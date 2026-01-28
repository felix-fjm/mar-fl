# ==================== Imports ====================
from __future__ import annotations
import copy
import gc
import logging
import os
import pickle
import queue
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from types import FrameType
from typing import  Any, Optional, Sequence
import torch
import torch.multiprocessing as tmp
import hivemind  # type: ignore
import wandb

# ==================== Runtime Settings ====================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.deterministic = False  # NOTE deactivated due to using set_all_seeds()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # NOTE added for H100 compatibility


# ==================== Local Imports ====================


# Machine learning tools
from ml_tasks import (
    SimpleCNN,
    ModernBERTClassifier,
    get_mnist_data_loaders,
    get_mnist_data_loaders_dirichlet,
    get_text_data_loaders,
    get_text_data_loaders_dirichlet,
    train_num_mini_batches_manually,
    evaluate,
    set_all_seeds,
)


# Config setup
from config_setup import parse_config


# Shared memory handling
from shared_memory_handling import (
    _dev_shm_usage_bytes,
    _dev_shm_total_usage_bytes,
    periodic_dict_cleanup,
)


# Peer synchronization
from peer_synchronization import count_num_participating_peers


# Peer aggregation logic
from peer_aggregation_logic import run_peer_aggregation_logic


# ==================== Logging ====================


# Logger configuration (stdout drop-in replacement for print, message-only formatting)
def _setup_logger(name: str = "mar_fl") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # NOTE avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))  # NOTE no extra prefix

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Setting up the logger
logger = _setup_logger()


# ==================== Events & Signals ====================


# Shutdown event
shutdown_event = threading.Event()


# Signal handler for cleanup
def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    shutdown_event.set()


# Signal handling
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ==================== Peers ====================


def peer_process(
    seed: int,
    split_type: str,
    dirichlet_alpha: float,
    ml_task: str,
    peer_id: int,
    bootstrap_address: Any,
    task_queue: Any,
    result_queue: Any,
    num_peers: int,
    peers_per_core: int,
    mar_group_size: int,
    mar_group_key_dim: int,
    mar_num_iterations: int,
    learning_rate: float,
    momentum: float,
    shared_model_dict: Any,
    valid_cores: Sequence[int],
    device: str,
    kn_dist: bool,
    kn_dist_iters: int,
    kn_dist_no_blending: bool,
    kd_all_candidates: bool,
    do_dp: bool,
    dp_clip_threshold: Optional[float],
    dp_noise_multiplier: Optional[float],
    model: Any,
    momentum_vector: Optional[Any] = None,
    next_batch_idx: Optional[int] = None,
) -> None:
    try:

        # Set random seeds for reproducibility
        set_all_seeds(seed)

        # Bind groups of peers_per_core peers to the same CPU core (e.g. set peers_per_core=2 when num_peers=64 while --ntasks=1 & --cpus-per-task=32)
        vcpu_id = valid_cores[(peer_id // peers_per_core) % len(valid_cores)]
        os.sched_setaffinity(0, {vcpu_id})

        # Initialize DHT instance
        dht = hivemind.DHT(initial_peers=bootstrap_address, start=True)
        logger.info(f"Peer {peer_id} started on vCPU: {list(os.sched_getaffinity(0))} | Process ID: {os.getpid()}")

        # Initialize model data
        if device == "cuda":
            assert torch.cuda.is_available(), f"[{datetime.now()}] WARNING: Peer {peer_id} requested CUDA but not available."
            torch.cuda.set_device(peer_id % torch.cuda.device_count())
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        if ml_task == "news":
            if split_type == "dirichlet":
                train_loader, _, test_loader = get_text_data_loaders_dirichlet(partition=peer_id, num_partitions=num_peers, seed=seed, alpha=dirichlet_alpha)
            else:
                train_loader, _, test_loader = get_text_data_loaders(partition=peer_id, num_partitions=num_peers, seed=seed)
        elif ml_task == "mnist":
            if split_type == "dirichlet":
                train_loader, _, test_loader = get_mnist_data_loaders_dirichlet(partition=peer_id, num_partitions=num_peers, seed=seed, alpha=dirichlet_alpha)
            else:
                train_loader, _, test_loader = get_mnist_data_loaders(partition=peer_id, num_partitions=num_peers, seed=seed)
        num_batches = len(train_loader)
        last_global_model = copy.deepcopy(model).to(device)
        if momentum_vector is None:
            momentum_vector = [torch.zeros_like(p, device=device) for p in model.parameters()]
        if next_batch_idx is None:
            next_batch_idx = 0
        logger.info(f"Peer {peer_id} using {device} as device for model training.")
        
        # Initialize DP variables
        used_dp_clip_threshold = dp_clip_threshold
        smoothed_delta_vector = None
        pb_delta = num_peers ** -1.1
        noise_steps = 0
        so_far_used_privacy_budget_approximated = 0
        so_far_used_privacy_budget_sweeped = 0

        # Set top-k ratio for KD
        if kd_all_candidates:
            top_k_ratio = 1.0
        else:
            top_k_ratio = 0.4

        while True:
            if shutdown_event.is_set(): 
                logger.info(f"Peer {peer_id} received shutdown signal, exiting loop.")
                break

            try:
                # Get a task from the task queue
                queue_item = task_queue.get(timeout=0.1)
                try:
                    target_peer, (task, iteration, task_value) = queue_item
                except (ValueError, TypeError) as e:
                    logger.info(f"Peer {peer_id} received a malformed task: {queue_item}. Error: {e}. Skipping.")
                    continue
                if target_peer not in (peer_id, "All"):
                    task_queue.put(queue_item)
                    continue
                logger.info(f"Peer {peer_id} got task: {task} with value {task_value}...")

                # ==================== Client Update ====================

                if task == "update":

                    # Train model on the peer's local and distinct data partition (core FL principle)
                    training_message = ""
                    training_loss = 0
                    start_time_update = time.time()
                    (
                        training_loss, 
                        training_message, 
                        next_batch_idx, 
                        momentum_vector, 
                        _
                    ) = train_num_mini_batches_manually( # NOTE increments the next batch index by the number of mini batches we just passed through the model
                        model=model, 
                        device=device, 
                        train_loader=train_loader, 
                        learning_rate=learning_rate,
                        momentum=momentum,
                        momentum_vector=momentum_vector, 
                        peer_id=peer_id,
                        num_mini_batches=task_value,
                        next_batch_idx=next_batch_idx
                    )
                    training_duration = time.time() - start_time_update
                    remaining_batches = num_batches - next_batch_idx

                    # Notify dispatcher
                    if iteration % 5 == 0:
                        training_message += f" Torch threads: {torch.get_num_threads()} (PID: {os.getpid()}). Running on CPU(s): {list(os.sched_getaffinity(0))}."
                    result_queue.put((3, iteration, peer_id, training_loss, training_message, (remaining_batches, training_duration)))

                # ==================== Model Aggregation ====================

                elif task == "aggregate":

                    # Unpack task value tuple
                    start_time_aggregation = time.time()
                    num_participating_peers, do_testing = task_value

                    # To align to real-world scenarios each peer has to compute the number of participating peers via DHT synchronization instead of using the information received from the dispatcher
                    num_participating_peers, _ = count_num_participating_peers(peer_id, iteration, num_peers, dht)

                    # In case of using DP do not include the CE loss during KD to avoid the respective privacy leak
                    kd_include_ce_loss = True
                    if do_dp:
                        used_dp_clip_threshold = dp_clip_threshold # NOTE for logging as we later overwrite the current threshold with an updated threshold
                        kd_include_ce_loss = False

                    # P2P aggregation logic including MAR, DP and KD
                    (
                        model, 
                        momentum_vector, 
                        last_global_model, 
                        smoothed_delta_vector,
                        noise_steps, 
                        dp_clip_threshold,
                        avg_group_length,
                        communicated_bytes, 
                        kl_factor, 
                        gradient_norm,
                        num_participating_peers,
                        group_logging,
                        so_far_used_privacy_budget_approximated,
                        so_far_used_privacy_budget_sweeped
                    ) = run_peer_aggregation_logic(
                        peer_id=peer_id,
                        iteration=iteration,
                        model=model,
                        momentum_vector=momentum_vector,
                        last_global_model=last_global_model,
                        dht=dht,
                        device=device,
                        mar_group_size=mar_group_size,
                        mar_group_key_dim=mar_group_key_dim,
                        mar_num_iterations=mar_num_iterations,
                        num_peers=num_peers,
                        num_participating_peers=num_participating_peers,
                        do_dp=do_dp,
                        dp_clip_threshold=dp_clip_threshold,
                        dp_noise_multiplier=dp_noise_multiplier,
                        pb_delta=pb_delta,
                        noise_steps=noise_steps,
                        kn_dist=kn_dist,
                        kn_dist_iters=kn_dist_iters,
                        kn_dist_no_blending=kn_dist_no_blending,
                        kd_include_ce_loss=kd_include_ce_loss,
                        top_k_ratio=top_k_ratio,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        train_loader=train_loader,
                        shared_model_dict=shared_model_dict,
                        smoothed_delta_vector=smoothed_delta_vector
                    )
                    
                    # If testing iteration then evaluate the model in a subprocess
                    aggregation_duration = time.time() - start_time_aggregation
                    if do_testing:
                        start_time_testing = time.time()    
                        test_acc, test_loss = evaluate(model, device, test_loader, label="Peer Test", peer_id=peer_id)
                        testing_duration = time.time() - start_time_testing
                        if ml_task == "news" and isinstance(test_acc, (int, float)) and test_acc > 50:
                            group_logging.append(f"Learning Rate will be decreased from {learning_rate} to 0.2.")
                        testing_duration = time.time() - start_time_testing
                        result_queue.put((4, iteration, peer_id, test_acc, test_loss, (group_logging, avg_group_length, communicated_bytes, kl_factor, used_dp_clip_threshold, so_far_used_privacy_budget_approximated, so_far_used_privacy_budget_sweeped, learning_rate, gradient_norm, aggregation_duration, testing_duration)))
                        if ml_task == "news" and isinstance(test_acc, (int, float)) and test_acc > 50:
                            learning_rate = 0.02
                    else:
                        result_queue.put((4, iteration, peer_id, 0, 0, (group_logging, avg_group_length, communicated_bytes, kl_factor, used_dp_clip_threshold, so_far_used_privacy_budget_approximated, so_far_used_privacy_budget_sweeped, learning_rate, gradient_norm, aggregation_duration, 0)))

                # ==================== Control Flow ====================
                
                # Avoid peers to crash during rare participation windows
                elif task == "skip":
                    logger.info(f"Peer {peer_id} skipping iteration {iteration}...")
                    continue
                
                # Shutdown the peer
                elif task == "shutdown":
                    logger.info(f"Peer {peer_id} shutting down...")
                    break
            
            except (queue.Empty, ValueError):
                # Keep waiting for tasks if none are available and skip value errors
                continue

    # Clean shutdown
    except Exception as e:
        logger.info(f"Peer process exception: {e}")
    finally:
        logger.info(f"Peer process {peer_id} exiting, cleaning up.")
        try:
            dht.shutdown()
        except Exception:
            pass
        try:
            shared_model_dict.clear()
            del shared_model_dict
        except Exception:
            pass
        try:
            task_queue.close()
            result_queue.close()
            task_queue.join_thread()
            result_queue.join_thread()
        except Exception:
            pass
        try:
            del task_queue
            del result_queue
        except Exception:
            pass
        gc.collect()


# ==================== Dispatcher ====================


def dispatcher(
    max_runtime: float,
    ml_task: str,
    shared_model_dict: Any,
    peer_processes: Sequence[Any],
    task_queue: Any,
    result_queue: Any,
    num_peers: int,
    local_update_participation_prob: int,
    aggregation_participation_prob: int,
    peer_dropout_likelihood: int,
    max_iterations: int,
    mini_batches_per_iteration: int,
    testing_frequency: int,
    dispatcher_patience: float,
    do_dp: bool,
) -> None:    
    
    # Logging
    logger.info("Dispatcher assigning tasks...")
    dispatcher_start_time = time.time()
    max_runtime_seconds = max_runtime * 3600

    # Parameter initialization
    best_loss, avg_loss, avg_acc = float("inf"), float("inf"), 0
    best_iteration = 0
    historical_losses = {}
    historical_accuracies = {}
    total_iterations = max_iterations
    k_update, k_aggregate = num_peers, num_peers
    do_testing = True

    # Each iteration contains one local model update per peer and one full mesh model aggregation
    for iteration in range(max_iterations):

        # Logging
        logger.info(f"DISPATCHER STARTING ITERATION {iteration+1}/{max_iterations}...")
        logger.info(datetime.now())
        start_time = time.time()

        # Track communication traffic
        iteration_sdict_bytes = 0 # NOTE truely relevant data traffic (model parameters & momentum vectors)
        iteration_tqueue_bytes, iteration_rqueue_bytes = 0, 0 # NOTE simulation-specific (dispatcher) traffic and would not occur in "productive mode" (self-controlling peers) => irrelevant for comparing with p2p baselines

        # ==================== Client Update ====================
        
        # Participation rates which determine heterogeneous peer participation
        if iteration >= 0: # NOTE optionally let all peers participate in the first iterations to enable initial intense global synchronization
            k_update, k_aggregate = int(num_peers * local_update_participation_prob / 100), int(num_peers * aggregation_participation_prob / 100)

        # Simulate heterogeneous peer participation
        participating_update_peers = random.sample(range(num_peers), k=k_update)
        
        # Trigger update tasks
        logger.info("Dispatcher puts \"update\"...")
        queue_object_a = ("update", iteration, mini_batches_per_iteration)
        queue_object_b = ("skip", iteration, None)
        queue_object_size_a = len(pickle.dumps(queue_object_a))
        queue_object_size_b = len(pickle.dumps(queue_object_b))
        for peer in range(num_peers): # NOTE assign "update" task only to selected peers s.t. this loop functions as DSpodFL's "Sporadic SGD" term
            if peer in participating_update_peers:
                task_queue.put((peer, queue_object_a))
                iteration_tqueue_bytes += queue_object_size_a
            else:
                task_queue.put((peer, queue_object_b))
                iteration_tqueue_bytes += queue_object_size_b
        
        # Wait for completion of all participating local updates
        results = []
        training_losses = []
        training_durations = []
        max_get_difference = 500
        last_get = time.time() + dispatcher_patience
        if iteration == 0:
            max_get_difference = 10 * 500
            last_get = time.time() + 10 * dispatcher_patience
            logger.info("Higher dispatcher patience during local updates because first FL iteration.")
        while len(results) < len(participating_update_peers) and (time.time() - last_get) < max_get_difference:
            if shutdown_event.is_set(): 
                logger.info("Dispatcher received shutdown signal, stopping...")
                break
            try:
                result_object = result_queue.get(timeout=0.1)
                iteration_rqueue_bytes += len(pickle.dumps(result_object))
                task_id, iteration_id, peer_id, training_loss, training_message, (remaining_batches, training_duration) = result_object
                if iteration_id != iteration or task_id != 3 or isinstance(training_loss,(int,float)) == False:
                    continue
                last_get = time.time()
                results.append(peer_id)
                training_losses.append(training_loss)
                training_durations.append(training_duration)
            except queue.Empty:
                continue
        missing_updates = set(participating_update_peers) - set(results)

        # Shutdown if missing results to avoid crashing script and leftover shared memory resources
        if missing_updates:
            logger.info(f"WARNING: Timeout! Missing update results from peers: {missing_updates}.")
            logger.info("DISPATCHER: Initiating shutdown due to missing peer updates.")
            for peer in range(num_peers):
                task_queue.put(("All", ("shutdown", max_iterations, 0)))
            return

        # Logging of peer status
        dead_peers = [peer_id for peer_id, p in enumerate(peer_processes) if not p.is_alive()]
        if dead_peers:
            logger.info(f"⚠️ Dead/non-responsive peers: {dead_peers}.")
        else:
            logger.info("🎉 All peer processes are alive.")

        # ==================== Model Aggregation ====================

        # Check if peers have to do model testing
        if iteration % testing_frequency == 0:
            do_testing = True
        else:
            do_testing = False

        # Simulate partial participation (aggregation_participation_prob) and network churn (peer_dropout_likelihood)
        if not do_dp or (do_dp and local_update_participation_prob == 100):
            participating_aggregation_peers = random.sample(range(num_peers), k=k_aggregate) # NOTE simulate partial participation
            if aggregation_participation_prob == local_update_participation_prob: # NOTE true partial participation (peer either participates in entire FL iteration or does not)
                participating_aggregation_peers = participating_update_peers
        else:
            participating_aggregation_peers = participating_update_peers # NOTE when doing DP we must have the same set of peers participating in update and aggregation to correctly compute the privacy budget
        if peer_dropout_likelihood > 0: # NOTE simulate network churn
            active_peers = []
            dropout_prob_float = peer_dropout_likelihood / 100.0
            for peer in participating_aggregation_peers:
                if random.random() > dropout_prob_float: # NOTE sampling from binomial distribution
                    active_peers.append(peer)
            dropped_count = len(participating_aggregation_peers) - len(active_peers)
            if dropped_count > 0:
                logger.info(f"{dropped_count} peers dropped out of aggregation due to network churn simulation.")            
            participating_aggregation_peers = active_peers
        num_participating_aggregation_peers = len(participating_aggregation_peers)

        # Trigger aggregation tasks
        logger.info("Dispatcher puts \"aggregate\"...")
        queue_object_values = (num_participating_aggregation_peers, do_testing)
        queue_object_a = ("aggregate", iteration, queue_object_values)
        queue_object_b = ("skip", iteration, None)
        queue_object_size_a = len(pickle.dumps(queue_object_a))
        queue_object_size_b = len(pickle.dumps(queue_object_b))
        for peer in range(num_peers): # NOTE assign "aggregate" task only to selected peers s.t. this loop functions as DSpodFL's "Sporadic aggregation" term
            if peer in participating_aggregation_peers:
                task_queue.put((peer, queue_object_a))
                iteration_tqueue_bytes += queue_object_size_a
            else:
                task_queue.put((peer, queue_object_b))
                iteration_tqueue_bytes += queue_object_size_b
        
        # Wait for completion of all participating model aggregations and evaluations
        results = []
        losses = []
        accs = []
        avg_group_lengths = []
        sfup_budgets_approximated = []
        sfup_budgets_sweeped = []
        learning_rates = []
        gradient_norms = []
        clipping_thresholds = []
        aggregation_durations = []
        testing_durations = []
        kl_factor = 0
        max_get_difference = 500
        last_get = time.time() + dispatcher_patience
        if iteration == 0:
            max_get_difference = 10 * 500
            last_get = time.time() + 10 * dispatcher_patience
            logger.info("Higher dispatcher patience during aggregation because first FL iteration.")
        while len(results) < len(participating_aggregation_peers) and (time.time() - last_get) < max_get_difference:
            if shutdown_event.is_set(): 
                logger.info("Dispatcher received shutdown signal, stopping...")
                break
            try:
                result_object = result_queue.get(timeout=0.1)
                iteration_rqueue_bytes += len(pickle.dumps(result_object))
                task_id, iteration_id, peer_id, acc, loss, (group_logging, avg_group_length, communicated_bytes, kl_factor, clipping_threshold, so_far_used_privacy_budget_approximated, so_far_used_privacy_budget_sweeped, peer_lr, gradient_norm, aggregation_duration, testing_duration) = result_object
                if iteration_id != iteration or task_id != 4 or isinstance(acc,(int,float)) == False or isinstance(loss,(int,float)) == False:
                    continue
                if do_testing:
                    acc_f = float(acc)
                    loss_f = float(loss)
                    accs.append(acc_f)
                    losses.append(loss_f)
                last_get = time.time()
                results.append(peer_id)
                avg_group_lengths.append(avg_group_length)
                sfup_budgets_approximated.append(so_far_used_privacy_budget_approximated)
                sfup_budgets_sweeped.append(so_far_used_privacy_budget_sweeped)
                learning_rates.append(peer_lr)
                gradient_norms.append(gradient_norm)
                clipping_thresholds.append(clipping_threshold)
                aggregation_durations.append(aggregation_duration)
                testing_durations.append(testing_duration)
                iteration_sdict_bytes += communicated_bytes
            except queue.Empty:
                continue
        missing_aggregations = set(participating_aggregation_peers) - set(results)

        # Shutdown if missing results to avoid crashing script and leftover shared memory resources
        if missing_aggregations:
            logger.info(f"WARNING: Timeout! Missing aggregation results from peers: {missing_aggregations}.")
            logger.info("DISPATCHER: Initiating shutdown due to missing peer aggregations.")
            for peer in range(num_peers):
                task_queue.put(("All", ("shutdown", max_iterations, 0)))
            return
        
        # ==================== Logging & Control Flow ====================

        # WandB Logging
        avg_training_loss = sum(training_losses) / len(training_losses) if training_losses else 10
        avg_training_duration = sum(training_durations) / len(training_durations) if training_durations else 0
        avg_group_length = sum(avg_group_lengths) / len(avg_group_lengths) if avg_group_lengths else 0
        average_sfup_budget_approximated = sum(sfup_budgets_approximated) / len(sfup_budgets_approximated) if sfup_budgets_approximated else 0
        average_sfup_budget_sweeped = sum(sfup_budgets_sweeped) / len(sfup_budgets_sweeped) if sfup_budgets_sweeped else 0
        avg_lr = sum(learning_rates) / len(learning_rates) if learning_rates else 0
        avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0
        avg_clipping_threshold = sum(clipping_thresholds) / len(clipping_thresholds) if clipping_thresholds else 0
        avg_aggregation_duration = sum(aggregation_durations) / len(aggregation_durations) if aggregation_durations else 0
        avg_testing_duration = sum(testing_durations) / len(testing_durations) if testing_durations else 0
        if do_testing:
            avg_loss = sum(losses) / len(losses) if losses else 10
            avg_acc = sum(accs) / len(accs) if accs else 0
            historical_losses[iteration] = avg_loss
            historical_accuracies[iteration] = avg_acc
        total_time = time.time() - start_time
        logger.info(f"Iteration {iteration+1} finished at {datetime.now()}.")
        logger.info(f"Iteration duration: {total_time:.2f} seconds.")
        wandb.log({
            "iteration": iteration+1,
            "participating_update_peers": len(participating_update_peers),
            "participating_aggregation_peers": len(participating_aggregation_peers),
            "dead_peers": len(dead_peers),
            "avg_training_loss": avg_training_loss,
            "testing_loss": avg_loss,
            "testing_accuracy": avg_acc,
            "avg_group_length": avg_group_length,
            "iteration_tqueue_bytes": iteration_tqueue_bytes,
            "iteration_rqueue_bytes": iteration_rqueue_bytes,
            "iteration_sdict_bytes": iteration_sdict_bytes,
            "iteration_duration": total_time,
            "kl_factor": kl_factor,
            "avg_so_far_used_privacy_budget_approximated": average_sfup_budget_approximated,
            "avg_so_far_used_privacy_budget_sweeped": average_sfup_budget_sweeped,
            "avg_learning_rate": avg_lr,
            "avg_gradient_norm": avg_gradient_norm,
            "avg_clipping_threshold": avg_clipping_threshold,
            "avg_training_duration": avg_training_duration,
            "avg_aggregation_duration": avg_aggregation_duration,
            "avg_testing_duration": avg_testing_duration
        })

        # Periodic cleanup of shared dictionary to prevent leaks from failed coordinators
        periodic_dict_cleanup(logger=logger, shared_dict=shared_model_dict, current_iteration=iteration)

        # Early stopping check
        if do_testing:

            # Stop if target accuracy is reached
            stop_acc = None
            if ml_task.lower() == "mnist":
                stop_acc = 95.0
            elif ml_task.lower() == "news":
                stop_acc = 50.0
            if stop_acc is not None and avg_acc >= stop_acc:
                total_iterations = iteration + 1
                logger.info(f"Early stopping triggered at round {total_iterations}: Reached {avg_acc:.2f}% accuracy (threshold: {stop_acc}%)")
                break

            # Stop if loss is higher than 10 (usually caused by too restrictive DP)
            if avg_loss > 10:
                total_iterations = iteration + 1
                logger.info(f"Early stopping triggered at iteration {total_iterations}: Current loss ({avg_loss:.4f}) is higher than 10 (usually caused by too restrictive DP).")
                break

            # Stop if loss has not improved compared to 50 iterations ago
            check_iteration = iteration - 50 # NOTE check iteration further ago would cause affected jobs to regularly run into Broken Pipe crashs (checking 50 iterations ago does not stop too early since peak accuracy was already seen for DP with NM values 0.2, 0.5, 1.0)
            if check_iteration in historical_losses:
                previous_loss = historical_losses[check_iteration]
                if avg_loss >= previous_loss:
                    total_iterations = iteration + 1
                    logger.info(f"Early stopping triggered at iteration {total_iterations}: Current loss ({avg_loss:.4f}) is not better than loss at iteration {check_iteration + 1} ({previous_loss:.4f}).")
                    break

            # Stop if accuracy has not improved compared to 50 iterations ago
            if check_iteration in historical_accuracies:
                previous_acc = historical_accuracies[check_iteration]
                if avg_acc <= previous_acc:
                    total_iterations = iteration + 1
                    logger.info(f"Early stopping triggered at iteration {total_iterations}: Current accuracy ({avg_acc:.2f}%) is not better than accuracy at iteration {check_iteration + 1} ({previous_acc:.2f}%).")
                    break

            # Stop if the job's max runtime is approaching
            elapsed_time = time.time() - dispatcher_start_time
            if elapsed_time >= (max_runtime_seconds - 20 * 60):
                total_iterations = iteration + 1
                logger.info(f"Early stopping triggered at iteration {total_iterations}: Time limit approaching. Elapsed time: {elapsed_time:.2f}s, Max runtime: {max_runtime_seconds:.2f}s.")
                break

    # Logging
    wandb.log({
        "total_iterations": total_iterations,
        "best_testing_loss": best_loss,
        "best_iteration": best_iteration
    })

    # Add shutdown tasks
    for peer in range(num_peers):
        task_queue.put(("All", ("shutdown", max_iterations, 0)))


# ==================== Main ====================


def main():

    # Shared memory check before any multiprocessing
    logger.info("[SHM-CHECK][BEFORE MP] /dev/shm usage before starting Manager/processes:")
    shm_before_mp_MB = _dev_shm_usage_bytes() / 1024 / 1024
    shm_before_mp_total_MB = _dev_shm_total_usage_bytes() / 1024 / 1024
    logger.info(f"    /dev/shm (psm_, torch_): ≈ {shm_before_mp_MB:.1f} MB")
    logger.info(f"    /dev/shm (ALL): ≈ {shm_before_mp_total_MB:.1f} MB\n")
    wandb.log({
        "shm_before_mp_MB": shm_before_mp_MB,
        "shm_before_mp_total_MB": shm_before_mp_total_MB,
    })

    # Context manager for multiprocessing.Manager()
    ctx = tmp.get_context('spawn')
    with ctx.Manager() as manager:
        shared_model_dict = manager.dict()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Start a single bootstrap node
        bootstrap_dht = hivemind.DHT(initial_peers=[], start=True)
        bootstrap_address = bootstrap_dht.get_visible_maddrs()
        start_time = time.time()
        logger.info(f"Bootstrap node started on {bootstrap_address} at {datetime.now()}.")

        # Get list of available CPU core indices
        valid_cores = list(os.sched_getaffinity(0))

        # Compute the overall number of training data samples
        num_peers = wandb.config.num_peers
        ml_task = wandb.config.ml_task
        overall_num_samples = 0
        for peer in range(num_peers):
            if ml_task == "news":
                if wandb.config.split_type == "dirichlet":
                    mock_train_loader, _, _ = get_text_data_loaders_dirichlet(partition=peer, num_partitions=num_peers, seed=wandb.config.seed, alpha=wandb.config.dirichlet_alpha)
                else:
                    mock_train_loader, _, _ = get_text_data_loaders(partition=peer, num_partitions=num_peers, seed=wandb.config.seed)
                num_samples = len(mock_train_loader.loader.dataset.indices)
            elif ml_task == "mnist":
                if wandb.config.split_type == "dirichlet":
                    mock_train_loader, _, _ = get_mnist_data_loaders_dirichlet(partition=peer, num_partitions=num_peers, seed=wandb.config.seed, alpha=wandb.config.dirichlet_alpha)
                else:
                    mock_train_loader, _, _ = get_mnist_data_loaders(partition=peer, num_partitions=num_peers, seed=wandb.config.seed)
                num_samples = len(mock_train_loader.dataset)
            overall_num_samples += num_samples

        # Preload the model to avoid Huggingface access issues in case of news task
        ml_task = wandb.config.ml_task
        logger.info(f"Main process is pre-loading the '{ml_task}' model...")
        if ml_task == "news":
            model = ModernBERTClassifier()
        elif ml_task == "mnist":
            model = SimpleCNN()
        logger.info("Model instance is created.")

        # Create persistent peer processes
        peer_processes = [
            ctx.Process(target=peer_process, args=(
                wandb.config.seed,
                wandb.config.split_type,
                wandb.config.dirichlet_alpha,
                ml_task,
                i, 
                bootstrap_address, 
                task_queue, 
                result_queue, 
                wandb.config.num_peers, 
                wandb.config.peers_per_core,
                wandb.config.mar_group_size,
                wandb.config.mar_group_key_dim,
                wandb.config.mar_num_iterations,
                wandb.config.learning_rate,
                wandb.config.momentum, 
                shared_model_dict,
                valid_cores,
                wandb.config.device,
                wandb.config.knowledge_distillation,
                wandb.config.knowledge_distillation_iters,
                wandb.config.knowledge_distillation_no_blending,
                wandb.config.knowledge_distillation_use_all_candidates,
                wandb.config.do_dp,
                wandb.config.dp_clip_threshold,
                wandb.config.dp_noise_multiplier,
                copy.deepcopy(model)
            ))
            for i in range(wandb.config.num_peers)
        ]

        # Start all peer processes
        for p in peer_processes:
            p.start()
        
        # Run dispatcher to assign tasks
        try:
            dispatcher(
                max_runtime=wandb.config.max_runtime,
                ml_task=ml_task,
                shared_model_dict=shared_model_dict,
                peer_processes=peer_processes, 
                task_queue=task_queue, 
                result_queue=result_queue,
                num_peers=wandb.config.num_peers, 
                local_update_participation_prob=wandb.config.local_update_participation_prob,
                aggregation_participation_prob=wandb.config.aggregation_participation_prob,
                peer_dropout_likelihood=wandb.config.peer_dropout_likelihood,
                max_iterations=wandb.config.max_iterations,
                mini_batches_per_iteration=wandb.config.mini_batches_per_iteration, 
                testing_frequency=wandb.config.testing_frequency,
                dispatcher_patience=wandb.config.dispatcher_patience,
                do_dp=wandb.config.do_dp
            )
        except Exception as e:
            logger.info(f"Main exception: {e}")
        finally:
            logger.info("Main cleanup: Notifying all peers to shut down...")
            
            # 1v4: notify: ask all peers to shut down gracefully
            for i in range(wandb.config.num_peers):
                try:
                    task_queue.put(("All", ("shutdown", -1, 0)), timeout=0.1)
                except queue.Full:
                    logger.info("Could not send shutdown task to all peers, queue is full.")
                    break
            logger.info("Main cleanup: Waiting for peer processes to join...")
            
            # 2v4: join: wait for processes to exit on their own
            for p in peer_processes:
                p.join(timeout=10)
            logger.info("Main cleanup: Terminating any remaining processes...")

            # 3v4: terminate: forcefully stop any process that is still running
            for p in peer_processes:
                if p.is_alive():
                    logger.info(f"Peer process {p.pid} did not exit gracefully. Terminating.")
                    p.terminate()
                    p.join(timeout=60)

            # 4v4: shutdown other resources
            try:
                logger.info("Main cleanup: Clearing all entries from the shared dictionary...")
                shared_model_dict.clear()
                logger.info("Main cleanup: Shared dictionary cleared.")
            except Exception as e:
                logger.info(f"Main cleanup: Could not clear shared_model_dict. Reason: {e}")
            try:
                task_queue.close()
                result_queue.close()
                task_queue.join_thread()
                result_queue.join_thread()
            except Exception:
                pass
            try:
                manager.shutdown()
            except Exception:
                pass
            try:
                del shared_model_dict
                del task_queue
                del result_queue
            except Exception:
                pass
            try:
                bootstrap_dht.shutdown()
            except Exception:
                pass
            gc.collect()
            total_time = time.time() - start_time
            logger.info(f"All peers have completed their tasks at {datetime.now()}.")
            logger.info(f"Total duration: {total_time:.2f} seconds.")

        # Logging
        total_time = time.time() - start_time
        logger.info(f"All peers have completed their tasks at {datetime.now()}.")
        logger.info(f"Total duration: {total_time:.2f} seconds.")

    # Shared memory check after multiprocessing cleanup
    logger.info("[SHM-CHECK][AFTER MP] /dev/shm usage after closing all multiprocessing resources:")
    shm_after_mp_MB = _dev_shm_usage_bytes() / 1024 / 1024
    shm_after_mp_total_MB = _dev_shm_total_usage_bytes() / 1024 / 1024
    logger.info(f"    /dev/shm (psm_, torch_): ≈ {shm_after_mp_MB:.1f} MB")
    logger.info(f"    /dev/shm (ALL): ≈ {shm_after_mp_total_MB:.1f} MB\n")
    wandb.log({
        "shm_after_mp_MB": shm_after_mp_MB,
        "shm_after_mp_total_MB": shm_after_mp_total_MB,
    })

    # Compute and log the left over and added shared memory allocation caused by my job
    shm_diff_MB = shm_after_mp_MB - shm_before_mp_MB
    shm_diff_total_MB = shm_after_mp_total_MB - shm_before_mp_total_MB
    logger.info(f"[SHM-CHECK][DIFF] Difference in /dev/shm allocation (psm_, torch_): {shm_diff_MB:.1f} MB")
    logger.info(f"[SHM-CHECK][DIFF] Difference in /dev/shm allocation (ALL): {shm_diff_total_MB:.1f} MB")
    wandb.log({
        "shm_diff_after_before_MB": shm_diff_MB,
        "shm_diff_after_before_total_MB": shm_diff_total_MB,
    })
    

# ==================== Argument Parsing & WandB Init ====================


if __name__ == "__main__":

    # Argument parsing
    (config, unique_run_name) = parse_config()

    # WandB config
    wandb_entity = "FedMoshpitAvg"
    wandb_project = "p2p_fl_niid"
    wandb_dir = "/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ga27fed2/wandb_runs"
    run = wandb.init(
        dir=wandb_dir,
        entity=wandb_entity,
        project=wandb_project,
        name=unique_run_name,
        config=config.__dict__,
    )

    # Start processes via spawn instead of fork
    tmp.set_start_method("spawn", force=True)

    # Let's go
    main()