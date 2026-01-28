# ==================== Imports ====================
import copy
import gc
import logging
import time
from datetime import datetime
from typing import List, Tuple, Any, Optional
import torch
import hivemind  # type: ignore


# ==================== Local Imports ====================


# Machine learning tools
from ml_tasks import knowledge_distillation


# Peer synchronization
from peer_synchronization import wait_for_all


# Moshpit all-reduce
from moshpit_all_reduce import (
    get_initial_index, 
    matchmake_group, 
    update_group_key, 
    communicate_models, 
    fedavg_aggregation
)


# Differential privacy
from differential_privacy import (
    compute_delta_vector, 
    clip_and_blur_delta_vector, 
    apply_delta_to_model, 
    dp_federated_aggregation, 
    compute_epsilon, 
    sweep_epsilon_enhanced
)


# ==================== Logging ====================


# Setting up the logger
logger = logging.getLogger("mar_fl")


# ==================== Peer Process Aggregation Logic ====================


def run_peer_aggregation_logic(
    peer_id: int, 
    iteration: int, 
    model: torch.nn.Module, 
    momentum_vector: List[torch.Tensor], 
    last_global_model: torch.nn.Module, 
    dht: hivemind.DHT, 
    device: torch.device,
    mar_group_size: int, 
    mar_group_key_dim: int, 
    mar_num_iterations: int, 
    num_peers: int, 
    num_participating_peers: int,
    do_dp: bool, 
    dp_clip_threshold: float, 
    dp_noise_multiplier: float, 
    pb_delta: float, 
    noise_steps: int,
    kn_dist: bool, 
    kn_dist_iters: int, 
    kn_dist_no_blending: bool, 
    kd_include_ce_loss: bool,
    top_k_ratio: float,
    learning_rate: float,
    momentum: float,
    train_loader: torch.utils.data.DataLoader,
    shared_model_dict: Any,
    smoothed_delta_vector: List[torch.Tensor]
) -> Tuple[
    torch.nn.Module,      # model
    List[torch.Tensor],   # momentum_vector
    torch.nn.Module,      # last_global_model
    List[torch.Tensor],   # smoothed_delta_vector
    int,                  # noise_steps
    float,                # dp_clip_threshold
    float,                # avg_group_length
    int,                  # communicated_bytes
    float,                # kl_factor
    float,                # gradient_norm
    int,                  # num_participating_peers
    List[str],            # group_logging
    float,                # so_far_used_privacy_budget_approximated
    float                 # so_far_used_privacy_budget_sweeped
]:
    prev_group_key = get_initial_index(peer_id, mar_group_size, mar_group_key_dim, num_peers)
    group_id_strings = []
    groups = []
    group_logging = []
    group_lengths = []
    noisy_clipped_delta_vector = None
    sigma_b = num_participating_peers / 20.0
    z_delta = (dp_noise_multiplier**-2 - (2 * sigma_b)**-2) ** -0.5
    binary_indicator = 1.0
    gradient_norm = 0.0
    communicated_bytes = 0
    kl_factor = 0
    round = 1
    use_case = 0
    if kn_dist and iteration < kn_dist_iters:
        use_case = 1
    else:
        if do_dp:
            use_case = 3
        else:
            use_case = 2
    for mar_iteration in range(mar_num_iterations):

        # Group-making
        group_id_str = "_".join(map(str, prev_group_key))
        group = matchmake_group(dht, group_id_str, peer_id, num_peers, num_participating_peers, iteration, mar_iteration)
        group_logging.append(f"PEER {peer_id} - ITER {iteration}-{mar_iteration}: {group}")
        group_lengths.append(len(group))
        group_id_strings.append(group_id_str)
        groups.append(group)

        # Communication and knowledge distillation or federated averaging
        chunk_index = 0
        if group:
            models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected = [], [], [], []
            
            # If knowledge distillation
            if use_case == 1:
                if do_dp and mar_iteration == 0: # NOTE if using DP first obtain a DP-safe local model before communicating it with other peers (DP only needed in first MAR iteration because we do not use CE loss in KD when using DP)
                    delta_vector = compute_delta_vector(model, last_global_model, device)
                    noisy_clipped_delta_vector, binary_indicator, gradient_norm = clip_and_blur_delta_vector(delta_vector, dp_clip_threshold, z_delta, num_participating_peers, device)
                    model = apply_delta_to_model(last_global_model, noisy_clipped_delta_vector)
                models_collected, chunk_index, communicated_bytes = communicate_models(device, peer_id, model, momentum_vector, group, group_id_str, shared_model_dict, dht, iteration, round, mar_iteration, use_case, communicated_bytes)
                if models_collected:
                    kd_start_time = time.time()
                    model, momentum_vector, kl_factor = knowledge_distillation(peer_id, model, device, train_loader, models_collected, momentum_vector, learning_rate, momentum, iteration, kn_dist_iters, kn_dist_no_blending, kd_include_ce_loss, top_k_ratio) # NOTE do not include CE loss when using DP
                    kd_end_time = time.time()
                    group_logging.append(f"KD in {(kd_end_time - kd_start_time):.2f}s with KL factor {kl_factor:.2f} and include_ce_loss={kd_include_ce_loss}")
                else:
                    logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any ML data from its own group: {group}.")
            
            # If regular federated averaging
            elif use_case == 2:
                models_collected, momentum_vectors_collected, chunk_index, communicated_bytes = communicate_models(device, peer_id, model, momentum_vector, group, group_id_str, shared_model_dict, dht, iteration, round, mar_iteration, use_case, communicated_bytes)
                if models_collected:
                    model, momentum_vector = fedavg_aggregation(model, models_collected, momentum_vector, momentum_vectors_collected)
                else:
                    logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any ML data from its own group: {group}.")
            
            # If DP federated averaging
            elif use_case == 3:
                if mar_iteration == 0:
                    delta_vector = compute_delta_vector(model, last_global_model, device)
                    noisy_clipped_delta_vector, binary_indicator, gradient_norm = clip_and_blur_delta_vector(delta_vector, dp_clip_threshold, z_delta, num_participating_peers, device)
                    if iteration > 0 and smoothed_delta_vector is not None:
                        smoothed_delta_vector = [0.9 * s + n for s, n in zip(smoothed_delta_vector, noisy_clipped_delta_vector)]
                    else:
                        smoothed_delta_vector = noisy_clipped_delta_vector
                    dp_safe_local_model = apply_delta_to_model(last_global_model, smoothed_delta_vector)
                dp_safe_local_models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, chunk_index, communicated_bytes = communicate_models(device, peer_id, dp_safe_local_model, momentum_vector, group, group_id_str, shared_model_dict, dht, iteration, round, mar_iteration, use_case, communicated_bytes, binary_indicator, smoothed_delta_vector)
                if dp_safe_local_models_collected:
                    last_mar_iteration = False
                    if mar_iteration < (mar_num_iterations - 1):
                        dp_safe_local_model, momentum_vector, binary_indicator, smoothed_delta_vector = dp_federated_aggregation(last_mar_iteration, num_participating_peers, dp_safe_local_model, dp_safe_local_models_collected, momentum_vector, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, dp_clip_threshold, device)
                    else:
                        last_mar_iteration = True
                        model, momentum_vector, smoothed_delta_vector, dp_clip_threshold = dp_federated_aggregation(last_mar_iteration, num_participating_peers, dp_safe_local_model, dp_safe_local_models_collected, momentum_vector, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, dp_clip_threshold, device)
                        last_global_model = type(model)().to(device)
                        last_global_model.load_state_dict(copy.deepcopy(model.state_dict()))
                else:
                    logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any ML data from its own group: {group}.")
            
            # Clean memory
            del models_collected
            del momentum_vectors_collected
            del binary_indicators_collected
            del smoothed_delta_vectors_collected
            torch.cuda.empty_cache()
            gc.collect()

            # Compute the next group key
            prev_group_key = update_group_key(prev_group_key, chunk_index, mar_group_size, mar_group_key_dim, num_peers)

        # Handle failed group matchmaking correctly
        else:
            logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} has an empty group.")
            chunk_index = peer_id % mar_group_size
            prev_group_key = update_group_key(prev_group_key, chunk_index, mar_group_size, mar_group_key_dim, num_peers)

    # Extra round of Moshpit All-Reduce with pre-defined groups if knowledge distillation iteration
    if kn_dist and iteration < kn_dist_iters:
        
        # Synchronization
        barrier_key = f"kdfa_{iteration}"
        dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
        wait_for_all(dht, barrier_key, num_peers, num_participating_peers, max_delays=400)

        # Moshpit All-Reduce
        round = 2
        use_case = 0
        if do_dp:
            use_case = 3
        else:
            use_case = 2
        for mar_iteration in range(mar_num_iterations):

            # Pre-defined groups
            group_id_str = group_id_strings[mar_iteration]
            group = groups[mar_iteration]
            group_logging.append(f"PEER {peer_id} - ITER {iteration}-{mar_iteration}: {group}")
            
            # Communication and federated averaging
            if group:
                models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected = [], [], [], []
                
                # If regular federated averaging
                if use_case == 2:
                    models_collected, momentum_vectors_collected, _, communicated_bytes = communicate_models(device, peer_id, model, momentum_vector, group, group_id_str, shared_model_dict, dht, iteration, round, mar_iteration, use_case, communicated_bytes)
                    if models_collected:
                        model, momentum_vector = fedavg_aggregation(model, models_collected, momentum_vector, momentum_vectors_collected)
                    else:
                        logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any ML data from its own group: {group}.")

                # If DP federated averaging
                elif use_case == 3:
                    if mar_iteration == 0:
                        post_kd_delta_vector = compute_delta_vector(model, last_global_model, device) # NOTE again compute the delta vector because in the meantime we improved the local model with KD but do not clip and add noise because we did not include the CE loss in KD and KD therefore only exhibits a tiny privacy leak (=> mention this leak as assumption in my analysis)
                        if iteration > 0 and smoothed_delta_vector is not None:
                            smoothed_delta_vector = [0.9 * s + n for s, n in zip(smoothed_delta_vector, post_kd_delta_vector)]
                        else:
                            smoothed_delta_vector = post_kd_delta_vector
                        dp_safe_local_model = apply_delta_to_model(last_global_model, smoothed_delta_vector)
                    dp_safe_local_models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, _, communicated_bytes = communicate_models(device, peer_id, dp_safe_local_model, momentum_vector, group, group_id_str, shared_model_dict, dht, iteration, round, mar_iteration, use_case, communicated_bytes, binary_indicator, smoothed_delta_vector)
                    if dp_safe_local_models_collected:
                        last_mar_iteration = False
                        if mar_iteration < (mar_num_iterations - 1):
                            dp_safe_local_model, momentum_vector, binary_indicator, smoothed_delta_vector = dp_federated_aggregation(last_mar_iteration, num_participating_peers, dp_safe_local_model, dp_safe_local_models_collected, momentum_vector, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, dp_clip_threshold, device)
                        else:
                            last_mar_iteration = True
                            model, momentum_vector, smoothed_delta_vector, dp_clip_threshold = dp_federated_aggregation(last_mar_iteration, num_participating_peers, dp_safe_local_model, dp_safe_local_models_collected, momentum_vector, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, dp_clip_threshold, device)
                            last_global_model = type(model)().to(device)
                            last_global_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    else:
                        logger.info(f"[{datetime.now()}] WARNING: Peer {peer_id} could not collect any ML data from its own group: {group}.")
                
                # Clean memory
                del models_collected
                del momentum_vectors_collected
                del binary_indicators_collected
                del smoothed_delta_vectors_collected
                torch.cuda.empty_cache()
                gc.collect()

    # In case of using DP save the new global model for the next model aggregation and compute the so far used privacy budget
    if do_dp:
        noise_steps += 1
        sample_rate = num_participating_peers / num_peers
        so_far_used_privacy_budget_approximated = compute_epsilon(noise_steps, pb_delta, sample_rate, dp_noise_multiplier) # NOTE conservative approximation of worst case epsilon
        so_far_used_privacy_budget_sweeped = sweep_epsilon_enhanced(noise_steps, pb_delta, sample_rate, dp_noise_multiplier) # NOTE sweeping of optimal epsilon as opacus would do
                    
    # Return parameters
    return (
        model, 
        momentum_vector, 
        last_global_model, 
        smoothed_delta_vector,
        noise_steps, 
        dp_clip_threshold,
        (sum(group_lengths) / len(group_lengths)) if group_lengths else 0, # avg_group_length
        communicated_bytes, 
        kl_factor, 
        gradient_norm,
        num_participating_peers,
        group_logging,
        so_far_used_privacy_budget_approximated, 
        so_far_used_privacy_budget_sweeped
    )