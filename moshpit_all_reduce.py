# ==================== Imports ====================
from __future__ import annotations
import copy
import gc
import logging
import math
from datetime import datetime
from typing import Any, MutableMapping, Optional, Sequence
import torch
import hivemind  # type: ignore


# ==================== Local Imports ====================


# Machine learning tools
from ml_tasks import ModernBERTClassifier


# Peer synchronization
from peer_synchronization import wait_for_all, wait_for_peers


# ==================== Logging ====================


# Setting up the logger
logger = logging.getLogger("mar_fl")


# ==================== Group Keys ====================


# Create a smart initial group_key from peer_id and group_size s.t. group_size peers receive the same group_key (if num_peers = group_size ^ group_key_dim) and otherwise ensuring best possible global aggregation
def get_initial_index(
    peer_id: int,
    group_size: int,
    group_key_dim: int,
    num_peers: int,
) -> list[int]:
    # perfect grid s.t. keeping legacy behavior
    if num_peers == (group_size**group_key_dim):
        return [(peer_id // (group_size ** (group_key_dim - j))) % group_size for j in range(1, group_key_dim)]

    # imperfect cases (incl. gs=4,gk=3,np=125 and gs=3,gk=4,np=125)
    bucket = peer_id // group_size  # contiguous groups of size gs
    phase = 0  # reshuffle phase (advanced each MAR round)

    # pack into a key of length (gk-1) used for matching
    dims = group_key_dim - 1
    if dims == 1:
        return [bucket]
    elif dims == 2:
        return [bucket, phase]
    else:  # dims >= 3
        # keep the last coord as "phase", other fillers are 0
        return [bucket] + [0] * (dims - 2) + [phase]


# Smart updating of the group key by moving the sliding window in a smart way
def update_group_key(
    previous_C_i: list[int],
    new_c_i: int,
    group_size: int,
    group_key_dim: int,
    num_peers: int,
) -> list[int]:
    # perfect grid → legacy sliding window
    if num_peers == (group_size ** group_key_dim):
        return previous_C_i[1:] + [new_c_i]

    # number of target groups (last one may be smaller)
    G = math.ceil(num_peers / group_size)
    dims = group_key_dim - 1

    # unpack old bucket and phase (we put phase in the last coordinate)
    old_bucket = previous_C_i[0]
    old_phase  = previous_C_i[-1] if dims >= 2 else 0

    # choose α schedule
    if (group_size, group_key_dim, num_peers) == (4, 3, 125):
        alpha_schedule = [1, 5]          # 2 updates (it=3 rounds)
    elif (group_size, group_key_dim, num_peers) == (3, 4, 125):
        alpha_schedule = [1, 5, 11]      # 3 updates (it=4 rounds)
    else:
        # generic: odd α co-prime with G; advance with phase
        a = 2 * old_phase + 1
        while math.gcd(a, G) != 1:
            a += 2
        alpha_schedule = [a]

    alpha = alpha_schedule[old_phase % len(alpha_schedule)]

    # bucket permutation: mixes by member position (chunk_index) across buckets
    new_bucket = (old_bucket + alpha * int(new_c_i)) % G
    new_phase  = (old_phase + 1) % max(1, len(alpha_schedule))

    # repack: keep last coord as phase so we can advance α next time
    if dims == 1:
        return [new_bucket]
    elif dims == 2:
        return [new_bucket, new_phase]
    else:  # dims >= 3
        return [new_bucket] + [0]*(dims-2) + [new_phase]


# ==================== Communication ====================


# Simplified matchmaking through filtering peers with identical group_key in the dht
def matchmake_group(
    dht: Any,
    group_id_str: str,
    peer_id: int,
    num_peers: int,
    num_participating_peers: int,
    outer_iteration: int,
    mar_iteration: int,
) -> list[int]:
    # 1v3: store the group key in the dht and inform all peers via dht
    group_key_iter = f"{group_id_str}-{outer_iteration}-{mar_iteration}"
    matchmake_key = f"matchmake_{group_key_iter}_{peer_id}"
    dht.store(matchmake_key, True, expiration_time=hivemind.get_dht_time() + 60)
    barrier_key = f"mg1_{outer_iteration}-{mar_iteration}"
    dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    wait_for_all(dht, barrier_key, num_peers, num_participating_peers)

    # 2v3: find peers with same group key in the dht
    group = []
    for peer in range(num_peers):
        check_key = f"matchmake_{group_key_iter}_{peer}"
        if dht.get(check_key, latest=True):
            group.append(peer)
    if peer_id not in group:
        group.append(peer_id)
    group = sorted(group)

    # 3v3: enforce symmetry in group formation by checking which group members peers with the same group key have gathered
    gathered_key = f"gathered_{group_key_iter}_{peer_id}"
    dht.store(gathered_key, group, expiration_time=hivemind.get_dht_time() + 60)  # NOTE store group list
    barrier_key = f"mg2_{outer_iteration}-{mar_iteration}"
    dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    wait_for_all(
        dht,
        barrier_key,
        num_peers,
        num_participating_peers,
    )  # NOTE wait_for_peers would not enforce symmetry

    group_checked: list[int] = []
    for peer in range(num_peers):
        check_key = f"gathered_{group_key_iter}_{peer}"
        result = dht.get(check_key, latest=True)
        if result and isinstance(result.value, list):
            group_checked.extend(result.value)
    group = sorted(set(group_checked))
    return group


# Communicate model parameters and momentum vectors within groups through writing to and reading from shared group dicts
def communicate_models(
    device: Any,
    peer_id: int,
    model: Any,
    momentum_vector: Any,
    group: Sequence[int],
    group_id_str: str,
    shared_model_dict: MutableMapping[Any, Any],
    dht: Any,
    outer_iteration: int,
    round: int,  # NOTE: required by your call-sites
    mar_iteration: int,
    use_case: int,
    communicated_bytes: int,
    binary_indicator: Optional[Any] = None,
    smoothed_delta_vector: Optional[Any] = None,
) -> Any:
    # 1v6: prepare data to be communicated
    group_key = f"{group_id_str}-{outer_iteration}-{round}-{mar_iteration}"
    if isinstance(model, ModernBERTClassifier):
        state_dict_to_send = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        momentum_to_send = [mv.cpu().clone() for p, mv in zip(model.parameters(), momentum_vector) if p.requires_grad]
    else:
        state_dict_to_send = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        momentum_to_send = [mv.cpu().clone() for mv in momentum_vector]

    # 2v6: store model & momentum vector in group dict and inform other peers via dht and wait for all group members to write
    if use_case == 1:
        shared_model_dict[(group_key, peer_id)] = {"model_state_dict": state_dict_to_send}
    elif use_case == 2:
        shared_model_dict[(group_key, peer_id)] = {
            "model_state_dict": state_dict_to_send,
            "momentum_vector": momentum_to_send,
        }
    elif use_case == 3:
        shared_model_dict[(group_key, peer_id)] = {
            "model_state_dict": state_dict_to_send,
            "momentum_vector": momentum_to_send,
            "binary_indicator": binary_indicator,
            "smoothed_delta_vector": [sdv.cpu().clone() for sdv in smoothed_delta_vector],
        }
    barrier_key = f"cm1_{outer_iteration}-{round}-{mar_iteration}"
    dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)
    _ = wait_for_peers(dht, barrier_key, group)

    # 3v6: try to read models & momentum vectors from all group members and inform other peers via dht
    models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected = [], [], [], []
    for peer in group:
        try:
            entry = shared_model_dict[(group_key, peer)]
            if "model_state_dict" in entry:
                if not (use_case == 1 and peer == peer_id):  # NOTE don't use own model as teacher
                    models_collected.append({k: v.to(device) for k, v in entry["model_state_dict"].items()})
            if "momentum_vector" in entry:
                momentum_vectors_collected.append([m.to(device) for m in entry["momentum_vector"]])
            if "binary_indicator" in entry:
                binary_indicators_collected.append(entry["binary_indicator"])
            if "smoothed_delta_vector" in entry:
                smoothed_delta_vectors_collected.append([sdv.to(device) for sdv in entry["smoothed_delta_vector"]])
        except KeyError:
            logger.warning(f"[{datetime.now()}] WARNING: Peer {peer_id} could not load model/momentum from peer {peer}.")
    barrier_key = f"cm2_{outer_iteration}-{round}-{mar_iteration}"
    dht.store(f"{barrier_key}_{peer_id}", True, expiration_time=hivemind.get_dht_time() + 60)

    # 4v6: track communication traffic by only counting models and momentum vectors collected from other peers
    collected_model_bytes = sum(sum(p.numel() * p.element_size() for p in peer_model.values()) for peer_model in models_collected)
    if use_case == 1:
        communicated_bytes += collected_model_bytes
    else:
        own_model_bytes = sum(p.numel() * p.element_size() for p in state_dict_to_send.values())
        communicated_bytes += collected_model_bytes - own_model_bytes
    if use_case == 2 or use_case == 3:
        collected_momentum_bytes = sum(sum(m.numel() * m.element_size() for m in peer_momentum) for peer_momentum in momentum_vectors_collected)
        own_momentum_bytes = sum(m.numel() * m.element_size() for m in momentum_to_send)
        communicated_bytes += collected_momentum_bytes - own_momentum_bytes
    if use_case == 3:
        others_binary_indicator_bytes = (len(binary_indicators_collected) - 1) * 4  # assume float32
        smoothed_delta_bytes = sum(sum(t.numel() * t.element_size() for t in peer_delta) for peer_delta in smoothed_delta_vectors_collected)
        own_smoothed_delta_bytes = sum(t.numel() * t.element_size() for t in smoothed_delta_vector)
        communicated_bytes += (smoothed_delta_bytes - own_smoothed_delta_bytes) + others_binary_indicator_bytes

    # 5v6: if we are the coordinator cleanup the group dict once all group members have read
    if peer_id == min(group):
        wait_for_peers(dht, barrier_key, group)
        for peer in group:
            try:
                if (group_key, peer) in shared_model_dict:
                    del shared_model_dict[(group_key, peer)]
            except KeyError:
                pass
        torch.cuda.empty_cache()
        gc.collect()

    # 6v6: compute the next chunk index and return it together with collected information
    chunk_index = 0
    if peer_id not in group:
        logger.warning(f"[{datetime.now()}] WARNING: Peer {peer_id} not found in its own group: {group}.")
    else:
        chunk_index = sorted(group).index(peer_id)

    if use_case == 1:
        return models_collected, chunk_index, communicated_bytes
    elif use_case == 2:
        return models_collected, momentum_vectors_collected, chunk_index, communicated_bytes
    elif use_case == 3:
        return models_collected, momentum_vectors_collected, binary_indicators_collected, smoothed_delta_vectors_collected, chunk_index, communicated_bytes


# ==================== Federated Averaging ====================


# Average models and momentum vectors according to FedAvg
def fedavg_aggregation(
    model: Any,
    models_collected: Any,
    momentum_vector: Any,
    momentum_vectors_collected: Any,
) -> tuple[Any, Any]:
    # 1v2: aggregate model weights
    averaged_state_dict = copy.deepcopy(models_collected[0])
    for key in averaged_state_dict:
        stacked = torch.stack([peer_model[key].float() for peer_model in models_collected])
        averaged_state_dict[key] = stacked.mean(dim=0)

    if isinstance(model, ModernBERTClassifier):
        model.load_state_dict(averaged_state_dict, strict=False)  # partial state_dict for head
    else:
        model.load_state_dict(averaged_state_dict, strict=True)

    # 2v2: aggregate momentum vectors
    if isinstance(model, ModernBERTClassifier):
        trainable_params_indices = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
        for i, global_idx in enumerate(trainable_params_indices):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[global_idx] = stacked.mean(dim=0)
    else:
        for i in range(len(momentum_vector)):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[i] = stacked.mean(dim=0)

    return model, momentum_vector