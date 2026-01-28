# ==================== Imports ====================
from __future__ import annotations
import copy
import math
from typing import Any
import torch


# ==================== Local Imports ====================


# Machine learning tools
from ml_tasks import ModernBERTClassifier


# ==================== Delta Vector ====================


# Compute delta vector between the updated local model and the last global model
def compute_delta_vector(model: Any, last_global_model: Any, device: Any) -> Any:
    last_global_state = {k: v.clone().detach() for k, v in last_global_model.state_dict().items()}
    with torch.no_grad():
        if isinstance(model, ModernBERTClassifier):
            delta_vector = [model.state_dict()[k] - last_global_state[k].to(device) for k, p in model.named_parameters() if p.requires_grad] # NOTE for BERT only compute the delta for the trainable head
        else:
            delta_vector = [model.state_dict()[k] - last_global_state[k].to(device) for k in model.state_dict()]
    return delta_vector


# Clip a delta vector with a clipping threshold and blur it with sampled noise to obtain the noisy clipped delta vector and its binary indicator
def clip_and_blur_delta_vector(
    delta_vector: Any,
    dp_clip_threshold: float,
    z_delta: float,
    num_participating_peers: int,
    device: Any,
) -> tuple[Any, Any, float]:
    
    # 1v2: clip the local delta vector
    global_norm = torch.linalg.norm(torch.cat([d.reshape(-1) for d in delta_vector]))
    binary_indicator = float(global_norm <= dp_clip_threshold)
    scale = min(1.0, dp_clip_threshold / (global_norm + 1e-8)) # NOTE ensure numerical stability by adding 1e-8
    clipped_delta_vector = [d.mul(scale) for d in delta_vector]
    
    # 2v2: add noise to each tensor in the clipped delta vector
    local_noise_std = (z_delta * dp_clip_threshold) / math.sqrt(num_participating_peers)
    noisy_clipped_delta_vector = []
    for d in clipped_delta_vector:
        noise = torch.normal(0, local_noise_std, size=d.shape, device=device)
        noisy_clipped_delta_vector.append(d + noise)
    return noisy_clipped_delta_vector, binary_indicator, global_norm.item()


# Apply a delta vector to a model to obtain an updated model
def apply_delta_to_model(model: Any, delta_vector: Any, eta_s: float = 0.1) -> Any:
    updated_model = copy.deepcopy(model)
    delta_iter = iter(delta_vector)
    with torch.no_grad():
        if isinstance(model, ModernBERTClassifier):
            for param in updated_model.parameters(): # NOTE for BERT iterate only over trainable parameters and apply the delta
                if param.requires_grad:
                    param.add_(next(delta_iter), alpha=eta_s)
        else:
            for param in updated_model.parameters():
                param.add_(next(delta_iter), alpha=eta_s)
    return updated_model


# ==================== Federated Averaging ====================


# Differentially private federated aggregation with adaptive clipping threshold
def dp_federated_aggregation(
    last_mar_iteration: bool,
    num_participating_peers: int,
    model: Any,
    models_collected: Any,
    momentum_vector: Any,
    momentum_vectors_collected: Any,
    binary_indicators_collected: Any,
    smoothed_delta_vectors_collected: Any,
    dp_clip_threshold: float,
    device: Any,
    eta_s: float = 0.1,
    eta_c: float = 0.2,
    gamma: float = 0.5,
    beta: float = 0.9,
) -> tuple[Any, Any, Any, Any]:
    
    # 1v5: aggregate models
    averaged_state_dict = copy.deepcopy(models_collected[0])
    for key in averaged_state_dict:
        stacked = torch.stack([peer_model[key].float() for peer_model in models_collected])
        averaged_state_dict[key] = stacked.mean(dim=0)
    if isinstance(model, ModernBERTClassifier):
        model.load_state_dict(averaged_state_dict, strict=False) # NOTE use strict=False to load the partial state_dict for the head
    else:
        model.load_state_dict(averaged_state_dict, strict=True)

    # 2v5: aggregate momentum vectors
    if isinstance(model, ModernBERTClassifier):
        trainable_params_indices = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
        for i, global_idx in enumerate(trainable_params_indices):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[global_idx] = stacked.mean(dim=0)
    else:
        for i in range(len(momentum_vector)):
            stacked = torch.stack([mv[i].float() for mv in momentum_vectors_collected])
            momentum_vector[i] = stacked.mean(dim=0)

    # 3v5: aggregate binary indicators
    indicators_tensor = torch.tensor(binary_indicators_collected, dtype=torch.float32, device=device)
    mean_indicator = indicators_tensor.mean()

    # 4v6: aggregate and reshape smoothed delta vectors
    stacked_smoothed_deltas = [torch.cat([param.reshape(-1) for param in delta]) for delta in smoothed_delta_vectors_collected]
    aggregated_smoothed_delta = torch.stack(stacked_smoothed_deltas, dim=0).mean(0)
    if isinstance(model, ModernBERTClassifier):
        template_tensors = [p for p in model.parameters() if p.requires_grad]
    else:
        template_tensors = list(model.state_dict().values())
    param_shapes = [p.shape for p in template_tensors]
    param_lengths = [p.numel() for p in template_tensors]
    reconstructed_smoothed_delta = []
    current_pos = 0
    for shape, length in zip(param_shapes, param_lengths):
        chunk = aggregated_smoothed_delta[current_pos : current_pos + length]
        reconstructed_smoothed_delta.append(chunk.view(shape))
        current_pos += length

    if last_mar_iteration: # NOTE if last MAR iteration

        # 5v6: add noise to aggregated indicator
        sigma_b = num_participating_peers / 20.0
        noise_b = torch.normal(mean=0.0, std=sigma_b, size=(1,), device=device) / num_participating_peers # NOTE divide by num_participating_peers because we add noise to the average while the paper adds noise to the not yet divided sum
        noisy_b_t = mean_indicator + noise_b
        noisy_b_t = torch.clamp(noisy_b_t, 0.0, 1.0)

        # 6v6: update clipping threshold
        updated_clip_threshold = dp_clip_threshold * torch.exp(-eta_c * (noisy_b_t - gamma))
        return model, momentum_vector, reconstructed_smoothed_delta, updated_clip_threshold.item()

    else: # NOTE if not last MAR iteration     
        return model, momentum_vector, mean_indicator.item(), reconstructed_smoothed_delta


# ==================== Privacy Budget ====================


# Compute the so far used privacy budget based on moments accountant approximation
def compute_epsilon(steps: int, delta: float, sample_rate: float, noise_multiplier: float) -> float:
    if noise_multiplier == 0:
        return float('inf')
    
    # 1v2: convert inputs
    sigma = noise_multiplier
    q = sample_rate
    T = steps
    l = 32 # NOTE common choice for order of Renyi Divergence

    # 2v2: compute Renyi DP (RDP) and convert to (ε, δ)-DP
    rdp = (q**2) * T * l / (2 * sigma**2)
    epsilon = rdp + math.log(1/delta) / (l - 1)
    return epsilon


# Sweep the so far used privacy budget based on moments accountant approximation in an enhanced procedure even closer to opacus
def sweep_epsilon_enhanced(steps: int, delta: float, sample_rate: float, noise_multiplier: float) -> float:
    if noise_multiplier == 0:
        return float('inf')
    
    # 1v2: convert inputs
    sigma = noise_multiplier
    q = sample_rate
    T = steps
    min_eps = float('inf')

    # 2v2: sweep over a range of l values (orders of Renyi Divergence) from 2 to 64 and compute the respective epsilon
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 65)) # NOTE create a list of l-values inspired by the opacus library's default
    for l in orders:
        if l == 1.0:
            continue
        rdp = (q**2) * T * l / (2 * sigma**2)
        eps = rdp + math.log(1/delta) / (l - 1)
        if eps < min_eps:
            min_eps = eps
    return min_eps