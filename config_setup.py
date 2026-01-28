# ==================== Imports ====================
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional
import wandb


# ==================== Local Imports ====================


# Machine learning tools
from ml_tasks import (
    get_mnist_data_loaders,
    get_mnist_data_loaders_dirichlet,
    get_text_data_loaders,
    get_text_data_loaders_dirichlet,
    set_all_seeds,
)

# ==================== Config ====================


# Data class for console arguments
@dataclass(frozen=True)
class FLConfig:
    # General & hardware
    seed: int
    ml_task: str
    device: str
    num_peers: int
    peers_per_core: int
    max_runtime: float

    # MAR protocol
    mar_group_size: int
    mar_group_key_dim: int
    mar_num_iterations: int
    local_update_participation_prob: float
    aggregation_participation_prob: float
    peer_dropout_likelihood: float

    # Training hyperparameters
    max_iterations: int
    mini_batches_per_iteration: int
    learning_rate: float
    momentum: float
    split_type: str
    dirichlet_alpha: float
    testing_frequency: int
    convergence_patience: int
    convergence_threshold: float
    dispatcher_patience: int

    # Knowledge distillation
    knowledge_distillation: bool
    knowledge_distillation_iters: int
    knowledge_distillation_no_blending: bool
    knowledge_distillation_use_all_candidates: bool

    # Differential privacy
    do_dp: bool
    dp_clip_threshold: float
    dp_noise_multiplier: float


# Generate a unique and informative wandb run name
def generate_unique_run_name(
    split_type: str,
    entity: str,
    project: str,
    seed: int,
    ml_task: str,
    num_peers: int,
    mini_batches_per_iteration: int,
    aggregation_participation_prob: int,
    local_update_participation_prob: int,
    peer_dropout_likelihood: int,
    do_kd: bool,
    kd_iters: int,
    kd_all_candidates: bool,
    do_dp: bool,
    dp_noise_multiplier: Optional[float],
    mar_group_size: int,
    mar_num_iters: int,
) -> str:
    api = wandb.Api()
    if kd_all_candidates:
        top_k_ratio = 1.0
    else:
        top_k_ratio = 0.4
    if do_kd:
        if do_dp:
            base_name = f"mar{num_peers}{ml_task}-k{kd_iters}-{top_k_ratio}-dp{dp_noise_multiplier}_b{mini_batches_per_iteration}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}_g{mar_group_size}_i{mar_num_iters}__s{seed}"
        else:
            base_name = f"mar{num_peers}{ml_task}-k{kd_iters}-{top_k_ratio}_b{mini_batches_per_iteration}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}_g{mar_group_size}_i{mar_num_iters}__s{seed}"
    else:
        if do_dp:
            base_name = f"mar{num_peers}{ml_task}-dp{dp_noise_multiplier}_b{mini_batches_per_iteration}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}_g{mar_group_size}_i{mar_num_iters}__s{seed}"
        else:
            base_name = f"mar{num_peers}{ml_task}_b{mini_batches_per_iteration}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}_g{mar_group_size}_i{mar_num_iters}__s{seed}"
            if local_update_participation_prob < 100:
                base_name = f"mar{num_peers}{ml_task}_b{mini_batches_per_iteration}_u{local_update_participation_prob}_a{aggregation_participation_prob}_d{peer_dropout_likelihood}_g{mar_group_size}_i{mar_num_iters}__s{seed}"
    if split_type == "dirichlet":
        base_name += "_niid"
    existing_runs = api.runs(f"{entity}/{project}", filters={"display_name": {"$regex": f"^{base_name}(_\\d+)?$"}})
    existing_names = {run.name for run in existing_runs}
    if base_name not in existing_names:
        return base_name
    else:
        suffix = 1
        while f"{base_name}_{suffix}" in existing_names:
            suffix += 1
        return f"{base_name}_{suffix}"


# Parsing of console arguments
def parse_config() -> Tuple[FLConfig, str]:
    # Create parser
    parser = argparse.ArgumentParser(description="MAR-FL")

    # General & hardware
    parser.add_argument("--sd", type=int, default=42, help="Seed for random value generation")
    parser.add_argument("--ml", type=str, default="mnist", help="ML task (mnist or news)")
    parser.add_argument("--dv", type=str, default="cuda", help="Device to use (cpu or cuda)")
    parser.add_argument("--np", type=int, default=64, help="Number of peers")
    parser.add_argument("--pp", type=int, default=1, help="Peers per core")
    parser.add_argument("--rt", type=int, default=12, help="Maximum allowed runtime in hours")

    # MAR protocol
    parser.add_argument("--gs", type=int, default=4, help="Moshpit All-Reduce group size (optimally np=gs^ni)")
    parser.add_argument("--gk", type=int, default=3, help="Moshpit All-Reduce group key dimension (optimally gk=ni)")
    parser.add_argument("--ni", type=int, default=3, help="Moshpit All-Reduce number of iterations (optimally np=gs^ni & gk=ni)")
    parser.add_argument("--up", type=int, default=100, help="Local update participation probability (between 0 and 100)")
    parser.add_argument("--ap", type=int, default=100, help="Aggregation participation probability (between 0 and 100)")
    parser.add_argument("--dl", type=int, default=0, help="Peer dropout likelihood (between 0 and 100)")

    # Training hyperparameters
    parser.add_argument("--im", type=int, default=0, help="Iterations mechanism approach (0 for 'mini batches per iteration' or 1 for 'iterations per dataset')")
    parser.add_argument("--iv", type=int, default=4, help="Iterations mechanism value (lower if higher number of peers)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate value (default 0.1)")
    parser.add_argument("--mm", type=float, default=0.9, help="Momentum value (default 0.9)")
    parser.add_argument("--sp", type=str, default="iid", choices=["iid","dirichlet"], help="How to split training data across peers")
    parser.add_argument("--al", type=float, default=1.0, help="Dirichlet alpha for non-IID when --sp=dirichlet")
    parser.add_argument("--tf", type=int, default=5, help="Testing frequency (default 5)")
    parser.add_argument("--cd", type=float, default=1e-3, help="Convergence threshold (default 1e-3)")
    parser.add_argument("--dp", type=int, default=180, help="Dispatcher patience (default 180 seconds)")

    # Knowledge distillation
    parser.add_argument("--kd", action="store_true", help="Use knowledge distillation during model aggregation?")
    parser.add_argument("--nb", action="store_true", help="Use knowledge distillation without blending during the entire run?")
    parser.add_argument("--ds", type=int, default=10, help="How often should a peer see his entire dataset while doing KD?")
    parser.add_argument("--ac", action="store_true", help="Distill knowledge from all collected candidate teacher models?")

    # Differential privacy
    parser.add_argument("--dp", action="store_true", help="Use differential privacy?")
    parser.add_argument("--nm", type=float, default=None, help="Noise multiplier for differential privacy")
    parser.add_argument("--ct", type=float, default=None, help="Initial clipping threshold for differential privacy")
    
    # Argument parsing
    args = parser.parse_args()

    # Max number of iterations and convergence patience
    max_iterations = 500
    if args.ml == "news":
        max_iterations = 1000
    convergence_patience = max_iterations * args.tf

    # Set random seeds for reproducibility
    seed = args.sd
    set_all_seeds(args.sd)

    # Number of mini-batches per iteration
    if args.ml == "news":
        if args.sp == "dirichlet":
            mock_train_loader, _, _ = get_text_data_loaders_dirichlet(partition=0, num_partitions=args.np, seed=args.sd, alpha=args.al)
        else:
            mock_train_loader, _, _ = get_text_data_loaders(partition=0, num_partitions=args.np, seed=args.sd)
    elif args.ml == "mnist":
        if args.sp == "dirichlet":
            mock_train_loader, _, _ = get_mnist_data_loaders_dirichlet(partition=0, num_partitions=args.np, seed=args.sd, alpha=args.al)
        else:
            mock_train_loader, _, _ = get_mnist_data_loaders(partition=0, num_partitions=args.np, seed=args.sd)
    num_batches_per_peer = len(mock_train_loader)
    if args.im == 0:
        mini_batches_per_iteration = args.iv
    elif args.im == 1:
        mini_batches_per_iteration = math.ceil(num_batches_per_peer / args.iv)
    else:
        raise RuntimeError("ERROR: Invalid iterations mechanism approach!")
    
    # If necessary determine number of iterations with knowledge distillation
    knowledge_distillation_iters = 0
    if args.nb:
        knowledge_distillation_iters = max_iterations # NOTE knowledge distillation without blending during the entire run
    else:
        if args.kd:
            knowledge_distillation_iters = args.ds * int(num_batches_per_peer / mini_batches_per_iteration) # NOTE each peer sees his entire training dataloader's batches args.ds times while using KD
    if args.kd: # NOTE to account for non-IID case
        if args.ml == "mnist": 
            if args.ds == 2:
                knowledge_distillation_iters = 8
            elif args.ds == 10:
                knowledge_distillation_iters = 40
        elif args.ml == "news":
            if args.ds == 2:
                knowledge_distillation_iters = 6
            elif args.ds == 10:
                knowledge_distillation_iters = 30

    # If not specified choose default DP values
    if args.ct is None:
        args.ct = 0.1
    if args.nm is None:
        args.nm = 1.0

    # If necessary adjust DP noise multiplier value # NOTE we must avoid negative (z_total**-2 - (2 * sigma_b)**-2) terms when calculating z_delta
    if args.dp and args.up < 100:
        args.ap = args.up # NOTE when doing DP we must have the same set of peers participating in update and aggregation to correctly compute the privacy budget unless every peer always conducts local updates
    minimum_sigma_b = int(args.np * args.ap / 100) / 20
    if args.nm >= 2 * minimum_sigma_b:
        old_noise_multiplier = args.nm
        args.nm = 2 * minimum_sigma_b - 0.01
        
    # Wandb run naming
    wandb_entity = "FedMoshpitAvg"
    wandb_project = "p2p_fl_niid"
    unique_run_name = generate_unique_run_name(args.sp, wandb_entity, wandb_project, args.sd, args.ml, args.np, mini_batches_per_iteration, args.ap, args.up, args.dl, args.kd, knowledge_distillation_iters, args.ac, args.dp, args.nm, args.gs, args.ni)

    return (
        FLConfig(
            # General & hardware
            seed=args.sd,
            ml_task=args.ml,
            device=args.dv,
            num_peers=args.np,
            peers_per_core=args.pp,
            max_runtime=args.rt,
            
            # MAR protocol
            mar_group_size=args.gs,
            mar_group_key_dim=args.gk,
            mar_num_iterations=args.ni,
            local_update_participation_prob=args.up,
            aggregation_participation_prob=args.ap,
            peer_dropout_likelihood=args.dl,
            
            # Training hyperparameters
            max_iterations=max_iterations,
            mini_batches_per_iteration=mini_batches_per_iteration,
            learning_rate=args.lr,
            momentum=args.mm,
            split_type=args.sp,
            dirichlet_alpha=args.al,
            testing_frequency=args.tf,
            convergence_patience=convergence_patience,
            convergence_threshold=args.cd,
            dispatcher_patience=args.dp,
            
            # Knowledge distillation
            knowledge_distillation=args.kd,
            knowledge_distillation_iters=knowledge_distillation_iters,
            knowledge_distillation_no_blending=args.nb,
            knowledge_distillation_use_all_candidates=args.ac,
            
            # Differential privacy
            do_dp=args.dp,
            dp_clip_threshold=args.ct,
            dp_noise_multiplier=args.nm
        ), 
        unique_run_name
    )