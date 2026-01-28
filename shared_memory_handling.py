# ==================== Imports ====================
from __future__ import annotations
import pathlib
import threading
import time
from datetime import datetime
from typing import Any, Mapping, MutableMapping, Optional, Sequence
import wandb


# ==================== Shared Memory Usage ====================


# Shared memory load measurement regarding python multiprocessing and torch
def _dev_shm_usage_bytes(select_prefixes: Sequence[str] = ("psm_", "torch_")) -> int:
    total = 0
    shm_path = pathlib.Path("/dev/shm")
    if not shm_path.exists():  # NOTE not a Linux box or no tmpfs mounted
        return 0
    for f in shm_path.iterdir():
        try:
            if not select_prefixes or f.name.startswith(select_prefixes):
                total += f.stat().st_size
        except FileNotFoundError:  # NOTE the file disappeared between listdir and stat – ignore
            pass
    return total


# Shared memory load measurement independent of library prefix
def _dev_shm_total_usage_bytes() -> int:
    total = 0
    shm_path = pathlib.Path("/dev/shm")
    if not shm_path.exists():
        return 0
    for f in shm_path.iterdir():
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total


# Shared memory logging
def _log_shared_memory(
    *,
    logger: Any,
    prefix: str,
    shared_dict: Mapping[str, Any],
    step: Optional[int] = None,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mb = _dev_shm_usage_bytes() / 1024 / 1024
    n_entries = len(shared_dict)
    logger.info(f"[{ts}][{prefix}] /dev/shm ≈ {mb:8.1f} MB | shared_model_dict entries: {n_entries}")
    log_dict = {
        "shm_intermediate_MB": mb,
        "shm_shared_dict_entries": n_entries,
    }
    if step is not None:
        log_dict["shm_step"] = step
    try:
        wandb.log(log_dict)
    except Exception as e:
        logger.info(f"[{ts}][{prefix}] WARNING: wandb.log failed: {e}")


# Constant shared memory watcher
def start_shm_watcher(
    *,
    logger: Any,
    shutdown_event: threading.Event,
    shared_dict: Mapping[str, Any],
    prefix: str,
    interval: int = 30,
) -> threading.Thread:
    def _worker() -> None:
        while not shutdown_event.is_set():
            _log_shared_memory(logger=logger, prefix=prefix, shared_dict=shared_dict)
            time.sleep(interval)
        _log_shared_memory(logger=logger, prefix=prefix + "-final", shared_dict=shared_dict)
    t = threading.Thread(target=_worker, name=f"{prefix}-shm-watcher", daemon=True)
    t.start()
    return t


# Periodic cleanup of the shared dictionary to avoid filling up the shared memory with stale data
def periodic_dict_cleanup(
    *,
    logger: Any,
    shared_dict: MutableMapping[Any, Any],
    current_iteration: int,
) -> None:
    if current_iteration == 0:
        return
    try:
        if not shared_dict.keys():
            logger.info(
                f"[Dispatcher] Periodic cleanup on iteration {current_iteration}: Shared dictionary is already empty."
            )
            return
        logger.info("[Dispatcher Cleanup] Clearing all stale entries from previous iterations...")
        shared_dict.clear()
        logger.info("[Dispatcher Cleanup] Shared dictionary cleared successfully.")
    except Exception as e:
        logger.info(f"[Dispatcher Cleanup] Error during shared dictionary cleanup: {e}")
