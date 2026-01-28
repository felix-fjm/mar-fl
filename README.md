# MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System

This repository contains the reference implementation for **MAR-FL**, a fully decentralized peer-to-peer Federated Learning system with communication-efficient aggregation, designed to be robust under partial participation and peer churn.

The system was developed as part of a first-author publication at **NeurIPS 2025 – AI4NextG Workshop (Poster)**.

---

## Overview

MAR-FL implements a fully decentralized Federated Learning architecture based on:

- Iterative group-based aggregation (**Moshpit All-Reduce**)
- Peer-to-peer synchronization without a central coordinator (**Distributed Hash Tables**)
- Optional and fully decentralized **Knowledge Distillation**
- Optional and fully decentralized **Differential Privacy**
- Robustness to peer churn and partial participation

The system is designed for execution in multi-process, distributed computing environments (e.g. Slurm-managed HPC clusters), where peer processes are emulated and optionally pinned to dedicated CPU cores or GPUs to approximate decentralized execution.

---

## Repository Structure

- `mar_fl.py` - Main script: experiment launcher, peer-process dispatcher, and emulated peer runtime.
- `config_setup.py` - Configuration (args/defaults), run setup, reproducibility utilities.
- `ml_tasks.py` - Task definitions plus training/evaluation loops (models, datasets, loss, metrics).
- `moshpit_all_reduce.py` - Iterative group-based aggregation (Moshpit All-Reduce) primitives.
- `peer_aggregation_logic.py` - Peer-side aggregation logic (group formation / update application).
- `peer_synchronization.py` - P2P coordination and state sync (peer discovery, rendezvous/scheduling).
- `differential_privacy.py` - Differential Privacy mechanisms (clipping/noise) integrated into training/aggregation.
- `shared_memory_handling.py` - Shared-memory monitoring and cleanup utilities (`/dev/shm`).

---

## Requirements

- Python **3.8**
- CUDA-enabled GPU (tested on NVIDIA H100)
- Linux environment (required for `/dev/shm` monitoring)

---

## Installation

It is recommended to use a **virtual environment**:

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
