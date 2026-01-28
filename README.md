# MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System

This repository contains the reference implementation for **MAR-FL**, a fully decentralized peer-to-peer Federated Learning system with communication-efficient aggregation, designed to be robust under partial participation and peer churn.

The system was developed as part of a first-author publication at **NeurIPS 2025 – AI4NextG Workshop (Poster)**.

---

## Overview

MAR-FL implements a fully decentralized Federated Learning architecture based on:

- Iterative group-based aggregation (**Moshpit All-Reduce**)
- Peer-to-peer synchronization without a central coordinator
- Optional and fully decentralized **Knowledge Distillation**
- Optional and fully decentralized **Differential Privacy**
- Robustness to peer churn and partial participation

The system is designed for execution on distributed environments (e.g. Slurm-based HPC clusters) and supports GPU-accelerated training.

---

## Repository Structure

- `mar_fl.py` — Main entry point  
- `config_setup.py` — Experiment and runtime configuration  
- `ml_tasks.py` — ML task definitions  
- `moshpit_all_reduce.py` — Group-based aggregation logic  
- `peer_aggregation_logic.py` — Peer-side aggregation  
- `peer_synchronization.py` — Decentralized peer synchronization  
- `differential_privacy.py` — DP mechanisms  
- `shared_memory_handling.py` — Shared-memory monitoring and cleanup  

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
