# ZeroChess-RL ♟️

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**ZeroChess-RL** is a high-performance, AlphaZero-inspired chess engine built from scratch in PyTorch. It features a continuous reinforcement learning loop that combines supervised pre-training on expert human games with self-play refinement using Monte Carlo Tree Search (MCTS).

The system is optimized for modern hardware, utilizing **Automatic Mixed Precision (AMP)** and **JIT compilation** to accelerate inference and training throughput by ~30%.

---

## 🚀 Key Features

* **Hybrid Training Pipeline:** Bootstraps knowledge from **14M+ expert games** (Lichess/CCRL) via Hugging Face streaming before refining strategies through self-play.
* **AlphaZero Architecture:** Implements a ResNet-based policy-value network guided by Monte Carlo Tree Search (MCTS).
* **High-Performance Inference:**
    * **JIT Compilation:** Uses `torch.compile` (PyTorch 2.0+) to reduce Python overhead.
    * **Mixed Precision:** Full support for FP16/TensorFloat32 on NVIDIA Ampere GPUs.
* **Cloud-Native Design:** Built to run seamlessly on **Modal** for distributed training and massive parallel game generation.
* **Efficient Data Streaming:** Custom data ingestion pipeline that processes millions of games without requiring terabytes of local storage.

---

## 🛠️ Architecture

The engine uses a **Residual Convolutional Neural Network (ResNet)** with the following specifications (configurable):

* **Input:** 18-channel board representation (Piece locations, turns, castling rights, etc.).
* **Backbone:** 3-5 Residual Blocks with 64-128 filters (lightweight for fast iteration).
* **Heads:**
    * **Policy Head:** Outputs move probabilities (Softmax).
    * **Value Head:** Outputs position evaluation (Tanh, -1 to 1).

**Search Algorithm:** Monte Carlo Tree Search (MCTS) with Upper Confidence Bound applied to Trees (UCT).

```bash
# Process 20,000 games and 10,000 puzzles
python data_ingestion.py --games 20000 --puzzles 10000
