# AlphaZero-Inspired Chessbot

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This is a high-performance, AlphaZero-inspired chess engine built from scratch in PyTorch. It features a continuous reinforcement learning loop that combines supervised pre-training on expert human games with self-play refinement using Monte Carlo Tree Search (MCTS).

The system is optimized for modern hardware, using **Automatic Mixed Precision (AMP)** and **JIT compilation** to accelerate inference and training throughput by **~30%**.

## Key Features

* **Hybrid Training Pipeline:** Bootstraps knowledge from a **14M+ expert games** dataset via HuggingFace streaming before refining strategies through self-play.
* **AlphaZero Architecture:** Implements a ResNet-based policy-value network guided by Monte Carlo Tree Search (MCTS).
* **High-Performance Inference, Achieving a 3.70x Increase in Speed:**
    * **JIT Compilation:** Uses `torch.compile` (PyTorch 2.0+) to reduce Python overhead.
    * **Mixed Precision:** Full support for FP16/TensorFloat32 on NVIDIA Ampere GPUs.
* **Cloud-Native Design:** Built to run seamlessly on **Modal** for distribute, faster training and massive parallel game generation.

## Architecture
The engine uses a **Residual Convolutional Neural Network (ResNet)** with the following specifications:

* **Input:** 18-channel board representation (Piece locations, turns, castling rights, etc.).
* **Backbone:** 12 Residual Blocks with 128 filters (lightweight for fast iteration).
* **Heads:**
    * **Policy Head:** Outputs move probabilities (Softmax).
    * **Value Head:** Outputs position evaluation (Tanh, -1 to 1).
**Search Algorithm:** Monte Carlo Tree Search (MCTS) with Upper Confidence Bound applied to Trees (UCT).


## Improvements 
This project is still being developed to further improve its performance, and integrate it into a website so it is more accessible and easier to play.

## Acknowledgements
This project was created during the Chesshacks hackathon in collaboration with Diya and Ian, as seen with their initial "spec/engine-improvement" branch. The initial framework for the project was also supplied by the hackathon. 
Recently, I have continued this project as a personal endeavour, improving the bot's performance and completely rewriting the code. These new changes are reflected on the new 'main' branch. 
