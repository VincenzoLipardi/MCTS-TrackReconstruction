
# MCTS for Quantum Ansatz Search in Variational Quantum Algorithms for Particle Track Reconstruction

## Overview

This repository presents an open-access code related to the paper "Variational Quantum Algorithms for Particle Track reconstruction" submitted to BNAIC 2024. We used a Monte Carlo Tree Search (MCTS) approach to design quantum ansatz for two different formulation of the particle track reconstruction problem within the framework of variational quantum algorithms.

## Key Concepts

- **Monte Carlo Tree Search (MCTS)**: MCTS is employed as a search strategy to navigate the space of quantum ansatz configurations. By iteratively building and exploring a search tree, MCTS guides the selection of promising ansatz configurations for further evaluation.

- **Variational Quantum Algorithms**: Particle Reconstruction is encoded as a variational quantum optimization problem, where the goal is to find the optimal parameters for a parameterized quantum circuit (ansatz) that minimizes a cost function representing the reconstruction quality.

- **Quantum Ansatz**: A parameterized quantum circuit is chosen as the ansatz, with the parameters representing the degrees of freedom to be optimized. MCTS efficiently explores the space of these parameters to find configurations that yield accurate reconstructions.

## Usage

1. **Clone the Repository**:

- setup a conda environment with python 3.10
- install the requirements with the following command: pip install -r requirements.txt
