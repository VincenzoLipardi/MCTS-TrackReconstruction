
# MCTS for Quantum Ansatz Search in Variational Quantum Algorithms for Particle Track Reconstruction

## Overview

This repository presents an open-access code related to the paper "Variational Quantum Algorithms for Particle Track reconstruction" submitted to BNAIC 2024. We used a Monte Carlo Tree Search (MCTS) approach to design quantum ansatz for two different formulation of the particle track reconstruction problem within the framework of variational quantum algorithms (VQAs).

## Key Concepts

- **Monte Carlo Tree Search (MCTS)**: classical search technique employed to design quantum circuits. At each node correspond a quantum circuit and at each edge (action/move) a modification of it. 
- **Variational Quantum Algorithms**: The particle track reconstruction is encoded as a variational quantum problem in two different setting.
- **Quantum Ansatz**: Parameterized quantum circuit chosen to sove a certain variational quantum algorithm by optimizing its angle parameters.

## Usage

1. **Clone the Repository**:

- Setup a conda environment with python 3.10
- Install the requirements with the following command: pip install -r requirements.txt

2. **Navigate in the project**

- Particle Track Problem encoded in the framework of VQAs": [evaluation_functions.py](/evaluation_functions.py), [vqe_tracking.py](/vqe_trackings.py), [vqls_tracking.py](/vqls_tracking.py).
- MCTS implementation: [mcts.py](/mcts.py)
