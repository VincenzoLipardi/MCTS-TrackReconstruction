import mcts
import pandas as pd
import math
import os.path
import numpy as np
from structure import Circuit
import matplotlib.pyplot as plt
from evaluation_functions import track_recon

root = mcts.Node(Circuit(variable_qubits=4, ancilla_qubits=0), max_depth=10)
final_state = mcts.mcts(root, budget=1000, branches=False, evaluation_function=track_recon, rollout_type='classical', roll_out_steps=0, choices={'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}, epsilon=None, stop_deterministic=False, verbose=True)
