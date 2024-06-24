import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from TrackHHL.utils import *

pauli_string = load_from_json('pauli_string.json')
c = load_from_json('coeffs_list.json')

pauli_decomp = qml.pauli_decompose(matrix)

def parse_hamiltonian(linear_combination):
    coeffs = linear_combination.coeffs
    observables = linear_combination.terms
    return coeffs, observables

# Parse the Hamiltonian
coeffs, observables = parse_hamiltonian(pauli_decomp)
H = qml.Hamiltonian(coeffs, observables)

# Define the device
dev = qml.device("default.qubit", wires=3)

# Define a more general ansatz
def ansatz(params, wires):
    qml.BasisState(np.array([0, 0, 0]), wires=wires)
    qml.RX(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[2])
    qml.CNOT(wires=[0, 1])
    qml.RX(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.RZ(params[5], wires=wires[2])
    qml.CNOT(wires=[1, 2])

# Define the cost function
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params, wires=[0, 1, 2])
    return qml.expval(H)

# Initialize parameters
np.random.seed(0)
params = np.random.normal(0, np.pi, 6)

# Choose an optimizer
opt = NesterovMomentumOptimizer(0.01)
steps = 100

for i in range(steps):
    params = opt.step(cost_fn, params)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1}: cost = {cost_fn(params)}")

# Output the final result
print("Final parameters:", params)
print("Final cost:", cost_fn(params))