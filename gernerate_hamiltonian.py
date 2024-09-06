import TrackHHL.trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import TrackHHL.trackhhl.toy.simple_generator as toy
from TrackHHL.utils import *
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

N_MODULES = 5
N_PARTICLES = 2
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0

detector = toy.SimpleDetectorGeometry(
    module_id=list(range(N_MODULES)),
    lx=[LX]*N_MODULES,
    ly=[LY]*N_MODULES,
    z=[i+Z_SPACING for i in range(N_MODULES)])

generator = toy.SimpleGenerator(
    detector_geometry=detector,
    theta_max=np.pi/6)
event = generator.generate_event(N_PARTICLES)
ham = hamiltonian.SimpleHamiltonian(
    epsilon=1e-3,
    gamma=2.0,
    delta=1.0)
ham.construct_hamiltonian(event=event)

matrix = ham.A.todense()
print(np.shape(matrix))
pauli_decomp = qml.pauli_decompose(matrix,pauli=True)
print(pauli_decomp)
pauli_string_result, coeffs_list = convert_to_pauli_string(pauli_decomp)
save_to_json(pauli_string_result, 'pauli_string.json')
save_to_json(coeffs_list, 'coeffs_list.json')

############################################################################
'''
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(matrix)

# Identify the ground state (smallest eigenvalue)
ground_state_index = np.argmin(eigenvalues)
ground_state_eigenvalue = eigenvalues[ground_state_index]
ground_state_eigenvector = eigenvectors[:, ground_state_index]

print("Eigenvalues:")
print(eigenvalues)
print(sum(abs(eigenvectors)))

print("\nGround state eigenvalue:")
print(ground_state_eigenvalue)

print("\nGround state eigenvector:")
print(ground_state_eigenvector)

H = qml.pauli_decompose(matrix)

# Define the device
dev = qml.device("default.qubit", wires=3)

# Define a more general ansatz
def ansatz(params, wires):
    for i in wires:
        qml.Hadamard(i)
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.RY(params[2], wires=wires[2])
    qml.RY(params[3], wires=wires[2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    for i in wires:
        qml.Hadamard(i)
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])
    qml.RY(params[6], wires=wires[2])
    qml.RY(params[7], wires=wires[2])
    qml.CNOT(wires=[1, 2])

# Define the cost function
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params, wires=[0, 1, 2])
    return qml.expval(H)

@qml.qnode(dev)
def state_fn(params):
    ansatz(params, wires=[0, 1, 2])
    return qml.state()


# Initialize parameters
np.random.seed(0)
params = np.random.normal(0, np.pi, 8)

# Choose an optimizer
#opt = qml.QNGOptimizer(stepsize=0.01, approx="block-diag")
opt = qml.AdamOptimizer(0.01)
#opt = NesterovMomentumOptimizer(0.01)
steps = 501

for i in range(steps):
    params = opt.step(cost_fn, params)
    if (i + 1) % 50 == 0:
        print(f"Step {i+1}: cost = {cost_fn(params)}")

# Output the final result
print("Final parameters:", params)
print("Final cost:", cost_fn(params))
print("Final State: ", state_fn(params))


import matplotlib.pyplot as plt
vqe_solution =  abs(state_fn(params))
#vqe_solution_binary = np.where(vqe_solution != 0, 1, 0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

classical_solution = ham.solve_classicaly()

ax1.bar(np.arange(len(classical_solution)), classical_solution, color="forestgreen")
ax1.set_xlim(-0.5, len(classical_solution) - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(len(vqe_solution)), vqe_solution, color="limegreen")
ax2.set_xlim(-0.5, len(vqe_solution) - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")


plt.show()'''