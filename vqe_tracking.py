import TrackHHL.trackhhl.toy.simple_generator as toy
from gen_dp import generate_hamiltonian
import TrackHHL.trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import pennylane as qml
import pennylane.numpy as np
from qiskit import QuantumCircuit
from vqe import VQE
import pickle

N_MODULES = 5
N_PARTICLES = 16
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0
params = {
    'alpha': 1.0,
    'beta': 1.0,
    'lambda': 100.0,} 

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

A, b, components, segments = generate_hamiltonian(event, params)
H = qml.pauli_decompose(A)

# Calculate eigenvalues and eigenvectors of the Hamiltonian matrix A
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Identify the ground state (smallest eigenvalue)
ground_state_index = np.argmin(eigenvalues)
#ground_state_eigenvalue = eigenvalues[ground_state_index]
ground_state_eigenvector = eigenvectors[:, ground_state_index]

# Prepare the dictionary to store in the .pkl file
"""data_to_store = {
    'Hamiltonian': H,
    'Ground State Eigenvector': ground_state_eigenvector
}"""

# Save the dictionary in a .pkl file
size = (N_MODULES-1)*(N_PARTICLES**2)
#filename = 'vqe_hamiltonian_'+str(size)+'.pkl'
filename = 'vqe_hamiltonian.pkl'
with open(filename, 'wb') as file:
    pickle.dump(H, file)
    #pickle.dump(data_to_store, file)





'''
print("Eigenvalues:")
print(eigenvalues)
print(sum(abs(eigenvectors)))

print("\nGround state eigenvalue:")
print(ground_state_eigenvalue)

print("\nGround state eigenvector:")
print(ground_state_eigenvector)




# Define the device
dev = qml.device("default.qubit", wires=3)

# Define a more general ansatz
def ansatz(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.RY(params[2], wires=wires[2])
    qml.RY(params[3], wires=wires[3])
    qml.RY(params[8], wires=wires[4])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])
    qml.RY(params[6], wires=wires[2])
    qml.RY(params[7], wires=wires[3])
    qml.RY(params[9], wires=wires[4])
    qml.CNOT(wires=[1, 2])

# Define the cost function
@qml.qnode(dev, diff_method="parameter-shift")
def cost_fn(params):
    ansatz(params, wires=[0, 1, 2, 3, 4])
    return qml.expval(H)

@qml.qnode(dev)
def state_fn(params):
    ansatz(params, wires=[0, 1, 2, 3, 4])
    return qml.state()

# Initialize parameters
np.random.seed(0)
params = np.random.normal(0, np.pi, 10)
theta = np.array(params, requires_grad=True)

# Choose an optimizer
#opt = qml.QNGOptimizer(stepsize=0.01, approx="block-diag")
opt = qml.AdamOptimizer(0.01)
#opt = NesterovMomentumOptimizer(0.01)
steps = 501
conv_tol = 1e-06

for i in range(steps):
    params, prev_val = opt.step_and_cost(cost_fn, params)
    val = cost_fn(params)
    conv = np.abs(val - prev_val)

    if i % 50 == 0:
        print(
            "Iteration = {:},  Cost = {:.8f} ,  Convergence parameter = {"
            ":.8f}".format(i, val, conv))

    #if conv <= conv_tol:
    #    break

# Output the final result
final_cost = cost_fn(params)
final_state = state_fn(params)

print("Final parameters:", params)
print("Final cost:", final_cost)
print("Final state vector:", final_state)

import matplotlib.pyplot as plt

vqe_solution =  ground_state_eigenvector * -1
normalized_array = (vqe_solution - np.min(vqe_solution)) / (np.max(vqe_solution) - np.min(vqe_solution))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(np.arange(len(final_state)), final_state, color="forestgreen")
ax1.set_xlim(-0.5, len(final_state) - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("VQE probabilities")

ax2.bar(np.arange(len(vqe_solution)), vqe_solution, color="limegreen")
ax2.set_xlim(-0.5, len(vqe_solution) - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Classical probabilities")

plt.show()'''