import heapq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from vqls_tracking import VQLS_XENO, load_from_json
from vqe import VQE
import pickle


def track_vqls(quantum_circuit, ansatz='all', cost=False, gradient=False):
    # Define the problem A = c_0 I + c_1 X_1 + c_2 X_2 + c_3 Z_3 Z_4
    problem = vqls_xeno

    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def track_vqe(quantum_circuit, ansatz='all', cost=False, gradient=False):
    filename = 'vqe_hamiltonian.pkl'
    with open(filename, 'rb') as file:
        loaded_H = pickle.load(file)
    problem = VQE(hamiltonian=loaded_H, n_qubits=len(loaded_H.wires))
    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def track_vqe_multi(quantum_circuit, ansatz='all', cost=False, gradient=False):
    filename = 'vqe_hamiltonian_multi_16.pkl'
    try:
        with open(filename, 'rb') as file:
            multi_H = pickle.load(file)
        print("Hamiltonian loaded successfully:", multi_H)
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled. It might be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")

    problem = VQE(hamiltonian=multi_H, n_qubits=4)
    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


# SYSTEMS OF LINEAR EQUATIONS
#pauli_string = load_from_json('pauli_string.json')
#c = load_from_json('coeffs_list.json')
## print(c, pauli_string)
#vqls_xeno = VQLS_XENO(c=c, pauli_string=pauli_string, qubits=4)