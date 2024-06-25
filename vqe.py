import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
from pennylane.ops.op_math import Sum

class VQE:
    def __init__(self, hamiltonian, n_qubits):
        if isinstance(hamiltonian, Sum):
            self.hamiltonian = hamiltonian
        else:
            if is_hermitian(hamiltonian):
                self.hamiltonian = qml.Hermitian(hamiltonian, wires=range(0, n_qubits))
            else:
                self.hamiltonian = qml.Hermitian(0.5 * (hamiltonian + hamiltonian.T.conj()), wires=range(0, n_qubits))
        self.dev = qml.device('default.qubit', wires=n_qubits)

    def costFunc(self, params, quantum_circuit=None, ansatz=""):
        """
        Expectation value of the ansatz state on the Hamiltonian.
        """
        if quantum_circuit is None:
            raise ValueError("quantum_circuit is None")

        def circuit(parameters):
            i = 0
            for instr, qubits, _ in quantum_circuit.data:
                name = instr.name.lower()
                if name == "rx":
                    qml.RX(instr.params[0] if ansatz == 'all' else parameters[i], wires=qubits[0].index)
                    if ansatz != 'all':
                        i += 1
                elif name == "ry":
                    qml.RY(instr.params[0] if ansatz == 'all' else parameters[i], wires=qubits[0].index)
                    if ansatz != 'all':
                        i += 1
                elif name == "rz":
                    qml.RZ(instr.params[0] if ansatz == 'all' else parameters[i], wires=qubits[0].index)
                    if ansatz != 'all':
                        i += 1
                elif name == "h":
                    qml.Hadamard(wires=qubits[0].index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0].index, qubits[1].index])

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def cost_fn(parameters):
            circuit(parameters)
            return qml.expval(self.hamiltonian)

        return cost_fn(params)

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return -self.costFunc(params, quantum_circuit, ansatz)

    def gradient_descent(self, quantum_circuit, max_iterations=500, conv_tol=1e-08):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)

        def cost_fn(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit, ansatz='')

        energy = [cost_fn(theta)]
        angle = [theta]

        for n in range(max_iterations):
            theta, prev_energy = opt.step_and_cost(cost_fn, theta)
            current_energy = cost_fn(theta)
            energy.append(current_energy)
            angle.append(theta)

            conv = np.abs(current_energy - prev_energy)

            if n % 2 == 0:
                print(f"Step = {n}, Energy = {current_energy:.8f} Ha")

            if conv <= conv_tol:
                print('Landscape is flat')
                break

        return energy


def get_parameters(quantum_circuit):
    parameters = []
    for instr, _, _ in quantum_circuit.data:
        if len(instr.params) > 0:
            parameters.append(instr.params[0])
    return parameters

def is_hermitian(matrix):
    """
    Check if a given matrix is Hermitian.

    A matrix is Hermitian if it is equal to its own conjugate transpose.

    Parameters:
    matrix (np.ndarray): The matrix to check.

    Returns:
    bool: True if the matrix is Hermitian, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is equal to its conjugate transpose
    return np.allclose(matrix, matrix.T.conj())



# Tutorial with a random matrix
matrix = np.random.rand(4, 4)

trial = VQE(hamiltonian=matrix, n_qubits=2)
qc = QuantumCircuit(2)
qc.h(0)
qc.ry(0.1, 1)
print("Gradient Descent on the angle parameters")
trial.gradient_descent(quantum_circuit=qc)


