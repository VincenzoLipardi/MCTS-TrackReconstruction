import json
import numpy as np
from pennylane.pauli import PauliSentence
from typing import List

def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def load_from_json(file_path): 
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def convert_to_pauli_string(pauli_sentence: PauliSentence) -> List[List[str]]:
    pauli_string = []
    coeffs_list = []
    for term, coefficient in pauli_sentence.items():
        #print(term)#,coefficient)
        #print(pauli_sentence.wires)
        pauli_term = []
        for qubit in range(len(pauli_sentence.wires)):
            op = term[qubit]
            if op == "I":
                pauli_term.append("I")
            elif op == "X":
                pauli_term.append("X")
            elif op == "Y":
                pauli_term.append("Y")
            elif op == "Z":
                pauli_term.append("Z")
        pauli_string.append(pauli_term)
        coeffs_list.append(coefficient)
    return pauli_string, coeffs_list

def reconstruct_matrix(pauli_strings, coefficients):
    n = 2 ** len(pauli_strings[0])  # Dimension of the matrix
    matrix = np.zeros((n, n), dtype=np.complex128)

    for pauli_string, coefficient in zip(pauli_strings, coefficients):
        matrix_term = np.eye(n, dtype=np.complex128)

        for qubit, operator in enumerate(pauli_string):
            if operator == 'X':
                pauli_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            elif operator == 'Y':
                pauli_matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            elif operator == 'Z':
                pauli_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            else:
                pauli_matrix = np.eye(2, dtype=np.complex128)

            if qubit == 0:
                kron_product = pauli_matrix
            else:
                kron_product = np.kron(kron_product, pauli_matrix)

        matrix_term *= coefficient
        matrix += kron_product * coefficient

    return matrix