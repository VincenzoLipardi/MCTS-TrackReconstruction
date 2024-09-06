import os
import pandas as pd
from structure import Circuit
import mcts
import numpy as np
import pickle
import pennylane as qml
from vqe import VQE as vqe


def get_filename(evaluation_function, criteria, budget, branches, iteration, epsilon, stop_deterministic, rollout_type, roll_out_steps, ucb, image, max_depth, qubits, gradient=False, gate_set='continuous'):
    """It creates the string of the file name that have to be saved or read."""
    ro = f'rollout_{rollout_type}/'
    ros = f'_rsteps_{roll_out_steps}'
    stop = '_stop' if stop_deterministic else ''
    
    if isinstance(branches, bool):
        branch = "dpw" if branches else "pw"
    elif isinstance(branches, int):
        branch = f'bf_{branches}'
    else:
        raise TypeError("branches must be either a boolean or an integer")
    
    eps = f'_eps_{epsilon}' if epsilon is not None else ''
    grad = '_gd' if gradient else ''
    
    if image:
        filename = f"{branch}{eps}{ros}{grad}{stop}"
        ro += 'images/'
    else:
        filename = f"{branch}{eps}_budget_{budget}{ros}_run_{iteration}{grad}{stop}"
    
    ucb_ = f'ucb{ucb}/'
    directory = 'experiments/'+str(2**qubits)+'x'+str(2**qubits)+f'/d_{max_depth}/{criteria}/{ucb_}{evaluation_function.__name__}/{gate_set}/{ro}'
    
    return directory, filename


def run_and_savepkl(evaluation_function, criteria, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, ucb, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=False, loading_bar=False):
    """
    Runs the MCTS on the problem defined in evaluation_function and saves the result (the best path) in a .pkl file.
    """
    directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration, epsilon, stop_deterministic, rollout_type, roll_out_steps, ucb, qubits=variable_qubits, image=False, max_depth=max_depth, gate_set=gate_set)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print("Directory created successfully!")
    
    file_path = os.path.join(directory, f'{filename}.pkl')
    
    if not os.path.exists(file_path):
        if isinstance(choices, dict):
            pass
        elif isinstance(choices, list) and len(choices) == 5:
            choices = {'a': choices[0], 'd': choices[1], 's': choices[2], 'c': choices[3], 'p': choices[4]}
        else:
            raise TypeError("choices must be either a dictionary or a list of length 5")
        
        # Define the root note
        root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)
        
        # Run the MCTS algorithm
        final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, criteria=criteria, rollout_type=rollout_type, roll_out_steps=roll_out_steps, choices=choices, epsilon=epsilon, stop_deterministic=stop_deterministic, ucb_value=ucb, verbose=verbose, loading_bar=loading_bar)
        
        # Save the results
        df = pd.DataFrame(final_state)
        df.to_pickle(file_path)
        print(f"Files saved in experiments/ as '{file_path}'")
    else:
        print(f'File already exists: {file_path}')


def add_columns(evaluation_function, criteria, qubits, budget, iter, max_depth, branches, epsilon, stop_deterministic, roll_out_steps, rollout_type, gradient, ucb, gate_set='continuous'):
    """Adds the column of the cost function during the search, and apply the gradient descent on the best circuit and save it the column Adam"""
    # Get best paths

    qc_path = get_paths(evaluation_function, criteria, qubits, max_depth, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, iter)[0]
    directory, filename = get_filename(evaluation_function, criteria, budget, branches, qubits=qubits, ucb=ucb, iteration=iter, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, image=False, max_depth=max_depth)
    df = pd.read_pickle(directory + filename + '.pkl')
    if 'cost' in df.columns:
        print('Cost column already created')
        column_cost = df['cost']
    else:
        # Create column of the cost values along the tree path
        column_cost = list(map(lambda x: evaluation_function(x, cost=True), qc_path))
        # Add the columns to the pickle file
        df['cost'] = column_cost
    if gradient:
        if 'Adam' in df.columns:
            print('Angle parameter optimization already performed')
        else:
            # Get last circuit in the tree path
            quantum_circuit_last = qc_path[-1]

            # Apply gradient on the last circuit and create a column to save it
            final_result = evaluation_function(quantum_circuit_last, ansatz='', cost=False, gradient=True)
            column_adam = [[None]]*df.shape[0]
            column_adam[-1] = final_result

            # Apply gradient on the best circuit if the best is not the last in the path
            if isinstance(column_cost, list):
                index = column_cost.index(min(column_cost))
            else:
                column_cost = np.array(column_cost)
                # index = column_cost.idxmin()
                index = np.argmin(column_cost)
            if index != len(qc_path):
                quantum_circuit_best = qc_path[index]
                best_result = evaluation_function(quantum_circuit_best, ansatz='', cost=False, gradient=True)
                column_adam[index] = best_result
            df["Adam"] = column_adam

        df.to_pickle(os.path.join(directory+filename + '.pkl'))
        print('Columns added to: ', directory+filename)


def postprocessing(evaluation_function, criteria, budget, iter, max_depth, branches, epsilon, stop_deterministic, roll_out_steps, rollout_type, ucb, qubits, gate_set='continuous'):

    directory, filename = get_filename(evaluation_function, criteria, budget, branches, qubits=qubits, ucb=ucb, iteration=iter, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, image=False, max_depth=max_depth)
    df = pd.read_pickle(directory + filename + '.pkl')
    print(directory + filename + '.pkl')
    if 'statevector' in df.columns and 'benchmark' in df.columns and 'success_rate' in df.columns:
        columns_to_delete = ['statevector', 'success_rate', 'benchmark']
        df = df.drop(columns=[col for col in columns_to_delete if col in df.columns])
    elif 'statevector' in df.columns and 'sampled' in df.columns and 'sampled' in df.columns:
        columns_to_delete = ['statevector', 'sampled', 'sampled']
        df = df.drop(columns=[col for col in columns_to_delete if col in df.columns])



    list_indices = df[df['Adam'].apply(lambda x: x != [None])].index
    quantum_circuit = df["qc"][list_indices[0]]
    size = 2**qubits
    with open('vqe_hamiltonian_'+str(size)+'.pkl', 'rb') as file:
        data_loaded = pickle.load(file)
    H = data_loaded['Hamiltonian']
    problem = vqe(H, qubits)
    energy, parameter_values = problem.gradient_descent(quantum_circuit=quantum_circuit)

    parameters = list(quantum_circuit.parameters)
    # Create a ParameterVector with 2 parameters
    param_dict = {parameters[i]: parameter_values[i] for i in range(len(parameters))}

    bound_qc = quantum_circuit.assign_parameters(param_dict)

    # Step 3: Convert the Qiskit circuit to a PennyLane template
    qml_template = qml.from_qiskit(bound_qc)

    # Step 4: Define a PennyLane device
    dev_1 = qml.device("default.qubit", wires=qubits)
    shots = 100000
    dev_2 = qml.device("default.qubit", wires=qubits, shots=shots)

    # Step 5: Define a QNode that will output the state vector
    @qml.qnode(dev_1)
    def circuit():
        qml_template()
        return qml.state()

    @qml.qnode(dev_2)
    def circuit_2():
        qml_template()
        return qml.sample()

    statevector = circuit()
    raw_samples = circuit_2()
    samples = []
    for sam in raw_samples:
        samples.append(int("".join(str(bs) for bs in sam), base=2))

    statevector_sampled = np.bincount(samples, minlength=2 ** qubits) / shots

    new_df = pd.DataFrame({
        'statevector': [statevector],
        'sampled': [statevector_sampled],
    })
    df = pd.concat([df, new_df], axis=1)
    df.to_pickle(os.path.join(directory + filename + '.pkl'))
    print(' Postprocessing Columns added to: ', directory + filename)


def get_paths(evaluation_function, criteria, qubits, max_depth, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, iteration):
    """ It opens the .pkl files and returns quantum circuits along the best path for all the independent run
    :return: four list of lists
    """
    directory, filename = get_filename(evaluation_function, criteria, budget, branches, qubits=qubits, iteration=iteration, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False, ucb=ucb, max_depth=max_depth)

    if os.path.isfile(directory+filename+'.pkl'):
        df = pd.read_pickle(directory+filename+'.pkl')
        qc_along_path = [circuit for circuit in df['qc']]
        children = df['children'].tolist()
        value = df['value'].tolist()
        visits = df['visits'].tolist()
    else:
        return FileNotFoundError
    return qc_along_path, children, visits, value


def same_sign(v1, v2):
    """
    Check if each component of the first vector has the same sign as the corresponding component in the second vector.

    Parameters:
    v1 (numpy array or list): First input vector.
    v2 (numpy array or list): Second input vector.

    Returns:
    int: 1 if all components have the same sign, 0 otherwise.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    # Check if the signs of corresponding elements are the same
    signs_match = np.sign(v1) == np.sign(v2)

    # Return 1 if all elements match, otherwise return 0
    return int(np.all(signs_match))

