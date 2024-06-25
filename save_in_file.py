import os
import pandas as pd
from structure import Circuit
import mcts
import numpy as np


def get_filename(evaluation_function, criteria, budget, branches, iteration, epsilon, stop_deterministic, rollout_type, roll_out_steps, ucb, image, gradient=False, gate_set='continuous'):
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
    directory = f'experiments/{criteria}/{ucb_}{evaluation_function.__name__}/{gate_set}/{ro}'
    
    return directory, filename

def run_and_savepkl(evaluation_function, criteria, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, ucb, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=False, loading_bar=False):
    """
    Runs the MCTS on the problem defined in evaluation_function and saves the result (the best path) in a .pkl file.
    """
    directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration, epsilon, stop_deterministic, rollout_type, roll_out_steps, ucb, image=False, gate_set=gate_set)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created successfully!")
    
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

def add_columns(evaluation_function, criteria, budget, n_iter, branches, epsilon, stop_deterministic, roll_out_steps, rollout_type, gradient, ucb, gate_set='continuous'):
    """Adds the column of the cost function during the search, and apply the gradient descent on the best circuit and save it the column Adam"""
    # Get best paths
    qc_path = get_paths(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, n_iter)[0]

    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, ucb=ucb, iteration=i, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, image=False)
        df = pd.read_pickle(directory + filename + '.pkl')
        if 'cost' in df.columns:
            print('Cost column already created')
            column_cost = df['cost']
        else:
            # Create column of the cost values along the tree path
            column_cost = list(map(lambda x: evaluation_function(x, cost=True), qc_path[i]))
            # Add the columns to the pickle file
            df['cost'] = column_cost
        if gradient:
            if 'Adam' in df.columns:
                print('Angle parameter optimization already performed')
            else:
                # Get last circuit in the tree path
                quantum_circuit_last = qc_path[i][-1]

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
                if index != len(qc_path[i]):
                    quantum_circuit_best = qc_path[i][index]
                    best_result = evaluation_function(quantum_circuit_best, ansatz='', cost=False, gradient=True)
                    column_adam[index] = best_result
                df["Adam"] = column_adam

        df.to_pickle(os.path.join(directory+filename + '.pkl'))
        print('Columns added to: ', directory+filename)

        
def get_paths(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, n_iter=10):
    """ It opens the .pkl files and returns quantum circuits along the best path for all the independent run
    :return: four list of lists
    """
    qc_along_path = []
    children, visits, value = [], [], []
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration=i, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False, ucb=ucb)

        if os.path.isfile(directory+filename+'.pkl'):
            df = pd.read_pickle(directory+filename+'.pkl')
            qc_along_path.append([circuit for circuit in df['qc']])
            children.append(df['children'].tolist())
            value.append(df['value'].tolist())
            visits. append(df['visits'].tolist())
        else:
            print('here')
            return FileNotFoundError
    return qc_along_path, children, visits, value
