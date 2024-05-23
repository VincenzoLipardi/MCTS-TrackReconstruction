import os
import pandas as pd
from structure import Circuit
import mcts


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

def run_and_savepkl(evaluation_function, criteria, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, ucb, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=False):
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
        final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, criteria=criteria, rollout_type=rollout_type, roll_out_steps=roll_out_steps, choices=choices, epsilon=epsilon, stop_deterministic=stop_deterministic, ucb_value=ucb, verbose=verbose)
        
        # Save the results
        df = pd.DataFrame(final_state)
        df.to_pickle(file_path)
        print(f"Files saved in experiments/ as '{file_path}'")
    else:
        print(f'File already exists: {file_path}')
