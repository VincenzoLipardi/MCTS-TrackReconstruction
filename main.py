from evaluation_functions import track_recon
import save_in_file as sif
# Define the necessary variables
OBJ_FUNC = track_recon
CRITERIA = "average_value"
VAR_q = 3
Anc_q = 0
BUDGET = 100
MAX_DEPTH = 10
ITERATIONS = 3
BRANCHES = False
P_ACTIONS = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
UCB = 0.5
GATE_SET = "continuous"  # Ensure it matches the expected case in the function
ROTYPE = "classic"
ROSTEPS = 0
GRADIENT=True

# Call the run_and_savepkl function with the defined variables
for i in range(ITERATIONS):
    sif.run_and_savepkl(
        evaluation_function=OBJ_FUNC,
        criteria=CRITERIA,
        variable_qubits=VAR_q,
        ancilla_qubits=Anc_q,
        budget=BUDGET,
        max_depth=MAX_DEPTH,
        iteration=i,
        branches=BRANCHES,
        choices=P_ACTIONS,
        epsilon=EPS,
        stop_deterministic=STOP,
        ucb=UCB,
        gate_set=GATE_SET,
        rollout_type=ROTYPE,
        roll_out_steps=ROSTEPS,
        verbose=False)
    
sif.add_columns(
    evaluation_function=OBJ_FUNC,
    criteria=CRITERIA,
    budget=BUDGET,
    n_iter=ITERATIONS,
    branches=BRANCHES,
    epsilon=EPS,
    stop_deterministic=STOP,
    roll_out_steps=ROSTEPS,
    rollout_type=ROTYPE,
    gradient=GRADIENT,
    ucb=UCB,
    gate_set=GATE_SET
)
