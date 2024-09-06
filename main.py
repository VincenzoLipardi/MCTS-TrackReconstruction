from evaluation_functions import track_vqe, track_vqe_multi
import save_in_file as sif

OBJ_FUNC = track_vqe
CRITERIA = "value"
VAR_q = 10
Anc_q = 0
BUDGET = 500000
MAX_DEPTH = 100
ITERATIONS = 2
BRANCHES = False
P_ACTIONS = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
UCB = 0.4
GATE_SET = "continuous"  # Ensure it matches the expected case in the function
ROTYPE = "classic"
ROSTEPS = 0
GRADIENT = True


def run_MCTQS(extra):
    init = extra*ITERATIONS
    for i in range(init, init+ITERATIONS):
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
            verbose=False,
            loading_bar=False)
    for i in range(init, init + ITERATIONS):
        sif.add_columns(
            evaluation_function=OBJ_FUNC,
            criteria=CRITERIA,
            qubits=VAR_q,
            budget=BUDGET,
            iter=i,
            max_depth=MAX_DEPTH,
            branches=BRANCHES,
            epsilon=EPS,
            stop_deterministic=STOP,
            roll_out_steps=ROSTEPS,
            rollout_type=ROTYPE,
            gradient=GRADIENT,
            ucb=UCB,
            gate_set=GATE_SET)
    for i in range(init, init + ITERATIONS):
        sif.postprocessing(
            evaluation_function=OBJ_FUNC,
            criteria=CRITERIA,
            budget=BUDGET,
            iter=i,
            max_depth=MAX_DEPTH,
            branches=BRANCHES,
            epsilon=EPS,
            stop_deterministic=STOP,
            roll_out_steps=ROSTEPS,
            rollout_type=ROTYPE,
            ucb=UCB,
            gate_set=GATE_SET,
            qubits=VAR_q)


def postprocess(extra):
    init = extra * ITERATIONS
    for i in range(init, init + ITERATIONS):
        sif.add_columns(
            evaluation_function=OBJ_FUNC,
            criteria=CRITERIA,
            qubits=VAR_q,
            budget=BUDGET,
            iter=i,
            max_depth=MAX_DEPTH,
            branches=BRANCHES,
            epsilon=EPS,
            stop_deterministic=STOP,
            roll_out_steps=ROSTEPS,
            rollout_type=ROTYPE,
            gradient=GRADIENT,
            ucb=UCB,
            gate_set=GATE_SET)
    for i in range(init, init + ITERATIONS):
        sif.postprocessing(
            evaluation_function=OBJ_FUNC,
            criteria=CRITERIA,
            budget=BUDGET,
            iter=i,
            max_depth=MAX_DEPTH,
            branches=BRANCHES,
            epsilon=EPS,
            stop_deterministic=STOP,
            roll_out_steps=ROSTEPS,
            rollout_type=ROTYPE,
            ucb=UCB,
            gate_set=GATE_SET,
            qubits=VAR_q)






