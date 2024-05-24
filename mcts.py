import random
import numpy as np
from structure import Circuit, GateSet
from qiskit import QuantumCircuit
from tqdm import tqdm


class Node:
    def __init__(self, state: Circuit, max_depth: int, parent=None):
        """
        A node of the tree store a lot of information in order to guide the search
        :param state: Circuit object. This is the quantum circuit stored in the node.
        :param parent: Node object. Parent Node
        """
        # Quantum circuit
        self.state = state
        # Is the circuit respecting the constraint of the hardware? boolean
        self.isTerminal = False
        # Parent node. Node object. The root node is the only one not having that.
        self.parent = parent
        # List of children of the node. list
        self.children = []
        # Number of times the node have been visited. integer
        self.visits = 0
        # Value is the total reward. float
        self.value = 0
        # Maximum quantum circuit depth
        self.max_depth = max_depth
        # Position of the node in terms of tree depth. integer
        self.tree_depth = 0 if parent is None else parent.tree_depth + 1
        # Gate set
        self.gate_set = 'continuous'
        # Control on the stop action of the node
        self.stop_is_done = False
        # Specify the action that created it, None only for root and stop nodes
        self.action = None
        # Counts the Controlled-NOT gates in the circuit
        self.counter_cx = self.state.circuit.count_ops().get('cx', 0)

    def __repr__(self):
        return "Tree depth: %s  -  Generated by Action: %s  -  Number of Children: %s  -  Visits: %s  -  Value: %s  -  Quantum Circuit (CNOT counts= %s):\n%s" % \
            (self.tree_depth, self.action, len(self.children), self.visits, self.value, self.counter_cx, self.state.circuit)

    def is_fully_expanded(self, branches, C=1, alpha=0.3):
        """
        :param branches: int or boolean. If true, progressive widening. if int the maximum number of branches is fixed.
        :param alpha: float (in [0,1]). Have to be chosen close to 1 in  strongly stochastic domains, close to 0 ow
        :param C: int. Hyperparameter
        :return: Boolean. True if the node is a leaf. False otherwise.
        """
        if isinstance(branches, bool):
            # Progressive Widening Techniques: adaptive number of branches
            t = self.visits
            k = np.ceil(C * (t ** alpha))
            if not branches:
                # Progressive Widening for discrete action space
                return len(self.children) >= k
            else:
                raise NotImplementedError   # Double Progressive Widening for Stochastic transitions
        elif isinstance(branches, int):     # Vanilla MCTS
            # The number of the tree's branches is fixed
            return len(self.children) >= branches
        else:
            raise TypeError


    def define_children(self, prob_choice, stop_deterministic, roll_out=False):
        """ Expand node by defining a new node applying a modification to the circuit
            :param stop_deterministic: boolean. If true the stop action is placed by default for each new node.
            :param prob_choice: dict. Probability to choose between the possible actions
            :param roll_out: boolean. True if it is used for the rollout (new nodes are temporary, not included in the tree)
            :return: Node """

        parent = self
        qc = parent.state.circuit.copy()
        # If we are rolling out we don't want to add the stop action in the rollout
        stop = self.stop_is_done
        if roll_out:
            stop = True

        def det_stop():

            new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)
            new_qc = new_qc(qc)
            if new_qc is None:
                # It chose to change parameters, but there are no parametrized gates. Or delete in a very shallow circuit,
                # then let's prevent this by allowing only the adding and swapping action
                temporary_prob_choice = {'a': 50, 's': 50, 'c': 0, 'd': 0}
                new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, temporary_prob_choice, stop)
                new_qc = new_qc(qc)
            """while check_equivalence(qc, new_qc):
                new_qc = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)(qc)"""
            new_node = node_from_qc(new_qc, parent_node=self, roll_out=roll_out)
            new_node.action = action
            self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
            return new_node

        def prob_stop():

            new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)
            new_qc = new_qc(qc)
            if new_qc == 'stop':
                # It means that get_legal_actions returned the STOP action, then we define this node as Terminal
                self.isTerminal = True
                self.stop_is_done = True
                self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
                return self
            else:

                if new_qc is None:
                    # It chose to change parameters, but there are no parametrized gates. Or delete in a very shallow circuit,
                    # then let's prevent this by allowing only the adding and swapping action
                    temporary_prob_choice = {'a': 50, 's': 50, 'c': 0, 'd': 0}
                    new_qc, action = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, temporary_prob_choice, stop)


                    new_qc = new_qc(qc)

                if isinstance(new_qc, QuantumCircuit):
                    new_state = Circuit(4, 1).building_state(new_qc)
                    new_node = Node(new_state, max_depth=self.max_depth, parent=self)
                    new_node.action = action
                    self.counter_cx = self.state.circuit.count_ops().get('cx', 0)
                    if not roll_out:
                        self.children.append(new_node)
                    return new_node
                else:
                    raise TypeError

        if stop_deterministic:
            return det_stop()
        else:

            return prob_stop()

    def best_child(self, criteria):
        children_with_values = [(child, child.value, child.visits) for child in self.children]
        if criteria == 'value':
            best = max(children_with_values, key=lambda x: x[1])[0]
        elif criteria == 'average_value':
            best = max(children_with_values, key=lambda x: x[1]/x[2])[0]
        elif criteria == 'visits':
            best = max(children_with_values, key=lambda x: x[2])[0]
        else:
            raise NotImplementedError
        return best


def node_from_qc(quantum_circuit, parent_node, roll_out):
    if isinstance(quantum_circuit, QuantumCircuit):
        new_state = Circuit(4, 1).building_state(quantum_circuit)
        new_child = Node(new_state, max_depth=parent_node.max_depth, parent=parent_node)
        if not roll_out:
            parent_node.children.append(new_child)
        return new_child
    else:
        raise TypeError


def select(node, exploration=.4):
    l = np.log(node.visits)
    children_with_values = [(child, child.value / child.visits + exploration * np.sqrt(l / child.visits)) for child in node.children]
    selected_child = max(children_with_values, key=lambda x: x[1])[0]
    return selected_child


def expand(node, prob_choice, stop_deterministic):
    new_node = node.define_children(prob_choice=prob_choice, stop_deterministic=stop_deterministic)

    if stop_deterministic:
        stop_node = node_from_qc(new_node.state.circuit, new_node, roll_out=False)
        stop_node.isTerminal = True

        return stop_node
    else:
        return new_node


def rollout(node, steps):
    new_node = node
    for i in range(steps):
        new_node = new_node.define_children(prob_choice={'a': 100, 'd': 0, 's': 0, 'c': 0, 'p': 0}, roll_out=True, stop_deterministic=False)
    return new_node


def evaluate(node, evaluation_function):
    return node.state.evaluation(evaluation_function)


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


def modify_prob_choice(dictionary, len_qc):

    keys = list(dictionary.keys())
    values = list(dictionary.values())

    modifications = [-40, 10, 10, 10, 10]
    modified_values = [max(0, v + m) for v, m in zip(values, modifications)]
    if len_qc < 6:
        modified_values[1] = 0
    # Normalize to ensure the sum is still 100
    modified_values = [v / sum(modified_values) * 100 for v in modified_values]
    # Normalize to ensure the sum is still 100
    modified_dict = dict(zip(keys, modified_values))
    return modified_dict


def commit(epsilon, current_node, criteria):
    # It commits to the best action if the node has explored enough

    if epsilon is not None:
        coin = random.uniform(0, 1)
        if coin >= epsilon:
            current_node = current_node.best_child(criteria=criteria)

    else:
        current_node = current_node.best_child(criteria=criteria)

    return current_node


def mcts(root, budget, evaluation_function, criteria, rollout_type, roll_out_steps, branches, choices, epsilon, stop_deterministic, ucb_value=0.4, verbose=False, loading_bar=False):
    prob_choice = {'a': 100, 'd': 0, 's': 0, 'c': 0}
    original_root = root
    if verbose:
        print('Root Node:\n', root)

    epoch_counter = 0
    evaluate(root, evaluation_function)
    root.visits = 1

    if loading_bar == True:
        budget_range = tqdm(range(budget))
    else:
        budget_range = range(budget)

    for _ in budget_range:
        current_node = root

        if current_node.visits > budget/20 and len(current_node.children) > 2:
            root = commit(epsilon, current_node, criteria)
            if verbose:
                print('commit to', root)
            current_node = root


        if verbose:
            print('Epoch Counter: ', epoch_counter)

        # Selection
        while not current_node.isTerminal and current_node.is_fully_expanded(branches=branches):
            current_node = select(current_node, ucb_value)
            if verbose:
                print('selection: ', current_node)

        # Expansion
        if not current_node.isTerminal:
            current_node = expand(current_node, prob_choice=prob_choice, stop_deterministic=stop_deterministic)
            if verbose:
                print('Node Expanded:\n', current_node)


        # Simulation ---> In this version is not used as rollout has to be 0
        if not current_node.isTerminal:
            if isinstance(roll_out_steps, int):
                leaf_node = rollout(current_node, steps=roll_out_steps)
                result = evaluate(leaf_node, evaluation_function)
                if roll_out_steps > 1 and rollout_type == 'max':
                    result_list = [result]
                    node_to_evaluate = leaf_node
                    for _ in range(roll_out_steps):
                        result_list.append(evaluate(node_to_evaluate.parent, evaluation_function))
                    result = max(result_list)
            else:
                raise TypeError

        else:
            result = evaluate(current_node, evaluation_function)

        if verbose:
            print('Reward: ', result)

        # Backpropagation
        backpropagate(current_node, result)

        epoch_counter += 1
        n_qubits = len(current_node.state.circuit.qubits)
        if current_node.tree_depth == 2*n_qubits:
            prob_choice = choices

    # Return the best
    best_node = original_root

    # path = []
    qc_path = []
    children = []
    value, visits = [], []
    while not best_node.isTerminal and len(best_node.children) >= 1:
        # path.append(best_node)
        qc_path.append(best_node.state.circuit)
        children.append(len(best_node.children))
        value.append(best_node.value)
        visits.append(best_node.visits)
        best_node = best_node.best_child(criteria=criteria)
    return {'qc': qc_path, 'children': children, 'visits': visits, 'value': value}
