import TrackHHL.trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
import TrackHHL.trackhhl.toy.simple_generator as toy
from TrackHHL.utils import *
import pennylane as qml

N_MODULES = 3
N_PARTICLES = 2
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0

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
ham.construct_hamiltonian(event=event)

matrix = ham.A.todense()
pauli_decomp = qml.pauli_decompose(matrix,pauli=True)
pauli_string_result, coeffs_list = convert_to_pauli_string(pauli_decomp)
save_to_json(pauli_string_result, 'pauli_string.json')
save_to_json(coeffs_list, 'coeffs_list.json')