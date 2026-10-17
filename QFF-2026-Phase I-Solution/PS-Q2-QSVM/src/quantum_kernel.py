import numpy as np
from qiskit.quantum_info import Statevector

def _statevector(feature_map, params, sample):
    bindings = {p: float(v) for p, v in zip(params, sample)}
    return Statevector.from_instruction(feature_map.assign_parameters(bindings))

def fidelity_kernel_matrix(X1, X2, feature_map, params):
    """K[i,j] = |<phi(X1[i])|phi(X2[j])>|^2 using ideal statevectors."""
    states1 = [_statevector(feature_map, params, x) for x in X1]
    states2 = states1 if X1 is X2 else [_statevector(feature_map, params, x) for x in X2]
    matrix = np.empty((len(X1), len(X2)), dtype=float)
    for i, state1 in enumerate(states1):
        for j, state2 in enumerate(states2):
            matrix[i, j] = float(np.abs(state1.data.conj() @ state2.data) ** 2)
    return np.clip(matrix, 0.0, 1.0)
