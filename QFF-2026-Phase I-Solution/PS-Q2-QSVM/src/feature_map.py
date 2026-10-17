import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

N_QUBITS = 4

def build_feature_map():
    """Chosen 4-qubit angle-encoding feature map; verify against full task guide."""
    x = ParameterVector("x", 2)
    qc = QuantumCircuit(N_QUBITS, name="moon_feature_map")
    for q in range(N_QUBITS):
        qc.h(q)
    for q in range(N_QUBITS):
        qc.ry(np.pi * x[q % 2], q)
    for q in range(N_QUBITS):
        qc.rz(np.pi * x[(q + 1) % 2], q)
    for q in range(N_QUBITS):
        qc.cx(q, (q + 1) % N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(np.pi * x[q % 2], q)
    return qc, list(x)
