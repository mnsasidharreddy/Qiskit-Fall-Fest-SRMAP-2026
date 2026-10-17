import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N_QUBITS = 4
N_LAYERS = 2
N_PARAMS = N_QUBITS * N_LAYERS * 2

def make_circuit(x, theta, measure=False):
    """4-qubit angle-encoded VQC. theta has two trainable rotations per qubit/layer."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS if measure else 0)
    for q in range(N_QUBITS):
        qc.ry(float(x[q]), q)
    k = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(float(theta[k]), q); k += 1
            qc.rz(float(theta[k]), q); k += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
    if measure:
        qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc

def predict_score_ideal(x, theta):
    state = Statevector.from_instruction(make_circuit(x, theta))
    probs = state.probabilities()
    p0 = sum(probs[i] for i in range(len(probs)) if (i & 1) == 0)
    return float(2 * p0 - 1)

def train_vqc(X, y, seed=42):
    """Train with a small classical optimizer against exact statevector scores."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-0.2, 0.2, N_PARAMS)
    labels = np.asarray(y) * 2 - 1

    idx = np.arange(len(X))
    if len(idx) > 100:
        idx = rng.choice(idx, size=100, replace=False)

    def loss(theta):
        scores = np.array([predict_score_ideal(X[i], theta) for i in idx])
        return float(np.mean(np.logaddexp(0, -labels[idx] * 2.0 * scores)) + 0.002*np.sum(theta**2))

    result = minimize(loss, theta0, method="COBYLA", options={"maxiter": 100, "rhobeg": 0.3})
    return result.x, {"training_loss": float(result.fun), "optimizer_success": bool(result.success),
                      "optimizer_message": str(result.message), "iterations": int(result.nfev)}

def score_to_label(score):
    return int(score > 0.0)
