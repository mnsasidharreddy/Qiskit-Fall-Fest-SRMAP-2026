import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

N_QUBITS, N_LAYERS = 4, 2

def build_vqc():
    x = ParameterVector("x", N_QUBITS)
    theta = ParameterVector("theta", N_LAYERS * N_QUBITS * 2)
    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(np.pi * x[q], q)
    k = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
    return qc, list(x), list(theta)

def _p1(circuit, xp, tp, features, weights):
    values = {p: float(v) for p, v in zip(xp, features)}
    values.update({p: float(v) for p, v in zip(tp, weights)})
    state = Statevector.from_instruction(circuit.assign_parameters(values))
    return float(np.clip(state.probabilities(qargs=[0])[1], 1e-7, 1-1e-7))

def train_vqc(X_train, y_train, maxiter=35, seed=42):
    qc, xp, tp = build_vqc()
    initial = np.random.default_rng(seed).uniform(-0.15, 0.15, len(tp))
    y = np.asarray(y_train, dtype=float)
    history = []
    def loss(w):
        p = np.array([_p1(qc, xp, tp, x, w) for x in X_train])
        return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
    def callback(w):
        history.append(loss(w))
    result = minimize(loss, initial, method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-7}, callback=callback)
    if not history or abs(history[-1] - result.fun) > 1e-12:
        history.append(float(result.fun))
    return {"circuit":qc, "x_params":xp, "theta_params":tp,
        "weights":np.asarray(result.x), "loss_history":history,
        "optimizer_success":bool(result.success), "optimizer_message":str(result.message),
        "final_loss":float(result.fun)}

def predict_proba(model, X):
    return np.array([_p1(model["circuit"], model["x_params"], model["theta_params"], x, model["weights"]) for x in X])

def predict(model, X, threshold=0.5):
    return (predict_proba(model, X) >= threshold).astype(int)

def bind_circuit_for_sample(model, features):
    values = {p: float(v) for p, v in zip(model["x_params"], features)}
    values.update({p: float(v) for p, v in zip(model["theta_params"], model["weights"])})
    return model["circuit"].assign_parameters(values)
