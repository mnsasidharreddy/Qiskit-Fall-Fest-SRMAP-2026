
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

N_QUBITS = 2
N_LAYERS = 2
RANDOM_SEED = 7

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def quantum_circuit(x, weights):
    """Encode input features and apply trainable quantum layers."""

    qml.AngleEmbedding(
        features=x,
        wires=range(N_QUBITS),
        rotation="X"
    )

    for layer in range(N_LAYERS):
        for wire in range(N_QUBITS):
            qml.RY(weights[layer, wire, 0], wires=wire)
            qml.RZ(weights[layer, wire, 1], wires=wire)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 0])

    return qml.expval(qml.PauliZ(0))


def predict_scores(X, weights):
    """Return quantum circuit scores between -1 and 1."""

    return np.array([
        float(quantum_circuit(x, weights))
        for x in X
    ])


def train_quantum_model(X_train, y_train, maxiter=150):
    """Train the variational quantum classifier using COBYLA."""

    rng = np.random.default_rng(RANDOM_SEED)

    initial_weights = rng.normal(
        loc=0.0,
        scale=0.15,
        size=(N_LAYERS, N_QUBITS, 2)
    )

    targets = 2 * np.asarray(y_train, dtype=float) - 1

    def loss_function(flat_weights):
        weights = flat_weights.reshape(
            N_LAYERS, N_QUBITS, 2
        )

        scores = predict_scores(X_train, weights)

        return float(np.mean((scores - targets) ** 2))

    result = minimize(
        loss_function,
        initial_weights.ravel(),
        method="COBYLA",
        options={
            "maxiter": maxiter,
            "rhobeg": 0.1,
            "tol": 1e-4,
            "catol": 1e-4
        }
    )

    trained_weights = result.x.reshape(
        N_LAYERS, N_QUBITS, 2
    )

    print("Optimizer message:", result.message)
    print("Optimizer converged:", result.success)
    print("Iterations performed:", result.nfev)
    print("Final training loss:", round(float(result.fun), 4))

    return trained_weights
