import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from src.problem import make_dataset
from src.feature_map import build_feature_map
from src.quantum_kernel import fidelity_kernel_matrix
from src.classifier import train_and_evaluate
from src.processors import load_processor_config
from src.routing import transpile_for_processor
from src.metrics import circuit_metrics, save_csv, save_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROCESSOR_A = ROOT / "processors" / "processor_a.json"
PROCESSOR_B = ROOT / "processors" / "processor_b.json"

def main():
    X_train, X_test, y_train, y_test, _ = make_dataset(random_state=42)
    print(f"Dataset: train={len(X_train)}, test={len(X_test)}, features={X_train.shape[1]}")
    feature_map, params = build_feature_map()

    start = time.perf_counter()
    K_train = fidelity_kernel_matrix(X_train, X_train, feature_map, params)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    K_test = fidelity_kernel_matrix(X_test, X_train, feature_map, params)
    test_seconds = time.perf_counter() - start

    assert K_train.shape == (len(X_train), len(X_train))
    assert np.allclose(K_train, K_train.T, atol=1e-8)
    assert np.allclose(np.diag(K_train), 1.0, atol=1e-8)
    assert K_test.shape == (len(X_test), len(X_train))

    _, _, classification = train_and_evaluate(K_train, K_test, y_train, y_test)
    print("\\nQSVM test metrics (ideal Statevector kernel):")
    for key, value in classification.items():
        print(f"  {key}: {value}")
    print(f"  train kernel simulator seconds: {train_seconds:.6f}")
    print(f"  test kernel simulator seconds: {test_seconds:.6f}")
    
    from qiskit.circuit import ParameterVector

    U, xp = build_feature_map()
    Uinv, xpp = build_feature_map()

    xp_prime = ParameterVector("xp", 2)
    Uinv = Uinv.assign_parameters({
        old: new for old, new in zip(xpp, xp_prime)
    })

    pair = U.compose(Uinv.inverse(), qubits=list(range(4)))

    bind = {p: 0.25 for p in xp}
    bind.update({p: 0.75 for p in xp_prime})
    pair = pair.assign_parameters(bind)


    cfgs = [
        load_processor_config(PROCESSOR_A),
        load_processor_config(PROCESSOR_B),
    ]
    rows = []
    unique_train_pairs = len(X_train) * (len(X_train) + 1) // 2
    for cfg in cfgs:
        compiled = transpile_for_processor(pair, cfg)
        row = circuit_metrics(compiled, cfg["name"], unique_train_pairs)
        row["processor_qubits_available"] = int(cfg["num_qubits"])
        rows.append(row)
        print(f"\\n{cfg['name']} resources:")
        for k, v in row.items():
            print(f"  {k}: {v}")

    save_csv(RESULTS / "tables" / "classification_metrics.csv",
             [{"metric": k, "value": json.dumps(v)} for k, v in classification.items()])
    save_csv(RESULTS / "tables" / "processor_comparison.csv", rows)
    save_json(RESULTS / "tables" / "summary.json", {
        "project": "QFF 2026 PS-Q2 QSVM",
        "dataset": {"name": "make_moons", "samples": 100, "features": 2, "noise": 0.1,
                    "random_state": 42, "train_samples": len(X_train), "test_samples": len(X_test)},
        "feature_map": "Chosen 4-qubit angle encoding with H, RY/RZ and ring CX entanglement",
        "kernel": "Ideal Statevector fidelity kernel",
        "classification": classification,
        "simulator_kernel_times_seconds": {"train": train_seconds, "test": test_seconds},
        "processor_comparison": rows
    })

    X_all, y_all = make_moons(n_samples=100, noise=0.1, random_state=42)
    fig, ax = plt.subplots()
    ax.scatter(X_all[:, 0], X_all[:, 1], c=y_all)
    ax.set_title("PS-Q2 make_moons dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "dataset.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(K_train, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Training fidelity kernel matrix")
    ax.set_xlabel("Training sample")
    ax.set_ylabel("Training sample")
    fig.colorbar(im, ax=ax, label="Fidelity")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "kernel_matrix.png", dpi=160)
    plt.close(fig)
    print("\\nFinished. Results saved under results/.")

if __name__ == "__main__":
    main()
