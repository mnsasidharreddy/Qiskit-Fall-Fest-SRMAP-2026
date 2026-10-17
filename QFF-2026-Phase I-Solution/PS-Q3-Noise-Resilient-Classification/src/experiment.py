import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from .model import make_circuit, predict_score_ideal, score_to_label
from .processors import load_processor, build_noise_model, build_coupling_map

def bit_zero_probability(counts):
    total = sum(counts.values())
    zeros = 0
    for bitstring, count in counts.items():
        if bitstring.replace(" ", "")[-1] == "0":
            zeros += count
    return zeros / total

def calibrate_readout(cfg, noise_model, shots):
    """Calibrate the measured physical qubit 0 with prepared |0> and |1> states."""
    n = cfg["physical_qubits"]
    backend = AerSimulator(noise_model=noise_model)
    circuits = []
    for prepared in (0, 1):
        qc = QuantumCircuit(n, 1)
        if prepared == 1:
            qc.x(0)
        qc.measure(0, 0)
        circuits.append(qc)
    result = backend.run(circuits, shots=shots, seed_simulator=42).result()
    c0, c1 = result.get_counts(0), result.get_counts(1)
    p_obs0_given0 = c0.get("0", 0) / shots
    p_obs1_given0 = c0.get("1", 0) / shots
    p_obs0_given1 = c1.get("0", 0) / shots
    p_obs1_given1 = c1.get("1", 0) / shots
    return np.array([[p_obs0_given0, p_obs0_given1],
                     [p_obs1_given0, p_obs1_given1]], dtype=float)

def mitigate_p0(p0_observed, assignment):
    """Invert the 2x2 readout assignment matrix and clip to valid probabilities."""
    observed = np.array([p0_observed, 1.0-p0_observed])
    try:
        corrected = np.linalg.solve(assignment, observed)
    except np.linalg.LinAlgError:
        return float(p0_observed)
    corrected = np.clip(corrected, 0.0, None)
    if corrected.sum() == 0:
        return float(p0_observed)
    corrected /= corrected.sum()
    return float(corrected[0])

def evaluate(X, y, theta, processor_name):
    cfg = load_processor(processor_name)
    shots = cfg["transpile"]["shots"]
    coupling = build_coupling_map(cfg)
    noise = build_noise_model(cfg)
    backend_noisy = AerSimulator(noise_model=noise)
    backend_ideal = AerSimulator()
    assignment = calibrate_readout(cfg, noise, shots)
    rows, depths, cx_counts, swap_counts = [], [], [], []

    for i, x in enumerate(X):
        logical = make_circuit(x, theta, measure=True)
        common = dict(coupling_map=coupling, basis_gates=cfg["basis_gates"],
                      optimization_level=cfg["transpile"]["optimization_level"],
                      seed_transpiler=cfg["transpile"]["seed_transpiler"],
                      initial_layout=list(range(4)))
        compiled = transpile(logical, **common)
        depths.append(compiled.depth())
        ops = compiled.count_ops()
        compiled_cx = int(ops.get("cx", 0))
        cx_counts.append(compiled_cx)
        logical_cx = int(logical.count_ops().get("cx", 0))
        extra_cx = max(0, compiled_cx - logical_cx)
        swap_counts.append(extra_cx / 3.0)

        ideal_circuit = transpile(logical, backend_ideal,
                                  optimization_level=1, seed_transpiler=42)
        ideal_counts = backend_ideal.run(ideal_circuit, shots=shots, seed_simulator=1000+i).result().get_counts()
        noisy_counts = backend_noisy.run(compiled, shots=shots, seed_simulator=2000+i).result().get_counts()
        p0_ideal = bit_zero_probability(ideal_counts)
        p0_noisy = bit_zero_probability(noisy_counts)
        p0_mitigated = mitigate_p0(p0_noisy, assignment)

        rows.append({
            "sample": i, "true_label": int(y[i]),
            "ideal_pred": score_to_label(2*p0_ideal-1),
            "noisy_pred": score_to_label(2*p0_noisy-1),
            "mitigated_pred": score_to_label(2*p0_mitigated-1),
            "ideal_score": 2*p0_ideal-1,
            "noisy_score": 2*p0_noisy-1,
            "mitigated_score": 2*p0_mitigated-1
        })

    df = pd.DataFrame(rows)
    summary = {
        "processor": cfg["name"],
        "ideal_accuracy": float(np.mean(df.ideal_pred == df.true_label)),
        "noisy_accuracy": float(np.mean(df.noisy_pred == df.true_label)),
        "mitigated_accuracy": float(np.mean(df.mitigated_pred == df.true_label)),
        "mean_transpiled_depth": float(np.mean(depths)),
        "mean_cx_count": float(np.mean(cx_counts)),
        "estimated_mean_swap_overhead": float(np.mean(swap_counts)),
        "readout_assignment_matrix": assignment.tolist()
    }
    return df, summary
