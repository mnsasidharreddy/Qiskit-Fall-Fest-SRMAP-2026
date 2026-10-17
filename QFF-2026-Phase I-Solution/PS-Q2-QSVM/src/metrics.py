import csv
import json
from pathlib import Path

def circuit_metrics(circuit, processor_name, kernel_pair_count):
    counts = circuit.count_ops()
    return {
        "processor": processor_name,
        "compiled_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "total_gates": int(sum(counts.values())),
        "cx_count": int(counts.get("cx", 0)),
        "swap_count": int(counts.get("swap", 0)),
        "operation_counts": json.dumps(dict(counts), sort_keys=True),
        "kernel_pair_count": int(kernel_pair_count),
        "estimated_compiled_gate_executions": int(sum(counts.values()) * kernel_pair_count),
    }

def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
