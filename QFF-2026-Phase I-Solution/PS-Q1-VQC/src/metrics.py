import pandas as pd

def circuit_metrics(circuit, processor_name, physical_qubits):
    ops = circuit.count_ops()
    return {"processor":processor_name, "logical_qubits":4,
        "physical_qubits":int(physical_qubits), "compiled_qubits":int(circuit.num_qubits),
        "depth":int(circuit.depth() or 0), "total_gates":int(circuit.size() or 0),
        "cx_count":int(ops.get("cx",0)), "swap_count":int(ops.get("swap",0)),
        "operation_counts":{str(k):int(v) for k,v in ops.items()}}

def save_metrics_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
