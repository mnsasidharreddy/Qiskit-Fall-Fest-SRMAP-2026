import json
from pathlib import Path
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

ROOT = Path(__file__).resolve().parents[1]

def load_processor(name):
    path = ROOT / "processors" / f"processor_{name}.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg

def build_noise_model(cfg):
    rates = cfg["noise"]
    model = NoiseModel(basis_gates=cfg["basis_gates"])
    one = depolarizing_error(rates["one_qubit_depolarizing_p"], 1)
    two = depolarizing_error(rates["cx_depolarizing_p"], 2)
    model.add_all_qubit_quantum_error(one, ["rz", "sx", "x"])
    model.add_all_qubit_quantum_error(two, ["cx"])
    p = rates["readout_error_p"]
    model.add_all_qubit_readout_error(ReadoutError([[1-p, p], [p, 1-p]]))
    return model

def build_coupling_map(cfg):
    return CouplingMap(cfg["coupling_map"])
