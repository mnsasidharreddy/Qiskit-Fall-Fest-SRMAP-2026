import json
from pathlib import Path
import networkx as nx

def load_processor_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    required = {"name","num_qubits","coupling_map","basis_gates"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    n = int(config["num_qubits"])
    for edge in config["coupling_map"]:
        if len(edge) != 2 or any(not isinstance(q,int) or q < 0 or q >= n for q in edge):
            raise ValueError(f"Invalid coupling edge: {edge}")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, config["coupling_map"]))
    config["graph"] = graph
    return config
