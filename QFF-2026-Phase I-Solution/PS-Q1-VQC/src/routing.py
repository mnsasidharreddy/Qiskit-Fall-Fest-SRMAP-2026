from qiskit import transpile

def transpile_for_processor(circuit, config):
    return transpile(circuit, basis_gates=config["basis_gates"],
        coupling_map=config["coupling_map"],
        optimization_level=int(config.get("optimization_level",1)),
        seed_transpiler=int(config.get("seed_transpiler",42)))
