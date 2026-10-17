# QFF 2026 PS-Q2 — Phase 1 Report

## 1. Task
Construct a fidelity quantum kernel on a four-qubit register, train a classical SVM using the precomputed kernel, and compare kernel evaluation cost on Processor A and Processor B.

## 2. Dataset and split
- Dataset: `sklearn.datasets.make_moons`
- Samples: 100; noise: 0.1; random state: 42
- Split: stratified 80% training / 20% testing
- Features: 2

## 3. Feature map
The implementation uses a chosen four-qubit angle-encoding circuit with Hadamard gates, data-dependent RY/RZ rotations, and a ring of CX entanglers. The task excerpt does not prescribe a unique feature map, so verify this choice against any additional official instructions.

## 4. Kernel and classifier
The ideal fidelity kernel is `K(x, x') = |<phi(x)|phi(x')>|^2`. Kernel entries use ideal Statevector simulation. The classical classifier is `SVC(kernel="precomputed")`.

## 5. Processor A/B comparison
The code compiles a representative overlap circuit for each processor and records depth, gate count, CX count, SWAP count, and a gate-count-based resource proxy. This proxy is not measured wall-clock runtime.

## 6. Results
Run `python main.py`; results are saved under `results/tables/` and `results/figures/`. Insert measured outputs and interpretation after successful execution.

## 7. Limitations
- Classification metrics use ideal statevector kernel values.
- Processor comparisons are transpiled resource comparisons, not real-device execution.
- No noise model is applied.
- Verify processor configurations and feature-map requirements against the official challenge materials.
- The kernel cost proxy is an estimate; actual device time depends on execution and queue overhead.

## 8. Reproducibility
Dataset and transpiler seeds are set to 42. Pin dependency versions after verifying the successful local environment.
