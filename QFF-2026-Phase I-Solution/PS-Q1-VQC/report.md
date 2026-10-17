# QFF 2026 PS-Q1 — Phase 1 Report

## 1. Specification

This implementation evaluates a four-qubit Variational Quantum Classifier (VQC) using the 200-sample synthetic binary-classification dataset. The dataset contains four features, with three informative features, and uses an 80/20 train-test split with random state 42.

## 2. Dataset and preprocessing

The dataset was generated using `make_classification` with 200 samples, four features, three informative features, and random state 42. The data was split into 160 training samples and 40 test samples.

The training set contains 80 samples from each class, while the test set contains 20 samples from each class. `MinMaxScaler` was fitted on the training data and used to transform the features.

## 3. Model

The VQC uses four logical qubits to encode the four input features. The circuit uses angle encoding, trainable rotation gates, and entangling CX gates. The model was trained using ideal statevector simulation with a maximum of 35 optimizer iterations and random seed 42.

The optimizer reported successful convergence. The final training loss was approximately 0.5402.

## 4. Processor A/B configurations

Processor A has five physical qubits and four coupling edges: (0,1), (1,2), (1,3), and (3,4).

Processor B has nine physical qubits and twelve coupling edges: (0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), and (8,0).

Both circuits were compiled using the project’s configured transpiler settings. The same trained logical circuit and held-out sample were used for the resource comparison.

## 5. Results

### VQC test metrics

Accuracy = 77.50%
Precision = 86.67%
Recall = 65.00%
F1 score = 74.29%
Final training loss = 0.5402
Optimizer success = True

### Logistic regression baseline

Accuracy = 87.50%
Precision = 89.47%
Recall = 85.00%
F1 score = 87.18%

### Compiled circuit comparison

| Metric | Processor A | Processor B |
| Physical qubits available | 5 | 9 |
| Logical qubits required | 4 | 4 |
| Compiled qubits | 5 | 9 |
| Circuit depth | 13 | 13 |
| Total gates | 38 | 38 |
| CX gates | 6 | 6 |
| SWAP gates | 0 | 0 |

The notebook saved the comparison table to `results/tables/processor_comparison.csv` and the summary to `results/tables/summary.json`. The training-loss and circuit-depth plots are saved under `results/figures/`.

## 6. Analysis

### Which processor has lower compiled circuit depth?

Neither processor has lower depth in this experiment. Both compiled circuits have a depth of 13.

### How do CX counts and SWAP counts differ?

Both processors have six CX gates and zero SWAP gates. CX gates are two-qubit controlled-NOT operations. SWAP gates exchange quantum states between qubits and may be inserted when routing is needed. No SWAP gates were present in either compiled circuit in this run.

### What test metrics did the model achieve?

The VQC achieved 77.50% accuracy, 86.67% precision, 65.00% recall, and an F1 score of 74.29%. The logistic regression baseline achieved 87.50% accuracy and an F1 score of 87.18%, outperforming the VQC on this test set.

### What are the limitations of ideal statevector training?

Ideal statevector simulation does not automatically model realistic hardware noise, such as gate errors, decoherence, or readout errors. Therefore, these classification metrics should not be described as noisy quantum hardware results.

### Does the authoritative PS-Q1 specification require noisy sampled A/B evaluation?

This has not yet been confirmed. The current notebook compares compiled circuit resources and does not report noisy A/B classification accuracy. The official challenge specification or organizer clarification must be checked before submission.

## 7. Limitations

The test set contains only 40 samples, so the measured classification metrics may vary with different dataset splits. The VQC was evaluated using ideal statevector simulation. Processor A/B benchmarking measured compiled circuit resources for one held-out sample, rather than noisy hardware classification accuracy.

## 8. Reproducibility

The notebook uses random state 42 for dataset generation and seed 42 for the VQC training and configured transpiler settings. The recorded dataset split contains 160 training samples and 40 test samples.

The experiment was run in a Python virtual environment with the project dependencies installed. Refer to `requirements.txt` for the dependency list.