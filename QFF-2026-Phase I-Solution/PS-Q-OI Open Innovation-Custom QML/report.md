
# Project Report: Custom Hybrid Quantum Neural Network for Noise-Aware Classification

## 1. Abstract

This project implements a custom Variational Quantum Classifier (VQC) for binary classification and compares it with a classical Logistic Regression baseline. Two software-simulated processor profiles with different readout-score noise levels are used to examine changes in classification performance.

The experiment uses a synthetic two-class dataset, PennyLane's quantum simulator, and standard classification metrics. In the recorded experiment, the classical baseline achieved 90% test accuracy, while the simulated Processor A and Processor B profiles achieved 62.5% and 60%, respectively. The quantum optimiser did not converge within the configured evaluation limit, so further training and repeated experiments are required.

## 2. Problem Statement

Design a custom Quantum Machine Learning model and document its feature encoding, classical comparison baseline, and behaviour under two different processor noise profiles.

The purpose is to create a reproducible software experiment that compares classification performance under a shared dataset split.

## 3. Objectives

- Implement a custom variational quantum classifier.
- Encode classical input features into quantum rotation angles.
- Train a classical Logistic Regression baseline.
- Simulate two different levels of noise in quantum output scores.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Save results as CSV and JSON files and generate an accuracy comparison chart.

## 4. Dataset and Preprocessing

A synthetic binary classification dataset is generated using scikit-learn's `make_moons` function.

- Total samples: 160
- Number of input features: 2
- Training samples: 120
- Testing samples: 40
- Dataset noise parameter: 0.18
- Random seed: 42

The dataset is split using stratified train-test splitting. StandardScaler is fitted on the training data and applied to both training and testing data.

The scaled features are clipped to [-2.5, 2.5] and mapped to rotation angles using:

`angle = clipped_feature × (π / 2.5)`

This preprocessing provides a consistent input representation for the quantum circuit.

## 5. Proposed Quantum Model

The model is a Variational Quantum Classifier implemented using PennyLane.

### 5.1 Circuit configuration

- Number of qubits: 2
- Number of trainable layers: 2
- Feature embedding: AngleEmbedding with X rotations
- Trainable gates: RY and RZ
- Entanglement: CNOT gates
- Measurement: Expectation value of Pauli-Z on qubit 0

The circuit output is a score between -1 and 1. Scores greater than or equal to zero are classified as class 1; negative scores are classified as class 0.

### 5.2 Training procedure

The circuit parameters are optimised using COBYLA from SciPy. The objective function is the mean squared error between circuit scores and targets encoded as -1 and +1.

In the recorded run, the optimiser stopped after reaching its maximum number of objective-function evaluations. It reported non-convergence and a final training loss of 0.7255. This status is included as a limitation rather than being presented as successful convergence.

## 6. Classical Baseline

Logistic Regression is used as the classical comparison model. It is trained on the prepared training data and evaluated on the same test set as the quantum model.

Using a common data split makes the comparison easier to interpret, although a single small test set does not establish general performance.

## 7. Simulated Processor Profiles

Two software-defined profiles are loaded from JSON configuration files.

| Parameter | Processor A | Processor B |
|---|---:|---:|
| Noise profile | Lower | Higher |
| Readout-score noise standard deviation | 0.03 | 0.12 |
| Seed offset | 100 | 200 |

Gaussian random perturbations are added to the quantum model's test scores. The perturbed scores are clipped to [-1, 1] before classification.

These profiles simulate different noise levels in software. They are not physical quantum processors and do not represent complete models of real quantum hardware noise.

## 8. Evaluation Metrics

The experiment reports:

- **Accuracy:** Fraction of predictions that are correct.
- **Precision:** Fraction of positive predictions that are correct.
- **Recall:** Fraction of actual positive cases identified correctly.
- **F1-score:** Harmonic mean of precision and recall.

## 9. Results

The following results were obtained in the recorded run using 40 test samples.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Classical Baseline | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| Processor A (simulated low-noise) | 0.6250 | 0.6316 | 0.6000 | 0.6154 |
| Processor B (simulated higher-noise) | 0.6000 | 0.5909 | 0.6500 | 0.6190 |

### 9.1 Interpretation

The classical baseline achieved the highest test accuracy at 90%. The simulated Processor A profile achieved 62.5%, and Processor B achieved 60%.

Processor A's accuracy was 2.5 percentage points higher than Processor B's in this run. However, this small difference from a single run is not sufficient to establish a general relationship between noise level and performance.

The quantum model's optimiser did not converge within the configured evaluation limit. Its current performance should therefore be treated as an initial experimental result.

## 10. Generated Outputs

The main experiment saves:

- `results/tables/metrics.csv`
- `results/summary.json`
- `results/figures/model_comparison.png`

These files contain the recorded evaluation metrics, structured results, and visual comparison.

## 11. Limitations

1. The dataset is small and synthetic.
2. The quantum optimiser did not report convergence in the recorded run.
3. The current quantum model underperformed the classical baseline.
4. Noise is applied as Gaussian perturbation to output scores rather than through a full quantum hardware noise model.
5. Only one fixed train-test split and one recorded experiment are reported.
6. No physical quantum processor was used.

## 12. Future Work

Future improvements may include:

- Testing multiple random seeds and reporting mean and standard deviation.
- Exploring different circuit architectures and optimisation settings.
- Evaluating additional synthetic and real-world datasets.
- Modelling quantum noise using explicit noisy quantum simulation channels.
- Comparing additional classical baselines.
- Investigating whether the quantum model can achieve stable optimisation and improved generalisation.

## 13. Conclusion

This project demonstrates an end-to-end software pipeline for custom quantum classification, classical baseline evaluation, and simulated noise-profile comparison. The experiment provides a reproducible starting point and documents both the measured results and the limitations.

The recorded results favour Logistic Regression over the current quantum model. Additional optimisation, more extensive experiments, and more realistic noise modelling are required before drawing broader conclusions about the performance of the quantum approach.
