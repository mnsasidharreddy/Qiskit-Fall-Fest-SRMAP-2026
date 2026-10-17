# QFF 2026 - PS-Q-OPEN: Open Innovation — Custom Quantum Machine Learning

**Project:** Custom Hybrid Quantum Neural Network for Noise-Aware Classification

## Task

Design and implement a custom Quantum Machine Learning (QML) model for binary classification. The project uses a custom Variational Quantum Classifier (VQC) and compares its performance with a classical Logistic Regression baseline. It also evaluates the model using two software-simulated noise profiles, Processor A and Processor B.

## Data / Setup

- Dataset: synthetic two-class `make_moons` dataset
- Total samples: 160
- Features: 2
- Training samples: 120
- Test samples: 40
- Random seed: 42
- Dataset noise parameter: 0.18
- Split: stratified 75/25 train/test split

## Structure

- `README.md` - Problem statement selected, setup, methodology, key results
- `main.py` - Main python file which runs code
- `main.ipynb` - End-to-end reproducible experiment notebook
- `requirements.txt` - Software dependencies (qiskit, qiskit-aer, networkx, etc.)
- `data/` - Folder containing raw and processed data
- `data/raw/` - Raw data or generated inputs
- `data/processed/` - Cleaned arrays or matrices
- `processors/` - Folder Containing Processor A & processor B Maps & Configs
- `processors/processor_A.json` - Participant-defined Processor A coupling map & config
- `processors/processor_B.json` - Participant-defined Processor B coupling map & config
- `src/` - Folder containing problem, processor, routing, metric, and visualization related files
- `src/_init _.py` - initialization code that runs when the package is imported
- `src/problem.py` - Problem formulation & cost Hamiltonian construction
- `src/processors.py` - Topology graph definition & noise model loaders
- `src/routing.py` - Transpilation & layout routing routines
- `src/metrics.py` - Circuit metrics (depth, CX, SWAP, fidelity)
- `src/visualization.py` - Plotting & coupling graph visualizers
- `results/` - Folder containing outputs like tables & figures
- `results/summary.json` - Thhe summary of result
- `results/tables/` - Exported CSV/JSON benchmark comparison tables
- `results/metrics.csv` - Calculated data about accuracy etc
- `results/figures/` - Exported comparison figures ( model_comparison, test_chart)
- `results/model_comparison.png` - Comparison of classical baseline & simulated qml
- `results/test_chart.png` - Comparison of accuracy of classical baseline & processor A & processor B
- `report.md` - Architectural analysis, trade-off interpretation, limitations
- `assets/` - dataset plots for reference
- `assets/figures/raw_dataset_distribution.png` - raw 2D dataset showing both classes for reference
- `assets/figures/training_split_distribution.png` - raw feature distribution for the training split for reference
- `assets/figures/processed_angle_features.png` - dataset after training-fitted standardization, clipping, and angle mapping for reference


## Setup

- Python 3.11 or a compatible version
- The libraries listed in `requirements.txt`

Create a virtual environment:

```bash
python -m venv .venv

.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

## Workflow

1. Generate the synthetic `make_moons` dataset and split it into training and test sets.
2. Standardise and scale the two input features for quantum angle encoding.
3. Build and train the two-qubit Variational Quantum Classifier.
4. Train Logistic Regression on the same data split as the classical baseline.
5. Add the configured simulated score noise for Processor A and Processor B.
6. Calculate accuracy, precision, recall, and F1-score.
7. Save the results and compare the quantum and classical models.

## Actual Measured Results

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Classical Logistic Regression | 90.00% | 90.00% | 90.00% | 90.00% |
| Processor A (simulated lower noise) | 62.50% | 63.16% | 60.00% | 61.54% |
| Processor B (simulated higher noise) | 60.00% | 59.09% | 65.00% | 61.90% |

Processor A uses a simulated score-noise standard deviation of `0.03`; Processor B uses `0.12`. These are software-simulated score perturbations, not results from physical quantum processors.

The recorded COBYLA optimisation reached its maximum number of objective evaluations and did not converge. The recorded final training loss was `0.7255`.

## Processor A/B Comparison

| Setting | Processor A | Processor B |
|---|---:|---:|
| Simulated noise level | Lower | Higher |
| Score-noise standard deviation | 0.03 | 0.12 |
| Random seed offset | 100 | 200 |
| Accuracy | 62.50% | 60.00% |
| Precision | 63.16% | 59.09% |
| Recall | 60.00% | 65.00% |
| F1-score | 61.54% | 61.90% |

The earlier OI README did not provide compiled circuit depth, total gate count, CX count, SWAP count, or physical-qubit counts for this two-qubit experiment. Those metrics are therefore not reported here.

## Conclusion

This project tests a small quantum classifier and compares it with Logistic Regression. In the recorded run, Logistic Regression performed better, with 90% accuracy. Processor A reached 62.5% accuracy and Processor B reached 60%.
The quantum model did not beat the classical model in this test. Next, we can try improving the training process, running the experiment with different random seeds, and testing more noise settings. This will help us understand how the model behaves in different conditions.

## Limitations

- The dataset is synthetic and contains only 160 samples.
- Processor A and Processor B are software-simulated noise profiles, not physical quantum processors.
- The simulated noise is added to model scores and does not reproduce all real quantum hardware noise.
- The optimizer reached its evaluation limit without convergence.
- Results come from one recorded run and may vary across repeated experiments.