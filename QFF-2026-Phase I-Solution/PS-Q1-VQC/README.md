# QFF 2026 - PS-Q1: Biomedical Signal Classifier

**Project:** Variational Quantum Classification for Arrhythmia & Biosignals

## Task
Build and train a 4-qubit Variational Quantum Classifier with angle encoding and parameterized rotation layers.
Compare classification accuracy, training convergence, and compiled circuit depth on Processor A vs B.

## Data / Setup
Generate 200 synthetic data samples (4 features, binary class) using
sklearn.datasets.make_classification(n_samples=200, n_features=4, n_informative=3, random_state=42) with an
80/20 train/test split.

## Structure
- `README.md` - Problem statement selected, setup, methodology, key results
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
- `results/tables/` - Exported CSV/JSON benchmark comparison tables
- `results/figures/` - Exported comparison figures (depth, energy curves, SWAP overhead)
- `report.md` - Architectural analysis, trade-off interpretation, limitations
- `assets/qff_phase1_latex_images/` - Optional LaTeX equation renders or visual assets

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open `main.ipynb` in Jupyter and run cells from top to bottom.

## Workflow

1) Generation of the synthetic dataset and split it into training and testing sets.

2) Building and training of the four-qubit Variational Quantum Classifier.

3) Comparing the coding results on Processor A and Processor B.

4) Evaluation of classification accuracy, circuit depth, CX count, SWAP routing overhead. Noise impact is not evaluated in the current experiment.

5) The results, graphs, and final analysis in the results/ directory and report.md are saved.

## Actual Measured Results

| Metric | VQC | Logistic regression |
|---|---:|---:|
| Accuracy | 77.50% | 87.50% |
| Precision | 86.67% | 89.47% |
| Recall | 65.00% | 85.00% |
| F1 score | 74.29% | 87.18% |

## Processor A/B comparison

| Metric | Processor A | Processor B |
|---|---:|---:|
| Circuit depth | 13 | 13 |
| Total gates | 38 | 38 |
| CX gates | 6 | 6 |
| SWAP gates | 0 | 0 |
| Physical qubits available | 5 | 9 |

## Conclusion
logistic regression outperformed the VQC on this test set and that both processors produced identical compiled circuit metrics for the tested circuit.