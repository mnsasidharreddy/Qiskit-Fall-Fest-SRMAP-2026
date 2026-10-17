# QFF 2026 - PS-Q3: Noise-Resilient Classification & Error Mitigation

**Project:** Zero Noise Extrapolation (ZNE) & Readout Mitigation in QML

## Task
Evaluate a trained VQC under ideal vs synthetic device noise. Apply at least one error mitigation strategy (e.g., Zero-
Noise Extrapolation via Mitiq or readout measurement mitigation) on Processor A vs B.

## Data / Setup
Use the same synthetic classification dataset and official synthetic noise parameters.

## Structure

- `README.md` - Project details, setup steps, and main results
- `main.py` - Runs the complete project
- `requirements.txt` - List of Python libraries needed
- `data/` - Stores the project data
- `data/raw/` - Stores the original dataset
- `data/raw/classification_raw.csv` - 
- `data/processed/` - Stores the data after processing
- `data/processed/classification_processed.csv` - 
- `processors/` - Stores Processor A and Processor B codes
- `processors/processor_a.json` - code for Processor A
- `processors/processor_b.json` - code for Processor B
- `src/` - Contains the main Python code files
- `src/__init__.py` - Initializes the Python package
- `src/data.py` - For Generating 200 sample data
- `src/experiment.py` - Calculates and evaluates probability
- `src/model.py` - Loads linear entanglement
- `src/processors.py` - Loads processor settings
- `results/` - Stores the project results
- `results/tables/` - Stores result tables
- `results/tables/predictions_A.csv` - Prediction values of 40 samples
- `results/tables/predictions_B.csv` - Prediction values of 40 samples
- `results/tables/training_info.json` - Loss & iterations related information
- `results/tables/processor_comparison.csv` - Comparision of processor A & B on accuracy, depth, cx count etc
- `results/figures/` - Stores graphs created by the project
- `results/figures/accuracy_comparison.png` - Accurary comparison of ideal, noisy and mitigated
- `assets/` - Stores images used in project documentation
- `assets/q3_workflow.png` - Workflow sequence picture
- `assets/accuracy_comparision_example.png` - Accuracy comparision of ideal, noisy and matigated picture sample
- `assets/dataset_distribution.png` - Distribution of synthethic classification dataset
- `report.md` - Explains the project, results, and conclusion

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The run may take a few minutes because it executes many circuit simulations. If needed for development, reduce `shots` in the processor JSON files, but restore 4096 for the official benchmark.

## Workflow

1. Created a dataset with 200 samples, 4 features, and 2 classes.
2. Split the data into 80% training data and 20% testing data.
3. Trained a 4-qubit Variational Quantum Classifier (VQC) using an ideal simulator.
4. Tested the trained model under ideal and synthetic noisy conditions.
5. Applied readout error mitigation to correct measurement errors.
6. Compared Processor A and Processor B using accuracy, circuit depth, and CX gate count.
7. Saved the results as CSV files, a JSON file, and graphs.

## Actual Measured Results

| Metric | Result |
|---|---:|
| Ideal accuracy | 80% |
| Noisy accuracy | 82.5% |
| Accuracy after mitigation | 82.5% |
| Mean circuit depth | 14.025 |
| Mean CX gate count | 6 |
| Estimated SWAP overhead | 0 |

Mitigation did not change the accuracy in this run. The optimizer reached its maximum number of evaluations without reporting successful convergence.

## Processor A/B Comparison

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits | 5 | 9 |
| Ideal accuracy | 80% | 80% |
| Noisy accuracy | 82.5% | 82.5% |
| Mitigated accuracy | 82.5% | 82.5% |
| Mean circuit depth | 14.025 | 14.025 |
| Mean CX gate count | 6 | 6 |
| Estimated SWAP overhead | 0 | 0 |

Both processors produced the same reported results in this run. Their circuit depth and CX gate count were also the same.

## Conclusion

This project tested a 4-qubit quantum classifier under ideal and synthetic noisy conditions. Readout error mitigation was applied, but it did not improve the accuracy in this run. Processor A and Processor B produced the same reported results.

The experiment used simulated noise rather than a real quantum computer. The results show how noise, measurement errors, and processor connectivity can be studied in quantum machine learning.

## References

- Qiskit Aer noise-model documentation: https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html
- Mitiq Zero-Noise Extrapolation guide: https://mitiq.readthedocs.io/en/stable/guide/zne.html
