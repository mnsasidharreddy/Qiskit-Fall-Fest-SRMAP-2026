# QFF 2026 — PS-Q2: High-Dimensional Quantum Kernel Classifier

**Project:** ZZFeatureMap Support Vector Machine (QSVC) on Benchmark Data

## Task
Construct a quantum kernel matrix using fidelity state overlap |⟨φ(x)|φ(x')⟩|2 on a 4-qubit register. Train a classical
SVM with the quantum kernel. Compare kernel evaluation cost on A vs B.

## Data / Setup
Generate dataset using sklearn.datasets.make_moons(n_samples=100, noise=0.1, random_state=42).

## Structure

- `README.md` - Project details, setup steps, and main results
- `main.py` - Runs the complete project
- `main.ipynb` - Runs the project step by step and shows outputs
- `requirements.txt` - List of Python libraries needed
- `data/` - Stores the project data
- `data/raw/` - Stores the original dataset
- `data/raw/make_moons_raw.csv` - Contains 100 original data samples for reference
- `data/processed/` - Stores the data after processing
- `data/processed/make_moons_processed.csv` - Contains scaled data and train/test details for reference
- `processors/` - Stores Processor A and Processor B codes
- `processors/processor_a.json` - code for Processor A
- `processors/processor_b.json` - code for Processor B
- `src/` - Contains the main Python code files
- `src/__init__.py` - Initializes the Python package
- `src/problem.py` - Creates the dataset and prepares the data
- `src/feature_map.py` - Creates the 4-qubit quantum circuit
- `src/quantum_kernel.py` - Calculates similarities between data samples
- `src/classifier.py` - Trains the SVM model and checks its performance
- `src/processors.py` - Loads processor settings
- `src/routing.py` - Helps compile and route quantum circuits
- `src/metrics.py` - Calculates model scores and circuit details
- `results/` - Stores the project results
- `results/tables/` - Stores result tables
- `results/tables/classification_metrics.csv` - Stores SVM performance scores
- `results/tables/processor_comparison.csv` - Compares Processor A and B
- `results/tables/summary.json` - Stores a summary of the results
- `results/figures/` - Stores graphs created by the project
- `results/figures/dataset.png` - Shows the dataset as a graph
- `results/figures/kernel_matrix.png` - Shows the quantum kernel matrix
- `assets/` - Stores images used in project documentation
- `assets/qsvm_images/` - Stores supporting project images
- `assets/qsvm_images/dataset_distribution.png` - Shows the dataset for reference
- `assets/qsvm_images/quantum_kernel_matrix.png` - Shows the kernel matrix for reference
- `assets/qsvm_images/processor_resource_comparison.png` - Compares processor resources for reference
- `report.md` - Explains the project, results, and conclusion

## Setup
Python 3.10 or 3.11 is recommended.

## Run on Windows
```bat
cd QFF-2026-PS-Q2-QSVM
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The script writes CSV/JSON metrics and PNG figures to `results/`.

## Workflow

1) Created a dataset with 100 samples using the make_moons function. The noise value was set to 0.1 and the random state was 42.

2) Divided the data into 80 training samples and 20 testing samples.

3) Used a four-qubit quantum circuit to convert the input data into quantum states.

4) Calculated the similarity between pairs of quantum states using the fidelity formula.

5) Used the calculated similarities to create a quantum kernel matrix.

6) Trained a classical SVM model using the kernel matrix.

7) Tested the model and calculated its performance.

8) Compiled a sample circuit for Processor A and Processor B and compared their circuit resources.

9) Saved the results as CSV, JSON, and image files.

## Actual Measured Results

| Metric | Measured Result |
|---|---:|
| Accuracy | 95% |
| Precision | 90.91% |
| Recall | 100% |
| F1-Score | 95.24% |
| Correct Predictions | 19 out of 20 |
| Training Kernel Time | 0.124259 seconds |
| Testing Kernel Time | 0.121057 seconds |


## Processor A & B comparison

| Metric | Processor A | Processor B |
|---|---:|---:|
| Available Qubits | 5 | 9 |
| Circuit Depth | 48 | 25 |
| Total Gates | 98 | 64 |
| CX Gates | 14 | 8 |
| SWAP Gates | 0 | 0 |
| Kernel Pair Count | 3240 | 3240 |
| Estimated Gate Executions | 317520 | 207360 |


## Conclusion

In this project, we built a quantum kernel using a four-qubit circuit and used it with a classical SVM model. The model achieved 95% accuracy on the test data, correctly classifying 19 out of 20 samples.

We also compared Processor A and Processor B. Processor B had a lower circuit depth and fewer gates in the compiled sample circuit. This shows that the processor's connection layout can affect the number of resources needed to run a quantum circuit.

The project helped us understand how quantum kernels can be combined with classical machine learning and how different processor layouts can affect circuit cost.