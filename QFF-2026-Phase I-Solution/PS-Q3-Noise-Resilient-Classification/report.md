# PS-Q3 Experimental Report

## Objective
Evaluate a trained 4-qubit VQC under ideal and synthetic noise and compare Processor A and Processor B, with readout measurement mitigation.

## Experimental setup
- Dataset: 200 samples, four features, two classes; `random_state=42`.
- Split: 80% train, 20% test.
- Circuit: angle encoding, two trainable rotation layers, nearest-neighbour CX entanglement.
- Noise: one-qubit depolarizing p=0.001, CX depolarizing p=0.005, readout p=0.01.
- Shots: 4096; transpilation optimization level 1; seed 42.

## Results
Paste the actual values from `results/tables/processor_comparison.csv` here after running `python main.py`.

| Processor | Ideal accuracy | Noisy accuracy | Mitigated accuracy | Mean depth | Mean CX | Estimated SWAP overhead |
|---|---:|---:|---:|---:|---:|---:|
| A | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |
| B | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment | Run experiment |

## Discussion
Explain whether noise reduces accuracy, whether readout mitigation helps, and how Processor A/B connectivity changes transpilation metrics. Do not claim mitigation improves results unless the measured results show it.

## Limitations
This is a synthetic simulator experiment, not a run on real quantum hardware. The mitigation corrects readout probability for the output qubit and does not undo gate depolarization. Results are affected by finite shots and the simple circuit ansatz.
