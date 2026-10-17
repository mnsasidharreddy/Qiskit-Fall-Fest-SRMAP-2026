
import json
from pathlib import Path

import pandas as pd

from src.problem import load_dataset, prepare_data
from src.qml_model import train_quantum_model, predict_scores
from src.classical_baseline import train_classical_baseline
from src.processors import load_processor_profiles, apply_readout_noise
from src.metrics import calculate_metrics
from src.visualization import plot_comparison


ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("CUSTOM HYBRID QUANTUM NEURAL NETWORK")
    print("Classical Baseline vs Simulated Processor Profiles")
    print("=" * 60)

    X, y = load_dataset(n_samples=160)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    print(f"\nDataset samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining classical baseline...")
    classical_model = train_classical_baseline(X_train, y_train)
    classical_predictions = classical_model.predict(X_test)

    results = [{
        "model": "Classical Baseline",
        **calculate_metrics(y_test, classical_predictions)
    }]

    print("\nTraining quantum model...")
    weights = train_quantum_model(X_train, y_train, maxiter=60)
    quantum_scores = predict_scores(X_test, weights)

    profiles_folder = ROOT / "processors"
    profiles = load_processor_profiles(profiles_folder)

    for profile in profiles:
        noisy_scores = apply_readout_noise(
            quantum_scores,
            profile
        )
        predictions = (noisy_scores >= 0).astype(int)

        results.append({
            "model": profile["name"],
            **calculate_metrics(y_test, predictions)
        })

    results_dir = ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(
        tables_dir / "metrics.csv",
        index=False
    )

    with (results_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    plot_comparison(
        results_frame,
        figures_dir / "model_comparison.png"
    )

    print("\nACTUAL TEST RESULTS")
    print(results_frame.to_string(index=False))
    print("\nSaved metrics: results/tables/metrics.csv")
    print("Saved summary: results/summary.json")
    print("Saved chart: results/figures/model_comparison.png")
    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()