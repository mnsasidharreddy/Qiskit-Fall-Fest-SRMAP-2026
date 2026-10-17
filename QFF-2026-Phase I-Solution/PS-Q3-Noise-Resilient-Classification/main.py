from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.data import make_data
from src.model import train_vqc
from src.experiment import evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

def main():
    X_train, X_test, y_train, y_test = make_data()
    theta, training_info = train_vqc(X_train, y_train)
    print("Training:", training_info)

    all_summaries = []
    for processor in ("A", "B"):
        print(f"Evaluating Processor {processor}...")
        predictions, summary = evaluate(X_test, y_test, theta, processor)
        predictions.to_csv(OUT / "tables" / f"predictions_{processor}.csv", index=False)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUT / "tables" / "processor_comparison.csv", index=False)
    (OUT / "tables" / "training_info.json").write_text(
        json.dumps(training_info, indent=2), encoding="utf-8"
    )
    print("\nCOMPARISON")
    print(summary_df.to_string(index=False))

    plot_df = summary_df.set_index("processor")[["ideal_accuracy", "noisy_accuracy", "mitigated_accuracy"]]
    ax = plot_df.plot(kind="bar", ylim=(0, 1), rot=0, figsize=(8, 5))
    ax.set_ylabel("Test accuracy")
    ax.set_title("PS-Q3: Ideal vs Noisy vs Readout-Mitigated")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "accuracy_comparison.png", dpi=160)
    print(f"\nSaved results to: {OUT}")

if __name__ == "__main__":
    main()
