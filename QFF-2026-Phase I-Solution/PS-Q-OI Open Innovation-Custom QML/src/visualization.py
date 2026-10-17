
from pathlib import Path
import matplotlib.pyplot as plt


def plot_comparison(frame, output_path):
    """Save a bar chart comparing test accuracy across models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = frame.plot(
        x="model",
        y="accuracy",
        kind="bar",
        legend=False,
        ylim=(0, 1),
        title="Classical Baseline vs Simulated QML"
    )

    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
