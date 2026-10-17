
import json
from pathlib import Path
import numpy as np


def load_processor_profiles(folder):
    profiles = []

    for path in sorted(Path(folder).glob("processor_*.json")):
        with path.open("r", encoding="utf-8") as file:
            profiles.append(json.load(file))

    if not profiles:
        raise FileNotFoundError(
            f"No processor configuration files found in {folder}"
        )

    return profiles


def apply_readout_noise(scores, profile, seed=42):
    """Apply simulated Gaussian perturbation to QML scores."""
    rng = np.random.default_rng(
        seed + int(profile["seed_offset"])
    )

    noise = rng.normal(
        loc=0.0,
        scale=float(profile["readout_noise_std"]),
        size=len(scores)
    )

    noisy_scores = np.asarray(scores) + noise

    return np.clip(noisy_scores, -1.0, 1.0)
