
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_dataset(n_samples=160):
    """Generate a reproducible binary classification dataset."""
    X, y = make_moons(
        n_samples=n_samples,
        noise=0.18,
        random_state=SEED
    )
    return X.astype(float), y.astype(int)


def prepare_data(X, y):
    """Split, scale, and encode features as quantum rotation angles."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -2.5, 2.5) * (np.pi / 2.5)
    X_test = np.clip(X_test, -2.5, 2.5) * (np.pi / 2.5)

    return X_train, X_test, y_train, y_test
