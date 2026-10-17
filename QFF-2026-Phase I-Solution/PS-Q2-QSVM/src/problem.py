from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def make_dataset(random_state=42, test_size=0.2):
    X, y = make_moons(n_samples=100, noise=0.1, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train) * 2 - 1
    X_test = scaler.transform(X_test) * 2 - 1
    return X_train, X_test, y_train, y_test, scaler
