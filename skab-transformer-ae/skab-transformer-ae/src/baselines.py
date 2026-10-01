"""Classical baselines that return an anomaly score per window."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def _flatten(x):
    return x.reshape(len(x), -1)


def isolation_forest_scores(x_train, x_val, x_test, seed=0):
    """Isolation Forest on flattened windows. Score = negated sklearn score_samples."""
    model = IsolationForest(n_estimators=200, random_state=seed)
    model.fit(_flatten(x_train))
    return (
        -model.score_samples(_flatten(x_val)),
        -model.score_samples(_flatten(x_test)),
    )


def pca_reconstruction_scores(x_train, x_val, x_test, variance=0.95, seed=0):
    """PCA fitted on normal windows. Score = mean squared reconstruction error."""
    model = PCA(n_components=variance, svd_solver="full", random_state=seed)
    model.fit(_flatten(x_train))

    def score(x):
        flat = _flatten(x)
        recon = model.inverse_transform(model.transform(flat))
        return np.mean((flat - recon) ** 2, axis=1)

    return score(x_val), score(x_test)
