"""Descriptor-based regressors used as baselines and as the base of the residual GNN.

Everything here works on a plain feature matrix, so it can be compared with the GNN on the
same folds and the same inputs.
"""
from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("ridge", "gp_iso", "gp_ard")


def grouped_splits(groups: np.ndarray, n_splits: int = 6):
    """GroupKFold splits that keep all structures of one dopant in the same fold."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return []  # nothing to split on, callers fall back to a simple default
    k = min(n_splits, n_groups)
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


class DescriptorModel:
    """Standardised descriptors followed by ridge or a Gaussian process.

    ridge:  alpha chosen by grouped cross validation inside the training data.
    gp_iso: Matern 5/2 with one shared length scale. Behaves well at N of about 50.
    gp_ard: one length scale per feature. Overfits easily at this N, kept as a check.
    """

    def __init__(self, kind: str = "gp_iso", seed: int = 0, restarts: int = 2):
        if kind not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{kind}', expected one of {MODEL_NAMES}")
        self.kind = kind
        self.seed = seed
        self.restarts = restarts
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> "DescriptorModel":
        xs = self.scaler.fit_transform(x)
        if self.kind == "ridge":
            splits = grouped_splits(groups) if groups is not None else []
            # cv=None falls back to efficient leave-one-out when there are too few groups
            self.model = RidgeCV(alphas=np.logspace(-2, 3, 25), cv=splits or None).fit(xs, y)
        else:
            length = np.ones(xs.shape[1]) if self.kind == "gp_ard" else 1.0
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                length_scale=length, length_scale_bounds=(0.3, 100.0), nu=2.5
            ) + WhiteKernel(0.1, (1e-4, 1.0))
            self.model = GaussianProcessRegressor(
                kernel, normalize_y=True, n_restarts_optimizer=self.restarts, random_state=self.seed
            ).fit(xs, y)
        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (mean, std). std is None for ridge, which has no predictive variance."""
        xs = self.scaler.transform(x)
        if self.kind == "ridge":
            return self.model.predict(xs).astype(np.float64), None
        mean, std = self.model.predict(xs, return_std=True)
        return mean.astype(np.float64), std.astype(np.float64)


def out_of_fold_predictions(
    kind: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 6, seed: int = 0
) -> np.ndarray:
    """Grouped out-of-fold predictions on the training set.

    The residual GNN learns from y minus these values. In-sample ridge or GP residuals would
    be too small, because the base model has already seen those points.
    """
    oof = np.full(len(y), float(np.mean(y)), dtype=np.float64)
    for train_idx, val_idx in grouped_splits(groups, n_splits):
        model = DescriptorModel(kind, seed=seed, restarts=1).fit(x[train_idx], y[train_idx], groups[train_idx])
        oof[val_idx] = model.predict(x[val_idx])[0]
    return oof
