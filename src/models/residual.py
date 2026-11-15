"""GNN that learns a correction on top of a descriptor model.

The descriptor model (GP by default) carries the chemistry that is easy to capture, mainly
mu_dopant and electronegativity trends. The GNN only has to explain what is left, and it is
kept small and heavily regularised so it cannot do much worse than its base.
"""
import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.models.baselines import DescriptorModel, out_of_fold_predictions
from src.models.train_gnn import evaluate, train_one_model


class ResidualDefectModel:
    """prediction = descriptor model + small descriptor-only GNN trained on out-of-fold residuals."""

    def __init__(
        self,
        base: str = "gp_iso",
        hidden_dim: int = 32,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 8,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.base_kind = base
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.base: Optional[DescriptorModel] = None
        self.gnn = None

    def fit(self, graphs: List[Data], x: np.ndarray, groups: np.ndarray) -> "ResidualDefectModel":
        """Fit on graphs with aligned descriptor matrix x and per-sample dopant labels."""
        y = np.array([float(g.y.item()) for g in graphs], dtype=np.float64)

        oof = out_of_fold_predictions(self.base_kind, x, y, groups, seed=self.seed)
        self.base = DescriptorModel(self.base_kind, seed=self.seed).fit(x, y, groups)

        residual_graphs = []
        for g, r in zip(graphs, y - oof):
            rg = copy.copy(g)
            rg.y = torch.tensor([float(r)], dtype=torch.float)
            residual_graphs.append(rg)

        self.gnn, _ = train_one_model(
            train_graphs=residual_graphs,
            val_graphs=None,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            seed=self.seed,
            hidden_dim=self.hidden_dim,
            device=self.device,
            model_kwargs={"use_embedding": False, "batch_norm": False, "dropout": 0.1},
        )
        return self

    def predict(self, graphs: List[Data], x: np.ndarray) -> Dict[str, Any]:
        """Return total prediction, base prediction and GNN correction (all in eV)."""
        base_mean, base_std = self.base.predict(x)
        correction = evaluate(self.gnn, graphs, y_mean=self.gnn.y_mean, y_std=self.gnn.y_std).predictions
        return {"pred": base_mean + correction, "base": base_mean, "correction": correction, "std": base_std}
