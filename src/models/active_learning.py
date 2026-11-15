"""Active learning and ensemble uncertainty estimation for defect formation energy."""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.structures_to_graphs import (
    build_graph_from_cif,
    process_cif_directory,
)
from src.data_pipeline.structures_to_graphs import attach_descriptors
from src.models.baselines import DescriptorModel
from src.models.train_gnn import DefectGNN, DopantGNN
from pymatgen.core import Element
from src.utils.log import get_logger
from src.utils.seed import set_seed

logger = get_logger("active_learning")

# Safe globals for PyTorch 2.6+ weights_only compatibility
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass

# Options for the descriptor-only DefectGNN (see DefectGNN docstring). global_dim must match
# the length of data.desc, i.e. len(RICH_FEATURE_NAMES).
DESCRIPTOR_GNN_KWARGS = {"use_embedding": False, "global_dim": 15, "batch_norm": False, "dropout": 0.1}

# Module-level cache so ensemble checkpoints are read once rather than per call
_CACHED_ENSEMBLE_MODELS: Optional[List[torch.nn.Module]] = None


def _instantiate_and_load_checkpoint(ckpt: Any, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Instantiate model architecture from checkpoint dict or raw state_dict."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        y_mean = float(ckpt.get("y_mean", 0.0))
        y_std = float(ckpt.get("y_std", 1.0))
        cfg = ckpt.get("config", {})
        in_feat = cfg.get("in_features", 9)
        hidden_dim = cfg.get("hidden_dim", 64)
        edge_dim = cfg.get("edge_dim", 8)
        model_kwargs = cfg.get("model_kwargs", {}) or {}
    else:
        state_dict = ckpt
        y_mean = 0.0
        y_std = 1.0
        in_feat = 1
        hidden_dim = 64
        edge_dim = 8
        model_kwargs = {}

    # Deduce hidden_dim from state_dict if available
    if "node_proj.weight" in state_dict:
        hidden_dim = state_dict["node_proj.weight"].size(0)
    elif "embedding.weight" in state_dict:
        hidden_dim = state_dict["embedding.weight"].size(0)

    is_defect = any(
        "conv1.lin_" in k or "element_embedding" in k or "node_proj" in k
        for k in state_dict.keys()
    )

    if is_defect:
        model = DefectGNN(in_features=in_feat, hidden_dim=hidden_dim, edge_dim=edge_dim, **model_kwargs)
    else:
        if "embedding.weight" in state_dict:
            in_feat = state_dict["embedding.weight"].size(1)
        model = DopantGNN(in_features=in_feat, hidden_dim=hidden_dim)

    model.load_state_dict(state_dict)
    model.y_mean = y_mean
    model.y_std = y_std

    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def load_ensemble_models(
    model_dir: str = "models",
    num_models: int = 5,
    device: Optional[torch.device] = None,
) -> List[torch.nn.Module]:
    """Load trained ensemble member models from disk checkpoints into memory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: List[torch.nn.Module] = []

    for i in range(num_models):
        candidate_paths = [
            os.path.join(model_dir, f"ensemble_model_{i}.pth"),
            os.path.join(model_dir, f"ensemble_model_{i}.pt"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                    model = _instantiate_and_load_checkpoint(ckpt, device=device)
                    models.append(model)
                    break
                except Exception as exc:
                    logger.warning("Could not load ensemble checkpoint %s: %s", path, exc)

    if not models:
        for fallback_name in ["defect_gnn.pth", "dopant_gnn.pth"]:
            fb_path = os.path.join(model_dir, fallback_name)
            if os.path.exists(fb_path):
                try:
                    ckpt = torch.load(fb_path, map_location=device, weights_only=False)
                    model = _instantiate_and_load_checkpoint(ckpt, device=device)
                    models.append(model)
                    logger.info("Loaded fallback single model from %s", fb_path)
                    break
                except Exception as exc:
                    logger.warning("Could not load fallback checkpoint %s: %s", fb_path, exc)

    return models


def predict_with_uncertainty(
    cif_or_graph: Union[str, Data],
    ensemble_models: Optional[List[torch.nn.Module]] = None,
) -> Tuple[float, float]:
    """Predict energy and epistemic uncertainty in eV using an ensemble of models.

    Args:
        cif_or_graph: CIF file path or PyG Data object.
        ensemble_models: Optional list of trained ensemble models.

    Returns:
        Tuple of (mean_energy, std_uncertainty) in eV.
    """
    global _CACHED_ENSEMBLE_MODELS

    if ensemble_models is None or len(ensemble_models) == 0:
        if _CACHED_ENSEMBLE_MODELS is None or len(_CACHED_ENSEMBLE_MODELS) == 0:
            _CACHED_ENSEMBLE_MODELS = load_ensemble_models()
        ensemble_models = _CACHED_ENSEMBLE_MODELS

    if not ensemble_models:
        raise ValueError("No ensemble models provided or found on disk in models/")

    if isinstance(cif_or_graph, str):
        graph = build_graph_from_cif(cif_or_graph)
    else:
        graph = cif_or_graph

    device = next(ensemble_models[0].parameters()).device
    batch = Batch.from_data_list([graph]).to(device)

    predictions_ev: List[float] = []

    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            if isinstance(model, DopantGNN) and batch.x is not None and batch.x.size(-1) > 1:
                batch_copy = batch.clone()
                batch_copy.x = batch.x[:, 0:1] / 100.0
                if batch_copy.edge_attr is not None and batch_copy.edge_attr.dim() > 1 and batch_copy.edge_attr.size(-1) > 1:
                    batch_copy.edge_attr = batch_copy.edge_attr[:, 0:1]
                    batch_copy.edge_weight = batch_copy.edge_attr[:, 0]
                out = model(batch_copy)
            else:
                out = model(batch)
            y_mean = getattr(model, "y_mean", 0.0)
            y_std = getattr(model, "y_std", 1.0)
            pred_ev = float(out.view(-1).item() * y_std + y_mean)
            predictions_ev.append(pred_ev)

    mean_pred = float(np.mean(predictions_ev))
    if len(predictions_ev) > 1:
        std_dev = float(np.std(predictions_ev, ddof=1))
    else:
        std_dev = 0.0

    return mean_pred, std_dev


def _load_graphs_dataset(data_path: Optional[str] = None) -> List[Data]:
    """Load graph dataset tensor from disk or generate from unrelaxed CIF directory."""
    data_path = data_path or CONFIG.graph_data_path

    if os.path.exists(data_path):
        try:
            graphs = torch.load(data_path, weights_only=False)
            if isinstance(graphs, list):
                return graphs
        except Exception as exc:
            logger.warning("Could not load graph tensor from %s: %s", data_path, exc)

    if os.path.exists(CONFIG.unrelaxed_dir):
        try:
            return process_cif_directory(
                unrelaxed_dir=CONFIG.unrelaxed_dir,
                csv_path=CONFIG.energies_csv_path,
                manifest_path=CONFIG.manifest_path,
                output_path=data_path,
            )
        except Exception as exc:
            logger.warning("Could not construct graphs from %s: %s", CONFIG.unrelaxed_dir, exc)

    return []


def _cluster_bootstrap(graphs: List[Data], rng: np.random.Generator) -> np.ndarray:
    """Bootstrap indices that resample whole dopant elements, not single structures.

    One dopant appears in several structures (both hosts, both SiC sites). Resampling single
    structures leaves a twin of almost every point in each member's training set, so the
    members agree and the ensemble std is too small exactly where the model has not seen a
    new element. Resampling elements keeps that spread. Falls back to plain resampling when
    graphs carry no dopant label.
    """
    n = len(graphs)
    labels = [getattr(g, "dopant_z", None) for g in graphs]
    if any(l is None for l in labels):
        return rng.choice(n, size=n, replace=True)
    labels = np.array([int(l) for l in labels])
    unique = np.unique(labels)
    if len(unique) < 2:
        return rng.choice(n, size=n, replace=True)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.where(labels == d)[0] for d in drawn])


class ActiveLearningLoop:
    """Active learning workflow using a bootstrapped GNN ensemble."""

    def __init__(
        self,
        initial_data_path: str = "data/graph_data/processed_graphs.pt",
        model_save_path: str = "models",
    ):
        self.data_path = initial_data_path
        self.model_save_path = model_save_path
        self.training_pool = _load_graphs_dataset(initial_data_path)
        self.models: List[torch.nn.Module] = []

    def load_ensemble(self, num_models: int = 5) -> List[torch.nn.Module]:
        """Load ensemble models from model directory into self.models."""
        self.models = load_ensemble_models(
            model_dir=self.model_save_path,
            num_models=num_models,
        )
        return self.models

    def train_ensemble(
        self,
        num_models: int = 5,
        epochs: int = 30,
        batch_size: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[torch.nn.Module]:
        """Train independent models with distinct seeds, target standardisation, and bootstrapped sampling."""
        global _CACHED_ENSEMBLE_MODELS
        print(f"\nTraining Bootstrapped Ensemble of {num_models} Models")
        os.makedirs(self.model_save_path, exist_ok=True)
        self.models = []
        n_samples = len(self.training_pool)

        if n_samples == 0:
            print("Warning: training_pool is empty.")
            return self.models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_feat = self.training_pool[0].x.size(-1) if self.training_pool[0].x is not None else 9
        use_defect_gnn = in_feat >= 6

        # Default to the descriptor-aware GNN when graphs have descriptors (see CONFIG)
        if model_kwargs is None and use_defect_gnn and CONFIG.use_descriptor_gnn:
            if all(getattr(g, "dopant_z", None) is not None for g in self.training_pool):
                attach_descriptors(self.training_pool, CONFIG.references_csv_path)
            if all(getattr(g, "desc", None) is not None for g in self.training_pool):
                model_kwargs = dict(CONFIG.descriptor_gnn_kwargs)

        for i in range(num_models):
            member_seed = 42 + i
            set_seed(member_seed)

            # Bootstrapped sampling with replacement using numpy Generator seeded per member
            rng = np.random.default_rng(member_seed)
            bootstrap_indices = _cluster_bootstrap(self.training_pool, rng)
            bootstrap_subset = [self.training_pool[idx] for idx in bootstrap_indices]

            # Target standardisation fitted on the bootstrap sample
            targets = [float(g.y.item()) for g in bootstrap_subset if g.y is not None]
            if targets:
                y_mean = float(np.mean(targets))
                y_std = float(np.std(targets)) if len(targets) > 1 and np.std(targets) > 1e-6 else 1.0
            else:
                y_mean = 0.0
                y_std = 1.0

            if use_defect_gnn:
                model = DefectGNN(in_features=in_feat, hidden_dim=64, edge_dim=8, **(model_kwargs or {}))
                model.fit_desc_scaler(bootstrap_subset)
            else:
                model = DopantGNN(in_features=in_feat, hidden_dim=64)

            model.y_mean = y_mean
            model.y_std = y_std
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            criterion = torch.nn.SmoothL1Loss()
            loader = DataLoader(bootstrap_subset, batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(epochs):
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    y_std_target = (batch.y.view(-1, 1) - y_mean) / y_std
                    loss = criterion(out, y_std_target)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Save checkpoint dict with state_dict, config, y_mean, y_std, seed
            model_file = os.path.join(self.model_save_path, f"ensemble_model_{i}.pth")
            checkpoint = {
                "state_dict": model.state_dict(),
                "config": {
                    "in_features": in_feat,
                    "hidden_dim": 64,
                    "edge_dim": 8,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "model_kwargs": dict(model_kwargs or {}),
                },
                "y_mean": y_mean,
                "y_std": y_std,
                "seed": member_seed,
            }
            torch.save(checkpoint, model_file)
            self.models.append(model)
            print(f"Trained member {i + 1}/{num_models} (y_mean={y_mean:.3f}, y_std={y_std:.3f})")

        _CACHED_ENSEMBLE_MODELS = self.models
        print(f"Saved {num_models} ensemble checkpoints to {self.model_save_path}/")
        return self.models

    def predict_with_uncertainty(self, new_graph_or_cif: Union[str, Data]) -> Tuple[float, float]:
        """Predict with the ensemble and measure epistemic uncertainty."""
        if not self.models:
            self.load_ensemble()
        return predict_with_uncertainty(new_graph_or_cif, ensemble_models=self.models)


def _train_transient_ensemble(
    train_graphs: List[Data],
    num_models: int = 5,
    epochs: int = 20,
    batch_size: int = 4,
    seed_base: int = 42,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> List[torch.nn.Module]:
    """Train transient in-memory ensemble members for pool benchmark iterations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: List[torch.nn.Module] = []
    n_train = len(train_graphs)
    in_feat = train_graphs[0].x.size(-1) if train_graphs[0].x is not None else 9
    use_defect_gnn = in_feat >= 6

    for m_idx in range(num_models):
        m_seed = seed_base + m_idx
        set_seed(m_seed)
        m_rng = np.random.default_rng(m_seed)

        boot_idx = _cluster_bootstrap(train_graphs, m_rng)
        boot_subset = [train_graphs[b] for b in boot_idx]

        targets = [float(g.y.item()) for g in boot_subset if g.y is not None]
        if targets:
            y_mean = float(np.mean(targets))
            y_std = float(np.std(targets)) if len(targets) > 1 and np.std(targets) > 1e-6 else 1.0
        else:
            y_mean = 0.0
            y_std = 1.0

        if use_defect_gnn:
            model = DefectGNN(in_features=in_feat, hidden_dim=64, edge_dim=8, **(model_kwargs or {}))
            model.fit_desc_scaler(boot_subset)
        else:
            model = DopantGNN(in_features=in_feat, hidden_dim=64)

        model.y_mean = y_mean
        model.y_std = y_std
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = torch.nn.SmoothL1Loss()
        loader = DataLoader(boot_subset, batch_size=min(batch_size, n_train), shuffle=True)

        model.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                y_std_target = (batch.y.view(-1, 1) - y_mean) / y_std
                loss = criterion(out, y_std_target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        models.append(model)

    return models


def plot_learning_curve(
    df_benchmark: pd.DataFrame,
    save_path: str = "results/al_learning_curve.png",
) -> None:
    """Plot mean held-out MAE across acquisition steps with shaded spread over repeats."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    df = df_benchmark.copy()
    if "surrogate" not in df.columns:
        df["surrogate"] = "ensemble"
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for k, ((surr, strat), sub_df) in enumerate(sorted(df.groupby(["surrogate", "strategy"]))):
        grouped = sub_df.groupby("n_train")["mae_ev"].agg(["mean", "std"]).sort_index()
        x = grouped.index.to_numpy()
        mean_mae = grouped["mean"].to_numpy()
        std_mae = grouped["std"].fillna(0.0).to_numpy()
        c = palette[k % len(palette)]
        ax.plot(x, mean_mae, marker="o", linewidth=2, color=c, label=f"{surr} / {strat}")
        ax.fill_between(x, mean_mae - std_mae, mean_mae + std_mae, color=c, alpha=0.15)

    ax.set_xlabel("Acquired Training Set Size", fontsize=11)
    ax.set_ylabel("Held-out MAE (eV)", fontsize=11)
    ax.set_title("Active Learning Pool Benchmark", fontsize=12, fontweight="medium")
    ax.grid(True, linestyle="dotted", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved learning curve plot to %s", save_path)


def _dopant_symbol(g: Data) -> str:
    """Dopant element symbol of a graph."""
    return Element.from_Z(int(g.dopant_z)).symbol


def _ensemble_predict(
    ensemble: List[torch.nn.Module], graphs: List[Data], device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std (eV) of an ensemble over a list of graphs."""
    means, stds = [], []
    for g in graphs:
        batch = Batch.from_data_list([g]).to(device)
        vals = []
        with torch.no_grad():
            for model in ensemble:
                vals.append(float(model(batch).view(-1).item() * model.y_std + model.y_mean))
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def _desc_matrix(graphs: List[Data]) -> np.ndarray:
    return np.stack([g.desc.view(-1).cpu().numpy() for g in graphs]).astype(np.float64)


def _fit_and_predict(
    surrogate: str,
    train_graphs: List[Data],
    query_sets: List[List[Data]],
    num_models: int,
    epochs: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
    model_kwargs: Optional[Dict[str, Any]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Train the surrogate on train_graphs and return (mean, std) for each query set."""
    if surrogate == "gp":
        y = np.array([float(g.y.item()) for g in train_graphs])
        groups = np.array([_dopant_symbol(g) for g in train_graphs])
        gp = DescriptorModel("gp_iso", seed=seed_base).fit(_desc_matrix(train_graphs), y, groups)
        outputs = []
        for graphs in query_sets:
            if not graphs:
                outputs.append((np.array([]), np.array([])))
                continue
            mean, std = gp.predict(_desc_matrix(graphs))
            outputs.append((mean, std))
        return outputs

    ensemble = _train_transient_ensemble(
        train_graphs=train_graphs,
        num_models=num_models,
        epochs=epochs,
        batch_size=batch_size,
        seed_base=seed_base,
        device=device,
        model_kwargs=model_kwargs,
    )
    return [
        _ensemble_predict(ensemble, graphs, device) if graphs else (np.array([]), np.array([]))
        for graphs in query_sets
    ]


def run_pool_benchmark(
    strategy: str = "both",
    seed_size: int = 6,
    batch_size: int = 4,
    budget: int = 20,
    n_repeats: int = 3,
    dataset: Optional[List[Data]] = None,
    data_path: Optional[str] = None,
    epochs: int = 20,
    num_models: int = 5,
    seed: int = 42,
    output_csv: str = "results/al_benchmark.csv",
    output_plot: str = "results/al_learning_curve.png",
    output_calibration_csv: str = "results/uncertainty_calibration.csv",
    device: Optional[torch.device] = None,
    holdout_by: str = "dopant",
    holdout_fraction: float = 0.25,
    surrogate: str = "ensemble",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pool-based active learning benchmark with a simulated oracle.

    Compares acquisition by maximum predictive std against random acquisition.

    holdout_by="dopant" reserves whole dopant elements as a fixed test set per repeat, so the
    score is comparable with leave-one-dopant-out CV. The older behaviour, scoring on the
    unacquired remainder (holdout_by="none"), lets a dopant sit in both train and test through
    its other site or host, which flatters every model. It is still reported as mae_pool_ev.

    surrogate="ensemble" uses the bootstrapped GNN ensemble, "gp" a Gaussian process on the
    graph-level descriptors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if surrogate not in ("ensemble", "gp"):
        raise ValueError("surrogate must be 'ensemble' or 'gp'")

    if dataset is None:
        dataset = _load_graphs_dataset(data_path)

    valid_dataset = [g for g in dataset if g.y is not None and not getattr(g, "ood_flag", False)]
    if surrogate == "gp" or (model_kwargs or {}).get("global_dim", 0) > 0:
        attach_descriptors(valid_dataset, CONFIG.references_csv_path)

    dopants_all = sorted({_dopant_symbol(g) for g in valid_dataset})
    n_test_dopants = max(1, int(round(holdout_fraction * len(dopants_all)))) if holdout_by == "dopant" else 0

    strategies_to_run = (
        ["uncertainty", "random"] if strategy.lower() in ["both", "all"] else [strategy.lower()]
    )
    benchmark_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []

    print(
        f"Active learning pool benchmark: {len(valid_dataset)} structures, {len(dopants_all)} dopants, "
        f"surrogate={surrogate}, holdout_by={holdout_by}, seed={seed_size}, batch={batch_size}, "
        f"budget={budget}, repeats={n_repeats}"
    )

    for strat in strategies_to_run:
        print(f"\nAcquisition Strategy: {strat.upper()}")

        for repeat_idx in range(n_repeats):
            # The split depends on the repeat only, so both strategies see identical folds
            split_rng = np.random.default_rng(seed + repeat_idx * 1000)
            run_rng = np.random.default_rng(seed + repeat_idx * 1000 + (1 if strat == "random" else 2))

            test_dopants = (
                set(split_rng.choice(dopants_all, size=n_test_dopants, replace=False).tolist())
                if n_test_dopants
                else set()
            )
            test_graphs = [g for g in valid_dataset if _dopant_symbol(g) in test_dopants]
            pool = [g for g in valid_dataset if _dopant_symbol(g) not in test_dopants]

            if len(pool) < seed_size + batch_size:
                print(f"Pool of {len(pool)} too small for seed_size={seed_size}; skipping repeat.")
                continue

            acquired = list(split_rng.choice(len(pool), size=seed_size, replace=False))
            candidates = [i for i in range(len(pool)) if i not in acquired]
            max_acquired = min(seed_size + budget, len(pool) - 1)
            step = 0

            while True:
                train_graphs = [pool[i] for i in acquired]
                cand_graphs = [pool[i] for i in candidates]

                (cand_mean, cand_std), (test_mean, test_std) = _fit_and_predict(
                    surrogate, train_graphs, [cand_graphs, test_graphs], num_models, epochs,
                    batch_size, seed + repeat_idx * 1000 + step * 100, device, model_kwargs,
                )

                cand_true = np.array([float(g.y.item()) for g in cand_graphs])
                pool_err = np.abs(cand_mean - cand_true)
                if len(test_graphs):
                    test_true = np.array([float(g.y.item()) for g in test_graphs])
                    eval_err = np.abs(test_mean - test_true)
                    eval_std = test_std
                else:
                    eval_err, eval_std = pool_err, cand_std

                mae = float(np.mean(eval_err))
                rmse = float(np.sqrt(np.mean(eval_err**2)))
                benchmark_rows.append({
                    "strategy": strat,
                    "surrogate": surrogate,
                    "repeat": repeat_idx,
                    "step": step,
                    "n_train": len(acquired),
                    "n_remainder": len(candidates),
                    "n_heldout": len(test_graphs),
                    "mae_ev": mae,
                    "rmse_ev": rmse,
                    "mae_pool_ev": float(np.mean(pool_err)),
                })

                if len(eval_std) > 2 and np.var(eval_std) > 1e-8 and np.var(eval_err) > 1e-8:
                    res = stats.spearmanr(eval_std, eval_err)
                    rho, p_val = float(res.statistic), float(res.pvalue)
                else:
                    rho, p_val = float("nan"), float("nan")
                calibration_rows.append({
                    "strategy": strat,
                    "surrogate": surrogate,
                    "repeat": repeat_idx,
                    "step": step,
                    "n_samples": len(eval_err),
                    "spearman_corr": rho,
                    "p_value": p_val,
                })
                print(
                    f"  [{strat}/{surrogate}] repeat {repeat_idx + 1}/{n_repeats} step {step}: "
                    f"N_train={len(acquired)}, held-out MAE={mae:.4f} eV, rho={rho:.3f}"
                )

                if len(acquired) >= max_acquired or not candidates:
                    break
                n_to_acquire = min(batch_size, len(candidates))
                if strat == "uncertainty":
                    order = np.argsort(-cand_std)[:n_to_acquire]
                    selected = [candidates[k] for k in order]
                else:
                    selected = list(run_rng.choice(candidates, size=n_to_acquire, replace=False))
                for sel in selected:
                    candidates.remove(sel)
                    acquired.append(sel)
                step += 1

    df_benchmark = pd.DataFrame(benchmark_rows)
    df_calibration = pd.DataFrame(calibration_rows)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_benchmark.to_csv(output_csv, index=False)
    print(f"\nSaved benchmark metrics to {output_csv}")
    os.makedirs(os.path.dirname(output_calibration_csv) or ".", exist_ok=True)
    df_calibration.to_csv(output_calibration_csv, index=False)
    print(f"Saved uncertainty calibration metrics to {output_calibration_csv}")

    if not df_benchmark.empty:
        plot_learning_curve(df_benchmark, save_path=output_plot)
        print(f"Saved learning curve plot to {output_plot}")

    return df_benchmark, df_calibration


def main(argv: Optional[List[str]] = None) -> None:
    """CLI execution entrypoint for ensemble training and active learning benchmark."""
    parser = argparse.ArgumentParser(
        description="Active learning ensemble training and simulated oracle benchmark."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run active learning simulated oracle pool benchmark.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["uncertainty", "random", "both"],
        help="Acquisition strategy to benchmark.",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=6,
        help="Initial training seed set size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch acquisition size per iteration.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Total acquisition budget count.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of independent benchmark repetitions.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs per ensemble model.",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of ensemble members.",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="ensemble",
        choices=["ensemble", "gp"],
        help="Model used for predictions and acquisition in the benchmark.",
    )
    parser.add_argument(
        "--holdout-by",
        type=str,
        default="dopant",
        choices=["dopant", "none"],
        help="Score on whole held-out dopants (default) or on the unacquired remainder.",
    )
    parser.add_argument(
        "--descriptor-gnn",
        action="store_true",
        help="Use the descriptor-only DefectGNN (no element embedding, descriptors in readout).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=CONFIG.graph_data_path,
        help="Path to graphs dataset tensor.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/al_benchmark.csv",
        help="Path for benchmark CSV output.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="results/al_learning_curve.png",
        help="Path for learning curve plot output.",
    )
    parser.add_argument(
        "--output-calibration",
        type=str,
        default="results/uncertainty_calibration.csv",
        help="Path for uncertainty calibration CSV output.",
    )
    args = parser.parse_args(argv)

    if args.benchmark:
        run_pool_benchmark(
            strategy=args.strategy,
            seed_size=args.seed_size,
            batch_size=args.batch_size,
            budget=args.budget,
            n_repeats=args.n_repeats,
            data_path=args.data_path,
            epochs=args.epochs,
            num_models=args.num_models,
            output_csv=args.output_csv,
            output_plot=args.output_plot,
            output_calibration_csv=args.output_calibration,
            surrogate=args.surrogate,
            holdout_by=args.holdout_by,
            model_kwargs=DESCRIPTOR_GNN_KWARGS if args.descriptor_gnn else None,
        )
    else:
        # Default behavior: train ensemble
        loop = ActiveLearningLoop(initial_data_path=args.data_path)
        if loop.training_pool:
            loop.train_ensemble(num_models=args.num_models, epochs=args.epochs)
        else:
            print(f"Dataset not found at {args.data_path}")


if __name__ == "__main__":
    main()
