"""Cross-validation and model evaluation benchmarks for defect formation energies."""
import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Element
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.structures_to_graphs import (
    build_graph_legacy,
    get_valence_electron_count,
    process_cif_directory,
)
from src.models.train_gnn import (
    DopantGNN,
    EvaluationMetrics,
    evaluate,
    train_one_model,
)
from src.utils.log import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluate")

# PyTorch safe globals compatibility
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass


@dataclass
class SampleData:
    """Unified container storing multimodal representations of a defect structure."""

    filename: str
    host: str
    dopant: str
    site: str
    y: float
    defect_graph: Data
    legacy_graph: Data
    descriptor: np.ndarray


def extract_dopant_descriptors(dopant_symbol: str, host: str) -> np.ndarray:
    """Extract physical and chemical descriptors for dopant element with host one-hot encoding.

    Descriptors:
        0: Atomic number Z
        1: Pauling electronegativity
        2: Covalent or atomic radius in Angstroms
        3: Periodic table group number
        4: Periodic table row or period number
        5: Valence electron count
        6: Host indicator for SiC (1.0 or 0.0)
        7: Host indicator for GaN (1.0 or 0.0)
    """
    try:
        el = Element(dopant_symbol)
        z = float(el.Z)
        electronegativity = float(getattr(el, "X", 0.0) or 0.0)
        covalent_radius = float(getattr(el, "atomic_radius", 1.5) or 1.5)
        group_val = float(getattr(el, "group", 1) or 1)
        period_val = float(getattr(el, "row", 1) or 1)
        val_count = float(get_valence_electron_count(el))
    except Exception:
        z = 0.0
        electronegativity = 0.0
        covalent_radius = 1.5
        group_val = 1.0
        period_val = 1.0
        val_count = 1.0

    host_sic = 1.0 if host == "SiC" else 0.0
    host_gan = 1.0 if host == "GaN" else 0.0

    return np.array(
        [
            z,
            electronegativity,
            covalent_radius,
            group_val,
            period_val,
            val_count,
            host_sic,
            host_gan,
        ],
        dtype=np.float32,
    )


def build_legacy_fallback_graph(defect_graph: Data) -> Data:
    """Create legacy-compatible isotropic graph from DefectGNN Data object."""
    if hasattr(defect_graph, "z") and defect_graph.z is not None:
        x_legacy = defect_graph.z.float().unsqueeze(-1) / 100.0
    elif defect_graph.x is not None and defect_graph.x.dim() > 1:
        x_legacy = defect_graph.x[:, 0:1] / 100.0
    else:
        x_legacy = torch.ones((defect_graph.num_nodes or 1, 1), dtype=torch.float)

    edge_index = getattr(defect_graph, "edge_index", torch.empty((2, 0), dtype=torch.long))
    num_edges = edge_index.size(1) if edge_index.dim() > 1 else 0

    if hasattr(defect_graph, "edge_attr") and defect_graph.edge_attr is not None:
        if defect_graph.edge_attr.size(0) == num_edges:
            edge_attr_legacy = defect_graph.edge_attr[:, 0:1]
        else:
            edge_attr_legacy = torch.ones((num_edges, 1), dtype=torch.float)
    else:
        edge_attr_legacy = torch.ones((num_edges, 1), dtype=torch.float)

    legacy_data = Data(
        x=x_legacy,
        edge_index=edge_index,
        edge_attr=edge_attr_legacy,
        y=defect_graph.y,
    )
    legacy_data.filename = getattr(defect_graph, "filename", "structure.cif")
    return legacy_data


def parse_sample_metadata(graph: Data) -> Tuple[str, str, str, str]:
    """Extract filename, host, dopant symbol, and site from graph attributes."""
    filename = str(getattr(graph, "filename", getattr(graph, "material_id", "")))
    host = str(getattr(graph, "host", ""))
    site = str(getattr(graph, "site", ""))
    dopant_symbol = ""

    if hasattr(graph, "dopant_z") and graph.dopant_z is not None:
        try:
            dopant_symbol = Element.from_Z(int(graph.dopant_z)).symbol
        except Exception:
            pass

    if not host and filename:
        if filename.startswith("SiC"):
            host = "SiC"
        elif filename.startswith("GaN"):
            host = "GaN"

    if not dopant_symbol and filename:
        match = re.search(r"_doped_([A-Za-z]+)", filename)
        if match:
            dopant_symbol = match.group(1)

    if not site and filename:
        match = re.search(r"_([0-9][a-z]_[0-9]+)\.cif", filename)
        if match:
            site = match.group(1)

    host = host or "unknown"
    dopant_symbol = dopant_symbol or "unknown"
    site = site or "unknown"
    filename = filename or f"{host}_doped_{dopant_symbol}_{site}.cif"

    return filename, host, dopant_symbol, site


def prepare_dataset_items(
    defect_graphs: List[Data],
    unrelaxed_dir: Optional[str] = None,
) -> List[SampleData]:
    """Convert DefectGNN graph instances into structured multimodal SampleData items."""
    items: List[SampleData] = []

    for g in defect_graphs:
        if g.y is None:
            continue

        filename, host, dopant, site = parse_sample_metadata(g)
        y_val = float(g.y.item())

        legacy_g: Optional[Data] = None
        if unrelaxed_dir and filename:
            cif_path = os.path.join(unrelaxed_dir, filename)
            if os.path.exists(cif_path):
                try:
                    legacy_g = build_graph_legacy(cif_path, y_val=y_val)
                except Exception as exc:
                    logger.debug("Failed building legacy graph from CIF %s: %s", cif_path, exc)

        if legacy_g is None:
            legacy_g = build_legacy_fallback_graph(g)

        descriptor = extract_dopant_descriptors(dopant, host)

        item = SampleData(
            filename=filename,
            host=host,
            dopant=dopant,
            site=site,
            y=y_val,
            defect_graph=g,
            legacy_graph=legacy_g,
            descriptor=descriptor,
        )
        items.append(item)

    return items


def load_evaluation_dataset(
    data_path: Optional[str] = None,
    unrelaxed_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> List[SampleData]:
    """Load graphs from disk tensor file or process unrelaxed CIF directory."""
    data_path = data_path or CONFIG.graph_data_path
    unrelaxed_dir = unrelaxed_dir or CONFIG.unrelaxed_dir
    csv_path = csv_path or CONFIG.energies_csv_path
    manifest_path = manifest_path or CONFIG.manifest_path

    defect_graphs: List[Data] = []

    if os.path.exists(data_path):
        try:
            loaded = torch.load(data_path, weights_only=False)
            if isinstance(loaded, list):
                defect_graphs = loaded
                logger.info("Loaded %d graphs from %s", len(defect_graphs), data_path)
        except Exception as exc:
            logger.warning("Failed loading tensor dataset %s: %s", data_path, exc)

    if not defect_graphs and os.path.exists(unrelaxed_dir):
        logger.info("Constructing graphs from CIFs in %s", unrelaxed_dir)
        defect_graphs = process_cif_directory(
            unrelaxed_dir=unrelaxed_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
            output_path=data_path,
        )

    return prepare_dataset_items(defect_graphs, unrelaxed_dir=unrelaxed_dir)


def evaluate_mean_predictor(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
) -> EvaluationMetrics:
    """Baseline predictor outputting empirical mean of training set targets."""
    mean_val = float(np.mean(train_targets)) if len(train_targets) > 0 else 0.0
    preds = np.full(len(test_targets), fill_value=mean_val, dtype=np.float64)

    diff = preds - test_targets
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    if len(test_targets) > 1 and np.var(test_targets) > 1e-8:
        r2 = float(r2_score(test_targets, preds))
    else:
        r2 = 0.0

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2, predictions=preds)


def evaluate_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> EvaluationMetrics:
    """Standardized Ridge regression baseline on physical descriptors and host one-hot."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)
    preds = model.predict(x_test_scaled).astype(np.float64)

    diff = preds - y_test
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    if len(y_test) > 1 and np.var(y_test) > 1e-8:
        r2 = float(r2_score(y_test, preds))
    else:
        r2 = 0.0

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2, predictions=preds)


def train_legacy_gcn(
    train_graphs: List[Data],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    seed: int = 42,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> DopantGNN:
    """Train legacy isotropic GCN model with target standardisation and cosine schedule."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_targets = [float(g.y.item()) for g in train_graphs if g.y is not None]
    if train_targets:
        y_mean = float(np.mean(train_targets))
        y_std = float(np.std(train_targets)) if len(train_targets) > 1 and np.std(train_targets) > 1e-6 else 1.0
    else:
        y_mean = 0.0
        y_std = 1.0

    in_feat = train_graphs[0].x.size(-1) if train_graphs and train_graphs[0].x is not None else 1
    model = DopantGNN(in_features=in_feat, hidden_dim=hidden_dim).to(device)
    model.y_mean = y_mean
    model.y_std = y_std

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = torch.nn.SmoothL1Loss()

    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    for _ in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y_std_target = (batch.y.view(-1, 1) - y_mean) / y_std
            loss = criterion(out, y_std_target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def run_lodo_cv(
    samples: List[SampleData],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    seed: int = 42,
    device: Optional[torch.device] = None,
    run_legacy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Dict[str, Dict[str, float]]]:
    """Execute Leave-One-Dopant-Out cross validation across hosts and sites.

    Returns:
        Tuple of (metrics_df, predictions_dict, targets_array, fold_summaries).
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_dopants = sorted(list(set(s.dopant for s in samples if s.dopant != "unknown")))
    if len(unique_dopants) < 2:
        raise ValueError(f"Need at least 2 distinct dopant elements for LODO CV, found {len(unique_dopants)}.")

    models = ["mean_predictor", "ridge_baseline"]
    if run_legacy:
        models.append("legacy_gcn")
    models.append("defect_gnn")

    records: List[Dict[str, Any]] = []
    preds_by_model: Dict[str, List[float]] = {m: [] for m in models}
    all_targets: List[float] = []

    for dopant in unique_dopants:
        train_items = [s for s in samples if s.dopant != dopant]
        test_items = [s for s in samples if s.dopant == dopant]

        if not test_items or not train_items:
            continue

        y_train = np.array([s.y for s in train_items], dtype=np.float64)
        y_test = np.array([s.y for s in test_items], dtype=np.float64)
        n_test = len(test_items)
        all_targets.extend(y_test.tolist())

        # 1. Training Mean Predictor
        eval_mean = evaluate_mean_predictor(y_train, y_test)
        preds_by_model["mean_predictor"].extend(eval_mean.predictions.tolist())
        records.append({
            "cv_type": "LODO",
            "model": "mean_predictor",
            "fold": dopant,
            "n_samples": n_test,
            "mae_ev": eval_mean.mae,
            "rmse_ev": eval_mean.rmse,
            "r2": eval_mean.r2,
        })

        # 2. Ridge Regression Baseline
        x_train = np.array([s.descriptor for s in train_items], dtype=np.float64)
        x_test = np.array([s.descriptor for s in test_items], dtype=np.float64)
        eval_ridge = evaluate_ridge(x_train, y_train, x_test, y_test)
        preds_by_model["ridge_baseline"].extend(eval_ridge.predictions.tolist())
        records.append({
            "cv_type": "LODO",
            "model": "ridge_baseline",
            "fold": dopant,
            "n_samples": n_test,
            "mae_ev": eval_ridge.mae,
            "rmse_ev": eval_ridge.rmse,
            "r2": eval_ridge.r2,
        })

        # 3. Legacy GCN Path
        if run_legacy:
            train_leg = [s.legacy_graph for s in train_items]
            test_leg = [s.legacy_graph for s in test_items]
            legacy_net = train_legacy_gcn(
                train_leg,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            eval_legacy = evaluate(
                legacy_net,
                test_leg,
                y_mean=legacy_net.y_mean,
                y_std=legacy_net.y_std,
                device=device,
            )
            preds_by_model["legacy_gcn"].extend(eval_legacy.predictions.tolist())
            records.append({
                "cv_type": "LODO",
                "model": "legacy_gcn",
                "fold": dopant,
                "n_samples": n_test,
                "mae_ev": eval_legacy.mae,
                "rmse_ev": eval_legacy.rmse,
                "r2": eval_legacy.r2,
            })

        # 4. DefectGNN
        train_def = [s.defect_graph for s in train_items]
        test_def = [s.defect_graph for s in test_items]
        defect_net, _ = train_one_model(
            train_graphs=train_def,
            val_graphs=None,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        eval_defect = evaluate(
            defect_net,
            test_def,
            y_mean=defect_net.y_mean,
            y_std=defect_net.y_std,
            device=device,
        )
        preds_by_model["defect_gnn"].extend(eval_defect.predictions.tolist())
        records.append({
            "cv_type": "LODO",
            "model": "defect_gnn",
            "fold": dopant,
            "n_samples": n_test,
            "mae_ev": eval_defect.mae,
            "rmse_ev": eval_defect.rmse,
            "r2": eval_defect.r2,
        })

    # Compute pooled metrics and fold statistics
    targets_arr = np.array(all_targets, dtype=np.float64)
    pred_arrays: Dict[str, np.ndarray] = {}
    fold_summaries: Dict[str, Dict[str, float]] = {}

    for m in models:
        p_arr = np.array(preds_by_model[m], dtype=np.float64)
        pred_arrays[m] = p_arr

        diff = p_arr - targets_arr
        pooled_mae = float(np.mean(np.abs(diff)))
        pooled_rmse = float(np.sqrt(np.mean(diff**2)))
        pooled_r2 = float(r2_score(targets_arr, p_arr)) if np.var(targets_arr) > 1e-8 else 0.0

        records.append({
            "cv_type": "LODO",
            "model": m,
            "fold": "pooled",
            "n_samples": len(targets_arr),
            "mae_ev": pooled_mae,
            "rmse_ev": pooled_rmse,
            "r2": pooled_r2,
        })

        fold_maes = [r["mae_ev"] for r in records if r["cv_type"] == "LODO" and r["model"] == m and r["fold"] != "pooled"]
        fold_summaries[m] = {
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes)),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "pooled_r2": pooled_r2,
        }

    df_metrics = pd.DataFrame(records)
    return df_metrics, pred_arrays, targets_arr, fold_summaries


def run_loho_cv(
    samples: List[SampleData],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    seed: int = 42,
    device: Optional[torch.device] = None,
    run_legacy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """Execute Leave-One-Host-Out exploratory transfer evaluation across host crystals."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_hosts = sorted(list(set(s.host for s in samples if s.host != "unknown")))
    if len(unique_hosts) < 2:
        logger.warning("Fewer than 2 hosts found. Skipping LOHO evaluation.")
        return pd.DataFrame(), {}, np.array([])

    models = ["mean_predictor", "ridge_baseline"]
    if run_legacy:
        models.append("legacy_gcn")
    models.append("defect_gnn")

    records: List[Dict[str, Any]] = []
    preds_by_model: Dict[str, List[float]] = {m: [] for m in models}
    all_targets: List[float] = []

    for host in unique_hosts:
        train_items = [s for s in samples if s.host != host]
        test_items = [s for s in samples if s.host == host]

        if not test_items or not train_items:
            continue

        y_train = np.array([s.y for s in train_items], dtype=np.float64)
        y_test = np.array([s.y for s in test_items], dtype=np.float64)
        n_test = len(test_items)
        all_targets.extend(y_test.tolist())

        # 1. Mean Predictor
        eval_mean = evaluate_mean_predictor(y_train, y_test)
        preds_by_model["mean_predictor"].extend(eval_mean.predictions.tolist())
        records.append({
            "cv_type": "LOHO_exploratory",
            "model": "mean_predictor",
            "fold": host,
            "n_samples": n_test,
            "mae_ev": eval_mean.mae,
            "rmse_ev": eval_mean.rmse,
            "r2": eval_mean.r2,
        })

        # 2. Ridge Baseline
        x_train = np.array([s.descriptor for s in train_items], dtype=np.float64)
        x_test = np.array([s.descriptor for s in test_items], dtype=np.float64)
        eval_ridge = evaluate_ridge(x_train, y_train, x_test, y_test)
        preds_by_model["ridge_baseline"].extend(eval_ridge.predictions.tolist())
        records.append({
            "cv_type": "LOHO_exploratory",
            "model": "ridge_baseline",
            "fold": host,
            "n_samples": n_test,
            "mae_ev": eval_ridge.mae,
            "rmse_ev": eval_ridge.rmse,
            "r2": eval_ridge.r2,
        })

        # 3. Legacy GCN
        if run_legacy:
            train_leg = [s.legacy_graph for s in train_items]
            test_leg = [s.legacy_graph for s in test_items]
            legacy_net = train_legacy_gcn(
                train_leg,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            eval_legacy = evaluate(
                legacy_net,
                test_leg,
                y_mean=legacy_net.y_mean,
                y_std=legacy_net.y_std,
                device=device,
            )
            preds_by_model["legacy_gcn"].extend(eval_legacy.predictions.tolist())
            records.append({
                "cv_type": "LOHO_exploratory",
                "model": "legacy_gcn",
                "fold": host,
                "n_samples": n_test,
                "mae_ev": eval_legacy.mae,
                "rmse_ev": eval_legacy.rmse,
                "r2": eval_legacy.r2,
            })

        # 4. DefectGNN
        train_def = [s.defect_graph for s in train_items]
        test_def = [s.defect_graph for s in test_items]
        defect_net, _ = train_one_model(
            train_graphs=train_def,
            val_graphs=None,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        eval_defect = evaluate(
            defect_net,
            test_def,
            y_mean=defect_net.y_mean,
            y_std=defect_net.y_std,
            device=device,
        )
        preds_by_model["defect_gnn"].extend(eval_defect.predictions.tolist())
        records.append({
            "cv_type": "LOHO_exploratory",
            "model": "defect_gnn",
            "fold": host,
            "n_samples": n_test,
            "mae_ev": eval_defect.mae,
            "rmse_ev": eval_defect.rmse,
            "r2": eval_defect.r2,
        })

    targets_arr = np.array(all_targets, dtype=np.float64)
    pred_arrays: Dict[str, np.ndarray] = {}

    for m in models:
        p_arr = np.array(preds_by_model[m], dtype=np.float64)
        pred_arrays[m] = p_arr
        diff = p_arr - targets_arr
        pooled_mae = float(np.mean(np.abs(diff)))
        pooled_rmse = float(np.sqrt(np.mean(diff**2)))
        pooled_r2 = float(r2_score(targets_arr, p_arr)) if np.var(targets_arr) > 1e-8 else 0.0

        records.append({
            "cv_type": "LOHO_exploratory",
            "model": m,
            "fold": "pooled",
            "n_samples": len(targets_arr),
            "mae_ev": pooled_mae,
            "rmse_ev": pooled_rmse,
            "r2": pooled_r2,
        })

    df_metrics = pd.DataFrame(records)
    return df_metrics, pred_arrays, targets_arr


def plot_parity_lodo(
    targets: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str = "results/parity_lodo.png",
) -> None:
    """Generate 2x2 parity scatter plots comparing model predictions to reference energies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_list = axes.flatten()

    model_display = [
        ("mean_predictor", "Mean Predictor"),
        ("ridge_baseline", "Ridge Baseline (Descriptors + Host)"),
        ("legacy_gcn", "Legacy GCN (Isotropic)"),
        ("defect_gnn", "DefectGNN (Center Readout)"),
    ]

    min_val = float(np.min(targets)) - 0.5 if len(targets) > 0 else -1.0
    max_val = float(np.max(targets)) + 0.5 if len(targets) > 0 else 1.0

    for ax, (model_key, display_name) in zip(axes_list, model_display):
        preds = predictions.get(model_key, np.array([]))
        if len(preds) == len(targets) and len(targets) > 0:
            diff = preds - targets
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            r2 = float(r2_score(targets, preds)) if np.var(targets) > 1e-8 else 0.0

            ax.scatter(targets, preds, alpha=0.75, edgecolors="k", linewidths=0.5, s=35, c="#1f77b4")
            ax.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="dashed", linewidth=1.5, label="Ideal")

            stats_text = f"MAE: {mae:.3f} eV\nRMSE: {rmse:.3f} eV\nR2: {r2:.3f}"
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )
        else:
            ax.text(0.5, 0.5, "No Predictions Available", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(display_name, fontsize=11, fontweight="medium")
        ax.set_xlabel("Reference Formation Energy (eV)", fontsize=10)
        ax.set_ylabel("Predicted Formation Energy (eV)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle="dotted", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved parity plot to %s", save_path)


def compute_plain_verdict(fold_summaries: Dict[str, Dict[str, float]]) -> str:
    """Evaluate whether DefectGNN outperforms the ridge baseline by more than fold standard deviation."""
    ridge_stats = fold_summaries.get("ridge_baseline", {})
    defect_stats = fold_summaries.get("defect_gnn", {})

    ridge_mae = ridge_stats.get("mean_mae", float("nan"))
    defect_mae = defect_stats.get("mean_mae", float("nan"))
    fold_std = defect_stats.get("std_mae", float("nan"))

    if np.isnan(ridge_mae) or np.isnan(defect_mae) or np.isnan(fold_std):
        return "Verdict: Incomplete benchmark results; comparison between DefectGNN and ridge baseline cannot be determined."

    margin = ridge_mae - defect_mae

    if margin > fold_std:
        return (
            f"Verdict: DefectGNN beats the ridge baseline by {margin:.4f} eV, "
            f"exceeding the fold-to-fold standard deviation of {fold_std:.4f} eV."
        )
    else:
        return (
            f"Verdict: DefectGNN does not beat the ridge baseline by more than the "
            f"fold-to-fold standard deviation (difference: {margin:.4f} eV, fold std: {fold_std:.4f} eV)."
        )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI execution entrypoint for model cross-validation benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Leave-One-Dopant-Out and Leave-One-Host-Out evaluation."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=CONFIG.graph_data_path,
        help="Path to graphs dataset tensor file.",
    )
    parser.add_argument(
        "--unrelaxed-dir",
        type=str,
        default=CONFIG.unrelaxed_dir,
        help="Directory of unrelaxed CIF files.",
    )
    parser.add_argument(
        "--energies-csv",
        type=str,
        default=CONFIG.energies_csv_path,
        help="Path to dopant energies CSV.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=CONFIG.manifest_path,
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/cv_metrics.csv",
        help="Destination path for evaluation metrics CSV.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="results/parity_lodo.png",
        help="Destination path for LODO parity scatter plot.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.train_epochs,
        help="Number of training epochs per fold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.batch_size,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CONFIG.learning_rate,
        help="Learning rate for neural network models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.seed,
        help="Global random seed.",
    )
    parser.add_argument(
        "--skip-legacy-gcn",
        action="store_true",
        help="Skip legacy isotropic GCN baseline execution.",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting model evaluation benchmark using device: %s", device)

    # 1. Load dataset samples
    samples = load_evaluation_dataset(
        data_path=args.data_path,
        unrelaxed_dir=args.unrelaxed_dir,
        csv_path=args.energies_csv,
        manifest_path=args.manifest,
    )

    if not samples:
        print("No valid labeled samples found for evaluation. Skipping cross validation.")
        logger.warning("No valid labeled samples found. Exiting.")
        return

    print(f"Loaded {len(samples)} structures across dopants and hosts for evaluation.")

    # 2. Leave-One-Dopant-Out Cross Validation
    print("Running Leave-One-Dopant-Out (LODO) cross validation...")
    df_lodo, preds_lodo, targets_lodo, lodo_summaries = run_lodo_cv(
        samples=samples,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=CONFIG.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        run_legacy=not args.skip_legacy_gcn,
    )

    # 3. Leave-One-Host-Out Exploratory Cross Validation
    print("Running Leave-One-Host-Out (LOHO - exploratory) transfer evaluation...")
    df_loho, _, _ = run_loho_cv(
        samples=samples,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=CONFIG.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        run_legacy=not args.skip_legacy_gcn,
    )

    # Combine metrics and save CSV
    all_metrics = pd.concat([df_lodo, df_loho], ignore_index=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    all_metrics.to_csv(args.output_csv, index=False)
    print(f"Saved evaluation metrics to {args.output_csv}")

    # 4. Generate Parity Plot
    if len(targets_lodo) > 0:
        plot_parity_lodo(targets_lodo, preds_lodo, save_path=args.output_plot)
        print(f"Saved parity plot to {args.output_plot}")

    # 5. Print summary table
    print("\nBenchmark Summary (Pooled Scores):")
    pooled_rows = all_metrics[all_metrics["fold"] == "pooled"]
    for _, row in pooled_rows.iterrows():
        print(
            f"  [{row['cv_type']}] {row['model']:<18} | "
            f"MAE: {row['mae_ev']:.4f} eV | RMSE: {row['rmse_ev']:.4f} eV | R2: {row['r2']:.4f}"
        )

    # 6. Final one-line verdict
    verdict = compute_plain_verdict(lodo_summaries)
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
