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
from src.models.train_gnn import DefectGNN, DopantGNN
from src.utils.log import get_logger
from src.utils.seed import set_seed

logger = get_logger("active_learning")

# Safe globals for PyTorch 2.6+ weights_only compatibility
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Data])
except Exception:
    pass

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
    else:
        state_dict = ckpt
        y_mean = 0.0
        y_std = 1.0
        in_feat = 1
        hidden_dim = 64
        edge_dim = 8

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
        model = DefectGNN(in_features=in_feat, hidden_dim=hidden_dim, edge_dim=edge_dim)
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

        for i in range(num_models):
            member_seed = 42 + i
            set_seed(member_seed)

            # Bootstrapped sampling with replacement using numpy Generator seeded per member
            rng = np.random.default_rng(member_seed)
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
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
                model = DefectGNN(in_features=in_feat, hidden_dim=64, edge_dim=8)
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

        boot_idx = m_rng.choice(n_train, size=n_train, replace=True)
        boot_subset = [train_graphs[b] for b in boot_idx]

        targets = [float(g.y.item()) for g in boot_subset if g.y is not None]
        if targets:
            y_mean = float(np.mean(targets))
            y_std = float(np.std(targets)) if len(targets) > 1 and np.std(targets) > 1e-6 else 1.0
        else:
            y_mean = 0.0
            y_std = 1.0

        if use_defect_gnn:
            model = DefectGNN(in_features=in_feat, hidden_dim=64, edge_dim=8)
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
    """Plot mean remainder MAE across acquisition steps with shaded error spread."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = {"uncertainty": "#1f77b4", "random": "#ff7f0e"}

    for strat in sorted(df_benchmark["strategy"].unique()):
        sub_df = df_benchmark[df_benchmark["strategy"] == strat]
        grouped = sub_df.groupby("n_train")["mae_ev"].agg(["mean", "std"]).sort_index()

        x = grouped.index.to_numpy()
        mean_mae = grouped["mean"].to_numpy()
        std_mae = grouped["std"].fillna(0.0).to_numpy()

        c = colors.get(strat, "#2ca02c")
        label = f"{strat.capitalize()} Acquisition"

        ax.plot(x, mean_mae, marker="o", linewidth=2, color=c, label=label)
        ax.fill_between(x, mean_mae - std_mae, mean_mae + std_mae, color=c, alpha=0.18)

    ax.set_xlabel("Acquired Training Set Size", fontsize=11)
    ax.set_ylabel("Remainder Pool MAE (eV)", fontsize=11)
    ax.set_title("Active Learning Simulated Oracle Benchmark", fontsize=12, fontweight="medium")
    ax.grid(True, linestyle="dotted", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved learning curve plot to %s", save_path)


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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute pool-based active learning simulated oracle benchmark.

    Evaluates greedy acquisition by maximum ensemble uncertainty versus random selection,
    recording untouched remainder pool MAE and calibration Spearman correlation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset is None:
        dataset = _load_graphs_dataset(data_path)

    # Filter out graphs missing labels or marked as out-of-distribution
    valid_dataset = [
        g for g in dataset
        if g.y is not None and not getattr(g, "ood_flag", False)
    ]

    total_pool_size = len(valid_dataset)
    if total_pool_size < seed_size + batch_size:
        logger.warning(
            "Valid dataset size %d too small for seed_size=%d and batch_size=%d",
            total_pool_size,
            seed_size,
            batch_size,
        )
        print(f"Dataset size {total_pool_size} too small to run benchmark.")
        return pd.DataFrame(), pd.DataFrame()

    strategies_to_run = (
        ["uncertainty", "random"] if strategy.lower() in ["both", "all"] else [strategy.lower()]
    )

    benchmark_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []

    print(
        f"Starting active learning pool benchmark: {len(valid_dataset)} samples, "
        f"seed={seed_size}, batch={batch_size}, budget={budget}, repeats={n_repeats}"
    )

    for strat in strategies_to_run:
        print(f"\nAcquisition Strategy: {strat.upper()}")

        for repeat_idx in range(n_repeats):
            run_rng = np.random.default_rng(seed + repeat_idx * 1000)

            # Draw initial seed indices
            all_indices = list(range(total_pool_size))
            seed_indices = list(run_rng.choice(total_pool_size, size=seed_size, replace=False))
            acquired_indices = list(seed_indices)
            candidate_indices = [idx for idx in all_indices if idx not in acquired_indices]

            max_acquired = min(seed_size + budget, total_pool_size - 1)
            step_count = 0

            while True:
                train_graphs = [valid_dataset[i] for i in acquired_indices]
                remainder_graphs = [valid_dataset[i] for i in candidate_indices]

                # Train ensemble on acquired indices
                ensemble = _train_transient_ensemble(
                    train_graphs=train_graphs,
                    num_models=num_models,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed_base=seed + repeat_idx * 1000 + step_count * 100,
                    device=device,
                )

                # Evaluate on untouched remainder
                rem_preds: List[float] = []
                rem_stds: List[float] = []
                rem_targets: List[float] = []

                for rem_g in remainder_graphs:
                    batch = Batch.from_data_list([rem_g]).to(device)
                    member_vals: List[float] = []
                    for model in ensemble:
                        with torch.no_grad():
                            out = model(batch)
                            val_ev = float(out.view(-1).item() * model.y_std + model.y_mean)
                            member_vals.append(val_ev)
                    mean_val = float(np.mean(member_vals))
                    std_val = float(np.std(member_vals, ddof=1)) if len(member_vals) > 1 else 0.0

                    rem_preds.append(mean_val)
                    rem_stds.append(std_val)
                    rem_targets.append(float(rem_g.y.item()))

                preds_arr = np.array(rem_preds, dtype=np.float64)
                targets_arr = np.array(rem_targets, dtype=np.float64)
                stds_arr = np.array(rem_stds, dtype=np.float64)
                abs_errors = np.abs(preds_arr - targets_arr)

                mae = float(np.mean(abs_errors))
                rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))

                benchmark_rows.append({
                    "strategy": strat,
                    "repeat": repeat_idx,
                    "step": step_count,
                    "n_train": len(acquired_indices),
                    "n_remainder": len(candidate_indices),
                    "mae_ev": mae,
                    "rmse_ev": rmse,
                })

                # Compute Spearman rank correlation between ensemble std and absolute error
                if len(stds_arr) > 2 and np.var(stds_arr) > 1e-8 and np.var(abs_errors) > 1e-8:
                    spearman_res = stats.spearmanr(stds_arr, abs_errors)
                    spearman_corr = float(spearman_res.statistic)
                    p_val = float(spearman_res.pvalue)
                else:
                    spearman_corr = float("nan")
                    p_val = float("nan")

                calibration_rows.append({
                    "strategy": strat,
                    "repeat": repeat_idx,
                    "step": step_count,
                    "n_samples": len(remainder_graphs),
                    "spearman_corr": spearman_corr,
                    "p_value": p_val,
                })

                print(
                    f"  [{strat}] Repeat {repeat_idx + 1}/{n_repeats} | "
                    f"Step {step_count}: N_train={len(acquired_indices)}, "
                    f"Remainder MAE={mae:.4f} eV, Spearman rho={spearman_corr:.3f}"
                )

                # Check termination
                if len(acquired_indices) >= max_acquired or len(candidate_indices) == 0:
                    break

                n_to_acquire = min(batch_size, len(candidate_indices))
                if n_to_acquire <= 0:
                    break

                if strat == "random":
                    selected = list(run_rng.choice(candidate_indices, size=n_to_acquire, replace=False))
                elif strat == "uncertainty":
                    cand_rankings = list(zip(candidate_indices, rem_stds))
                    cand_rankings.sort(key=lambda x: x[1], reverse=True)
                    selected = [item[0] for item in cand_rankings[:n_to_acquire]]
                else:
                    selected = list(run_rng.choice(candidate_indices, size=n_to_acquire, replace=False))

                for sel_idx in selected:
                    candidate_indices.remove(sel_idx)
                    acquired_indices.append(sel_idx)

                step_count += 1

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
