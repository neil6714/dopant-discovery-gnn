"""Diagnostics that explain why the GNN does or does not beat the descriptor baselines.

Four checks, each answering one question:

1. embedding coverage: does the held-out dopant have a trained embedding row? (It cannot.)
2. embedding ablation: does removing the element embedding, and adding mu_dopant to the
   readout, change leave-one-dopant-out error for the same architecture?
3. relaxed-geometry ceiling: does the GNN improve when it sees relaxed instead of unrelaxed
   structures? If not, geometry adds little beyond the dopant identity at this data size.
4. identity ceiling: how much variance can be explained by dopant, host and site labels alone?

Results go to results/diagnostics.csv.
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.structures_to_graphs import build_graph_from_cif
from src.models.evaluate import SampleData, load_evaluation_dataset
from src.models.train_gnn import evaluate, train_one_model
from src.utils.descriptors import RICH_FEATURE_NAMES
from src.utils.log import get_logger
from src.utils.seed import set_seed

logger = get_logger("diagnostics")


def _row(check: str, variant: str, preds: np.ndarray, y: np.ndarray, note: str = "") -> Dict[str, Any]:
    diff = preds - y
    return {
        "check": check,
        "variant": variant,
        "n": len(y),
        "mae_ev": float(np.mean(np.abs(diff))),
        "rmse_ev": float(np.sqrt(np.mean(diff**2))),
        "r2": float(r2_score(y, preds)) if np.var(y) > 1e-8 else 0.0,
        "note": note,
    }


def embedding_coverage(samples: List[SampleData]) -> Dict[str, Any]:
    """Fraction of LODO folds whose held-out dopant element never appears in the training nodes."""
    unseen = 0
    dopants = sorted({s.dopant for s in samples})
    for dopant in dopants:
        train_z = set()
        for s in samples:
            if s.dopant != dopant:
                train_z.update(s.defect_graph.z.tolist())
        held_z = {int(s.defect_graph.dopant_z) for s in samples if s.dopant == dopant}
        if not (held_z & train_z):
            unseen += 1
    return {
        "check": "embedding_coverage",
        "variant": "default DefectGNN",
        "n": len(dopants),
        "mae_ev": np.nan,
        "rmse_ev": np.nan,
        "r2": np.nan,
        "note": f"{unseen}/{len(dopants)} LODO folds hold out an element whose embedding row is untrained",
    }


def lodo_gnn(
    samples: List[SampleData],
    graph_of,
    model_kwargs: Dict[str, Any],
    hidden_dim: int,
    epochs: int,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Leave-one-dopant-out predictions of a DefectGNN variant."""
    preds = np.zeros(len(samples))
    dopants = np.array([s.dopant for s in samples])
    for dopant in np.unique(dopants):
        test_idx = np.where(dopants == dopant)[0]
        train_graphs = [graph_of(s) for i, s in enumerate(samples) if dopants[i] != dopant]
        test_graphs = [graph_of(samples[i]) for i in test_idx]
        net, _ = train_one_model(
            train_graphs, None, epochs=epochs, weight_decay=weight_decay, seed=seed,
            hidden_dim=hidden_dim, device=device, model_kwargs=model_kwargs,
        )
        preds[test_idx] = evaluate(net, test_graphs, y_mean=net.y_mean, y_std=net.y_std, device=device).predictions
    return preds


def identity_ceiling(samples: List[SampleData], seed: int) -> Dict[str, Any]:
    """Random-split R2 of ridge on one-hot dopant, host and site labels (optimistic by design)."""
    labels = np.array([[s.dopant, s.host, s.site] for s in samples])
    x = OneHotEncoder(handle_unknown="ignore").fit_transform(labels).toarray()
    y = np.array([s.y for s in samples])
    preds = np.zeros(len(y))
    counts = np.zeros(len(y))
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed).split(x):
        preds[te] += Ridge(alpha=0.1).fit(x[tr], y[tr]).predict(x[te])
        counts[te] += 1
    return _row(
        "identity_ceiling", "ridge on one-hot dopant+host+site, random 5-fold",
        preds / counts, y, "random split, so the same dopant is in train and test: an upper bound",
    )


def run_diagnostics(
    samples: List[SampleData],
    relaxed_dir: Optional[str],
    epochs: int,
    seed: int,
    output_csv: str,
) -> pd.DataFrame:
    """Run all four checks and write the results CSV."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.array([s.y for s in samples])
    rows: List[Dict[str, Any]] = [embedding_coverage(samples)]

    def unrelaxed(s: SampleData):
        return s.defect_graph

    variants = {
        "default (embedding, no global descriptors)": ({}, 64, 1e-4),
        "no embedding": ({"use_embedding": False}, 32, 1e-3),
        "no embedding + mu_dopant in readout": (
            {"use_embedding": False, "global_dim": len(RICH_FEATURE_NAMES), "batch_norm": False}, 32, 1e-3,
        ),
    }
    for name, (kwargs, hidden, wd) in variants.items():
        logger.info("LODO embedding ablation: %s", name)
        preds = lodo_gnn(samples, unrelaxed, kwargs, hidden, epochs, wd, seed, device)
        rows.append(_row("embedding_ablation", name, preds, y))

    if relaxed_dir and os.path.isdir(relaxed_dir):
        logger.info("Building relaxed-geometry graphs from %s", relaxed_dir)
        relaxed_graphs: Dict[str, Any] = {}
        for s in samples:
            path = os.path.join(relaxed_dir, s.filename)
            if os.path.exists(path):
                meta = {"host": s.host, "dopant": s.dopant, "site": s.site}
                relaxed_graphs[s.filename] = build_graph_from_cif(path, y_val=s.y, manifest_meta=meta)
        if len(relaxed_graphs) == len(samples):
            kwargs = {"use_embedding": False, "global_dim": len(RICH_FEATURE_NAMES), "batch_norm": False}
            for label, getter in (
                ("unrelaxed input", unrelaxed),
                ("relaxed input", lambda s: relaxed_graphs[s.filename]),
            ):
                preds = lodo_gnn(samples, getter, kwargs, 32, epochs, 1e-3, seed, device)
                rows.append(_row("relaxed_ceiling", label, preds, y))
        else:
            rows.append({"check": "relaxed_ceiling", "variant": "skipped", "n": 0, "mae_ev": np.nan,
                         "rmse_ev": np.nan, "r2": np.nan,
                         "note": f"relaxed CIFs found for {len(relaxed_graphs)}/{len(samples)} structures"})
    else:
        rows.append({"check": "relaxed_ceiling", "variant": "skipped", "n": 0, "mae_ev": np.nan,
                     "rmse_ev": np.nan, "r2": np.nan, "note": "no relaxed CIF directory available"})

    rows.append(identity_ceiling(samples, seed))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def summarise(df: pd.DataFrame) -> str:
    """Plain-language summary of what the numbers say."""
    lines = []
    cov = df[df.check == "embedding_coverage"]
    if len(cov):
        lines.append(cov.iloc[0]["note"] + ".")
    abl = df[df.check == "embedding_ablation"].set_index("variant")["mae_ev"]
    if len(abl) == 3:
        d = abl.iloc[0] - abl.iloc[2]
        lines.append(
            f"LODO MAE: default {abl.iloc[0]:.3f} eV, no embedding {abl.iloc[1]:.3f} eV, "
            f"no embedding with mu_dopant {abl.iloc[2]:.3f} eV ({'better' if d > 0 else 'worse'} than default by {abs(d):.3f} eV)."
        )
    rel = df[(df.check == "relaxed_ceiling") & (df.variant != "skipped")].set_index("variant")["mae_ev"]
    if len(rel) == 2:
        lines.append(
            f"Relaxed input MAE {rel['relaxed input']:.3f} eV versus unrelaxed {rel['unrelaxed input']:.3f} eV."
        )
    ident = df[df.check == "identity_ceiling"]
    if len(ident):
        lines.append(f"Identity-only random-split R2: {ident.iloc[0]['r2']:.3f}.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Diagnose GNN versus descriptor baselines.")
    parser.add_argument("--data-path", default=CONFIG.graph_data_path)
    parser.add_argument("--unrelaxed-dir", default=CONFIG.unrelaxed_dir)
    parser.add_argument("--relaxed-dir", default=CONFIG.relaxed_dir)
    parser.add_argument("--epochs", type=int, default=CONFIG.train_epochs)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output-csv", default="results/diagnostics.csv")
    args = parser.parse_args(argv)

    samples = load_evaluation_dataset(data_path=args.data_path, unrelaxed_dir=args.unrelaxed_dir)
    if not samples:
        print("No labelled samples found.")
        return
    df = run_diagnostics(samples, args.relaxed_dir, args.epochs, args.seed, args.output_csv)
    print(df.round(3).to_string(index=False))
    print("\n" + summarise(df))


if __name__ == "__main__":
    main()
