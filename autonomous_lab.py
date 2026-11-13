"""Active learning screening loop for defect candidate structures."""
import argparse
import datetime
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import CONFIG
from src.data_pipeline.calculate_energies import evaluate_single_structure, write_csv_atomically
from src.data_pipeline.formation_energy import add_formation_energies
from src.data_pipeline.structures_to_graphs import process_cif_directory
from src.models.active_learning import (
    ActiveLearningLoop,
    _load_graphs_dataset,
    load_ensemble_models,
    predict_with_uncertainty,
)
from src.utils.log import get_logger

logger = get_logger("autonomous_lab")

# Narrow warning suppression specifically to third-party library deprecations
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=FutureWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


def ensure_ensemble(
    graphs_path: str = "data/graph_data/processed_graphs.pt",
    models_dir: str = "models",
    num_models: int = 5,
    epochs: int = 30,
) -> ActiveLearningLoop:
    """Ensure trained ensemble members are ready in memory, loading or training as needed."""
    al_loop = ActiveLearningLoop(initial_data_path=graphs_path, model_save_path=models_dir)

    all_exist = all(
        os.path.exists(os.path.join(models_dir, f"ensemble_model_{i}.pth"))
        or os.path.exists(os.path.join(models_dir, f"ensemble_model_{i}.pt"))
        for i in range(num_models)
    )

    if all_exist:
        try:
            loaded_models = load_ensemble_models(model_dir=models_dir, num_models=num_models)
            if len(loaded_models) == num_models:
                al_loop.models = loaded_models
                logger.info("Loaded %d existing ensemble members from %s", num_models, models_dir)
                return al_loop
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            logger.warning("Failed loading complete ensemble from %s: %s", models_dir, exc)

    # Train ensemble if checkpoints are missing or loading was incomplete
    if not al_loop.training_pool:
        al_loop.training_pool = _load_graphs_dataset(graphs_path)

    if al_loop.training_pool:
        print(f"Training ensemble of {num_models} models on {len(al_loop.training_pool)} structures...")
        al_loop.train_ensemble(num_models=num_models, epochs=epochs)
    else:
        logger.warning("No training graphs found to fit ensemble at %s", graphs_path)

    return al_loop


def generate_hypothesis(
    dopant: str,
    host: str = "SiC",
    site: Optional[str] = None,
    base_cif: Optional[str] = None,
    output_dir: str = "data/candidates",
    manifest_path: str = "data/manifest.csv",
    relaxed_hosts_dir: str = "data/relaxed/hosts",
) -> str:
    """Generate unrelaxed candidate supercell substituting a target dopant at an inequivalent site.

    Supports arbitrary host crystals and symmetry-inequivalent cation positions.

    Args:
        dopant: Element symbol of the dopant to insert.
        host: Host crystal identifier (SiC or GaN).
        site: Optional Wyckoff site identifier (such as 2b_0 or 2a_1).
        base_cif: Optional path to host primitive cell.
        output_dir: Directory where candidate CIF structure is written.
        manifest_path: Path to defect manifest CSV.
        relaxed_hosts_dir: Directory containing pre-relaxed host structures.

    Returns:
        Path to the generated candidate CIF file.
    """
    host_def = CONFIG.hosts.get(host)
    cation_symbol = host_def.cation_symbol if host_def else ("Si" if host == "SiC" else "Ga")
    default_base = host_def.cif_path if host_def else f"data/raw/{host}_base.cif"
    base_cif_path = base_cif or default_base

    os.makedirs(output_dir, exist_ok=True)

    # Check for pre-relaxed host unit cell
    relaxed_cif = os.path.join(relaxed_hosts_dir, f"{host}.cif")
    if os.path.exists(relaxed_cif):
        host_cif = relaxed_cif
    else:
        host_cif = base_cif_path
        if not os.path.exists(host_cif):
            raise FileNotFoundError(f"Host structure CIF not found at {host_cif}")

    base_struct = Structure.from_file(host_cif)

    # Enumerate symmetry-inequivalent sites for target cation
    sga = SpacegroupAnalyzer(base_struct)
    symm = sga.get_symmetrized_structure()

    target_sites: List[Dict[str, Any]] = []
    site_counter = 0

    for group_idx, indices in enumerate(symm.equivalent_indices):
        first_idx = indices[0]
        site_obj = base_struct[first_idx]
        if site_obj.species.elements[0].symbol == cation_symbol:
            wyckoff = symm.wyckoff_symbols[group_idx]
            name = f"{wyckoff}_{site_counter}"
            target_sites.append({
                "base_idx": first_idx,
                "wyckoff": wyckoff,
                "site_name": name,
            })
            site_counter += 1

    if not target_sites:
        raise ValueError(f"No cation sites for element {cation_symbol} found in host {host}")

    selected_site_info = None
    if site:
        for s_info in target_sites:
            if s_info["site_name"] == site or s_info["wyckoff"] == site:
                selected_site_info = s_info
                break

    if selected_site_info is None:
        selected_site_info = target_sites[0]

    # Construct 3x3x3 supercell and substitute dopant
    supercell = base_struct.copy()
    supercell.make_supercell([3, 3, 3])
    supercell.replace(selected_site_info["base_idx"], dopant)

    site_name = selected_site_info["site_name"]
    wyckoff_label = selected_site_info["wyckoff"]
    cif_filename = f"{host}_doped_{dopant}_{site_name}.cif"
    output_path = os.path.join(output_dir, cif_filename)
    supercell.to(fmt="cif", filename=output_path)

    # Update manifest tracking
    dopant_z = int(Element(dopant).Z)
    is_ood = bool(dopant in CONFIG.candidate_dopants or dopant == "U" or dopant_z == 92)

    new_row = {
        "filename": cif_filename,
        "host": host,
        "site": site_name,
        "wyckoff": wyckoff_label,
        "dopant": dopant,
        "dopant_z": dopant_z,
        "status": "candidate",
        "ood_flag": is_ood,
        "path": output_path,
        "paths": output_path,
    }

    if os.path.exists(manifest_path):
        df_manifest = pd.read_csv(manifest_path)
        df_manifest = df_manifest[df_manifest["filename"] != cif_filename]
        df_manifest = pd.concat([df_manifest, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_manifest = pd.DataFrame([new_row])

    write_csv_atomically(df_manifest, manifest_path)
    logger.info("Saved candidate hypothesis to %s and registered in %s", output_path, manifest_path)
    return output_path


def _load_manifest_candidates(
    manifest_path: str = "data/manifest.csv",
    candidates_dir: str = "data/candidates",
) -> List[Dict[str, Any]]:
    """Retrieve active candidate records matching status 'candidate' from manifest."""
    candidates: List[Dict[str, Any]] = []

    if os.path.exists(manifest_path):
        try:
            df_manifest = pd.read_csv(manifest_path)
            cand_rows = df_manifest[df_manifest["status"] == "candidate"]
            for _, row in cand_rows.iterrows():
                fname = str(row["filename"]).strip()
                path = str(row.get("path", os.path.join(candidates_dir, fname)))
                if not os.path.exists(path):
                    path = os.path.join(candidates_dir, fname)

                if os.path.exists(path):
                    candidates.append({
                        "filename": fname,
                        "path": path,
                        "host": str(row.get("host", "SiC")),
                        "dopant": str(row.get("dopant", "unknown")),
                        "site": str(row.get("site", "unknown")),
                        "wyckoff": str(row.get("wyckoff", "unknown")),
                        "ood_flag": bool(row.get("ood_flag", False)),
                    })
        except Exception as exc:
            logger.warning("Could not read manifest at %s: %s", manifest_path, exc)

    # Fallback to scanning candidate directory if manifest has no active entries
    if not candidates and os.path.exists(candidates_dir):
        files = sorted([f for f in os.listdir(candidates_dir) if f.endswith(".cif")])
        for f in files:
            full_path = os.path.join(candidates_dir, f)
            candidates.append({
                "filename": f,
                "path": full_path,
                "host": "SiC" if "SiC" in f else "GaN",
                "dopant": f.split("_doped_")[1].split("_")[0] if "_doped_" in f else "unknown",
                "site": "unknown",
                "wyckoff": "unknown",
                "ood_flag": "U" in f,
            })

    return candidates


def _append_live_al_log(
    log_csv: str,
    record: Dict[str, Any],
) -> None:
    """Atomically append out-of-sample prediction and simulation results to CSV."""
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
    df_row = pd.DataFrame([record])

    if os.path.exists(log_csv):
        df_row.to_csv(log_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_csv, mode="w", header=True, index=False)


def run_autonomous_lab(
    candidates_dir: str = "data/candidates",
    manifest_path: str = "data/manifest.csv",
    csv_path: str = "data/dopant_energies.csv",
    graphs_path: str = "data/graph_data/processed_graphs.pt",
    references_csv: str = "data/references.csv",
    models_dir: str = "models",
    uncertainty_threshold: Optional[float] = None,
    batch_size: int = 1,
    budget: int = 5,
    num_models: int = 5,
    epochs: int = 25,
    dry_run: bool = False,
    log_csv: str = "results/live_al_log.csv",
    processed_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run live active learning screening loop ranking candidates by epistemic uncertainty.

    The uncertainty_threshold is a screening cutoff in eV, not a probability value.
    Out-of-sample prediction errors are recorded to results/live_al_log.csv before retraining.

    Args:
        candidates_dir: Directory containing candidate structure CIF files.
        manifest_path: Path to defect manifest CSV.
        csv_path: Path to dopant energies CSV.
        graphs_path: Path to graph dataset tensor file.
        references_csv: Path to elemental and host reference CSV.
        models_dir: Directory containing ensemble model checkpoints.
        uncertainty_threshold: Screening cutoff in eV for ensemble epistemic uncertainty.
        batch_size: Number of candidate structures simulated per round.
        budget: Maximum total number of candidates to simulate across rounds.
        num_models: Number of models in the ensemble.
        epochs: Training epochs per ensemble member upon retraining.
        dry_run: If True, ranks candidates and reports acquisitions without calling CHGNet.
        log_csv: Path for out-of-sample active learning log CSV.
        processed_dir: Backward-compatible alias for candidates_dir.

    Returns:
        List of acquired and evaluated candidate records.
    """
    candidates_dir = processed_dir or candidates_dir
    if uncertainty_threshold is None:
        uncertainty_threshold = float(CONFIG.uncertainty_threshold)

    print("Active Learning Screening Loop")
    print(f"Settings: batch_size={batch_size}, budget={budget}, threshold={uncertainty_threshold:.4f} eV")

    # 1. Initialize or load ensemble
    al_loop = ensure_ensemble(
        graphs_path=graphs_path,
        models_dir=models_dir,
        num_models=num_models,
        epochs=epochs,
    )

    remaining_budget = budget
    completed_records: List[Dict[str, Any]] = []
    round_num = 1

    while remaining_budget > 0:
        candidates = _load_manifest_candidates(
            manifest_path=manifest_path,
            candidates_dir=candidates_dir,
        )

        if not candidates:
            print("\nNo candidate structures with status 'candidate' remaining. Screening finished.")
            break

        print(f"\nRound {round_num}: Evaluating {len(candidates)} Candidate Structure(s)")

        # 2. Query ensemble predictions and epistemic uncertainties in eV
        candidate_evaluations: List[Dict[str, Any]] = []
        for cand in candidates:
            cif_file = cand["path"]
            try:
                pred_e_form, pred_std = predict_with_uncertainty(
                    cif_file,
                    ensemble_models=al_loop.models,
                )
            except Exception as exc:
                logger.warning("Could not predict for candidate %s: %s", cif_file, exc)
                continue

            cand_entry = dict(cand)
            cand_entry["pred_e_form"] = pred_e_form
            cand_entry["pred_std"] = pred_std
            cand_entry["exceeds_threshold"] = bool(pred_std > uncertainty_threshold)
            candidate_evaluations.append(cand_entry)

        if not candidate_evaluations:
            print("No valid candidate predictions obtained. Exiting loop.")
            break

        # 3. Rank candidates descending by ensemble standard deviation (in eV)
        ranked = sorted(candidate_evaluations, key=lambda x: x["pred_std"], reverse=True)

        print("\nCandidate Rankings (by ensemble std):")
        print(f"  {'Filename':<30} | {'Host':<5} | {'Dopant':<6} | {'Site':<6} | {'Pred E_form (eV)':<16} | {'Ensemble std (eV)':<18} | {'Exceeds Cutoff'}")
        print("  " + "-" * 105)
        for c in ranked:
            print(
                f"  {c['filename']:<30} | {c['host']:<5} | {c['dopant']:<6} | {c['site']:<6} | "
                f"{c['pred_e_form']:>16.4f} | {c['pred_std']:>18.4f} | {str(c['exceeds_threshold']):<14}"
            )

        # 4. Handle dry run mode
        if dry_run:
            eligible_dry = [c for c in ranked if c["exceeds_threshold"]]
            batch_dry = eligible_dry[:min(batch_size, remaining_budget)]
            print(f"\n[DRY RUN] Would simulate {len(batch_dry)} candidate(s) exceeding {uncertainty_threshold:.4f} eV cutoff:")
            for item in batch_dry:
                print(f"  • {item['filename']} (pred E_form: {item['pred_e_form']:.4f} eV, ensemble std: {item['pred_std']:.4f} eV)")
            return ranked

        # 5. Filter candidates exceeding uncertainty threshold
        eligible = [c for c in ranked if c["exceeds_threshold"]]
        if not eligible:
            max_std = ranked[0]["pred_std"] if ranked else 0.0
            print(
                f"\nAll candidate uncertainties are within screening cutoff (max ensemble std: {max_std:.4f} eV <= {uncertainty_threshold:.4f} eV). "
                f"No additional CHGNet relaxation required."
            )
            break

        # 6. Select top candidates up to batch_size and remaining budget
        n_acquire = min(batch_size, remaining_budget, len(eligible))
        selected_batch = eligible[:n_acquire]
        print(f"\nSelected {len(selected_batch)} candidate(s) for CHGNet relaxation (remaining budget: {remaining_budget}):")

        for item in selected_batch:
            fname = item["filename"]
            cif_path = item["path"]
            print(f"\nRunning CHGNet relaxation for candidate: {fname}")

            # Execute relaxation with single-point energy calculation
            sim_res = evaluate_single_structure(cif_path, csv_path=csv_path)
            doped_converged = bool(sim_res.get("converged", True))

            # Calculate defect formation energy in the cation-rich limit
            sim_e_form = float("nan")
            try:
                df_updated = add_formation_energies(
                    energies_csv=csv_path,
                    references_csv=references_csv,
                    manifest_csv=manifest_path,
                )
                match = df_updated[df_updated["filename"] == fname]
                if not match.empty and "e_form_ev" in match.columns:
                    sim_e_form = float(match.iloc[-1]["e_form_ev"])
            except Exception as exc:
                logger.warning("Could not calculate formation energy for %s: %s", fname, exc)

            if np.isnan(sim_e_form):
                sim_e_form = float(sim_res.get("energy_per_atom", 0.0))

            # Compute strictly out-of-sample error before retraining
            abs_error = abs(item["pred_e_form"] - sim_e_form)

            print("CHGNet relaxation complete:")
            print(f"  Simulated formation energy: {sim_e_form:.4f} eV")
            print(f"  Predicted formation energy: {item['pred_e_form']:.4f} eV")
            print(f"  Ensemble std:               {item['pred_std']:.4f} eV")
            print(f"  Out-of-sample error:        {abs_error:.4f} eV")

            # Log to live AL CSV before retraining
            log_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "candidate": fname,
                "host": item["host"],
                "dopant": item["dopant"],
                "site": item["site"],
                "pred_e_form_ev": item["pred_e_form"],
                "ensemble_std_ev": item["pred_std"],
                "sim_e_form_ev": sim_e_form,
                "absolute_error_ev": abs_error,
                "converged": doped_converged,
            }
            _append_live_al_log(log_csv, log_record)
            print(f"Logged out-of-sample error to {log_csv}")

            # Update candidate status in manifest from candidate to relaxed
            if os.path.exists(manifest_path):
                df_man = pd.read_csv(manifest_path)
                mask = df_man["filename"] == fname
                if mask.any():
                    df_man.loc[mask, "status"] = "relaxed"
                    write_csv_atomically(df_man, manifest_path)
                    print(f"Updated status for {fname} in {manifest_path} to relaxed")

            item["sim_e_form"] = sim_e_form
            item["abs_error"] = abs_error
            completed_records.append(item)
            remaining_budget -= 1

        # 7. Retrain ensemble on updated training set
        print("\nRebuilding graph dataset and retraining ensemble...")
        try:
            process_cif_directory(
                unrelaxed_dir=CONFIG.unrelaxed_dir,
                csv_path=csv_path,
                manifest_path=manifest_path,
                output_path=graphs_path,
            )
            al_loop.training_pool = _load_graphs_dataset(graphs_path)
            al_loop.train_ensemble(num_models=num_models, epochs=epochs)
            print("Ensemble retraining complete.")
        except Exception as exc:
            logger.warning("Ensemble retraining encountered an error: %s", exc)

        round_num += 1

    print(f"\nActive learning screening complete. Evaluated {len(completed_records)} candidate(s).")
    return completed_records


def run_autonomous_discovery(
    target_dopant: str,
    host: str = "SiC",
    site: Optional[str] = None,
    base_cif: Optional[str] = None,
    dry_run: bool = False,
    budget: int = 1,
) -> List[Dict[str, Any]]:
    """Generate candidate structure and execute active learning screening round."""
    generate_hypothesis(dopant=target_dopant, host=host, site=site, base_cif=base_cif)
    return run_autonomous_lab(budget=budget, dry_run=dry_run)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI execution entrypoint for live active learning screening."""
    parser = argparse.ArgumentParser(
        description="Active learning screening loop for defect candidates."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rank candidates and print acquisition decisions without calling CHGNet.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="Maximum number of candidate structures to simulate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of candidate structures to simulate per round.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Screening cutoff in eV for ensemble epistemic uncertainty.",
    )
    parser.add_argument(
        "--dopant",
        type=str,
        default=None,
        help="Optional target dopant symbol to generate as hypothesis before screening.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="SiC",
        help="Host crystal name for hypothesis generation.",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Site identifier for hypothesis generation.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=str,
        default="data/candidates",
        help="Directory of candidate CIF structures.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=CONFIG.manifest_path,
        help="Path to manifest CSV file.",
    )
    args = parser.parse_args(argv)

    if args.dopant:
        print(f"Generating hypothesis for dopant {args.dopant} in {args.host}...")
        generate_hypothesis(
            dopant=args.dopant,
            host=args.host,
            site=args.site,
            manifest_path=args.manifest,
        )

    run_autonomous_lab(
        candidates_dir=args.candidates_dir,
        manifest_path=args.manifest,
        uncertainty_threshold=args.threshold,
        batch_size=args.batch_size,
        budget=args.budget,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
