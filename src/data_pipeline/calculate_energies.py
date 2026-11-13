"""Evaluate and relax structures with CHGNet and record energy labels."""
import argparse
from datetime import datetime, timezone
import os
import sys
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional
import numpy as np
import pandas as pd
from pymatgen.core import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.utils.log import get_logger

logger = get_logger("calculate_energies")

OUTPUT_COLUMNS = [
    "filename",
    "n_atoms",
    "e_total_ev",
    "e_per_atom_ev",
    "converged",
    "n_steps",
    "max_force",
    "relax_cell",
    "fmax",
    "chgnet_version",
]


class RelaxationResult(NamedTuple):
    """Container for structure relaxation outputs."""

    final_structure: Structure
    total_energy: float
    energy_per_atom: float
    n_steps: int
    max_force: float
    converged: bool


def get_chgnet_version() -> str:
    """Retrieve installed CHGNet library version."""
    try:
        import chgnet

        return getattr(chgnet, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def write_csv_atomically(df: pd.DataFrame, csv_path: str) -> None:
    """Write DataFrame to CSV atomically using a temporary file and rename.

    Args:
        df: DataFrame to write.
        csv_path: Destination CSV file path.
    """
    csv_dir = os.path.dirname(csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=csv_dir, prefix="energies_", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, csv_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def log_relaxation_failure(
    filename: str,
    exc: Exception,
    log_dir: str = "logs",
    failures_csv: str = "relax_failures.csv",
) -> None:
    """Append a relaxation failure record to logs/relax_failures.csv.

    Args:
        filename: Name of the structure file that failed.
        exc: Exception instance raised during relaxation.
        log_dir: Directory where the failures log is stored.
        failures_csv: Filename for the failures CSV.
    """
    os.makedirs(log_dir, exist_ok=True)
    failures_path = os.path.join(log_dir, failures_csv)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "exception_type": type(exc).__name__,
        "error_message": str(exc),
    }
    df_err = pd.DataFrame([record])
    if os.path.exists(failures_path):
        df_err.to_csv(failures_path, mode="a", header=False, index=False)
    else:
        df_err.to_csv(failures_path, mode="w", header=True, index=False)


def relax_structure(
    structure: Structure,
    relaxer: Optional[Any] = None,
    fmax: float = 0.05,
    steps: int = 1500,
    relax_cell: bool = False,
) -> RelaxationResult:
    """Relax structure with CHGNet and compute energy from single point prediction.

    Args:
        structure: pymatgen Structure to relax.
        relaxer: Optional StructOptimizer instance.
        fmax: Force convergence threshold in eV/Angstrom.
        steps: Maximum relaxation steps.
        relax_cell: Whether to relax lattice vectors.

    Returns:
        RelaxationResult with final structure, total energy, energy per atom,
        step count, maximum residual force, and convergence status.
    """
    if relaxer is None:
        try:
            from chgnet.model.dynamics import StructOptimizer

            relaxer = StructOptimizer()
        except ImportError:
            from chgnet.model import StructOptimizer

            relaxer = StructOptimizer()

    relax_res = relaxer.relax(
        structure,
        fmax=fmax,
        steps=steps,
        relax_cell=relax_cell,
        verbose=False,
    )

    final_structure = relax_res["final_structure"]

    # Trajectory step count
    traj = relax_res.get("trajectory")
    if traj is not None and hasattr(traj, "energies") and traj.energies:
        n_steps = max(0, len(traj.energies) - 1)
    else:
        n_steps = 0

    # Obtain model for single point prediction on relaxed geometry
    chgnet_model = None
    if hasattr(relaxer, "calculator") and hasattr(relaxer.calculator, "model"):
        chgnet_model = relaxer.calculator.model
    if chgnet_model is None:
        try:
            from chgnet.model.model import CHGNet

            chgnet_model = CHGNet.load()
        except ImportError:
            from chgnet.model import CHGNet

            chgnet_model = CHGNet.load()

    # Single point prediction on relaxed structure
    pred = chgnet_model.predict_structure(final_structure)
    if "e" not in pred:
        raise KeyError(
            f"CHGNet prediction dictionary missing expected 'e' key. Found: {list(pred.keys())}"
        )

    # In CHGNet (v0.4.2), 'e' is energy per atom in eV
    raw_e = pred["e"]
    e_per_atom = float(raw_e.item() if hasattr(raw_e, "item") else raw_e)

    # Calculate maximum residual force magnitude across atoms
    if "f" in pred and pred["f"] is not None:
        forces = np.asarray(pred["f"])
        atom_forces = np.linalg.norm(forces, axis=-1)
        max_force = float(np.max(atom_forces))
    elif traj is not None and hasattr(traj, "forces") and traj.forces:
        last_forces = np.asarray(traj.forces[-1])
        atom_forces = np.linalg.norm(last_forces, axis=-1)
        max_force = float(np.max(atom_forces))
    else:
        max_force = 0.0

    converged = bool(max_force <= fmax)
    n_atoms = len(final_structure)
    e_total = e_per_atom * n_atoms

    # Sanity asserts: physical range [-12, 0] eV/atom and consistency between total and per-atom
    assert -12.0 <= e_per_atom <= 0.0, (
        f"Energy per atom ({e_per_atom:.4f} eV) outside expected physical range [-12, 0] eV"
    )
    assert np.isclose(e_total, e_per_atom * n_atoms, atol=1e-5), (
        f"Total energy ({e_total:.6f} eV) does not equal per-atom energy ({e_per_atom:.6f} eV) multiplied by {n_atoms}"
    )

    return RelaxationResult(
        final_structure=final_structure,
        total_energy=e_total,
        energy_per_atom=e_per_atom,
        n_steps=n_steps,
        max_force=max_force,
        converged=converged,
    )


def evaluate_single_structure(
    cif_path: str,
    csv_path: str = "data/dopant_energies.csv",
    relaxer: Optional[Any] = None,
    relaxed_dir: str = "data/relaxed",
    fmax: Optional[float] = None,
    steps: Optional[int] = None,
    relax_cell: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate and relax a single CIF structure, saving relaxed coordinates to data/relaxed/.

    Input CIF files are never overwritten.

    Args:
        cif_path: Path to input structure CIF.
        csv_path: Path to energies CSV file.
        relaxer: Optional StructOptimizer instance.
        relaxed_dir: Destination directory for relaxed structures.
        fmax: Force threshold in eV/Angstrom.
        steps: Maximum relaxation steps.
        relax_cell: Whether to relax lattice parameters.

    Returns:
        Dictionary containing both legacy keys (filename, total_energy, energy_per_atom)
        and new detailed outputs.
    """
    if fmax is None:
        fmax = CONFIG.relax_fmax
    if steps is None:
        steps = CONFIG.relax_steps
    if relax_cell is None:
        relax_cell = CONFIG.relax_cell_doped

    target_cif_path = cif_path
    if not os.path.exists(target_cif_path):
        candidate = os.path.join(CONFIG.processed_dir, os.path.basename(cif_path))
        if os.path.exists(candidate):
            target_cif_path = candidate
        else:
            raise FileNotFoundError(f"Structure CIF not found at {cif_path} or {candidate}")

    structure = Structure.from_file(target_cif_path)
    res = relax_structure(
        structure,
        relaxer=relaxer,
        fmax=fmax,
        steps=steps,
        relax_cell=relax_cell,
    )

    filename = os.path.basename(cif_path)
    os.makedirs(relaxed_dir, exist_ok=True)
    relaxed_path = os.path.join(relaxed_dir, filename)

    # Write relaxed structure to data/relaxed/ without modifying input CIF
    res.final_structure.to(fmt="cif", filename=relaxed_path)

    chgnet_ver = get_chgnet_version()
    n_atoms = len(res.final_structure)

    record = {
        "filename": filename,
        "total_energy": res.total_energy,
        "energy_per_atom": res.energy_per_atom,
        "n_atoms": n_atoms,
        "e_total_ev": res.total_energy,
        "e_per_atom_ev": res.energy_per_atom,
        "converged": res.converged,
        "n_steps": res.n_steps,
        "max_force": res.max_force,
        "relax_cell": relax_cell,
        "fmax": fmax,
        "chgnet_version": chgnet_ver,
        "relaxed_cif_path": relaxed_path,
    }

    if csv_path:
        csv_record = {k: record[k] for k in OUTPUT_COLUMNS if k in record}
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([csv_record])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([csv_record])
        else:
            df = pd.DataFrame([csv_record])

        df = df.drop_duplicates(subset=["filename"], keep="last")
        col_order = [c for c in OUTPUT_COLUMNS if c in df.columns] + [
            c for c in df.columns if c not in OUTPUT_COLUMNS
        ]
        df = df[col_order]
        write_csv_atomically(df, csv_path)

    return record


def calculate_energies(
    manifest_path: str = "data/manifest.csv",
    csv_path: str = "data/dopant_energies.csv",
    relaxed_dir: str = "data/relaxed",
    unrelaxed_dir: str = "data/unrelaxed",
    only: Optional[str] = None,
    limit: Optional[int] = None,
    skip_done: bool = False,
    relax_cell: Optional[bool] = None,
    fmax: Optional[float] = None,
    steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Relax crystal structures listed in manifest and record energies to CSV.

    Args:
        manifest_path: Path to manifest CSV.
        csv_path: Path to energy results CSV.
        relaxed_dir: Destination folder for relaxed CIFs.
        unrelaxed_dir: Fallback directory to search for CIFs if manifest is absent.
        only: Substring filter for filename selection.
        limit: Maximum number of structures to process.
        skip_done: Whether to skip entries already converged in CSV.
        relax_cell: Whether to relax unit cell dimensions.
        fmax: Convergence threshold in eV/Angstrom.
        steps: Maximum relaxation steps.

    Returns:
        List of result records.
    """
    if fmax is None:
        fmax = CONFIG.relax_fmax
    if steps is None:
        steps = CONFIG.relax_steps
    if relax_cell is None:
        relax_cell = CONFIG.relax_cell_doped

    # 1. Determine input structure paths
    candidate_items: List[Dict[str, str]] = []
    if os.path.exists(manifest_path):
        try:
            df_manifest = pd.read_csv(manifest_path)
            for _, row in df_manifest.iterrows():
                fname = str(row.get("filename", "")).strip()
                cpath = str(row.get("path", "")).strip()
                if not fname and cpath:
                    fname = os.path.basename(cpath)
                if not cpath and fname:
                    cpath = os.path.join(unrelaxed_dir, fname)
                if fname:
                    candidate_items.append({"filename": fname, "path": cpath})
        except Exception as exc:
            logger.warning("Failed to read manifest %s: %s", manifest_path, exc)

    if not candidate_items:
        for search_dir in [unrelaxed_dir, "data/candidates"]:
            if os.path.exists(search_dir):
                files = sorted([f for f in os.listdir(search_dir) if f.endswith(".cif")])
                for f in files:
                    candidate_items.append({"filename": f, "path": os.path.join(search_dir, f)})

    # 2. Filter candidate items
    completed_filenames = set()
    if skip_done and os.path.exists(csv_path):
        try:
            df_done = pd.read_csv(csv_path)
            if "filename" in df_done.columns:
                if "converged" in df_done.columns:
                    completed_filenames = set(
                        df_done[df_done["converged"] == True]["filename"].astype(str)  # noqa: E712
                    )
                else:
                    completed_filenames = set(df_done["filename"].astype(str))
        except Exception as exc:
            logger.warning("Could not parse existing CSV for skip_done check: %s", exc)

    filtered_items = []
    for item in candidate_items:
        fname = item["filename"]
        if only and only not in fname:
            continue
        if skip_done and fname in completed_filenames:
            continue
        filtered_items.append(item)

    if limit is not None and limit > 0:
        filtered_items = filtered_items[:limit]

    logger.info("Found %d structure(s) to process.", len(filtered_items))
    if not filtered_items:
        return []

    # 3. Instantiate relaxer once
    try:
        from chgnet.model.dynamics import StructOptimizer

        relaxer = StructOptimizer()
    except ImportError:
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer()

    results = []
    for idx, item in enumerate(filtered_items, start=1):
        fname = item["filename"]
        cpath = item["path"]
        logger.info("[%d/%d] Relaxing %s...", idx, len(filtered_items), fname)
        try:
            record = evaluate_single_structure(
                cif_path=cpath,
                csv_path=csv_path,
                relaxer=relaxer,
                relaxed_dir=relaxed_dir,
                fmax=fmax,
                steps=steps,
                relax_cell=relax_cell,
            )
            results.append(record)
            logger.info("Success: %s (e_per_atom=%.4f eV/atom)", fname, record["e_per_atom_ev"])
        except Exception as exc:
            logger.error("Failed %s: %s", fname, exc)
            log_relaxation_failure(filename=fname, exc=exc, log_dir=CONFIG.logs_dir)

    logger.info("Processed %d structure(s).", len(results))
    return results


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for batch structure relaxation."""
    parser = argparse.ArgumentParser(
        description="Relax crystal structures with CHGNet and compute energy labels."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV containing structures to process.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CONFIG.energies_csv_path,
        help="Path to output dopant energies CSV.",
    )
    parser.add_argument(
        "--relaxed-dir",
        type=str,
        default="data/relaxed",
        help="Destination directory for relaxed CIF structures.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Filter candidates matching this filename substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of structures to process.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip structures already converged in the output CSV.",
    )
    parser.add_argument(
        "--relax-cell",
        action="store_true",
        help="Allow lattice parameters to relax during optimization.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=CONFIG.relax_fmax,
        help="Force convergence threshold in eV/Angstrom.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=CONFIG.relax_steps,
        help="Maximum optimization steps.",
    )
    args = parser.parse_args(argv)

    calculate_energies(
        manifest_path=args.manifest,
        csv_path=args.csv,
        relaxed_dir=args.relaxed_dir,
        only=args.only,
        limit=args.limit,
        skip_done=args.skip_done,
        relax_cell=args.relax_cell,
        fmax=args.fmax,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
