"""Relax host primitive cells and elemental references with cell optimization."""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from pymatgen.core import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.calculate_energies import relax_structure, write_csv_atomically
from src.utils.log import get_logger

logger = get_logger("relax_references")

# Note: Magnetic elemental references (Fe, Co, Ni, Mn, Cr) are handled
# implicitly by CHGNet, which represents an approximation of spin states.


def load_source_material_ids(sources_csv: Optional[str] = None) -> Dict[str, str]:
    """Load reference material_id mapping from sources.csv if available.

    Args:
        sources_csv: Path to sources CSV file.

    Returns:
        Dictionary mapping element symbol to Materials Project material_id.
    """
    if sources_csv is None:
        sources_csv = os.path.join(CONFIG.raw_dir, "references", "sources.csv")

    source_map: Dict[str, str] = {}
    if os.path.exists(sources_csv):
        try:
            df = pd.read_csv(sources_csv)
            for _, row in df.iterrows():
                el = str(row.get("element", "")).strip()
                mid = str(row.get("material_id", "")).strip()
                if el and mid:
                    source_map[el] = mid
        except Exception as exc:
            logger.warning("Could not parse %s: %s", sources_csv, exc)

    return source_map


def relax_references(
    raw_dir: str = "data/raw",
    relaxed_hosts_dir: str = "data/relaxed/hosts",
    relaxed_refs_dir: str = "data/relaxed/references",
    output_csv: str = "data/references.csv",
    fmax: Optional[float] = None,
    steps: Optional[int] = None,
    force: bool = False,
    only: Optional[str] = None,
    only_unconverged: bool = False,
) -> List[Dict[str, Any]]:
    """Relax host primitive cells and elemental references with lattice optimization.

    Args:
        raw_dir: Base directory containing raw host CIFs and references/.
        relaxed_hosts_dir: Output folder for relaxed host structures.
        relaxed_refs_dir: Output folder for relaxed elemental references.
        output_csv: Destination CSV path for reference energies.
        fmax: Force convergence threshold in eV/Angstrom.
        steps: Maximum optimization steps.
        force: Whether to re-relax references even if already present.
        only: Substring filter to selectively process names.
        only_unconverged: Redo only the entries flagged unconverged in output_csv, starting
            from their last relaxed structure and with twice the step budget.

    Returns:
        List of reference energy records.
    """
    if fmax is None:
        fmax = CONFIG.relax_fmax
    if steps is None:
        steps = CONFIG.relax_steps

    os.makedirs(relaxed_hosts_dir, exist_ok=True)
    os.makedirs(relaxed_refs_dir, exist_ok=True)

    redo_names = None
    if only_unconverged:
        if not os.path.exists(output_csv):
            raise FileNotFoundError(f"{output_csv} not found, nothing to re-relax.")
        existing = pd.read_csv(output_csv)
        redo_names = set(existing.loc[~existing["converged"].astype(bool), "name"].astype(str))
        steps = steps * 2
        logger.info("Re-relaxing %d unconverged entr(ies): %s", len(redo_names), sorted(redo_names))

    source_ids = load_source_material_ids(os.path.join(raw_dir, "references", "sources.csv"))

    # Instantiate StructOptimizer once
    try:
        from chgnet.model.dynamics import StructOptimizer

        relaxer = StructOptimizer()
    except ImportError:
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer()

    records: List[Dict[str, Any]] = []

    # 1. Relax host primitive cells
    for host_name, host_def in CONFIG.hosts.items():
        if only and only not in host_name:
            continue
        if redo_names is not None and host_name not in redo_names:
            continue

        raw_cif = os.path.join(raw_dir, f"{host_name}_base.cif")
        if not os.path.exists(raw_cif):
            raw_cif = host_def.cif_path

        if not os.path.exists(raw_cif):
            logger.warning("Host raw CIF not found at %s. Skipping %s.", raw_cif, host_name)
            continue

        out_cif = os.path.join(relaxed_hosts_dir, f"{host_name}.cif")
        source_id = host_def.material_id

        logger.info("Relaxing primitive cell for host %s from %s...", host_name, raw_cif)
        struct = Structure.from_file(raw_cif)
        prim_struct = struct.get_primitive_structure()

        res = relax_structure(
            structure=prim_struct,
            relaxer=relaxer,
            fmax=fmax,
            steps=steps,
            relax_cell=True,
        )

        # Fatal stop if a host calculation fails to converge
        if not res.converged:
            logger.critical(
                "CRITICAL: Host %s failed to converge! max_force=%.5f eV/A > fmax=%.5f eV/A (steps=%d)",
                host_name,
                res.max_force,
                fmax,
                res.n_steps,
            )
            raise RuntimeError(
                f"Host structure {host_name} did not converge during reference relaxation: "
                f"max_force={res.max_force:.5f} eV/A > fmax={fmax:.5f} eV/A."
            )

        res.final_structure.to(fmt="cif", filename=out_cif)
        logger.info(
            "Host %s converged: e_per_atom=%.4f eV/atom, n_atoms=%d",
            host_name,
            res.energy_per_atom,
            len(res.final_structure),
        )

        records.append(
            {
                "name": host_name,
                "e_per_atom_ev": res.energy_per_atom,
                "n_atoms": len(res.final_structure),
                "converged": res.converged,
                "max_force": res.max_force,
                "source_material_id": source_id,
            }
        )

    # 2. Relax elemental references
    raw_refs_dir = os.path.join(raw_dir, "references")
    if os.path.exists(raw_refs_dir):
        ref_files = sorted([f for f in os.listdir(raw_refs_dir) if f.endswith(".cif")])
        for rfile in ref_files:
            el_name = os.path.splitext(rfile)[0]
            if only and only not in el_name:
                continue
            if redo_names is not None and el_name not in redo_names:
                continue

            raw_cif = os.path.join(raw_refs_dir, rfile)
            out_cif = os.path.join(relaxed_refs_dir, rfile)
            if redo_names is not None and os.path.exists(out_cif):
                raw_cif = out_cif  # continue from where the previous run stopped
            source_id = source_ids.get(el_name, "")

            logger.info("Relaxing elemental reference %s with lattice relaxation...", el_name)
            struct = Structure.from_file(raw_cif)

            res = relax_structure(
                structure=struct,
                relaxer=relaxer,
                fmax=fmax,
                steps=steps,
                relax_cell=True,
            )

            # Loudly report non-converged references
            if not res.converged:
                logger.warning(
                    "WARNING: NON-CONVERGED REFERENCE: %s max_force=%.5f eV/A > fmax=%.5f eV/A (steps=%d)",
                    el_name,
                    res.max_force,
                    fmax,
                    res.n_steps,
                )
            else:
                logger.info(
                    "Reference %s converged: e_per_atom=%.4f eV/atom, n_atoms=%d",
                    el_name,
                    res.energy_per_atom,
                    len(res.final_structure),
                )

            res.final_structure.to(fmt="cif", filename=out_cif)

            records.append(
                {
                    "name": el_name,
                    "e_per_atom_ev": res.energy_per_atom,
                    "n_atoms": len(res.final_structure),
                    "converged": res.converged,
                    "max_force": res.max_force,
                    "source_material_id": source_id,
                }
            )

    # 3. Write data/references.csv atomically
    if records:
        df_new = pd.DataFrame(records)
        if os.path.exists(output_csv) and not force:
            try:
                df_existing = pd.read_csv(output_csv)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["name"], keep="last")
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new.drop_duplicates(subset=["name"], keep="last")

        write_csv_atomically(df_combined, output_csv)
        logger.info("Wrote %d reference records to %s", len(df_combined), output_csv)

    return records


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for relaxing reference crystal structures."""
    parser = argparse.ArgumentParser(
        description="Relax host primitive cells and elemental references with cell relaxation."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=CONFIG.raw_dir,
        help="Path to raw structures directory.",
    )
    parser.add_argument(
        "--relaxed-hosts-dir",
        type=str,
        default="data/relaxed/hosts",
        help="Destination directory for relaxed host structures.",
    )
    parser.add_argument(
        "--relaxed-refs-dir",
        type=str,
        default="data/relaxed/references",
        help="Destination directory for relaxed elemental references.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/references.csv",
        help="Destination path for references energy CSV.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation and overwrite existing reference records.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Selectively process structures containing this substring.",
    )
    parser.add_argument(
        "--rerelax-unconverged",
        action="store_true",
        help="Redo only references flagged unconverged in the CSV, with a doubled step budget.",
    )
    args = parser.parse_args(argv)

    relax_references(
        raw_dir=args.raw_dir,
        relaxed_hosts_dir=args.relaxed_hosts_dir,
        relaxed_refs_dir=args.relaxed_refs_dir,
        output_csv=args.output_csv,
        fmax=args.fmax,
        steps=args.steps,
        force=args.force,
        only=args.only,
        only_unconverged=args.rerelax_unconverged,
    )


if __name__ == "__main__":
    main()
