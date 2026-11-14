"""Calculate defect formation energies for substitutional dopants in the cation-rich limit.

Formula:
    E_f = E_doped_total - E_host_supercell_total + mu_X - mu_D

Where:
    E_f: Formation energy of the neutral substitutional defect in eV (fixed cell).
    E_doped_total: Total energy of the relaxed doped 3x3x3 supercell in eV.
    E_host_supercell_total: Total energy of the pristine 3x3x3 supercell relaxed with fixed cell (ions only) in eV.
    mu_X: Chemical potential of the substituted host cation X in eV.
    mu_D: Chemical potential of the dopant D in eV.

Sign Convention:
    Positive E_f represents an endothermic substitution requiring energy input to incorporate
    the dopant atom into the host crystal lattice relative to the elemental reservoir.
    Lower or negative E_f values indicate thermodynamically favorable incorporation.

Thermodynamic Limit:
    Calculations are evaluated in the cation-rich (metal-rich) limit where the host cation chemical
    potential mu_X equals the energy per atom of the relaxed elemental bulk phase (mu_X = mu_X_bulk).
    The dopant chemical potential mu_D likewise equals the per-atom energy of its relaxed elemental
    reference phase (mu_D = mu_D_bulk).
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pymatgen.core import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import CONFIG
from src.data_pipeline.calculate_energies import relax_structure, write_csv_atomically
from src.utils.log import get_logger

logger = get_logger("formation_energy")


def relax_host_supercell(
    host_name: str,
    raw_dir: str = "data/raw",
    relaxed_hosts_dir: str = "data/relaxed/hosts",
    references_csv: str = "data/references.csv",
    relaxer: Optional[Any] = None,
    fmax: Optional[float] = None,
    steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Relax pristine 3x3x3 host supercell (ions only, fixed cell) and store in references.csv.

    Args:
        host_name: Host name (SiC or GaN).
        raw_dir: Directory containing raw base crystal CIFs.
        relaxed_hosts_dir: Directory containing pre-relaxed host CIFs.
        references_csv: Path to references CSV file.
        relaxer: Optional StructOptimizer instance.
        fmax: Force threshold in eV/Angstrom.
        steps: Maximum relaxation steps.

    Returns:
        Dictionary containing relaxation record for the pristine supercell.
    """
    if fmax is None:
        fmax = CONFIG.relax_fmax
    if steps is None:
        steps = CONFIG.relax_steps

    # Load host structure
    relaxed_cif = os.path.join(relaxed_hosts_dir, f"{host_name}.cif")
    if os.path.exists(relaxed_cif):
        cif_path = relaxed_cif
    else:
        cif_path = os.path.join(raw_dir, f"{host_name}_base.cif")
        if not os.path.exists(cif_path):
            cif_path = CONFIG.hosts[host_name].cif_path

    logger.info("Constructing pristine 3x3x3 supercell for %s from %s...", host_name, cif_path)
    base_struct = Structure.from_file(cif_path)
    supercell = base_struct.copy()
    supercell.make_supercell([3, 3, 3])

    # Fixed-cell relaxation (ions only)
    res = relax_structure(
        structure=supercell,
        relaxer=relaxer,
        fmax=fmax,
        steps=steps,
        relax_cell=False,
    )

    entry_name = f"{host_name}_supercell"
    source_id = CONFIG.hosts[host_name].material_id
    record = {
        "name": entry_name,
        "e_per_atom_ev": res.energy_per_atom,
        "n_atoms": len(res.final_structure),
        "converged": res.converged,
        "source_material_id": source_id,
    }

    # Atomically update references.csv
    if os.path.exists(references_csv):
        try:
            df_ref = pd.read_csv(references_csv)
            df_ref = pd.concat([df_ref, pd.DataFrame([record])], ignore_index=True)
            df_ref = df_ref.drop_duplicates(subset=["name"], keep="last")
        except Exception:
            df_ref = pd.DataFrame([record])
    else:
        df_ref = pd.DataFrame([record])

    write_csv_atomically(df_ref, references_csv)
    logger.info("Saved pristine supercell reference for %s to %s", host_name, references_csv)
    return record


def add_formation_energies(
    energies_csv: str = "data/dopant_energies.csv",
    references_csv: str = "data/references.csv",
    manifest_csv: str = "data/manifest.csv",
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate formation energies in the cation-rich limit and append columns to energy records.

    Adds columns:
        e_form_ev: Neutral defect formation energy in eV.
        converged_all: Boolean indicating all underlying relaxations converged.
        ood_flag: Boolean indicating out-of-distribution candidate status.
        host, cation, dopant, e_host_supercell_total_ev, mu_cation_ev, mu_dopant_ev: Raw terms.

    Args:
        energies_csv: Path to dopant energies CSV.
        references_csv: Path to reference energies CSV.
        manifest_csv: Path to defect manifest CSV.
        output_csv: Path to save updated CSV (defaults to overwriting energies_csv atomically).

    Returns:
        Updated DataFrame with computed formation energies and raw terms.
    """
    if not os.path.exists(energies_csv):
        raise FileNotFoundError(f"Dopant energies CSV not found at {energies_csv}")
    if not os.path.exists(references_csv):
        raise FileNotFoundError(f"References CSV not found at {references_csv}")

    df_energies = pd.read_csv(energies_csv)
    df_refs = pd.read_csv(references_csv)

    manifest_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(manifest_csv):
        try:
            df_manifest = pd.read_csv(manifest_csv)
            for _, row in df_manifest.iterrows():
                fname = str(row.get("filename", "")).strip()
                if fname:
                    manifest_map[fname] = row.to_dict()
        except Exception as exc:
            logger.warning("Could not parse manifest at %s: %s", manifest_csv, exc)

    # Reference lookup tables
    ref_energies: Dict[str, float] = {}
    ref_converged: Dict[str, bool] = {}
    ref_natoms: Dict[str, int] = {}

    for _, row in df_refs.iterrows():
        name = str(row["name"]).strip()
        ref_energies[name] = float(row["e_per_atom_ev"])
        ref_converged[name] = bool(row["converged"])
        ref_natoms[name] = int(row.get("n_atoms", 1))

    # Compute formation energy row by row
    records = []
    for _, row in df_energies.iterrows():
        fname = str(row["filename"]).strip()
        raw_e = row.get("e_total_ev")
        if pd.isna(raw_e):
            raw_e = row.get("total_energy", np.nan)
        e_doped_total = float(raw_e)
        doped_converged = bool(row.get("converged", True))

        # Metadata from manifest or filename
        meta = manifest_map.get(fname, {})
        host_name = meta.get("host")
        dopant = meta.get("dopant")
        ood_flag = bool(meta.get("ood_flag", False))

        if not host_name or not dopant:
            # Parse from filename pattern: <Host>_doped_<Dopant>_<Site>.cif
            parts = fname.replace(".cif", "").split("_doped_")
            if len(parts) == 2:
                host_name = parts[0]
                rest = parts[1].split("_")
                dopant = rest[0]
            else:
                host_name = "SiC" if "SiC" in fname else ("GaN" if "GaN" in fname else "unknown")
                dopant = "unknown"

        if dopant == "U":
            ood_flag = True

        # Host cation
        host_def = CONFIG.hosts.get(host_name)
        cation = host_def.cation_symbol if host_def else ("Si" if host_name == "SiC" else "Ga")

        default_n = 216 if host_name == "SiC" else 108
        raw_n = row.get("n_atoms")
        n_atoms_sc = int(raw_n) if pd.notna(raw_n) else default_n

        # 1. Host supercell total energy
        host_sc_key = f"{host_name}_supercell"
        if host_sc_key in ref_energies:
            e_host_sc_per_atom = ref_energies[host_sc_key]
            n_atoms_sc = ref_natoms.get(host_sc_key, n_atoms_sc)
            e_host_supercell_total = e_host_sc_per_atom * n_atoms_sc
            host_sc_conv = ref_converged.get(host_sc_key, True)
        elif host_name in ref_energies:
            e_host_sc_per_atom = ref_energies[host_name]
            e_host_supercell_total = e_host_sc_per_atom * n_atoms_sc
            host_sc_conv = ref_converged.get(host_name, True)
        else:
            e_host_supercell_total = np.nan
            host_sc_conv = False

        # 2. Chemical potentials mu_X (cation) and mu_D (dopant)
        mu_cation = ref_energies.get(cation, np.nan)
        cation_conv = ref_converged.get(cation, False)

        mu_dopant = ref_energies.get(dopant, np.nan)
        dopant_conv = ref_converged.get(dopant, False)

        # 3. Calculate formation energy: E_f = E_doped - E_host + mu_X - mu_D
        if np.isnan(e_doped_total) or np.isnan(e_host_supercell_total) or np.isnan(mu_cation) or np.isnan(mu_dopant):
            e_form = np.nan
            converged_all = False
        else:
            e_form = float(e_doped_total - e_host_supercell_total + mu_cation - mu_dopant)
            converged_all = bool(doped_converged and host_sc_conv and cation_conv and dopant_conv)

        row_dict = row.to_dict()
        row_dict["host"] = host_name
        row_dict["cation"] = cation
        row_dict["dopant"] = dopant
        row_dict["e_host_supercell_total_ev"] = e_host_supercell_total
        row_dict["mu_cation_ev"] = mu_cation
        row_dict["mu_dopant_ev"] = mu_dopant
        row_dict["e_form_ev"] = e_form
        row_dict["converged_all"] = converged_all
        row_dict["ood_flag"] = ood_flag

        records.append(row_dict)

    df_out = pd.DataFrame(records)

    target_path = output_csv if output_csv is not None else energies_csv
    write_csv_atomically(df_out, target_path)
    logger.info("Updated formation energies in %s (%d records)", target_path, len(df_out))
    return df_out


def check_formation_energies(df: pd.DataFrame) -> None:
    """Print summary statistics per host and issue warnings for out-of-range or unconverged entries.

    Args:
        df: DataFrame containing computed formation energy records.
    """
    if "e_form_ev" not in df.columns:
        logger.warning("No 'e_form_ev' column found in provided DataFrame.")
        return

    hosts = df["host"].unique() if "host" in df.columns else ["all"]

    print("\n" + "=" * 55)
    print(" FORMATION ENERGY SUMMARY (CATION-RICH LIMIT)")
    print("=" * 55)

    for host in hosts:
        df_host = df[df["host"] == host] if host != "all" else df
        ef = df_host["e_form_ev"].dropna()

        if len(ef) == 0:
            print(f"Host {host}: No valid formation energies computed.")
            continue

        min_val = float(ef.min())
        max_val = float(ef.max())
        med_val = float(ef.median())

        print(
            f"Host {host:4s} | Count: {len(ef):2d} | Min: {min_val:7.4f} eV | Max: {max_val:7.4f} eV | Median: {med_val:7.4f} eV"
        )

        # Check expected physical range [-5.0, 15.0] eV
        outliers = df_host[(df_host["e_form_ev"] < -5.0) | (df_host["e_form_ev"] > 15.0)]
        if len(outliers) > 0:
            logger.warning(
                "WARNING: Host %s has %d value(s) outside expected range [-5, +15] eV:\n%s",
                host,
                len(outliers),
                outliers[["filename", "e_form_ev"]].to_string(index=False),
            )

        # Check convergence flags
        if "converged_all" in df_host.columns:
            unconverged = df_host[df_host["converged_all"] == False]  # noqa: E712
            if len(unconverged) > 0:
                logger.warning(
                    "WARNING: Host %s has %d entry/entries with unconverged underlying relaxations.",
                    host,
                    len(unconverged),
                )

    print("=" * 55 + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for calculating formation energies."""
    parser = argparse.ArgumentParser(
        description="Calculate defect formation energies for dopants in the cation-rich limit."
    )
    parser.add_argument(
        "--energies-csv",
        type=str,
        default=CONFIG.energies_csv_path,
        help="Path to dopant energies CSV.",
    )
    parser.add_argument(
        "--references-csv",
        type=str,
        default=CONFIG.references_csv_path,
        help="Path to reference energies CSV.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default=CONFIG.manifest_path,
        help="Path to defect manifest CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional destination path (defaults to updating energies-csv).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print summary statistics and validate formation energy values.",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.energies_csv) or not os.path.exists(args.references_csv):
        print("Required CSV files not found for calculation. Running checks on existing records if available.")
        if os.path.exists(args.energies_csv):
            df = pd.read_csv(args.energies_csv)
            if args.check:
                check_formation_energies(df)
        return

    df_result = add_formation_energies(
        energies_csv=args.energies_csv,
        references_csv=args.references_csv,
        manifest_csv=args.manifest_csv,
        output_csv=args.output_csv,
    )

    if args.check:
        check_formation_energies(df_result)


if __name__ == "__main__":
    main()
