"""Evaluate and relax processed structures with CHGNet and write energy labels."""
import os
import pandas as pd
from pymatgen.core import Structure

try:
    from chgnet.model import StructOptimizer, CHGNet
except ImportError:
    from chgnet.model.dynamics import StructOptimizer
    from chgnet.model.model import CHGNet


def evaluate_single_structure(cif_path, csv_path="data/dopant_energies.csv", relaxer=None):
    """Relax a single structure using CHGNet StructOptimizer, update CIF, and record energy."""
    if relaxer is None:
        relaxer = StructOptimizer()

    target_cif_path = cif_path
    if not os.path.exists(target_cif_path):
        candidate = os.path.join("data/processed", os.path.basename(cif_path))
        if os.path.exists(candidate):
            target_cif_path = candidate

    structure = Structure.from_file(target_cif_path)
    result = relaxer.relax(structure, fmax=0.1, verbose=False)

    final_structure = result["final_structure"]
    if "trajectory" in result and hasattr(result["trajectory"], "energies"):
        total_energy = result["trajectory"].energies[-1]
    elif "trajectory" in result and isinstance(result["trajectory"], dict) and "energies" in result["trajectory"]:
        total_energy = result["trajectory"]["energies"][-1]
    elif "energy" in result:
        total_energy = result["energy"]
    elif "e" in result:
        total_energy = result["e"]
    else:
        total_energy = result["trajectory"].energies[-1]

    if hasattr(total_energy, "item"):
        total_energy = total_energy.item()
    total_energy = float(total_energy)
    energy_per_atom = total_energy / len(final_structure)

    # Overwrite / save the relaxed .cif in data/processed/
    final_structure.to(fmt="cif", filename=target_cif_path)
    processed_dir = "data/processed"
    processed_path = os.path.join(processed_dir, os.path.basename(cif_path))
    if os.path.exists(processed_dir) and os.path.abspath(target_cif_path) != os.path.abspath(processed_path):
        final_structure.to(fmt="cif", filename=processed_path)

    # Deduplication and updating csv_path
    filename = os.path.basename(cif_path)
    record = {
        "filename": filename,
        "total_energy": total_energy,
        "energy_per_atom": energy_per_atom,
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df = df.drop_duplicates(subset=["filename"], keep="last")
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    return record


def calculate_energies(processed_dir="data/processed", csv_path="data/dopant_energies.csv"):
    """Evaluate and relax all CIF files in processed_dir and write deduplicated energy labels."""
    print("Loading CHGNet Structure Optimizer...")
    relaxer = StructOptimizer()
    results = []

    if not os.path.exists(processed_dir):
        print(f"Error: Directory '{processed_dir}' not found.")
        return results

    # Sort files
    files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".cif")])

    for file in files:
        path = os.path.join(processed_dir, file)
        print(f"Processing {file}...", end=" ")
        try:
            record = evaluate_single_structure(path, csv_path=csv_path, relaxer=relaxer)
            results.append(record)
            print("Success")
        except Exception as e:
            print(f"Failed: {repr(e)}")

    if len(results) > 0:
        print(f"\nSaved {len(results)} calculations to {csv_path}")
    else:
        print("\nError: No files were successfully processed.")

    return results


if __name__ == "__main__":
    calculate_energies()