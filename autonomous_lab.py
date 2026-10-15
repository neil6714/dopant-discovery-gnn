"""Run the autonomous active learning workflow for candidate dopants."""
import glob
import os
import shutil
import sys
import warnings
import pandas as pd
import torch
from pymatgen.core import Structure

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_pipeline.calculate_energies import evaluate_single_structure
from src.data_pipeline.structures_to_graphs import build_graph_from_cif, process_cif_directory
from src.models.active_learning import ActiveLearningLoop, predict_with_uncertainty


def ensure_ensemble(
    graphs_path="data/graph_data/processed_graphs.pt",
    models_dir="models",
    num_models=5,
    epochs=30,
):
    """Ensure a trained 5-model ensemble is ready."""
    al_loop = ActiveLearningLoop(graphs_path, models_dir)
    all_exist = all(
        os.path.exists(os.path.join(models_dir, f"ensemble_model_{i}.pth"))
        for i in range(num_models)
    )

    if all_exist:
        try:
            from src.models.train_gnn import DopantGNN

            loaded_models = []
            for i in range(num_models):
                m = DopantGNN()
                m.load_state_dict(
                    torch.load(os.path.join(models_dir, f"ensemble_model_{i}.pth"), weights_only=False)
                )
                loaded_models.append(m)
            al_loop.models = loaded_models
            return al_loop
        except Exception:
            pass

    # Ensure dataset exists before training
    if not os.path.exists(graphs_path):
        process_cif_directory(output_path=graphs_path)
        al_loop = ActiveLearningLoop(graphs_path, models_dir)

    al_loop.train_ensemble(num_models=num_models, epochs=epochs)
    return al_loop


def generate_hypothesis(target_dopant, base_cif="data/raw/SiC_base.cif", output_dir="data/processed"):
    """Generate an unrelaxed hypothesis structure with a target dopant."""
    print(f"Generating theoretical hypothesis structure for dopant '{target_dopant}'...")
    base_struct = Structure.from_file(base_cif)
    base_struct.make_supercell([3, 3, 3])

    replace_index = next(
        i for i, site in enumerate(base_struct) if site.species.elements[0].symbol == "Si"
    )
    base_struct.replace(replace_index, target_dopant)

    os.makedirs(output_dir, exist_ok=True)
    hypothesis_path = os.path.join(output_dir, f"SiC_doped_{target_dopant}_HYPOTHESIS.cif")
    base_struct.to(fmt="cif", filename=hypothesis_path)
    print(f"Saved hypothesis to {hypothesis_path}")
    return hypothesis_path


def run_autonomous_lab(
    processed_dir="data/processed",
    csv_path="data/dopant_energies.csv",
    graphs_path="data/graph_data/processed_graphs.pt",
    models_dir="models",
    uncertainty_threshold=0.05,
):
    """Run closed-loop active learning over candidate hypothesis structures."""
    print("=" * 65)
    print(" STARTING AUTONOMOUS LAB: HYPOTHESIS EVALUATION LOOP")
    print("=" * 65)

    # 1. Ensure initial ensemble is trained
    al_loop = ensure_ensemble(graphs_path=graphs_path, models_dir=models_dir)

    # 2. Scan data/processed/*_HYPOTHESIS.cif
    pattern = os.path.join(processed_dir, "*_HYPOTHESIS.cif")
    hypothesis_files = sorted(glob.glob(pattern))

    if not hypothesis_files:
        print(f"No hypothesis files found matching '{pattern}'.")
        return

    print(f"\nFound {len(hypothesis_files)} candidate hypothesis structure(s):")
    for hf in hypothesis_files:
        print(f"  • {os.path.basename(hf)}")

    for hyp_path in hypothesis_files:
        hyp_filename = os.path.basename(hyp_path)
        print(f"\n" + "-" * 50)
        print(f"[ACTIVE LEARNING] Investigating Candidate: {hyp_filename}")
        print("-" * 50)

        # 3. Query the 5-model ensemble
        pred_mean, pred_std = predict_with_uncertainty(hyp_path, ensemble_models=al_loop.models)
        print(f" -> AI Predicted Energy:          {pred_mean:.4f} eV/atom")
        print(f" -> Epistemic Uncertainty (std):  {pred_std:.4f} eV/atom")

        # 4. Check uncertainty threshold
        if pred_std > uncertainty_threshold:
            print(f"\n[DECISION GATE] Uncertainty ({pred_std:.4f}) > threshold ({uncertainty_threshold}).")
            print(" -> Requesting quantum simulation (CHGNet StructOptimizer relaxation)...")

            # 4a. Trigger the CHGNet oracle
            oracle_res = evaluate_single_structure(hyp_path, csv_path=csv_path)
            real_energy = oracle_res["total_energy"]
            real_energy_per_atom = oracle_res["energy_per_atom"]
            error_before = abs(pred_mean - real_energy_per_atom)

            print(f" -> SIMULATION COMPLETE:")
            print(f"    Total Energy:    {real_energy:.4f} eV")
            print(f"    Energy per atom: {real_energy_per_atom:.4f} eV/atom")
            print(f"    AI Error Before: {error_before:.4f} eV/atom")

            # 4b. Promote structure from hypothesis to validated dataset
            validated_filename = hyp_filename.replace("_HYPOTHESIS.cif", ".cif")
            validated_path = os.path.join(processed_dir, validated_filename)
            shutil.copyfile(hyp_path, validated_path)
            print(f" -> Promoted {hyp_filename} to validated dataset: {validated_filename}")

            # Record validated structure in dopant_energies.csv
            df = pd.read_csv(csv_path)
            new_record = {
                "filename": validated_filename,
                "total_energy": real_energy,
                "energy_per_atom": real_energy_per_atom,
            }
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df = df.drop_duplicates(subset=["filename"], keep="last")
            df.to_csv(csv_path, index=False)
            print(f" -> Recorded {validated_filename} in {csv_path}")

            # 4c. Rebuild graph dataset
            print(" -> Rebuilding graph dataset...")
            process_cif_directory(
                processed_dir=processed_dir,
                csv_path=csv_path,
                output_path=graphs_path,
            )

            # 4d. Retrain 5-model ensemble on expanded dataset
            print(" -> Retraining 5-model GNN ensemble on expanded dataset...")
            al_loop = ActiveLearningLoop(graphs_path, models_dir)
            al_loop.train_ensemble(num_models=5, epochs=30)

            # 4e. Re-evaluate (new_mean, new_std) on newly validated structure
            new_mean, new_std = predict_with_uncertainty(validated_path, ensemble_models=al_loop.models)
            error_after = abs(new_mean - real_energy_per_atom)

            print("\n" + "=" * 55)
            print(" CLOSED-LOOP ACTIVE LEARNING SUMMARY")
            print("=" * 55)
            print(f" Structure:               {validated_filename}")
            print(f" Ground Truth Energy:     {real_energy_per_atom:.4f} eV/atom")
            print(f" Prediction (Before/After): {pred_mean:.4f} -> {new_mean:.4f} eV/atom")
            print(f" Absolute Error:          {error_before:.4f} -> {error_after:.4f} eV/atom")
            print(f" Epistemic Uncertainty:   {pred_std:.4f} -> {new_std:.4f} eV/atom")
            if pred_std > new_std:
                print(f" Verified Uncertainty Reduction: {(pred_std - new_std):.4f} eV/atom")
            print("=" * 55)

        else:
            print(f" -> AI is confident (std={pred_std:.4f} <= {uncertainty_threshold}).")
            print(" -> No quantum simulation required. Candidate accepted.")

    print("\nAUTONOMOUS LAB RUN COMPLETE.")


def run_autonomous_discovery(target_dopant, base_cif="data/raw/SiC_base.cif"):
    """Workflow entrypoint for discovering a specific candidate dopant."""
    generate_hypothesis(target_dopant, base_cif=base_cif)
    run_autonomous_lab()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_autonomous_lab()