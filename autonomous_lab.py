import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import pandas as pd
from copy import deepcopy
from pymatgen.core import Structure
from chgnet.model.model import CHGNet
from torch_geometric.data import Data

from src.models.active_learning import ActiveLearningLoop
from src.data_pipeline.structures_to_graphs import cif_to_graph

def run_autonomous_discovery(target_dopant, base_cif="data/raw/SiC_base.cif"):
    print(f"\n🚀 STARTING AUTONOMOUS LAB: Investigating {target_dopant}")
    
    # --- STEP 1: Generate the Unseen Hypothesis ---
    print(f"\n[1/5] Generating theoretical structure for {target_dopant}...")
    base_struct = Structure.from_file(base_cif)
    base_struct.make_supercell([3, 3, 3])
    
    replace_index = next(i for i, site in enumerate(base_struct) if site.species.elements[0].symbol == "Si")
    base_struct.replace(replace_index, target_dopant)
    
    # Save the temporary hypothesis
    hypothesis_path = f"data/processed/SiC_doped_{target_dopant}_HYPOTHESIS.cif"
    base_struct.to(fmt="cif", filename=hypothesis_path)
    
    # Convert hypothesis to a Math Graph (with a dummy energy of 0.0 for now)
    hypothesis_graph = cif_to_graph(hypothesis_path, energy_label=0.0)

    # --- STEP 2: Ask the AI (Active Learning) ---
    print("\n[2/5] Consulting GNN Ensemble...")
    loop = ActiveLearningLoop("data/graph_data/processed_graphs.pt", "models/ensemble_gnn.pth")
    loop.train_ensemble(num_models=5, epochs=20) # Quick train for the demo
    
    mean_pred, uncertainty = loop.predict_with_uncertainty(hypothesis_graph)
    print(f" -> AI Predicted Energy: {mean_pred:.4f} eV")
    print(f" -> AI Uncertainty: {uncertainty:.4f}")
    
    # --- STEP 3: The Decision Gate ---
    if uncertainty > 0.001: 
        print(f"\n[3/5] Uncertainty threshold exceeded! AI requesting quantum simulation...")
        
        # --- STEP 4: The Simulator (CHGNet) ---
        print(f" -> Waking up Universal Potential to calculate ground truth...")
        chgnet_model = CHGNet.load()
        prediction = chgnet_model.predict_structure(base_struct)
        
        # Extract the real energy
        real_energy = prediction.get('e', prediction.get('energy'))
        if hasattr(real_energy, "item"): real_energy = real_energy.item()
        real_energy_per_atom = real_energy / len(base_struct)
        
        print(f" -> SIMULATION COMPLETE. True Energy: {real_energy_per_atom:.4f} eV")
        print(f" -> The AI was off by: {abs(mean_pred - real_energy_per_atom):.4f} eV")
        
        # --- STEP 5: Close the Loop (Retrain) ---
        print("\n[5/5] Closing the loop. Appending new knowledge...")
        
        # 1. Update CSV
        df = pd.read_csv("data/dopant_energies.csv")
        new_row = pd.DataFrame([{"filename": f"SiC_doped_{target_dopant}.cif", "total_energy": real_energy, "energy_per_atom": real_energy_per_atom}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("data/dopant_energies.csv", index=False)
        
        # 2. Update Graphs
        real_graph = cif_to_graph(hypothesis_path, energy_label=real_energy_per_atom)
        loop.training_pool.append(real_graph)
        torch.save(loop.training_pool, "data/graph_data/processed_graphs.pt")
        
        print(" -> Dataset updated. The AI is now permanently smarter.")
        
    else:
        print("\n[3/5] AI is highly confident. No simulation needed. Proceeding to next candidate...")

    print("\nAUTONOMOUS DISCOVERY CYCLE COMPLETE")

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")
    target_dopant="U"
    run_autonomous_discovery(target_dopant)