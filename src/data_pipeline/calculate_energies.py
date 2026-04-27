import os
import pandas as pd
from pymatgen.core import Structure
from chgnet.model.model import CHGNet

def calculate_energies():
    print("Loading CHGNet Universal Potential...")
    model = CHGNet.load()
    processed_dir = "data/processed"
    results = []

    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} not found!")
        return

    # Sort files 
    files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".cif")])

    for file in files:
        path = os.path.join(processed_dir, file)
        print(f"Processing {file}...", end=" ")
        
        try:
            struct = Structure.from_file(path)
            prediction = model.predict_structure(struct)
            
            
            if 'e' in prediction:
                total_energy = prediction['e']
            elif 'energy' in prediction:
                total_energy = prediction['energy']
            else:
                print(f"FAILED! Unrecognized AI output: {prediction.keys()}")
                continue
            
            
            if hasattr(total_energy, "item"):
                total_energy = total_energy.item()
                
            energy_per_atom = total_energy / len(struct)
            
            results.append({
                "filename": file, 
                "total_energy": float(total_energy),
                "energy_per_atom": float(energy_per_atom)
            })
            print("Success!")
            
        except Exception as e:
           
            print(f"FAILED! Exact Error: {repr(e)}")

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv("data/dopant_energies.csv", index=False)
        print(f"\nSaved {len(results)} successful calculations to data/dopant_energies.csv")
    else:
        print("\nCritical Failure: No files were successfully processed.")

if __name__ == "__main__":
    calculate_energies()