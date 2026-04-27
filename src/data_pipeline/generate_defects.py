import os
from pymatgen.core import Structure
from copy import deepcopy


os.makedirs("data/processed", exist_ok=True)

def generate_doped_structures(base_cif_path, target_element, dopants, output_prefix):
    print(f"\nProcessing {base_cif_path}...")
    
    base_struct = Structure.from_file(base_cif_path)
    
    # Create a 3x3x3 Supercell
    base_struct.make_supercell([3, 3, 3])
    print(f"Created supercell with {len(base_struct)} atoms.")

    # Find the index of the first target atom 
    replace_index = -1
    for i, site in enumerate(base_struct):
        if site.species.elements[0].symbol == target_element:
            replace_index = i
            break
            
    if replace_index == -1:
        print(f"Error: Could not find {target_element} in structure.")
        return

    # Generate a new 3D structure for every dopant 
    for dopant in dopants:
        # Make a fresh copy of the perfect supercell
        doped_struct = deepcopy(base_struct)
        
        # Substitute the target atom with the new transition metal
        doped_struct.replace(replace_index, dopant)
        
        # Save the new defective structure as a .cif file
        file_name = f"data/processed/{output_prefix}_doped_{dopant}.cif"
        doped_struct.to(fmt="cif", filename=file_name)
        print(f" -> Generated {target_element} replaced by {dopant}: {file_name}")

if __name__ == "__main__":
    # A list of 3d transition metals to test as potential dopants
    transition_metals = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
    
    # 1. Dope Silicon Carbide (Replacing one Silicon atom)
    generate_doped_structures(
        base_cif_path="data/raw/SiC_base.cif",
        target_element="Si",
        dopants=transition_metals,
        output_prefix="SiC"
    )
    
    # 2. Dope Gallium Oxide (Replacing one Gallium atom)
    generate_doped_structures(
        base_cif_path="data/raw/Ga2O3_base.cif",
        target_element="Ga",
        dopants=transition_metals,
        output_prefix="Ga2O3"
    )
    
    