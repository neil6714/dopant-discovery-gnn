import os
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

def cif_to_graph(cif_path, energy_label):
    # 1. Load structure
    struct = Structure.from_file(cif_path)
    
    # 2. Node Features: Atomic numbers (Z)
    node_features = torch.tensor([[site.specie.Z] for site in struct], dtype=torch.float)
    
    # 3. Edge Index: Find neighbors within 5 Angstroms
    edge_index = []
    edge_attr = []
    all_neighbors = struct.get_all_neighbors(r=5.0)
    
    for i, neighbors in enumerate(all_neighbors):
        for nb in neighbors:
            edge_index.append([i, nb.index])
            edge_attr.append([nb.nn_distance]) # Distance between atoms
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 4. Target Label = energy
    y = torch.tensor([energy_label], dtype=torch.float)
    
    # Combine into a PyG Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)

if __name__ == "__main__":

    df = pd.read_csv("data/dopant_energies.csv")
    graph_data_list = []
    
    print("Converting CIF files to Graphs...")
    for index, row in df.iterrows():
        cif_path = os.path.join("data/processed", row['filename'])
        energy = row['energy_per_atom']
        
        graph = cif_to_graph(cif_path, energy)
        graph_data_list.append(graph)
    
    os.makedirs("data/graph_data", exist_ok=True)
    torch.save(graph_data_list, "data/graph_data/processed_graphs.pt")
    
    print("Saved to data/graph_data/processed_graphs.pt")