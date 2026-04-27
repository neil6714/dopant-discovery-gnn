# Autonomous Graph Neural Network for Semiconductor Doping

An end-to-end Active Learning pipeline that accelerates traditional Density Functional Theory (DFT) dopant simulations from 3 days to <0.5 seconds using PyTorch Geometric and Universal Machine Learning Potentials.

This project autonomously generates 3D crystal structures, predicts their thermodynamic stability (formation energy) using a custom Graph Neural Network (GNN), and utilizes deep ensemble bootstrapping to identify out-of-distribution materials for targeted quantum simulation.

---

## 🔬 Pipeline Architecture

The system operates in a closed-loop Active Learning cycle broken into three phases:

1. **Automated Crystal Generation (`pymatgen`)**
   * Downloads base structures (e.g., Silicon Carbide) via the Materials Project API.
   * Procedurally generates 54-atom supercells and injects theoretical transition metal dopants.
2. **Graph Neural Network Predictor (`torch_geometric`)**
   * Converts 3D `.cif` files into rotation-invariant mathematical graphs (nodes = atomic numbers, edges = atomic distances).
   * Passes the graphs through multiple Message Passing (GCNConv) layers to learn local chemical environments and predict formation energy.
3. **Autonomous Active Learning Loop (Uncertainty Quantification)**
   * Trains an ensemble of 5 independent GNNs using bootstrapped data subsets.
   * Calculates the variance (Standard Deviation) of the ensemble's predictions on unseen materials.
   * **Decision Gate:** If uncertainty exceeds a set threshold (e.g., >0.05), the system automatically triggers the `CHGNet` Universal ML Potential to simulate the ground-truth physics, updates the dataset, and permanently retrains the model.

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, PyTorch Geometric (GNNs)
* **Materials Informatics:** CHGNet, Pymatgen, Materials Project API
* **Data Engineering:** Pandas, NumPy

## 📂 Repository Structure

```text
dopant-discovery-gnn/
├── src/
│   ├── data_pipeline/
│   │   ├── 1_extract_base_crystals.py
│   │   ├── 2_generate_defects.py
│   │   ├── 3_calculate_energies.py
│   │   └── 4_structures_to_graphs.py
│   └── models/
│       ├── train_gnn.py
│       └── 6_active_learning.py
├── 7_autonomous_lab.py            # Master orchestration script
└── requirements.txt
