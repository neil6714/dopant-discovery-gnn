# Autonomous Graph Neural Network for Semiconductor Doping

An end-to-end active learning pipeline for accelerating semiconductor dopant screening using Graph Neural Networks (GNNs) and universal machine learning potentials.

The system generates doped crystal structures, predicts their formation energies using a custom GNN, and uses ensemble-based uncertainty estimation to identify materials that require further evaluation. This creates an automated feedback loop that reduces the need for expensive first-principles calculations during the initial screening stage.

---

## Overview

Traditional DFT calculations for evaluating doped semiconductor structures can take hours or days for a single material. This project uses machine learning to reduce the computational cost of screening large numbers of candidate structures.

The pipeline combines:

* Automated crystal and defect generation
* Graph-based representation of crystal structures
* GNN-based formation energy prediction
* Ensemble-based uncertainty estimation
* CHGNet-based energy calculations for selected candidates
* Iterative active learning to improve the model

The current pipeline is designed to reduce the prediction time from several days of DFT computation to **less than 0.5 seconds for a trained GNN inference**.

---

## Pipeline

The workflow consists of three main stages.

### 1. Crystal and Defect Generation

Base crystal structures are obtained using the Materials Project API and processed using `pymatgen`.

The pipeline then:

* Downloads base structures such as Silicon Carbide
* Generates 54-atom supercells
* Selects suitable atomic sites
* Introduces theoretical transition-metal dopants
* Saves the resulting structures as `.cif` files

### 2. Graph Neural Network Prediction

Crystal structures are converted into graphs for processing by a GNN.

Each structure is represented using:

* **Nodes:** atoms represented primarily through their atomic numbers
* **Edges:** neighboring atomic interactions based on interatomic distances
* **Graph structure:** local chemical environments within the crystal

The graph is then passed through multiple `GCNConv` message-passing layers. The model learns relationships between local atomic environments and predicts the formation energy of the structure.

### 3. Active Learning and Uncertainty Estimation

To identify structures where the model is uncertain, the pipeline trains an ensemble of five independent GNN models.

Each model is trained using a bootstrapped subset of the available training data.

For an unseen material:

1. All five models generate a formation energy prediction.
2. The mean prediction is used as the model's estimate.
3. The standard deviation across the ensemble is used as an uncertainty measure.
4. If the uncertainty exceeds the predefined threshold, the structure is selected for further evaluation.
5. CHGNet is used to calculate a higher-quality energy estimate.
6. The new result is added to the training dataset.
7. The GNN ensemble is retrained with the expanded dataset.

This creates a closed-loop active learning system that focuses computational resources on structures where the model needs additional information.

---

## Tech Stack

### Deep Learning

* PyTorch
* PyTorch Geometric
* Graph Convolutional Networks (GCNs)

### Materials Informatics

* CHGNet
* Pymatgen
* Materials Project API
* CIF crystal structures

### Data Processing

* NumPy
* Pandas

---

## Repository Structure

```text
dopant-discovery-gnn/
│
├── src/
│   ├── data_pipeline/
│   │   ├── extract_base_crystals.py
│   │   ├── generate_defects.py
│   │   ├── calculate_energies.py
│   │   └── structures_to_graphs.py
│   │
│   └── models/
│       ├── train_gnn.py
│       └── active_learning.py
│
├── autonomous_lab.py             # Main pipeline orchestration
└── requirements.txt
```

---

## How It Works

At a high level, the complete workflow can be summarized as:

```text
Base Crystal Structures
          │
          ▼
   Generate Supercells
          │
          ▼
     Add Dopants
          │
          ▼
   Convert to Graphs
          │
          ▼
    GNN Prediction
          │
          ▼
  Ensemble Uncertainty
          │
     ┌────┴────┐
     │         │
 Low Uncertainty   High Uncertainty
     │         │
     ▼         ▼
 Accept     CHGNet Evaluation
               │
               ▼
        Update Training Data
               │
               ▼
         Retrain GNN Ensemble
```

The goal is to build a model that becomes progressively better at screening doped semiconductor structures as new training data is generated.

---

## Active Learning Strategy

The active learning loop uses ensemble disagreement as a measure of model uncertainty.

For a candidate structure, the ensemble produces predictions:

[
E_1, E_2, ..., E_5
]

The prediction uncertainty is estimated using the standard deviation:

[
\sigma_E = \mathrm{Std}(E_1, E_2, ..., E_5)
]

If:

[
\sigma_E > 0.05
]

the candidate is considered sufficiently uncertain to trigger a CHGNet evaluation.

The resulting energy is then incorporated into the training dataset for the next iteration.

---

## Project Goals

The project is aimed at developing an automated workflow for machine-learning-assisted materials discovery.

The main objectives are:

* Reduce the computational cost of dopant screening
* Automatically explore new crystal configurations
* Identify promising semiconductor dopants
* Quantify model uncertainty rather than relying only on point predictions
* Continuously improve the model through active learning
* Build a scalable pipeline for high-throughput materials discovery

---

## Future Improvements

Potential extensions to the current pipeline include:

* Incorporating additional semiconductor host materials
* Expanding the range of dopant elements and defect configurations
* Replacing the current GCN architecture with more expressive crystal GNNs
* Adding structural relaxation before energy evaluation
* Comparing CHGNet predictions against DFT calculations
* Improving uncertainty calibration
* Adding formation-energy and stability-based candidate ranking
* Scaling the active learning loop to larger candidate spaces

---
