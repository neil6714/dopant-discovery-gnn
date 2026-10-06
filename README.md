# Dopant Discovery GNN

A graph neural network pipeline for evaluating transition-metal dopants in 4H-SiC and GaN with CHGNet energy labels.

## Overview

This project generates doped crystal structures, evaluates them with CHGNet, converts the results into graph data, and trains a graph neural network. The target is the CHGNet energy stored for each structure. The repository also includes an ensemble-based active learning workflow that estimates prediction uncertainty for a new candidate.

The two host materials are:

- 4H-SiC, downloaded as `SiC_base.cif`
- GaN, downloaded as `GaN_base.cif`

The defect generator tests Ti, V, Cr, Mn, Fe, Co, Ni, Cu, and Zn by replacing the first matching host atom in each supercell.

## Pipeline

The workflow has three stages.

### 1. Structure and defect generation

`extract_base_crystals.py` downloads the SiC and GaN host structures from the Materials Project. `generate_defects.py` creates a 3x3x3 supercell and replaces one host atom with each transition-metal dopant.

The resulting supercells contain 216 atoms for SiC and 108 atoms for GaN. The structures are written as CIF files under `data/processed/`.

### 2. CHGNet labeling and graph conversion

`calculate_energies.py` evaluates every processed CIF with CHGNet and writes the results to `data/dopant_energies.csv`. `structures_to_graphs.py` reads the CSV and converts each structure into a PyTorch Geometric graph.

The graph node feature is the raw atomic number. Neighbor edges are created for atoms within 5 Angstroms, and the edge distances are stored in the graph data.

### 3. GNN training and active learning

`train_gnn.py` trains a three-layer GCN on the saved graph data and writes the model weights to `models/dopant_gnn.pth`.

`autonomous_lab.py` trains an ensemble of five GNN models, predicts the energy and uncertainty for a new SiC dopant hypothesis, reports metrics on the training pool, and evaluates the hypothesis with CHGNet when the uncertainty is greater than `0.001`. When a simulation is requested, it appends the result to the CSV and graph dataset.

## Repository Structure

```text
dopant-discovery-gnn/
├── .env.example
├── .gitignore
├── README.md
├── autonomous_lab.py
├── data/
│   ├── dopant_energies.csv
│   ├── graph_data/
│   │   └── processed_graphs.pt
│   ├── processed/
│   │   ├── GaN_doped_*.cif
│   │   └── SiC_doped_*.cif
│   └── raw/
│       ├── GaN_base.cif
│       └── SiC_base.cif
├── models/
│   └── dopant_gnn.pth
├── requirements.txt
└── src/
    ├── data_pipeline/
    │   ├── calculate_energies.py
    │   ├── extract_base_crystals.py
    │   ├── generate_defects.py
    │   └── structures_to_graphs.py
    └── models/
        ├── active_learning.py
        └── train_gnn.py
```

## Setup

Create an environment with Conda:

```bash
conda create -n dopant-gnn python=3.10
conda activate dopant-gnn
```

Or use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\\Scripts\\Activate.ps1
```

From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template and set a Materials Project API key:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

```text
MP_API_KEY=your_materials_project_key
```

## Usage

Run these commands from the repository root, in order:

```bash
python src/data_pipeline/extract_base_crystals.py
python src/data_pipeline/generate_defects.py
python src/data_pipeline/calculate_energies.py
python src/data_pipeline/structures_to_graphs.py
python src/models/train_gnn.py
python autonomous_lab.py
```

The first command requires a valid Materials Project key. The final command uses the default SiC host and the dopant `U` unless its source code is changed.

## Active Learning Strategy

The active learning loop trains five independent GNN models. For each model, the training pool is shuffled and the first 80 percent of the pool is used for training.

For a candidate graph, each model produces a CHGNet-energy prediction. The loop returns the mean prediction and the population standard deviation across the five predictions. In `autonomous_lab.py`, CHGNet evaluation is requested when:

```text
uncertainty > 0.001
```

The resulting CHGNet value is appended to `data/dopant_energies.csv` and the graph is appended to `data/graph_data/processed_graphs.pt`.

## Limitations

1. Structures are unrelaxed single-point CHGNet evaluations with one dopant at one site.
2. `energy_per_atom` in the CSV is currently divided by atom count twice because CHGNet already returns per-atom energy. The stored values are therefore about 100x too small. The correct value is in the `total_energy` column. This is a known issue to fix on the next data regeneration.
3. Metrics in `autonomous_lab.py` are computed on the training pool and are not held-out metrics.
4. The retraining step appends new data but does not yet refit the ensemble.
5. The GNN uses raw atomic number as its node feature and ignores edge distances during message passing.

## Future Work

- Fix the CSV energy normalization and regenerate the dataset.
- Add structural relaxation and more than one dopant site per host.
- Use held-out validation and test splits for reported metrics.
- Refit the ensemble after adding new labeled structures.
- Add chemically meaningful node features and edge-distance features.
- Add reproducible experiment configuration and model checkpoints.
