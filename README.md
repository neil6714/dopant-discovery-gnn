# MLIP-guided GNN for dopant screening

A graph neural network pipeline for evaluating transition-metal and actinide dopants in 4H-SiC and GaN with CHGNet universal potential relaxation and energy labels.

## Overview

This project generates doped crystal structures, relaxes and evaluates them with CHGNet, converts the atomic geometries into edge-aware graph representations, and trains a graph neural network (GNN) ensemble. The learning target is the relaxed formation energy per atom. The repository includes an active learning loop that quantifies epistemic prediction uncertainty across a bootstrapped ensemble and executes an autonomous closed-loop retraining cycle when candidate uncertainty exceeds a defined threshold.

The two host materials are:

- 4H-SiC, provided as `SiC_base.cif`
- GaN, provided as `GaN_base.cif`

The defect generator models candidate dopants (including transition metals and heavy elements such as Zr and U) by substituting host atoms in the supercell.

## Pipeline

The workflow consists of three primary stages:

### 1. Structure and Defect Generation

`extract_base_crystals.py` fetches the SiC and GaN host structures from the Materials Project API. `generate_defects.py` builds 3x3x3 supercells (216 atoms for SiC, 108 atoms for GaN) and substitutes single host lattice sites with candidate dopants. Processed structures are written as CIF files under `data/processed/`.

### 2. CHGNet Relaxation, Energy Labeling, and Graph Conversion

- `calculate_energies.py`: Relaxes unrelaxed supercells using `StructOptimizer` from `chgnet.model` (`fmax=0.1 eV/Å`) to resolve steric clashes introduced by large dopant radii. It overwrites `data/processed/*.cif` with the relaxed atomic coordinates, computes `energy_per_atom = total_energy / len(final_structure)` from the final trajectory state, and records the results in `data/dopant_energies.csv` with deduplication (`drop_duplicates(subset=["filename"], keep="last")`). It also exposes `evaluate_single_structure(cif_path)` for on-demand potential relaxation and energy evaluation.
- `structures_to_graphs.py`: Exposes `build_graph_from_cif(cif_path, y_val=None)` and batch-converts verified CIFs into PyTorch Geometric (PyG) graphs.
  - **Node features ($x$):** Normalized atomic numbers ($Z / 100.0$, shape `[num_nodes, 1]`) to prevent gradient instability when evaluating heavy dopants (e.g., Zr, U).
  - **Edge connectivity and attributes (`edge_index`, `edge_attr`):** Derived from radial neighbor cutoffs ($r \le 5.0$ Å) using inverse distance weights ($1.0 / (\text{distance} + 10^{-5})$).
  - **Dataset filtering:** Unvalidated hypothesis structures (`*_HYPOTHESIS.cif`) are excluded from training until ground-truth potential simulation labels are registered.
  - **PyTorch 2.6+ Compatibility:** Safe globals registration and `weights_only=False` handling ensure serialization compatibility.

### 3. Edge-Aware GNN Training and Closed-Loop Active Learning

- `train_gnn.py`: Implements `DopantGNN`, featuring a linear node embedding layer, three distance-weighted `GCNConv` layers consuming edge attributes, global mean pooling, and a 2-layer MLP prediction head. Models are trained using `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) and `SmoothL1Loss()`, saving weights to `models/dopant_gnn.pth`.
- `active_learning.py`: Defines `ActiveLearningLoop`, which trains a 5-model ensemble (`models/ensemble_model_0.pth` ... `models/ensemble_model_4.pth`) using distinct random seeds and bootstrapped sampling with replacement. Exposes `predict_with_uncertainty(cif_or_graph, ensemble_models)` returning mean prediction and ensemble standard deviation.
- `autonomous_lab.py`: Executes the autonomous hypothesis evaluation loop:
  1. Scans candidate hypothesis structures (`data/processed/*_HYPOTHESIS.cif`).
  2. Queries the 5-model ensemble for predicted formation energy and epistemic uncertainty (`pred_std`).
  3. When candidate uncertainty exceeds the threshold (`pred_std > 0.05` or candidate is unvalidated):
     - Executes the CHGNet `StructOptimizer` simulation via `evaluate_single_structure`.
     - Promotes the structure (`*_HYPOTHESIS.cif` -> `*.cif`) and updates `data/dopant_energies.csv`.
     - Rebuilds `data/graph_data/processed_graphs.pt`.
     - Retrains the 5-model ensemble on the updated training pool.
     - Re-evaluates prediction and uncertainty reduction on the validated material.

## Repository Structure

```text
dopant-discovery-gnn/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── autonomous_lab.py
├── requirements.txt
├── data/
│   ├── README.md
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
│   ├── dopant_gnn.pth
│   ├── ensemble_model_0.pth
│   └── ...
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

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables for Materials Project API access:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Set your API key in `.env`:

```text
MP_API_KEY=your_materials_project_key
```

## Usage

Execute pipeline stages sequentially from the repository root:

```bash
# 1. Download base crystal structures (requires MP_API_KEY)
python -m src.data_pipeline.extract_base_crystals

# 2. Generate defect supercells
python -m src.data_pipeline.generate_defects

# 3. Perform CHGNet relaxation and compute per-atom energy
python -m src.data_pipeline.calculate_energies

# 4. Construct graph representations
python -m src.data_pipeline.structures_to_graphs

# 5. Train the base edge-aware DopantGNN model
python -m src.models.train_gnn

# 6. Train the 5-model active learning ensemble
python -m src.models.active_learning

# 7. Run the closed-loop autonomous active learning cycle
python autonomous_lab.py
```

## Active Learning Strategy

The active learning framework implements bootstrapped ensemble sampling:

1. **Ensemble Diversity:** Five independent `DopantGNN` models are instantiated with distinct random seeds (`42 + i`) and trained over bootstrapped subsets sampled with replacement from the validated graph pool.
2. **Epistemic Uncertainty Estimation:** For candidate graphs, the ensemble outputs predictions $\{y_1, \dots, y_5\}$. Epistemic uncertainty is quantified as the sample standard deviation $\sigma = \sqrt{\frac{1}{M-1} \sum_{i=1}^M (y_i - \bar{y})^2}$.
3. **Simulation Query Gate:** If $\sigma > 0.05$ eV/atom or if the candidate is unvalidated, the CHGNet potential simulation (`StructOptimizer`) is triggered.
4. **Closed-Loop Retraining:** Labeled data is appended to the ground-truth dataset, duplicate records are pruned, graph representations are rebuilt, and the ensemble is refitted to reduce model variance on subsequent discovery iterations.

## Current Scope & Limitations

1. **Defect Configurations:** Focuses on single-site substitutional dopants in $3\times3\times3$ supercells. Interstitials, defect complexes, and variable doping concentrations are not currently covered.
2. **Machine-Learned Potential Approximations:** Ground truth labels rely on CHGNet universal potentials; subsequent experimental verification or high-throughput DFT calculations can be integrated as secondary validation stages.
3. **Graph Architecture:** Utilizes edge-weighted isotropic message passing (`GCNConv`). Higher-order directional or equivariant graph architectures (e.g., PaiNN, EGNN, Equiformer) represent natural extensions for capturing three-dimensional symmetry and bond angles.
