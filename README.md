# Dopant formation energy surrogate

A machine-learned screening pipeline for substitutional transition-metal and actinide dopants in 4H-SiC and GaN using CHGNet relaxations and crystal graph neural networks.

## Overview

This project builds a fast screening surrogate for defect formation energies in wide-bandgap semiconductors. In this pipeline, the training label is the neutral substitutional defect formation energy in eV calculated from CHGNet potential relaxations in the cation-rich limit. Relaxations are performed with fixed lattice parameters (ions relaxed at fixed cell volume), which avoids cell deformation artifacts in supercells while resolving local steric strain.

Input structures given to the graph neural network are unrelaxed substituted supercells. Because unrelaxed geometries are available immediately without running atomistic relaxations, the neural network acts as a rapid pre-screening tool to prioritize candidates before running machine-learned potential or density functional theory calculations.

Both symmetry-inequivalent silicon sites in 4H-SiC (Wyckoff positions 2b and 2a) are explicitly generated and distinguished. For GaN, dopants substitute on the single inequivalent gallium site (Wyckoff 2b). The training dopant set spans 25 transition metals across the 3d, 4d, and 5d rows. Uranium is treated strictly as an out-of-distribution candidate and is never included in the training set.

Earlier versions of this repository contained energy scaling inconsistencies and unrelaxed structures in the training label tables. Those legacy files have been moved to `archive/legacy_v0/` with details documented in `archive/legacy_v0/NOTES.md`. The current codebase was developed and tested using Python 3.10.

## Pipeline

The data pipeline runs through seven sequential steps:

1. Base crystal extraction (`src.data_pipeline.extract_base_crystals`): downloads primitive unit cells for 4H-SiC (mp-11714) and GaN (mp-804) alongside elemental reference structures from the Materials Project API, recording material identifiers in `data/raw/references/sources.csv`.
2. Reference relaxation (`src.data_pipeline.relax_references`): optimizes the cell and ionic positions of the host primitive cells and elemental references using CHGNet, saving records to `data/references.csv`.
3. Defect generation (`src.data_pipeline.generate_defects`): builds 3x3x3 supercells from the relaxed host lattices, substitutes the target cation across all symmetry-inequivalent sites using `SpacegroupAnalyzer`, and writes unrelaxed structures to `data/unrelaxed/` while logging metadata to `data/manifest.csv`.
4. Energy relaxation (`src.data_pipeline.calculate_energies`): relaxes defect supercells with fixed cell dimensions using CHGNet, writes relaxed structures to `data/relaxed/`, and appends total and per-atom energies to `data/dopant_energies.csv`.
5. Formation energy calculation (`src.data_pipeline.formation_energy`): evaluates the neutral formation energy in the cation-rich limit ($E_f = E_{\text{doped}} - E_{\text{host\_sc}} + \mu_{\text{cation}} - \mu_{\text{dopant}}$) using the relaxed host supercell energy and elemental chemical potentials, saving results in `data/dopant_energies.csv`.
6. Graph construction (`src.data_pipeline.structures_to_graphs`): extracts defect-centered local clusters within a 6.0 Angstrom radius, constructs edges within 4.5 Angstroms with 8 Gaussian radial basis function attributes, computes 9-dimensional node features, and attaches target formation energies.
7. Model training and evaluation (`src.models.train_gnn` and `src.models.evaluate`): trains `DefectGNN` with target standardisation and cosine annealing, and runs cross-validation against baseline models.

## Usage

Run the pipeline from the repository root:

```bash
# 1. Download base host structures and elemental references (requires MP_API_KEY)
python -m src.data_pipeline.extract_base_crystals

# 2. Relax host primitive cells and elemental references
python -m src.data_pipeline.relax_references

# 3. Generate unrelaxed substituted defect supercells
python -m src.data_pipeline.generate_defects

# 4. Perform CHGNet relaxation on defect supercells
python -m src.data_pipeline.calculate_energies

# 5. Compute formation energies in the cation-rich limit
python -m src.data_pipeline.formation_energy

# 6. Convert unrelaxed crystal structures into graph datasets
python -m src.data_pipeline.structures_to_graphs

# 7. Train DefectGNN model
python -m src.models.train_gnn

# 8. Run cross-validation benchmarks
python -m src.models.evaluate

# 9. Run active learning pool benchmark
python -m src.models.active_learning --benchmark

# 10. Run live screening loop in dry-run mode
python autonomous_lab.py --dry-run
```

## Active Learning

Active learning is implemented in two complementary workflows:

1. Simulated oracle pool benchmark (`src.models.active_learning --benchmark`): evaluates active acquisition on the labeled dataset. Starting from a small initial seed set, it compares greedy acquisition by maximum ensemble standard deviation against random acquisition. After each batch, the ensemble is retrained and evaluated on the untouched remainder pool. Results are saved to `results/al_benchmark.csv`, `results/al_learning_curve.png`, and `results/uncertainty_calibration.csv` (recording the Spearman rank correlation between ensemble standard deviation and prediction error).
2. Live screening loop (`autonomous_lab.py`): evaluates active candidates from `data/candidates/` tracked in `data/manifest.csv`. In each round, candidates are ranked by ensemble standard deviation in eV. Structures exceeding the screening cutoff from configuration are relaxed with CHGNet. Out-of-sample prediction errors are recorded to `results/live_al_log.csv` before retraining the ensemble, and candidate status in the manifest is updated without copying or renaming files.

## Results

to be filled after the first full run

## Repository Structure

```text
dopant-discovery-gnn/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── autonomous_lab.py
├── requirements.txt
├── archive/
│   └── legacy_v0/
│       ├── NOTES.md
│       ├── dopant_energies.csv
│       ├── dopant_gnn.pth
│       ├── processed_graphs.pt
│       └── processed_unrelaxed_site0/
├── data/
│   ├── README.md
│   ├── candidates/
│   ├── dopant_energies.csv
│   ├── manifest.csv
│   ├── references.csv
│   ├── graph_data/
│   │   └── processed_graphs.pt
│   ├── raw/
│   │   ├── GaN_base.cif
│   │   ├── SiC_base.cif
│   │   └── references/
│   ├── relaxed/
│   │   ├── hosts/
│   │   └── references/
│   └── unrelaxed/
├── models/
│   ├── defect_gnn.pth
│   └── ensemble_model_*.pth
├── results/
│   ├── al_benchmark.csv
│   ├── al_learning_curve.png
│   ├── cv_metrics.csv
│   ├── live_al_log.csv
│   ├── parity_lodo.png
│   └── uncertainty_calibration.csv
├── src/
│   ├── config.py
│   ├── data_pipeline/
│   │   ├── calculate_energies.py
│   │   ├── extract_base_crystals.py
│   │   ├── formation_energy.py
│   │   ├── generate_defects.py
│   │   ├── relax_references.py
│   │   └── structures_to_graphs.py
│   ├── models/
│   │   ├── active_learning.py
│   │   ├── evaluate.py
│   │   └── train_gnn.py
│   └── utils/
│       ├── log.py
│       └── seed.py
└── tests/
    ├── test_active_learning.py
    ├── test_autonomous_lab.py
    ├── test_defects.py
    ├── test_evaluate.py
    ├── test_formation_energy.py
    ├── test_graphs.py
    ├── test_manifest.py
    └── test_model_shapes.py
```

## Setup

Create a Python 3.10 environment using Conda:

```bash
conda create -n dopant-gnn python=3.10
conda activate dopant-gnn
```

Or using python venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install pinned dependencies:

```bash
pip install -r requirements.txt
```

Configure Materials Project access by copying `.env.example` to `.env` and setting `MP_API_KEY`:

```text
MP_API_KEY=your_materials_project_api_key
```

## Limitations

1. CHGNet is a pretrained machine-learned interatomic potential and is least reliable for actinides such as uranium, where localized 5f electron behavior and complex oxidation states challenge universal potential representations.
2. Defect formation energies are calculated strictly for the neutral charge state ($q = 0$). Charged defect states, thermodynamic charge transition levels, and band bending corrections are not computed.
3. Spin states and magnetic moments are handled implicitly through CHGNet without explicit validation against spin-polarized density functional theory reference calculations.
4. The pipeline currently models single-site substitutional defects in fixed supercell volumes. Interstitial dopants, vacancy-dopant pairs, and defect complexes are not represented.
5. The dataset consists of 75 substituted configurations across 25 dopants. This is a compact dataset designed for methodology demonstration and cross-validation benchmarking.
6. The host transfer evaluation between 4H-SiC and GaN is exploratory because only two host crystal structures are included in the dataset.
