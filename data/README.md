# Data directory

This directory organizes crystal structures, energy records, defect metadata, and graph datasets across all pipeline stages.

## Directory organization

- `raw/`: contains primitive unit cells for host crystals (`SiC_base.cif` for 4H-SiC, `GaN_base.cif` for GaN) and elemental reference structures downloaded from the Materials Project API under `references/`. Provenance details and material identifiers are recorded in `raw/references/sources.csv`.
- `relaxed/`: contains structures relaxed with CHGNet. Pre-relaxed host primitive cells are in `hosts/`, relaxed elemental reference crystals are in `references/`, and relaxed defect supercells are written directly to this directory.
- `unrelaxed/`: contains 75 unrelaxed 3x3x3 defect supercells generated across symmetry-inequivalent sites (2b_0 and 2a_1 in 4H-SiC, 2b_0 in GaN) for 3d, 4d, and 5d dopants. These unrelaxed structures serve as the input geometries for graph representation.
- `candidates/`: contains candidate structures reserved for active learning screening (such as uranium-doped SiC at both silicon sites).
- `graph_data/`: contains the PyTorch Geometric graph dataset tensor file `processed_graphs.pt`.
- `archive/legacy_v0/`: preserves the initial repository files, unrelaxed site 0 structures, and legacy checkpoints for provenance. Detailed notes on earlier energy scale discrepancies and generator snapshots are documented in `archive/legacy_v0/NOTES.md`.

## CSV tracking tables

- `manifest.csv`: the central tracking registry connecting filenames, host crystals, symmetry sites, Wyckoff positions, dopant symbols, atomic numbers, lifecycle status flags (`unrelaxed`, `relaxed`, `candidate`), out-of-distribution indicators, and filesystem paths.
- `references.csv`: stores relaxed per-atom energies, atom counts, convergence flags, and Materials Project identifiers for host primitive cells, pristine supercells, and elemental reference phases. These energies define the elemental chemical potentials ($\mu$).
- `dopant_energies.csv`: stores relaxation outputs from CHGNet, including total energy in eV, energy per atom in eV, step counts, maximum residual force, convergence flags, and calculated substitutional defect formation energies (`e_form_ev`) evaluated in the cation-rich limit.
