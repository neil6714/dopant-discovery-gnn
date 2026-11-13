# Legacy v0 Snapshot Notes

These files are preserved for provenance only.

## Findings

1. Energy label scaling inconsistency in CSV
In `dopant_energies.csv`, the nine 3d transition metal rows per host record `total_energy` near -7.5 eV (for SiC) and -6.3 eV (for GaN), which are per-atom magnitudes rather than supercell totals. Their corresponding `energy_per_atom` values are approximately -0.035 eV/atom and -0.058 eV/atom, indicating that these rows were divided by the supercell atom count twice. The U and Zr rows record true supercell totals (approximately -1629 eV for 216 atoms), giving `energy_per_atom` around -7.54 eV/atom. The label column combines two different scales.

2. Structure files and snapshot reproducibility
Every CIF file in `data/processed/` has identical coordinates and cell volume across dopants, showing that they are unrelaxed generator output. In contrast, the CSV entries and `processed_graphs.pt` were built from relaxed calculations, and include `SiC_doped_U` and `SiC_doped_Zr`, which do not exist as validated CIFs on disk. The snapshot cannot be reproduced from files on disk.

3. Model checkpoint weights compatibility
The checkpoint `models/dopant_gnn.pth` contains keys for `conv1`, `conv2`, `conv3`, and `out_layer`, but has no `embedding` or `mlp` keys. Consequently, it does not load into the current `DopantGNN` architecture definition.
