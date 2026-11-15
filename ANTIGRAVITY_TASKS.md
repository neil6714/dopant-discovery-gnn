# Tasks for Antigravity (run everything on the Colab T4)

Unzip `dopant_gnn_fixes_overlay.zip` over the repo root so the files land in place. Then work
through the steps below in order. Do not delete any existing data or code. Stop and report if a
step fails instead of working around it.

## What the overlay changes (read before running)

- `generate_defects.py`: fixes a site-index bug. The old code reused a unit-cell index on the
  supercell, so every SiC `2a_1` structure was the `2b_0` site again.
- `clean_data.py` (new): archives legacy rows and duplicate-site data into `archive/legacy_v1/`.
- `formation_energy.py`, `relax_references.py`: no legacy-column fallback, Mn reference
  re-relaxation, `max_force` stored for references.
- `src/utils/descriptors.py` (new), `structures_to_graphs.py`: graph-level descriptor vector
  `data.desc` including mu_dopant.
- `train_gnn.py`: `DefectGNN` options `use_embedding`, `global_dim`, `batch_norm`, `dropout`.
- `baselines.py`, `residual.py` (new): ridge/GP baselines and a GP + residual GNN.
- `evaluate.py`: new rows ridge_rich, gp_iso, gp_ard, defect_gnn_desc, residual_gnn.
- `diagnostics.py` (new): embedding coverage, ablation, relaxed ceiling, identity ceiling.
- `active_learning.py`: dopant-level (cluster) bootstrap, descriptor GNN as default ensemble,
  pool benchmark that scores on held-out dopants, GP surrogate option.
- `config.py`: 8 extra non-transition-metal dopants, uncertainty cutoff 0.25 eV.

## Steps

1. `pytest tests/ -q` (expect all to pass).
2. `python -m src.data_pipeline.clean_data` and read the dry-run output, then
   `python -m src.data_pipeline.clean_data --apply`.
3. `python -m src.data_pipeline.extract_base_crystals` (downloads references for the new dopants).
4. `python -m src.data_pipeline.relax_references --only ""` is not needed. Run
   `python -m src.data_pipeline.relax_references` for the new elements, then
   `python -m src.data_pipeline.relax_references --rerelax-unconverged`.
5. `python -m src.data_pipeline.generate_defects`
6. `python -m src.data_pipeline.calculate_energies --skip-done`
7. `python -m src.data_pipeline.formation_energy --check`. Report min/max per host and any
   non-converged rows. Expect Mn to be included now.
8. `python -m src.data_pipeline.structures_to_graphs`
9. `python -m src.models.evaluate` then `python -m src.models.diagnostics`
10. `python -m src.models.active_learning --benchmark --surrogate gp`
    `python -m src.models.active_learning --benchmark --descriptor-gnn`
11. `python autonomous_lab.py --dry-run`, then `python autonomous_lab.py`.
12. Update README Section 6 with only numbers from the new `results/*.csv`. Replace the old
    tables, keep the correction notice, and write the missing Section 6.4 comparing models.
    If a GP or ridge model beats the GNN variants, say so plainly.
13. Commit as "site fix, descriptor models, rerun on corrected data" and push.
