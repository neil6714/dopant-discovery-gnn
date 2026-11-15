# legacy_v1: data set aside after the site-index fix

Created by `python -m src.data_pipeline.clean_data --apply`. Nothing here was deleted from
the project, only moved out of the active files.

- `dopant_energies_before_cleanup.csv`: the full CSV exactly as it was.
- `legacy_rows.csv`: 20 rows from the first pipeline version. `total_energy` in those rows is
  a per-atom value, so their `e_form_ev` is not meaningful.
- `duplicate_site_rows.csv`: rows for `*_2a_1` structures. The old generator replaced a
  translated copy of the 2b site instead of the 2a site, so these repeat the 2b energies.
  The structures have to be regenerated and relaxed again.
- `processed_graphs_before_site_fix.pt`: graph file that still contains the duplicate sites.
- `duplicate_site_cifs/`: unrelaxed or relaxed CIFs for the duplicate sites, if present.
