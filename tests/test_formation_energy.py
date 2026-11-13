"""Unit tests for defect formation energy calculation and sign conventions."""
import numpy as np
import pandas as pd

from src.data_pipeline.formation_energy import add_formation_energies


def test_formation_energy_formula_hand_checkable(tmp_path):
    """Verify formation energy calculation against hand-computed values.

    Formula (cation-rich limit):
        E_f = E_doped_total - E_host_supercell_total + mu_cation - mu_dopant

    Hand check:
        E_doped_total = -1620.0 eV
        E_host_supercell_total = -1625.0 eV (216 atoms * -7.523148 eV/atom)
        mu_cation = -5.0 eV/atom
        mu_dopant = -7.0 eV/atom
        E_f = -1620.0 - (-1625.0) + (-5.0) - (-7.0) = 5.0 - 5.0 + 7.0 = 7.0 eV
    """
    energies_csv = tmp_path / "dopant_energies.csv"
    refs_csv = tmp_path / "references.csv"
    manifest_csv = tmp_path / "manifest.csv"

    # Synthetic energies record
    df_energies = pd.DataFrame([
        {
            "filename": "SiC_doped_Ti_2b_0.cif",
            "total_energy": -1620.0,
            "energy_per_atom": -7.5,
            "n_atoms": 216,
            "converged": True,
        }
    ])
    df_energies.to_csv(energies_csv, index=False)

    # Reference energies record
    df_refs = pd.DataFrame([
        {"name": "SiC_supercell", "e_per_atom_ev": -1625.0 / 216.0, "n_atoms": 216, "converged": True},
        {"name": "Si", "e_per_atom_ev": -5.0, "n_atoms": 1, "converged": True},
        {"name": "Ti", "e_per_atom_ev": -7.0, "n_atoms": 1, "converged": True},
    ])
    df_refs.to_csv(refs_csv, index=False)

    # Defect manifest
    df_manifest = pd.DataFrame([
        {
            "filename": "SiC_doped_Ti_2b_0.cif",
            "host": "SiC",
            "site": "2b_0",
            "dopant": "Ti",
            "dopant_z": 22,
            "status": "relaxed",
            "ood_flag": False,
        }
    ])
    df_manifest.to_csv(manifest_csv, index=False)

    df_out = add_formation_energies(
        energies_csv=str(energies_csv),
        references_csv=str(refs_csv),
        manifest_csv=str(manifest_csv),
    )

    computed_e_form = float(df_out.iloc[0]["e_form_ev"])
    expected_e_form = 7.0
    assert np.isclose(computed_e_form, expected_e_form, atol=1e-5)


def test_formation_energy_sign_convention():
    """Verify sign convention where lower/negative values indicate thermodynamic stability."""
    e_doped_stable = -1630.0
    e_doped_unstable = -1610.0
    e_host = -1625.0
    mu_x = -5.0
    mu_d = -5.0

    # Negative E_f implies exothermic substitution relative to reservoirs
    e_f_stable = e_doped_stable - e_host + mu_x - mu_d
    e_f_unstable = e_doped_unstable - e_host + mu_x - mu_d

    assert e_f_stable == -5.0
    assert e_f_unstable == 15.0
    assert e_f_stable < 0.0 < e_f_unstable
