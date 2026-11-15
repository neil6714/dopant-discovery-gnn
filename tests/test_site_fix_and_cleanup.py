"""Regression tests for the supercell site-index bug and the data cleanup script."""
import os
import sys

import pandas as pd
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline import clean_data
from src.data_pipeline.generate_defects import supercell_index_of_site

RAW_SIC = "data/raw/SiC_base.cif"


def test_supercell_index_matches_by_position():
    lat = Lattice.hexagonal(3.0, 5.0)
    prim = Structure(lat, ["Si", "Si", "C"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5], [0, 0, 0.4]])
    sc = prim.copy()
    sc.make_supercell([3, 3, 3])
    for i in range(2):
        j = supercell_index_of_site(sc, prim[i].coords)
        assert sc[j].species.elements[0].symbol == "Si"
        assert sc[j].coords == pytest.approx(prim[i].coords, abs=1e-6)
    # the two Si sites must map to different supercell atoms
    assert supercell_index_of_site(sc, prim[0].coords) != supercell_index_of_site(sc, prim[1].coords)


@pytest.mark.skipif(not os.path.exists(RAW_SIC), reason="raw SiC host not present")
def test_generated_sic_sites_are_actually_different(tmp_path):
    from src.data_pipeline.generate_defects import generate_doped_structures

    generate_doped_structures(RAW_SIC, "Si", ["Ti"], "SiC", output_dir=str(tmp_path), relaxed_hosts_dir="/nonexistent")
    a = Structure.from_file(tmp_path / "SiC_doped_Ti_2b_0.cif")
    b = Structure.from_file(tmp_path / "SiC_doped_Ti_2a_1.cif")
    assert not StructureMatcher(ltol=0.01, stol=0.01, angle_tol=0.1).fit(a, b)


def test_clean_energies_archives_without_deleting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(clean_data, "ARCHIVE", str(tmp_path / "archive"))
    csv = tmp_path / "e.csv"
    pd.DataFrame([
        {"filename": "SiC_doped_Co.cif", "total_energy": -7.5, "energy_per_atom": -0.03, "e_total_ev": None, "e_form_ev": 1600.0},
        {"filename": "SiC_doped_Ti_2b_0.cif", "total_energy": None, "energy_per_atom": None, "e_total_ev": -1627.0, "e_form_ev": -0.26},
        {"filename": "SiC_doped_Ti_2a_1.cif", "total_energy": None, "energy_per_atom": None, "e_total_ev": -1627.0, "e_form_ev": -0.26},
    ]).to_csv(csv, index=False)

    active = clean_data.clean_energies(str(csv), apply=True)
    assert list(active["filename"]) == ["SiC_doped_Ti_2b_0.cif"]
    assert "total_energy" not in active.columns
    arch = tmp_path / "archive"
    assert len(pd.read_csv(arch / "dopant_energies_before_cleanup.csv")) == 3
    assert len(pd.read_csv(arch / "legacy_rows.csv")) == 1
    assert len(pd.read_csv(arch / "duplicate_site_rows.csv")) == 1

    # a second run must not overwrite the archived original with the cleaned file
    clean_data.clean_energies(str(csv), apply=True)
    assert len(pd.read_csv(arch / "dopant_energies_before_cleanup.csv")) == 3


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(clean_data, "ARCHIVE", str(tmp_path / "archive"))
    csv = tmp_path / "e.csv"
    pd.DataFrame([{"filename": "x_2a_1.cif", "e_total_ev": -1.0}]).to_csv(csv, index=False)
    before = csv.read_text()
    clean_data.clean_energies(str(csv), apply=False)
    assert csv.read_text() == before and not (tmp_path / "archive").exists()
