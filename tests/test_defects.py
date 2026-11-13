"""Unit tests for defect site enumeration and symmetry analysis."""
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def test_toy_two_site_cell_inequivalent_sites():
    """Verify that a toy two-site rock-salt binary cell yields exactly one inequivalent cation site."""
    lattice = Lattice.cubic(4.0)
    # Binary crystal with one cation and one anion
    struct = Structure(lattice, ["Si", "C"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    sga = SpacegroupAnalyzer(struct)
    symm = sga.get_symmetrized_structure()

    si_inequivalent_indices = [
        eq[0] for eq in symm.equivalent_indices if struct[eq[0]].species.elements[0].symbol == "Si"
    ]
    assert len(si_inequivalent_indices) == 1


def test_toy_broken_symmetry_two_cation_sites():
    """Verify that breaking lattice symmetry separates cations into two inequivalent sites."""
    lattice = Lattice.from_parameters(4.0, 5.0, 6.0, 90.0, 90.0, 90.0)
    # Two Si atoms at positions with no inversion or mirror symmetry relating them
    struct = Structure(
        lattice,
        ["Si", "Si", "C"],
        [[0.0, 0.0, 0.0], [0.5, 0.2, 0.1], [0.5, 0.5, 0.5]],
    )

    sga = SpacegroupAnalyzer(struct)
    symm = sga.get_symmetrized_structure()

    si_inequivalent_indices = [
        eq[0] for eq in symm.equivalent_indices if struct[eq[0]].species.elements[0].symbol == "Si"
    ]
    assert len(si_inequivalent_indices) == 2
