"""Physical descriptors for dopant elements, shared by the graph builder and the baselines.

The same feature vector feeds ridge, the Gaussian process, and the GNN readout, so any
difference between those models comes from the learner and not from the inputs.
"""
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Element

# Host cation that the dopant replaces. Kept here so descriptors do not import the config.
HOST_CATION = {"SiC": "Si", "GaN": "Ga"}
HOSTS = ("SiC", "GaN")

# Raw atomic number is left out on purpose: it is an arbitrary integer and the GNN already
# sees it through its element embedding. Everything below has a physical reading.
RICH_FEATURE_NAMES: List[str] = [
    "electronegativity",
    "atomic_radius",
    "group",
    "period",
    "valence_electrons",
    "d_electrons",
    "mendeleev_number",
    "first_ionization_ev",
    "mu_dopant_ev",
    "delta_electronegativity",
    "delta_radius",
    "delta_valence",
    "host_SiC",
    "host_GaN",
    "second_site",
]

_REF_CACHE: Dict[str, Dict[str, float]] = {}


def valence_electron_count(element: Element) -> int:
    """Count electrons outside the noble gas core from the electronic structure string."""
    total = 0
    for part in str(getattr(element, "electronic_structure", "")).split("."):
        if part.startswith("["):
            continue
        match = re.search(r"(\d+)$", part)
        if match:
            total += int(match.group(1))
    return total if total > 0 else int(element.group or 1)


def d_electron_count(element: Element) -> int:
    """Occupation of the outermost d subshell in the neutral ground state (0 if none)."""
    d_shells = [s for s in element.full_electronic_structure if s[1] == "d"]
    if not d_shells:
        return 0
    outer = max(s[0] for s in d_shells)
    return int(sum(s[2] for s in d_shells if s[0] == outer))


def _first_ionization(element: Element) -> float:
    energies = getattr(element, "ionization_energies", None)
    if energies and energies[0]:
        return float(energies[0])
    return 7.0  # rough mid-table value, only hit for elements without tabulated data


def load_reference_energies(references_csv: str = "data/references.csv") -> Dict[str, float]:
    """Return {element: energy per atom (eV)} from the relaxed elemental references."""
    key = os.path.abspath(references_csv)
    if key not in _REF_CACHE:
        if not os.path.exists(references_csv):
            _REF_CACHE[key] = {}
        else:
            df = pd.read_csv(references_csv)
            _REF_CACHE[key] = {str(n): float(e) for n, e in zip(df["name"], df["e_per_atom_ev"])}
    return _REF_CACHE[key]


def site_is_second(site: Optional[str]) -> float:
    """1.0 for the second inequivalent cation site (name like '2a_1'), else 0.0."""
    match = re.search(r"_(\d+)$", str(site or ""))
    return 1.0 if match and int(match.group(1)) >= 1 else 0.0


def rich_descriptor(
    dopant: str,
    host: str,
    site: Optional[str] = None,
    mu_dopant: Optional[float] = None,
    references_csv: str = "data/references.csv",
) -> np.ndarray:
    """Build the rich descriptor vector for one defect (order given by RICH_FEATURE_NAMES)."""
    el = Element(dopant)
    cation = Element(HOST_CATION.get(host, "Si"))

    if mu_dopant is None:
        mu_dopant = load_reference_energies(references_csv).get(dopant, np.nan)
    if mu_dopant is None or np.isnan(mu_dopant):
        mu_dopant = 0.0  # missing reference: neutral value, flagged by the caller if it matters

    radius = float(el.atomic_radius or 1.5)
    cation_radius = float(cation.atomic_radius or 1.5)
    valence = float(valence_electron_count(el))

    return np.array(
        [
            float(el.X or 1.5),
            radius,
            float(el.group or 3),
            float(el.row),
            valence,
            float(d_electron_count(el)),
            float(el.mendeleev_no),
            _first_ionization(el),
            float(mu_dopant),
            float(el.X or 1.5) - float(cation.X or 1.5),
            radius - cation_radius,
            valence - float(valence_electron_count(cation)),
            1.0 if host == "SiC" else 0.0,
            1.0 if host == "GaN" else 0.0,
            site_is_second(site),
        ],
        dtype=np.float32,
    )


def rich_descriptor_matrix(
    dopants: Iterable[str],
    hosts: Iterable[str],
    sites: Iterable[Optional[str]],
    references_csv: str = "data/references.csv",
) -> np.ndarray:
    """Stack rich descriptors for parallel lists of dopants, hosts and sites."""
    rows = [
        rich_descriptor(d, h, s, references_csv=references_csv)
        for d, h, s in zip(dopants, hosts, sites)
    ]
    return np.stack(rows) if rows else np.zeros((0, len(RICH_FEATURE_NAMES)), dtype=np.float32)
