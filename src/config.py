"""Central configuration dataclass for the dopant discovery pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class HostDef:
    """Specification of a host crystal."""

    name: str
    cif_path: str
    cation_symbol: str
    anion_symbol: str
    material_id: str


@dataclass
class Config:
    """Project-wide parameters for data, relaxation, graphs, and models."""

    # Paths
    raw_dir: str = "data/raw"
    unrelaxed_dir: str = "data/unrelaxed"
    relaxed_dir: str = "data/relaxed"
    candidates_dir: str = "data/candidates"
    processed_dir: str = "data/unrelaxed"  # backward compatibility alias
    graph_data_path: str = "data/graph_data/processed_graphs.pt"
    energies_csv_path: str = "data/dopant_energies.csv"
    manifest_path: str = "data/manifest.csv"
    references_csv_path: str = "data/references.csv"
    models_dir: str = "models"
    dopant_gnn_path: str = "models/dopant_gnn.pth"
    logs_dir: str = "logs"
    results_dir: str = "results"
    archive_dir: str = "archive/legacy_v0"

    # Host definitions
    hosts: Dict[str, HostDef] = field(
        default_factory=lambda: {
            "SiC": HostDef(
                name="SiC",
                cif_path="data/raw/SiC_base.cif",
                cation_symbol="Si",
                anion_symbol="C",
                material_id="mp-11714",
            ),
            "GaN": HostDef(
                name="GaN",
                cif_path="data/raw/GaN_base.cif",
                cation_symbol="Ga",
                anion_symbol="N",
                material_id="mp-804",
            ),
        }
    )

    # Dopant lists
    dopants_3d: List[str] = field(
        default_factory=lambda: ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
    )
    dopants_4d: List[str] = field(
        default_factory=lambda: ["Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd"]
    )
    dopants_5d: List[str] = field(
        default_factory=lambda: ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]
    )
    candidate_dopants: List[str] = field(default_factory=lambda: ["U"])

    # Relaxation settings
    relax_fmax: float = 0.05
    relax_steps: int = 1500
    relax_cell_host: bool = True
    relax_cell_doped: bool = False

    # Graph settings
    defect_cluster_radius: float = 6.0
    edge_cutoff: float = 4.5
    num_rbf: int = 8
    rbf_min: float = 0.5
    rbf_max: float = 6.0

    # Training and active learning hyperparameters
    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_epochs: int = 50
    ensemble_epochs: int = 30
    num_ensemble_models: int = 5
    # Screening cutoff in eV on ensemble std. The old 0.05 eV was below the ensemble std of
    # nearly every candidate (about 0.3 to 0.6 eV at this data size), so it never discriminated.
    uncertainty_threshold: float = 0.25
    train_split: float = 0.8

    # Non-transition-metal dopants. The 24 transition metals alone leave the model with one
    # chemistry family, so held-out elements are far from everything seen in training.
    extra_dopants: List[str] = field(default_factory=lambda: ["Be", "Mg", "Ca", "Al", "Ge", "Sn", "Sc", "Y"])
    include_extra_dopants: bool = True

    # Ensemble members use the descriptor-aware DefectGNN (no element embedding, mu_dopant in
    # the readout) whenever the graphs carry data.desc.
    use_descriptor_gnn: bool = True
    descriptor_gnn_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"use_embedding": False, "global_dim": 15, "batch_norm": False, "dropout": 0.1}
    )

    @property
    def training_dopants(self) -> List[str]:
        """Return the dopant list used for training (3d, 4d, 5d and optional extras)."""
        base = self.dopants_3d + self.dopants_4d + self.dopants_5d
        if self.include_extra_dopants:
            base = base + [d for d in self.extra_dopants if d not in base]
        return base


# Global default configuration instance
CONFIG = Config()
