"""Central configuration dataclass for the dopant discovery pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List


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
    uncertainty_threshold: float = 0.05
    train_split: float = 0.8

    @property
    def training_dopants(self) -> List[str]:
        """Return combined list of 3d, 4d, and 5d dopants for training."""
        return self.dopants_3d + self.dopants_4d + self.dopants_5d


# Global default configuration instance
CONFIG = Config()
