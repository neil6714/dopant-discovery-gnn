"""Tests for the descriptor module, GP and residual models, DefectGNN options and the pool benchmark."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.active_learning import run_pool_benchmark
from src.models.baselines import DescriptorModel, grouped_splits, out_of_fold_predictions
from src.models.residual import ResidualDefectModel
from src.models.train_gnn import DefectGNN, train_one_model
from src.utils.descriptors import RICH_FEATURE_NAMES, rich_descriptor, site_is_second
from pymatgen.core import Element

DOPANTS = ["Ti", "V", "Cr", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Nb", "Mo", "Pd"]


def make_graph(symbol: str, host: str = "SiC", site: str = "2b_0") -> Data:
    """Tiny two-node graph with the same layout as the real builder, including data.desc."""
    z = Element(symbol).Z
    x = torch.tensor(
        [[float(z), 1.5, 1.4, 4.0, 4.0, 4.0, 1.0, 1.0, 0.0],
         [14.0, 1.9, 1.1, 14.0, 3.0, 4.0, 0.0, 1.0, 0.0]], dtype=torch.float)
    g = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 8),
             y=torch.tensor([float(rich_descriptor(symbol, host, site)[8]) * -0.5 + 0.1 * z / 10]))
    g.z = torch.tensor([z, 14])
    g.dopant_z, g.host, g.site, g.ood_flag = z, host, site, False
    g.desc = torch.tensor(rich_descriptor(symbol, host, site)).unsqueeze(0)
    g.filename = f"{host}_doped_{symbol}_{site}.cif"
    return g


def test_rich_descriptor_shape_and_flags():
    vec = rich_descriptor("Pd", "SiC", "2a_1", mu_dopant=-5.2)
    assert vec.shape == (len(RICH_FEATURE_NAMES),)
    named = dict(zip(RICH_FEATURE_NAMES, vec))
    assert named["mu_dopant_ev"] == pytest.approx(-5.2)
    assert named["host_SiC"] == 1.0 and named["host_GaN"] == 0.0
    assert named["second_site"] == 1.0
    assert named["d_electrons"] == 10.0  # Pd is 4d10


def test_site_flag():
    assert site_is_second("2b_0") == 0.0
    assert site_is_second("2a_1") == 1.0
    assert site_is_second(None) == 0.0


def test_descriptor_models_predict_with_std():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.05, size=20)
    groups = np.array([i % 5 for i in range(20)])
    for kind in ("ridge", "gp_iso", "gp_ard"):
        mean, std = DescriptorModel(kind, restarts=0).fit(x, y, groups).predict(x[:4])
        assert mean.shape == (4,)
        assert (std is None) == (kind == "ridge")


def test_single_group_falls_back_instead_of_crashing():
    x = np.random.default_rng(1).normal(size=(6, 3))
    y = x[:, 0]
    groups = np.array(["a"] * 6)
    assert grouped_splits(groups) == []
    DescriptorModel("ridge").fit(x, y, groups).predict(x)
    assert out_of_fold_predictions("ridge", x, y, groups).shape == (6,)


def test_defect_gnn_options_forward_and_checkpoint_buffers():
    graphs = [make_graph(s) for s in DOPANTS[:6]]
    model = DefectGNN(in_features=9, hidden_dim=16, use_embedding=False, global_dim=len(RICH_FEATURE_NAMES), batch_norm=False)
    model.fit_desc_scaler(graphs)
    assert not torch.allclose(model.desc_std, torch.ones_like(model.desc_std))
    from torch_geometric.data import Batch
    out = model(Batch.from_data_list(graphs[:3]))
    assert out.shape == (3, 1)
    assert "desc_mean" in model.state_dict()
    # Without the embedding a held-out element still gets a defined input
    assert model.element_embedding is None


def test_default_defect_gnn_unchanged_by_new_options():
    model = DefectGNN(in_features=9, hidden_dim=16)
    assert model.element_embedding is not None and model.global_dim == 0
    assert not hasattr(model, "desc_mean")


def test_residual_model_fits_and_predicts():
    graphs = [make_graph(s) for s in DOPANTS]
    x = np.stack([g.desc.view(-1).numpy() for g in graphs]).astype(np.float64)
    groups = np.array([Element.from_Z(g.dopant_z).symbol for g in graphs])
    model = ResidualDefectModel(epochs=3, batch_size=4, device=torch.device("cpu")).fit(graphs[:10], x[:10], groups[:10])
    out = model.predict(graphs[10:], x[10:])
    assert out["pred"].shape == (2,)
    assert np.allclose(out["pred"], out["base"] + out["correction"])


def test_train_one_model_with_descriptor_options():
    graphs = [make_graph(s) for s in DOPANTS]
    net, _ = train_one_model(graphs, None, epochs=2, batch_size=4, hidden_dim=16, device=torch.device("cpu"),
                             model_kwargs={"use_embedding": False, "global_dim": len(RICH_FEATURE_NAMES), "batch_norm": False})
    assert net.global_dim == len(RICH_FEATURE_NAMES)


@pytest.mark.parametrize("surrogate", ["ensemble", "gp"])
def test_pool_benchmark_holds_out_whole_dopants(tmp_path, surrogate):
    graphs = [make_graph(s) for s in DOPANTS] + [make_graph(s, site="2a_1") for s in DOPANTS]
    df, cal = run_pool_benchmark(
        strategy="both", seed_size=6, batch_size=4, budget=4, n_repeats=1, dataset=graphs, epochs=1,
        num_models=2, device=torch.device("cpu"), surrogate=surrogate, holdout_by="dopant",
        output_csv=str(tmp_path / "b.csv"), output_plot=str(tmp_path / "p.png"),
        output_calibration_csv=str(tmp_path / "c.csv"),
        model_kwargs={"use_embedding": False, "global_dim": len(RICH_FEATURE_NAMES), "batch_norm": False},
    )
    assert not df.empty
    assert (df["n_heldout"] > 0).all()
    assert {"mae_ev", "mae_pool_ev", "surrogate"} <= set(df.columns)
    # both strategies share the same held-out dopants, so their step-0 test sizes match
    first = df[df.step == 0].groupby("strategy")["n_heldout"].first()
    assert first.nunique() == 1
