"""Unit tests for active learning screening loop and candidate hypothesis workflows."""
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autonomous_lab import (
    _append_live_al_log,
    generate_hypothesis,
    run_autonomous_lab,
)
from src.models.train_gnn import DefectGNN


def test_generate_hypothesis_any_host_and_site():
    """Verify hypothesis generation across hosts and sites with manifest tracking."""
    temp_dir = tempfile.mkdtemp()
    manifest_file = os.path.join(temp_dir, "manifest.csv")
    try:
        # Generate hypothesis for SiC site 2b_0
        out_sic = generate_hypothesis(
            dopant="Ti",
            host="SiC",
            site="2b_0",
            output_dir=temp_dir,
            manifest_path=manifest_file,
        )
        assert os.path.exists(out_sic)
        assert "SiC_doped_Ti_2b_0.cif" in out_sic

        # Verify manifest entry
        assert os.path.exists(manifest_file)
        df_man = pd.read_csv(manifest_file)
        assert len(df_man) == 1
        assert df_man.iloc[0]["status"] == "candidate"
        assert df_man.iloc[0]["host"] == "SiC"
        assert df_man.iloc[0]["dopant"] == "Ti"

        # Generate hypothesis for GaN
        out_gan = generate_hypothesis(
            dopant="V",
            host="GaN",
            output_dir=temp_dir,
            manifest_path=manifest_file,
        )
        assert os.path.exists(out_gan)
        assert "GaN_doped_V_" in out_gan

        df_man2 = pd.read_csv(manifest_file)
        assert len(df_man2) == 2
        assert "GaN" in df_man2["host"].values
    finally:
        shutil.rmtree(temp_dir)


def test_append_live_al_log():
    """Verify atomic logging of out-of-sample predictions and errors."""
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, "live_al_log.csv")
    try:
        record1 = {
            "timestamp": "2026-11-14T01:00:00",
            "candidate": "SiC_doped_U_2b_0.cif",
            "host": "SiC",
            "dopant": "U",
            "site": "2b_0",
            "pred_e_form_ev": 1.5,
            "ensemble_std_ev": 0.08,
            "sim_e_form_ev": 1.4,
            "absolute_error_ev": 0.1,
            "converged": True,
        }
        _append_live_al_log(log_file, record1)
        assert os.path.exists(log_file)
        df = pd.read_csv(log_file)
        assert len(df) == 1
        assert df.iloc[0]["candidate"] == "SiC_doped_U_2b_0.cif"
        assert np.isclose(df.iloc[0]["absolute_error_ev"], 0.1)

        # Append second record
        record2 = dict(record1)
        record2["candidate"] = "SiC_doped_U_2a_1.cif"
        _append_live_al_log(log_file, record2)
        df2 = pd.read_csv(log_file)
        assert len(df2) == 2
    finally:
        shutil.rmtree(temp_dir)


def test_dry_run_ranking_and_selection():
    """Verify that dry-run mode ranks candidates by ensemble uncertainty without invoking simulation."""
    temp_dir = tempfile.mkdtemp()
    cand_dir = os.path.join(temp_dir, "candidates")
    models_dir = os.path.join(temp_dir, "models")
    manifest_file = os.path.join(temp_dir, "manifest.csv")
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    try:
        # Create 2 synthetic ensemble members with distinct predictions
        m1 = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
        m1.y_mean = 1.0
        m1.y_std = 1.0
        torch.save(
            {"state_dict": m1.state_dict(), "y_mean": 1.0, "y_std": 1.0, "seed": 42},
            os.path.join(models_dir, "ensemble_model_0.pth"),
        )

        m2 = DefectGNN(in_features=9, hidden_dim=32, edge_dim=8)
        m2.y_mean = 2.0
        m2.y_std = 1.0
        torch.save(
            {"state_dict": m2.state_dict(), "y_mean": 2.0, "y_std": 1.0, "seed": 43},
            os.path.join(models_dir, "ensemble_model_1.pth"),
        )

        # Generate candidates in test environment
        c1 = generate_hypothesis("Ti", host="SiC", site="2b_0", output_dir=cand_dir, manifest_path=manifest_file)
        c2 = generate_hypothesis("V", host="SiC", site="2b_0", output_dir=cand_dir, manifest_path=manifest_file)
        assert os.path.exists(c1)
        assert os.path.exists(c2)

        # Execute dry run
        ranked = run_autonomous_lab(
            candidates_dir=cand_dir,
            manifest_path=manifest_file,
            models_dir=models_dir,
            num_models=2,
            uncertainty_threshold=0.01,
            batch_size=1,
            budget=1,
            dry_run=True,
        )

        assert len(ranked) == 2
        assert "pred_std" in ranked[0]
        # Candidate status in manifest remains 'candidate' because no simulation was run
        df_man = pd.read_csv(manifest_file)
        assert all(s == "candidate" for s in df_man["status"])
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running test_generate_hypothesis_any_host_and_site...")
    test_generate_hypothesis_any_host_and_site()
    print("Running test_append_live_al_log...")
    test_append_live_al_log()
    print("Running test_dry_run_ranking_and_selection...")
    test_dry_run_ranking_and_selection()
    print("All autonomous lab tests passed successfully!")
