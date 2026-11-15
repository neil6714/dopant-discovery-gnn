"""Unit tests for defect manifest integrity and status validation."""
import os
import pandas as pd

from src.config import CONFIG


def test_manifest_no_duplicate_filenames():
    """Verify all filenames registered in the defect manifest are strictly unique."""
    manifest_path = CONFIG.manifest_path
    assert os.path.exists(manifest_path), f"Manifest file missing at {manifest_path}"

    df = pd.read_csv(manifest_path)
    assert not df["filename"].duplicated().any(), (
        f"Found duplicate filenames in manifest: {df[df['filename'].duplicated()]['filename'].tolist()}"
    )


def test_manifest_valid_status_values():
    """Verify that every record has a recognized status category."""
    manifest_path = CONFIG.manifest_path
    assert os.path.exists(manifest_path), f"Manifest file missing at {manifest_path}"

    df = pd.read_csv(manifest_path)
    valid_statuses = {"unrelaxed", "relaxed", "candidate", "needs_regeneration"}
    actual_statuses = set(df["status"].dropna().unique())

    assert actual_statuses.issubset(valid_statuses), (
        f"Found invalid status values in manifest: {actual_statuses - valid_statuses}"
    )


def test_manifest_required_columns():
    """Verify presence of essential defect metadata columns."""
    manifest_path = CONFIG.manifest_path
    df = pd.read_csv(manifest_path)
    required_cols = {"filename", "host", "site", "wyckoff", "dopant", "dopant_z", "status", "ood_flag", "path"}
    assert required_cols.issubset(set(df.columns))
