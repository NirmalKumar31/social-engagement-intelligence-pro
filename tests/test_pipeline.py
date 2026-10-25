"""Data-contract tests on pipeline outputs (run after `make all`)."""
import pandas as pd
import pytest

from sei import config as C

pytestmark = pytest.mark.skipif(not (C.PROCESSED / "user_experiment.parquet").exists(), reason="run `make all` first")


def test_user_table_contract():
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    assert df.user_id.is_unique and len(df) == C.N_USERS
    assert set(df.arm) == set(C.ARMS)
    assert df.churned.isin([0, 1]).all()
    assert (df.exp_minutes_per_day >= 0).all()


def test_churn_features_do_not_peek_into_label_window():
    from sei.features import load_raw
    from sei.retention import SCORING_DAY, build_features
    users, daily = load_raw()
    f = build_features(users, daily)
    assert (f.active_days_last3 >= 0).all() and (f.active_days_last3 <= 3).all()
    assert "late_minutes" not in f.columns and not any(c.startswith("exp_") for c in f.columns)
    assert SCORING_DAY < C.EXP_DAYS - 1


def test_assignment_is_deterministic():
    from sei.simulate import assign_arm
    import numpy as np
    ids = np.arange(1, 1001)
    assert (assign_arm(ids) == assign_arm(ids)).all()


def test_powerbi_export_consistent():
    import json
    bim = json.loads((C.POWERBI / "SEI_Dashboard.SemanticModel" / "model.bim").read_text())
    names = {t["name"] for t in bim["model"]["tables"]}
    for t in names - {"_Measures"}:
        assert (C.POWERBI / "data" / f"{t}.csv").exists()


def test_pbip_is_fabric_git_ready():
    import json
    for item, kind in [("SemanticModel", "SemanticModel"), ("Report", "Report")]:
        meta = json.loads((C.POWERBI / f"SEI_Dashboard.{item}" / ".platform").read_text())
        assert meta["metadata"]["type"] == kind
    bim = json.loads((C.POWERBI / "SEI_Dashboard.SemanticModel" / "model.bim").read_text())
    assert bim["model"]["expressions"][0]["name"] == "DataPath"


def test_powerbi_columns_unique_case_insensitive():
    import json
    bim = json.loads((C.POWERBI / "SEI_Dashboard.SemanticModel" / "model.bim").read_text())
    for t in bim["model"]["tables"]:
        names = [c["name"].lower() for c in t["columns"]]
        assert len(names) == len(set(names)), t["name"]
