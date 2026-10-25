"""Regression tests for the 2026-10-25 audit (docs/AUDIT_2026-10-25.md). Each test names its finding."""
import json

import numpy as np
import pandas as pd
import pytest

from sei import config as C
from sei import stats as S
from sei.decision import decide

HAVE_DATA = (C.PROCESSED / "user_experiment.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run the pipeline first")


# ---------------------------------------------------------------- S01 decision rule
def _results(primary=(0.08, 0.06, 0.10, 1e-6), **guards):
    """Build a metric_results-shaped frame for two arms; guards: metric -> (rel, lo, hi)."""
    rows = []
    for arm in C.TREATMENTS:
        rel, lo, hi, p = primary
        rows.append({"arm": arm, "metric": "minutes", "rel_lift": rel, "rel_ci_low": lo, "rel_ci_high": hi, "p_adj_bh": p})
        for m in C.GUARDRAILS:
            g = guards.get(m, (0.0, -0.005, 0.005))
            rows.append({"arm": arm, "metric": m, "rel_lift": g[0], "rel_ci_low": g[1], "rel_ci_high": g[2], "p_adj_bh": 0.5})
    return pd.DataFrame(rows)


def test_s01_all_guardrails_pass_launches():
    assert decide(_results())["recommendation_state"] == "LAUNCH"


def test_s01_null_primary_does_not_launch():
    d = decide(_results(primary=(0.001, -0.01, 0.012, 0.8)))
    assert d["recommendation_state"] == "DO NOT LAUNCH" and d["recommended_arm"] is None


def test_s01_reversed_primary_does_not_launch():
    assert decide(_results(primary=(-0.05, -0.07, -0.03, 1e-6)))["recommendation_state"] == "DO NOT LAUNCH"


def test_s01_critical_guardrail_harm_blocks_launch():
    assert decide(_results(churned=(0.15, 0.10, 0.20)))["recommendation_state"] == "DO NOT LAUNCH"


def test_s01_inconclusive_critical_guardrail_is_not_treated_as_safe():
    # not significant, but the CI cannot rule out a harm larger than the margin
    d = decide(_results(churned=(0.01, -0.05, 0.07)))
    assert d["recommendation_state"] == "DO NOT LAUNCH"
    assert d["arms"]["ai_feed"]["guardrails"]["churned"]["status"] == "INCONCLUSIVE"


def test_s01_standard_guardrail_harm_is_conditional():
    assert decide(_results(neg_feedback_rate=(0.08, 0.06, 0.10)))["recommendation_state"] == "CONDITIONAL"


@needs_data
def test_s01_memo_is_generated_and_labelled_synthetic():
    memo = (C.REPORTS / "EXECUTIVE_MEMO.md").read_text()
    summary = json.loads((C.REPORTS / "experiment_summary.json").read_text())
    assert C.SYNTHETIC_NOTICE in memo
    assert summary["decision"]["recommendation_state"] in memo


# ---------------------------------------------------------------- S10 CUPED estimand
def test_s10_cuped_relative_lift_uses_raw_control_denominator():
    rng = np.random.default_rng(0)
    x_t, x_c = rng.normal(10, 3, 300), rng.normal(12, 3, 300)        # deliberately imbalanced pre-period
    r = S.cuped(x_t + rng.normal(1, 1, 300), x_c + rng.normal(0, 1, 300), x_t, x_c)
    assert r["rel_lift"] == pytest.approx(r["diff"] / r["mean_control"])
    assert "mean_control_adj" in r and r["mean_control_adj"] != r["mean_control"]


# ---------------------------------------------------------------- S06 campaign never chooses a guaranteed loss
def test_s06_adverse_economics_send_to_nobody():
    from sei.retention import campaign_value
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000); y = (rng.uniform(0, 1, 5000) < p).astype(int)
    cost, save, value = 3.0, 0.05, 5.0          # a known churner is worth at most 0.25 < cost
    threshold = cost / (save * value)           # break-even rule
    assert threshold > 1 and campaign_value(p, y, threshold, cost, save, value) == 0.0


# ---------------------------------------------------------------- S02 point-in-time cohort and features
@needs_data
def test_s02_features_unchanged_when_future_is_perturbed():
    from sei.features import load_raw
    from sei.retention import SCORING_DAY, build_features
    users, daily = load_raw()
    perturbed = daily.copy()
    future = perturbed.day > SCORING_DAY
    perturbed.loc[future, ["minutes", "likes", "comments", "ad_clicks"]] *= 7
    perturbed = perturbed[~(future & (perturbed.user_id % 3 == 0))]       # also delete some future activity
    pd.testing.assert_frame_equal(build_features(users, daily), build_features(users, perturbed))


@needs_data
def test_s02_no_scored_user_had_already_quit():
    s = json.loads((C.REPORTS / "retention_summary.json").read_text())["churn_model"]["simulator_benchmark"]
    assert s["already_quit_at_scoring"] == 0


# ---------------------------------------------------------------- S03 survival endpoint
@needs_data
def test_s03_survival_events_censored_consistently():
    km = pd.read_csv(C.TABLES / "km_survival.csv")
    from sei.retention import LAST_START
    assert km.day.max() <= LAST_START


# ---------------------------------------------------------------- S04 sequential prior uses pre-period only
@needs_data
def test_s04_mixture_scale_is_prespecified_and_diagnosed_on_real_design():
    sq = json.loads((C.REPORTS / "experiment_summary.json").read_text())["sequential"]
    assert sq["tau_source"].startswith("5% of pre-period") and "re-randomised" in sq["null_design"]


# ---------------------------------------------------------------- S08 control baseline clears arm filters
def test_s08_control_measures_remove_arm_filters():
    from sei.powerbi import MEASURES
    for name, expr, *_ in MEASURES:
        if '"control"' in expr:
            assert "REMOVEFILTERS(dim_arm)" in expr, name


# ---------------------------------------------------------------- S09 pandas extension dtypes
def test_s09_dtype_handles_extension_strings():
    from sei.powerbi import _dtype
    assert _dtype(pd.Series(["a", "b"], dtype="string")) == ("string", "type text")
    assert _dtype(pd.Series([1, 2], dtype="Int64")) == ("int64", "Int64.Type")
    assert _dtype(pd.Series([True, False])) == ("int64", "Int64.Type")


# ---------------------------------------------------------------- S12 virality threshold from training data
@needs_data
def test_s12_virality_threshold_from_training_only():
    v = json.loads((C.REPORTS / "virality_summary.json").read_text())
    assert v["threshold_source"] == "training articles only" and v["train_until"] < v["test_from"]


# ---------------------------------------------------------------- S14 Power BI arm identity for HTE tables
def test_s14_hte_tables_relate_to_arm_dimension():
    from sei.powerbi import RELATIONSHIPS
    linked = {r[0] for r in RELATIONSHIPS if r[2] == "dim_arm"}
    assert {"qini_curves", "cate_deciles"} <= linked
