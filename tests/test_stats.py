"""Calibration tests for the estimators."""
import numpy as np
import pytest

from sei import stats as S


def test_welch_aa_false_positive_rate():
    rng = np.random.default_rng(0)
    p = [S.welch(rng.normal(10, 3, 800), rng.normal(10, 3, 800))["p_value"] for _ in range(2000)]
    assert 0.035 < np.mean(np.array(p) < 0.05) < 0.065


def test_welch_ci_coverage_of_relative_lift():
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(1500):
        r = S.welch(rng.normal(11, 4, 1000), rng.normal(10, 4, 1000))
        covered += r["rel_ci_low"] <= 0.10 <= r["rel_ci_high"]
    assert 0.93 < covered / 1500 < 0.97


def test_cuped_unbiased_and_reduces_variance():
    rng = np.random.default_rng(2)
    est, vr = [], []
    for _ in range(400):
        x_t, x_c = rng.normal(20, 6, 2000), rng.normal(20, 6, 2000)
        y_t = 0.8 * x_t + rng.normal(0, 3, 2000) + 1.0
        y_c = 0.8 * x_c + rng.normal(0, 3, 2000)
        r = S.cuped(y_t, y_c, x_t, x_c)
        est.append(r["diff"]); vr.append(r["variance_reduction"])
    assert abs(np.mean(est) - 1.0) < 0.03
    assert np.mean(vr) > 0.6  # corr^2 = 0.8^2*36 / (0.8^2*36 + 9) ~= 0.72


def test_ratio_metric_delta_method_calibrated():
    rng = np.random.default_rng(3)
    p = []
    for _ in range(1500):
        den_t, den_c = rng.poisson(50, 1000), rng.poisson(50, 1000)
        ctr_t, ctr_c = rng.beta(2, 150, 1000), rng.beta(2, 150, 1000)  # user-level heterogeneity
        p.append(S.ratio_metric(rng.binomial(den_t, ctr_t), den_t, rng.binomial(den_c, ctr_c), den_c)["p_value"])
    assert 0.035 < np.mean(np.array(p) < 0.05) < 0.065


def test_srm_detects_imbalance():
    assert not S.srm_check({"a": 10_050, "b": 9_950})["srm_detected"]
    assert S.srm_check({"a": 10_500, "b": 9_500})["srm_detected"]


def test_msprt_controls_type_one_error_iid_textbook_case():
    """Only the i.i.d. case has an exact guarantee."""
    rng = np.random.default_rng(4)
    k = np.arange(1, 29)
    se = np.sqrt(2 / (k * 300))
    hits = 0
    for _ in range(2000):
        d = (rng.normal(0, 1 / np.sqrt(300), 28).cumsum() - rng.normal(0, 1 / np.sqrt(300), 28).cumsum()) / k
        hits += S.msprt_pvalue(d, se, tau2=(2 * se[-1]) ** 2)[-1] < 0.05
    assert hits / 2000 <= 0.05


def test_power_formula_roundtrip():
    n = S.required_n_per_arm(sd=10, mde_abs=0.5)
    assert S.mde(10, n) == pytest.approx(0.5, rel=0.01)
    assert S.required_n_per_arm(10, 0.5, variance_reduction=0.5) == pytest.approx(n / 2, rel=0.01)


def test_bayes_expected_loss_nonnegative_and_probability_sane():
    b = S.bayes_normal(diff=1.0, se=0.5)
    assert 0.97 < b["p_better"] < 0.98 and b["expected_loss_if_ship"] >= 0
