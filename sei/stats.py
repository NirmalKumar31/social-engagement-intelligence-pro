"""Reusable experimentation statistics.

Every estimator returns plain dicts so results flow straight into tables, dashboards and Power BI.
Unit tests in tests/test_stats.py check calibration (A/A false-positive rates, CI coverage).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

Z = stats.norm.ppf(0.975)


def welch(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Difference in means with Welch SE, plus a delta-method CI on the relative lift."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    mt, mc = y_t.mean(), y_c.mean()
    vt, vc = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    diff, se = mt - mc, np.sqrt(vt + vc)
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(diff / se)) if se > 0 else 1.0
    rel = diff / mc if mc else np.nan
    # Var(mt/mc) by the delta method (independent groups)
    rel_se = np.sqrt(vt / mc**2 + vc * mt**2 / mc**4) if mc else np.nan
    return {"mean_control": mc, "mean_treatment": mt, "diff": diff, "se": se,
            "ci_low": diff - z * se, "ci_high": diff + z * se, "p_value": p,
            "rel_lift": rel, "rel_ci_low": rel - z * rel_se, "rel_ci_high": rel + z * rel_se,
            "n_control": len(y_c), "n_treatment": len(y_t)}


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    cov = np.cov(y, x, ddof=1)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0


def cuped(y_t, y_c, x_t, x_c, alpha: float = 0.05) -> dict:
    """CUPED (Deng et al. 2013): regress out a pre-period covariate with a pooled theta.

    The adjustment keeps each arm's mean comparable (x is centred on the pooled mean) so the
    relative lift is still expressed against the raw control mean.
    """
    y = np.concatenate([y_t, y_c]); x = np.concatenate([x_t, x_c])
    theta = cuped_theta(y, x)
    mx = x.mean()
    adj_t = y_t - theta * (x_t - mx)
    adj_c = y_c - theta * (x_c - mx)
    out = welch(adj_t, adj_c, alpha)
    raw = welch(y_t, y_c, alpha)
    out["theta"] = theta
    out["variance_reduction"] = 1 - (out["se"] ** 2) / (raw["se"] ** 2)
    out["mean_control"], out["mean_treatment"] = raw["mean_control"], raw["mean_treatment"]
    return out


def ratio_metric(num_t, den_t, num_c, den_c, alpha: float = 0.05) -> dict:
    """Ratio-of-sums metric (e.g. CTR = clicks / impressions) with delta-method variance.

    Randomisation is per user, so a per-event t-test would understate variance; the delta method
    treats each user as the unit of analysis.
    """
    def _ratio(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        n = len(num); mn, md = num.mean(), den.mean()
        r = mn / md
        cov = np.cov(num, den, ddof=1)
        var = (cov[0, 0] / md**2 - 2 * mn * cov[0, 1] / md**3 + mn**2 * cov[1, 1] / md**4) / n
        return r, var
    rt, vt = _ratio(num_t, den_t)
    rc, vc = _ratio(num_c, den_c)
    diff, se = rt - rc, np.sqrt(vt + vc)
    z = stats.norm.ppf(1 - alpha / 2)
    rel_se = np.sqrt(vt / rc**2 + vc * rt**2 / rc**4)
    return {"mean_control": rc, "mean_treatment": rt, "diff": diff, "se": se,
            "ci_low": diff - z * se, "ci_high": diff + z * se,
            "p_value": 2 * stats.norm.sf(abs(diff / se)), "rel_lift": diff / rc,
            "rel_ci_low": diff / rc - z * rel_se, "rel_ci_high": diff / rc + z * rel_se,
            "n_control": len(num_c), "n_treatment": len(num_t)}


def poisson_bootstrap_quantile(y_t, y_c, q: float = 0.5, n_boot: int = 500, seed: int = 0) -> dict:
    """Poisson bootstrap CI for a difference in quantiles (scales to large n, streaming-friendly)."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.sort(np.asarray(y_t, float)), np.sort(np.asarray(y_c, float))

    def wq(sorted_y, w):
        cw = np.cumsum(w)
        return sorted_y[np.searchsorted(cw, q * cw[-1])]
    diffs = np.array([wq(y_t, rng.poisson(1, len(y_t))) - wq(y_c, rng.poisson(1, len(y_c)))
                      for _ in range(n_boot)])
    point = np.quantile(y_t, q) - np.quantile(y_c, q)
    return {"diff": point, "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975),
            "p_value": min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}


def srm_check(observed: dict, expected_share: dict | None = None) -> dict:
    """Sample-ratio-mismatch chi-square test. p < 0.001 is the conventional alarm threshold."""
    keys = list(observed)
    obs = np.array([observed[k] for k in keys], float)
    share = np.array([expected_share[k] for k in keys]) if expected_share else np.full(len(keys), 1 / len(keys))
    chi2, p = stats.chisquare(obs, obs.sum() * share)
    return {"chi2": chi2, "p_value": p, "srm_detected": p < 0.001, "observed": dict(zip(keys, obs.astype(int)))}


def smd(x_t, x_c) -> float:
    """Standardised mean difference for covariate balance (|SMD| < 0.1 is balanced)."""
    x_t, x_c = np.asarray(x_t, float), np.asarray(x_c, float)
    pooled = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return (x_t.mean() - x_c.mean()) / pooled if pooled > 0 else 0.0


def adjust_pvalues(p: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    return multipletests(np.asarray(p, float), method=method)[1]


def required_n_per_arm(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                       variance_reduction: float = 0.0) -> int:
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = sd**2 * (1 - variance_reduction)
    return int(np.ceil(2 * var * (za + zb) ** 2 / mde_abs**2))


def mde(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8,
        variance_reduction: float = 0.0) -> float:
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) * np.sqrt(2 * sd**2 * (1 - variance_reduction) / n_per_arm)


def msprt_pvalue(diffs: np.ndarray, ses: np.ndarray, tau2: float) -> np.ndarray:
    """Always-valid p-values via the mixture SPRT with a normal mixing prior (Johari et al. 2017).

    diffs/ses are the running estimate and its SE at each look. Returns the running-minimum
    always-valid p-value, which may be monitored continuously without inflating type-I error.
    """
    diffs, ses = np.asarray(diffs, float), np.asarray(ses, float)
    v = ses**2
    lam = np.sqrt(v / (v + tau2)) * np.exp(tau2 * diffs**2 / (2 * v * (v + tau2)))
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))


def bayes_normal(diff: float, se: float, prior_sd: float | None = None) -> dict:
    """Posterior for a difference under a normal likelihood and a N(0, prior_sd^2) skeptical prior."""
    if prior_sd is None:
        post_mean, post_sd = diff, se
    else:
        w = prior_sd**2 / (prior_sd**2 + se**2)
        post_mean, post_sd = w * diff, np.sqrt(w) * se
    p_better = stats.norm.sf(0, loc=post_mean, scale=post_sd)
    # Expected loss of shipping = E[max(0, -delta)]
    z = post_mean / post_sd
    exp_loss = post_sd * stats.norm.pdf(z) - post_mean * stats.norm.cdf(-z)
    return {"post_mean": post_mean, "post_sd": post_sd, "p_better": p_better,
            "expected_loss_if_ship": exp_loss,
            "cred_low": post_mean - Z * post_sd, "cred_high": post_mean + Z * post_sd}


def beta_binomial(succ_t: int, n_t: int, succ_c: int, n_c: int, draws: int = 200_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pt = rng.beta(1 + succ_t, 1 + n_t - succ_t, draws)
    pc = rng.beta(1 + succ_c, 1 + n_c - succ_c, draws)
    d = pt - pc
    return {"p_treatment_higher": (d > 0).mean(), "diff_mean": d.mean(),
            "cred_low": np.quantile(d, 0.025), "cred_high": np.quantile(d, 0.975)}
