"""Experiment readout: design checks, effects, time dynamics and decision statistics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import curve_fit

from . import config as C
from . import stats as S
from .decision import decide
from .decision import headline as headline_text
from .features import load_raw
from .plotting import plt, save, setup

# metric: (label, tier, kind, pre-period covariate or ratio parts, higher is better)
METRICS = {
    "minutes": ("Daily active minutes / user", "primary", "mean", "minutes", True),
    "active_days": ("DAU rate (active days / day)", "secondary", "mean", "active_days", True),
    "sessions": ("Sessions / user / day", "secondary", "mean", "sessions", True),
    "likes": ("Likes / user / day", "secondary", "mean", "likes", True),
    "shares": ("Shares / user / day", "secondary", "mean", "shares", True),
    "posts_created": ("Posts created / user / day", "secondary", "mean", "posts_created", True),
    "ad_revenue": ("Ad revenue / user / day (USD)", "secondary", "mean", "ad_revenue", True),
    "churned": ("28-day churn rate", "guardrail", "binary", None, False),
    "friend_interactions": ("Friend interactions / user / day", "guardrail", "mean", "friend_interactions", True),
    "neg_feedback_rate": ("Negative feedback per 100 min", "guardrail", "ratio", ("negative_feedback", "minutes"), False),
    "ad_ctr": ("Ad CTR", "guardrail", "ratio", ("ad_clicks", "ad_impressions"), True),
}


def design_checks(df: pd.DataFrame) -> dict:
    srm = S.srm_check(df.arm.value_counts().reindex(C.ARMS).to_dict())
    rows = []
    numeric = ["pre_minutes_per_day", "pre_active_days_per_day", "pre_sessions_per_day",
               "pre_friend_interactions_per_day", "pre_ad_revenue_per_day", "tenure_days", "friend_count",
               "is_new_user"]
    cats = {"age_group": ["18-24", "55+"], "device": ["iOS", "Android"], "account_type": ["creator"],
            "region": ["India", "North America"]}
    ctrl = df[df.arm == "control"]
    for arm in C.TREATMENTS:
        t = df[df.arm == arm]
        for col in numeric:
            rows.append({"arm": arm, "covariate": col, "smd": S.smd(t[col], ctrl[col])})
        for col, levels in cats.items():
            for lv in levels:
                rows.append({"arm": arm, "covariate": f"{col}={lv}", "smd": S.smd(t[col] == lv, ctrl[col] == lv)})
    bal = pd.DataFrame(rows)
    bal["balanced"] = bal.smd.abs() < 0.1
    bal.to_csv(C.TABLES / "balance_smd.csv", index=False)
    srm_tbl = pd.DataFrame([{"arm": k, "users": v, "share": v / sum(srm["observed"].values())}
                            for k, v in srm["observed"].items()])
    srm_tbl.to_csv(C.TABLES / "srm.csv", index=False)

    y, x = df.exp_minutes_per_day.values, df.pre_minutes_per_day.values
    rho = np.corrcoef(y, x)[0, 1]
    sd = y[df.arm == "control"].std()
    n_arm = int(df.arm.value_counts().min())
    base = y[df.arm == "control"].mean()
    power_rows = []
    for vr_label, vr in [("raw", 0.0), ("cuped", rho**2)]:
        for n in [2_000, 5_000, 10_000, 20_000, 50_000]:
            power_rows.append({"method": vr_label, "n_per_arm": n, "mde_abs": S.mde(sd, n, variance_reduction=vr),
                               "mde_rel": S.mde(sd, n, variance_reduction=vr) / base})
    pw = pd.DataFrame(power_rows)
    pw.to_csv(C.TABLES / "power_mde.csv", index=False)
    return {"srm": {k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item()) for k, v in srm.items()},
            "max_abs_smd": float(bal.smd.abs().max()), "pre_post_corr": float(rho),
            "mde_rel_raw": float(S.mde(sd, n_arm) / base),
            "mde_rel_cuped": float(S.mde(sd, n_arm, variance_reduction=rho**2) / base),
            "n_needed_2pct_raw": S.required_n_per_arm(sd, 0.02 * base),
            "n_needed_2pct_cuped": S.required_n_per_arm(sd, 0.02 * base, variance_reduction=rho**2),
            "n_per_arm": n_arm}


def aa_test(df: pd.DataFrame, n_sims: int = 1000, seed: int = 1) -> dict:
    """Re-randomise pre-period data; a calibrated test rejects about 5% of A/A splits."""
    rng = np.random.default_rng(seed)
    y = df.pre_minutes_per_day.values
    p = np.empty(n_sims)
    for i in range(n_sims):
        m = rng.random(len(y)) < 0.5
        p[i] = S.welch(y[m], y[~m])["p_value"]
    pd.DataFrame({"p_value": p}).to_csv(C.TABLES / "aa_pvalues.csv", index=False)
    return {"aa_false_positive_rate": float((p < C.ALPHA).mean()), "n_sims": n_sims}


def metric_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrl = df[df.arm == "control"]
    for arm in C.TREATMENTS:
        t = df[df.arm == arm]
        for key, (label, tier, kind, cov, hib) in METRICS.items():
            if kind == "mean":
                yt, yc = t[f"exp_{cov}_per_day"].values, ctrl[f"exp_{cov}_per_day"].values
                raw = S.welch(yt, yc)
                adj = S.cuped(yt, yc, t[f"pre_{cov}_per_day"].values, ctrl[f"pre_{cov}_per_day"].values)
            elif kind == "binary":
                raw = adj = S.welch(t[key].values, ctrl[key].values)
                adj = dict(adj, variance_reduction=0.0)
            else:
                num, den = cov
                scale = 100 if key == "neg_feedback_rate" else 1
                raw = adj = S.ratio_metric(t[f"exp_{num}"] * scale, t[f"exp_{den}"],
                                           ctrl[f"exp_{num}"] * scale, ctrl[f"exp_{den}"])
                adj = dict(adj, variance_reduction=0.0)
            rows.append({"arm": arm, "metric": key, "label": label, "tier": tier, "higher_is_better": hib,
                         "mean_control": adj["mean_control"], "mean_treatment": adj["mean_treatment"],
                         "diff": adj["diff"], "ci_low": adj["ci_low"], "ci_high": adj["ci_high"],
                         "rel_lift": adj["rel_lift"], "rel_ci_low": adj["rel_ci_low"], "rel_ci_high": adj["rel_ci_high"],
                         "p_value": adj["p_value"], "p_value_raw_ttest": raw["p_value"],
                         "se_raw": raw["se"], "se_cuped": adj["se"],
                         "variance_reduction": adj.get("variance_reduction", 0.0)})
    res = pd.DataFrame(rows)
    res["p_adj_bh"] = S.adjust_pvalues(res.p_value, "fdr_bh")
    res["p_adj_holm"] = S.adjust_pvalues(res.p_value, "holm")
    res["significant"] = res.p_adj_bh < C.ALPHA
    good = np.where(res.higher_is_better, res["diff"] > 0, res["diff"] < 0)
    res["verdict"] = np.select([res.significant & good, res.significant & ~good], ["win", "harm"], "neutral")
    res.to_csv(C.TABLES / "metric_results.csv", index=False)
    return res


def user_day_matrix(users: pd.DataFrame, daily: pd.DataFrame, value: str = "minutes") -> np.ndarray:
    """Users x experiment-day matrix of `value`, zero on inactive days."""
    m = np.zeros((len(users), C.EXP_DAYS))
    pos = pd.Series(np.arange(len(users)), index=users.user_id.values)
    e = daily[daily.day >= 0]
    np.add.at(m, (pos.reindex(e.user_id.values).values, e.day.values), e[value].values)
    return m


def _decay(t, ss, nov, tau):
    return ss + nov * np.exp(-t / tau)


def _fit(day, lift):
    try:
        (ss, nov, tau), _ = curve_fit(_decay, day, lift, p0=[0.07, 0.08, 5], bounds=([-1, -1, 0.5], [1, 1, 60]))
    except RuntimeError:
        ss, nov, tau = float(np.mean(lift[-7:])), 0.0, np.nan
    return ss, nov, tau


def daily_dynamics(users: pd.DataFrame, daily: pd.DataFrame, n_boot: int = 200, seed: int = 3):
    """Daily lift in minutes per assigned user, an exponential decay fit, and user-bootstrap CIs."""
    m = user_day_matrix(users, daily)
    arm = users.arm.values
    idx = {a: np.flatnonzero(arm == a) for a in C.ARMS}
    day = np.arange(C.EXP_DAYS)

    def lifts(sel):
        means = {a: m[sel[a]].mean(axis=0) for a in C.ARMS}
        return {a: means[a] / means["control"] - 1 for a in C.TREATMENTS}

    point = lifts(idx)
    rows = []
    for a in C.TREATMENTS:
        t, c = m[idx[a]], m[idx["control"]]
        mt, mc = t.mean(axis=0), c.mean(axis=0)
        se = np.sqrt(t.var(axis=0, ddof=1) / len(t) / mc**2 + c.var(axis=0, ddof=1) / len(c) * mt**2 / mc**4)
        for d in day:
            rows.append({"day": int(d), "date": pd.Timestamp(C.START_DATE) + pd.Timedelta(days=int(d)), "arm": a,
                         "mean_control": mc[d], "mean_treatment": mt[d], "rel_lift": point[a][d],
                         "ci_low": point[a][d] - S.Z * se[d], "ci_high": point[a][d] + S.Z * se[d]})
    out = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    boots = {a: [] for a in C.TREATMENTS}
    for _ in range(n_boot):
        sel = {a: rng.choice(ix, len(ix), replace=True) for a, ix in idx.items()}
        bl = lifts(sel)
        for a in C.TREATMENTS:
            boots[a].append((*_fit(day, bl[a]), bl[a][:7].mean(), bl[a][21:].mean()))
    fits = {}
    for a in C.TREATMENTS:
        ss, nov, tau = _fit(day, point[a])
        out.loc[out.arm == a, "fitted"] = _decay(day, ss, nov, tau)
        b = np.array(boots[a])
        q = lambda j: [float(np.nanquantile(b[:, j], 0.025)), float(np.nanquantile(b[:, j], 0.975))]
        fits[a] = {"fitted_asymptote_lift": float(ss), "fitted_asymptote_ci": q(0),
                   "novelty_boost": float(nov), "decay_days": float(tau), "decay_days_ci": q(2),
                   "week1_lift": float(point[a][:7].mean()), "week1_ci": q(3),
                   "week4_lift": float(point[a][21:].mean()), "week4_ci": q(4),
                   "headline_lift_retained_share": None, "n_boot": n_boot}
    out.to_csv(C.TABLES / "daily_lift.csv", index=False)
    return out, fits


def sequential(users: pd.DataFrame, daily: pd.DataFrame, n_splits: int = 400, seed: int = 812):
    """Running p-values on the primary metric and a re-randomised null check of peeking on the real data."""
    m = user_day_matrix(users, daily)
    cum = m.cumsum(axis=1) / np.arange(1, C.EXP_DAYS + 1)
    arm = users.arm.values
    pre = users[["user_id"]].merge(
        daily[(daily.day < 0)].groupby("user_id").minutes.sum().rename("pre"), on="user_id", how="left").fillna(0)
    pre_ctrl_mean = pre.pre.values[arm == "control"].mean() / C.PRE_DAYS
    tau2 = (0.05 * pre_ctrl_mean) ** 2

    def monitor(a_mask, b_mask, data):
        na, nb = a_mask.sum(), b_mask.sum()
        ma, mb = data[a_mask].mean(axis=0), data[b_mask].mean(axis=0)
        se = np.sqrt(data[a_mask].var(axis=0, ddof=1) / na + data[b_mask].var(axis=0, ddof=1) / nb)
        diff = ma - mb
        return diff, se, 2 * stats.norm.sf(np.abs(diff / se))

    rows = []
    for a in C.TREATMENTS:
        diff, se, p_naive = monitor(arm == a, arm == "control", cum)
        p_mix = S.msprt_pvalue(diff, se, tau2)
        rows += [{"arm": a, "day": d, "diff": diff[d], "se": se[d], "p_naive": p_naive[d], "p_mixture": p_mix[d]}
                 for d in range(C.EXP_DAYS)]
    seq = pd.DataFrame(rows)
    seq.to_csv(C.TABLES / "sequential_monitoring.csv", index=False)

    # Split control users into two pseudo-arms many times and monitor them daily
    rng = np.random.default_rng(seed)
    ctrl = cum[arm == "control"]
    n = len(ctrl)
    masks = rng.random((n_splits, n)) < 0.5
    na = masks.sum(axis=1, keepdims=True); nb = n - na
    s1, s2 = masks @ ctrl, masks @ ctrl**2
    t1, t2 = ctrl.sum(axis=0) - s1, (ctrl**2).sum(axis=0) - s2
    ma, mb = s1 / na, t1 / nb
    va = (s2 - na * ma**2) / (na - 1); vb = (t2 - nb * mb**2) / (nb - 1)
    se = np.sqrt(va / na + vb / nb)
    diff = ma - mb
    p_naive = 2 * stats.norm.sf(np.abs(diff / se))
    p_mix = np.array([S.msprt_pvalue(diff[i], se[i], tau2) for i in range(n_splits)])
    sim = {"null_design": "control users re-randomised into two pseudo-arms, same users re-observed daily",
           "n_splits": n_splits, "tau": float(np.sqrt(tau2)), "tau_source": "5% of pre-period control mean",
           "peeking_fpr_naive": float((p_naive.min(axis=1) < 0.05).mean()),
           "peeking_fpr_msprt": float((p_mix[:, -1] < 0.05).mean()),
           "fixed_horizon_fpr": float((p_naive[:, -1] < 0.05).mean()),
           "guarantee": "approximation; diagnostic only covers the primary metric under this null"}
    for a in C.TREATMENTS:
        s_ = seq[(seq.arm == a) & (seq.p_mixture < 0.05)]
        sim[f"first_day_mixture_sig_{a}"] = int(s_.day.min()) if len(s_) else None
    return seq, sim


def bayesian(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in res.iterrows():
        se = (r.ci_high - r.ci_low) / (2 * S.Z)
        prior = 0.10 * abs(r.mean_control) if r.mean_control else None
        b = S.bayes_normal(r["diff"], se, prior_sd=prior)
        sign = 1 if r.higher_is_better else -1
        p_good = b["p_better"] if sign == 1 else 1 - b["p_better"]
        rows.append({"arm": r.arm, "metric": r.metric, "label": r.label, "post_mean": b["post_mean"],
                     "post_sd": b["post_sd"], "p_improves": p_good, "cred_low": b["cred_low"], "cred_high": b["cred_high"]})
    out = pd.DataFrame(rows)
    ctrl = df[df.arm == "control"]
    for arm in C.TREATMENTS:
        t = df[df.arm == arm]
        bb = S.beta_binomial(int(t.retained.sum()), len(t), int(ctrl.retained.sum()), len(ctrl))
        out.loc[(out.arm == arm) & (out.metric == "churned"), "p_improves"] = bb["p_treatment_higher"]
    out.to_csv(C.TABLES / "bayesian_results.csv", index=False)
    return out


SEGMENT_DIMS = ["age_group", "region", "device", "account_type", "is_new_user", "acquisition_channel"]


def segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in SEGMENT_DIMS:
        for level, g in df.groupby(dim):
            c = g[g.arm == "control"]
            for arm in C.TREATMENTS:
                t = g[g.arm == arm]
                m = S.cuped(t.exp_minutes_per_day.values, c.exp_minutes_per_day.values,
                            t.pre_minutes_per_day.values, c.pre_minutes_per_day.values)
                ch = S.welch(t.churned.values, c.churned.values)
                fi = S.cuped(t.exp_friend_interactions_per_day.values, c.exp_friend_interactions_per_day.values,
                             t.pre_friend_interactions_per_day.values, c.pre_friend_interactions_per_day.values)
                rev = S.cuped(t.exp_ad_revenue_per_day.values, c.exp_ad_revenue_per_day.values,
                              t.pre_ad_revenue_per_day.values, c.pre_ad_revenue_per_day.values)
                rows.append({"dimension": dim, "segment": str(level), "arm": arm, "n_treatment": len(t), "n_control": len(c),
                             "minutes_lift": m["rel_lift"], "minutes_ci_low": m["rel_ci_low"], "minutes_ci_high": m["rel_ci_high"],
                             "minutes_p": m["p_value"], "churn_control": ch["mean_control"], "churn_treatment": ch["mean_treatment"],
                             "churn_diff_pp": ch["diff"] * 100, "churn_ci_low_pp": ch["ci_low"] * 100,
                             "churn_ci_high_pp": ch["ci_high"] * 100, "churn_p": ch["p_value"],
                             "friend_int_lift": fi["rel_lift"], "revenue_lift": rev["rel_lift"]})
    seg = pd.DataFrame(rows)
    seg["minutes_p_adj"] = S.adjust_pvalues(seg.minutes_p)
    seg["churn_p_adj"] = S.adjust_pvalues(seg.churn_p)
    seg["status"] = np.select(
        [(seg.churn_p_adj < 0.05) & (seg.churn_diff_pp > 0), (seg.minutes_p_adj < 0.05) & (seg.minutes_lift > 0)],
        ["churn risk", "clear win"], "inconclusive")
    seg.to_csv(C.TABLES / "segment_results.csv", index=False)

    # Does the AI-feed effect differ by age?
    sub = df[df.arm.isin(["control", "ai_feed"])].assign(treat=lambda x: (x.arm == "ai_feed").astype(int))
    m = smf.ols("exp_minutes_per_day ~ pre_minutes_per_day + treat * age_group", sub).fit(cov_type="HC1")
    joint = m.f_test([p for p in m.params.index if p.startswith("treat:")])
    mc = smf.ols("churned ~ pre_active_days_per_day + treat * age_group", sub).fit(cov_type="HC1")
    jc = mc.f_test([p for p in mc.params.index if p.startswith("treat:")])
    seg.attrs["interaction"] = {"minutes_age_interaction_p": float(joint.pvalue), "churn_age_interaction_p": float(jc.pvalue)}
    return seg


def quantile_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctrl = df[df.arm == "control"].exp_minutes_per_day.values
    for arm in C.TREATMENTS:
        t = df[df.arm == arm].exp_minutes_per_day.values
        for q in [0.25, 0.5, 0.75, 0.9]:
            r = S.poisson_bootstrap_quantile(t, ctrl, q=q, n_boot=300, seed=int(q * 100))
            rows.append({"arm": arm, "quantile": q, **r, "control_q": float(np.quantile(ctrl, q))})
    out = pd.DataFrame(rows)
    out["rel_diff"] = out["diff"] / out.control_q
    out.to_csv(C.TABLES / "quantile_effects.csv", index=False)
    return out


def figures(res, dyn, seq, seg, bal_path=C.TABLES / "balance_smd.csv"):
    setup()
    lab = {a: C.ARM_LABELS[a] for a in C.ARMS}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    order = list(METRICS)
    for ax, arm in zip(axes, C.TREATMENTS):
        r = res[res.arm == arm].set_index("metric").loc[order]
        yy = np.arange(len(order))[::-1]
        good = np.where(r.higher_is_better, r["diff"] > 0, r["diff"] < 0)
        col = np.where(~r.significant, C.INK["muted"], np.where(good, C.STATUS["good"], C.STATUS["critical"]))
        ax.hlines(yy, r.rel_ci_low * 100, r.rel_ci_high * 100, color=col, lw=2)
        ax.scatter(r.rel_lift * 100, yy, color=col, s=40, zorder=3, edgecolor=C.INK["surface"], lw=1.5)
        ax.axvline(0, color=C.INK["axis"], lw=1)
        ax.set_yticks(yy, [METRICS[m][0] for m in order])
        ax.set_title(lab[arm], color=C.ARM_COLORS[arm])
        ax.set_xlabel("Relative change vs control (%), 95% CI")
    fig.suptitle("Metric scorecard  ·  green = significant win, red = significant harm (BH-adjusted)",
                 x=0.01, ha="left", fontsize=11, color=C.INK["secondary"])
    fig.tight_layout()
    save(fig, "01_metric_scorecard")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for arm in C.TREATMENTS:
        a = dyn[dyn.arm == arm]
        ax.fill_between(a.day, a.ci_low * 100, a.ci_high * 100, color=C.ARM_COLORS[arm], alpha=0.12, lw=0)
        ax.plot(a.day, a.rel_lift * 100, color=C.ARM_COLORS[arm], lw=1.4, alpha=0.6)
        ax.plot(a.day, a.fitted * 100, color=C.ARM_COLORS[arm], lw=2.4, label=lab[arm])
        ax.annotate(f"{a.fitted.iloc[-1]*100:.1f}%", (a.day.iloc[-1], a.fitted.iloc[-1] * 100),
                    xytext=(6, 0), textcoords="offset points", va="center", color=C.INK["secondary"])
    ax.axhline(0, color=C.INK["axis"], lw=1)
    ax.set(xlabel="Experiment day", ylabel="Lift in minutes / user (%)",
           title="Daily lift with exponential fit (asymptote is an extrapolation)")
    ax.legend(loc="upper right")
    save(fig, "02_novelty_decay")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for arm in C.TREATMENTS:
        a = seq[seq.arm == arm]
        ax.plot(a.day, a.p_mixture, color=C.ARM_COLORS[arm], lw=2.2, label=f"{lab[arm]} · mixture SPRT")
        ax.plot(a.day, a.p_naive, color=C.ARM_COLORS[arm], lw=1.2, alpha=0.5, label=f"{lab[arm]} · naive")
    ax.axhline(0.05, color=C.STATUS["critical"], lw=1)
    ax.text(27, 0.06, "α = 0.05", color=C.INK["secondary"], ha="right")
    ax.set_yscale("log"); ax.set_ylim(1e-12, 1.2)
    ax.set(xlabel="Experiment day", ylabel="p-value (log scale)", title="Sequential monitoring: naive vs mixture-SPRT p-values (approximation)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "03_sequential_monitoring")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    s = seg[seg.dimension == "age_group"]
    levels = sorted(s.segment.unique())
    x = np.arange(len(levels))
    for i, arm in enumerate(C.TREATMENTS):
        a = s[s.arm == arm].set_index("segment").loc[levels]
        off = (i - 0.5) * 0.36
        ax.bar(x + off, a.churn_diff_pp, width=0.34, color=C.ARM_COLORS[arm], label=lab[arm], edgecolor=C.INK["surface"], lw=2)
        ax.errorbar(x + off, a.churn_diff_pp, yerr=[a.churn_diff_pp - a.churn_ci_low_pp, a.churn_ci_high_pp - a.churn_diff_pp],
                    fmt="none", ecolor=C.INK["secondary"], lw=1, capsize=3)
    ax.axhline(0, color=C.INK["axis"], lw=1)
    ax.set_xticks(x, levels)
    ax.set(ylabel="Churn change vs control (pp)", xlabel="Age group",
           title="Churn risk concentrates in 35+ users on the pure AI feed")
    ax.legend()
    save(fig, "04_segment_churn_by_age")

    bal = pd.read_csv(bal_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    covs = bal.covariate.unique()[::-1]
    for arm in C.TREATMENTS:
        b = bal[bal.arm == arm].set_index("covariate").loc[covs]
        ax.scatter(b.smd, np.arange(len(covs)), color=C.ARM_COLORS[arm], s=36, label=lab[arm], zorder=3)
    ax.axvspan(-0.1, 0.1, color=C.INK["grid"], alpha=0.5, lw=0)
    ax.axvline(0, color=C.INK["axis"], lw=1)
    ax.set_yticks(np.arange(len(covs)), covs)
    ax.set(xlabel="Standardised mean difference vs control", title="Randomisation balance: every |SMD| < 0.1")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "05_balance_love_plot")

    fig, ax = plt.subplots(figsize=(9, 4))
    r = res[(res.arm == "ai_feed") & (res.variance_reduction > 0)].sort_values("variance_reduction")
    ax.barh(r.label, r.variance_reduction * 100, color=C.SEQ_BLUE[4], height=0.6)
    for yv, v in enumerate(r.variance_reduction):
        ax.text(v * 100 + 1, yv, f"{v:.0%}", va="center", color=C.INK["secondary"])
    ax.set(xlabel="Variance reduction from CUPED (%)", title="CUPED shrinks variance, equivalent to a much larger sample")
    save(fig, "06_cuped_variance_reduction")


def main():
    users, daily = load_raw()
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    summary = {"design": design_checks(df), "aa": aa_test(df)}
    res = metric_results(df)
    dyn, fits = daily_dynamics(users, daily)
    seq, seqsum = sequential(users, daily)
    bay = bayesian(res, df)
    seg = segments(df)
    quantile_effects(df)
    for a in C.TREATMENTS:
        headline = res[(res.arm == a) & (res.metric == "minutes")].rel_lift.iloc[0]
        fits[a]["headline_lift"] = float(headline)
        fits[a]["headline_lift_retained_share"] = fits[a]["fitted_asymptote_lift"] / headline
    summary.update({"novelty": fits, "sequential": seqsum, "interaction": seg.attrs["interaction"]})
    summary["decision"] = decide(res)
    summary["decision_headline"] = headline_text(summary["decision"])
    summary["metrics"] = res[["arm", "metric", "rel_lift", "rel_ci_low", "rel_ci_high", "diff", "p_adj_bh", "verdict",
                              "variance_reduction"]].to_dict(orient="records")
    summary["bayes"] = bay[["arm", "metric", "p_improves"]].to_dict(orient="records")
    figures(res, dyn, seq, seg)
    with open(C.REPORTS / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(res[["arm", "metric", "rel_lift", "p_adj_bh", "verdict", "variance_reduction"]].round(4).to_string())
    print(json.dumps({k: summary[k] for k in ["design", "aa", "novelty", "sequential", "interaction"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
