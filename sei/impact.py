"""12-month revenue scenario model under explicit assumptions.

Decomposition (one coherent endpoint, the 7-day lapse from retention.py):
    monthly revenue of a cohort = users not yet lapsed (t)  x  revenue per not-lapsed user-month
  * revenue rate   = ad revenue / person-days at risk, post-novelty window (days 14-27), counting only
                     days before a user's first lapse. Treatment changes it by (1 + rev_lift).
  * lapse hazard   = lapses / person-days at risk over days 0-21; treatment multiplies it by the rate ratio.
Both parameters come from the SAME bootstrap resample of users per draw, with the control arm resampled
once and shared by both contrasts, so their joint and shared-control dependence is kept.

Transport assumptions (not identified by a 4-week test and shown as scenarios, not as uncertainty):
  * a monthly base lapse rate for the full population (config.MONTHLY_BASE_CHURN),
  * whether the 4-week effects persist, half-decay or fade by month 12.
"Share of draws > 0" is a fraction of simulations under these assumptions, not an empirical probability.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .features import load_raw
from .plotting import plt, save, setup
from .retention import LAST_START, activity_matrix, first_lapse_start

MONTH_DAYS = 30.4
MONTHS = 12
PERSISTENCE = {"persist": lambda t: np.ones_like(t, float),
               "half-decay by month 12": lambda t: 1 - 0.5 * t / (MONTHS - 1),
               "fade to zero by month 12": lambda t: 1 - t / (MONTHS - 1)}


def per_user_inputs(users: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    act = activity_matrix(users, daily)
    start = first_lapse_start(act)
    T = np.where(start >= 0, start, LAST_START).clip(min=0.5)
    end = np.where(start >= 0, start, C.EXP_DAYS)        # at risk on days < first lapse start
    lo = C.EXP_DAYS // 2
    risk_days = np.clip(end - lo, 0, C.EXP_DAYS - lo)
    rev = np.zeros((len(users), C.EXP_DAYS))
    pos = pd.Series(np.arange(len(users)), index=users.user_id.values)
    e = daily[daily.day >= 0]
    np.add.at(rev, (pos.reindex(e.user_id.values).values, e.day.values), e.ad_revenue.values)
    day = np.arange(C.EXP_DAYS)
    mask = (day[None, :] >= lo) & (day[None, :] < end[:, None])
    return pd.DataFrame({"arm": users.arm.values, "event": (start >= 0).astype(int), "T": T,
                         "risk_days": risk_days, "risk_revenue": (rev * mask).sum(axis=1)})


def estimates(u: pd.DataFrame, sel: dict[str, np.ndarray]) -> dict:
    g = {a: u.iloc[ix] for a, ix in sel.items()}
    rate = {a: g[a].risk_revenue.sum() / g[a].risk_days.sum() for a in C.ARMS}
    haz = {a: g[a].event.sum() / g[a]["T"].sum() for a in C.ARMS}
    return {"base_rate": rate["control"],
            **{a: {"rev_lift": rate[a] / rate["control"] - 1, "rate_ratio": haz[a] / haz["control"]} for a in C.TREATMENTS}}


def project(rev_lift, rr, rev_month, persistence="persist", base_churn=C.MONTHLY_BASE_CHURN, pop=C.MAU_POPULATION):
    """Cumulative incremental-able revenue path (USD) for a cohort of `pop` users. Vectorised over draws."""
    t = np.arange(MONTHS)
    k = PERSISTENCE[persistence](t)[None, :]
    lift = 1 + np.atleast_1d(rev_lift)[:, None] * k
    rr_t = 1 + (np.atleast_1d(rr)[:, None] - 1) * k
    m = 1 - (1 - base_churn) ** rr_t
    surv = np.cumprod(np.c_[np.ones((m.shape[0], 1)), 1 - m[:, :-1]], axis=1)
    return (pop * surv * rev_month * lift).cumsum(axis=1)


def main(n_boot: int = 1000):
    users, daily = load_raw()
    u = per_user_inputs(users, daily)
    idx = {a: np.flatnonzero(u.arm.values == a) for a in C.ARMS}
    point = estimates(u, idx)
    rng = np.random.default_rng(C.SEED)
    draws = [estimates(u, {a: rng.choice(ix, len(ix)) for a, ix in idx.items()}) for _ in range(n_boot)]
    rev0 = point["base_rate"] * MONTH_DAYS
    par = {a: {"rev_lift": np.array([d[a]["rev_lift"] for d in draws]), "rate_ratio": np.array([d[a]["rate_ratio"] for d in draws])}
           for a in C.TREATMENTS}
    pd.DataFrame({f"{a}_{k}": par[a][k] for a in C.TREATMENTS for k in ("rev_lift", "rate_ratio")}).to_csv(
        C.TABLES / "impact_param_draws.csv", index=False)
    corr = {a: float(np.corrcoef(par[a]["rev_lift"], par[a]["rate_ratio"])[0, 1]) for a in C.TREATMENTS}
    cross = float(np.corrcoef(par["ai_feed"]["rev_lift"], par["ai_feed_social"]["rev_lift"])[0, 1])
    base = lambda pers: project(np.zeros(1), np.ones(1), rev0, pers)[0]

    scen_rows, fan, hist, cum = [], [], [], {}
    for pers in PERSISTENCE:
        for a in C.TREATMENTS:
            cum[(pers, a)] = project(par[a]["rev_lift"], par[a]["rate_ratio"], rev0, pers) - base(pers)
        diff = cum[(pers, "ai_feed_social")][:, -1] - cum[(pers, "ai_feed")][:, -1]
        for name, arr in [("AI feed vs control", cum[(pers, "ai_feed")][:, -1]),
                          ("AI + friend boost vs control", cum[(pers, "ai_feed_social")][:, -1]),
                          ("Friend boost minus AI feed", diff)]:
            v = arr / 1e6
            scen_rows.append({"persistence": pers, "comparison": name, "p05_musd": np.quantile(v, 0.05),
                              "p50_musd": np.median(v), "p95_musd": np.quantile(v, 0.95), "mean_musd": v.mean(),
                              "prob_positive": (v > 0).mean()})
    scen = pd.DataFrame(scen_rows)
    scen.to_csv(C.TABLES / "impact_scenarios.csv", index=False)
    res = scen[scen.persistence == "persist"].drop(columns="persistence")
    res.to_csv(C.TABLES / "impact_monte_carlo.csv", index=False)

    for a in C.TREATMENTS:
        c = cum[("persist", a)] / 1e6
        fan += [{"arm": a, "month": mth + 1, "p05": np.quantile(c[:, mth], 0.05), "p50": np.median(c[:, mth]),
                 "p95": np.quantile(c[:, mth], 0.95)} for mth in range(MONTHS)]
    fan = pd.DataFrame(fan)
    fan.to_csv(C.TABLES / "impact_fan.csv", index=False)
    finals = {a: cum[("persist", a)][:, -1] / 1e6 for a in C.TREATMENTS}
    edges = np.linspace(min(np.quantile(v, 0.001) for v in finals.values()), max(np.quantile(v, 0.999) for v in finals.values()), 41)
    for a, v in finals.items():
        cnt, _ = np.histogram(v, bins=edges)
        hist += [{"arm": a, "bin_mid_musd": (edges[i] + edges[i + 1]) / 2, "share_of_draws": c / len(v)} for i, c in enumerate(cnt)]
    pd.DataFrame(hist).to_csv(C.TABLES / "impact_distribution.csv", index=False)

    # Sensitivity: one-way ranges plus a joint adverse stress case (marginal bounds, not a probability)
    q = lambda a, k, p: float(np.quantile(par[a][k], p))

    def fb_minus_ai(la=None, lb=None, ra=None, rb=None, churn=C.MONTHLY_BASE_CHURN, pop=C.MAU_POPULATION, rev=rev0,
                    pers="persist"):
        A, B = point["ai_feed"], point["ai_feed_social"]
        va = project(np.array([A["rev_lift"] if la is None else la]), np.array([A["rate_ratio"] if ra is None else ra]),
                     rev, pers, churn, pop)[0, -1]
        vb = project(np.array([B["rev_lift"] if lb is None else lb]), np.array([B["rate_ratio"] if rb is None else rb]),
                     rev, pers, churn, pop)[0, -1]
        bl = project(np.zeros(1), np.ones(1), rev, pers, churn, pop)[0, -1]
        return ((vb - bl) - (va - bl)) / 1e6
    central = fb_minus_ai()
    sens = [
        ("Monthly base lapse 2% / 6%", fb_minus_ai(churn=0.02), fb_minus_ai(churn=0.06)),
        ("Eligible MAU 20M / 60M", fb_minus_ai(pop=20e6), fb_minus_ai(pop=60e6)),
        ("Revenue per user -30% / +30%", fb_minus_ai(rev=rev0 * 0.7), fb_minus_ai(rev=rev0 * 1.3)),
        ("AI-feed revenue lift 2.5% / 97.5%", fb_minus_ai(la=q("ai_feed", "rev_lift", .025)), fb_minus_ai(la=q("ai_feed", "rev_lift", .975))),
        ("AI-feed lapse rate ratio 2.5% / 97.5%", fb_minus_ai(ra=q("ai_feed", "rate_ratio", .025)), fb_minus_ai(ra=q("ai_feed", "rate_ratio", .975))),
        ("Friend-boost lapse rate ratio 2.5% / 97.5%", fb_minus_ai(rb=q("ai_feed_social", "rate_ratio", .025)), fb_minus_ai(rb=q("ai_feed_social", "rate_ratio", .975))),
        ("Effects persist / fade by month 12", fb_minus_ai(pers="persist"), fb_minus_ai(pers="fade to zero by month 12")),
    ]
    sens = pd.DataFrame([{"assumption": n, "low_musd": lo, "high_musd": hi, "central_musd": central, "swing": abs(hi - lo)}
                         for n, lo, hi in sens]).sort_values("swing")
    sens.to_csv(C.TABLES / "impact_sensitivity.csv", index=False)
    stress = fb_minus_ai(la=q("ai_feed", "rev_lift", .975), lb=q("ai_feed_social", "rev_lift", .025),
                         ra=q("ai_feed", "rate_ratio", .025), rb=q("ai_feed_social", "rate_ratio", .975), churn=0.02)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for a in C.TREATMENTS:
        f = fan[fan.arm == a]
        axes[0].fill_between(f.month, f.p05, f.p95, color=C.ARM_COLORS[a], alpha=0.15, lw=0)
        axes[0].plot(f.month, f.p50, color=C.ARM_COLORS[a], lw=2.2, label=C.ARM_LABELS[a])
        axes[0].annotate(f"${f.p50.iloc[-1]:,.0f}M", (12, f.p50.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                         va="center", color=C.INK["secondary"])
    axes[0].axhline(0, color=C.INK["axis"], lw=1)
    axes[0].set(xlabel="Months after rollout", ylabel="Cumulative incremental ad revenue ($M)",
                title="Scenario: effects persist (median, 90% band of parameter draws)")
    axes[0].legend(fontsize=8, loc="upper left")
    yy = np.arange(len(sens))
    axes[1].barh(yy, sens.high_musd - sens.central_musd, left=sens.central_musd, color=C.SEQ_BLUE[4], height=0.5)
    axes[1].barh(yy, sens.low_musd - sens.central_musd, left=sens.central_musd, color=C.SEQ_BLUE[1], height=0.5)
    axes[1].axvline(central, color=C.INK["secondary"], lw=1)
    axes[1].axvline(0, color=C.STATUS["critical"], lw=1)
    axes[1].scatter([stress], [len(sens) - 0.5], marker="v", color=C.STATUS["critical"], zorder=3)
    axes[1].annotate(f"joint adverse stress: ${stress:,.0f}M", (stress, len(sens) - 0.5), xytext=(6, 4),
                     textcoords="offset points", fontsize=8, color=C.INK["secondary"])
    axes[1].set_yticks(yy, sens.assumption, fontsize=8)
    axes[1].set(xlabel="Friend boost minus AI feed, 12-month value ($M)", title="Sensitivity (one-way) and a joint stress case")
    fig.tight_layout()
    save(fig, "17_business_impact")

    out = {"method": __doc__.strip(), "point": point, "rev_per_at_risk_user_month": rev0,
           "param_ci": {a: {k: [q(a, k, .025), q(a, k, .975)] for k in ("rev_lift", "rate_ratio")} for a in C.TREATMENTS},
           "param_correlation_rev_vs_hazard": corr, "cross_arm_rev_lift_correlation": cross,
           "results": res.to_dict(orient="records"), "scenarios": scen.to_dict(orient="records"),
           "sensitivity": sens.to_dict(orient="records"), "joint_stress_fb_minus_ai_musd": stress, "n_boot": n_boot}
    with open(C.REPORTS / "impact_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ["point", "param_ci", "param_correlation_rev_vs_hazard", "cross_arm_rev_lift_correlation",
                                          "joint_stress_fb_minus_ai_musd"]}, indent=1, default=float))
    print(scen.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
