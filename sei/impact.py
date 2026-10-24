"""12-month business impact under uncertainty.

Short experiments measure a 28-day snapshot; the decision needs a 12-month view where churn
compounds. Every draw samples the steady-state revenue lift (post-novelty, CUPED) and the Cox
hazard ratio from their sampling distributions and projects a cohort forward month by month.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import stats as S
from .plotting import plt, save, setup

MONTH_DAYS = 30.4


def inputs(df: pd.DataFrame, hr: pd.DataFrame) -> dict:
    ctrl = df[df.arm == "control"]
    rev_alive = ctrl.loc[ctrl.churned == 0, "late_ad_revenue_per_day"].mean() * MONTH_DAYS
    out = {"rev_per_alive_user_month": rev_alive}
    hr = hr.set_index("covariate")
    for arm in C.TREATMENTS:
        t = df[df.arm == arm]
        r = S.cuped(t.late_ad_revenue_per_day.values, ctrl.late_ad_revenue_per_day.values,
                    t.pre_ad_revenue_per_day.values, ctrl.pre_ad_revenue_per_day.values)
        row = hr.loc[f"arm_{arm}"]
        out[arm] = {"rev_lift": r["rel_lift"], "rev_lift_se": (r["rel_ci_high"] - r["rel_ci_low"]) / (2 * S.Z),
                    "log_hr": np.log(row.hazard_ratio),
                    "log_hr_se": (np.log(row.hr_ci_high) - np.log(row.hr_ci_low)) / (2 * S.Z)}
    return out


def project(rev_lift, hr, rev_month, months=12, base_churn=C.MONTHLY_BASE_CHURN, pop=C.MAU_POPULATION):
    """Cumulative revenue (USD) by month for a cohort of `pop` users. Vectorised over draws."""
    m = 1 - (1 - base_churn) ** np.atleast_1d(hr)[:, None]
    t = np.arange(months)[None, :]
    surv = (1 - m) ** t
    monthly = pop * surv * rev_month * (1 + np.atleast_1d(rev_lift)[:, None])
    return monthly.cumsum(axis=1)


def main():
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    hr = pd.read_csv(C.TABLES / "cox_hazard_ratios.csv")
    inp = inputs(df, hr)
    rng = np.random.default_rng(C.SEED)
    rev0 = inp["rev_per_alive_user_month"]
    base = project(np.zeros(1), np.ones(1), rev0)[0]
    cum = {}
    for arm in C.TREATMENTS:
        a = inp[arm]
        lift = rng.normal(a["rev_lift"], a["rev_lift_se"], C.MC_DRAWS)
        h = np.exp(rng.normal(a["log_hr"], a["log_hr_se"], C.MC_DRAWS))
        cum[arm] = project(lift, h, rev0) - base
    rows = []
    for name, arr in [("AI feed vs control", cum["ai_feed"]), ("AI + friend boost vs control", cum["ai_feed_social"]),
                      ("Friend boost minus AI feed", cum["ai_feed_social"] - cum["ai_feed"])]:
        final = arr[:, -1] / 1e6
        rows.append({"comparison": name, "p05_musd": np.quantile(final, 0.05), "p50_musd": np.median(final),
                     "p95_musd": np.quantile(final, 0.95), "mean_musd": final.mean(), "prob_positive": (final > 0).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "impact_monte_carlo.csv", index=False)

    fan = []
    for arm in C.TREATMENTS:
        for mth in range(12):
            v = cum[arm][:, mth] / 1e6
            fan.append({"arm": arm, "month": mth + 1, "p05": np.quantile(v, 0.05), "p50": np.median(v), "p95": np.quantile(v, 0.95)})
    fan = pd.DataFrame(fan)
    fan.to_csv(C.TABLES / "impact_fan.csv", index=False)
    edges = np.linspace(min(np.quantile(cum[a][:, -1], 0.001) for a in C.TREATMENTS) / 1e6,
                        max(np.quantile(cum[a][:, -1], 0.999) for a in C.TREATMENTS) / 1e6, 61)
    hist = []
    for arm in C.TREATMENTS:
        cnt, _ = np.histogram(cum[arm][:, -1] / 1e6, bins=edges)
        hist += [{"arm": arm, "bin_mid_musd": (edges[i] + edges[i + 1]) / 2, "share_of_draws": c / C.MC_DRAWS}
                 for i, c in enumerate(cnt)]
    pd.DataFrame(hist).to_csv(C.TABLES / "impact_distribution.csv", index=False)

    # Sensitivity of the B-vs-A decision to each assumption (one at a time, low/high)
    def b_minus_a(base_churn=C.MONTHLY_BASE_CHURN, pop=C.MAU_POPULATION, rev=rev0, la=None, lb=None, ha=None, hb=None):
        A, B = inp["ai_feed"], inp["ai_feed_social"]
        va = project(np.array([la if la is not None else A["rev_lift"]]), np.array([ha if ha is not None else np.exp(A["log_hr"])]),
                     rev, base_churn=base_churn, pop=pop)[0, -1]
        vb = project(np.array([lb if lb is not None else B["rev_lift"]]), np.array([hb if hb is not None else np.exp(B["log_hr"])]),
                     rev, base_churn=base_churn, pop=pop)[0, -1]
        return (vb - va) / 1e6
    A, B = inp["ai_feed"], inp["ai_feed_social"]
    sens = []
    central = b_minus_a()
    for label, lo, hi in [
        ("Monthly base churn 2% / 6%", b_minus_a(base_churn=0.02), b_minus_a(base_churn=0.06)),
        ("Eligible MAU 20M / 60M", b_minus_a(pop=20e6), b_minus_a(pop=60e6)),
        ("Revenue per user -30% / +30%", b_minus_a(rev=rev0 * 0.7), b_minus_a(rev=rev0 * 1.3)),
        ("AI-feed revenue lift at 95% CI low / high", b_minus_a(la=A["rev_lift"] - S.Z * A["rev_lift_se"]),
         b_minus_a(la=A["rev_lift"] + S.Z * A["rev_lift_se"])),
        ("AI-feed churn HR at 95% CI low / high", b_minus_a(ha=np.exp(A["log_hr"] - S.Z * A["log_hr_se"])),
         b_minus_a(ha=np.exp(A["log_hr"] + S.Z * A["log_hr_se"]))),
        ("Friend-boost churn HR at 95% CI low / high", b_minus_a(hb=np.exp(B["log_hr"] - S.Z * B["log_hr_se"])),
         b_minus_a(hb=np.exp(B["log_hr"] + S.Z * B["log_hr_se"]))),
    ]:
        sens.append({"assumption": label, "low_musd": lo, "high_musd": hi, "central_musd": central, "swing": abs(hi - lo)})
    sens = pd.DataFrame(sens).sort_values("swing")
    sens.to_csv(C.TABLES / "impact_sensitivity.csv", index=False)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for arm in C.TREATMENTS:
        f = fan[fan.arm == arm]
        axes[0].fill_between(f.month, f.p05, f.p95, color=C.ARM_COLORS[arm], alpha=0.15, lw=0)
        axes[0].plot(f.month, f.p50, color=C.ARM_COLORS[arm], lw=2.2, label=C.ARM_LABELS[arm])
        axes[0].annotate(f"${f.p50.iloc[-1]:,.0f}M", (12, f.p50.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                         va="center", color=C.INK["secondary"])
    axes[0].axhline(0, color=C.INK["axis"], lw=1)
    axes[0].set(xlabel="Months after full rollout", ylabel="Cumulative incremental ad revenue ($M)",
                title="12-month projection (median, 90% band)")
    axes[0].legend(fontsize=8, loc="upper left")
    yy = np.arange(len(sens))
    axes[1].barh(yy, sens.high_musd - sens.central_musd, left=sens.central_musd, color=C.SEQ_BLUE[4], height=0.5)
    axes[1].barh(yy, sens.low_musd - sens.central_musd, left=sens.central_musd, color=C.SEQ_BLUE[1], height=0.5)
    axes[1].axvline(central, color=C.INK["secondary"], lw=1)
    axes[1].axvline(0, color=C.STATUS["critical"], lw=1)
    axes[1].set_yticks(yy, sens.assumption, fontsize=8)
    axes[1].set(xlabel="Friend boost minus AI feed, 12-month value ($M)", title="Sensitivity: what would flip the decision?")
    fig.tight_layout()
    save(fig, "17_business_impact")
    out = {"inputs": inp, "results": res.to_dict(orient="records"), "sensitivity": sens.to_dict(orient="records")}
    with open(C.REPORTS / "impact_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
