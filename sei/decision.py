"""Launch decision rule.

A guardrail passes only if its adjusted CI rules out harm beyond the margin. An arm is DO NOT LAUNCH if the
primary metric doesn't win or a critical guardrail doesn't pass, CONDITIONAL if a standard guardrail doesn't pass,
and LAUNCH otherwise.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from . import config as C

RANK = {"LAUNCH": 2, "CONDITIONAL": 1, "DO NOT LAUNCH": 0}


def guardrail_status(rel: float, lo95: float, hi95: float, direction: str, margin: float, z_adj: float) -> tuple[str, float]:
    se = (hi95 - lo95) / (2 * stats.norm.ppf(0.975))
    lo, hi = rel - z_adj * se, rel + z_adj * se
    harm_bound, far_bound = (hi, lo) if direction == "increase" else (-lo, -hi)
    if harm_bound < margin:
        return "PASS", harm_bound
    if far_bound > margin:
        return "FAIL", harm_bound
    return "INCONCLUSIVE", harm_bound


def decide(results: pd.DataFrame, arms: list[str] | None = None) -> dict:
    """results: metric_results-shaped frame (arm, metric, rel_lift, rel_ci_low, rel_ci_high, p_adj_bh)."""
    arms = arms or sorted(results.arm.unique())
    n_tests = len(C.GUARDRAILS) * len(arms)
    z_adj = stats.norm.ppf(1 - C.GUARDRAIL_FAMILY_ALPHA / n_tests)
    out = {"arms": {}, "rule": "guardrail non-inferiority, Bonferroni-adjusted", "z_guardrail": z_adj, "n_guardrail_tests": n_tests}
    for arm in arms:
        r = results[results.arm == arm].set_index("metric")
        p = r.loc[C.PRIMARY_METRIC]
        primary_win = bool(p.p_adj_bh < C.ALPHA and p.rel_lift > 0)
        guards = {}
        for metric, (direction, margin, severity) in C.GUARDRAILS.items():
            g = r.loc[metric]
            status, bound = guardrail_status(g.rel_lift, g.rel_ci_low, g.rel_ci_high, direction, margin, z_adj)
            guards[metric] = {"status": status, "severity": severity, "margin": margin, "direction": direction,
                              "rel_change": float(g.rel_lift), "harm_bound": float(bound)}
        critical_ok = all(v["status"] == "PASS" for v in guards.values() if v["severity"] == "critical")
        standard_issues = [m for m, v in guards.items() if v["severity"] == "standard" and v["status"] != "PASS"]
        if not primary_win or not critical_ok:
            state = "DO NOT LAUNCH"
        elif standard_issues:
            state = "CONDITIONAL"
        else:
            state = "LAUNCH"
        out["arms"][arm] = {"state": state, "primary_win": primary_win, "primary_lift": float(p.rel_lift),
                            "guardrails": guards, "blocking": [m for m, v in guards.items()
                                                               if v["severity"] == "critical" and v["status"] != "PASS"],
                            "mitigations_required": standard_issues}
    best = max(arms, key=lambda a: (RANK[out["arms"][a]["state"]], out["arms"][a]["primary_lift"]))
    out["recommended_arm"] = best if out["arms"][best]["state"] != "DO NOT LAUNCH" else None
    out["recommendation_state"] = out["arms"][best]["state"]
    return out


LABEL = {"minutes": "daily minutes", "churned": "churn", "friend_interactions": "friend interactions",
         "neg_feedback_rate": "negative feedback", "ad_ctr": "ad CTR"}


def headline(d: dict) -> str:
    """One-sentence recommendation generated from the decision states."""
    arm = d["recommended_arm"]
    if arm is None:
        return "Do not launch any variant: no arm wins on the primary metric with every critical guardrail passing."
    a = d["arms"][arm]
    name = C.ARM_LABELS[arm]
    if a["state"] == "LAUNCH":
        return f"Launch {name}: primary metric wins and every guardrail passes its non-inferiority margin."
    issues = ", ".join(f"{LABEL[m]} ({a['guardrails'][m]['rel_change']:+.1%}, {a['guardrails'][m]['status'].lower()})"
                       for m in a["mitigations_required"])
    return (f"Conditional launch of {name}: it wins on {LABEL[C.PRIMARY_METRIC]} and passes the critical churn "
            f"guardrail, but fails {issues}; ship only with a mitigation and a guardrail monitor on the ramp.")
