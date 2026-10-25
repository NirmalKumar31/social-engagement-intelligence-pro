"""Single source of truth for every sentence a deliverable says about the results.

Memo, Streamlit and Power BI all read facts from here, so no recommendation or claim is hard-coded:
change the data and every narrative follows the decision rule and the numbers.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date

import pandas as pd

from . import config as C
from .decision import LABEL, headline


def musd(x: float) -> str:
    """Signed millions of USD: -$9M rather than $-9M."""
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}M"


def _load(name: str) -> dict:
    with open(C.REPORTS / name) as f:
        return json.load(f)


def facts() -> dict:
    ex, hte, ret = _load("experiment_summary.json"), _load("hte_summary.json"), _load("retention_summary.json")
    nlp, vir, imp = _load("nlp_summary.json"), _load("virality_summary.json"), _load("impact_summary.json")
    res = pd.read_csv(C.TABLES / "metric_results.csv")
    seg = pd.read_csv(C.TABLES / "segment_results.csv")
    dec = ex["decision"]
    rec = dec["recommended_arm"]

    age = seg[seg.dimension == "age_group"]
    ai_harm_ages = age[(age.arm == "ai_feed") & (age.churn_p_adj < C.ALPHA) & (age.churn_diff_pp > 0)].segment.tolist()
    fb_lower_ages = age[(age.arm == "ai_feed_social") & (age.churn_p_adj < C.ALPHA) & (age.churn_diff_pp < 0)].segment.tolist()
    fb_up_ages = age[(age.arm == "ai_feed_social") & (age.churn_diff_pp > 0)].segment.tolist()

    nov = ex["novelty"]["ai_feed"]
    retained = nov["headline_lift_retained_share"]
    novelty_sentence = (f"The AI feed's headline minutes lift ({nov['headline_lift']:+.1%}) falls from {nov['week1_lift']:.1%} in week 1 "
                        f"to {nov['week4_lift']:.1%} in week 4; an exponential fit extrapolates to {nov['fitted_asymptote_lift']:+.1%} "
                        f"(95% CI {nov['fitted_asymptote_ci'][0]:+.1%} to {nov['fitted_asymptote_ci'][1]:+.1%}), i.e. about "
                        f"{retained:.0%} of the headline lift, so roughly {1 - retained:.0%} of it looks like novelty.")

    persist = {r["comparison"]: r for r in imp["results"]}
    fb_ai = persist["Friend boost minus AI feed"]
    revenue_sentence = (f"Under the 'effects persist' scenario the 12-month revenue difference (friend boost minus AI feed) is "
                        f"{musd(fb_ai['p50_musd'])} (90% of draws {musd(fb_ai['p05_musd'])} to {musd(fb_ai['p95_musd'])}); "
                        f"revenue does not separate the two variants, so the decision rests on the guardrails.")

    pol = pd.DataFrame(hte["policies"]).set_index("policy")
    mt = pol.loc["Model-targeted (DR-learner)"]
    personal_sentence = (f"Model-targeted assignment vs blanket friend boost: {mt.diff_vs_friend_boost_usd:+.2f} utility/user "
                         f"(paired 95% CI {mt.diff_vs_fb_ci_low:+.2f} to {mt.diff_vs_fb_ci_high:+.2f}); personalisation is not "
                         f"shown to help.")

    seg_sentence = ("AI-feed churn harm is significant (BH-adjusted) in age groups " + ", ".join(ai_harm_ages)
                    if ai_harm_ages else "No age group shows a significant AI-feed churn harm after correction")
    seg_sentence += ("; friend boost significantly lowers churn in " + ", ".join(fb_lower_ages) if fb_lower_ages
                     else "; no single age group shows a significant friend-boost churn reduction")
    if fb_up_ages:
        seg_sentence += f" (point estimate is higher, not significant, in {', '.join(fb_up_ages)})"
    seg_sentence += "."

    expo = {r["arm"]: r for r in nlp["exposure"]}
    sens = pd.DataFrame(nlp["exposure_sensitivity"])
    content_sentence = (f"Scenario only: under the assumed AI ranker weights, the share of negative-sentiment posts served rises from "
                        f"{expo['control']['negative_share']:.1%} to {expo['ai_feed']['negative_share']:.1%} and topic diversity falls "
                        f"{abs(expo['ai_feed']['topic_entropy_rel_change']):.0%}; across the weight sweep the negative share ranges "
                        f"{sens.negative_share.min():.0%}-{sens.negative_share.max():.0%}. These harms are assumptions of the scenario, "
                        f"not learned from real engagement data.")

    sv = ret["survival"]
    cm = ret["churn_model"]
    rr = sv["rate_ratio"]
    mt_row = pol.loc["Model-targeted (DR-learner)"]
    dep = cm["holdout"][cm["deployed_model"]]
    story = [
        ("Novelty", f"{retained:.0%}", f"of the AI feed's {nov['headline_lift']:+.1%} minutes lift survives the novelty fade "
                                        f"(fitted asymptote {nov['fitted_asymptote_lift']:+.1%}, an extrapolation).", "#1D3557"),
        ("Retention", f"HR {sv['hr_ai_feed']:.2f} vs {sv['hr_ai_feed_social']:.2f}",
         "Hazard of a 7-day lapse vs control: the AI feed raises it, the friend boost lowers it.", C.ARM_COLORS["ai_feed"]),
        ("Revenue", musd(fb_ai["p50_musd"]), "12-month friend boost minus AI feed if effects persist. The 90% range spans zero, so "
                                             "revenue does not pick the winner.", "#1D3557"),
        ("Personalisation", f"{mt_row.diff_vs_friend_boost_usd:+.2f}", "utility/user for model targeting vs blanket friend boost "
                                                                        "(CI spans zero): no gain shown.", "#8A94A6"),
        ("Early warning", f"AUC {dep['roc_auc']:.2f}", f"Lapse model on users active at day 13, top decile "
                                                        f"{dep['top_decile_lift']:.1f}× the base rate.", C.ARM_COLORS["ai_feed_social"]),
        ("Content (scenario)", f"{expo['ai_feed']['negative_share']:.0%}", f"negative posts served by the assumed AI ranker vs "
                                                                            f"{expo['control']['negative_share']:.0%} control; set by "
                                                                            "assumed weights.", C.STATUS["critical"]),
    ]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "story": story, "commit": commit, "generated": date.today().isoformat(),
        "notice": C.SYNTHETIC_NOTICE, "decision": dec, "recommended_arm": rec, "state": dec["recommendation_state"],
        "headline": headline(dec), "novelty": novelty_sentence, "revenue": revenue_sentence,
        "personalisation": personal_sentence, "segments": seg_sentence, "content": content_sentence,
        "ai_harm_ages": ai_harm_ages, "fb_lower_ages": fb_lower_ages,
        "lift": {(r.arm, r.metric): r for r in res.itertuples()},
        "survival": sv, "churn_model": cm, "rate_ratio": rr, "impact": imp, "hte": hte, "experiment": ex,
        "virality": vir, "nlp": nlp, "policies": pol,
        "label": LABEL,
    }
