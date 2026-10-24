"""Executive decision memo.

Always writes a deterministic memo built directly from the analysis outputs (so every number is
traceable). With `--llm`, Claude rewrites it into a polished narrative; if no credentials are
configured, or the call fails, the deterministic memo stands on its own.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config as C


def _load(name):
    with open(C.REPORTS / name) as f:
        return json.load(f)


def _m(metrics, arm, metric):
    return next(r for r in metrics if r["arm"] == arm and r["metric"] == metric)


def build_memo() -> tuple[str, dict]:
    ex, hte, ret = _load("experiment_summary.json"), _load("hte_summary.json"), _load("retention_summary.json")
    nlp, vir, imp = _load("nlp_summary.json"), _load("virality_summary.json"), _load("impact_summary.json")
    M = ex["metrics"]
    a_min, b_min = _m(M, "ai_feed", "minutes"), _m(M, "ai_feed_social", "minutes")
    a_ch, b_ch = _m(M, "ai_feed", "churned"), _m(M, "ai_feed_social", "churned")
    a_fi, b_fi = _m(M, "ai_feed", "friend_interactions"), _m(M, "ai_feed_social", "friend_interactions")
    a_rev, b_rev = _m(M, "ai_feed", "ad_revenue"), _m(M, "ai_feed_social", "ad_revenue")
    nov = ex["novelty"]
    pol = pd.DataFrame(hte["policies"]).set_index("policy")
    mc = pd.DataFrame(imp["results"]).set_index("comparison")
    expo = pd.DataFrame(nlp["exposure"]).set_index("arm")
    seg = pd.read_csv(C.TABLES / "segment_results.csv")
    risk = seg[(seg.arm == "ai_feed") & (seg.status == "churn risk")]
    sm = pd.DataFrame(nlp["sentiment"]["metrics"]).set_index("model")
    cm = ret["churn_model"]["holdout"]["LightGBM + isotonic"]
    d = ex["design"]
    pb = pd.read_csv(C.TABLES / "virality_playbook.csv")
    wd = pb[pb.dimension == "weekday"].set_index("level")
    wk_lift = (wd.loc[["saturday", "sunday"], "viral_rate"] * wd.loc[["saturday", "sunday"], "articles"]).sum() / wd.loc[["saturday", "sunday"], "articles"].sum() / (
        (wd.drop(["saturday", "sunday"]).viral_rate * wd.drop(["saturday", "sunday"]).articles).sum() / wd.drop(["saturday", "sunday"]).articles.sum())
    ch = pb[pb.dimension == "channel"].set_index("level")
    sm_lift = ch.loc["Social Media", "viral_rate"] / ((ch.viral_rate * ch.articles).sum() / ch.articles.sum())
    pct = lambda x: f"{x * 100:+.1f}%"

    memo = f"""# Feed Ranking Experiment: Launch Decision Memo

**Recommendation: ship the AI feed + friend boost variant (`ai_feed_social`) to all eligible users, and do not
ship the pure AI feed.** Then run a follow-up on model-targeted arm assignment, which adds
${pol.loc['Model-targeted (DR-learner)', 'uplift_vs_control_usd'] - pol.loc['Ship AI + friend boost to all', 'uplift_vs_control_usd']:.2f}/user/yr on top
(within noise today, so it needs a dedicated test before it becomes policy).

## 1. Is the experiment trustworthy?
| Check | Result |
|---|---|
| Sample-ratio mismatch | χ² p = {d['srm']['p_value']:.3f} (no SRM) |
| Covariate balance | max \\|SMD\\| = {d['max_abs_smd']:.3f} (threshold 0.10) |
| A/A false-positive rate | {ex['aa']['aa_false_positive_rate']:.1%} over {ex['aa']['n_sims']:,} re-randomisations (target 5%) |
| Power | MDE {d['mde_rel_raw']:.1%} raw → **{d['mde_rel_cuped']:.1%} with CUPED** at {d['n_per_arm']:,} users/arm |
| Peeking risk | naive daily peeking inflates false positives to {ex['sequential']['peeking_fpr_naive']:.0%}; the mSPRT always-valid monitor keeps it at {ex['sequential']['peeking_fpr_msprt']:.1%} |

## 2. What happened?
| Metric (CUPED / delta-method, BH-adjusted) | AI feed | AI + friend boost |
|---|---|---|
| Daily minutes / user (primary) | {pct(a_min['rel_lift'])} | {pct(b_min['rel_lift'])} |
| Steady-state minutes lift (after novelty) | {pct(nov['ai_feed']['steady_state_lift'])} | {pct(nov['ai_feed_social']['steady_state_lift'])} |
| Ad revenue / user | {pct(a_rev['rel_lift'])} | {pct(b_rev['rel_lift'])} |
| 28-day churn (guardrail) | **{pct(a_ch['rel_lift'])}** ({a_ch['verdict']}) | **{pct(b_ch['rel_lift'])}** ({b_ch['verdict']}) |
| Friend interactions (guardrail) | **{pct(a_fi['rel_lift'])}** ({a_fi['verdict']}) | {pct(b_fi['rel_lift'])} ({b_fi['verdict']}) |
| Cox hazard ratio for churn | {ret['survival']['hr_ai_feed']:.2f} | {ret['survival']['hr_ai_feed_social']:.2f} |

- **The AI feed's headline lift is inflated by novelty.** Week-1 lift was {nov['ai_feed']['week1_lift']:.1%} and week-4 lift was
  {nov['ai_feed']['week4_lift']:.1%}; the fitted steady state is {nov['ai_feed']['steady_state_lift']:.1%}, with novelty decaying on a ~{nov['ai_feed']['decay_days']:.0f}-day time constant.
  Once novelty is stripped out, the AI feed beats the friend-boost variant by only {(nov['ai_feed']['steady_state_lift'] - nov['ai_feed_social']['steady_state_lift']) * 100:.1f} pp of time spent.
- **It buys that edge with retention.** Churn rises {a_ch['rel_lift']:.0%} on the AI feed, concentrated in
  {', '.join(risk[risk.dimension == 'age_group'].segment.tolist()) or 'older'} age groups (age × treatment interaction p = {ex['interaction']['churn_age_interaction_p']:.4f}).
  The friend boost *reduces* churn by {abs(b_ch['rel_lift']):.0%}.

## 3. Content health (responsible-AI guardrails)
On real posts, the engagement-optimised AI ranker cut feed topic diversity by {abs(expo.loc['ai_feed', 'topic_entropy_rel_change']):.0%} and raised
the share of negative-sentiment posts served from {expo.loc['control', 'negative_share']:.1%} to {expo.loc['ai_feed', 'negative_share']:.1%}.
The friend boost halves both harms ({expo.loc['ai_feed_social', 'negative_share']:.1%} negative share). Sentiment is scored by our TweetEval model
(macro-recall {sm.loc['TF-IDF + LogReg (ours)', 'macro_recall']:.3f} vs VADER {sm.loc['VADER (lexicon baseline)', 'macro_recall']:.3f}).

## 4. What is it worth?
12-month Monte Carlo ({C.MC_DRAWS:,} draws, {C.MAU_POPULATION / 1e6:.0f}M eligible MAU, compounding churn):
| Rollout | Median incremental revenue | 90% interval |
|---|---|---|
| AI feed | ${mc.loc['AI feed vs control', 'p50_musd']:,.0f}M | ${mc.loc['AI feed vs control', 'p05_musd']:,.0f}M – ${mc.loc['AI feed vs control', 'p95_musd']:,.0f}M |
| AI + friend boost | **${mc.loc['AI + friend boost vs control', 'p50_musd']:,.0f}M** | ${mc.loc['AI + friend boost vs control', 'p05_musd']:,.0f}M – ${mc.loc['AI + friend boost vs control', 'p95_musd']:,.0f}M |

P(friend boost beats AI feed over 12 months) = **{mc.loc['Friend boost minus AI feed', 'prob_positive']:.2%}**. No single assumption in the sensitivity
analysis flips the decision.

## 5. Personalisation and retention levers
- DR-learner CATE model: model-targeted assignment is worth ${pol.loc['Model-targeted (DR-learner)', 'uplift_vs_control_usd']:.2f}/user/yr vs
  ${pol.loc['Ship AI + friend boost to all', 'uplift_vs_control_usd']:.2f} for the blanket friend-boost launch (IPW, held-out users). Effects of the AI feed are strongly
  heterogeneous; the friend boost's are nearly uniform, so blanket rollout is low-regret.
- Churn early-warning model (leakage-free, scored at day 13): ROC-AUC {cm['roc_auc']:.3f}, top-decile lift {cm['top_decile_lift']:.1f}x.
  Targeting the top {ret['churn_model']['campaign_optimal_fraction']:.0%} riskiest users with a retention nudge nets ≈${ret['churn_model']['campaign_optimal_profit_musd']:.1f}M at scale.
- Creator playbook (real data, {vir['n_articles']:,} articles, time-split AUC {vir['auc_lightgbm']:.3f}): weekend posts go viral at
  {wk_lift:.1f}x the weekday rate and Social Media-channel posts at {sm_lift:.1f}x the overall rate. Surface this in creator tooling.

## 6. Next steps
1. Roll out `ai_feed_social` at 10% → 50% → 100% with the churn and friend-interaction guardrails on an always-valid monitor.
2. Holdback 2% on control for 90 days to measure long-run retention directly (validates the Monte Carlo).
3. A/B test the DR-learner arm-assignment policy against blanket friend boost.
4. Add a negativity-exposure cap to the ranker objective; re-measure diversity.
"""
    return memo, {"experiment": ex, "hte": hte, "retention": ret, "nlp": nlp, "virality": vir, "impact": imp}


def llm_polish(memo: str) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=("You are a principal product data scientist writing for a VP of Product. Rewrite the memo you are "
                    "given into a crisp one-page executive brief in Markdown. Keep every number exactly as given, do "
                    "not invent figures, keep the recommendation, and lead with the decision and the single most "
                    "important reason."),
            messages=[{"role": "user", "content": memo}],
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        print(f"LLM polish skipped (credentials): {e}")
        return None
    except anthropic.APIConnectionError as e:
        print(f"LLM polish skipped (network): {e}")
        return None
    except anthropic.APIStatusError as e:
        print(f"LLM polish skipped (API {e.status_code}): {e}")
        return None
    except TypeError as e:  # no credentials resolved at all
        print(f"LLM polish skipped: {e}")
        return None
    if response.stop_reason == "refusal":
        return None
    return "".join(b.text for b in response.content if b.type == "text") or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also write a Claude-polished brief")
    args = ap.parse_args()
    memo, _ = build_memo()
    (C.REPORTS / "EXECUTIVE_MEMO.md").write_text(memo)
    print(memo)
    if args.llm:
        polished = llm_polish(memo)
        if polished:
            (C.REPORTS / "EXECUTIVE_BRIEF_AI.md").write_text(polished)
            print("wrote reports/EXECUTIVE_BRIEF_AI.md")


if __name__ == "__main__":
    main()
