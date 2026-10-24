# Feed Ranking Experiment: Launch Decision Memo

**Recommendation: ship the AI feed + friend boost variant (`ai_feed_social`) to all eligible users, and do not
ship the pure AI feed.** Then run a follow-up on model-targeted arm assignment, which adds
$0.14/user/yr on top
(within noise today, so it needs a dedicated test before it becomes policy).

## 1. Is the experiment trustworthy?
| Check | Result |
|---|---|
| Sample-ratio mismatch | χ² p = 0.099 (no SRM) |
| Covariate balance | max \|SMD\| = 0.019 (threshold 0.10) |
| A/A false-positive rate | 5.1% over 1,000 re-randomisations (target 5%) |
| Power | MDE 2.1% raw → **1.2% with CUPED** at 19,756 users/arm |
| Peeking risk | naive daily peeking inflates false positives to 27%; the mSPRT always-valid monitor keeps it at 0.9% |

## 2. What happened?
| Metric (CUPED / delta-method, BH-adjusted) | AI feed | AI + friend boost |
|---|---|---|
| Daily minutes / user (primary) | +11.6% | +7.9% |
| Steady-state minutes lift (after novelty) | +7.7% | +6.7% |
| Ad revenue / user | +11.5% | +8.1% |
| 28-day churn (guardrail) | **+17.0%** (harm) | **-11.1%** (win) |
| Friend interactions (guardrail) | **-9.0%** (harm) | +10.7% (win) |
| Cox hazard ratio for churn | 1.19 | 0.88 |

- **The AI feed's headline lift is inflated by novelty.** Week-1 lift was 17.4% and week-4 lift was
  7.5%; the fitted steady state is 7.7%, with novelty decaying on a ~7-day time constant.
  Once novelty is stripped out, the AI feed beats the friend-boost variant by only 0.9 pp of time spent.
- **It buys that edge with retention.** Churn rises 17% on the AI feed, concentrated in
  35-44, 45-54, 55+ age groups (age × treatment interaction p = 0.0025).
  The friend boost *reduces* churn by 11%.

## 3. Content health (responsible-AI guardrails)
On real posts, the engagement-optimised AI ranker cut feed topic diversity by 43% and raised
the share of negative-sentiment posts served from 18.9% to 36.7%.
The friend boost halves both harms (24.6% negative share). Sentiment is scored by our TweetEval model
(macro-recall 0.628 vs VADER 0.570).

## 4. What is it worth?
12-month Monte Carlo (20,000 draws, 40M eligible MAU, compounding churn):
| Rollout | Median incremental revenue | 90% interval |
|---|---|---|
| AI feed | $57M | $39M – $74M |
| AI + friend boost | **$103M** | $87M – $119M |

P(friend boost beats AI feed over 12 months) = **99.97%**. No single assumption in the sensitivity
analysis flips the decision.

## 5. Personalisation and retention levers
- DR-learner CATE model: model-targeted assignment is worth $2.19/user/yr vs
  $2.05 for the blanket friend-boost launch (IPW, held-out users). Effects of the AI feed are strongly
  heterogeneous; the friend boost's are nearly uniform, so blanket rollout is low-regret.
- Churn early-warning model (leakage-free, scored at day 13): ROC-AUC 0.896, top-decile lift 6.9x.
  Targeting the top 9% riskiest users with a retention nudge nets ≈$2.5M at scale.
- Creator playbook (real data, 39,644 articles, time-split AUC 0.732): weekend posts go viral at
  1.5x the weekday rate and Social Media-channel posts at 1.5x the overall rate. Surface this in creator tooling.

## 6. Next steps
1. Roll out `ai_feed_social` at 10% → 50% → 100% with the churn and friend-interaction guardrails on an always-valid monitor.
2. Holdback 2% on control for 90 days to measure long-run retention directly (validates the Monte Carlo).
3. A/B test the DR-learner arm-assignment policy against blanket friend boost.
4. Add a negativity-exposure cap to the ranker objective; re-measure diversity.
