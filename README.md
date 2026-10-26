# Social Engagement Intelligence

Experiment analysis, causal ML and retention modelling for a feed-ranking launch decision.

> The experiment data is simulated (`sei/simulate.py`). Only the NLP and virality modules use real public data, so
> the numbers illustrate the methods, not a real product. An independent audit of an earlier version is in
> [`docs/AUDIT_2026-10-25.md`](docs/AUDIT_2026-10-25.md), with the fixes in
> [`docs/AUDIT_RESPONSE_2026-10-25.md`](docs/AUDIT_RESPONSE_2026-10-25.md).

## The question

A social app tested two AI-ranked feeds against its current feed on 60,000 users for 28 days. The pure AI "For You"
feed has the biggest headline lift. Should it ship?

The answer comes from a written decision rule ([`sei/decision.py`](sei/decision.py)):

- **Pure AI feed: do not launch.** It fails the churn guardrail (+17% final-week inactivity), and also friend
  interactions (−9%) and negative feedback (+19%).
- **AI feed + friend boost: conditional launch.** It wins on minutes (+7.9%) and passes churn (−11%) and friend
  interactions (+11%), but fails the negative-feedback guardrail (+8.1% against a 5% margin). It ships only with a
  mitigation and a guardrail monitor during the ramp.
- 12-month ad revenue doesn't separate the two: friend boost minus AI feed is −$9M (90% of draws −$32M to +$14M) if
  effects persist. The decision rests on the guardrails.
- Personalised assignment doesn't beat launching friend boost to everyone (−0.24 utility per user, 95% CI −1.46 to
  +0.91).

![Metric scorecard](reports/figures/01_metric_scorecard.png)

## What's in it

| Area | Methods | Result |
|---|---|---|
| Simulation | 42-day daily event log, user-level latent traits, AR(1) behaviour, novelty decay, churn hazard | 1.76M active user-days |
| Experiment checks | SRM, covariate balance, A/A re-randomisation, power | SRM p = 0.10, A/A false positives 5.1% |
| Effects | Welch SE, CUPED, delta method for ratios, quantile bootstrap, Benjamini-Hochberg | CUPED removes ~68% of variance |
| Decision rule | Guardrail non-inferiority margins, Bonferroni, LAUNCH / CONDITIONAL / DO NOT LAUNCH | See above |
| Time dynamics | Daily lift with an exponential fit, mixture-SPRT monitoring | ~66% of the AI-feed lift survives the novelty fade |
| Heterogeneity | Segment effects, interaction tests, T- and DR-learner, checked against simulator counterfactuals | Churn harm concentrated in users 35+ |
| Policies | Horvitz-Thompson IPW, paired bootstrap differences, oracle regret | No policy beats blanket friend boost |
| Retention | Time to first 7-day lapse, Kaplan-Meier, Cox PH; lapse early-warning model | HR 1.08 / 0.91; AUC 0.79 |
| NLP (real data) | TweetEval sentiment model vs VADER, NMF topics, feed-exposure scenario | Macro-recall 0.628 vs 0.570 |
| Virality (real data) | UCI news, whole-date split, LightGBM + SHAP | AUC 0.73 (0.69 without share-history features) |
| Revenue | Revenue per not-lapsed user-day x survival, joint bootstrap, persistence scenarios | Variants indistinguishable |
| Reporting | Memo generated from results, Streamlit app, Power BI report | [`reports/EXECUTIVE_MEMO.md`](reports/EXECUTIVE_MEMO.md) |

## Figures

| | |
|---|---|
| ![Novelty](reports/figures/02_novelty_decay.png) | ![KM](reports/figures/12_kaplan_meier.png) |
| ![Segments](reports/figures/04_segment_churn_by_age.png) | ![Policy](reports/figures/08_policy_values.png) |
| ![Content](reports/figures/15_responsible_ai_guardrails.png) | ![Impact](reports/figures/17_business_impact.png) |

All figures are in [`reports/figures/`](reports/figures) and all result tables in [`reports/tables/`](reports/tables).

## Running it

```bash
make setup        # venv with pinned versions (Python 3.12)
make all          # download data, run the pipeline (~7 min), run tests
make fixture      # same pipeline on a 3,000-user fixture, as CI does
make dashboard    # Streamlit app
```

Modules also run on their own, e.g. `python -m sei.experiment`. `python -m sei.pipeline --llm` additionally writes a
Claude-polished brief if Anthropic credentials are available.

**Power BI.** The report is called *Feed Ranking Launch Review*. In a Fabric workspace, import
`powerbi/Feed_Ranking_Launch_Review.ipynb` and run it; on Windows, open `powerbi/Feed_Ranking_Launch_Review.pbip`.
See [`powerbi/README.md`](powerbi/README.md). The DAX has been checked statically, not in a live Power BI engine.

## Exploring the data

| | |
|---|---|
| `make sql` | SQL prompt over every table (`.tables` lists them) |
| `make examples` | Runs [`queries/examples.sql`](queries/examples.sql) |
| [`notebooks/explore.ipynb`](notebooks/explore.ipynb) | SQL, pandas, Plotly and a one-line CUPED `ab_test()` |

The main tables are `users`, `daily_activity`, `user_experiment`, `churn_scores` and `user_cate`.

## Layout

```
sei/          analysis package
dashboard/    Streamlit app
powerbi/      Power BI project, CSVs, DAX, Fabric notebook
reports/      memo, summaries, figures, tables
tests/        statistical, data and report checks (35 tests)
docs/         audit and response
queries/      example SQL
notebooks/    exploration notebook
```

## Caveats

- The experiment is simulated. Simulator internals are used only to validate models, never as features.
- TweetEval (Barbieri et al., 2020) and UCI Online News Popularity (Fernandes et al., 2015) are real.
- The content-exposure results come from assumed ranker weights.
- Guardrail margins were set after the first results were seen.
- Business numbers (40M users, $38 per retained user, 3.5% monthly lapse, 18% nudge save rate) are assumptions in
  `sei/config.py`. Revenue outputs are scenarios.
