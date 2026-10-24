# Social Engagement Intelligence: Experimentation, Causal ML & Retention Analytics

**Business question.** A social app tested two AI-ranked feeds against its current feed on 60,000 users for 28 days.
The pure AI "For You" feed has the biggest headline lift. Should we ship it?

**Answer: no. Ship the AI feed + friend boost variant.** The AI feed's +11.6% time-spent lift is mostly novelty
(steady state +7.7%), and it costs a **+17% rise in churn**, −9% friend interactions, and a feed that serves twice as
much negative content. The friend-boost variant keeps most of the engagement (+6.7% steady state), *cuts* churn by 11%,
and is worth **$103M vs $57M over 12 months** at 40M MAU (P(better) = 99.97% across 20,000 Monte Carlo draws).

![Metric scorecard](reports/figures/01_metric_scorecard.png)

---

## What's inside

| Module | Methods | Key result |
|---|---|---|
| **Simulation engine** (`sei/simulate.py`) | 42-day user-level event log, latent-trait heterogeneous effects, AR(1) behaviour, novelty decay, emergent churn hazard, salted-hash assignment | 1.76M active user-days; pre/post correlation 0.82 |
| **Experiment design** (`sei/experiment.py`) | SRM χ², covariate SMD balance, A/A re-randomisation, power/MDE | SRM p = 0.10 · max \|SMD\| 0.019 · A/A FPR 5.1% |
| **Effect estimation** | Welch + **CUPED**, **delta method** for ratio metrics, Poisson-bootstrap quantile effects, Benjamini-Hochberg & Holm | CUPED removes ~68% of variance (MDE 2.1% → 1.2%) |
| **Time dynamics** | Daily lift + exponential novelty-decay fit, **mSPRT always-valid p-values** | Naive daily peeking: 27% false positives; mSPRT: 0.9% |
| **Decision statistics** | Bayesian posteriors with skeptical prior, P(improves), expected loss, beta-binomial retention | |
| **Segments & HTE** (`sei/hte.py`) | Segment CUPED effects (BH-adjusted), interaction OLS (HC1), **T-learner & cross-fitted DR-learner**, Qini curves, ground-truth validation | Churn harm concentrated in 35+ users (interaction p = 0.0025) |
| **Policy learning** | 3-arm CATE policy, **IPW off-policy evaluation** with paired bootstrap, depth-3 policy tree | Model-targeted policy: +$2.19 vs +$2.05/user/yr for blanket friend boost (overlapping CIs) |
| **Retention** (`sei/retention.py`) | Kaplan-Meier, log-rank, **Cox PH**; leakage-free early-warning model (LogReg vs LightGBM), isotonic calibration, SHAP, campaign profit curve | HR 1.19 vs 0.88 · churn ROC-AUC 0.896, top-decile lift 6.9× |
| **NLP on real text** (`sei/nlp.py`) | TweetEval (60K labelled tweets): TF-IDF word+char LogReg vs VADER, NMF topics, ranker exposure simulation | Macro-recall 0.628 vs 0.570 · AI feed: −43% diversity, negative share 19% → 37% |
| **Virality on real data** (`sei/virality.py`) | UCI Online News Popularity (39,644 articles), time-based split, LightGBM + SHAP, creator playbook | AUC 0.732 (logistic 0.708) |
| **Business impact** (`sei/impact.py`) | 12-month Monte Carlo with compounding churn, one-way sensitivity (tornado) | No single assumption flips the decision |
| **Executive memo** (`sei/report.py`) | Deterministic memo from outputs + optional Claude API brief | [`reports/EXECUTIVE_MEMO.md`](reports/EXECUTIVE_MEMO.md) |
| **Dashboards** | Streamlit (9 pages, live calculators) + **Power BI PBIP** (9 pages, 30 DAX measures) | [`dashboard/`](dashboard/app.py) · [`powerbi/`](powerbi/README.md) |

## What v2 fixes from the original project

| Issue in [v1](https://github.com/sakshiraade/social-engagement-intelligence) | v2 |
|---|---|
| One row per user; pre and post metrics drawn independently, so CUPED is impossible | Daily event log with autocorrelated behaviour; CUPED removes ~68% of variance |
| `days_since_last_login` generated **from the churn label**, then used as a model feature (target leakage → inflated AUC) | Point-in-time features (≤ day 13) and a future label window (days 14–27); a test enforces it |
| SMOTE + Platt scaling on leaked features | Class-aware LightGBM, isotonic calibration, honest AUC 0.896 with a Brier-score comparison |
| "Experiment week" randomly assigned per user, so novelty can't really be measured | Real calendar time: daily lift with fitted decay (τ ≈ 7 days) |
| 15 caption templates repeated 50K times as the NLP corpus | 60K real human-labelled tweets with official train/val/test splits and a published benchmark |
| Two arms, fixed-horizon t-tests only | Three arms, delta method, sequential (mSPRT), Bayesian, quantile effects, multiple-testing control |
| Segment cuts only | Causal ML (DR-learner), Qini, off-policy policy evaluation, validation vs ground truth |
| Revenue "estimate" not tied to uncertainty | Monte Carlo over sampling distributions with compounding churn + sensitivity analysis |
| Notebooks only, no tests | Importable package, one-command pipeline, 12 unit/contract tests |

## Selected figures

| | |
|---|---|
| ![Novelty](reports/figures/02_novelty_decay.png) | ![KM](reports/figures/12_kaplan_meier.png) |
| ![Segments](reports/figures/04_segment_churn_by_age.png) | ![Policy](reports/figures/08_policy_values.png) |
| ![Responsible AI](reports/figures/15_responsible_ai_guardrails.png) | ![Impact](reports/figures/17_business_impact.png) |

All 17 figures are in [`reports/figures/`](reports/figures), all result tables in [`reports/tables/`](reports/tables).

## Quickstart

```bash
make setup        # python3 -m venv .venv && pip install -r requirements.txt
make all          # download data → run pipeline (~4 min) → run tests
make dashboard    # streamlit run dashboard/app.py
```

Individual modules run on their own: `python -m sei.experiment`, `python -m sei.hte`, and so on.
`python -m sei.pipeline --llm` also writes a Claude-polished executive brief (`claude-opus-5-5`). It uses the
`ANTHROPIC_API_KEY` or `ant auth login` credentials, and is skipped gracefully when neither is present.

**Power BI:** see [`powerbi/README.md`](powerbi/README.md). Open `powerbi/SEI_Dashboard.pbip` in Power BI Desktop
(Windows) after setting the `DataFolder` parameter, or regenerate with
`make powerbi DATA_FOLDER="C:\path\to\powerbi\data\"`.

## Project layout

```
sei/                 analysis package (config, simulate, features, stats, experiment, hte, retention,
                     nlp, virality, impact, report, powerbi, pipeline, download)
dashboard/app.py     Streamlit app
powerbi/             PBIP project + star-schema CSVs + DAX + theme
reports/             EXECUTIVE_MEMO.md, *_summary.json, figures/, tables/
tests/               statistical calibration + data-contract tests
data/                raw/ & processed/ (regenerated), external/ (downloaded)
```

## Data & honesty notes

- The experiment is **simulated**: no public dataset has randomised feed-ranking experiments with pre-period logs.
  The simulator's ground-truth individual effects are kept in a separate file and used **only** to validate the
  causal models, never as features.
- The NLP and virality modules use **real public data**: TweetEval sentiment (Barbieri et al., 2020;
  SemEval-2017) and UCI Online News Popularity (Fernandes et al., 2015, CC BY 4.0).
- Content exposure by arm comes from simulated ranking policies applied to real posts. It shows the
  *mechanism*; production ranker logs would be needed to measure it for real.
- Business figures (40M MAU, $38 LTV, 3.5% monthly base churn) are configurable assumptions in `sei/config.py`.
