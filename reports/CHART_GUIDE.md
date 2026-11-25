# Reading the charts

Start with these three. They explain the main decision without needing the rest of the model diagnostics.

| Chart | Question | Takeaway |
|---|---|---|
| [Feed comparison](figures/00_project_summary.png) | What changed? | Friend boost offered a better balance of activity and inactivity, with a remaining feedback issue. |
| [Engagement over time](figures/02_novelty_decay.png) | Did the first-week improvement last? | The initial lift faded. A fitted long-term trend is an extrapolation. |
| [Results by age](figures/04_segment_churn_by_age.png) | Who was affected? | The pure AI feed's inactivity increase was concentrated in older age groups. |

All three describe simulated experiment data. Inactivity means no activity in the final
seven days; it does not establish that a user left permanently.

## More detail

- [Full scorecard](figures/01_metric_scorecard.png): all metrics and uncertainty intervals.
- [Group balance](figures/05_balance_love_plot.png): checks whether groups looked similar before the experiment.
- [Adjustment using prior activity](figures/06_cuped_variance_reduction.png): reduces noise in the comparisons.
- [Evidence over time](figures/03_sequential_monitoring.png): compares repeated-testing approaches.
- [First inactive period](figures/12_kaplan_meier.png): when users first experience a seven-day lapse.
- [Personalized assignment](figures/08_policy_values.png): no clear gain over giving everyone friend boost.
- [Ranking possible benefits](figures/07_qini_curves.png): checks whether a model can rank people by expected benefit.
- [Risk prediction](figures/09_churn_roc_calibration.png) and [model explanations](figures/10_churn_shap_beeswarm.png):
  predictive diagnostics, not evidence that changing an input causes retention.
- [Retention campaign estimate](figures/11_retention_profit_curve.png): depends on assumed response and contact costs.
- [Sentiment](figures/13_sentiment_benchmark.png) and [article popularity](figures/16_virality_drivers.png):
  separate real-data examples.
- [Topic exposure](figures/14_topic_exposure_by_arm.png) and [content scenario](figures/15_responsible_ai_guardrails.png):
  feed behavior depends on assumed ranking weights.
- [Revenue scenarios](figures/17_business_impact.png): projections under stated assumptions, not earned revenue.

The [technical report](TECHNICAL_REPORT.md) explains the methods and remaining limitations.
