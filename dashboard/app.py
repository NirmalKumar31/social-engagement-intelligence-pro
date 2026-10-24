"""Social Engagement Intelligence: interactive decision dashboard.

Run:  streamlit run dashboard/app.py   (after `make all` has produced reports/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sei import config as C  # noqa: E402
from sei import stats as S  # noqa: E402
from sei.impact import project  # noqa: E402

st.set_page_config(page_title="Social Engagement Intelligence", page_icon="📊", layout="wide")

T, R = C.TABLES, C.REPORTS
LAB = C.ARM_LABELS
COL = C.ARM_COLORS
GOOD, BAD, MUTED = C.STATUS["good"], C.STATUS["critical"], C.INK["muted"]

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1400px;}
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
  .verdict {padding: .9rem 1.1rem; border-radius: 8px; border: 1px solid rgba(11,11,11,.1); margin-bottom: .8rem;}
</style>""", unsafe_allow_html=True)


def fig_style(fig: go.Figure, h: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50 if title else 20, b=10), title=title,
                      title_font=dict(size=15), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(137,135,129,.18)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(137,135,129,.18)", zeroline=False)
    return fig


@st.cache_data
def tbl(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


@st.cache_data
def js(name: str) -> dict:
    return json.loads((R / name).read_text())


if not (R / "experiment_summary.json").exists():
    st.error("No analysis outputs found. Run `make all` (or `python -m sei.pipeline`) first.")
    st.stop()

ex, hte, ret = js("experiment_summary.json"), js("hte_summary.json"), js("retention_summary.json")
nlp, vir, imp = js("nlp_summary.json"), js("virality_summary.json"), js("impact_summary.json")
res = tbl("metric_results")

PAGES = ["Executive summary", "Experiment readout", "Time dynamics", "Segments", "Personalisation (HTE)",
         "Retention & churn risk", "Content & responsible AI", "Creator virality (real data)", "Business impact simulator"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("3-arm feed-ranking experiment · 60,000 users · 42 days of event logs · "
                   "plus real TweetEval (60K tweets) and UCI news (39,644 articles) data.")


def lift(arm, metric, key="rel_lift"):
    return res[(res.arm == arm) & (res.metric == metric)][key].iloc[0]


# ======================================================================= pages
if page == "Executive summary":
    st.title("Feed ranking experiment: launch decision")
    mc = pd.DataFrame(imp["results"]).set_index("comparison")
    st.markdown(f"""<div class="verdict" style="border-left: 5px solid {GOOD}">
    <b>✅ Recommendation: ship <i>AI feed + friend boost</i>; do not ship the pure AI feed.</b><br>
    The AI feed's headline lift is inflated by novelty and paid for with churn. Over 12 months the friend-boost variant is worth
    <b>${mc.loc['AI + friend boost vs control', 'p50_musd']:,.0f}M</b> vs <b>${mc.loc['AI feed vs control', 'p50_musd']:,.0f}M</b>
    (P(better) = {mc.loc['Friend boost minus AI feed', 'prob_positive']:.2%}).</div>""", unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("AI feed · minutes", f"{lift('ai_feed', 'minutes'):+.1%}")
    c[1].metric("Friend boost · minutes", f"{lift('ai_feed_social', 'minutes'):+.1%}")
    c[2].metric("AI feed · churn", f"{lift('ai_feed', 'churned'):+.1%}", "harm", delta_color="inverse")
    c[3].metric("Friend boost · churn", f"{lift('ai_feed_social', 'churned'):+.1%}", "win")
    c[4].metric("AI feed · steady-state lift", f"{ex['novelty']['ai_feed']['steady_state_lift']:+.1%}",
                f"wk1 {ex['novelty']['ai_feed']['week1_lift']:.0%}", delta_color="off")
    c[5].metric("Churn model AUC", f"{ret['churn_model']['holdout']['LightGBM + isotonic']['roc_auc']:.3f}")

    l, r = st.columns([1.2, 1])
    with l:
        dl = tbl("daily_lift")
        fig = go.Figure()
        for arm in C.TREATMENTS:
            a = dl[dl.arm == arm]
            fig.add_scatter(x=a.day, y=a.rel_lift * 100, mode="lines", line=dict(color=COL[arm], width=1), opacity=.45,
                            showlegend=False, hovertemplate="day %{x}: %{y:.1f}%")
            fig.add_scatter(x=a.day, y=a.fitted * 100, mode="lines", line=dict(color=COL[arm], width=3), name=LAB[arm])
        fig.update_yaxes(title="Lift in minutes/user (%)"); fig.update_xaxes(title="Experiment day")
        st.plotly_chart(fig_style(fig, 360, "Novelty decay: the headline lift is not the steady state"), width="stretch")
    with r:
        show = res[res.tier != "secondary"][["arm", "label", "rel_lift", "p_adj_bh", "verdict"]].copy()
        show["arm"] = show.arm.map(LAB)
        show["rel_lift"] = show.rel_lift.map(lambda v: f"{v:+.1%}")
        show["p_adj_bh"] = show.p_adj_bh.map(lambda v: f"{v:.1e}")
        st.markdown("**Primary + guardrail metrics** (CUPED / delta method, BH-adjusted)")
        st.dataframe(show.rename(columns={"label": "metric", "rel_lift": "lift", "p_adj_bh": "p (BH)"}),
                     hide_index=True, width="stretch", height=330)
    memo = R / "EXECUTIVE_MEMO.md"
    if memo.exists():
        with st.expander("📄 Full executive memo"):
            st.markdown(memo.read_text())

elif page == "Experiment readout":
    st.title("Experiment readout")
    d = ex["design"]
    c = st.columns(5)
    c[0].metric("SRM χ² p-value", f"{d['srm']['p_value']:.3f}", "no mismatch", delta_color="off")
    c[1].metric("Max |SMD|", f"{d['max_abs_smd']:.3f}", "< 0.10 balanced", delta_color="off")
    c[2].metric("A/A false-positive rate", f"{ex['aa']['aa_false_positive_rate']:.1%}", "target 5%", delta_color="off")
    c[3].metric("Pre/post correlation", f"{d['pre_post_corr']:.2f}")
    c[4].metric("MDE (raw → CUPED)", f"{d['mde_rel_cuped']:.1%}", f"from {d['mde_rel_raw']:.1%}", delta_color="off")

    arm = st.segmented_control("Treatment arm", C.TREATMENTS, format_func=LAB.get, default="ai_feed")
    r = res[res.arm == arm].iloc[::-1]
    good = np.where(r.higher_is_better, r["diff"] > 0, r["diff"] < 0)
    colors = np.where(~r.significant.astype(bool), MUTED, np.where(good, GOOD, BAD))
    fig = go.Figure(go.Scatter(
        x=r.rel_lift * 100, y=r.label, mode="markers", marker=dict(size=11, color=colors, line=dict(color="white", width=2)),
        error_x=dict(type="data", symmetric=False, array=(r.rel_ci_high - r.rel_lift) * 100,
                     arrayminus=(r.rel_lift - r.rel_ci_low) * 100, color="#888", thickness=1.5),
        customdata=np.c_[r.p_adj_bh, r.variance_reduction * 100],
        hovertemplate="%{y}<br>lift %{x:.2f}%<br>p (BH) %{customdata[0]:.2e}<br>CUPED var. reduction %{customdata[1]:.0f}%<extra></extra>"))
    fig.add_vline(0, line_color="#c3c2b7")
    fig.update_xaxes(title="Relative change vs control (%), 95% CI")
    st.plotly_chart(fig_style(fig, 470, "Metric scorecard · green = win, red = harm, gray = not significant"), width="stretch")

    l, r_ = st.columns(2)
    with l:
        aa = tbl("aa_pvalues")
        fig = go.Figure(go.Histogram(x=aa.p_value, xbins=dict(start=0, end=1, size=0.05), marker_color=C.SEQ_BLUE[3]))
        fig.add_hline(len(aa) / 20, line_dash="solid", line_color=MUTED)
        fig.update_xaxes(title="p-value"); fig.update_yaxes(title="A/A tests")
        st.plotly_chart(fig_style(fig, 320, "A/A p-values are uniform: the pipeline is calibrated"), width="stretch")
    with r_:
        st.markdown("**Power calculator** (live, using this experiment's variance)")
        df_u = pd.read_parquet(C.PROCESSED / "user_experiment.parquet", columns=["arm", "exp_minutes_per_day"])
        sd = df_u[df_u.arm == "control"].exp_minutes_per_day.std()
        base = df_u[df_u.arm == "control"].exp_minutes_per_day.mean()
        mde_pct = st.slider("Minimum detectable lift (%)", 0.5, 10.0, 2.0, 0.5)
        power = st.slider("Power", 0.5, 0.95, 0.8, 0.05)
        use_cuped = st.toggle("Use CUPED", True)
        vr = d["pre_post_corr"] ** 2 if use_cuped else 0.0
        n = S.required_n_per_arm(sd, mde_pct / 100 * base, power=power, variance_reduction=vr)
        st.metric("Users needed per arm", f"{n:,}", f"{'with' if use_cuped else 'without'} CUPED · {vr:.0%} variance removed",
                  delta_color="off")

elif page == "Time dynamics":
    st.title("Time dynamics: novelty & sequential monitoring")
    nv = ex["novelty"]
    c = st.columns(4)
    for i, arm in enumerate(C.TREATMENTS):
        c[2 * i].metric(f"{LAB[arm]} · week 1 → week 4", f"{nv[arm]['week1_lift']:.1%} → {nv[arm]['week4_lift']:.1%}")
        c[2 * i + 1].metric(f"{LAB[arm]} · fitted steady state", f"{nv[arm]['steady_state_lift']:.1%}",
                            f"novelty τ ≈ {nv[arm]['decay_days']:.1f} days", delta_color="off")
    dl = tbl("daily_lift")
    fig = go.Figure()
    for arm in C.TREATMENTS:
        a = dl[dl.arm == arm]
        fig.add_scatter(x=np.r_[a.day, a.day[::-1]], y=np.r_[a.ci_high, a.ci_low[::-1]] * 100, fill="toself",
                        fillcolor=COL[arm], opacity=.12, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=a.day, y=a.rel_lift * 100, mode="lines+markers", marker=dict(size=5), line=dict(color=COL[arm], width=1.5),
                        name=f"{LAB[arm]} (daily)")
        fig.add_scatter(x=a.day, y=a.fitted * 100, mode="lines", line=dict(color=COL[arm], width=3), name=f"{LAB[arm]} (fit)")
    fig.update_xaxes(title="Experiment day"); fig.update_yaxes(title="Lift (%)")
    st.plotly_chart(fig_style(fig, 420, "Daily lift with 95% CI and exponential novelty-decay fit"), width="stretch")

    sq = ex["sequential"]
    st.markdown(f"""**Why always-valid inference?** In {4000:,} simulated A/A experiments with daily peeks, stopping at the first
    p < 0.05 produced a **{sq['peeking_fpr_naive']:.0%}** false-positive rate. The mSPRT always-valid p-value kept it at
    **{sq['peeking_fpr_msprt']:.1%}**, while the fixed-horizon test alone sits at {sq['fixed_horizon_fpr']:.1%}.""")
    s = tbl("sequential_monitoring")
    fig = go.Figure()
    for arm in C.TREATMENTS:
        a = s[s.arm == arm]
        fig.add_scatter(x=a.day, y=a.p_always_valid, name=f"{LAB[arm]} · always-valid", line=dict(color=COL[arm], width=3))
        fig.add_scatter(x=a.day, y=a.p_naive, name=f"{LAB[arm]} · naive", line=dict(color=COL[arm], width=1), opacity=.5)
    fig.add_hline(0.05, line_color=BAD)
    fig.update_yaxes(type="log", title="p-value"); fig.update_xaxes(title="Experiment day")
    st.plotly_chart(fig_style(fig, 380, "Running p-values on the primary metric"), width="stretch")

elif page == "Segments":
    st.title("Segment deep-dive")
    seg = tbl("segment_results")
    dim = st.segmented_control("Dimension", list(seg.dimension.unique()), default="age_group")
    s = seg[seg.dimension == dim]
    metric = st.radio("Metric", ["churn_diff_pp", "minutes_lift", "friend_int_lift", "revenue_lift"], horizontal=True,
                      format_func={"churn_diff_pp": "Churn change (pp)", "minutes_lift": "Minutes lift",
                                   "friend_int_lift": "Friend interactions lift", "revenue_lift": "Revenue lift"}.get)
    fig = go.Figure()
    for arm in C.TREATMENTS:
        a = s[s.arm == arm]
        y = a[metric] * (1 if metric == "churn_diff_pp" else 100)
        kw = {}
        if metric == "churn_diff_pp":
            kw = dict(error_y=dict(type="data", symmetric=False, array=a.churn_ci_high_pp - a.churn_diff_pp,
                                   arrayminus=a.churn_diff_pp - a.churn_ci_low_pp, color="#666"))
        elif metric == "minutes_lift":
            kw = dict(error_y=dict(type="data", symmetric=False, array=(a.minutes_ci_high - a.minutes_lift) * 100,
                                   arrayminus=(a.minutes_lift - a.minutes_ci_low) * 100, color="#666"))
        fig.add_bar(x=a.segment, y=y, name=LAB[arm], marker_color=COL[arm], **kw)
    fig.update_layout(barmode="group", bargap=.3)
    fig.update_yaxes(title="pp" if metric == "churn_diff_pp" else "%")
    st.plotly_chart(fig_style(fig, 420), width="stretch")
    st.caption(f"Formal heterogeneity tests (AI feed × age, HC1-robust OLS): minutes p = {ex['interaction']['minutes_age_interaction_p']:.1e}, "
               f"churn p = {ex['interaction']['churn_age_interaction_p']:.4f}")
    st.dataframe(s[["segment", "arm", "n_treatment", "minutes_lift", "minutes_p_adj", "churn_control", "churn_treatment",
                    "churn_diff_pp", "churn_p_adj", "status"]].round(4), hide_index=True, width="stretch")

elif page == "Personalisation (HTE)":
    st.title("Heterogeneous treatment effects & rollout policy")
    st.caption("Outcome: net annual user value = steady-state ad revenue × 365 − churned × LTV. DR-learner with cross-fitting; "
               "policies valued by IPW on a 35% held-out fold.")
    pol = tbl("policy_values").sort_values("uplift_vs_control_usd")
    fig = go.Figure(go.Bar(x=pol.uplift_vs_control_usd, y=pol.policy, orientation="h",
                           marker_color=[C.SEQ_BLUE[5] if "Model" in p else C.SEQ_BLUE[2] for p in pol.policy],
                           error_x=dict(type="data", symmetric=False, array=pol.uplift_ci_high - pol.uplift_vs_control_usd,
                                        arrayminus=pol.uplift_vs_control_usd - pol.uplift_ci_low, color="#666"),
                           customdata=pol.annual_value_at_scale_musd,
                           hovertemplate="%{y}<br>$%{x:.2f}/user/yr<br>≈ $%{customdata:,.0f}M/yr at scale<extra></extra>"))
    fig.update_xaxes(title="USD per user per year vs control (paired-bootstrap 95% CI)")
    st.plotly_chart(fig_style(fig, 340, "Policy value"), width="stretch")
    l, r = st.columns(2)
    with l:
        arm = st.segmented_control("Arm", C.TREATMENTS, format_func=LAB.get, default="ai_feed", key="qarm")
        q = tbl("qini_curves")
        fig = go.Figure()
        for model, w in [("DR-learner", 3), ("T-learner", 1.5)]:
            a = q[(q.arm == arm) & (q.model == model)]
            fig.add_scatter(x=a.fraction_targeted * 100, y=a.cumulative_gain / 1e3, name=model,
                            line=dict(color=COL[arm] if model == "DR-learner" else C.INK["secondary"], width=w))
        fig.add_scatter(x=a.fraction_targeted * 100, y=a.random / 1e3, name="Random", line=dict(color=MUTED, width=1.2))
        fig.update_xaxes(title="% targeted"); fig.update_yaxes(title="Cumulative gain ($K)")
        st.plotly_chart(fig_style(fig, 360, "Qini curve"), width="stretch")
        h = hte[arm]
        st.caption(f"Qini AUC ratio vs random: DR {h['qini_auc_ratio_DR-learner']:.2f} · T {h['qini_auc_ratio_T-learner']:.2f}. "
                   f"Spearman ρ with simulator ground truth: DR {h['spearman_vs_truth_dr']:.2f}.")
    with r:
        dc = tbl("cate_deciles")
        a = dc[dc.arm == arm]
        fig = go.Figure()
        fig.add_bar(x=a.decile, y=a.predicted_cate, name="Predicted", marker_color=C.SEQ_BLUE[2])
        fig.add_bar(x=a.decile, y=a.observed_uplift, name="Observed (held-out)", marker_color=C.SEQ_BLUE[5])
        fig.update_layout(barmode="group"); fig.update_xaxes(title="Predicted-CATE decile"); fig.update_yaxes(title="USD/user/yr")
        st.plotly_chart(fig_style(fig, 360, "CATE calibration by decile"), width="stretch")
    with st.expander("Interpretable policy tree (AI feed advantage over friend boost)"):
        st.code((R / "policy_tree_rules.txt").read_text())

elif page == "Retention & churn risk":
    st.title("Retention & churn early warning")
    sv, cm = ret["survival"], ret["churn_model"]
    c = st.columns(5)
    c[0].metric("Cox HR · AI feed", f"{sv['hr_ai_feed']:.2f}", "higher hazard", delta_color="inverse")
    c[1].metric("Cox HR · friend boost", f"{sv['hr_ai_feed_social']:.2f}", "lower hazard")
    c[2].metric("Log-rank p", f"{sv['logrank_p']:.1e}")
    c[3].metric("ROC-AUC (holdout)", f"{cm['holdout']['LightGBM + isotonic']['roc_auc']:.3f}")
    c[4].metric("Top-decile lift", f"{cm['holdout']['LightGBM + isotonic']['top_decile_lift']:.1f}×")
    l, r = st.columns(2)
    with l:
        km = tbl("km_survival")
        fig = go.Figure()
        for arm in C.ARMS:
            a = km[km.arm == arm]
            fig.add_scatter(x=a.day, y=a.survival * 100, line_shape="hv", name=LAB[arm], line=dict(color=COL[arm], width=2.5))
        fig.update_xaxes(title="Experiment day", range=[0, C.EXP_DAYS - C.CHURN_WINDOW]); fig.update_yaxes(title="% not churned")
        st.plotly_chart(fig_style(fig, 360, "Kaplan-Meier retention"), width="stretch")
    with r:
        mt = tbl("churn_model_metrics").set_index("model")
        st.markdown("**Model comparison (held-out)**. Note how the class-weighted logistic is badly calibrated (Brier score).")
        st.dataframe(mt[["roc_auc", "pr_auc", "brier", "log_loss", "top_decile_lift", "cv_auc"]].round(4), width="stretch")
        shp = tbl("churn_shap_importance").head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=shp.mean_abs_shap, y=shp.feature, orientation="h", marker_color=C.SEQ_BLUE[4]))
        st.plotly_chart(fig_style(fig, 260, "Top churn drivers (mean |SHAP|)"), width="stretch")
    st.subheader("Retention campaign economics")
    pc = tbl("retention_profit_curve")
    a, b, c_ = st.columns(3)
    cost = a.slider("Cost per targeted user ($)", 0.1, 3.0, C.RETENTION_CAMPAIGN_COST, 0.05)
    save_rate = b.slider("Share of targeted churners saved", 0.05, 0.5, C.RETENTION_CAMPAIGN_SAVE_RATE, 0.01)
    ltv = c_.slider("LTV of a saved user ($)", 5.0, 120.0, C.LTV_PER_RETAINED_USER, 1.0)
    scale = pc.profit_at_scale_musd.iloc[-1] / pc.profit_test_usd.iloc[-1] if pc.profit_test_usd.iloc[-1] else 0
    prof = (pc.churners_caught * save_rate * ltv - pc.users_targeted * cost) * scale
    best = prof.idxmax()
    fig = go.Figure(go.Scatter(x=pc.fraction_targeted * 100, y=prof, line=dict(color=C.SEQ_BLUE[5], width=3),
                               hovertemplate="target top %{x:.0f}%: $%{y:.2f}M<extra></extra>"))
    fig.add_vline(pc.fraction_targeted[best] * 100, line_color=MUTED)
    fig.add_hline(0, line_color="#c3c2b7")
    fig.update_xaxes(title="% of scored users targeted (highest risk first)"); fig.update_yaxes(title="Net value at scale ($M)")
    st.plotly_chart(fig_style(fig, 340, f"Optimum: target top {pc.fraction_targeted[best]:.0%} → ${prof[best]:,.2f}M "
                                        f"(recall {pc.recall[best]:.0%}, precision {pc.precision[best]:.0%})"), width="stretch")

elif page == "Content & responsible AI":
    st.title("Content intelligence & responsible-AI guardrails")
    sm = tbl("sentiment_model_metrics").set_index("model")
    c = st.columns(4)
    c[0].metric("Sentiment macro-recall (ours)", f"{sm.iloc[1].macro_recall:.3f}", f"VADER {sm.iloc[0].macro_recall:.3f}", delta_color="off")
    ce = tbl("content_exposure_summary").set_index("arm")
    c[1].metric("Negative posts served · AI feed", f"{ce.loc['ai_feed', 'negative_share']:.1%}",
                f"control {ce.loc['control', 'negative_share']:.1%}", delta_color="off")
    c[2].metric("Feed diversity · AI feed", f"{ce.loc['ai_feed', 'topic_entropy']:.2f}",
                f"{ce.loc['ai_feed', 'topic_entropy_rel_change']:+.0%} vs control", delta_color="inverse")
    c[3].metric("Feed diversity · friend boost", f"{ce.loc['ai_feed_social', 'topic_entropy']:.2f}",
                f"{ce.loc['ai_feed_social', 'topic_entropy_rel_change']:+.0%} vs control", delta_color="inverse")
    tx = tbl("topic_exposure_by_arm").merge(tbl("topic_info")[["topic_id", "label"]], on="topic_id")
    fig = go.Figure()
    for arm in C.ARMS:
        a = tx[tx.arm == arm]
        fig.add_bar(x=a.label, y=a.share * 100, name=LAB[arm], marker_color=COL[arm])
    fig.update_layout(barmode="group"); fig.update_yaxes(title="% of impressions")
    st.plotly_chart(fig_style(fig, 400, "Topic exposure by arm (NMF topics discovered on real posts)"), width="stretch")
    st.caption("Exposure is simulated: each arm's ranking objective serves real TweetEval posts to 3,000 sampled users per arm. "
               "The sentiment model is trained and scored on the official TweetEval splits.")
    st.dataframe(tbl("topic_info")[["topic_id", "label", "top_words", "n_posts", "share_negative_pred"]], hide_index=True,
                 width="stretch")

elif page == "Creator virality (real data)":
    st.title("What makes content spread? (real data)")
    c = st.columns(4)
    c[0].metric("Articles", f"{vir['n_articles']:,}")
    c[1].metric("Time-split AUC · LightGBM", f"{vir['auc_lightgbm']:.3f}", f"logistic {vir['auc_logistic']:.3f}", delta_color="off")
    c[2].metric("Precision in top 10% scored", f"{vir['precision_top10pct']:.0%}", f"base {vir['base_viral_rate_test']:.0%}", delta_color="off")
    c[3].metric("Viral threshold", f"{vir['viral_threshold_shares']:,.0f} shares")
    l, r = st.columns(2)
    with l:
        sh = tbl("virality_shap_importance").head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=sh.mean_abs_shap, y=sh.feature, orientation="h", marker_color=C.SEQ_BLUE[4]))
        st.plotly_chart(fig_style(fig, 420, "Top drivers (mean |SHAP|)"), width="stretch")
    with r:
        pb = tbl("virality_playbook")
        dim = st.selectbox("Playbook dimension", pb.dimension.unique())
        a = pb[pb.dimension == dim].sort_values("viral_rate")
        fig = go.Figure(go.Bar(x=a.viral_rate * 100, y=a.level, orientation="h", marker_color=C.SEQ_BLUE[4],
                               customdata=np.c_[a.articles, a.median_shares],
                               hovertemplate="%{y}: %{x:.1f}% viral<br>%{customdata[0]:,} articles · median %{customdata[1]:,.0f} shares<extra></extra>"))
        fig.update_xaxes(title="% of articles in top-quartile shares")
        st.plotly_chart(fig_style(fig, 380, "Viral rate by level"), width="stretch")

elif page == "Business impact simulator":
    st.title("12-month business impact simulator")
    inp = imp["inputs"]
    a, b, c_ = st.columns(3)
    mau = a.slider("Eligible MAU (millions)", 5, 150, int(C.MAU_POPULATION / 1e6), 5) * 1e6
    base_churn = b.slider("Monthly base churn", 0.01, 0.10, C.MONTHLY_BASE_CHURN, 0.005)
    rev_mult = c_.slider("Revenue per user multiplier", 0.5, 2.0, 1.0, 0.05)
    rng = np.random.default_rng(0)
    draws = 5000
    rev0 = inp["rev_per_alive_user_month"] * rev_mult
    base = project(np.zeros(1), np.ones(1), rev0, base_churn=base_churn, pop=mau)[0]
    cum = {}
    for arm in C.TREATMENTS:
        x = inp[arm]
        cum[arm] = project(rng.normal(x["rev_lift"], x["rev_lift_se"], draws), np.exp(rng.normal(x["log_hr"], x["log_hr_se"], draws)),
                           rev0, base_churn=base_churn, pop=mau) - base
    k = st.columns(3)
    k[0].metric("Friend boost · median 12-mo value", f"${np.median(cum['ai_feed_social'][:, -1]) / 1e6:,.0f}M")
    k[1].metric("AI feed · median 12-mo value", f"${np.median(cum['ai_feed'][:, -1]) / 1e6:,.0f}M")
    k[2].metric("P(friend boost > AI feed)", f"{(cum['ai_feed_social'][:, -1] > cum['ai_feed'][:, -1]).mean():.1%}")
    fig = go.Figure()
    months = np.arange(1, 13)
    for arm in C.TREATMENTS:
        lo, mid, hi = (np.quantile(cum[arm], q, axis=0) / 1e6 for q in (0.05, 0.5, 0.95))
        fig.add_scatter(x=np.r_[months, months[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=COL[arm], opacity=.15,
                        line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=months, y=mid, name=LAB[arm], line=dict(color=COL[arm], width=3))
    fig.update_xaxes(title="Months after rollout"); fig.update_yaxes(title="Cumulative incremental revenue ($M)")
    st.plotly_chart(fig_style(fig, 420, "Projection (median, 90% band)"), width="stretch")
    st.caption("Each draw samples the post-novelty CUPED revenue lift and the Cox churn hazard ratio from their sampling "
               "distributions, then compounds churn month by month.")
