"""Export a Power BI star schema and generate a Power BI Project (PBIP).

Outputs (all under powerbi/):
  data/*.csv                      star schema + analysis result tables
  SEI_Dashboard.pbip              open this in Power BI Desktop (Windows, PBIP preview feature on)
  SEI_Dashboard.SemanticModel/    model.bim: tables, relationships, DAX measures, DataPath parameter (GitHub or local)
  SEI_Dashboard.Report/           report.json: 9 report pages
  SEI_Theme.json                  palette theme (View > Themes > Browse for themes)
  measures.dax                    every DAX measure, for copy/paste into any model
"""
from __future__ import annotations

import argparse
import json
import shutil
import uuid

import numpy as np
import pandas as pd

from . import config as C

OUT = C.POWERBI
DATA = OUT / "data"
NAME = "SEI_Dashboard"
SEGMENTS = ["age_group", "region", "device", "account_type"]
SUMS = ["sessions", "minutes", "likes", "comments", "dms_sent", "shares", "posts_created", "negative_feedback",
        "ad_impressions", "ad_clicks", "ad_revenue"]


# ---------------------------------------------------------------- data export
def export_data() -> dict[str, pd.DataFrame]:
    users = pd.read_parquet(C.RAW / "users.parquet")
    daily = pd.read_parquet(C.RAW / "daily_activity.parquet")
    ux = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    tables: dict[str, pd.DataFrame] = {}

    days = np.arange(-C.PRE_DAYS, C.EXP_DAYS)
    dates = pd.Timestamp(C.START_DATE) + pd.to_timedelta(days, unit="D")
    tables["dim_date"] = pd.DataFrame({
        "date": dates, "experiment_day": days, "period": np.where(days < 0, "Pre-period", "Experiment"),
        "experiment_week": np.where(days < 0, 0, days // 7 + 1), "weekday": dates.day_name(),
        "is_weekend": (dates.dayofweek >= 5).astype(int)})
    tables["dim_arm"] = pd.DataFrame({"arm": C.ARMS, "arm_label": [C.ARM_LABELS[a] for a in C.ARMS],
                                      "is_treatment": [0, 1, 1], "sort_order": [1, 2, 3],
                                      "color_hex": [C.ARM_COLORS[a] for a in C.ARMS]})

    # Daily fact at (date x arm x segment) grain: ~28K rows, additive sums + assigned-user denominators
    d = daily.merge(users[["user_id", "arm"] + SEGMENTS], on="user_id")
    agg = d.groupby(["day", "arm"] + SEGMENTS)[SUMS].sum()
    agg["active_users"] = d.groupby(["day", "arm"] + SEGMENTS).size()
    assigned = users.groupby(["arm"] + SEGMENTS).size().rename("users_assigned").reset_index()
    grid = assigned.merge(pd.DataFrame({"day": days}), how="cross")
    fact = grid.merge(agg.reset_index(), on=["day", "arm"] + SEGMENTS, how="left").fillna(0)
    fact["date"] = pd.Timestamp(C.START_DATE) + pd.to_timedelta(fact.day, unit="D")
    fact["friend_interactions"] = fact.comments + fact.dms_sent
    fact = fact.drop(columns="day")
    int_cols = [c for c in SUMS + ["active_users", "users_assigned", "friend_interactions"] if c not in ("minutes", "ad_revenue")]
    fact[int_cols] = fact[int_cols].astype(int)
    fact["minutes"] = fact.minutes.round(1)
    fact["ad_revenue"] = fact.ad_revenue.round(4)
    tables["fact_daily"] = fact

    # User fact
    keep = ["user_id", "arm", "age_group", "region", "device", "account_type", "acquisition_channel", "tenure_days",
            "friend_count", "is_new_user", "churned", "pre_minutes_per_day", "exp_minutes_per_day",
            "pre_active_days_per_day", "exp_active_days_per_day", "exp_friend_interactions_per_day",
            "pre_ad_revenue_per_day", "exp_ad_revenue_per_day", "late_ad_revenue_per_day"]
    fu = ux[keep].copy()
    sc = pd.read_parquet(C.PROCESSED / "churn_scores.parquet")[["user_id", "churn_probability", "risk_tier", "risk_decile",
                                                                "actual_churn_next_14d"]]
    ca = pd.read_parquet(C.PROCESSED / "user_cate.parquet")[["user_id", "cate_ai_feed", "cate_ai_feed_social", "policy_arm"]]
    fu = fu.merge(sc, on="user_id", how="left").merge(ca, on="user_id", how="left")
    fu["risk_tier"] = fu.risk_tier.fillna("Not scored (inactive at day 13)")
    fu["policy_arm"] = fu.policy_arm.fillna("Not in held-out fold")
    tables["fact_user"] = fu.round(5)

    # Analysis result tables
    topic = pd.read_csv(C.TABLES / "topic_info.csv")
    tx = pd.read_csv(C.TABLES / "topic_exposure_by_arm.csv").merge(topic[["topic_id", "label"]], on="topic_id")
    simple = {
        "metric_results": "metric_results.csv", "segment_results": "segment_results.csv", "daily_lift": "daily_lift.csv",
        "sequential_monitoring": "sequential_monitoring.csv", "power_mde": "power_mde.csv", "balance_smd": "balance_smd.csv",
        "bayesian_results": "bayesian_results.csv", "km_survival": "km_survival.csv", "cox_hazard_ratios": "cox_hazard_ratios.csv",
        "churn_model_metrics": "churn_model_metrics.csv", "churn_shap_importance": "churn_shap_importance.csv",
        "retention_profit_curve": "retention_profit_curve.csv", "policy_values": "policy_values.csv",
        "qini_curves": "qini_curves.csv", "cate_deciles": "cate_deciles.csv", "topic_info": "topic_info.csv",
        "content_exposure_summary": "content_exposure_summary.csv", "sentiment_model_metrics": "sentiment_model_metrics.csv",
        "virality_playbook": "virality_playbook.csv", "virality_shap_importance": "virality_shap_importance.csv",
        "virality_weekly": "virality_weekly.csv", "impact_fan": "impact_fan.csv", "impact_distribution": "impact_distribution.csv",
        "impact_monte_carlo": "impact_monte_carlo.csv", "impact_sensitivity": "impact_sensitivity.csv",
    }
    for t, f in simple.items():
        tables[t] = pd.read_csv(C.TABLES / f)
    tables["topic_exposure_by_arm"] = tx
    tables["daily_lift"]["date"] = pd.to_datetime(tables["daily_lift"]["date"])
    tables["virality_weekly"]["week"] = pd.to_datetime(tables["virality_weekly"]["week"])
    for t in ["metric_results", "segment_results"]:
        for c in tables[t].columns:
            if tables[t][c].dtype == bool:
                tables[t][c] = tables[t][c].astype(int)
    tables["metric_results"]["metric_label"] = tables["metric_results"]["label"]

    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    for t, df in tables.items():
        out = df.copy()
        for c in out.columns:
            if np.issubdtype(out[c].dtype, np.datetime64):
                out[c] = out[c].dt.strftime("%Y-%m-%d")
        out.to_csv(DATA / f"{t}.csv", index=False)
    return tables


# ---------------------------------------------------------------- semantic model
PCT_COLS = {"rel_lift", "rel_ci_low", "rel_ci_high", "minutes_lift", "minutes_ci_low", "minutes_ci_high", "friend_int_lift",
            "revenue_lift", "variance_reduction", "mde_rel", "survival", "ci_low_surv", "churn_probability", "share",
            "negative_share", "share_of_draws", "viral_rate", "recall", "precision", "p_improves", "fraction_targeted",
            "share_ai_feed", "share_ai_feed_social", "share_control", "prob_positive", "share_negative_pred",
            "share_positive_pred", "macro_recall", "macro_f1", "accuracy", "churn_control", "churn_treatment",
            "topic_entropy_rel_change", "negative_share_rel_change", "top_topic_share", "share_35_plus"}


def _dtype(s: pd.Series) -> tuple[str, str]:
    if np.issubdtype(s.dtype, np.datetime64):
        return "dateTime", "type date"
    if np.issubdtype(s.dtype, np.integer):
        return "int64", "Int64.Type"
    if np.issubdtype(s.dtype, np.floating):
        return "double", "type number"
    return "string", "type text"


def m_expression(table: str, df: pd.DataFrame, source: str) -> list[str]:
    types = ", ".join(f'{{"{c}", {_dtype(df[c])[1]}}}' for c in df.columns)
    # Static base URL + RelativePath keeps the query refreshable in the Power BI Service (no gateway needed)
    content = (f'Web.Contents("https://raw.githubusercontent.com", [RelativePath = DataPath & "{table}.csv"])'
               if source == "github" else f'File.Contents(DataPath & "{table}.csv")')
    return [
        "let",
        f'    Source = Csv.Document({content}, [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
        "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
        '    Blanks = Table.TransformColumns(Promoted, {}, each if _ = "" then null else _),',
        f'    Typed = Table.TransformColumnTypes(Blanks, {{{types}}}, "en-US")',
        "in",
        "    Typed",
    ]


MEASURES = [
    # (name, expression, format, display folder)
    ("Users", "COUNTROWS(fact_user)", "#,0", "Experiment"),
    ("User-Days", "SUM(fact_daily[users_assigned])", "#,0", "Engagement"),
    ("Minutes per User-Day", "DIVIDE(SUM(fact_daily[minutes]), [User-Days])", "0.00", "Engagement"),
    ("DAU Rate", "DIVIDE(SUM(fact_daily[active_users]), [User-Days])", "0.0%", "Engagement"),
    ("Sessions per User-Day", "DIVIDE(SUM(fact_daily[sessions]), [User-Days])", "0.00", "Engagement"),
    ("Friend Interactions per User-Day", "DIVIDE(SUM(fact_daily[friend_interactions]), [User-Days])", "0.000", "Engagement"),
    ("Posts per User-Day", "DIVIDE(SUM(fact_daily[posts_created]), [User-Days])", "0.000", "Engagement"),
    ("Revenue per User-Day", "DIVIDE(SUM(fact_daily[ad_revenue]), [User-Days])", "$0.0000", "Monetisation"),
    ("Ad CTR", "DIVIDE(SUM(fact_daily[ad_clicks]), SUM(fact_daily[ad_impressions]))", "0.00%", "Monetisation"),
    ("Negative Feedback per 100 Min", "DIVIDE(SUM(fact_daily[negative_feedback]), SUM(fact_daily[minutes])) * 100", "0.000", "Guardrails"),
    ("Minutes per User-Day (Control)", 'CALCULATE([Minutes per User-Day], dim_arm[arm] = "control")', "0.00", "Lift vs Control"),
    ("Minutes Lift vs Control", "DIVIDE([Minutes per User-Day], [Minutes per User-Day (Control)]) - 1", "+0.0%;-0.0%;0.0%", "Lift vs Control"),
    ("Revenue Lift vs Control", 'DIVIDE([Revenue per User-Day], CALCULATE([Revenue per User-Day], dim_arm[arm] = "control")) - 1', "+0.0%;-0.0%;0.0%", "Lift vs Control"),
    ("Friend Interactions Lift vs Control", 'DIVIDE([Friend Interactions per User-Day], CALCULATE([Friend Interactions per User-Day], dim_arm[arm] = "control")) - 1', "+0.0%;-0.0%;0.0%", "Lift vs Control"),
    ("Churn Rate", "AVERAGE(fact_user[churned])", "0.00%", "Retention"),
    ("Churn Rate (Control)", 'CALCULATE([Churn Rate], dim_arm[arm] = "control")', "0.00%", "Retention"),
    ("Churn Change vs Control (pp)", "([Churn Rate] - [Churn Rate (Control)]) * 100", "+0.00;-0.00;0.00", "Retention"),
    ("Avg Churn Risk", "AVERAGE(fact_user[churn_probability])", "0.0%", "Retention"),
    ("High-Risk Users", 'CALCULATE(COUNTROWS(fact_user), fact_user[risk_tier] IN {"High", "Critical"})', "#,0", "Retention"),
    ("AI Feed Minutes Lift", 'CALCULATE(MAX(metric_results[rel_lift]), metric_results[arm] = "ai_feed", metric_results[metric] = "minutes")', "+0.0%;-0.0%", "Headline KPIs"),
    ("Friend Boost Minutes Lift", 'CALCULATE(MAX(metric_results[rel_lift]), metric_results[arm] = "ai_feed_social", metric_results[metric] = "minutes")', "+0.0%;-0.0%", "Headline KPIs"),
    ("AI Feed Churn Change", 'CALCULATE(MAX(metric_results[rel_lift]), metric_results[arm] = "ai_feed", metric_results[metric] = "churned")', "+0.0%;-0.0%", "Headline KPIs"),
    ("Friend Boost Churn Change", 'CALCULATE(MAX(metric_results[rel_lift]), metric_results[arm] = "ai_feed_social", metric_results[metric] = "churned")', "+0.0%;-0.0%", "Headline KPIs"),
    ("Friend Boost 12M Value ($M)", 'CALCULATE(MAX(impact_monte_carlo[p50_musd]), impact_monte_carlo[comparison] = "AI + friend boost vs control")', "$#,0\"M\"", "Headline KPIs"),
    ("AI Feed 12M Value ($M)", 'CALCULATE(MAX(impact_monte_carlo[p50_musd]), impact_monte_carlo[comparison] = "AI feed vs control")', "$#,0\"M\"", "Headline KPIs"),
    ("P(Friend Boost Beats AI Feed)", 'CALCULATE(MAX(impact_monte_carlo[prob_positive]), impact_monte_carlo[comparison] = "Friend boost minus AI feed")', "0.0%", "Headline KPIs"),
    ("Significant Wins", 'CALCULATE(COUNTROWS(metric_results), metric_results[verdict] = "win")', "0", "Headline KPIs"),
    ("Significant Harms", 'CALCULATE(COUNTROWS(metric_results), metric_results[verdict] = "harm")', "0", "Headline KPIs"),
    ("Best Policy Uplift ($/user/yr)", "MAX(policy_values[uplift_vs_control_usd])", "$0.00", "Headline KPIs"),
    ("Churn Model AUC", 'CALCULATE(MAX(churn_model_metrics[roc_auc]), churn_model_metrics[model] = "LightGBM + isotonic")', "0.000", "Headline KPIs"),
]

RELATIONSHIPS = [
    ("fact_daily", "date", "dim_date", "date"), ("fact_daily", "arm", "dim_arm", "arm"),
    ("fact_user", "arm", "dim_arm", "arm"), ("metric_results", "arm", "dim_arm", "arm"),
    ("segment_results", "arm", "dim_arm", "arm"), ("daily_lift", "arm", "dim_arm", "arm"),
    ("km_survival", "arm", "dim_arm", "arm"), ("impact_fan", "arm", "dim_arm", "arm"),
    ("topic_exposure_by_arm", "arm", "dim_arm", "arm"), ("content_exposure_summary", "arm", "dim_arm", "arm"),
    ("sequential_monitoring", "arm", "dim_arm", "arm"), ("impact_distribution", "arm", "dim_arm", "arm"),
]


def build_model(tables: dict[str, pd.DataFrame], data_path: str, source: str) -> dict:
    tmsl_tables = []
    for t, df in tables.items():
        cols = []
        for c in df.columns:
            dt, _ = _dtype(df[c])
            col = {"name": c, "dataType": dt, "sourceColumn": c, "summarizeBy": "none" if dt in ("string", "dateTime") or c.endswith("_id") else "sum",
                   "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{t}.{c}"))}
            if dt == "dateTime":
                col["formatString"] = "yyyy-mm-dd"
            elif c in PCT_COLS or c.endswith("_lift") or c.endswith("_rel_change"):
                col["formatString"] = "0.0%"
            elif dt == "double":
                col["formatString"] = "#,0.0000"
            if t == "dim_arm" and c == "arm_label":
                col["sortByColumn"] = "sort_order"
            cols.append(col)
        tmsl_tables.append({"name": t, "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, t)), "columns": cols,
                            "partitions": [{"name": t, "mode": "import", "source": {"type": "m", "expression": m_expression(t, df, source)}}]})
    tmsl_tables.append({
        "name": "_Measures", "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, "_Measures")),
        "columns": [{"name": "Measure Holder", "dataType": "string", "sourceColumn": "Measure Holder", "isHidden": True,
                     "summarizeBy": "none", "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, "_Measures.holder"))}],
        "measures": [{"name": n, "expression": e, "formatString": f, "displayFolder": folder,
                      "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, f"measure.{n}"))} for n, e, f, folder in MEASURES],
        "partitions": [{"name": "_Measures", "mode": "import", "source": {"type": "m", "expression": [
            "let", '    Source = #table(type table [#"Measure Holder" = text], {})', "in", "    Source"]}}],
    })
    rels = [{"name": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{a}.{b}->{c}.{d}")), "fromTable": a, "fromColumn": b,
             "toTable": c, "toColumn": d} for a, b, c, d in RELATIONSHIPS]
    return {
        "name": NAME, "compatibilityLevel": 1567,
        "model": {
            "culture": "en-US", "defaultPowerBIDataSourceVersion": "powerBI_V3", "sourceQueryCulture": "en-US",
            "dataAccessOptions": {"legacyRedirects": True, "returnErrorValuesAsNull": True},
            "tables": tmsl_tables, "relationships": rels,
            "expressions": [{"name": "DataPath", "kind": "m", "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, "DataPath")),
                             "expression": f'"{data_path}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'}],
            "annotations": [{"name": "PBI_QueryOrder", "value": json.dumps(["DataPath"] + list(tables) + ["_Measures"])},
                            {"name": "__PBI_TimeIntelligenceEnabled", "value": "0"}],
        },
    }


# ---------------------------------------------------------------- report
AGG = {"sum": 0, "avg": 1, "min": 3, "max": 4}
BASE_THEME = "CY24SU06"


class Page:
    def __init__(self, name: str):
        self.name, self.visuals = name, []

    def add(self, vtype, x, y, w, h, roles: dict, title: str | None = None, objects: dict | None = None):
        """roles: role -> list of fields; field = ('m', table, measure) | ('c', table, col) | ('a', table, col, agg)."""
        aliases, frm, select, proj = {}, [], [], {}
        for role, fields in roles.items():
            proj[role] = []
            for f in fields:
                kind, table, prop = f[0], f[1], f[2]
                if table not in aliases:
                    aliases[table] = f"t{len(aliases)}"
                    frm.append({"Name": aliases[table], "Entity": table, "Type": 0})
                src = {"SourceRef": {"Source": aliases[table]}}
                if kind == "m":
                    ref = f"{table}.{prop}"
                    sel = {"Measure": {"Expression": src, "Property": prop}, "Name": ref}
                elif kind == "c":
                    ref = f"{table}.{prop}"
                    sel = {"Column": {"Expression": src, "Property": prop}, "Name": ref}
                else:
                    fn = f[3]
                    ref = f"{fn.capitalize()}({table}.{prop})"
                    sel = {"Aggregation": {"Expression": {"Column": {"Expression": src, "Property": prop}}, "Function": AGG[fn]},
                           "Name": ref}
                if not any(s["Name"] == ref for s in select):
                    select.append(sel)
                proj[role].append({"queryRef": ref, **({"active": True} if role in ("Category", "Rows") else {})})
        sv = {"visualType": vtype, "projections": proj,
              "prototypeQuery": {"Version": 2, "From": frm, "Select": select}, "drillFilterOtherVisuals": True}
        if objects:
            sv["objects"] = objects
        if title:
            sv["vcObjects"] = {"title": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}},
                                                         "text": {"expr": {"Literal": {"Value": f"'{title}'"}}}}}]}
        self._push(x, y, w, h, sv)

    def text(self, x, y, w, h, runs: list[tuple[str, str, bool]]):
        paragraphs = [{"textRuns": [{"value": t, "textStyle": {"fontSize": size, **({"fontWeight": "bold"} if bold else {})}}]}
                      for t, size, bold in runs]
        sv = {"visualType": "textbox", "drillFilterOtherVisuals": True,
              "objects": {"general": [{"properties": {"paragraphs": paragraphs}}]}}
        self._push(x, y, w, h, sv)

    def _push(self, x, y, w, h, sv):
        z = len(self.visuals) * 1000
        vid = uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}.{len(self.visuals)}").hex[:20]
        cfg = {"name": vid, "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": z, "width": w, "height": h, "tabOrder": z}}],
               "singleVisual": sv}
        self.visuals.append({"x": x, "y": y, "z": z, "width": w, "height": h, "config": json.dumps(cfg), "filters": "[]"})

    def section(self, i):
        return {"name": f"ReportSection{uuid.uuid5(uuid.NAMESPACE_URL, self.name).hex[:18]}", "displayName": self.name,
                "displayOption": 1, "width": 1280, "height": 720, "ordinal": i, "config": json.dumps({"filterSortOrder": 3}),
                "filters": "[]",
                "visualContainers": self.visuals}


def m(t, n): return ("m", t, n)
def col(t, n): return ("c", t, n)
def agg(t, n, f="sum"): return ("a", t, n, f)


ARM = col("dim_arm", "arm_label")


def header(p: Page, title: str, sub: str):
    p.text(20, 8, 1240, 64, [(title, "20pt", True), (sub, "10pt", False)])


def build_report() -> dict:
    pages = []
    # 1. Executive summary
    p = Page("Executive Summary")
    header(p, "Feed Ranking Experiment · Launch Decision",
           "3-arm randomised test · 60,000 users · 14-day pre-period + 28-day experiment · Recommendation: ship AI feed + friend boost")
    cards = [("Users", "Users in experiment"), ("AI Feed Minutes Lift", "AI feed: minutes lift"),
             ("Friend Boost Minutes Lift", "Friend boost: minutes lift"), ("AI Feed Churn Change", "AI feed: churn change"),
             ("Friend Boost Churn Change", "Friend boost: churn change"), ("Friend Boost 12M Value ($M)", "Friend boost: 12-mo value")]
    for i, (meas, t) in enumerate(cards):
        p.add("card", 20 + i * 208, 84, 196, 110, {"Values": [m("_Measures", meas)]}, t)
    p.add("lineChart", 20, 206, 760, 250, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "Minutes per User-Day")],
                                           "Series": [ARM]}, "Daily minutes per assigned user (pre-period → experiment)")
    p.add("lineChart", 20, 466, 760, 240, {"Category": [col("daily_lift", "day")], "Y": [agg("daily_lift", "rel_lift")],
                                           "Series": [ARM]}, "Daily lift vs control: novelty decays")
    p.add("tableEx", 792, 206, 468, 500, {"Values": [col("metric_results", "arm"), col("metric_results", "metric_label"),
                                                     agg("metric_results", "rel_lift"), agg("metric_results", "p_adj_bh"),
                                                     col("metric_results", "verdict")]},
          "Metric scorecard (CUPED, BH-adjusted)")
    pages.append(p)

    # 2. Experiment scorecard & validity
    p = Page("Experiment Scorecard")
    header(p, "Experiment Scorecard & Validity", "SRM, balance, A/A calibration, CUPED power gains, and every metric with 95% CIs")
    p.add("slicer", 20, 84, 300, 120, {"Values": [ARM]}, "Arm")
    p.add("card", 340, 84, 220, 120, {"Values": [m("_Measures", "Significant Wins")]}, "Significant wins")
    p.add("card", 580, 84, 220, 120, {"Values": [m("_Measures", "Significant Harms")]}, "Significant harms (guardrails)")
    p.add("clusteredBarChart", 20, 216, 620, 490, {"Category": [col("metric_results", "metric_label")],
                                                   "Y": [agg("metric_results", "rel_lift")], "Series": [ARM]},
          "Relative lift vs control by metric")
    p.add("lineChart", 652, 84, 608, 300, {"Category": [col("power_mde", "n_per_arm")], "Y": [agg("power_mde", "mde_rel")],
                                           "Series": [col("power_mde", "method")]}, "Minimum detectable effect: raw vs CUPED")
    p.add("tableEx", 652, 396, 608, 310, {"Values": [col("metric_results", "arm"), col("metric_results", "metric_label"),
                                                     agg("metric_results", "rel_ci_low"), agg("metric_results", "rel_ci_high"),
                                                     agg("metric_results", "variance_reduction"), agg("metric_results", "p_value")]},
          "Confidence intervals & CUPED variance reduction")
    pages.append(p)

    # 3. Daily trends explorer
    p = Page("Engagement Trends")
    header(p, "Engagement Trends Explorer", "Slice the daily fact by segment. Every measure is per assigned user (intent-to-treat)")
    for i, s in enumerate(SEGMENTS):
        p.add("slicer", 20 + i * 312, 84, 300, 90, {"Values": [col("fact_daily", s)]}, s.replace("_", " ").title())
    p.add("lineChart", 20, 186, 615, 255, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "Minutes Lift vs Control")],
                                           "Series": [ARM]}, "Minutes lift vs control")
    p.add("lineChart", 645, 186, 615, 255, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "DAU Rate")],
                                            "Series": [ARM]}, "DAU rate")
    p.add("lineChart", 20, 451, 615, 255, {"Category": [col("dim_date", "date")],
                                           "Y": [m("_Measures", "Friend Interactions per User-Day")], "Series": [ARM]},
          "Friend interactions per user-day (guardrail)")
    p.add("lineChart", 645, 451, 615, 255, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "Revenue per User-Day")],
                                            "Series": [ARM]}, "Ad revenue per user-day")
    pages.append(p)

    # 4. Segments
    p = Page("Segment Deep-Dive")
    header(p, "Who Wins, Who Churns", "CUPED minutes lift and churn change by segment · BH-adjusted across all segment tests")
    p.add("slicer", 20, 84, 260, 200, {"Values": [col("segment_results", "dimension")]}, "Segment dimension")
    p.add("clusteredColumnChart", 292, 84, 968, 300, {"Category": [col("segment_results", "segment")],
                                                     "Y": [agg("segment_results", "churn_diff_pp")], "Series": [ARM]},
          "Churn change vs control (percentage points)")
    p.add("clusteredColumnChart", 20, 396, 620, 310, {"Category": [col("segment_results", "segment")],
                                                     "Y": [agg("segment_results", "minutes_lift")], "Series": [ARM]},
          "Minutes lift vs control")
    p.add("tableEx", 652, 396, 608, 310, {"Values": [col("segment_results", "dimension"), col("segment_results", "segment"),
                                                     col("segment_results", "arm"), agg("segment_results", "minutes_lift"),
                                                     agg("segment_results", "churn_diff_pp"), col("segment_results", "status")]},
          "Segment verdicts")
    pages.append(p)

    # 5. Retention
    p = Page("Retention & Churn Risk")
    header(p, "Retention & Churn Early Warning", "Kaplan-Meier by arm · leakage-free LightGBM risk model (scored day 13) · campaign economics")
    p.add("card", 20, 84, 200, 100, {"Values": [m("_Measures", "Churn Model AUC")]}, "Churn model ROC-AUC")
    p.add("card", 230, 84, 200, 100, {"Values": [m("_Measures", "High-Risk Users")]}, "High/critical-risk users")
    p.add("card", 440, 84, 200, 100, {"Values": [m("_Measures", "Churn Rate")]}, "28-day churn rate")
    p.add("lineChart", 20, 196, 620, 250, {"Category": [col("km_survival", "day")], "Y": [agg("km_survival", "survival", "avg")],
                                           "Series": [ARM]}, "Kaplan-Meier: share of users not yet churned")
    p.add("clusteredBarChart", 652, 84, 608, 362, {"Category": [col("churn_shap_importance", "feature")],
                                                   "Y": [agg("churn_shap_importance", "mean_abs_shap")]}, "Churn drivers (mean |SHAP|)")
    p.add("lineChart", 20, 456, 620, 250, {"Category": [col("retention_profit_curve", "fraction_targeted")],
                                           "Y": [agg("retention_profit_curve", "profit_at_scale_musd")]},
          "Retention campaign profit curve ($M at scale)")
    p.add("clusteredColumnChart", 652, 456, 608, 250, {"Category": [col("fact_user", "risk_tier")],
                                                      "Y": [m("_Measures", "Users")], "Series": [ARM]}, "Users by churn-risk tier")
    pages.append(p)

    # 6. HTE
    p = Page("Personalisation (HTE)")
    header(p, "Heterogeneous Effects & Rollout Policy", "DR-learner CATE on net annual user value · IPW policy evaluation on held-out users")
    p.add("card", 20, 84, 240, 100, {"Values": [m("_Measures", "Best Policy Uplift ($/user/yr)")]}, "Best policy uplift ($/user/yr)")
    p.add("clusteredBarChart", 20, 196, 620, 250, {"Category": [col("policy_values", "policy")],
                                                   "Y": [agg("policy_values", "uplift_vs_control_usd")]},
          "Net annual value per user vs control, by rollout policy")
    p.add("slicer", 272, 84, 368, 100, {"Values": [col("qini_curves", "arm")]}, "Qini: treatment arm")
    p.add("lineChart", 652, 84, 608, 362, {"Category": [col("qini_curves", "fraction_targeted")],
                                           "Y": [agg("qini_curves", "cumulative_gain")],
                                           "Series": [col("qini_curves", "model")]}, "Qini curve: model vs random targeting")
    p.add("clusteredColumnChart", 20, 456, 1240, 250, {"Category": [col("cate_deciles", "decile")],
                                                      "Y": [agg("cate_deciles", "predicted_cate"), agg("cate_deciles", "observed_uplift")]},
          "CATE calibration by decile: predicted vs observed uplift (USD/user/yr)")
    pages.append(p)

    # 7. Content
    p = Page("Content & Responsible AI")
    header(p, "Content Health & Responsible-AI Guardrails", "Real TweetEval posts · NMF topics · sentiment model · simulated ranker exposure")
    p.add("clusteredColumnChart", 20, 84, 1240, 300, {"Category": [col("topic_exposure_by_arm", "label")],
                                                     "Y": [agg("topic_exposure_by_arm", "share")], "Series": [ARM]},
          "Share of impressions by topic")
    p.add("clusteredColumnChart", 20, 396, 400, 310, {"Category": [ARM], "Y": [agg("content_exposure_summary", "negative_share")]},
          "Share of negative posts served")
    p.add("clusteredColumnChart", 432, 396, 400, 310, {"Category": [ARM], "Y": [agg("content_exposure_summary", "topic_entropy")]},
          "Feed diversity (normalised topic entropy)")
    p.add("tableEx", 844, 396, 416, 310, {"Values": [col("sentiment_model_metrics", "model"),
                                                     agg("sentiment_model_metrics", "macro_recall"),
                                                     agg("sentiment_model_metrics", "macro_f1"),
                                                     agg("sentiment_model_metrics", "accuracy")]},
          "Sentiment model benchmark (TweetEval test)")
    pages.append(p)

    # 8. Impact
    p = Page("Business Impact")
    header(p, "12-Month Business Impact", "Monte Carlo over steady-state revenue lift and churn hazard ratio · 40M eligible MAU")
    p.add("card", 20, 84, 300, 100, {"Values": [m("_Measures", "Friend Boost 12M Value ($M)")]}, "Friend boost: median 12-mo value")
    p.add("card", 330, 84, 300, 100, {"Values": [m("_Measures", "AI Feed 12M Value ($M)")]}, "AI feed: median 12-mo value")
    p.add("card", 640, 84, 300, 100, {"Values": [m("_Measures", "P(Friend Boost Beats AI Feed)")]}, "P(friend boost > AI feed)")
    p.add("lineChart", 20, 196, 620, 260, {"Category": [col("impact_fan", "month")],
                                           "Y": [agg("impact_fan", "p50")], "Series": [ARM]},
          "Cumulative incremental revenue, median ($M)")
    p.add("clusteredColumnChart", 652, 196, 608, 260, {"Category": [col("impact_distribution", "bin_mid_musd")],
                                                      "Y": [agg("impact_distribution", "share_of_draws")], "Series": [ARM]},
          "Distribution of 12-month value across simulations ($M)")
    p.add("tableEx", 20, 468, 620, 238, {"Values": [col("impact_monte_carlo", "comparison"), agg("impact_monte_carlo", "p05_musd"),
                                                    agg("impact_monte_carlo", "p50_musd"), agg("impact_monte_carlo", "p95_musd"),
                                                    agg("impact_monte_carlo", "prob_positive")]}, "Monte Carlo summary ($M)")
    p.add("clusteredBarChart", 652, 468, 608, 238, {"Category": [col("impact_sensitivity", "assumption")],
                                                    "Y": [agg("impact_sensitivity", "low_musd"), agg("impact_sensitivity", "high_musd")]},
          "Sensitivity: friend boost minus AI feed ($M)")
    pages.append(p)

    # 9. Virality
    p = Page("Creator Virality (Real Data)")
    header(p, "What Makes Content Spread", "UCI Online News Popularity · 39,644 real articles · time-split LightGBM · SHAP drivers")
    p.add("clusteredBarChart", 20, 84, 500, 622, {"Category": [col("virality_shap_importance", "feature")],
                                                  "Y": [agg("virality_shap_importance", "mean_abs_shap")]}, "Virality drivers (mean |SHAP|)")
    p.add("slicer", 532, 84, 728, 70, {"Values": [col("virality_playbook", "dimension")]}, "Playbook dimension")
    p.add("clusteredColumnChart", 532, 166, 728, 270, {"Category": [col("virality_playbook", "level")],
                                                      "Y": [agg("virality_playbook", "viral_rate", "avg")]},
          "Viral rate (top-quartile shares) by level")
    p.add("lineChart", 532, 448, 728, 258, {"Category": [col("virality_weekly", "week")],
                                            "Y": [agg("virality_weekly", "viral_rate", "avg")]}, "Weekly viral rate")
    pages.append(p)

    # Header mirrors a report.json known to render in the Power BI Service: the base theme is a built-in one
    # that the service resolves itself, and an explicit (empty) report-level filter list is required.
    config = {"version": "5.55", "themeCollection": {"baseTheme": {"name": BASE_THEME, "version": "5.56", "type": 2}},
              "activeSectionIndex": 0, "defaultDrillFilterOtherVisuals": True, "filterSortOrder": 3,
              "linguisticSchemaSyncVersion": 2,
              "settings": {"useNewFilterPaneExperience": True, "allowChangeFilterTypes": True,
                           "useStylableVisualContainerHeader": True, "queryLimitOption": 6, "useEnhancedTooltips": True,
                           "exportDataMode": 1, "useDefaultAggregateDisplayName": True},
              "objects": {"section": [{"properties": {"verticalAlignment": {"expr": {"Literal": {"Value": "'Top'"}}}}}]}}
    return {"config": json.dumps(config), "filters": "[]", "layoutOptimization": 0,
            "resourcePackages": [{"resourcePackage": {"disabled": False, "name": "SharedResources", "type": 2, "items": [
                {"name": BASE_THEME, "path": f"BaseThemes/{BASE_THEME}.json", "type": 202}]}}],
            "sections": [pg.section(i) for i, pg in enumerate(pages)]}


def theme() -> dict:
    return {"name": "SEI Analytics", "dataColors": [C.ARM_COLORS["control"], C.ARM_COLORS["ai_feed"], C.ARM_COLORS["ai_feed_social"],
                                                    "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
            "background": "#FCFCFB", "foreground": "#0B0B0B", "tableAccent": "#2a78d6",
            "good": C.STATUS["good"], "neutral": "#898781", "bad": C.STATUS["critical"],
            "maximum": "#0d366b", "center": "#f0efec", "minimum": "#cde2fb",
            "textClasses": {"title": {"fontFace": "Segoe UI Semibold", "fontSize": 12, "color": "#0B0B0B"},
                            "label": {"fontFace": "Segoe UI", "fontSize": 9, "color": "#52514E"},
                            "callout": {"fontFace": "Segoe UI Semibold", "fontSize": 26, "color": "#0B0B0B"}},
            "visualStyles": {"*": {"*": {"background": [{"color": {"solid": {"color": "#FFFFFF"}}, "transparency": 0}],
                                         "border": [{"show": True, "color": {"solid": {"color": "#E1E0D9"}}, "radius": 6}]}}}}


def platform(item_type: str) -> str:
    """Fabric Git-integration metadata; without it a synced workspace ignores the item folder."""
    return json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": item_type, "displayName": NAME},
        "config": {"version": "2.0", "logicalId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAME}.{item_type}"))},
    }, indent=2)


def write_pbip(tables, data_path: str, source: str):
    model_dir, report_dir = OUT / f"{NAME}.SemanticModel", OUT / f"{NAME}.Report"
    for d in (model_dir, report_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    (OUT / f"{NAME}.pbip").write_text(json.dumps({"version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
                                                 "settings": {"enableAutoRecovery": True}}, indent=2))
    (model_dir / "definition.pbism").write_text(json.dumps({"version": "1.0", "settings": {}}, indent=2))
    (model_dir / "model.bim").write_text(json.dumps(build_model(tables, data_path, source), indent=2))
    (model_dir / ".platform").write_text(platform("SemanticModel"))
    (report_dir / ".platform").write_text(platform("Report"))
    (report_dir / "definition.pbir").write_text(json.dumps(
        {"version": "1.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}, "byConnection": None}}, indent=2))
    (report_dir / "report.json").write_text(json.dumps(build_report(), indent=2))
    (OUT / "SEI_Theme.json").write_text(json.dumps(theme(), indent=2))
    (OUT / ".gitignore").write_text("**/.pbi/localSettings.json\n**/.pbi/cache.abf\n")
    dax = ["// DAX measures for the SEI semantic model (table: _Measures)\n"]
    for n, e, f, folder in MEASURES:
        dax.append(f"// [{folder}]  format: {f}\n{n} =\n    {e}\n")
    (OUT / "measures.dax").write_text("\n".join(dax))


def write_excel(tables: dict[str, pd.DataFrame]):
    """One workbook, one Excel table per sheet: importable by the Power BI web app (Service) on any OS."""
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    path = OUT / f"{NAME}_data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for t, df in tables.items():
            df.to_excel(xw, sheet_name=t, index=False)
            ws = xw.sheets[t]
            ref = f"A1:{get_column_letter(df.shape[1])}{len(df) + 1}"
            tab = Table(displayName=t, ref=ref)
            tab.tableStyleInfo = TableStyleInfo(name="TableStyleLight9", showRowStripes=True)
            ws.add_table(tab)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["github", "local"], default="github",
                    help="github: read CSVs from the public repo (works in Desktop AND the Power BI Service); "
                         "local: read from a folder on disk (Desktop only)")
    ap.add_argument("--github-path", default=C.GITHUB_DATA_PATH, help="owner/repo/branch/path/ to powerbi/data/")
    ap.add_argument("--data-folder", default="C:\\SEI\\powerbi\\data\\",
                    help="absolute path (with trailing slash) to powerbi/data, used with --source local")
    args = ap.parse_args()
    if args.source == "github":
        data_path = args.github_path if args.github_path.endswith("/") else args.github_path + "/"
    else:
        data_path = args.data_folder if args.data_folder.endswith(("\\", "/")) else args.data_folder + "\\"
    tables = export_data()
    write_pbip(tables, data_path, args.source)
    xlsx = write_excel(tables)
    print(f"wrote {xlsx.name} for the Power BI web app")
    total = sum(len(t) for t in tables.values())
    print(f"exported {len(tables)} tables ({total:,} rows) to {DATA} and wrote {NAME}.pbip "
          f"(source = {args.source}, DataPath = {data_path})")


if __name__ == "__main__":
    main()
