"""Power BI report layout (legacy report.json) with an embedded design system.

All styling is written into the report itself (page backgrounds, card containers, fonts, series colours),
so it renders identically in Power BI Desktop and the Power BI Service without a custom theme file.

Design system
  page      #F4F6FA background, 1280x720, 20px margins, 16px gutters
  header    deep-navy band with white title + muted subtitle
  cards     white, 10px radius, hairline border, soft shadow, visual header hidden
  type      Segoe UI Semibold titles 11pt, KPI values 24pt
  colour    arm identity fixed everywhere (control blue, AI feed orange, friend boost aqua); single-series navy
"""
from __future__ import annotations

import json
import uuid

from . import config as C

BASE_THEME = "CY24SU06"
AGG = {"sum": 0, "avg": 1, "min": 3, "max": 4}

NAVY, NAVY_2, INK, INK_2, MUTED = "#0F1B2D", "#1D3557", "#1F2937", "#4B5563", "#8A94A6"
PAGE_BG, CARD_BG, BORDER, BAND = "#F4F6FA", "#FFFFFF", "#E3E7EE", "#F7F9FC"
SERIES = "#2B4C7E"          # single-series charts (blue is reserved for the control arm)
SERIES_2 = "#9AA8BE"        # comparison series
GOOD, BAD = C.STATUS["good"], C.STATUS["critical"]
ARM_LABEL_COLORS = {C.ARM_LABELS[a]: C.ARM_COLORS[a] for a in C.ARMS}

W, H, M, G = 1280, 720, 20, 16
TOP = 92          # content starts below the header band
BOTTOM = H - M


# ---------------------------------------------------------------- literal helpers
def lit(v: str) -> dict:
    return {"expr": {"Literal": {"Value": v}}}


def txt(s: str) -> dict:
    return lit("'" + s.replace("'", "''") + "'")


def num(x: float) -> dict:
    return lit(f"{x}D")


def flag(b: bool) -> dict:
    return lit("true" if b else "false")


def colour(hex_: str) -> dict:
    return {"solid": {"color": txt(hex_)}}


def props(**kw) -> list[dict]:
    return [{"properties": kw}]


def series_colours(entity: str, prop: str, mapping: dict[str, str]) -> list[dict]:
    """dataPoint fill per category/series value, so identity colours never depend on sort order."""
    return [{"properties": {"fill": colour(c)},
             "selector": {"data": [{"scopeId": {"Comparison": {
                 "ComparisonKind": 0,
                 "Left": {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}},
                 "Right": {"Literal": {"Value": "'" + v + "'"}}}}}]}}
            for v, c in mapping.items()]


def measure_colours(mapping: dict[str, str]) -> list[dict]:
    """dataPoint fill per measure/aggregation (multi-Y charts); keys are queryRefs."""
    return [{"properties": {"fill": colour(c)}, "selector": {"metadata": ref}} for ref, c in mapping.items()]


ARM_COLOURS = series_colours("dim_arm", "arm_label", ARM_LABEL_COLORS)


def card_container(title: str | None) -> dict:
    vc = {
        "background": props(show=flag(True), color=colour(CARD_BG), transparency=num(0)),
        "border": props(show=flag(True), color=colour(BORDER), radius=num(10)),
        "dropShadow": props(show=flag(True), color=colour(NAVY), transparency=num(90), position=txt("Outer"),
                            preset=txt("Custom"), shadowBlur=num(14), shadowDistance=num(3), angle=num(90),
                            shadowSpread=num(0)),
        "visualHeader": props(show=flag(False)),
        "padding": props(top=num(10), bottom=num(8), left=num(12), right=num(12)),
    }
    if title:
        vc["title"] = props(show=flag(True), text=txt(title), fontColor=colour(INK), fontSize=num(11),
                            fontFamily=txt("Segoe UI Semibold"), alignment=txt("left"))
    else:
        vc["title"] = props(show=flag(False))
    return vc


def axes(value_title: bool = False) -> dict:
    return {
        "categoryAxis": props(show=flag(True), labelColor=colour(INK_2), fontSize=num(9), showAxisTitle=flag(False),
                              gridlineShow=flag(False)),
        "valueAxis": props(show=flag(True), labelColor=colour(MUTED), fontSize=num(9), showAxisTitle=flag(value_title),
                           gridlineShow=flag(True), gridlineColor=colour("#EDF0F5"), gridlineThickness=num(1)),
        "legend": props(show=flag(True), position=txt("TopLeft"), labelColor=colour(INK_2), fontSize=num(9),
                        showTitle=flag(False)),
    }


def line_objects(colours: list[dict] | None = None) -> dict:
    o = axes()
    o["lineStyles"] = props(strokeWidth=num(3), showMarker=flag(False), lineChartType=txt("smooth"))
    o["dataPoint"] = colours or [{"properties": {"fill": colour(SERIES)}}]
    return o


def bar_objects(colours: list[dict] | None = None, labels: bool = True, fmt_units: int = 0) -> dict:
    o = axes()
    o["dataPoint"] = colours or [{"properties": {"fill": colour(SERIES)}}]
    o["labels"] = props(show=flag(labels), color=colour(INK_2), fontSize=num(8), labelDisplayUnits=num(fmt_units))
    return o


def table_objects() -> dict:
    return {
        "columnHeaders": props(fontColor=colour("#FFFFFF"), backColor=colour(NAVY_2), fontSize=num(9),
                               fontFamily=txt("Segoe UI Semibold"), wordWrap=flag(True)),
        "values": props(fontColorPrimary=colour(INK), backColorPrimary=colour(CARD_BG),
                        fontColorSecondary=colour(INK), backColorSecondary=colour(BAND), fontSize=num(9)),
        "grid": props(gridVertical=flag(False), gridHorizontal=flag(True), gridHorizontalColor=colour("#EEF1F6"),
                      rowPadding=num(5), outlineColor=colour(BORDER)),
        "total": props(totals=flag(False)),
    }


def card_objects(value_colour: str = NAVY) -> dict:
    return {"labels": props(color=colour(value_colour), fontSize=num(24), fontFamily=txt("Segoe UI Semibold")),
            "categoryLabels": props(show=flag(False))}


def slicer_objects() -> dict:
    return {"data": props(mode=txt("Dropdown")),
            "header": props(show=flag(False)),
            "items": props(fontColor=colour(INK), fontSize=num(10), background=colour(CARD_BG))}


# ---------------------------------------------------------------- page model
def m(t, n): return ("m", t, n)
def col(t, n): return ("c", t, n)
def agg(t, n, f="sum"): return ("a", t, n, f)


ARM = col("dim_arm", "arm_label")


class Page:
    def __init__(self, name: str):
        self.name, self.visuals = name, []

    def add(self, vtype, x, y, w, h, roles: dict, title: str | None = None, objects: dict | None = None,
            container: dict | None = None):
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
              "prototypeQuery": {"Version": 2, "From": frm, "Select": select}, "drillFilterOtherVisuals": True,
              "vcObjects": container if container is not None else card_container(title)}
        if objects:
            sv["objects"] = objects
        self._push(x, y, w, h, sv)

    def text(self, x, y, w, h, paragraphs: list[list[tuple[str, dict]]], bg: str | None = None,
             border: str | None = None, radius: int = 10):
        """paragraphs: list of paragraphs, each a list of (text, textStyle) runs."""
        paras = [{"textRuns": [{"value": t, "textStyle": style} for t, style in runs]} for runs in paragraphs]
        vc = {"visualHeader": props(show=flag(False)), "title": props(show=flag(False)),
              "padding": props(top=num(8), bottom=num(6), left=num(16), right=num(16))}
        if bg:
            vc["background"] = props(show=flag(True), color=colour(bg), transparency=num(0))
        if border:
            vc["border"] = props(show=flag(True), color=colour(border), radius=num(radius))
        sv = {"visualType": "textbox", "drillFilterOtherVisuals": True,
              "objects": {"general": [{"properties": {"paragraphs": paras}}]}, "vcObjects": vc}
        self._push(x, y, w, h, sv)

    def _push(self, x, y, w, h, sv):
        z = len(self.visuals) * 1000
        vid = uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}.{len(self.visuals)}").hex[:20]
        cfg = {"name": vid, "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": z, "width": w, "height": h, "tabOrder": z}}],
               "singleVisual": sv}
        self.visuals.append({"x": x, "y": y, "z": z, "width": w, "height": h, "config": json.dumps(cfg), "filters": "[]"})

    def section(self, i):
        cfg = {"filterSortOrder": 3, "objects": {
            "background": props(color=colour(PAGE_BG), transparency=num(0)),
            "outspace": props(color=colour("#E9EDF3"), transparency=num(0)),
            "outspacePane": props(backgroundColor=colour(CARD_BG), foregroundColor=colour(INK), transparency=num(0))}}
        return {"name": f"ReportSection{uuid.uuid5(uuid.NAMESPACE_URL, self.name).hex[:18]}", "displayName": self.name,
                "displayOption": 1, "width": W, "height": H, "ordinal": i, "config": json.dumps(cfg), "filters": "[]",
                "visualContainers": self.visuals}


def style(size: int, colour_: str, bold: bool = False) -> dict:
    s = {"fontSize": f"{size}pt", "color": colour_, "fontFamily": "Segoe UI Semibold" if bold else "Segoe UI"}
    if bold:
        s["fontWeight"] = "bold"
    return s


def header(p: Page, kicker: str, title: str, sub: str):
    p.text(0, 0, W, 76, [[(kicker.upper(), style(8, "#7FA7D9", True))],
                         [(title, style(18, "#FFFFFF", True))],
                         [(sub, style(9, "#B7C3D6"))]], bg=NAVY, radius=0)


def grid(n: int, width: int = W - 2 * M, x0: int = M) -> list[tuple[int, int]]:
    """n equal columns -> [(x, w)]."""
    w = (width - (n - 1) * G) / n
    return [(round(x0 + i * (w + G)), round(w)) for i in range(n)]


def kpis(p: Page, y: int, h: int, items: list[tuple[str, str, str]], width: int = W - 2 * M, x0: int = M):
    """items: (measure, title, value colour)."""
    for (x, w), (meas, title, c) in zip(grid(len(items), width, x0), items):
        p.add("card", x, y, w, h, {"Values": [m("_Measures", meas)]}, title, card_objects(c))


# ---------------------------------------------------------------- pages
def build_report() -> dict:
    L, R = grid(2)
    pages = []

    # 1 ── Executive summary
    p = Page("Executive Summary")
    header(p, "Feed ranking experiment · launch review",
           "Ship the AI feed + friend boost, not the pure AI feed",
           "3-arm randomised test · 60,000 users · 14-day pre-period + 28-day experiment · CUPED, BH-adjusted")
    p.text(M, TOP, W - 2 * M, 50, [[("✔  RECOMMENDATION   ", style(9, GOOD, True)),
                                    ("The AI feed's headline lift is mostly novelty and costs +17% churn. The friend-boost "
                                     "variant keeps most of the engagement, cuts churn 11%, and is worth ~$103M vs ~$57M over 12 months.",
                                     style(10, INK))]], bg="#EAF6EC", border="#BFE3C5")
    kpis(p, TOP + 66, 96, [("Users", "Users in experiment", NAVY),
                           ("AI Feed Minutes Lift", "AI feed · minutes lift", NAVY),
                           ("Friend Boost Minutes Lift", "Friend boost · minutes lift", NAVY),
                           ("AI Feed Churn Change", "AI feed · churn change", BAD),
                           ("Friend Boost Churn Change", "Friend boost · churn change", GOOD),
                           ("Friend Boost 12M Value ($M)", "Friend boost · 12-month value", GOOD)])
    y2 = TOP + 66 + 96 + G
    lw, rx, rw = 760, M + 760 + G, W - 2 * M - 760 - G
    h2 = (BOTTOM - y2 - G) // 2
    p.add("lineChart", M, y2, lw, h2, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "Minutes per User-Day")],
                                       "Series": [ARM]}, "Daily minutes per assigned user · pre-period → experiment",
          line_objects(ARM_COLOURS))
    p.add("lineChart", M, y2 + h2 + G, lw, h2, {"Category": [col("daily_lift", "day")], "Y": [agg("daily_lift", "rel_lift")],
                                                "Series": [ARM]}, "Lift vs control by experiment day · novelty decays",
          line_objects(ARM_COLOURS))
    p.add("tableEx", rx, y2, rw, BOTTOM - y2, {"Values": [col("metric_results", "arm"), col("metric_results", "metric_label"),
                                                          agg("metric_results", "rel_lift"), agg("metric_results", "p_adj_bh"),
                                                          col("metric_results", "verdict")]},
          "Metric scorecard", table_objects())
    pages.append(p)

    # 2 ── Experiment scorecard
    p = Page("Experiment Scorecard")
    header(p, "Experiment validity & results", "Every metric, every arm, with uncertainty",
           "SRM χ² p = 0.10 · max |SMD| 0.019 · A/A false-positive rate 5.1% · CUPED removes ~68% of variance")
    (sx, sw), (c1x, c1w), (c2x, c2w) = grid(3, L[1], L[0])
    p.add("slicer", sx, TOP, sw, 96, {"Values": [ARM]}, "Arm", slicer_objects())
    p.add("card", c1x, TOP, c1w, 96, {"Values": [m("_Measures", "Significant Wins")]}, "Significant wins", card_objects(GOOD))
    p.add("card", c2x, TOP, c2w, 96, {"Values": [m("_Measures", "Significant Harms")]}, "Guardrail harms", card_objects(BAD))
    p.add("clusteredBarChart", L[0], TOP + 96 + G, L[1], BOTTOM - TOP - 96 - G,
          {"Category": [col("metric_results", "metric_label")], "Y": [agg("metric_results", "rel_lift")], "Series": [ARM]},
          "Relative lift vs control", bar_objects(ARM_COLOURS))
    hh = (BOTTOM - TOP - G) // 2
    p.add("lineChart", R[0], TOP, R[1], hh, {"Category": [col("power_mde", "n_per_arm")], "Y": [agg("power_mde", "mde_rel")],
                                             "Series": [col("power_mde", "method")]},
          "Minimum detectable effect by sample size · raw vs CUPED",
          line_objects(series_colours("power_mde", "method", {"raw": SERIES_2, "cuped": SERIES})))
    p.add("tableEx", R[0], TOP + hh + G, R[1], BOTTOM - TOP - hh - G,
          {"Values": [col("metric_results", "arm"), col("metric_results", "metric_label"), agg("metric_results", "rel_ci_low"),
                      agg("metric_results", "rel_ci_high"), agg("metric_results", "variance_reduction")]},
          "95% confidence intervals & CUPED variance reduction", table_objects())
    pages.append(p)

    # 3 ── Engagement trends
    p = Page("Engagement Trends")
    header(p, "Engagement explorer", "Slice every daily metric by segment",
           "All measures are per assigned user-day (intent-to-treat), so they stay correct under any filter")
    for (x, w), s in zip(grid(4), ["age_group", "region", "device", "account_type"]):
        p.add("slicer", x, TOP, w, 64, {"Values": [col("fact_daily", s)]}, s.replace("_", " ").title(), slicer_objects())
    y = TOP + 64 + G
    hh = (BOTTOM - y - G) // 2
    for (x, w), (yy, meas, t) in zip([L, R, L, R], [
            (y, "Minutes Lift vs Control", "Minutes lift vs control"),
            (y, "DAU Rate", "DAU rate"),
            (y + hh + G, "Friend Interactions per User-Day", "Friend interactions per user-day · guardrail"),
            (y + hh + G, "Revenue per User-Day", "Ad revenue per user-day")]):
        p.add("lineChart", x, yy, w, hh, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", meas)], "Series": [ARM]},
              t, line_objects(ARM_COLOURS))
    pages.append(p)

    # 4 ── Segments
    p = Page("Segment Deep-Dive")
    header(p, "Heterogeneity", "Who wins, who churns",
           "CUPED minutes lift and churn change by segment · Benjamini-Hochberg across all segment tests")
    p.add("slicer", M, TOP, 300, 64, {"Values": [col("segment_results", "dimension")]}, "Segment dimension", slicer_objects())
    p.text(M + 300 + G, TOP, W - 2 * M - 300 - G, 64,
           [[("Reading the charts  ", style(9, INK, True)),
             ("Churn harm on the pure AI feed concentrates in users aged 35+ (age × treatment interaction p = 0.0025). "
              "The friend boost lowers churn in every age group.", style(9, INK_2))]], bg=CARD_BG, border=BORDER)
    y = TOP + 64 + G
    h1 = 270
    p.add("clusteredColumnChart", M, y, W - 2 * M, h1, {"Category": [col("segment_results", "segment")],
                                                        "Y": [agg("segment_results", "churn_diff_pp")], "Series": [ARM]},
          "Churn change vs control (percentage points)", bar_objects(ARM_COLOURS))
    y2 = y + h1 + G
    p.add("clusteredColumnChart", L[0], y2, L[1], BOTTOM - y2, {"Category": [col("segment_results", "segment")],
                                                                "Y": [agg("segment_results", "minutes_lift")], "Series": [ARM]},
          "Minutes lift vs control", bar_objects(ARM_COLOURS, labels=False))
    p.add("tableEx", R[0], y2, R[1], BOTTOM - y2,
          {"Values": [col("segment_results", "segment"), col("segment_results", "arm"), agg("segment_results", "minutes_lift"),
                      agg("segment_results", "churn_diff_pp"), col("segment_results", "status")]},
          "Segment verdicts", table_objects())
    pages.append(p)

    # 5 ── Retention
    p = Page("Retention & Churn Risk")
    header(p, "Retention", "Survival by arm and a churn early-warning system",
           "Kaplan-Meier · Cox PH (AI feed HR 1.19, friend boost HR 0.88) · leakage-free LightGBM scored at day 13")
    kpis(p, TOP, 96, [("Churn Model AUC", "Churn model ROC-AUC", NAVY), ("Churn Rate", "28-day churn rate", NAVY),
                      ("Avg Churn Risk", "Average predicted risk", NAVY), ("High-Risk Users", "High / critical-risk users", BAD)])
    y = TOP + 96 + G
    hh = (BOTTOM - y - G) // 2
    p.add("lineChart", L[0], y, L[1], hh, {"Category": [col("km_survival", "day")], "Y": [agg("km_survival", "survival", "avg")],
                                           "Series": [ARM]}, "Kaplan-Meier · share of users not yet churned",
          line_objects(ARM_COLOURS))
    p.add("clusteredBarChart", R[0], y, R[1], hh, {"Category": [col("churn_shap_importance", "feature")],
                                                   "Y": [agg("churn_shap_importance", "mean_abs_shap")]},
          "What drives churn risk · mean |SHAP|", bar_objects(labels=False))
    p.add("areaChart", L[0], y + hh + G, L[1], hh, {"Category": [col("retention_profit_curve", "fraction_targeted")],
                                                    "Y": [agg("retention_profit_curve", "profit_at_scale_musd")]},
          "Retention campaign profit curve · $M at scale", line_objects())
    p.add("clusteredColumnChart", R[0], y + hh + G, R[1], hh, {"Category": [col("fact_user", "risk_tier")],
                                                              "Y": [m("_Measures", "Users")], "Series": [ARM]},
          "Users by churn-risk tier", bar_objects(ARM_COLOURS, labels=False))
    pages.append(p)

    # 6 ── Personalisation
    p = Page("Personalisation (HTE)")
    header(p, "Causal ML", "Heterogeneous effects & rollout policy",
           "Cross-fitted DR-learner on net annual user value · IPW off-policy evaluation on 35% held-out users")
    (k1x, k1w), (sx, sw) = grid(2, L[1], L[0])
    p.add("card", k1x, TOP, k1w, 96, {"Values": [m("_Measures", "Best Policy Uplift ($/user/yr)")]},
          "Best policy uplift · $/user/yr", card_objects(GOOD))
    p.add("slicer", sx, TOP, sw, 96, {"Values": [col("qini_curves", "arm")]}, "Qini curve · treatment arm", slicer_objects())
    p.text(R[0], TOP, R[1], 96, [[("How to read this  ", style(9, INK, True)),
                                  ("The AI feed's effect is strongly heterogeneous (targetable); the friend boost's is nearly "
                                   "uniform, so a blanket rollout is low-regret. The model-targeted policy leads, but its "
                                   "CI overlaps the blanket friend-boost launch.", style(9, INK_2))]], bg=CARD_BG, border=BORDER)
    y = TOP + 96 + G
    hh = (BOTTOM - y - G) // 2
    p.add("clusteredBarChart", L[0], y, L[1], hh, {"Category": [col("policy_values", "policy")],
                                                   "Y": [agg("policy_values", "uplift_vs_control_usd")]},
          "Net annual value per user vs control · by rollout policy", bar_objects())
    p.add("lineChart", R[0], y, R[1], hh, {"Category": [col("qini_curves", "fraction_targeted")],
                                           "Y": [agg("qini_curves", "cumulative_gain")], "Series": [col("qini_curves", "model")]},
          "Qini curve · cumulative gain from targeting",
          line_objects(series_colours("qini_curves", "model", {"DR-learner": SERIES, "T-learner": SERIES_2})))
    p.add("clusteredColumnChart", M, y + hh + G, W - 2 * M, hh,
          {"Category": [col("cate_deciles", "decile")],
           "Y": [agg("cate_deciles", "predicted_cate"), agg("cate_deciles", "observed_uplift")]},
          "CATE calibration by decile · predicted vs observed uplift ($/user/yr)",
          bar_objects(measure_colours({"Sum(cate_deciles.predicted_cate)": SERIES_2,
                                       "Sum(cate_deciles.observed_uplift)": SERIES}), labels=False))
    pages.append(p)

    # 7 ── Content health
    p = Page("Content & Responsible AI")
    header(p, "Responsible AI", "Content health guardrails",
           "Real TweetEval posts · NMF topics · TF-IDF sentiment model (macro-recall 0.628 vs VADER 0.570) · simulated ranker exposure")
    h1 = 300
    p.add("clusteredColumnChart", M, TOP, W - 2 * M, h1, {"Category": [col("topic_exposure_by_arm", "label")],
                                                          "Y": [agg("topic_exposure_by_arm", "share")], "Series": [ARM]},
          "Share of impressions by topic", bar_objects(ARM_COLOURS, labels=False))
    y = TOP + h1 + G
    (ax, aw), (bx, bw), (cx, cw) = grid(3)
    p.add("clusteredColumnChart", ax, y, aw, BOTTOM - y, {"Category": [ARM], "Y": [agg("content_exposure_summary", "negative_share")]},
          "Share of negative posts served", bar_objects(ARM_COLOURS))
    p.add("clusteredColumnChart", bx, y, bw, BOTTOM - y, {"Category": [ARM], "Y": [agg("content_exposure_summary", "topic_entropy")]},
          "Feed diversity · normalised topic entropy", bar_objects(ARM_COLOURS))
    p.add("tableEx", cx, y, cw, BOTTOM - y, {"Values": [col("sentiment_model_metrics", "model"),
                                                       agg("sentiment_model_metrics", "macro_recall"),
                                                       agg("sentiment_model_metrics", "macro_f1"),
                                                       agg("sentiment_model_metrics", "accuracy")]},
          "Sentiment model · TweetEval test set", table_objects())
    pages.append(p)

    # 8 ── Business impact
    p = Page("Business Impact")
    header(p, "Decision economics", "12-month business impact under uncertainty",
           "20,000 Monte Carlo draws over post-novelty revenue lift and Cox churn hazard ratios · 40M eligible MAU")
    kpis(p, TOP, 96, [("Friend Boost 12M Value ($M)", "Friend boost · median 12-month value", GOOD),
                      ("AI Feed 12M Value ($M)", "AI feed · median 12-month value", NAVY),
                      ("P(Friend Boost Beats AI Feed)", "P(friend boost beats AI feed)", GOOD)])
    y = TOP + 96 + G
    hh = (BOTTOM - y - G) // 2
    p.add("lineChart", L[0], y, L[1], hh, {"Category": [col("impact_fan", "month")], "Y": [agg("impact_fan", "p50")],
                                           "Series": [ARM]}, "Cumulative incremental revenue · median, $M",
          line_objects(ARM_COLOURS))
    p.add("clusteredColumnChart", R[0], y, R[1], hh, {"Category": [col("impact_distribution", "bin_mid_musd")],
                                                      "Y": [agg("impact_distribution", "share_of_draws")], "Series": [ARM]},
          "Distribution of 12-month value across simulations · $M", bar_objects(ARM_COLOURS, labels=False))
    p.add("tableEx", L[0], y + hh + G, L[1], hh,
          {"Values": [col("impact_monte_carlo", "comparison"), agg("impact_monte_carlo", "p05_musd"),
                      agg("impact_monte_carlo", "p50_musd"), agg("impact_monte_carlo", "p95_musd"),
                      agg("impact_monte_carlo", "prob_positive")]}, "Monte Carlo summary · $M", table_objects())
    p.add("clusteredBarChart", R[0], y + hh + G, R[1], hh,
          {"Category": [col("impact_sensitivity", "assumption")],
           "Y": [agg("impact_sensitivity", "low_musd"), agg("impact_sensitivity", "high_musd")]},
          "Sensitivity · friend boost minus AI feed, $M (low / high)",
          bar_objects(measure_colours({"Sum(impact_sensitivity.low_musd)": SERIES_2,
                                       "Sum(impact_sensitivity.high_musd)": SERIES}), labels=False))
    pages.append(p)

    # 9 ── Creator virality
    p = Page("Creator Virality (Real Data)")
    header(p, "Creator insights · real data", "What makes content spread",
           "UCI Online News Popularity · 39,644 Mashable articles · time-split LightGBM (AUC 0.732) · SHAP drivers")
    lw = 480
    p.add("clusteredBarChart", M, TOP, lw, BOTTOM - TOP, {"Category": [col("virality_shap_importance", "feature")],
                                                          "Y": [agg("virality_shap_importance", "mean_abs_shap")]},
          "Virality drivers · mean |SHAP|", bar_objects(labels=False))
    rx, rw = M + lw + G, W - 2 * M - lw - G
    p.add("slicer", rx, TOP, rw, 64, {"Values": [col("virality_playbook", "dimension")]}, "Playbook dimension", slicer_objects())
    y = TOP + 64 + G
    hh = (BOTTOM - y - G) // 2
    p.add("clusteredColumnChart", rx, y, rw, hh, {"Category": [col("virality_playbook", "level")],
                                                  "Y": [agg("virality_playbook", "viral_rate", "avg")]},
          "Viral rate · share of articles in the top quartile of shares", bar_objects())
    p.add("areaChart", rx, y + hh + G, rw, hh, {"Category": [col("virality_weekly", "week")],
                                                "Y": [agg("virality_weekly", "viral_rate", "avg")]},
          "Weekly viral rate", line_objects())
    pages.append(p)

    config = {"version": "5.55", "themeCollection": {"baseTheme": {"name": BASE_THEME, "version": "5.56", "type": 2}},
              "activeSectionIndex": 0, "defaultDrillFilterOtherVisuals": True, "filterSortOrder": 3,
              "linguisticSchemaSyncVersion": 2,
              "settings": {"useNewFilterPaneExperience": True, "allowChangeFilterTypes": True,
                           "useStylableVisualContainerHeader": True, "queryLimitOption": 6, "useEnhancedTooltips": True,
                           "exportDataMode": 1, "useDefaultAggregateDisplayName": True},
              "objects": {"section": [{"properties": {"verticalAlignment": {"expr": {"Literal": {"Value": "'Top'"}}}}}],
                          "outspacePane": [{"properties": {"expanded": {"expr": {"Literal": {"Value": "false"}}}}}]}}
    return {"config": json.dumps(config), "filters": "[]", "layoutOptimization": 0,
            "resourcePackages": [{"resourcePackage": {"disabled": False, "name": "SharedResources", "type": 2, "items": [
                {"name": BASE_THEME, "path": f"BaseThemes/{BASE_THEME}.json", "type": 202}]}}],
            "sections": [pg.section(i) for i, pg in enumerate(pages)]}
