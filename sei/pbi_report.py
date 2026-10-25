"""Power BI report layout (legacy report.json) with an embedded design system.

All styling is written into the report itself (page backgrounds, card containers, fonts, series colours),
so it renders identically in Power BI Desktop and the Power BI Service without a custom theme file.

Design system
  page      #F4F6FA background, 1280x720, 24px margins, 20px gutters
  header    deep-navy band: kicker, title, subtitle
  cards     white, 10px radius, hairline border, soft shadow, visual header hidden
  type      Power BI's Segoe UI font stack (with web fallbacks, so Macs never drop to serif)
  colour    arm identity fixed everywhere (control blue, AI feed orange, friend boost aqua); single-series navy
  text      short display columns for every category axis and table (see powerbi.add_display_columns)
"""
from __future__ import annotations

import json
import uuid

from . import config as C

BASE_THEME = "CY24SU06"
AGG = {"sum": 0, "avg": 1, "min": 3, "max": 4}

FONT = "'Segoe UI', wf_segoe-ui_normal, helvetica, arial, sans-serif"
FONT_SB = "'Segoe UI Semibold', wf_segoe-ui_semibold, helvetica, arial, sans-serif"
FONT_B = "'Segoe UI Bold', wf_segoe-ui_bold, helvetica, arial, sans-serif"

NAVY, NAVY_2, INK, INK_2, MUTED = "#0F1B2D", "#1D3557", "#1F2937", "#4B5563", "#8A94A6"
PAGE_BG, CARD_BG, BORDER, BAND = "#F4F6FA", "#FFFFFF", "#E3E7EE", "#F7F9FC"
SERIES = "#2B4C7E"          # single-series charts (blue is reserved for the control arm)
SERIES_2 = "#9AA8BE"        # comparison series
GOOD, BAD = C.STATUS["good"], C.STATUS["critical"]
ARM_LABEL_COLORS = {C.ARM_LABELS[a]: C.ARM_COLORS[a] for a in C.ARMS}

W, H, M, G = 1280, 720, 24, 20
BAND_H = 96
TOP = BAND_H + 20
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


def visual_filter(table: str, column: str, values: list[str]) -> dict:
    return {"name": uuid.uuid5(uuid.NAMESPACE_URL, f"{table}.{column}.{values}").hex[:20],
            "expression": {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": column}},
            "filter": {"Version": 2, "From": [{"Name": "f", "Entity": table, "Type": 0}],
                       "Where": [{"Condition": {"In": {
                           "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": column}}],
                           "Values": [[{"Literal": {"Value": "'" + v + "'"}}] for v in values]}}}]},
            "type": "Categorical", "howCreated": 1}


# ---------------------------------------------------------------- object presets
def card_container(title: str | None) -> dict:
    vc = {
        "background": props(show=flag(True), color=colour(CARD_BG), transparency=num(0)),
        "border": props(show=flag(True), color=colour(BORDER), radius=num(10)),
        "dropShadow": props(show=flag(True), color=colour(NAVY), transparency=num(92), position=txt("Outer"),
                            preset=txt("Custom"), shadowBlur=num(12), shadowDistance=num(2), angle=num(90),
                            shadowSpread=num(0)),
        "visualHeader": props(show=flag(False)),
    }
    if title:
        vc["title"] = props(show=flag(True), text=txt(title), fontColor=colour(INK), fontSize=num(11),
                            fontFamily=txt(FONT_SB), alignment=txt("left"))
    else:
        vc["title"] = props(show=flag(False))
    return vc


def axes(legend: bool) -> dict:
    return {
        "categoryAxis": props(show=flag(True), labelColor=colour(INK_2), fontSize=num(9), fontFamily=txt(FONT),
                              showAxisTitle=flag(False), gridlineShow=flag(False), maxMarginFactor=num(35)),
        "valueAxis": props(show=flag(True), labelColor=colour(MUTED), fontSize=num(8), fontFamily=txt(FONT),
                           showAxisTitle=flag(False), gridlineShow=flag(True), gridlineColor=colour("#EDF0F5"),
                           gridlineThickness=num(1)),
        "legend": props(show=flag(legend), position=txt("Top"), labelColor=colour(INK_2), fontSize=num(9),
                        fontFamily=txt(FONT), showTitle=flag(False)),
    }


def line_objects(colours: list[dict] | None = None, legend: bool = True) -> dict:
    o = axes(legend)
    o["lineStyles"] = props(strokeWidth=num(3), showMarker=flag(False))
    o["dataPoint"] = colours or [{"properties": {"fill": colour(SERIES)}}]
    return o


def bar_objects(colours: list[dict] | None = None, labels: bool = False, legend: bool = True) -> dict:
    o = axes(legend)
    o["dataPoint"] = colours or [{"properties": {"fill": colour(SERIES)}}]
    o["labels"] = props(show=flag(labels), color=colour(INK_2), fontSize=num(8), fontFamily=txt(FONT))
    return o


def table_objects() -> dict:
    return {
        "columnHeaders": props(fontColor=colour("#FFFFFF"), backColor=colour(NAVY_2), fontSize=num(9),
                               fontFamily=txt(FONT_SB), wordWrap=flag(True), alignment=txt("Left")),
        "values": props(fontColorPrimary=colour(INK), backColorPrimary=colour(CARD_BG), fontColorSecondary=colour(INK),
                        backColorSecondary=colour(BAND), fontSize=num(9), fontFamily=txt(FONT), wordWrap=flag(False)),
        "grid": props(gridVertical=flag(False), gridHorizontal=flag(True), gridHorizontalColor=colour("#EEF1F6"),
                      rowPadding=num(4), outlineColor=colour(BORDER), textSize=num(9)),
        "total": props(totals=flag(False)),
    }


def card_objects(value_colour: str = NAVY) -> dict:
    return {"labels": props(color=colour(value_colour), fontSize=num(22), fontFamily=txt(FONT_SB)),
            "categoryLabels": props(show=flag(False))}


def slicer_objects() -> dict:
    return {"data": props(mode=txt("Dropdown")),
            "header": props(show=flag(False)),
            "items": props(fontColor=colour(INK), fontSize=num(10), fontFamily=txt(FONT), background=colour(CARD_BG))}


# ---------------------------------------------------------------- page model
def m(t, n): return ("m", t, n)
def col(t, n): return ("c", t, n)
def agg(t, n, f="sum"): return ("a", t, n, f)


ARM = col("dim_arm", "arm_label")


class Page:
    def __init__(self, name: str):
        self.name, self.visuals = name, []

    def add(self, vtype, x, y, w, h, roles: dict, title: str | None = None, objects: dict | None = None,
            filters: list[dict] | None = None):
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
                    sel = {"Measure": {"Expression": src, "Property": prop}, "Name": ref, "NativeReferenceName": prop}
                elif kind == "c":
                    ref = f"{table}.{prop}"
                    sel = {"Column": {"Expression": src, "Property": prop}, "Name": ref, "NativeReferenceName": prop}
                else:
                    fn = f[3]
                    ref = f"{fn.capitalize()}({table}.{prop})"
                    sel = {"Aggregation": {"Expression": {"Column": {"Expression": src, "Property": prop}}, "Function": AGG[fn]},
                           "Name": ref, "NativeReferenceName": prop}
                if not any(s["Name"] == ref for s in select):
                    select.append(sel)
                proj[role].append({"queryRef": ref, **({"active": True} if role in ("Category", "Rows") else {})})
        sv = {"visualType": vtype, "projections": proj,
              "prototypeQuery": {"Version": 2, "From": frm, "Select": select}, "drillFilterOtherVisuals": True,
              "vcObjects": card_container(title)}
        if objects:
            sv["objects"] = objects
        self._push(x, y, w, h, sv, filters)

    def text(self, x, y, w, h, paragraphs: list[list[tuple[str, dict]]], bg: str | None = None,
             border: str | None = None, radius: int = 10):
        """paragraphs: list of paragraphs, each a list of (text, textStyle) runs."""
        paras = [{"textRuns": [{"value": t, "textStyle": s} for t, s in runs]} for runs in paragraphs]
        vc = {"visualHeader": props(show=flag(False)), "title": props(show=flag(False))}
        vc["background"] = props(show=flag(bool(bg)), color=colour(bg or CARD_BG), transparency=num(0))
        vc["border"] = props(show=flag(bool(border)), color=colour(border or BORDER), radius=num(radius))
        sv = {"visualType": "textbox", "drillFilterOtherVisuals": True,
              "objects": {"general": [{"properties": {"paragraphs": paras}}]}, "vcObjects": vc}
        self._push(x, y, w, h, sv)

    def _push(self, x, y, w, h, sv, filters=None):
        z = len(self.visuals) * 1000
        vid = uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}.{len(self.visuals)}").hex[:20]
        cfg = {"name": vid, "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": z, "width": w, "height": h, "tabOrder": z}}],
               "singleVisual": sv}
        self.visuals.append({"x": x, "y": y, "z": z, "width": w, "height": h, "config": json.dumps(cfg),
                             "filters": json.dumps(filters or [])})

    def section(self, i):
        cfg = {"filterSortOrder": 3, "objects": {
            "background": props(color=colour(PAGE_BG), transparency=num(0)),
            "outspace": props(color=colour("#E9EDF3"), transparency=num(0))}}
        return {"name": f"ReportSection{uuid.uuid5(uuid.NAMESPACE_URL, self.name).hex[:18]}", "displayName": self.name,
                "displayOption": 1, "width": W, "height": H, "ordinal": i, "config": json.dumps(cfg), "filters": "[]",
                "visualContainers": self.visuals}


def style(size: float, colour_: str, weight: str = "normal") -> dict:
    family = {"normal": FONT, "semibold": FONT_SB, "bold": FONT_B}[weight]
    s = {"fontSize": f"{size}pt", "color": colour_, "fontFamily": family}
    if weight == "bold":
        s["fontWeight"] = "bold"
    return s


def header(p: Page, kicker: str, title: str, sub: str):
    p.text(0, 0, W, BAND_H, [[(kicker.upper(), style(8, "#7FA7D9", "semibold"))],
                             [(title, style(17, "#FFFFFF", "semibold"))],
                             [(sub, style(9, "#B7C3D6"))]], bg=NAVY, radius=0)


def callout(p: Page, x, y, w, h, lead: str, body: str, tone: str = "info"):
    bg, br, lc = {"good": ("#EAF6EC", "#BFE3C5", GOOD), "info": (CARD_BG, BORDER, NAVY_2)}[tone]
    p.text(x, y, w, h, [[(lead + "   ", style(9.5, lc, "semibold")), (body, style(9.5, INK))]], bg=bg, border=br)


def grid(n: int, width: int = W - 2 * M, x0: int = M) -> list[tuple[int, int]]:
    """n equal columns -> [(x, w)]."""
    w = (width - (n - 1) * G) / n
    return [(round(x0 + i * (w + G)), round(w)) for i in range(n)]


def rows(n: int, y0: int, y1: int = BOTTOM) -> list[tuple[int, int]]:
    h = (y1 - y0 - (n - 1) * G) / n
    return [(round(y0 + i * (h + G)), round(h)) for i in range(n)]


def kpis(p: Page, y: int, h: int, items: list[tuple[str, str, str]], width: int = W - 2 * M, x0: int = M):
    """items: (measure, title, value colour)."""
    for (x, w), (meas, title, c) in zip(grid(len(items), width, x0), items):
        p.add("card", x, y, w, h, {"Values": [m("_Measures", meas)]}, title, card_objects(c))


KPI_H = 92


# ---------------------------------------------------------------- pages
def build_report() -> dict:
    L, R = grid(2)
    pages = []

    # 1 ── Executive summary
    p = Page("Executive Summary")
    header(p, "Feed ranking experiment · launch review", "Ship the AI feed + friend boost, not the pure AI feed",
           "3-arm randomised test · 60,000 users · 14-day pre-period + 28-day experiment")
    callout(p, M, TOP, W - 2 * M, 44, "✔ RECOMMENDATION",
            "The AI feed's lift is mostly novelty and costs +17% churn. Friend boost keeps most of the engagement, "
            "cuts churn 11%, and is worth ~$103M vs ~$57M over 12 months.", "good")
    y = TOP + 44 + G
    kpis(p, y, KPI_H, [("AI Feed Minutes Lift", "AI feed · minutes", NAVY),
                       ("Friend Boost Minutes Lift", "Friend boost · minutes", NAVY),
                       ("AI Feed Churn Change", "AI feed · churn", BAD),
                       ("Friend Boost Churn Change", "Friend boost · churn", GOOD),
                       ("Friend Boost 12M Value ($M)", "Friend boost · 12-mo value", GOOD)])
    y += KPI_H + G
    lw = 780
    rx, rw = M + lw + G, W - 2 * M - lw - G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("lineChart", M, y1, lw, h1, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", "Minutes per User-Day")],
                                       "Series": [ARM]}, "Daily minutes per user", line_objects(ARM_COLOURS))
    p.add("lineChart", M, y2, lw, h2, {"Category": [col("daily_lift", "day")], "Y": [agg("daily_lift", "rel_lift")],
                                       "Series": [ARM]}, "Lift vs control by experiment day", line_objects(ARM_COLOURS))
    p.add("tableEx", rx, y, rw, BOTTOM - y, {"Values": [col("metric_results", "Metric name"), col("metric_results", "Arm name"),
                                                        col("metric_results", "Lift"), col("metric_results", "Result")]},
          "Metric scorecard", table_objects())
    pages.append(p)

    # 2 ── Experiment scorecard
    p = Page("Experiment Scorecard")
    header(p, "Experiment validity & results", "Every metric, every arm, with uncertainty",
           "SRM p = 0.10 · max |SMD| 0.019 · A/A false-positive rate 5.1% · CUPED removes ~68% of variance")
    (sx, sw), (c1x, c1w), (c2x, c2w) = grid(3, L[1], L[0])
    p.add("slicer", sx, TOP, sw, KPI_H, {"Values": [ARM]}, "Arm", slicer_objects())
    p.add("card", c1x, TOP, c1w, KPI_H, {"Values": [m("_Measures", "Significant Wins")]}, "Significant wins", card_objects(GOOD))
    p.add("card", c2x, TOP, c2w, KPI_H, {"Values": [m("_Measures", "Significant Harms")]}, "Guardrail harms", card_objects(BAD))
    y = TOP + KPI_H + G
    p.add("clusteredBarChart", L[0], y, L[1], BOTTOM - y,
          {"Category": [col("metric_results", "Metric name")], "Y": [agg("metric_results", "rel_lift")], "Series": [ARM]},
          "Relative lift vs control", bar_objects(ARM_COLOURS, labels=True))
    (y1, h1), (y2, h2) = rows(2, TOP)
    p.add("lineChart", R[0], y1, R[1], h1, {"Category": [col("power_mde", "n_per_arm")],
                                            "Y": [agg("power_mde", "Minimum detectable lift")],
                                            "Series": [col("power_mde", "Method name")]},
          "Minimum detectable lift by users per arm",
          line_objects(series_colours("power_mde", "Method name", {"Raw t-test": SERIES_2, "With CUPED": SERIES})))
    p.add("tableEx", R[0], y2, R[1], h2, {"Values": [col("metric_results", "Metric name"), col("metric_results", "Arm name"),
                                                     col("metric_results", "CI low"), col("metric_results", "CI high"),
                                                     col("metric_results", "Variance removed")]},
          "95% confidence intervals", table_objects())
    pages.append(p)

    # 3 ── Engagement trends
    p = Page("Engagement Trends")
    header(p, "Engagement explorer", "Slice every daily metric by segment",
           "All measures are per assigned user-day (intent-to-treat), so they stay correct under any filter")
    for (x, w), (s, t) in zip(grid(4), [("age_group", "Age group"), ("region", "Region"), ("device", "Device"),
                                        ("account_type", "Account type")]):
        p.add("slicer", x, TOP, w, 76, {"Values": [col("fact_daily", s)]}, t, slicer_objects())
    y = TOP + 76 + G
    (y1, h1), (y2, h2) = rows(2, y)
    for (x, w), (yy, hh, meas, t) in zip([L, R, L, R], [
            (y1, h1, "Minutes Lift vs Control", "Minutes lift vs control"),
            (y1, h1, "DAU Rate", "DAU rate"),
            (y2, h2, "Friend Interactions per User-Day", "Friend interactions per user-day"),
            (y2, h2, "Revenue per User-Day", "Ad revenue per user-day")]):
        p.add("lineChart", x, yy, w, hh, {"Category": [col("dim_date", "date")], "Y": [m("_Measures", meas)], "Series": [ARM]},
              t, line_objects(ARM_COLOURS))
    pages.append(p)

    # 4 ── Segments
    p = Page("Segment Deep-Dive")
    header(p, "Heterogeneity", "Who wins, who churns",
           "CUPED minutes lift and churn change by segment · Benjamini-Hochberg across all segment tests")
    callout(p, M, TOP, W - 2 * M, 44, "READ THIS",
            "AI-feed churn harm concentrates in users aged 35+ (age × treatment p = 0.0025). "
            "The friend boost lowers churn in every age group.")
    y = TOP + 44 + G
    (y1, h1), (y2, h2) = rows(2, y)
    age = [visual_filter("segment_results", "dimension", ["age_group"])]
    region = [visual_filter("segment_results", "dimension", ["region"])]
    p.add("clusteredColumnChart", L[0], y1, L[1], h1, {"Category": [col("segment_results", "Segment value")],
                                                      "Y": [agg("segment_results", "churn_diff_pp")], "Series": [ARM]},
          "Churn change by age (pp)", bar_objects(ARM_COLOURS, labels=True), filters=age)
    p.add("clusteredColumnChart", R[0], y1, R[1], h1, {"Category": [col("segment_results", "Segment value")],
                                                      "Y": [agg("segment_results", "churn_diff_pp")], "Series": [ARM]},
          "Churn change by region (pp)", bar_objects(ARM_COLOURS, labels=True), filters=region)
    p.add("clusteredColumnChart", L[0], y2, L[1], h2, {"Category": [col("segment_results", "Segment value")],
                                                      "Y": [agg("segment_results", "minutes_lift")], "Series": [ARM]},
          "Minutes lift by age", bar_objects(ARM_COLOURS, labels=True), filters=age)
    p.add("tableEx", R[0], y2, R[1], h2, {"Values": [col("segment_results", "Dimension name"),
                                                     col("segment_results", "Segment value"), col("segment_results", "Arm name"),
                                                     col("segment_results", "Churn change (pp)"), col("segment_results", "Verdict")]},
          "All segment verdicts", table_objects())
    pages.append(p)

    # 5 ── Retention
    p = Page("Retention & Churn Risk")
    header(p, "Retention", "Survival by arm and a churn early-warning system",
           "Kaplan-Meier · Cox PH (AI feed HR 1.19, friend boost HR 0.88) · leakage-free LightGBM scored at day 13")
    kpis(p, TOP, KPI_H, [("Churn Model AUC", "Churn model ROC-AUC", NAVY), ("Churn Rate", "28-day churn rate", NAVY),
                         ("Avg Churn Risk", "Average predicted risk", NAVY), ("High-Risk Users", "High-risk users", BAD)])
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("lineChart", L[0], y1, L[1], h1, {"Category": [col("km_survival", "day")], "Y": [agg("km_survival", "survival", "avg")],
                                            "Series": [ARM]}, "Users not yet churned (Kaplan-Meier)", line_objects(ARM_COLOURS))
    p.add("clusteredBarChart", R[0], y1, R[1], h1 + G + h2,
          {"Category": [col("churn_shap_importance", "Feature name")], "Y": [agg("churn_shap_importance", "mean_abs_shap")]},
          "What drives churn risk (mean |SHAP|)", bar_objects(legend=False))
    p.add("areaChart", L[0], y2, L[1], h2, {"Category": [col("retention_profit_curve", "fraction_targeted")],
                                            "Y": [agg("retention_profit_curve", "profit_at_scale_musd")]},
          "Retention campaign profit ($M) by share targeted", line_objects(legend=False))
    pages.append(p)

    # 6 ── Personalisation
    p = Page("Personalisation (HTE)")
    header(p, "Causal ML", "Heterogeneous effects & rollout policy",
           "Cross-fitted DR-learner on net annual user value · IPW off-policy evaluation on 35% held-out users")
    (k1x, k1w), (sx, sw) = grid(2, L[1], L[0])
    p.add("card", k1x, TOP, k1w, KPI_H, {"Values": [m("_Measures", "Best Policy Uplift ($/user/yr)")]},
          "Best policy · $/user/yr", card_objects(GOOD))
    p.add("slicer", sx, TOP, sw, KPI_H, {"Values": [col("qini_curves", "arm")]}, "Qini: treatment arm", slicer_objects())
    callout(p, R[0], TOP, R[1], KPI_H, "HOW TO READ",
            "The AI feed's effect varies a lot by user (targetable). The friend boost's is nearly uniform, so a "
            "blanket launch is low-regret. Model targeting leads, but its CI overlaps blanket friend boost.")
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("clusteredBarChart", L[0], y1, L[1], h1, {"Category": [col("policy_values", "Policy name")],
                                                    "Y": [agg("policy_values", "uplift_vs_control_usd")]},
          "Net value vs control ($/user/yr) by policy", bar_objects(labels=True, legend=False))
    p.add("lineChart", R[0], y1, R[1], h1, {"Category": [col("qini_curves", "fraction_targeted")],
                                            "Y": [agg("qini_curves", "cumulative_gain")], "Series": [col("qini_curves", "model")]},
          "Qini curve: gain from targeting",
          line_objects(series_colours("qini_curves", "model", {"DR-learner": SERIES, "T-learner": SERIES_2})))
    p.add("clusteredColumnChart", M, y2, W - 2 * M, h2,
          {"Category": [col("cate_deciles", "decile")],
           "Y": [agg("cate_deciles", "predicted_cate"), agg("cate_deciles", "observed_uplift")]},
          "CATE calibration by decile: predicted vs observed ($/user/yr)",
          bar_objects(measure_colours({"Sum(cate_deciles.predicted_cate)": SERIES_2,
                                       "Sum(cate_deciles.observed_uplift)": SERIES})))
    pages.append(p)

    # 7 ── Content health
    p = Page("Content & Responsible AI")
    header(p, "Responsible AI", "Content health guardrails",
           "Real TweetEval posts · NMF topics · sentiment model (macro-recall 0.628 vs VADER 0.570) · simulated ranker exposure")
    (y1, h1), (y2, h2) = rows(2, TOP)
    p.add("clusteredColumnChart", M, y1, W - 2 * M, h1, {"Category": [col("topic_exposure_by_arm", "Topic")],
                                                         "Y": [agg("topic_exposure_by_arm", "Impression share")],
                                                         "Series": [ARM]},
          "Share of impressions by topic", bar_objects(ARM_COLOURS))
    (ax, aw), (bx, bw), (cx, cw) = grid(3)
    p.add("clusteredColumnChart", ax, y2, aw, h2, {"Category": [ARM], "Y": [agg("content_exposure_summary", "Negative share")]},
          "Negative posts served", bar_objects(ARM_COLOURS, labels=True, legend=False))
    p.add("clusteredColumnChart", bx, y2, bw, h2, {"Category": [ARM], "Y": [agg("content_exposure_summary", "Topic diversity")]},
          "Feed diversity (topic entropy)", bar_objects(ARM_COLOURS, labels=True, legend=False))
    p.add("tableEx", cx, y2, cw, h2, {"Values": [col("sentiment_model_metrics", "Model name"),
                                                 col("sentiment_model_metrics", "Macro recall"),
                                                 col("sentiment_model_metrics", "Macro F1"),
                                                 col("sentiment_model_metrics", "Accuracy %")]},
          "Sentiment model (TweetEval test)", table_objects())
    pages.append(p)

    # 8 ── Business impact
    p = Page("Business Impact")
    header(p, "Decision economics", "12-month business impact under uncertainty",
           "20,000 Monte Carlo draws over post-novelty revenue lift and Cox churn hazard ratios · 40M eligible MAU")
    kpis(p, TOP, KPI_H, [("Friend Boost 12M Value ($M)", "Friend boost · median 12-mo", GOOD),
                         ("AI Feed 12M Value ($M)", "AI feed · median 12-mo", NAVY),
                         ("P(Friend Boost Beats AI Feed)", "P(friend boost wins)", GOOD)])
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("lineChart", L[0], y1, L[1], h1, {"Category": [col("impact_fan", "month")], "Y": [agg("impact_fan", "p50")],
                                            "Series": [ARM]}, "Cumulative incremental revenue ($M, median)",
          line_objects(ARM_COLOURS))
    p.add("clusteredColumnChart", R[0], y1, R[1], h1, {"Category": [col("impact_distribution", "bin_mid_musd")],
                                                       "Y": [agg("impact_distribution", "share_of_draws")], "Series": [ARM]},
          "Distribution of 12-month value ($M)", bar_objects(ARM_COLOURS))
    p.add("tableEx", L[0], y2, L[1], h2, {"Values": [col("impact_monte_carlo", "Scenario"), col("impact_monte_carlo", "P5 ($M)"),
                                                     col("impact_monte_carlo", "Median ($M)"), col("impact_monte_carlo", "P95 ($M)"),
                                                     col("impact_monte_carlo", "P(> 0)")]},
          "Monte Carlo summary", table_objects())
    p.add("clusteredBarChart", R[0], y2, R[1], h2,
          {"Category": [col("impact_sensitivity", "Driver")],
           "Y": [agg("impact_sensitivity", "Low case ($M)"), agg("impact_sensitivity", "High case ($M)")]},
          "Sensitivity: friend boost minus AI feed ($M)",
          bar_objects(measure_colours({"Sum(impact_sensitivity.Low case ($M))": SERIES_2,
                                       "Sum(impact_sensitivity.High case ($M))": SERIES})))
    pages.append(p)

    # 9 ── Creator virality
    p = Page("Creator Virality (Real Data)")
    header(p, "Creator insights · real data", "What makes content spread",
           "UCI Online News Popularity · 39,644 Mashable articles · time-split LightGBM (AUC 0.732) · SHAP drivers")
    lw = 500
    p.add("clusteredBarChart", M, TOP, lw, BOTTOM - TOP,
          {"Category": [col("virality_shap_importance", "Feature name")], "Y": [agg("virality_shap_importance", "mean_abs_shap")]},
          "Virality drivers (mean |SHAP|)", bar_objects(legend=False))
    rx, rw = M + lw + G, W - 2 * M - lw - G
    p.add("slicer", rx, TOP, rw, 76, {"Values": [col("virality_playbook", "Playbook view")]}, "Playbook view", slicer_objects())
    y = TOP + 76 + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("clusteredColumnChart", rx, y1, rw, h1, {"Category": [col("virality_playbook", "Level name")],
                                                   "Y": [agg("virality_playbook", "Viral rate", "avg")]},
          "Viral rate (top-quartile shares)", bar_objects(labels=True, legend=False))
    p.add("areaChart", rx, y2, rw, h2, {"Category": [col("virality_weekly", "week")],
                                        "Y": [agg("virality_weekly", "viral_rate", "avg")]},
          "Weekly viral rate", line_objects(legend=False))
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
