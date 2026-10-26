"""Power BI report layout (legacy report.json) with the styling built in."""
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
FOOTER_H = 18
BOTTOM = H - M - FOOTER_H


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
    """Colour each series by its value so colours don't depend on sort order."""
    return [{"properties": {"fill": colour(c)},
             "selector": {"data": [{"scopeId": {"Comparison": {
                 "ComparisonKind": 0,
                 "Left": {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}},
                 "Right": {"Literal": {"Value": "'" + v + "'"}}}}}]}}
            for v, c in mapping.items()]


def measure_colours(mapping: dict[str, str]) -> list[dict]:
    """Colour each measure in a multi-measure chart. Keys are queryRefs."""
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
    p.text(0, 0, W, BAND_H, [[("SYNTHETIC DATA · " + kicker.upper(), style(8, "#7FA7D9", "semibold"))],
                             [(title, style(17, "#FFFFFF", "semibold"))],
                             [(sub, style(9, "#B7C3D6"))]], bg=NAVY, radius=0)


def callout(p: Page, x, y, w, h, lead: str, body: str, tone: str = "info"):
    bg, br, lc = {"good": ("#EAF6EC", "#BFE3C5", GOOD), "warn": ("#FFF6E5", "#F5D9A6", "#9A6700"),
                  "bad": ("#FDECEC", "#F3C2C2", BAD), "info": (CARD_BG, BORDER, NAVY_2)}[tone]
    p.text(x, y, w, h, [[(lead + "   ", style(9.5, lc, "semibold")), (body, style(9.5, INK))]], bg=bg, border=br)


def grid(n: int, width: int = W - 2 * M, x0: int = M) -> list[tuple[int, int]]:
    """n equal columns -> [(x, w)]."""
    w = (width - (n - 1) * G) / n
    return [(round(x0 + i * (w + G)), round(w)) for i in range(n)]


def rows(n: int, y0: int, y1: int = BOTTOM) -> list[tuple[int, int]]:
    h = (y1 - y0 - (n - 1) * G) / n
    return [(round(y0 + i * (h + G)), round(h)) for i in range(n)]


def accent(p: Page, x, y, h, colour_: str):
    """Thin coloured strip on a card's left edge."""
    p.text(x, y + 14, 4, h - 28, [[(" ", style(1, colour_))]], bg=colour_, radius=2)


def kpis(p: Page, y: int, h: int, items: list[tuple[str, str, str]], width: int = W - 2 * M, x0: int = M):
    """One card per (measure, title, colour), each with an accent strip."""
    for (x, w), (meas, title, c) in zip(grid(len(items), width, x0), items):
        p.add("card", x, y, w, h, {"Values": [m("_Measures", meas)]}, title, card_objects(c))
        accent(p, x, y, h, c)


def insight(p: Page, x, y, w, h, kicker: str, stat: str, body: str, colour_: str):
    """Story card: small kicker, big number, one-sentence takeaway."""
    p.text(x, y, w, h, [[(kicker.upper(), style(8, colour_, "semibold"))],
                        [(stat, style(22, NAVY, "semibold"))],
                        [(body, style(9.5, INK_2))]], bg=CARD_BG, border=BORDER)
    accent(p, x, y, h, colour_)


STATE_TONE = {"LAUNCH": (GOOD, "#EAF6EC", "#BFE3C5", "✔"), "CONDITIONAL": ("#9A6700", "#FFF6E5", "#F5D9A6", "◐"),
              "DO NOT LAUNCH": (BAD, "#FDECEC", "#F3C2C2", "✖")}


def chip(p: Page, x, y, w, h, title: str, state: str, detail: str):
    fg, bg, br, icon = STATE_TONE[state]
    p.text(x, y, w, h, [[(title.upper(), style(8, INK_2, "semibold"))],
                        [(f"{icon}  {state}", style(16, fg, "semibold"))],
                        [(detail, style(9, INK))]], bg=bg, border=br)


def footer(p: Page, F: dict, i: int, n: int):
    p.text(0, H - FOOTER_H - 4, W, FOOTER_H + 4,
           [[(f"SYNTHETIC DATA  ·  {DISPLAY_NAME}  ·  run {F['commit']}  ·  generated {F['generated']}  ·  page {i} of {n}",
              style(7, MUTED))]])


KPI_H = 92
DISPLAY_NAME = "Feed Ranking Launch Review"


def build_report(F: dict) -> dict:
    """Build report.json from narrative.facts()."""
    L, R = grid(2)
    pages = []
    ex, sv, cm, imp, vir = F["experiment"], F["survival"], F["churn_model"], F["impact"], F["virality"]
    d = ex["design"]
    tone = {"LAUNCH": "good", "CONDITIONAL": "warn", "DO NOT LAUNCH": "bad"}[F["state"]]

    p = Page("Overview")
    header(p, "Feed ranking experiment · launch review", DISPLAY_NAME,
           "Simulated 3-arm randomised test · 60,000 users · 14-day pre-period + 28-day experiment · every statement below is "
           "generated from the results")
    callout(p, M, TOP, W - 2 * M, 52, f"DECISION · {F['state']}", F["headline"], tone)
    st = F["story"]
    y0 = TOP + 52 + G
    card_h = 132
    for idx, ((x, w), (k, stat, body, c)) in enumerate(zip(grid(3) + grid(3), st)):
        insight(p, x, y0 + (idx // 3) * (card_h + G), w, card_h, k, stat, body, c)
    y = y0 + 2 * card_h + 2 * G
    p.add("lineChart", L[0], y, L[1], BOTTOM - y, {"Category": [col("dim_date", "date")],
                                                   "Y": [m("_Measures", "Minutes per User-Day")], "Series": [ARM]},
          "Daily minutes per user (pre-period → experiment)", line_objects(ARM_COLOURS))
    p.add("clusteredBarChart", R[0], y, R[1], BOTTOM - y,
          {"Category": [col("metric_results", "Metric name")], "Y": [agg("metric_results", "rel_lift")], "Series": [ARM]},
          "Primary metric and guardrails: lift vs control", bar_objects(ARM_COLOURS, labels=True),
          filters=[visual_filter("metric_results", "tier", ["primary", "guardrail"])])
    pages.append(p)

    p = Page("Decision & Guardrails")
    header(p, "Launch decision rule", "Which arm can ship, and what blocks the other",
           f"Guardrails: one-sided non-inferiority, Bonferroni across {F['decision']['n_guardrail_tests']} comparisons · "
           "PASS rules out harm beyond the margin · margins written after first results (disclosed)")
    a_ai, a_fb = F["decision"]["arms"]["ai_feed"], F["decision"]["arms"]["ai_feed_social"]
    lab = F["label"]

    def detail(a):
        bits = []
        if a["blocking"]:
            bits.append("blocked by " + ", ".join(lab[x] for x in a["blocking"]))
        if a["mitigations_required"]:
            bits.append("needs mitigation for " + ", ".join(lab[x] for x in a["mitigations_required"]))
        return f"Primary {a['primary_lift']:+.1%} minutes · " + ("; ".join(bits) if bits else "all guardrails pass")
    (cx1, cw1), (cx2, cw2) = grid(2)
    chip(p, cx1, TOP, cw1, 88, C.ARM_LABELS["ai_feed"], a_ai["state"], detail(a_ai))
    chip(p, cx2, TOP, cw2, 88, C.ARM_LABELS["ai_feed_social"], a_fb["state"], detail(a_fb))
    y = TOP + 88 + G
    p.add("tableEx", L[0], y, L[1], BOTTOM - y,
          {"Values": [col("decision_guardrails", "Arm name"), col("decision_guardrails", "Guardrail"),
                      col("decision_guardrails", "Observed change"), col("decision_guardrails", "Allowed harm"),
                      col("decision_guardrails", "Status")]}, "Guardrail matrix", table_objects())
    p.add("clusteredBarChart", R[0], y, R[1], BOTTOM - y,
          {"Category": [col("decision_guardrails", "Guardrail")], "Y": [agg("decision_guardrails", "Harm bound")],
           "Series": [ARM]}, "Worst-case harm (adjusted bound) vs margin", bar_objects(ARM_COLOURS, labels=True))
    pages.append(p)

    p = Page("Results Snapshot")
    header(p, "Experiment results", "Engagement up in both arms; retention splits them",
           "CUPED-adjusted lifts vs control · Benjamini-Hochberg across 22 tests · daily lift shows the novelty decay")
    y = TOP
    kpis(p, y, KPI_H, [("AI Feed Minutes Lift", "AI feed · minutes", C.ARM_COLORS["ai_feed"]),
                       ("Friend Boost Minutes Lift", "Friend boost · minutes", C.ARM_COLORS["ai_feed_social"]),
                       ("AI Feed Churn Change", "AI feed · churn", BAD),
                       ("Friend Boost Churn Change", "Friend boost · churn", GOOD),
                       ("Significant Harms", "Guardrail harms (BH)", BAD)])
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

    p = Page("Experiment Scorecard")
    header(p, "Experiment validity & results", "Every metric, every arm, with uncertainty",
           f"SRM p = {d['srm']['p_value']:.2f} · max |SMD| {d['max_abs_smd']:.3f} · A/A false-positive rate "
           f"{ex['aa']['aa_false_positive_rate']:.1%} · CUPED MDE {d['mde_rel_raw']:.1%} → {d['mde_rel_cuped']:.1%}")
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

    p = Page("Segment Deep-Dive")
    header(p, "Heterogeneity", "Who wins, who churns",
           "CUPED minutes lift and churn change by segment · Benjamini-Hochberg across all segment tests")
    callout(p, M, TOP, W - 2 * M, 44, "READ THIS", F["segments"])
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

    p = Page("Retention & Lapse Risk")
    header(p, "Retention", "Time to first 7-day lapse, and a lapse early-warning model",
           f"Cox HR AI feed {sv['hr_ai_feed']:.2f}, friend boost {sv['hr_ai_feed_social']:.2f} · early warning scores users "
           f"active on day 13 · deployed: {cm['deployed_model']}")
    kpis(p, TOP, KPI_H, [("Churn Model AUC", "Lapse model ROC-AUC", NAVY), ("Churn Rate", "Final-week inactivity rate", NAVY),
                         ("Avg Churn Risk", "Average predicted lapse risk", NAVY), ("High-Risk Users", "High-risk users", BAD)])
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("lineChart", L[0], y1, L[1], h1, {"Category": [col("km_survival", "day")], "Y": [agg("km_survival", "survival", "avg")],
                                            "Series": [ARM]}, "Users without a 7-day lapse yet (Kaplan-Meier)", line_objects(ARM_COLOURS))
    p.add("clusteredBarChart", R[0], y1, R[1], h1 + G + h2,
          {"Category": [col("churn_shap_importance", "Feature name")], "Y": [agg("churn_shap_importance", "mean_abs_shap")]},
          "What drives lapse risk (LightGBM, mean |SHAP|)", bar_objects(legend=False))
    p.add("areaChart", L[0], y2, L[1], h2, {"Category": [col("retention_profit_curve", "fraction_targeted")],
                                            "Y": [agg("retention_profit_curve", "profit_at_scale_musd")]},
          "Nudge campaign value ($M, assumed save rate) by share targeted", line_objects(legend=False))
    pages.append(p)

    p = Page("Personalisation")
    header(p, "Causal ML", "Heterogeneous effects & rollout policy",
           "Cross-fitted DR-learner on an assumed utility score · Horvitz-Thompson IPW on 35% held-out users · "
           "validated against simulator counterfactuals")
    p.add("card", L[0], TOP, 280, KPI_H, {"Values": [m("_Measures", "Best Policy Uplift ($/user/yr)")]},
          "Best policy · utility/user", card_objects(NAVY))
    callout(p, L[0] + 280 + G, TOP, W - 2 * M - 280 - G, KPI_H, "HOW TO READ", F["personalisation"])
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("clusteredBarChart", L[0], y1, L[1], h1, {"Category": [col("policy_values", "Policy name")],
                                                    "Y": [agg("policy_values", "uplift_vs_control_usd")]},
          "Utility vs control (per user) by policy", bar_objects(labels=True, legend=False))
    p.add("lineChart", R[0], y1, R[1], h1, {"Category": [col("qini_curves", "fraction_targeted")],
                                            "Y": [agg("qini_curves", "cumulative_gain")], "Series": [ARM]},
          "Cumulative uplift gain (DR-learner) by share targeted", line_objects(ARM_COLOURS),
          filters=[visual_filter("qini_curves", "model", ["DR-learner"])])
    p.add("clusteredColumnChart", M, y2, W - 2 * M, h2,
          {"Category": [col("cate_deciles", "decile")], "Y": [agg("cate_deciles", "observed_uplift")], "Series": [ARM]},
          "Observed held-out uplift by predicted-effect decile (per arm)", bar_objects(ARM_COLOURS))
    pages.append(p)

    p = Page("Content Health (Scenario)")
    sm = F["nlp"]["sentiment"]["metrics"]
    header(p, "Responsible AI · scenario", "Content health under ASSUMED ranker weights",
           f"Held-out TweetEval posts · sentiment macro-recall {sm[1]['macro_recall']:.3f} vs VADER {sm[0]['macro_recall']:.3f} · "
           "harms are built into the scenario's weights, not learned from engagement")
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

    p = Page("Revenue Scenarios")
    header(p, "Decision economics · scenario model", "12-month revenue under explicit assumptions",
           f"{imp['n_boot']:,} joint bootstrap draws (shared control) · {C.MONTHLY_BASE_CHURN:.1%} assumed monthly lapse rate · "
           f"{C.MAU_POPULATION / 1e6:.0f}M users · scenario: effects persist")
    kpis(p, TOP, KPI_H, [("Friend Boost 12M Value ($M)", "Friend boost · median 12-mo", NAVY),
                         ("AI Feed 12M Value ($M)", "AI feed · median 12-mo", NAVY),
                         ("Share of Draws FB Beats AI", "Share of draws: friend boost > AI", NAVY)])
    y = TOP + KPI_H + G
    (y1, h1), (y2, h2) = rows(2, y)
    p.add("lineChart", L[0], y1, L[1], h1, {"Category": [col("impact_fan", "month")], "Y": [agg("impact_fan", "p50")],
                                            "Series": [ARM]}, "Cumulative incremental revenue ($M, median)",
          line_objects(ARM_COLOURS))
    p.add("clusteredColumnChart", R[0], y1, R[1], h1, {"Category": [col("impact_distribution", "bin_mid_musd")],
                                                       "Y": [agg("impact_distribution", "share_of_draws")], "Series": [ARM]},
          "Distribution of 12-month value ($M)", bar_objects(ARM_COLOURS))
    p.add("tableEx", L[0], y2, L[1], h2, {"Values": [col("impact_scenarios", "Effect persistence"), col("impact_scenarios", "Scenario"),
                                                     col("impact_scenarios", "Median ($M)"), col("impact_scenarios", "90% range"),
                                                     col("impact_scenarios", "Share of draws > 0")]},
          "All persistence scenarios", table_objects())
    p.add("clusteredBarChart", R[0], y2, R[1], h2,
          {"Category": [col("impact_sensitivity", "Driver")],
           "Y": [agg("impact_sensitivity", "Low case ($M)"), agg("impact_sensitivity", "High case ($M)")]},
          "Sensitivity: friend boost minus AI feed ($M)",
          bar_objects(measure_colours({"Sum(impact_sensitivity.Low case ($M))": SERIES_2,
                                       "Sum(impact_sensitivity.High case ($M))": SERIES})))
    pages.append(p)

    p = Page("Creator Virality (Real)")
    header(p, "Creator insights · real data", "What makes content spread",
           f"UCI news · ordered retrospective benchmark · AUC {vir['auc_lightgbm']:.3f} "
           f"({vir['auc_lightgbm_without_unverified_features']:.3f} without share-history features) · playbook is descriptive")
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

    for i, pg in enumerate(pages, 1):
        pg.name = f"{i} · {pg.name}"
        footer(pg, F, i, len(pages))

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
