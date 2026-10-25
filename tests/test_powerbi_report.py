"""Static validation of the generated Power BI report (no Power BI engine needed).

Every visual field, visual filter value, colour selector and DAX column reference must exist in the exported model and
data; visuals must stay on the 1280x720 canvas and must not overlap (except 4px accent strips drawn on cards).
"""
import json
import re

import pandas as pd
import pytest

from sei import config as C
from sei.powerbi import NAME

REPORT = C.POWERBI / f"{NAME}.Report" / "report.json"
MODEL = C.POWERBI / f"{NAME}.SemanticModel" / "model.bim"
pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def model():
    bim = json.loads(MODEL.read_text())
    return {t["name"]: ({c["name"] for c in t["columns"]}, {m["name"] for m in t.get("measures", [])}) for t in bim["model"]["tables"]}, bim


@pytest.fixture(scope="module")
def visuals():
    rep = json.loads(REPORT.read_text())
    return [(s["displayName"], v, json.loads(v["config"])["singleVisual"]) for s in rep["sections"] for v in s["visualContainers"]]


def _values(table, column):
    return set(pd.read_csv(C.POWERBI / "data" / f"{table}.csv")[column].astype(str))


def test_fields_exist(model, visuals):
    tabs, _ = model
    for page, _, sv in visuals:
        if "prototypeQuery" not in sv:
            continue
        alias = {f["Name"]: f["Entity"] for f in sv["prototypeQuery"]["From"]}
        for q in sv["prototypeQuery"]["Select"]:
            node = q.get("Measure") or q.get("Column") or q["Aggregation"]["Expression"]["Column"]
            ent, prop = alias[node["Expression"]["SourceRef"]["Source"]], node["Property"]
            pool = tabs[ent][1] if "Measure" in q else tabs[ent][0]
            assert prop in pool, f"{page}: {ent}[{prop}]"


def test_filters_and_colour_selectors_match_data(visuals):
    for page, v, sv in visuals:
        for f in json.loads(v["filters"]):
            col = f["expression"]["Column"]
            ent, prop = col["Expression"]["SourceRef"]["Entity"], col["Property"]
            for val in f["filter"]["Where"][0]["Condition"]["In"]["Values"]:
                assert val[0]["Literal"]["Value"].strip("'") in _values(ent, prop), f"{page}: filter {ent}[{prop}]"
        refs = {q["Name"] for q in sv.get("prototypeQuery", {}).get("Select", [])}
        for dp in sv.get("objects", {}).get("dataPoint", []):
            sel = dp.get("selector") or {}
            if "metadata" in sel:
                assert sel["metadata"] in refs, f"{page}: {sel['metadata']}"
            if "data" in sel:
                c = sel["data"][0]["scopeId"]["Comparison"]
                ent, prop = c["Left"]["Column"]["Expression"]["SourceRef"]["Entity"], c["Left"]["Column"]["Property"]
                assert c["Right"]["Literal"]["Value"].strip("'") in _values(ent, prop), f"{page}: colour {ent}[{prop}]"


def test_dax_column_references_exist(model):
    tabs, bim = model
    for t in bim["model"]["tables"]:
        for m in t.get("measures", []):
            for table, col in re.findall(r"(\w+)\[([^\]]+)\]", m["expression"]):
                assert table in tabs and col in tabs[table][0], f"{m['name']}: {table}[{col}]"


def test_layout_on_canvas_without_overlaps(visuals):
    by_page = {}
    for page, v, sv in visuals:
        x, y, w, h = v["x"], v["y"], v["width"], v["height"]
        assert x >= 0 and y >= 0 and x + w <= 1280.5 and y + h <= 720.5, f"{page}: {sv['visualType']} off canvas"
        if sv["visualType"] == "textbox" and w <= 6:
            continue                                   # accent strip, drawn on top of a card on purpose
        for (bx, by, bw, bh, bt) in by_page.get(page, []):
            overlap = x < bx + bw - 0.5 and bx < x + w - 0.5 and y < by + bh - 0.5 and by < y + h - 0.5
            assert not overlap, f"{page}: {sv['visualType']} overlaps {bt}"
        by_page.setdefault(page, []).append((x, y, w, h, sv["visualType"]))
