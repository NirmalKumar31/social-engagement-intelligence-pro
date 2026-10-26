"""Run the whole pipeline and write a run manifest: python -m sei.pipeline [--skip-nlp] [--llm]"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata

from . import config as C
from . import experiment, features, hte, impact, nlp, oracle, powerbi, report, retention, simulate, virality

SOURCE_PATHS = ["sei", "dashboard", "tests", "queries", "Makefile", "requirements.txt", "requirements-lock.txt"]
PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "lightgbm", "shap", "lifelines", "pyarrow"]


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def config_fingerprint() -> dict:
    return {"seed": C.SEED, "n_users": C.N_USERS, "pre_days": C.PRE_DAYS, "exp_days": C.EXP_DAYS,
            "guardrails": C.GUARDRAILS, "monthly_base_churn": C.MONTHLY_BASE_CHURN, "mau": C.MAU_POPULATION}


def write_manifest(stages: list[dict], started: str):
    inputs = {str(p.relative_to(C.ROOT)): _sha(p) for p in sorted(C.EXTERNAL.rglob("*")) if p.is_file() and p.suffix != ".zip"}
    outputs = {str(p.relative_to(C.WORK)): _sha(p) for d in (C.REPORTS, C.POWERBI / "data")
               for p in sorted(d.rglob("*")) if p.is_file() and p.suffix in {".json", ".csv", ".md", ".txt"}
               and p.name != "run_manifest.json"}
    manifest = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": _git("rev-parse", "HEAD"),
                # outputs change during a run, so only source files count as dirty
                "git_source_dirty": bool(_git("status", "--porcelain", "--", *SOURCE_PATHS)),
                "python": platform.python_version(), "packages": {p: metadata.version(p) for p in PACKAGES},
                "config": config_fingerprint(), "workdir": str(C.WORK), "stages": stages,
                "inputs_sha256_16": inputs, "outputs_sha256_16": outputs}
    with open(C.REPORTS / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nlp", action="store_true", help="reuse NLP outputs from a previous run with the same config")
    ap.add_argument("--llm", action="store_true", help="also generate a Claude-polished executive brief")
    args = ap.parse_args()
    for cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=cat)
    if args.skip_nlp:
        prev = C.REPORTS / "run_manifest.json"
        ok = prev.exists() and json.loads(prev.read_text()).get("config", {}).get("seed") == C.SEED \
            and json.loads(prev.read_text()).get("config", {}).get("n_users") == C.N_USERS \
            and (C.REPORTS / "nlp_summary.json").exists()
        if not ok:
            sys.exit("--skip-nlp refused: no NLP outputs from a run with the same seed and user count. Run without it.")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    steps = [("simulate", simulate.main), ("features", features.main), ("experiment", experiment.main),
             ("oracle", oracle.main), ("hte", hte.main), ("retention", retention.main),
             ("nlp", None if args.skip_nlp else nlp.main), ("virality", virality.main), ("impact", impact.main),
             ("powerbi", powerbi.main), ("report", report.main)]
    stages = []
    for name, fn in steps:
        if fn is None:
            stages.append({"stage": name, "status": "skipped (reused)"})
            continue
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        argv, sys.argv = sys.argv, [sys.argv[0]] + (["--llm"] if name == "report" and args.llm else [])
        try:
            fn()
        finally:
            sys.argv = argv
        stages.append({"stage": name, "status": "ran", "seconds": round(time.time() - t0, 1)})
        print(f"--- {name} done in {time.time() - t0:.1f}s")
    write_manifest(stages, started)
    print(f"\nrun manifest: {C.REPORTS / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
