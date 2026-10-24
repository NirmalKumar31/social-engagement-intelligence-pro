"""Run the full pipeline end to end:  python -m sei.pipeline [--skip-nlp] [--llm]"""
import argparse
import sys
import time
import warnings

from . import experiment, features, hte, impact, nlp, powerbi, report, retention, simulate, virality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nlp", action="store_true", help="skip the (slower) NLP module if its outputs exist")
    ap.add_argument("--llm", action="store_true", help="also generate a Claude-polished executive brief")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    steps = [("simulate", simulate.main), ("features", features.main), ("experiment", experiment.main),
             ("hte", hte.main), ("retention", retention.main), ("nlp", None if args.skip_nlp else nlp.main),
             ("virality", virality.main), ("impact", impact.main), ("powerbi", lambda: powerbi.main()),
             ("report", report.main)]
    for name, fn in steps:
        if fn is None:
            continue
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        argv, sys.argv = sys.argv, [sys.argv[0]] + (["--llm"] if name == "report" and args.llm else [])
        try:
            fn()
        finally:
            sys.argv = argv
        print(f"--- {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
