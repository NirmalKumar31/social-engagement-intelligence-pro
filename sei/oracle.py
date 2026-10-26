"""Counterfactual outcomes for every user under every arm, by re-simulating with common seeds.

Used only to validate the causal models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import build_user_table
from .simulate import simulate_daily

REPS = 16


def outcome(df: pd.DataFrame) -> np.ndarray:
    """Assumed utility score (USD-like units): 365 x late-window (days 14-27) ad revenue/day - churned x LTV."""
    return (df.late_ad_revenue_per_day * C.ANNUAL_DAYS - df.churned * C.LTV_PER_RETAINED_USER).values


def main(reps: int = REPS):
    users = pd.read_parquet(C.RAW / "users.parquet")
    truth = pd.read_parquet(C.RAW / "user_ground_truth.parquet").set_index("user_id").reindex(users.user_id)
    base = users.copy()
    base["_eng"], base["_social"], base["_novelty"] = (truth.true_engagement.values, truth.true_social_affinity.values,
                                                       truth.true_novelty_seeking.values)
    ys = {a: np.zeros((reps, len(users))) for a in C.ARMS}
    for r in range(reps):
        for a in C.ARMS:
            u = base.assign(arm=a)
            daily, _ = simulate_daily(np.random.default_rng(10_000 + r), u)
            df = build_user_table(u.drop(columns=["_eng", "_social", "_novelty"]), daily)
            ys[a][r] = outcome(df)
        print(f"oracle replication {r + 1}/{reps} done")
    out = pd.DataFrame({"user_id": users.user_id.values})
    for a in C.ARMS:
        out[f"y_{a}"] = ys[a].mean(axis=0)
    for a in C.TREATMENTS:
        eff = ys[a] - ys["control"]
        out[f"oracle_cate_{a}"] = eff.mean(axis=0)
        out[f"oracle_cate_{a}_mc_se"] = eff.std(axis=0, ddof=1) / np.sqrt(reps)
    out.to_parquet(C.PROCESSED / "oracle_outcomes.parquet", index=False)
    for a in C.TREATMENTS:
        print(f"{a}: mean oracle CATE {out[f'oracle_cate_{a}'].mean():.3f}, "
              f"median per-user MC se {out[f'oracle_cate_{a}_mc_se'].median():.3f}")


if __name__ == "__main__":
    main()
