"""Roll the event log up to the user grain used by every downstream analysis."""
from __future__ import annotations

import pandas as pd

from . import config as C

COUNT_COLS = ["sessions", "minutes", "likes", "comments", "dms_sent", "shares", "posts_created",
              "stories_viewed", "negative_feedback", "ad_impressions", "ad_clicks", "ad_revenue"]


def load_raw():
    users = pd.read_parquet(C.RAW / "users.parquet")
    daily = pd.read_parquet(C.RAW / "daily_activity.parquet")
    return users, daily


def _window(daily: pd.DataFrame, lo: int, hi: int, prefix: str) -> pd.DataFrame:
    w = daily[(daily.day >= lo) & (daily.day <= hi)]
    agg = w.groupby("user_id")[COUNT_COLS].sum()
    agg["active_days"] = w.groupby("user_id").size()
    agg["friend_interactions"] = agg.comments + agg.dms_sent
    return agg.add_prefix(prefix)


def build_user_table(users: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    pre = _window(daily, -C.PRE_DAYS, -1, "pre_")
    exp = _window(daily, 0, C.EXP_DAYS - 1, "exp_")
    late = _window(daily, C.EXP_DAYS // 2, C.EXP_DAYS - 1, "late_")  # post-novelty steady state
    df = users.set_index("user_id").join(pre).join(exp).join(late).fillna(0).reset_index()
    # Per-day normalised metrics: inactive days count as zero (intent-to-treat)
    for c in COUNT_COLS + ["active_days", "friend_interactions"]:
        df[f"pre_{c}_per_day"] = df[f"pre_{c}"] / C.PRE_DAYS
        df[f"exp_{c}_per_day"] = df[f"exp_{c}"] / C.EXP_DAYS
        df[f"late_{c}_per_day"] = df[f"late_{c}"] / (C.EXP_DAYS - C.EXP_DAYS // 2)
    last_active = daily[daily.day >= 0].groupby("user_id").day.max()
    df["last_active_day"] = df.user_id.map(last_active).fillna(-1).astype(int)
    df["churned"] = (df.last_active_day < C.EXP_DAYS - C.CHURN_WINDOW).astype(int)
    df["retained"] = 1 - df.churned
    return df


def main():
    users, daily = load_raw()
    df = build_user_table(users, daily)
    df.to_parquet(C.PROCESSED / "user_experiment.parquet", index=False)
    print(f"user table: {df.shape}, churn rate {df.churned.mean():.3%}")


if __name__ == "__main__":
    main()
