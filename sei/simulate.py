"""Simulate the three-arm feed experiment as a daily event log."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config as C

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
REGIONS = ["North America", "Europe", "LATAM", "India", "APAC"]
DEVICES = ["iOS", "Android", "Web"]
ACCOUNT_TYPES = ["casual", "creator", "business"]
CHANNELS = ["organic", "paid_social", "referral", "search"]
ECPM = {"North America": 9.0, "Europe": 6.0, "APAC": 4.0, "LATAM": 2.5, "India": 1.0}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def assign_arm(user_ids: np.ndarray, salt: str = "feed_rank_v2_2026q3") -> np.ndarray:
    buckets = np.array(
        [int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:8], 16) % 3 for u in user_ids]
    )
    return np.array(C.ARMS)[buckets]


def simulate_users(rng: np.random.Generator, n: int) -> pd.DataFrame:
    uid = np.arange(1, n + 1)
    age = rng.choice(AGE_GROUPS, n, p=[0.30, 0.31, 0.19, 0.12, 0.08])
    region = rng.choice(REGIONS, n, p=[0.22, 0.24, 0.16, 0.20, 0.18])
    p_ios = np.select([region == "North America", np.isin(region, ["India", "LATAM"])], [0.58, 0.22], 0.42)
    u = rng.random(n)
    device = np.where(u < p_ios, "iOS", np.where(u < 0.93, "Android", "Web"))
    acct = rng.choice(ACCOUNT_TYPES, n, p=[0.72, 0.20, 0.08])
    channel = rng.choice(CHANNELS, n, p=[0.45, 0.25, 0.20, 0.10])
    tenure = np.clip(rng.gamma(1.3, 520, n), 7, 3650).astype(int)
    friends = np.clip(rng.lognormal(4.4 + 0.12 * (age != "18-24") + 0.0004 * tenure.clip(0, 1500), 0.85, n), 0, 5000).astype(int)
    following = np.clip(rng.lognormal(5.3, 0.8, n), 1, 7500).astype(int)
    followers = np.clip(rng.lognormal(4.3 + 1.8 * (acct == "creator") + 1.2 * (acct == "business"), 1.1, n), 0, 2_000_000).astype(int)

    # Latent traits
    eng = (rng.normal(0, 1, n) + 0.25 * (age == "18-24") + 0.15 * (age == "25-34") - 0.2 * (age == "55+")
           + 0.40 * (acct == "creator") - 0.35 * (tenure < 90) + 0.10 * np.isin(region, ["India", "LATAM"])
           - 0.40 * (device == "Web") - 0.15 * (channel == "paid_social"))
    social = _sigmoid(rng.normal(0, 1, n) + 0.45 * np.isin(age, ["35-44", "45-54", "55+"])
                      + 0.55 * np.log(np.maximum(friends, 1) / 100) - 0.45 * (acct == "creator")
                      - 0.30 * (acct == "business") + 0.25 * (channel == "referral"))
    novelty = _sigmoid(rng.normal(0, 1, n) + 0.70 * (age == "18-24") + 0.30 * (age == "25-34")
                       - 0.30 * (age == "45-54") - 0.60 * (age == "55+") + 0.25 * (acct == "creator"))
    users = pd.DataFrame({
        "user_id": uid, "age_group": age, "region": region, "device": device, "account_type": acct,
        "acquisition_channel": channel, "tenure_days": tenure, "friend_count": friends,
        "following_count": following, "follower_count": followers,
    })
    users["is_new_user"] = (users.tenure_days < 90).astype(int)
    users["arm"] = assign_arm(uid)
    users["_eng"], users["_social"], users["_novelty"] = eng, social, novelty
    return users


def simulate_daily(rng: np.random.Generator, users: pd.DataFrame):
    n = len(users)
    days = np.arange(-C.PRE_DAYS, C.EXP_DAYS)
    dates = pd.Timestamp(C.START_DATE) + pd.to_timedelta(days, unit="D")
    eng, s, v = users._eng.values, users._social.values, users._novelty.values
    arm = users.arm.values
    is_a, is_b = arm == "ai_feed", arm == "ai_feed_social"
    creator = (users.account_type == "creator").values
    business = (users.account_type == "business").values
    ecpm = users.region.map(ECPM).values
    new = users.is_new_user.values

    base_logit = 1.05 + 1.25 * eng
    base_logmin = 2.95 + 0.42 * eng
    like_rate = rng.gamma(4, 0.08, n)  # likes per minute
    comment_rate = 0.06 * (0.4 + s) * rng.gamma(4, 0.25, n)
    dm_rate = 0.09 * (0.2 + 1.3 * s) * rng.gamma(4, 0.25, n)
    share_rate = rng.gamma(3, 0.012, n)
    post_rate = np.where(creator, 0.55, np.where(business, 0.35, 0.06)) * rng.gamma(4, 0.25, n)
    ctr = np.clip(rng.normal(0.011, 0.003, n), 0.002, 0.04)
    neg_rate = rng.gamma(2, 0.004, n)

    # True individual treatment effects
    steady_a = 0.075 + 0.150 * v - 0.100 * s
    steady_b = 0.030 + 0.090 * v - 0.030 * s
    novelty_a, novelty_b = 0.09 * (0.5 + v), 0.07 * (0.5 + v)
    hazard0 = 0.0028 * np.exp(-0.75 * eng + 0.55 * new)
    haz_mult_a = np.exp(1.30 * s - 0.55 * v - 0.30)
    haz_mult_b = np.exp(0.20 * s - 0.45 * v - 0.02)

    shock_a = rng.normal(0, 0.6, n)
    shock_m = rng.normal(0, 0.35, n)
    alive = np.ones(n, dtype=bool)
    churn_day = np.full(n, -1)
    rows = []
    for t, d in zip(days, dates):
        shock_a = 0.65 * shock_a + rng.normal(0, 0.6 * np.sqrt(1 - 0.65**2), n)
        shock_m = 0.60 * shock_m + rng.normal(0, 0.35 * np.sqrt(1 - 0.60**2), n)
        weekend = d.dayofweek >= 5
        logit = base_logit + shock_a + (0.15 if weekend else 0.0) + 0.004 * t
        logmin = base_logmin + shock_m + (0.08 if weekend else 0.0)
        m_mult = np.ones(n)
        comm_mult, dm_mult, like_mult = np.ones(n), np.ones(n), np.ones(n)
        share_mult, post_mult, neg_mult, ctr_mult = np.ones(n), np.ones(n), np.ones(n), np.ones(n)
        if t >= 0:
            decay = np.exp(-t / 5.0)
            ramp = t / (C.EXP_DAYS - 1)
            m_mult = np.where(is_a, 1 + steady_a + novelty_a * decay, m_mult)
            m_mult = np.where(is_b, 1 + steady_b + novelty_b * decay, m_mult)
            logit = logit + np.where(is_a, 0.10 * v - 0.22 * s * ramp, 0) + np.where(is_b, 0.05 * v - 0.03 * s * ramp, 0)
            comm_mult = np.where(is_a, 0.80, np.where(is_b, 1.04, 1.0))
            dm_mult = np.where(is_a, 0.84, np.where(is_b, 1.02, 1.0))
            like_mult = np.where(is_a, 1.09, np.where(is_b, 1.06, 1.0))
            share_mult = np.where(is_a, 1.14, np.where(is_b, 1.09, 1.0))
            post_mult = np.where(is_a, np.where(creator, 1.12, 0.90), np.where(is_b, np.where(creator, 1.07, 1.03), 1.0))
            neg_mult = np.where(is_a, 1.18, np.where(is_b, 1.07, 1.0))
            ctr_mult = np.where(is_a, 1.04, np.where(is_b, 1.03, 1.0))
            hz = hazard0 * np.where(is_a, haz_mult_a, np.where(is_b, haz_mult_b, 1.0))
            newly = alive & (rng.random(n) < hz)
            churn_day[newly] = t
            alive &= ~newly
        active = alive & (rng.random(n) < _sigmoid(logit))
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue
        mean_min = np.exp(logmin[idx]) * m_mult[idx]
        minutes = np.round(rng.gamma(2.5, mean_min / 2.5), 1).clip(0.5, 600)
        sessions = 1 + rng.poisson(minutes / 7.0)
        likes = rng.poisson(minutes * like_rate[idx] * like_mult[idx])
        comments = rng.poisson(minutes * comment_rate[idx] * comm_mult[idx])
        dms = rng.poisson(minutes * dm_rate[idx] * dm_mult[idx])
        shares = rng.poisson(minutes * share_rate[idx] * share_mult[idx])
        posts = rng.poisson(post_rate[idx] * post_mult[idx])
        stories = rng.poisson(minutes * 0.55)
        neg = rng.poisson(minutes * neg_rate[idx] * neg_mult[idx])
        impr = rng.poisson(minutes * 0.9)
        clicks = rng.binomial(impr, np.clip(ctr[idx] * ctr_mult[idx], 0, 1))
        revenue = impr * ecpm[idx] / 1000 + clicks * 0.02
        rows.append(pd.DataFrame({
            "user_id": users.user_id.values[idx], "day": np.int16(t), "date": d,
            "sessions": sessions.astype(np.int16), "minutes": minutes.astype(np.float32),
            "likes": likes.astype(np.int32), "comments": comments.astype(np.int32),
            "dms_sent": dms.astype(np.int32), "shares": shares.astype(np.int32),
            "posts_created": posts.astype(np.int16), "stories_viewed": stories.astype(np.int32),
            "negative_feedback": neg.astype(np.int16), "ad_impressions": impr.astype(np.int32),
            "ad_clicks": clicks.astype(np.int16), "ad_revenue": revenue.astype(np.float32),
        }))
    daily = pd.concat(rows, ignore_index=True)

    truth = pd.DataFrame({
        "user_id": users.user_id.values, "true_engagement": eng, "true_social_affinity": s,
        "true_novelty_seeking": v,
        "true_tau_minutes_ai_feed": steady_a, "true_tau_minutes_ai_feed_social": steady_b,
        "true_hazard_mult_ai_feed": haz_mult_a, "true_hazard_mult_ai_feed_social": haz_mult_b,
        "latent_churn_day": churn_day,
    })
    return daily, truth


def main():
    rng = np.random.default_rng(C.SEED)
    users = simulate_users(rng, C.N_USERS)
    daily, truth = simulate_daily(rng, users)
    users = users.drop(columns=["_eng", "_social", "_novelty"])
    users.to_parquet(C.RAW / "users.parquet", index=False)
    daily.to_parquet(C.RAW / "daily_activity.parquet", index=False)
    truth.to_parquet(C.RAW / "user_ground_truth.parquet", index=False)
    print(f"users={len(users):,}  daily rows={len(daily):,}  arms={users.arm.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
