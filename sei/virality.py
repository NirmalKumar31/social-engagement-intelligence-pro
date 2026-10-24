"""What makes content spread? A real-data model on UCI Online News Popularity (39,644 Mashable articles).

* Time-based split (train on older articles, test on the newest 25%) to mimic production scoring
* LightGBM classifier for "viral" (top-quartile shares) and a log-shares regressor
* SHAP drivers + creator-facing playbook tables (best channel / weekday / media mix)
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .plotting import plt, save, setup

CHANNELS = {"data_channel_is_lifestyle": "Lifestyle", "data_channel_is_entertainment": "Entertainment",
            "data_channel_is_bus": "Business", "data_channel_is_socmed": "Social Media",
            "data_channel_is_tech": "Tech", "data_channel_is_world": "World"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def load() -> pd.DataFrame:
    df = pd.read_csv(C.EXTERNAL / "uci_news" / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv")
    df.columns = df.columns.str.strip()
    df["channel"] = "Other"
    for col, name in CHANNELS.items():
        df.loc[df[col] == 1, "channel"] = name
    df["weekday"] = "monday"
    for d in WEEKDAYS:
        df.loc[df[f"weekday_is_{d}"] == 1, "weekday"] = d
    df["publish_date"] = pd.to_datetime(df.url.str.extract(r"/(\d{4}/\d{2}/\d{2})/")[0], format="%Y/%m/%d")
    return df


def main():
    df = load()
    viral_cut = df.shares.quantile(0.75)
    df["viral"] = (df.shares >= viral_cut).astype(int)
    feats = [c for c in df.columns if c not in {"url", "timedelta", "shares", "viral", "channel", "weekday", "publish_date"}]
    df = df.sort_values("publish_date").reset_index(drop=True)
    cut = int(len(df) * 0.75)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
    clf = lgb.LGBMClassifier(**params).fit(tr[feats], tr.viral)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)).fit(tr[feats], tr.viral)
    reg = lgb.LGBMRegressor(**params).fit(tr[feats], np.log1p(tr.shares))
    p, pb = clf.predict_proba(te[feats])[:, 1], base.predict_proba(te[feats])[:, 1]
    lp = reg.predict(te[feats])
    metrics = {
        "n_articles": int(len(df)), "train_until": str(tr.publish_date.max().date()), "test_from": str(te.publish_date.min().date()),
        "viral_threshold_shares": float(viral_cut),
        "auc_lightgbm": float(roc_auc_score(te.viral, p)), "auc_logistic": float(roc_auc_score(te.viral, pb)),
        "r2_log_shares": float(r2_score(np.log1p(te.shares), lp)),
        "mae_shares_median_abs": float(mean_absolute_error(te.shares, np.expm1(lp))),
        "precision_top10pct": float(te.viral[p >= np.quantile(p, 0.9)].mean()), "base_viral_rate_test": float(te.viral.mean()),
    }
    sample = te[feats].sample(3000, random_state=0)
    sv = shap.TreeExplainer(clf).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TABLES / "virality_shap_importance.csv", index=False)

    playbook = []
    for dim in ["channel", "weekday"]:
        g = df.groupby(dim).agg(articles=("shares", "size"), median_shares=("shares", "median"), viral_rate=("viral", "mean"))
        playbook.append(g.reset_index().rename(columns={dim: "level"}).assign(dimension=dim))
    df["images_bucket"] = pd.cut(df.num_imgs, [-1, 0, 1, 5, 15, 200], labels=["0", "1", "2-5", "6-15", "16+"]).astype(str)
    df["videos_bucket"] = pd.cut(df.num_videos, [-1, 0, 1, 5, 200], labels=["0", "1", "2-5", "6+"]).astype(str)
    df["title_sentiment_bucket"] = pd.cut(df.title_sentiment_polarity, [-1.01, -0.1, 0.1, 1.01],
                                          labels=["negative", "neutral", "positive"]).astype(str)
    for dim in ["images_bucket", "videos_bucket", "title_sentiment_bucket"]:
        g = df.groupby(dim).agg(articles=("shares", "size"), median_shares=("shares", "median"), viral_rate=("viral", "mean"))
        playbook.append(g.reset_index().rename(columns={dim: "level"}).assign(dimension=dim))
    pb_tbl = pd.concat(playbook)[["dimension", "level", "articles", "median_shares", "viral_rate"]]
    pb_tbl.to_csv(C.TABLES / "virality_playbook.csv", index=False)
    weekly = df.set_index("publish_date").resample("W").agg({"shares": "median", "viral": "mean", "url": "count"}).reset_index()
    weekly.columns = ["week", "median_shares", "viral_rate", "articles"]
    weekly.to_csv(C.TABLES / "virality_weekly.csv", index=False)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    top = imp.head(12)[::-1]
    axes[0].barh(top.feature, top.mean_abs_shap, color=C.SEQ_BLUE[4], height=0.6)
    axes[0].set(title="Top virality drivers (mean |SHAP|)", xlabel="Impact on log-odds of going viral")
    ch = pb_tbl[pb_tbl.dimension == "channel"].sort_values("viral_rate")
    axes[1].barh(ch.level, ch.viral_rate * 100, color=C.SEQ_BLUE[4], height=0.6)
    for yv, v in enumerate(ch.viral_rate):
        axes[1].text(v * 100 + 0.5, yv, f"{v:.0%}", va="center", color=C.INK["secondary"])
    axes[1].set(title="Viral rate by content channel", xlabel="% of articles in top-quartile shares")
    fig.suptitle(f"Real data: 39,644 articles · time-split AUC {metrics['auc_lightgbm']:.3f} (LightGBM) vs "
                 f"{metrics['auc_logistic']:.3f} (logistic)", x=0.01, ha="left", color=C.INK["secondary"])
    fig.tight_layout()
    save(fig, "16_virality_drivers")
    metrics["top_drivers"] = imp.feature.head(8).tolist()
    with open(C.REPORTS / "virality_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
