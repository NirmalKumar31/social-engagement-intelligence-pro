"""Retention: survival analysis by arm + a leakage-free churn early-warning model.

Early-warning framing (point-in-time correct):
  observation window = pre-period + experiment days 0-13  ->  features
  outcome window     = experiment days 14-27              ->  label (no activity at all)
  cohort             = users active at least once in days 7-13 (still "alive" at scoring time)
Nothing computed from the outcome window ever enters a feature.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import load_raw
from .plotting import plt, save, setup

SCORING_DAY = 13
CATS = ["age_group", "region", "device", "account_type", "acquisition_channel", "arm"]


# ---------------------------------------------------------------- survival
def survival(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["T"] = np.where(d.churned == 1, d.last_active_day + 1, C.EXP_DAYS).clip(0, C.EXP_DAYS)
    d["E"] = d.churned
    km_rows = []
    for arm in C.ARMS:
        g = d[d.arm == arm]
        km = KaplanMeierFitter().fit(g["T"], g["E"], label=arm)
        sf = km.survival_function_.reset_index()
        sf.columns = ["day", "survival"]
        ci = km.confidence_interval_.reset_index(drop=True)
        sf["ci_low"], sf["ci_high"], sf["arm"] = ci.iloc[:, 0], ci.iloc[:, 1], arm
        km_rows.append(sf)
    km = pd.concat(km_rows)
    km.to_csv(C.TABLES / "km_survival.csv", index=False)
    lr = multivariate_logrank_test(d["T"], d.arm, d["E"])
    cox_df = pd.get_dummies(d[["T", "E", "arm", "age_group", "is_new_user", "pre_active_days_per_day",
                               "pre_friend_interactions_per_day"]], columns=["arm", "age_group"], drop_first=False, dtype=float)
    cox_df = cox_df.drop(columns=["arm_control", "age_group_25-34"])
    cph = CoxPHFitter(penalizer=0.001).fit(cox_df, "T", "E")
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
    hr.columns = ["covariate", "hazard_ratio", "hr_ci_low", "hr_ci_high", "p_value"]
    hr.to_csv(C.TABLES / "cox_hazard_ratios.csv", index=False)
    return {"logrank_p": float(lr.p_value),
            "hr_ai_feed": float(hr.set_index("covariate").loc["arm_ai_feed", "hazard_ratio"]),
            "hr_ai_feed_social": float(hr.set_index("covariate").loc["arm_ai_feed_social", "hazard_ratio"]),
            "cox_concordance": float(cph.concordance_index_)}, km, hr


# ---------------------------------------------------------------- early-warning features
def build_features(users: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    obs = daily[daily.day <= SCORING_DAY]
    out = users.set_index("user_id")[["age_group", "region", "device", "account_type", "acquisition_channel", "arm",
                                      "tenure_days", "friend_count", "following_count", "follower_count", "is_new_user"]].copy()
    out["log_followers"] = np.log1p(out.pop("follower_count"))
    obs = obs.assign(week=np.floor_divide(obs.day, 7))  # -2,-1,0,1
    wk = obs.pivot_table(index="user_id", columns="week", values="minutes", aggfunc=["sum", "count"]).fillna(0)
    for w in [-2, -1, 0, 1]:
        out[f"minutes_w{w}"] = wk[("sum", w)].reindex(out.index).fillna(0) if ("sum", w) in wk else 0
        out[f"active_days_w{w}"] = wk[("count", w)].reindex(out.index).fillna(0) if ("count", w) in wk else 0
    tot = obs.groupby("user_id")[["minutes", "sessions", "comments", "dms_sent", "likes", "shares", "posts_created",
                                  "negative_feedback", "ad_clicks", "ad_impressions"]].sum().reindex(out.index).fillna(0)
    out["recency_days"] = SCORING_DAY - obs.groupby("user_id").day.max().reindex(out.index).fillna(-C.PRE_DAYS - 1)
    out["minutes_per_session"] = tot.minutes / tot.sessions.replace(0, np.nan)
    out["friend_interactions_rate"] = (tot.comments + tot.dms_sent) / tot.minutes.replace(0, np.nan) * 100
    out["like_rate"] = tot.likes / tot.minutes.replace(0, np.nan)
    out["neg_feedback_rate"] = tot.negative_feedback / tot.minutes.replace(0, np.nan) * 100
    out["posts_total"] = tot.posts_created
    out["ctr"] = tot.ad_clicks / tot.ad_impressions.replace(0, np.nan)
    out["engagement_trend"] = (out.minutes_w1 + 1) / (out["minutes_w-1"] + out["minutes_w-2"] + 2) * 2
    out["activity_drop"] = (out["active_days_w-2"] + out["active_days_w-1"]) / 2 - out.active_days_w1
    for c in CATS:
        out[c] = out[c].astype("category")
    return out.fillna({"minutes_per_session": 0, "friend_interactions_rate": 0, "like_rate": 0,
                       "neg_feedback_rate": 0, "ctr": 0})


def churn_model(users, daily) -> dict:
    feats = build_features(users, daily)
    alive = daily[(daily.day >= SCORING_DAY - 6) & (daily.day <= SCORING_DAY)].user_id.unique()
    future = daily[daily.day > SCORING_DAY].user_id.unique()
    X = feats.loc[feats.index.isin(alive)]
    y = (~X.index.isin(future)).astype(int)
    num = [c for c in X.columns if c not in CATS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=C.SEED, stratify=y)

    logit = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), num),
                                                 ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])),
                      ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])
    gbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=80,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_auc = {"logistic": roc_auc_score(y_tr, cross_val_predict(logit, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]),
              "lightgbm": roc_auc_score(y_tr, cross_val_predict(gbm, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1])}
    logit.fit(X_tr, y_tr)
    gbm.fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(lgb.LGBMClassifier(**gbm.get_params()), method="isotonic", cv=5).fit(X_tr, y_tr)

    preds = {"Logistic regression": logit.predict_proba(X_te)[:, 1], "LightGBM (raw)": gbm.predict_proba(X_te)[:, 1],
             "LightGBM + isotonic": cal.predict_proba(X_te)[:, 1]}
    rows, roc_rows, cal_rows = [], [], []
    base = y_te.mean()
    for name, p in preds.items():
        top = p >= np.quantile(p, 0.9)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p),
                     "brier": brier_score_loss(y_te, p), "log_loss": log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)),
                     "top_decile_lift": y_te[top].mean() / base, "top_decile_precision": y_te[top].mean()})
        fpr, tpr, _ = roc_curve(y_te, p)
        sel = np.linspace(0, len(fpr) - 1, 200).astype(int)
        roc_rows.append(pd.DataFrame({"model": name, "fpr": fpr[sel], "tpr": tpr[sel]}))
        fr, mp = calibration_curve(y_te, p, n_bins=12, strategy="quantile")
        cal_rows.append(pd.DataFrame({"model": name, "mean_predicted": mp, "fraction_positive": fr}))
    metrics = pd.DataFrame(rows)
    metrics["cv_auc"] = metrics.model.map({"Logistic regression": cv_auc["logistic"], "LightGBM (raw)": cv_auc["lightgbm"],
                                           "LightGBM + isotonic": cv_auc["lightgbm"]})
    metrics.to_csv(C.TABLES / "churn_model_metrics.csv", index=False)
    pd.concat(roc_rows).to_csv(C.TABLES / "churn_roc.csv", index=False)
    pd.concat(cal_rows).to_csv(C.TABLES / "churn_calibration.csv", index=False)

    # SHAP on the raw booster (calibration is monotone, so rankings are unchanged)
    sample = X_te.sample(4000, random_state=0)
    sv = shap.TreeExplainer(gbm).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TABLES / "churn_shap_importance.csv", index=False)

    # Retention-campaign economics on calibrated probabilities
    p = preds["LightGBM + isotonic"]
    order = np.argsort(-p)
    yy = y_te[order]
    n = len(yy)
    scale = C.MAU_POPULATION * (len(X) / len(users)) / n  # population users represented by one test user
    prof = []
    for frac in np.linspace(0.01, 1, 100):
        k = int(frac * n)
        tp = yy[:k].sum()
        value = tp * C.RETENTION_CAMPAIGN_SAVE_RATE * C.LTV_PER_RETAINED_USER - k * C.RETENTION_CAMPAIGN_COST
        prof.append({"fraction_targeted": frac, "users_targeted": k, "churners_caught": int(tp),
                     "recall": tp / yy.sum(), "precision": tp / max(k, 1), "profit_test_usd": value,
                     "profit_at_scale_musd": value * scale / 1e6})
    prof = pd.DataFrame(prof)
    prof.to_csv(C.TABLES / "retention_profit_curve.csv", index=False)
    best = prof.loc[prof.profit_test_usd.idxmax()]

    # Score every alive user out-of-fold for dashboards
    oof = cross_val_predict(CalibratedClassifierCV(lgb.LGBMClassifier(**gbm.get_params()), method="isotonic", cv=3),
                            X, y, cv=StratifiedKFold(3, shuffle=True, random_state=1), method="predict_proba")[:, 1]
    scores = X[["age_group", "region", "device", "account_type", "arm"]].astype(str).copy()
    scores["churn_probability"] = oof
    scores["actual_churn_next_14d"] = y
    scores["risk_decile"] = pd.qcut(pd.Series(oof).rank(method="first"), 10, labels=False).values + 1
    scores["risk_tier"] = pd.cut(oof, [-0.01, 0.05, 0.15, 0.35, 1.0], labels=["Low", "Medium", "High", "Critical"]).astype(str)
    top_feat = imp.feature.head(6).tolist()
    scores = scores.join(X[[f for f in top_feat if f not in CATS]])
    scores.reset_index().to_parquet(C.PROCESSED / "churn_scores.parquet", index=False)

    # Figures
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = {"Logistic regression": C.INK["secondary"], "LightGBM (raw)": C.SEQ_BLUE[2], "LightGBM + isotonic": C.SEQ_BLUE[5]}
    for r in pd.concat(roc_rows).groupby("model"):
        name, g = r
        auc = metrics.set_index("model").loc[name, "roc_auc"]
        axes[0].plot(g.fpr, g.tpr, color=cols[name], lw=2, label=f"{name} · AUC {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color=C.INK["muted"], lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Churn early-warning: ROC (held-out)")
    axes[0].legend(fontsize=8, loc="lower right")
    for name, g in pd.concat(cal_rows).groupby("model"):
        axes[1].plot(g.mean_predicted, g.fraction_positive, marker="o", ms=5, color=cols[name], lw=1.8, label=name)
    axes[1].plot([0, 1], [0, 1], color=C.INK["muted"], lw=1)
    axes[1].set(xlabel="Predicted churn probability", ylabel="Observed churn rate", title="Calibration (reliability curve)",
                xlim=(0, 0.9), ylim=(0, 0.9))
    axes[1].legend(fontsize=8)
    save(fig, "09_churn_roc_calibration")

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, max_display=14, show=False, plot_size=None)
    plt.title("What drives churn risk (SHAP, LightGBM)", loc="left", fontweight="bold")
    save(fig, "10_churn_shap_beeswarm")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(prof.fraction_targeted * 100, prof.profit_at_scale_musd, color=C.SEQ_BLUE[5], lw=2.2)
    ax.axvline(best.fraction_targeted * 100, color=C.INK["muted"], lw=1)
    ax.annotate(f"Optimum: target top {best.fraction_targeted:.0%}\n≈ ${best.profit_at_scale_musd:,.1f}M net",
                (best.fraction_targeted * 100, best.profit_at_scale_musd), xytext=(12, -30), textcoords="offset points",
                color=C.INK["secondary"])
    ax.axhline(0, color=C.INK["axis"], lw=1)
    ax.set(xlabel="% of at-risk-cohort users targeted (highest risk first)", ylabel="Net campaign value at scale ($M)",
           title="Retention campaign profit curve")
    save(fig, "11_retention_profit_curve")

    return {"cohort_size": int(len(X)), "base_churn_rate": float(y.mean()), "cv_auc": cv_auc,
            "holdout": metrics.set_index("model").to_dict(orient="index"),
            "top_drivers": imp.feature.head(8).tolist(),
            "campaign_optimal_fraction": float(best.fraction_targeted),
            "campaign_optimal_profit_musd": float(best.profit_at_scale_musd),
            "campaign_recall_at_optimum": float(best.recall)}


def main():
    users, daily = load_raw()
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    surv, km, hr = survival(df)
    model = churn_model(users, daily)
    setup()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for arm in C.ARMS:
        g = km[km.arm == arm]
        ax.step(g.day, g.survival * 100, where="post", color=C.ARM_COLORS[arm], lw=2, label=C.ARM_LABELS[arm])
        ax.fill_between(g.day, g.ci_low * 100, g.ci_high * 100, step="post", color=C.ARM_COLORS[arm], alpha=0.12, lw=0)
    ax.set(xlabel="Experiment day", ylabel="Users not yet churned (%)",
           title=f"Kaplan-Meier retention · AI feed HR {surv['hr_ai_feed']:.2f}, friend boost HR {surv['hr_ai_feed_social']:.2f}",
           ylim=(88, 100.5), xlim=(0, C.EXP_DAYS - C.CHURN_WINDOW))
    ax.legend(loc="lower left")
    save(fig, "12_kaplan_meier")
    out = {"survival": surv, "churn_model": model}
    with open(C.REPORTS / "retention_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
