"""Time to first 7-day lapse by arm, and a lapse early-warning model scored on day 13.

A lapse is 7 days in a row with no activity. A start on day d is only fully observed when d <= 21.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time
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
LAPSE = C.CHURN_WINDOW
LAST_START = C.EXP_DAYS - LAPSE  # 21
CATS = ["age_group", "region", "device", "account_type", "acquisition_channel", "arm"]


def activity_matrix(users: pd.DataFrame, daily: pd.DataFrame) -> np.ndarray:
    act = np.zeros((len(users), C.EXP_DAYS), bool)
    pos = pd.Series(np.arange(len(users)), index=users.user_id.values)
    e = daily[daily.day >= 0]
    act[pos.reindex(e.user_id.values).values, e.day.values] = True
    return act


def first_lapse_start(act: np.ndarray, first: int = 0, last: int = LAST_START) -> np.ndarray:
    """Start day of the first fully observed lapse with start in [first, last]; -1 if none."""
    out = np.full(len(act), -1)
    for d in range(first, last + 1):
        hit = (out < 0) & ~act[:, d:d + LAPSE].any(axis=1)
        out[hit] = d
    return out


def survival(users: pd.DataFrame, act: np.ndarray, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    start = first_lapse_start(act)
    d = df[["user_id", "arm", "age_group", "is_new_user", "pre_active_days_per_day",
            "pre_friend_interactions_per_day"]].copy()
    d["E"] = (start >= 0).astype(int)
    d["T"] = np.where(start >= 0, start, LAST_START).astype(float)
    d["T"] = d["T"].clip(lower=0.5)  # lifelines needs T > 0
    km_rows, rmst = [], {}
    for arm in C.ARMS:
        g = d[d.arm == arm]
        km = KaplanMeierFitter().fit(g["T"], g["E"], label=arm)
        sf = km.survival_function_.reset_index()
        sf.columns = ["day", "survival"]
        ci = km.confidence_interval_.reset_index(drop=True)
        sf["ci_low"], sf["ci_high"], sf["arm"] = ci.iloc[:, 0], ci.iloc[:, 1], arm
        km_rows.append(sf)
        rmst[arm] = float(restricted_mean_survival_time(km, t=LAST_START))
    km = pd.concat(km_rows)
    km.to_csv(C.TABLES / "km_survival.csv", index=False)
    lr = multivariate_logrank_test(d["T"], d.arm, d["E"])

    cox_df = pd.get_dummies(d.drop(columns=["user_id"]), columns=["arm", "age_group"], dtype=float)
    cox_df = cox_df.drop(columns=["arm_control", "age_group_25-34"])
    cph = CoxPHFitter(penalizer=0.001).fit(cox_df, "T", "E")
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
    hr.columns = ["covariate", "hazard_ratio", "hr_ci_low", "hr_ci_high", "p_value"]
    hr.to_csv(C.TABLES / "cox_hazard_ratios.csv", index=False)
    ph = proportional_hazard_test(cph, cox_df, time_transform="rank").summary
    ph_p = {k: float(ph.loc[f"arm_{k}", "p"]) for k in C.TREATMENTS}

    # Rate ratio per person-day at risk (what impact.py assumes)
    rates = d.groupby("arm").agg(events=("E", "sum"), person_days=("T", "sum"))
    rr = {}
    for arm in C.TREATMENTS:
        r = (rates.loc[arm, "events"] / rates.loc[arm, "person_days"]) / (rates.loc["control", "events"] / rates.loc["control", "person_days"])
        se = np.sqrt(1 / rates.loc[arm, "events"] + 1 / rates.loc["control", "events"])
        rr[arm] = {"rate_ratio": float(r), "ci": [float(r * np.exp(-1.96 * se)), float(r * np.exp(1.96 * se))]}

    # Benchmark only: same model on the simulator's true quit day
    truth = pd.read_parquet(C.RAW / "user_ground_truth.parquet").set_index("user_id").reindex(d.user_id)
    o = cox_df.copy()
    lq = truth.latent_churn_day.values
    o["E"] = (lq >= 0).astype(int)
    o["T"] = np.where(lq >= 0, lq + 0.5, C.EXP_DAYS).astype(float)
    ocph = CoxPHFitter(penalizer=0.001).fit(o, "T", "E")
    oracle = {k: float(ocph.summary.loc[f"arm_{k}", "exp(coef)"]) for k in C.TREATMENTS}

    hrs = hr.set_index("covariate")
    summary = {"endpoint": f"time to first {LAPSE}-day lapse (start day), censored at day {LAST_START}",
               "events": {a: int(rates.loc[a, "events"]) for a in C.ARMS},
               "logrank_p": float(lr.p_value),
               "hr_ai_feed": float(hrs.loc["arm_ai_feed", "hazard_ratio"]),
               "hr_ai_feed_social": float(hrs.loc["arm_ai_feed_social", "hazard_ratio"]),
               "hr_ci": {a: [float(hrs.loc[f"arm_{a}", "hr_ci_low"]), float(hrs.loc[f"arm_{a}", "hr_ci_high"])] for a in C.TREATMENTS},
               "ph_test_p": ph_p, "cox_concordance": float(cph.concordance_index_),
               "rmst_days": rmst, "rmst_diff_vs_control": {a: rmst[a] - rmst["control"] for a in C.TREATMENTS},
               "rate_ratio": rr,
               "simulator_benchmark_hr_latent_quit": oracle}
    return summary, km


def build_features(users: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    obs = daily[daily.day <= SCORING_DAY]
    out = users.set_index("user_id")[["age_group", "region", "device", "account_type", "acquisition_channel", "arm",
                                      "tenure_days", "friend_count", "following_count", "follower_count", "is_new_user"]].copy()
    out["log_followers"] = np.log1p(out.pop("follower_count"))
    obs = obs.assign(week=np.floor_divide(obs.day, 7))  # -2, -1, 0, 1
    wk = obs.pivot_table(index="user_id", columns="week", values="minutes", aggfunc=["sum", "count"]).fillna(0)
    for w in [-2, -1, 0, 1]:
        out[f"minutes_w{w}"] = wk[("sum", w)].reindex(out.index).fillna(0) if ("sum", w) in wk else 0
        out[f"active_days_w{w}"] = wk[("count", w)].reindex(out.index).fillna(0) if ("count", w) in wk else 0
    tot = obs.groupby("user_id")[["minutes", "sessions", "comments", "dms_sent", "likes", "shares", "posts_created",
                                  "negative_feedback", "ad_clicks", "ad_impressions"]].sum().reindex(out.index).fillna(0)
    last3 = obs[obs.day > SCORING_DAY - 3].groupby("user_id").size().reindex(out.index).fillna(0)
    out["active_days_last3"] = last3
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


def campaign_value(p: np.ndarray, y: np.ndarray, threshold: float, cost: float, save_rate: float, value: float) -> float:
    """Net value of targeting everyone with score >= threshold (threshold > 1 means send to nobody)."""
    t = p >= threshold
    return float((y[t].sum() * save_rate * value) - t.sum() * cost)


def campaign_curve(p: np.ndarray, y: np.ndarray, cost: float, save_rate: float, value: float) -> pd.DataFrame:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    n = len(ys)
    rows = [{"fraction_targeted": 0.0, "users_targeted": 0, "churners_caught": 0, "recall": 0.0, "precision": np.nan,
             "threshold": np.inf, "profit_usd": 0.0}]
    for frac in np.linspace(0.01, 1, 100):
        k = int(round(frac * n))
        tp = ys[:k].sum()
        rows.append({"fraction_targeted": frac, "users_targeted": k, "churners_caught": int(tp), "recall": tp / max(ys.sum(), 1),
                     "precision": tp / max(k, 1), "threshold": ps[k - 1], "profit_usd": tp * save_rate * value - k * cost})
    return pd.DataFrame(rows)


def churn_model(users, daily, act) -> dict:
    feats = build_features(users, daily)
    pos = pd.Series(np.arange(len(users)), index=users.user_id.values)
    lapse = first_lapse_start(act, first=SCORING_DAY + 1)
    cohort_ids = users.user_id.values[act[:, SCORING_DAY]]
    X = feats.loc[cohort_ids]
    y = (lapse[pos[cohort_ids].values] >= 0).astype(int)
    truth = pd.read_parquet(C.RAW / "user_ground_truth.parquet").set_index("user_id").latent_churn_day.reindex(cohort_ids).values
    bench = {"positives": int(y.sum()), "already_quit_at_scoring": int(((truth >= 0) & (truth <= SCORING_DAY) & (y == 1)).sum()),
             "permanent_quit_after_scoring": int(((truth > SCORING_DAY) & (y == 1)).sum()),
             "temporary_lapse": int(((truth < 0) & (y == 1)).sum())}

    num = [c for c in X.columns if c not in CATS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=C.SEED, stratify=y)
    pre = lambda: ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
    logit = Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=3000, C=0.5))])
    logit_bal = Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))])
    gbm_params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=80, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    gbm = lgb.LGBMClassifier(**gbm_params)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_auc = {"logistic": roc_auc_score(y_tr, cross_val_predict(logit, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]),
              "lightgbm": roc_auc_score(y_tr, cross_val_predict(gbm, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1])}
    logit.fit(X_tr, y_tr); logit_bal.fit(X_tr, y_tr); gbm.fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(lgb.LGBMClassifier(**gbm_params), method="isotonic", cv=5).fit(X_tr, y_tr)

    preds = {"Logistic regression": logit.predict_proba(X_te)[:, 1],
             "Logistic (class-weighted)": logit_bal.predict_proba(X_te)[:, 1],
             "LightGBM (raw)": gbm.predict_proba(X_te)[:, 1],
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
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        cal_rows.append(pd.DataFrame({"model": name, "mean_predicted": mp, "fraction_positive": fr}))
    metrics = pd.DataFrame(rows)
    metrics["cv_auc"] = metrics.model.map(lambda m: cv_auc["lightgbm"] if "LightGBM" in m else cv_auc["logistic"])
    metrics.to_csv(C.TABLES / "churn_model_metrics.csv", index=False)
    pd.concat(roc_rows).to_csv(C.TABLES / "churn_roc.csv", index=False)
    pd.concat(cal_rows).to_csv(C.TABLES / "churn_calibration.csv", index=False)

    sample = X_te.sample(min(4000, len(X_te)), random_state=0)
    sv = shap.TreeExplainer(gbm).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TABLES / "churn_shap_importance.csv", index=False)

    if cv_auc["logistic"] >= cv_auc["lightgbm"]:
        deployed, make_model = "Logistic regression", lambda: Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=3000, C=0.5))])
    else:
        deployed, make_model = "LightGBM + isotonic", lambda: CalibratedClassifierCV(lgb.LGBMClassifier(**gbm_params),
                                                                                    method="isotonic", cv=3)

    metrics["deployed"] = (metrics.model == deployed).astype(int)
    metrics.to_csv(C.TABLES / "churn_model_metrics.csv", index=False)

    cost, save_rate, value = C.RETENTION_CAMPAIGN_COST, C.RETENTION_CAMPAIGN_SAVE_RATE, C.LTV_PER_RETAINED_USER
    oof_tr = cross_val_predict(make_model(), X_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=2),
                               method="predict_proba")[:, 1]
    train_curve = campaign_curve(oof_tr, y_tr, cost, save_rate, value)
    chosen = train_curve.loc[train_curve.profit_usd.idxmax()]
    rules = {"Depth chosen on training OOF": float(chosen.threshold),
             "Break-even rule (p > cost / (save x value))": cost / (save_rate * value),
             "Send to nobody": np.inf}
    p_te = preds[deployed]
    scale = C.MAU_POPULATION * (len(X) / len(users)) / len(y_te)  # population users per test user
    rng = np.random.default_rng(5)
    boot_idx = [rng.integers(0, len(y_te), len(y_te)) for _ in range(500)]
    rule_rows = []
    for name, thr in rules.items():
        v = campaign_value(p_te, y_te, thr, cost, save_rate, value) * scale / 1e6
        b = np.array([campaign_value(p_te[i], y_te[i], thr, cost, save_rate, value) for i in boot_idx]) * scale / 1e6
        rule_rows.append({"rule": name, "threshold": thr, "share_targeted": float((p_te >= thr).mean()),
                          "test_value_musd": v, "ci_low": float(np.quantile(b, 0.025)), "ci_high": float(np.quantile(b, 0.975))})
    rule_tbl = pd.DataFrame(rule_rows)
    rule_tbl.to_csv(C.TABLES / "retention_campaign_rules.csv", index=False)
    test_curve = campaign_curve(p_te, y_te, cost, save_rate, value)
    test_curve["profit_at_scale_musd"] = test_curve.profit_usd * scale / 1e6
    test_curve["profit_test_usd"] = test_curve.profit_usd
    test_curve.to_csv(C.TABLES / "retention_profit_curve.csv", index=False)

    # Out-of-fold scores for every cohort user
    oof = cross_val_predict(make_model(), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=1), method="predict_proba")[:, 1]
    scores = X[["age_group", "region", "device", "account_type", "arm"]].astype(str).copy()
    scores["churn_probability"] = oof
    scores["actual_churn_next_14d"] = y
    scores["risk_decile"] = pd.qcut(pd.Series(oof).rank(method="first"), 10, labels=False).values + 1
    scores["risk_tier"] = pd.cut(oof, [-0.01, 0.02, 0.06, 0.15, 1.0], labels=["Low", "Medium", "High", "Critical"]).astype(str)
    top_feat = imp.feature.head(6).tolist()
    scores = scores.join(X[[f for f in top_feat if f not in CATS]])
    scores.reset_index().rename(columns={"index": "user_id"}).to_parquet(C.PROCESSED / "churn_scores.parquet", index=False)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = {"Logistic regression": C.INK["secondary"], "Logistic (class-weighted)": C.INK["muted"],
            "LightGBM (raw)": C.SEQ_BLUE[2], "LightGBM + isotonic": C.SEQ_BLUE[5]}
    for name, g in pd.concat(roc_rows).groupby("model"):
        auc = metrics.set_index("model").loc[name, "roc_auc"]
        axes[0].plot(g.fpr, g.tpr, color=cols[name], lw=2, label=f"{name} · AUC {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color=C.INK["muted"], lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="Lapse early-warning: ROC (held-out, active-on-day-13 cohort)")
    axes[0].legend(fontsize=8, loc="lower right")
    for name, g in pd.concat(cal_rows).groupby("model"):
        axes[1].plot(g.mean_predicted, g.fraction_positive, marker="o", ms=5, color=cols[name], lw=1.8, label=name)
    lim = float(pd.concat(cal_rows)[["mean_predicted", "fraction_positive"]].max().max()) * 1.05
    axes[1].plot([0, lim], [0, lim], color=C.INK["muted"], lw=1)
    axes[1].set(xlabel="Predicted lapse probability", ylabel="Observed lapse rate", title="Calibration (reliability curve)",
                xlim=(0, lim), ylim=(0, lim))
    axes[1].legend(fontsize=8)
    save(fig, "09_churn_roc_calibration")

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, max_display=14, show=False, plot_size=None)
    plt.title("What drives lapse risk (SHAP, LightGBM)", loc="left", fontweight="bold")
    save(fig, "10_churn_shap_beeswarm")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(test_curve.fraction_targeted * 100, test_curve.profit_at_scale_musd, color=C.SEQ_BLUE[5], lw=2.2,
            label="Held-out test curve (descriptive)")
    chosen_row = rule_tbl.iloc[0]
    ax.axvline(chosen_row.share_targeted * 100, color=C.INK["muted"], lw=1)
    ax.annotate(f"Depth frozen on training data: {chosen_row.share_targeted:.0%} targeted\n"
                f"test value ${chosen_row.test_value_musd:,.2f}M [{chosen_row.ci_low:,.2f}, {chosen_row.ci_high:,.2f}]",
                (chosen_row.share_targeted * 100, chosen_row.test_value_musd), xytext=(14, -34), textcoords="offset points",
                color=C.INK["secondary"])
    ax.axhline(0, color=C.INK["axis"], lw=1)
    ax.set(xlabel="% of active users targeted (highest risk first)", ylabel="Net campaign value at scale ($M)",
           title=f"Retention nudge economics (assumed {save_rate:.0%} of lapsers saved, ${cost:.2f}/user, ${value:.0f} value)")
    ax.legend(loc="lower left")
    save(fig, "11_retention_profit_curve")

    return {"task": "predict a 7-day lapse starting days 14-21 among users active on day 13",
            "deployed_model": deployed, "deployed_selected_by": "training cross-validated AUC",
            "cohort_size": int(len(X)), "base_churn_rate": float(y.mean()), "simulator_benchmark": bench,
            "cv_auc": cv_auc, "holdout": metrics.set_index("model").to_dict(orient="index"),
            "top_drivers": imp.feature.head(8).tolist(),
            "campaign_rules": rule_tbl.replace({np.inf: None}).to_dict(orient="records"),
            "campaign_optimal_fraction": float(rule_tbl.iloc[0].share_targeted),
            "campaign_optimal_profit_musd": float(rule_tbl.iloc[0].test_value_musd),
            "campaign_value_ci_musd": [float(rule_tbl.iloc[0].ci_low), float(rule_tbl.iloc[0].ci_high)],
            "campaign_assumptions": {"cost": cost, "save_rate": save_rate, "value": value,
                                     "note": "save_rate is assumed, not measured; needs a randomised nudge holdout"}}


def main():
    users, daily = load_raw()
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    act = activity_matrix(users, daily)
    surv, km = survival(users, act, df)
    model = churn_model(users, daily, act)
    setup()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for arm in C.ARMS:
        g = km[km.arm == arm]
        ax.step(g.day, g.survival * 100, where="post", color=C.ARM_COLORS[arm], lw=2, label=C.ARM_LABELS[arm])
        ax.fill_between(g.day, g.ci_low * 100, g.ci_high * 100, step="post", color=C.ARM_COLORS[arm], alpha=0.12, lw=0)
    ax.set(xlabel="Experiment day", ylabel="Users without a 7-day lapse yet (%)", xlim=(0, LAST_START),
           title=f"Time to first 7-day lapse · Cox HR: AI feed {surv['hr_ai_feed']:.2f}, friend boost {surv['hr_ai_feed_social']:.2f}")
    ax.legend(loc="lower left")
    save(fig, "12_kaplan_meier")
    out = {"survival": surv, "churn_model": model}
    with open(C.REPORTS / "retention_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
