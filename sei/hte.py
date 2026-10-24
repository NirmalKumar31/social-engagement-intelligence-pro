"""Heterogeneous treatment effects and personalised rollout policy.

Outcome = annualised net user value (USD) = steady-state ad revenue/day * 365 - churned * LTV.
Revenue uses the post-novelty window (experiment days 14-27) so week-1 novelty is not annualised.
It folds the engagement upside and the retention downside into one number a PM can act on.

* T-learner and cross-fitted DR-learner (doubly robust pseudo-outcomes, known propensity)
* Qini / uplift curves on a held-out fold
* Validation against the simulator's ground-truth individual effects
* Three-arm policy learning, valued off-policy with IPW on held-out data, plus an
  interpretable depth-3 "policy tree" that turns the model into rollout rules
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

from . import config as C
from .plotting import plt, save, setup

PRE = ["minutes", "active_days", "sessions", "likes", "comments", "dms_sent", "shares", "posts_created",
       "ad_revenue", "friend_interactions", "negative_feedback"]
CATS = ["age_group", "region", "device", "account_type", "acquisition_channel"]
LGB_PARAMS = dict(n_estimators=350, learning_rate=0.04, num_leaves=24, min_child_samples=150,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[f"pre_{c}_per_day" for c in PRE] + ["tenure_days", "friend_count", "following_count", "is_new_user"]].copy()
    X["log_followers"] = np.log1p(df.follower_count)
    X = pd.concat([X, pd.get_dummies(df[CATS], dtype=float)], axis=1)
    X.columns = [c.replace(" ", "_").replace("+", "plus").replace("-", "_") for c in X.columns]
    return X


def outcome(df: pd.DataFrame) -> np.ndarray:
    return (df.late_ad_revenue_per_day * C.ANNUAL_DAYS - df.churned * C.LTV_PER_RETAINED_USER).values


def _reg():
    return lgb.LGBMRegressor(**LGB_PARAMS)


def t_learner(Xtr, ytr, ttr, Xte):
    m1, m0 = _reg().fit(Xtr[ttr == 1], ytr[ttr == 1]), _reg().fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict(Xte) - m0.predict(Xte)


def dr_learner(Xtr, ytr, ttr, Xte, e: float, seed: int = 0):
    phi = np.zeros(len(ytr))
    for fit_idx, pred_idx in KFold(3, shuffle=True, random_state=seed).split(Xtr):
        Xf, yf, tf = Xtr.iloc[fit_idx], ytr[fit_idx], ttr[fit_idx]
        m1 = _reg().fit(Xf[tf == 1], yf[tf == 1])
        m0 = _reg().fit(Xf[tf == 0], yf[tf == 0])
        Xp, yp, tp = Xtr.iloc[pred_idx], ytr[pred_idx], ttr[pred_idx]
        mu1, mu0 = m1.predict(Xp), m0.predict(Xp)
        phi[pred_idx] = mu1 - mu0 + tp * (yp - mu1) / e - (1 - tp) * (yp - mu0) / (1 - e)
    final = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": 250, "min_child_samples": 400}).fit(Xtr, phi)
    return final.predict(Xte), final


def qini_curve(tau_hat, y, t, n_points: int = 50) -> pd.DataFrame:
    order = np.argsort(-tau_hat)
    y, t = y[order], t[order]
    n = len(y)
    rows = []
    for frac in np.linspace(0, 1, n_points + 1)[1:]:
        k = int(frac * n)
        yt, yc = y[:k][t[:k] == 1], y[:k][t[:k] == 0]
        uplift = (yt.mean() - yc.mean()) * k if len(yt) and len(yc) else 0.0
        rows.append({"fraction_targeted": frac, "cumulative_gain": uplift})
    out = pd.DataFrame(rows)
    total = out.cumulative_gain.iloc[-1]
    out["random"] = out.fraction_targeted * total
    return out


def ipw_policy_value(policy: np.ndarray, arm: np.ndarray, y: np.ndarray, probs: dict, n_boot: int = 400, seed: int = 0):
    """Self-normalised IPW value of a policy, with bootstrap draws (same seed => paired across policies)."""
    w = np.array([1 / probs[a] for a in arm]) * (policy == arm)
    vals = w * y
    rng = np.random.default_rng(seed)
    boots = np.array([vals[idx].sum() / w[idx].sum() for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    return vals.sum() / w.sum(), boots


def main():
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    truth = pd.read_parquet(C.RAW / "user_ground_truth.parquet")
    df = df.merge(truth, on="user_id")
    X = feature_matrix(df)
    y = outcome(df)
    probs = (df.arm.value_counts() / len(df)).to_dict()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.35, random_state=C.SEED, stratify=df.arm)
    te = df.iloc[idx_te].reset_index(drop=True)
    X_te = X.iloc[idx_te].reset_index(drop=True)
    y_te = y[idx_te]

    cate, qinis, summary = {}, [], {}
    for arm in C.TREATMENTS:
        mtr = df.arm.iloc[idx_tr].isin(["control", arm]).values
        tr_idx = idx_tr[mtr]
        Xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
        ttr = (df.arm.iloc[tr_idx] == arm).astype(int).values
        e = ttr.mean()
        tau_t = t_learner(Xtr, ytr, ttr, X_te)
        tau_dr, final_model = dr_learner(Xtr, ytr, ttr, X_te, e)
        cate[arm] = tau_dr
        te[f"cate_{arm}"] = tau_dr
        te[f"cate_t_{arm}"] = tau_t
        mte = te.arm.isin(["control", arm]).values
        t_te = (te.arm[mte] == arm).astype(int).values
        for name, tau in [("DR-learner", tau_dr), ("T-learner", tau_t)]:
            q = qini_curve(tau[mte], y_te[mte], t_te)
            q["model"], q["arm"] = name, arm
            qinis.append(q)
        # Ground-truth check: does the model rank users like their true (latent) effects do?
        # True annualised value effect ~ revenue lift (tau_minutes * baseline revenue) minus churn-hazard harm
        true_val = (te[f"true_tau_minutes_{arm}"] * te.pre_ad_revenue_per_day * C.ANNUAL_DAYS
                    - (te[f"true_hazard_mult_{arm}"] - 1) * C.LTV_PER_RETAINED_USER * 0.08)
        summary[arm] = {
            "mean_cate_usd": float(tau_dr.mean()),
            "share_positive": float((tau_dr > 0).mean()),
            "spearman_vs_truth_dr": float(spearmanr(tau_dr, true_val).correlation),
            "spearman_vs_truth_t": float(spearmanr(tau_t, true_val).correlation),
        }
        imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp.head(15).rename("importance").to_frame().assign(arm=arm).to_csv(C.TABLES / f"hte_drivers_{arm}.csv")

    qini = pd.concat(qinis)
    qini.to_csv(C.TABLES / "qini_curves.csv", index=False)
    for arm in C.TREATMENTS:
        for model in ["DR-learner", "T-learner"]:
            q = qini[(qini.arm == arm) & (qini.model == model)]
            summary[arm][f"qini_auc_ratio_{model}"] = float(np.trapezoid(q.cumulative_gain, q.fraction_targeted)
                                                            / np.trapezoid(q.random, q.fraction_targeted))

    # Policy learning: choose the arm with the highest predicted net value per user
    pred = np.column_stack([np.zeros(len(te)), cate["ai_feed"], cate["ai_feed_social"]])
    best = np.array(C.ARMS)[pred.argmax(axis=1)]
    under35 = te.age_group.isin(["18-24", "25-34"]).values
    policies = {
        "Ship nothing (control)": np.full(len(te), "control"),
        "Ship AI feed to all": np.full(len(te), "ai_feed"),
        "Ship AI + friend boost to all": np.full(len(te), "ai_feed_social"),
        "Rule: AI feed <35, friend boost 35+": np.where(under35, "ai_feed", "ai_feed_social"),
        "Model-targeted (DR-learner)": best,
    }
    rows = []
    _, ctrl_boots = ipw_policy_value(policies["Ship nothing (control)"], te.arm.values, y_te, probs)
    for name, pol in policies.items():
        v, boots = ipw_policy_value(pol, te.arm.values, y_te, probs)
        diff = boots - ctrl_boots  # paired bootstrap: CI on the uplift itself
        rows.append({"policy": name, "value_per_user_usd": v, "uplift_ci_low": np.quantile(diff, 0.025),
                     "uplift_ci_high": np.quantile(diff, 0.975),
                     "share_ai_feed": (pol == "ai_feed").mean(), "share_ai_feed_social": (pol == "ai_feed_social").mean(),
                     "share_control": (pol == "control").mean()})
    pol = pd.DataFrame(rows)
    base = pol.loc[0, "value_per_user_usd"]
    pol["uplift_vs_control_usd"] = pol.value_per_user_usd - base
    pol["annual_value_at_scale_musd"] = pol.uplift_vs_control_usd * C.MAU_POPULATION / 1e6
    pol.to_csv(C.TABLES / "policy_values.csv", index=False)

    # Interpretable rules: depth-3 tree on the AI-feed-minus-friend-boost advantage
    adv = cate["ai_feed"] - cate["ai_feed_social"]
    readable = te[["tenure_days", "friend_count", "pre_minutes_per_day", "pre_friend_interactions_per_day", "is_new_user"]].copy()
    readable["age_35_plus"] = (~under35).astype(int)
    readable["is_creator"] = (te.account_type == "creator").astype(int)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=800, random_state=0).fit(readable, adv)
    rules = export_text(tree, feature_names=list(readable.columns), decimals=2)
    (C.REPORTS / "policy_tree_rules.txt").write_text(
        "Predicted annual value advantage (USD/user) of AI feed over AI+friend boost\n"
        "(positive = AI feed better for this leaf)\n\n" + rules)

    te["policy_arm"] = best
    te["cate_best_usd"] = pred.max(axis=1)
    te[["user_id", "arm", "age_group", "region", "device", "account_type", "cate_ai_feed", "cate_ai_feed_social",
        "policy_arm", "cate_best_usd"]].to_parquet(C.PROCESSED / "user_cate.parquet", index=False)
    deciles = []
    for arm in C.TREATMENTS:
        te["_dec"] = pd.qcut(te[f"cate_{arm}"].rank(method="first"), 10, labels=False) + 1
        for d, g in te.groupby("_dec"):
            g2 = g[g.arm.isin(["control", arm])]
            yy = outcome(g2)
            obs = yy[(g2.arm == arm).values].mean() - yy[(g2.arm == "control").values].mean()
            deciles.append({"arm": arm, "decile": int(d), "predicted_cate": g[f"cate_{arm}"].mean(), "observed_uplift": obs,
                            "share_35_plus": (~g.age_group.isin(["18-24", "25-34"])).mean()})
    pd.DataFrame(deciles).to_csv(C.TABLES / "cate_deciles.csv", index=False)

    summary["policies"] = pol.to_dict(orient="records")
    with open(C.REPORTS / "hte_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Figures
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, arm in zip(axes, C.TREATMENTS):
        for model, ls in [("DR-learner", "-"), ("T-learner", "-")]:
            q = qini[(qini.arm == arm) & (qini.model == model)]
            ax.plot(q.fraction_targeted * 100, q.cumulative_gain / 1e3, lw=2.2 if model == "DR-learner" else 1.4,
                    color=C.ARM_COLORS[arm] if model == "DR-learner" else C.INK["secondary"], label=model)
        ax.plot(q.fraction_targeted * 100, q.random / 1e3, color=C.INK["muted"], lw=1.2, label="Random targeting")
        ax.set(title=f"Qini curve · {C.ARM_LABELS[arm]}", xlabel="% of users targeted (highest predicted effect first)",
               ylabel="Cumulative net value gain ($K)")
        ax.legend(fontsize=8)
    save(fig, "07_qini_curves")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    p = pol.sort_values("value_per_user_usd")
    colors = [C.SEQ_BLUE[5] if "Model" in n else C.SEQ_BLUE[2] for n in p.policy]
    ax.barh(p.policy, p.uplift_vs_control_usd, color=colors, height=0.55,
            xerr=[p.uplift_vs_control_usd - p.uplift_ci_low, p.uplift_ci_high - p.uplift_vs_control_usd],
            ecolor=C.INK["secondary"])
    ax.axvline(0, color=C.INK["axis"], lw=1)
    for yv, (v, s) in enumerate(zip(p.uplift_vs_control_usd, p.annual_value_at_scale_musd)):
        ax.text(max(v, 0) + 0.15, yv + 0.22, f"${s:,.0f}M / yr at {C.MAU_POPULATION/1e6:.0f}M MAU", color=C.INK["secondary"], fontsize=8)
    ax.set(xlabel="Net annual value per user vs control (USD, IPW on held-out users, paired-bootstrap 95% CI)",
           title="Model-targeted rollout leads, but overlaps the blanket friend-boost launch")
    save(fig, "08_policy_values")
    print(json.dumps(summary, indent=1, default=float))
    print(rules)


if __name__ == "__main__":
    main()
