"""Heterogeneous treatment effects and rollout policies.

The outcome is an assumed utility score (365 x late-window revenue/day - churned x LTV), not a dollar forecast.
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
from .oracle import outcome
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


def gain_curve(tau_hat, y, t, n_points: int = 50) -> pd.DataFrame:
    """Cumulative uplift gain by share targeted, highest predicted effect first."""
    order = np.argsort(-tau_hat)
    y, t = y[order], t[order]
    n = len(y)
    rows = [{"fraction_targeted": 0.0, "cumulative_gain": 0.0}]
    for frac in np.linspace(0, 1, n_points + 1)[1:]:
        k = int(frac * n)
        yt, yc = y[:k][t[:k] == 1], y[:k][t[:k] == 0]
        uplift = (yt.mean() - yc.mean()) * k if len(yt) and len(yc) else 0.0
        rows.append({"fraction_targeted": frac, "cumulative_gain": uplift})
    out = pd.DataFrame(rows)
    out["random"] = out.fraction_targeted * out.cumulative_gain.iloc[-1]
    return out


def ht_policy_value(policy: np.ndarray, arm: np.ndarray, y: np.ndarray, p_design: float, boot_idx) -> tuple[float, np.ndarray]:
    """Horvitz-Thompson policy value with the known design probability. Shared boot_idx pairs the draws."""
    vals = (policy == arm) * y / p_design
    return float(vals.mean()), np.array([vals[i].mean() for i in boot_idx])


def oracle_metrics(pred: np.ndarray, oracle: np.ndarray, mc_se: np.ndarray) -> dict:
    mse = np.mean((pred - oracle) ** 2)
    noise = np.mean(mc_se ** 2)
    dec = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"d": dec, "p": pred, "o": oracle}).groupby("d").mean()
    slope = float(np.polyfit(cal.p, cal.o, 1)[0])
    return {"spearman": float(spearmanr(pred, oracle).correlation), "rmse": float(np.sqrt(mse)),
            "rmse_noise_corrected": float(np.sqrt(max(mse - noise, 0))), "bias": float(np.mean(pred - oracle)),
            "decile_calibration_slope": slope}


def main():
    df = pd.read_parquet(C.PROCESSED / "user_experiment.parquet")
    oracle = pd.read_parquet(C.PROCESSED / "oracle_outcomes.parquet")
    df = df.merge(oracle, on="user_id")
    X = feature_matrix(df)
    y = outcome(df)
    p_design = 1 / len(C.ARMS)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.35, random_state=C.SEED, stratify=df.arm)
    te = df.iloc[idx_te].reset_index(drop=True)
    X_te = X.iloc[idx_te].reset_index(drop=True)
    X_trall = X.iloc[idx_tr].reset_index(drop=True)
    tr_all = df.iloc[idx_tr].reset_index(drop=True)
    y_te = y[idx_te]

    cate, cate_tr, gains, summary = {}, {}, [], {}
    for arm in C.TREATMENTS:
        mtr = df.arm.iloc[idx_tr].isin(["control", arm]).values
        tr_idx = idx_tr[mtr]
        Xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
        ttr = (df.arm.iloc[tr_idx] == arm).astype(int).values
        e = ttr.mean()
        tau_t = t_learner(Xtr, ytr, ttr, X_te)
        tau_dr, final_model = dr_learner(Xtr, ytr, ttr, X_te, e)
        cate[arm], cate_tr[arm] = tau_dr, final_model.predict(X_trall)
        te[f"cate_{arm}"], te[f"cate_t_{arm}"] = tau_dr, tau_t
        mte = te.arm.isin(["control", arm]).values
        t_te = (te.arm[mte] == arm).astype(int).values
        for name, tau in [("DR-learner", tau_dr), ("T-learner", tau_t)]:
            g = gain_curve(tau[mte], y_te[mte], t_te)
            g["model"], g["arm"] = name, arm
            gains.append(g)
        orc, se = te[f"oracle_cate_{arm}"].values, te[f"oracle_cate_{arm}_mc_se"].values
        summary[arm] = {"mean_cate": float(tau_dr.mean()), "mean_oracle_cate": float(orc.mean()),
                        "share_positive": float((tau_dr > 0).mean()), "share_positive_oracle": float((orc > 0).mean()),
                        "oracle_dr": oracle_metrics(tau_dr, orc, se), "oracle_t": oracle_metrics(tau_t, orc, se),
                        "oracle_note": "oracle = mean over 16 common-seed re-simulations per user; per-user MC error reported"}
        imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp.head(15).rename("importance").to_frame().assign(arm=arm).to_csv(C.TABLES / f"hte_drivers_{arm}.csv")

    gain = pd.concat(gains)
    gain.to_csv(C.TABLES / "qini_curves.csv", index=False)
    for arm in C.TREATMENTS:
        for model in ["DR-learner", "T-learner"]:
            g = gain[(gain.arm == arm) & (gain.model == model)]
            rnd = np.trapezoid(g.random, g.fraction_targeted)
            summary[arm][f"gain_area_ratio_{model}"] = (float(np.trapezoid(g.cumulative_gain, g.fraction_targeted) / rnd)
                                                        if abs(rnd) > 1e-9 else None)

    # Readable policy: a shallow tree fit on training-user predictions, valued on test users
    def readable(frame):
        r = frame[["tenure_days", "friend_count", "pre_minutes_per_day", "pre_friend_interactions_per_day", "is_new_user"]].copy()
        r["age_35_plus"] = (~frame.age_group.isin(["18-24", "25-34"])).astype(int)
        r["is_creator"] = (frame.account_type == "creator").astype(int)
        return r
    adv_tr = cate_tr["ai_feed"] - cate_tr["ai_feed_social"]
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=800, random_state=0).fit(readable(tr_all), adv_tr)
    rules = export_text(tree, feature_names=list(readable(tr_all).columns), decimals=2)
    (C.REPORTS / "policy_tree_rules.txt").write_text(
        "Surrogate policy tree trained on TRAINING-user DR-learner predictions of the annual utility advantage\n"
        "of AI feed over AI+friend boost (positive = AI feed). Its own value is evaluated on held-out users.\n\n" + rules)
    tree_policy = np.where(tree.predict(readable(te)) > 0, "ai_feed", "ai_feed_social")

    pred = np.column_stack([np.zeros(len(te)), cate["ai_feed"], cate["ai_feed_social"]])
    best = np.array(C.ARMS)[pred.argmax(axis=1)]
    under35 = te.age_group.isin(["18-24", "25-34"]).values
    policies = {
        "Ship nothing (control)": np.full(len(te), "control"),
        "Ship AI feed to all": np.full(len(te), "ai_feed"),
        "Ship AI + friend boost to all": np.full(len(te), "ai_feed_social"),
        "Rule: AI feed <35, friend boost 35+": np.where(under35, "ai_feed", "ai_feed_social"),
        "Policy tree (trained on train)": tree_policy,
        "Model-targeted (DR-learner)": best,
    }
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(te), len(te)) for _ in range(1000)]
    vals = {n: ht_policy_value(p, te.arm.values, y_te, p_design, boot_idx) for n, p in policies.items()}
    oracle_y = te[[f"y_{a}" for a in C.ARMS]].values
    oracle_best = oracle_y.max(axis=1).mean()
    rows = []
    for n, p in policies.items():
        v, b = vals[n]
        d_ctrl = b - vals["Ship nothing (control)"][1]
        d_fb = b - vals["Ship AI + friend boost to all"][1]
        chosen = oracle_y[np.arange(len(te)), [C.ARMS.index(a) for a in p]]
        rows.append({"policy": n, "value_per_user_usd": v,
                     "uplift_vs_control_usd": v - vals["Ship nothing (control)"][0],
                     "uplift_ci_low": float(np.quantile(d_ctrl, 0.025)), "uplift_ci_high": float(np.quantile(d_ctrl, 0.975)),
                     "diff_vs_friend_boost_usd": v - vals["Ship AI + friend boost to all"][0],
                     "diff_vs_fb_ci_low": float(np.quantile(d_fb, 0.025)), "diff_vs_fb_ci_high": float(np.quantile(d_fb, 0.975)),
                     "share_draws_beating_fb": float((d_fb > 0).mean()),
                     "oracle_value_per_user": float(chosen.mean()), "oracle_regret_per_user": float(oracle_best - chosen.mean()),
                     "share_ai_feed": (p == "ai_feed").mean(), "share_ai_feed_social": (p == "ai_feed_social").mean(),
                     "share_control": (p == "control").mean()})
    pol = pd.DataFrame(rows)
    pol["annual_value_at_scale_musd"] = pol.uplift_vs_control_usd * C.MAU_POPULATION / 1e6
    pol.to_csv(C.TABLES / "policy_values.csv", index=False)

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
                            "oracle_cate": g[f"oracle_cate_{arm}"].mean(),
                            "share_35_plus": (~g.age_group.isin(["18-24", "25-34"])).mean()})
    pd.DataFrame(deciles).to_csv(C.TABLES / "cate_deciles.csv", index=False)

    summary["policies"] = pol.to_dict(orient="records")
    summary["oracle_best_value_per_user"] = float(oracle_best)
    with open(C.REPORTS / "hte_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, arm in zip(axes, C.TREATMENTS):
        for model in ["DR-learner", "T-learner"]:
            g = gain[(gain.arm == arm) & (gain.model == model)]
            ax.plot(g.fraction_targeted * 100, g.cumulative_gain / 1e3, lw=2.2 if model == "DR-learner" else 1.4,
                    color=C.ARM_COLORS[arm] if model == "DR-learner" else C.INK["secondary"], label=model)
        ax.plot(g.fraction_targeted * 100, g.random / 1e3, color=C.INK["muted"], lw=1.2, label="Random targeting")
        ax.set(title=f"Cumulative uplift gain · {C.ARM_LABELS[arm]}", xlabel="% of users targeted (highest predicted effect first)",
               ylabel="Cumulative utility gain (thousands)")
        ax.legend(fontsize=8)
    save(fig, "07_qini_curves")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    p = pol.sort_values("value_per_user_usd")
    colors = [C.SEQ_BLUE[5] if "Model" in n else C.SEQ_BLUE[2] for n in p.policy]
    ax.barh(p.policy, p.uplift_vs_control_usd, color=colors, height=0.55,
            xerr=[p.uplift_vs_control_usd - p.uplift_ci_low, p.uplift_ci_high - p.uplift_vs_control_usd],
            ecolor=C.INK["secondary"])
    ax.axvline(0, color=C.INK["axis"], lw=1)
    m = pol.set_index("policy").loc["Model-targeted (DR-learner)"]
    ax.set(xlabel="Utility score per user vs control (Horvitz-Thompson IPW on held-out users, paired 95% CI)",
           title=f"Policy values · model minus blanket friend boost: {m.diff_vs_friend_boost_usd:+.2f} "
                 f"[{m.diff_vs_fb_ci_low:+.2f}, {m.diff_vs_fb_ci_high:+.2f}]")
    save(fig, "08_policy_values")
    print(json.dumps({k: v for k, v in summary.items() if k != "policies"}, indent=1, default=float))
    print(pol[["policy", "uplift_vs_control_usd", "diff_vs_friend_boost_usd", "diff_vs_fb_ci_low", "diff_vs_fb_ci_high",
               "share_draws_beating_fb", "oracle_regret_per_user"]].round(3).to_string(index=False))
    print(rules)


if __name__ == "__main__":
    main()
