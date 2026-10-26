"""Sentiment model and topics on TweetEval, plus a feed-exposure scenario under assumed ranker weights."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config as C
from .plotting import plt, save, setup

LABELS = ["negative", "neutral", "positive"]
N_TOPICS = 10


def clean(text: str) -> str:
    text = re.sub(r"@user", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_tweets():
    d = {s: pd.read_parquet(C.EXTERNAL / "tweet_eval" / f"{s}.parquet") for s in ["train", "validation", "test"]}
    for s, df in d.items():
        df["clean"] = df.text.map(clean)
        df["split"] = s
    return d


def sentiment_model(d) -> tuple[dict, object]:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000, sublinear_tf=True)
    Xtr = sparse.hstack([word.fit_transform(d["train"].clean), char.fit_transform(d["train"].clean)]).tocsr()
    tr = lambda s: sparse.hstack([word.transform(d[s].clean), char.transform(d[s].clean)]).tocsr()
    Xva, Xte = tr("validation"), tr("test")
    best = None
    for c in [0.5, 1.0, 2.0, 4.0]:
        m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(Xtr, d["train"].label)
        score = recall_score(d["validation"].label, m.predict(Xva), average="macro")
        if best is None or score > best[0]:
            best = (score, c, m)
    _, c_best, model = best
    pred = model.predict(Xte)
    vader = SentimentIntensityAnalyzer()
    comp = d["test"].clean.map(lambda t: vader.polarity_scores(t)["compound"])
    vpred = np.where(comp >= 0.05, 2, np.where(comp <= -0.05, 0, 1))
    y = d["test"].label.values
    rows = []
    for name, p in [("VADER (lexicon baseline)", vpred), ("TF-IDF + LogReg (ours)", pred)]:
        rows.append({"model": name, "macro_recall": recall_score(y, p, average="macro"),
                     "macro_f1": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p),
                     "f1_negative": f1_score(y, p, labels=[0], average="macro"),
                     "f1_positive": f1_score(y, p, labels=[2], average="macro")})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.TABLES / "sentiment_model_metrics.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(y, pred, normalize="true"), index=LABELS, columns=LABELS)
    cm.to_csv(C.TABLES / "sentiment_confusion_matrix.csv")
    corpus = pd.concat(d.values(), ignore_index=True)
    Xall = sparse.hstack([word.transform(corpus.clean), char.transform(corpus.clean)]).tocsr()
    proba = model.predict_proba(Xall)
    corpus["p_negative"], corpus["p_neutral"], corpus["p_positive"] = proba.T
    corpus["pred_sentiment"] = np.array(LABELS)[proba.argmax(axis=1)]
    return {"best_C": c_best, "metrics": metrics.to_dict(orient="records")}, corpus


def topics(corpus: pd.DataFrame) -> pd.DataFrame:
    stop = list(ENGLISH_STOP_WORDS | {"amp", "just", "like", "tomorrow", "today", "day", "night", "going", "im", "dont",
                                      "got", "get", "may", "th", "st", "nd", "u", "ll", "ve", "don", "didn", "doesn"})
    vec = TfidfVectorizer(stop_words=stop, max_df=0.08, min_df=15, max_features=12_000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(corpus.clean.str.lower())
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    corpus["topic_id"] = W.argmax(axis=1)
    corpus["topic_strength"] = W.max(axis=1)
    rows = []
    for k in range(N_TOPICS):
        top = vocab[np.argsort(-nmf.components_[k])[:10]]
        g = corpus[corpus.topic_id == k]
        rows.append({"topic_id": k, "label": " / ".join(top[:3]), "top_words": ", ".join(top), "n_posts": len(g),
                     "share_negative_pred": (g.pred_sentiment == "negative").mean(),
                     "share_positive_pred": (g.pred_sentiment == "positive").mean(),
                     "mean_p_negative": g.p_negative.mean()})
    info = pd.DataFrame(rows)
    info.to_csv(C.TABLES / "topic_info.csv", index=False)
    return info


RANKERS = {"control": (0.6, 0.0), "ai_feed": (2.2, 2.2), "ai_feed_social": (1.4, 0.8)}  # (concentration, engagement weight)


def exposure_simulation(corpus: pd.DataFrame, n_users_per_arm: int = 3000, impressions: int = 150,
                        rankers: dict | None = None, save_outputs: bool = True) -> pd.DataFrame:
    """Serve held-out posts to sampled users under each arm's assumed ranker."""
    rng = np.random.default_rng(C.SEED)
    rankers = rankers or RANKERS
    users = pd.read_parquet(C.RAW / "users.parquet")
    users = users[users.arm.isin(list(rankers))]
    sample = users.groupby("arm", group_keys=False).apply(lambda g: g.sample(min(n_users_per_arm, len(g)), random_state=0))
    # Test-split posts only, so the sentiment model never saw them
    items = corpus[(corpus.topic_strength > 0) & (corpus.split == "test")].reset_index(drop=True)
    topic_items = [np.flatnonzero(items.topic_id.values == k) for k in range(N_TOPICS)]
    # Assumption: negative posts draw more reactions
    eng_score = 0.9 * items.p_negative.values + 0.3 * items.p_positive.values
    rows, topic_rows = [], []
    for _, u in sample.iterrows():
        aff = rng.dirichlet(np.full(N_TOPICS, 0.7))
        conc, w_eng = rankers[u.arm]
        p_topic = aff**conc / (aff**conc).sum()
        counts = rng.multinomial(impressions, p_topic)
        served = []
        for k, cnt in enumerate(counts):
            if cnt == 0:
                continue
            pool = topic_items[k]
            cand = rng.choice(pool, size=min(len(pool), cnt * 8), replace=False)
            w = np.exp(w_eng * eng_score[cand])
            served.append(rng.choice(cand, size=cnt, replace=False, p=w / w.sum()))
        served = np.concatenate(served)
        share = np.bincount(items.topic_id.values[served], minlength=N_TOPICS) / impressions
        nz = share[share > 0]
        engage_p = 1 / (1 + np.exp(-(-3.0 + 4.0 * aff[items.topic_id.values[served]] + 0.8 * items.p_negative.values[served])))
        rows.append({"user_id": u.user_id, "arm": u.arm, "age_group": u.age_group,
                     "topic_entropy": float(-(nz * np.log(nz)).sum() / np.log(N_TOPICS)),
                     "top_topic_share": float(share.max()), "distinct_topics": int((share > 0).sum()),
                     "negative_share": float((items.pred_sentiment.values[served] == "negative").mean()),
                     "mean_p_negative": float(items.p_negative.values[served].mean()),
                     "expected_engagements": float(engage_p.sum())})
        for k in range(N_TOPICS):
            topic_rows.append((u.arm, k, share[k]))
    ux = pd.DataFrame(rows)
    if not save_outputs:
        return ux
    ux.to_parquet(C.PROCESSED / "content_exposure_users.parquet", index=False)
    tx = pd.DataFrame(topic_rows, columns=["arm", "topic_id", "share"]).groupby(["arm", "topic_id"]).share.mean().reset_index()
    tx.to_csv(C.TABLES / "topic_exposure_by_arm.csv", index=False)
    summ = []
    ctrl = ux[ux.arm == "control"]
    for arm in [a for a in C.ARMS if a in set(ux.arm)]:
        g = ux[ux.arm == arm]
        r = {"arm": arm}
        for m in ["topic_entropy", "top_topic_share", "distinct_topics", "negative_share", "expected_engagements"]:
            r[m] = g[m].mean()
            if arm != "control":
                r[f"{m}_rel_change"] = g[m].mean() / ctrl[m].mean() - 1
                r[f"{m}_p"] = ttest_ind(g[m], ctrl[m], equal_var=False).pvalue
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(C.TABLES / "content_exposure_summary.csv", index=False)
    return summ


def exposure_sensitivity(corpus: pd.DataFrame) -> pd.DataFrame:
    """Rerun the AI-feed scenario over a grid of ranker weights."""
    rows = []
    for conc in (0.6, 1.4, 2.2):
        for w in (0.0, 0.8, 1.5, 2.2, 3.0):
            ux = exposure_simulation(corpus, n_users_per_arm=600, rankers={"ai_feed": (conc, w)}, save_outputs=False)
            rows.append({"concentration": conc, "engagement_weight": w, "negative_share": ux.negative_share.mean(),
                         "topic_entropy": ux.topic_entropy.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "exposure_sensitivity.csv", index=False)
    return out


def figures(metrics, info, tx, summ):
    setup()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    m = pd.DataFrame(metrics)
    x = np.arange(3)
    for i, (_, r) in enumerate(m.iterrows()):
        vals = [r.macro_recall, r.macro_f1, r.accuracy]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=[C.INK["muted"], C.SEQ_BLUE[5]][i], label=r.model,
               edgecolor=C.INK["surface"], lw=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 0.5) * 0.36, v + 0.01, f"{v:.3f}", ha="center", color=C.INK["secondary"], fontsize=8)
    ax.set_xticks(x, ["Macro recall (official)", "Macro F1", "Accuracy"])
    ax.set(ylim=(0, 0.85), title="Sentiment model on the TweetEval test set (12,284 real tweets)")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "13_sentiment_benchmark")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = info.set_index("topic_id").label
    x = np.arange(N_TOPICS)
    for i, arm in enumerate(C.ARMS):
        a = tx[tx.arm == arm].set_index("topic_id").share.reindex(x)
        ax.bar(x + (i - 1) * 0.27, a * 100, width=0.25, color=C.ARM_COLORS[arm], label=C.ARM_LABELS[arm],
               edgecolor=C.INK["surface"], lw=1.5)
    ax.set_xticks(x, [labels[k] for k in x], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="Share of impressions (%)", title="Topic exposure by arm (NMF topics on real posts)")
    ax.legend(fontsize=8)
    save(fig, "14_topic_exposure_by_arm")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, t) in zip(axes, [("topic_entropy", "Feed diversity (normalised entropy)"),
                                  ("negative_share", "Share of negative posts served"),
                                  ("expected_engagements", "Expected engagements / 150 impressions")]):
        vals = summ.set_index("arm")[m].reindex(C.ARMS)
        ax.bar(range(3), vals, color=[C.ARM_COLORS[a] for a in C.ARMS], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f"{v:.2f}" if m != "negative_share" else f"{v:.1%}", ha="center", color=C.INK["secondary"])
        ax.set_xticks(range(3), ["Control", "AI feed", "AI + friends"])
        ax.set_title(t, fontsize=10)
    fig.suptitle("Scenario: under ASSUMED ranker weights, engagement-optimised ranking narrows feeds and raises negative-post share",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "15_responsible_ai_guardrails")


def main():
    d = load_tweets()
    sent, corpus = sentiment_model(d)
    info = topics(corpus)
    summ = exposure_simulation(corpus)
    sens = exposure_sensitivity(corpus)
    tx = pd.read_csv(C.TABLES / "topic_exposure_by_arm.csv")
    figures(sent["metrics"], info, tx, summ)
    corpus[["split", "label", "pred_sentiment", "p_negative", "topic_id"]].to_parquet(C.PROCESSED / "tweet_corpus_scored.parquet", index=False)
    out = {"sentiment": sent, "topics": info[["topic_id", "label", "n_posts", "share_negative_pred"]].to_dict(orient="records"),
           "exposure": summ.to_dict(orient="records"), "exposure_sensitivity": sens.to_dict(orient="records"),
           "exposure_note": "scenario under assumed ranker weights on held-out posts; not learned from real engagement"}
    with open(C.REPORTS / "nlp_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
