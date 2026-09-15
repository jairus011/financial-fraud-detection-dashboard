import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def evaluate(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y, score)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"n": len(y), "fraud_count": int(y.sum()), "prevalence": float(y.mean()),
            "threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)) if both else None,
            "average_precision": float(average_precision_score(y, score)) if both else None,
            "pr_auc": float(auc(recall, precision)) if both else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "alert_count": int(pred.sum()), "alert_rate": float(pred.mean())}


def choose_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    table = pd.DataFrame({"threshold": t, "precision": p[:-1], "recall": r[:-1], "f1": f1})
    best = table.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, False], kind="stable").iloc[0]
    return float(best.threshold), table


def grouped_bootstrap(y, scores, groups, threshold, repeats=400, seed=42):
    """Resample customer groups within test, preserving within-customer rows."""
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(repeats):
        idx = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) == 2:
            records.append(evaluate(y[idx], scores[idx], threshold))
    keys = ["precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc"]
    return {"method": "Customer-group percentile bootstrap on fixed test predictions; conditional on the fitted model, not training/selection uncertainty.",
            "repeats": len(records), "seed": seed,
            "intervals_95": {k: [float(x) for x in np.percentile([r[k] for r in records], [2.5, 97.5])] for k in keys}}
