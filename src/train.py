"""Rebuild the full project from the supplied CSV: python -m src.train."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .alerts import simulate_alerts
from .config import EXPECTED_SHA256, RAW_DATA, ROOT, SEED, TARGET
from .data import chronological_split, load_clean, write_etl
from .evaluation import choose_threshold, evaluate, grouped_bootstrap
from .features import build_pipeline
from .predict import PRODUCTION_INPUTS, score_transactions


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def candidates():
    result = {}
    for weighted in [False, True]:
        suffix = "balanced" if weighted else "unweighted"
        weight = "balanced" if weighted else None
        estimators = [
            ("Logistic Regression", LogisticRegression(C=1.0, max_iter=2000, class_weight=weight, random_state=SEED), True),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, class_weight=weight, random_state=SEED), False),
            ("Random Forest", RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=8, max_features="sqrt",
                                                    class_weight="balanced_subsample" if weighted else None, n_jobs=2, random_state=SEED), False),
            ("Histogram Gradient Boosting", HistGradientBoostingClassifier(max_iter=160, learning_rate=.06, max_leaf_nodes=15,
                    min_samples_leaf=25, l2_regularization=2.0, class_weight=weight, early_stopping=False, random_state=SEED), False),
        ]
        for name, estimator, scale in estimators:
            result[f"{name} ({suffix})"] = build_pipeline(estimator, scale=scale, include_history=False)
    return result


def run(data_path=RAW_DATA, output_root=ROOT, make_figures=True):
    output_root = Path(output_root)
    for sub in ["outputs", "models", "data/processed", "outputs/figures"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    out, models = output_root / "outputs", output_root / "models"
    df, audit = load_clean(data_path)
    # A different dataset requires a deliberate new protocol, never mixed results.
    if audit["source_sha256"] != EXPECTED_SHA256:
        raise ValueError("Dataset hash differs from the final supplied CSV. Review the protocol before changing EXPECTED_SHA256.")
    write_etl(df, output_root)
    save_json(out / "data_audit.json", audit)
    parts = chronological_split(df)
    train, val, test = (parts[s] for s in ["train", "validation", "test"])
    split_info = {}
    for name, frame in parts.items():
        split_info[name] = {"rows": len(frame), "frauds": int(frame[TARGET].sum()),
                            "fraud_rate": float(frame[TARGET].mean()), "start": str(frame.Transaction_Date.min()),
                            "end": str(frame.Transaction_Date.max()), "unique_customers": int(frame.Customer_ID.nunique())}
    split_info["customer_overlap"] = {f"{a}_{b}": len(set(parts[a].Customer_ID) & set(parts[b].Customer_ID))
                                        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    split_info["policy"] = "Chronological 60/20/20; identical timestamps not split; returning customers permitted, IDs excluded."
    save_json(out / "split_summary.json", split_info)
    split_ids = pd.concat([p[["Transaction_ID", "Customer_ID", "Transaction_Date", TARGET]].assign(Split=s)
                           for s, p in parts.items()], ignore_index=True)
    split_ids.to_csv(out / "split_assignments.csv", index=False)
    print(f"Audited {len(df)} rows. Split: {len(train)}/{len(val)}/{len(test)}.", flush=True)
    fitted, thresholds, records, validation_scores = {}, {}, [], {}
    with threadpool_limits(limits=2):
        for name, pipeline in candidates().items():
            pipeline.fit(train[PRODUCTION_INPUTS], train[TARGET])
            scores = pipeline.predict_proba(val[PRODUCTION_INPUTS])[:, 1]
            threshold, _ = choose_threshold(val[TARGET], scores)
            fitted[name], thresholds[name], validation_scores[name] = pipeline, threshold, scores
            metrics = evaluate(val[TARGET], scores, threshold)
            records.append({"model": name, "split": "validation", "kind": "supervised", **metrics})
            print(f"Validation {name}: AP={metrics['average_precision']:.4f}, F1={metrics['f1']:.4f}", flush=True)
        # Freeze model selection using validation only; the test set is evaluated below.
        ranking = sorted(records, key=lambda r: (-r["average_precision"], r["model"]))
        selected = ranking[0]["model"]
        final_pipeline, threshold = fitted[selected], thresholds[selected]
        selection = {"model": selected, "criterion": "Highest validation average precision; name breaks exact AP ties",
                     "threshold": threshold, "threshold_policy": "Validation maximum F1; ties favour recall",
                     "validation": ranking[0], "fit_partition": "train only",
                     "excluded_features": ["Transaction_ID", "Customer_ID", TARGET, "Suspicious_Keyword",
                                           "Previous_Transactions", "Average_Spend", "Account_Age_Days"],
                     "input_columns": PRODUCTION_INPUTS, "dataset_sha256": audit["source_sha256"], "seed": SEED}
        save_json(models / "selection.json", selection)
        print(f"FROZEN selection: {selected}, threshold={threshold:.8f}", flush=True)
        _, threshold_curve = choose_threshold(val[TARGET], validation_scores[selected])
        threshold_curve.to_csv(out / "validation_thresholds.csv", index=False)

        isolation = build_pipeline(IsolationForest(n_estimators=250, max_samples=256, contamination="auto",
                                                  random_state=SEED, n_jobs=2), scale=True, include_history=False)
        isolation.fit(train[PRODUCTION_INPUTS])  # No labels passed to fit.
        train_anomaly = -isolation.score_samples(train[PRODUCTION_INPUTS])
        anomaly_threshold = float(np.quantile(train_anomaly, .90))
        anomaly_bundle = {"pipeline": isolation, "threshold": anomaly_threshold,
                          "review_budget": .10, "threshold_policy": "90th percentile of unlabelled training anomaly scores",
                          "score_direction": "Higher is more anomalous", "dataset_sha256": audit["source_sha256"]}
        model_bundle = {"pipeline": final_pipeline, **selection}
        joblib.dump(model_bundle, models / "fraud_model.joblib", compress=3)
        joblib.dump(anomaly_bundle, models / "isolation_forest.joblib", compress=3)

        dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train[TARGET])
        curves, test_predictions = [], test[["Transaction_ID", "Customer_ID", "Transaction_Date", TARGET]].copy()
        for name, pipeline in fitted.items():
            scores = pipeline.predict_proba(test[PRODUCTION_INPUTS])[:, 1]
            records.append({"model": name, "split": "test", "kind": "supervised", **evaluate(test[TARGET], scores, thresholds[name])})
            test_predictions[name] = scores
            fpr, tpr, _ = roc_curve(test[TARGET], scores)
            p, r, _ = precision_recall_curve(test[TARGET], scores)
            curves.extend([{"model": name, "curve": "ROC", "x": float(x), "y": float(y)} for x, y in zip(fpr, tpr)])
            curves.extend([{"model": name, "curve": "PR", "x": float(x), "y": float(y)} for x, y in zip(r, p)])
        for split, frame in [("validation", val), ("test", test)]:
            ds = dummy.predict_proba(np.zeros((len(frame), 1)))[:, 1]
            records.append({"model": "Dummy prior / no fraud alerts", "split": split, "kind": "baseline", **evaluate(frame[TARGET], ds, .5)})
            scores = -isolation.score_samples(frame[PRODUCTION_INPUTS])
            records.append({"model": "Isolation Forest", "split": split, "kind": "unsupervised", **evaluate(frame[TARGET], scores, anomaly_threshold)})
            if split == "test":
                test_predictions["Isolation Forest"] = scores
                fpr, tpr, _ = roc_curve(frame[TARGET], scores)
                p, r, _ = precision_recall_curve(frame[TARGET], scores)
                curves.extend([{"model": "Isolation Forest", "curve": "ROC", "x": float(x), "y": float(y)} for x, y in zip(fpr, tpr)])
                curves.extend([{"model": "Isolation Forest", "curve": "PR", "x": float(x), "y": float(y)} for x, y in zip(r, p)])
        comparison = pd.DataFrame(records)
        comparison["selected"] = comparison.model.eq(selected)
        comparison.to_csv(out / "model_comparison.csv", index=False)
        pd.DataFrame(curves).to_csv(out / "performance_curves.csv", index=False)
        test_predictions.to_csv(out / "test_model_scores.csv", index=False)
        final_metrics = evaluate(test[TARGET], test_predictions[selected], threshold)
        metrics = {"final_model": selected, "test": final_metrics, "validation": evaluate(val[TARGET], validation_scores[selected], threshold),
                   "isolation_forest_test": next(r for r in records if r["model"] == "Isolation Forest" and r["split"] == "test"),
                   "test_confidence_intervals": grouped_bootstrap(test[TARGET], test_predictions[selected], test.Customer_ID, threshold)}
        slices = []
        for label, mask in [("seen in training", test.Customer_ID.isin(train.Customer_ID)),
                            ("unseen in training", ~test.Customer_ID.isin(train.Customer_ID))]:
            if mask.any():
                slices.append({"customer_slice": label, **evaluate(test.loc[mask, TARGET], test_predictions.loc[mask, selected], threshold)})
        metrics["customer_slices"] = slices
        save_json(out / "metrics.json", metrics)

        # Validation-only diagnostics: neither variant can replace the frozen artifact.
        sensitivity = [{"variant": "Final: transaction-time fields only", "validation_ap": float(average_precision_score(val[TARGET], validation_scores[selected]))}]
        for label, history, keyword in [("Add unverified history (diagnostic only)", True, False),
                                        ("Add suspicious keyword (diagnostic only)", False, True)]:
            diagnostic = build_pipeline(clone(final_pipeline.named_steps["model"]),
                          scale=isinstance(final_pipeline.named_steps["model"], LogisticRegression), include_history=history, include_keyword=keyword)
            diagnostic.fit(train, train[TARGET])
            sensitivity.append({"variant": label, "validation_ap": float(average_precision_score(val[TARGET], diagnostic.predict_proba(val)[:, 1]))})
        pd.DataFrame(sensitivity).to_csv(out / "feature_sensitivity.csv", index=False)
        importance = permutation_importance(final_pipeline, val[PRODUCTION_INPUTS], val[TARGET], scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
        pd.DataFrame({"feature": PRODUCTION_INPUTS, "validation_ap_drop_mean": importance.importances_mean,
                      "validation_ap_drop_std": importance.importances_std}).sort_values("validation_ap_drop_mean", ascending=False).to_csv(out / "feature_importance.csv", index=False)

    scored = score_transactions(df, model_bundle, anomaly_bundle)
    scored = scored.merge(split_ids[["Transaction_ID", "Split"]], on="Transaction_ID", validate="one_to_one")
    scored["Score_Context"] = scored.Split.map({"train": "in-sample", "validation": "selection/tuning", "test": "held-out evaluation"})
    scored.to_csv(out / "scored_transactions.csv", index=False)
    scored.loc[scored.Split.eq("test")].to_csv(out / "test_predictions.csv", index=False)
    simulate_alerts(scored.loc[scored.Split.eq("test")], out / "fraud_alerts.jsonl")
    df.loc[:, PRODUCTION_INPUTS].head(8).to_csv(output_root / "data/prediction_example.csv", index=False)
    runtime = {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p)
               for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "streamlit", "plotly", "matplotlib"]}}
    save_json(out / "runtime_versions.json", runtime)
    save_json(models / "artifact_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.glob("*.joblib")})
    if make_figures:
        from .plots import make_report_figures
        make_report_figures(df, scored, comparison, metrics, out)
    print(json.dumps({"selected_model": selected, "test": final_metrics}, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output_dir, make_figures=not args.no_figures)
