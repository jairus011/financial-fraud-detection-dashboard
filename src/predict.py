"""Batch and single-record prediction use the exact saved preprocessing pipelines."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CATEGORICAL, ROOT
from .data import parse_dates

PRODUCTION_INPUTS = ["Transaction_Date", "Transaction_Amount", "Is_International", *CATEGORICAL]


def load_artifacts(model_dir=None):
    path = Path(model_dir) if model_dir else ROOT / "models"
    # joblib/pickle files must be trusted. The UI never accepts uploaded model files.
    return joblib.load(path / "fraud_model.joblib"), joblib.load(path / "isolation_forest.joblib")


def validate_inputs(df):
    missing = sorted(set(PRODUCTION_INPUTS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if len(df) == 0:
        raise ValueError("Supply at least one transaction.")
    if parse_dates(df.Transaction_Date).isna().any():
        raise ValueError("Transaction_Date must use DD-MM-YYYY HH:MM or ISO format.")
    for c in ["Transaction_Amount", "Is_International"]:
        s = df[c]
        n = pd.to_numeric(s, errors="coerce")
        if (s.notna() & (~np.isfinite(n) | n.lt(0))).any():
            raise ValueError(f"{c} must be a nonnegative number, or blank for imputation.")
        if c == "Is_International" and (n.notna() & ~n.isin([0, 1])).any():
            raise ValueError("Is_International must be 0, 1, or blank.")


def score_transactions(df, model_bundle=None, anomaly_bundle=None, model_dir=None):
    validate_inputs(df)
    if model_bundle is None or anomaly_bundle is None:
        model_bundle, anomaly_bundle = load_artifacts(model_dir)
    result = df.copy().reset_index(drop=True)
    scores = model_bundle["pipeline"].predict_proba(df)[:, 1]
    anomaly = -anomaly_bundle["pipeline"].score_samples(df)
    result["Fraud_Score"] = scores
    result["Predicted_Fraud"] = (scores >= model_bundle["threshold"]).astype(int)
    result["Anomaly_Score"] = anomaly
    result["Anomaly_Flag"] = (anomaly >= anomaly_bundle["threshold"]).astype(int)
    result["Review_Status"] = np.select(
        [result.Predicted_Fraud.eq(1), result.Anomaly_Flag.eq(1)],
        ["Model review", "Anomaly review"], default="Lower score")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/batch_predictions.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    scores = score_transactions(pd.read_csv(args.input), model_dir=args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"Scored {len(scores)} transactions -> {args.output}")
