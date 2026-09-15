import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.alerts import simulate_alerts
from src.config import EXPECTED_SHA256, ROOT
from src.data import chronological_split, load_clean, write_etl
from src.evaluation import evaluate
from src.predict import PRODUCTION_INPUTS, load_artifacts, score_transactions


@pytest.fixture(scope="module")
def data():
    return load_clean()[0]


@pytest.fixture(scope="module")
def bundles():
    return load_artifacts()


def test_final_dataset_identity(data):
    _, audit = load_clean()
    assert audit["source_sha256"] == EXPECTED_SHA256
    assert data.shape == (5000, 14)
    assert data.Fraudulent.sum() == 482
    assert audit["history_caution"]["account_age_decreases"] == 606


def test_temporal_partitions_are_disjoint(data):
    p = chronological_split(data)
    assert [len(p[s]) for s in p] == [3000, 1000, 1000]
    assert p["train"].Transaction_Date.max() < p["validation"].Transaction_Date.min()
    assert p["validation"].Transaction_Date.max() < p["test"].Transaction_Date.min()
    sets = [set(frame.Transaction_ID) for frame in p.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_equal_timestamps_stay_together(data):
    copy = data.copy()
    copy.loc[2999, "Transaction_Date"] = copy.loc[3000, "Transaction_Date"]
    parts = chronological_split(copy)
    assert parts["train"].Transaction_Date.max() < parts["validation"].Transaction_Date.min()
    assert len(parts["train"]) == 2999


def test_sqlite_round_trip(data, tmp_path):
    db = write_etl(data, tmp_path)
    with sqlite3.connect(db) as conn:
        rows, frauds, amount = conn.execute("SELECT COUNT(*), SUM(Fraudulent), SUM(Transaction_Amount) FROM transactions").fetchone()
    assert rows == 5000 and frauds == 482
    assert amount == pytest.approx(395903.32)


def test_preprocessor_statistics_are_training_only(data, bundles):
    bundle, _ = bundles
    pipeline = bundle["pipeline"]
    train = chronological_split(data)["train"]
    assert set(pipeline["features"].feature_names_in_) == set(PRODUCTION_INPUTS)
    features = pipeline["features"].transform(train)
    numeric = pipeline["preprocess"].named_transformers_["numeric"]
    cols = pipeline["preprocess"].transformers_[0][2]
    expected = features[cols].median().to_numpy()
    np.testing.assert_allclose(numeric["impute"].statistics_, expected)
    assert numeric["scale"].n_samples_seen_ == len(train)
    assert not pipeline["features"].include_history
    assert not pipeline["features"].include_keyword


def test_saved_model_matches_evaluation_outputs(bundles):
    expected = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    actual = score_transactions(expected, *bundles)
    np.testing.assert_allclose(actual.Fraud_Score, expected.Fraud_Score, atol=1e-12)
    np.testing.assert_allclose(actual.Anomaly_Score, expected.Anomaly_Score, atol=1e-12)
    np.testing.assert_array_equal(actual.Predicted_Fraud, expected.Predicted_Fraud)
    np.testing.assert_array_equal(actual.Anomaly_Flag, expected.Anomaly_Flag)


def test_single_and_batch_predictions_match(data, bundles):
    batch = score_transactions(data.tail(5), *bundles)
    singles = [score_transactions(data.iloc[[i]], *bundles).Fraud_Score.iloc[0] for i in range(len(data)-5, len(data))]
    np.testing.assert_allclose(batch.Fraud_Score, singles, atol=1e-12)


def test_ignored_target_ids_and_unverified_fields_cannot_change_predictions(data, bundles):
    sample = data.tail(20).copy()
    before = score_transactions(sample, *bundles)
    sample["Fraudulent"] = 1 - sample.Fraudulent
    for col in ["Transaction_ID", "Customer_ID", "Suspicious_Keyword"]:
        sample[col] = "changed"
    for col in ["Average_Spend", "Previous_Transactions", "Account_Age_Days"]:
        sample[col] = 999999
    after = score_transactions(sample, *bundles)
    np.testing.assert_array_equal(before.Fraud_Score, after.Fraud_Score)
    np.testing.assert_array_equal(before.Anomaly_Score, after.Anomaly_Score)


def test_missing_values_and_unseen_categories(data, bundles):
    sample = data.tail(2).copy()
    sample["Transaction_Amount"] = np.nan
    sample["Is_International"] = np.nan
    sample["Device_Type"] = np.nan
    sample["Location"] = "Previously unseen location"
    scores = score_transactions(sample, *bundles)
    assert np.isfinite(scores.Fraud_Score).all()
    assert np.isfinite(scores.Anomaly_Score).all()


@pytest.mark.parametrize("column,value", [("Transaction_Amount", -1), ("Transaction_Amount", "bad"),
                                        ("Is_International", 2), ("Transaction_Date", "not-a-date")])
def test_invalid_prediction_input_rejected(data, bundles, column, value):
    row = data.tail(1).copy()
    row[column] = value
    with pytest.raises(ValueError):
        score_transactions(row, *bundles)


def test_missing_columns_and_empty_upload_rejected(data, bundles):
    with pytest.raises(ValueError, match="Missing input columns"):
        score_transactions(data.drop(columns="Device_Type"), *bundles)
    with pytest.raises(ValueError, match="at least one"):
        score_transactions(data.head(0), *bundles)


def test_reported_metrics_recompute_exactly(bundles):
    scores = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    expected = json.loads((ROOT / "outputs/metrics.json").read_text())["test"]
    actual = evaluate(scores.Fraudulent, scores.Fraud_Score, bundles[0]["threshold"])
    for metric in ["precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc", "tp", "fp", "fn", "tn"]:
        assert actual[metric] == pytest.approx(expected[metric])


def test_alerts_only_include_threshold_exceedances(tmp_path):
    scores = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    records = simulate_alerts(scores, tmp_path / "alerts.jsonl", limit=25)
    expected_ids = set(scores.loc[scores.Predicted_Fraud.eq(1), "Transaction_ID"])
    assert len(records) == 25
    assert all(r["transaction_id"] in expected_ids and r["status"] == "SIMULATED_ONLY" for r in records)
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 25
