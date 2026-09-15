"""Rebuild in an empty output folder and compare with the committed results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import score_transactions

with tempfile.TemporaryDirectory(prefix="fraud-fresh-run-") as temp:
    dest = Path(temp)
    subprocess.run([sys.executable, "-m", "src.train", "--output-dir", str(dest)], cwd=ROOT, check=True)
    original = json.loads((ROOT / "outputs/metrics.json").read_text())
    fresh = json.loads((dest / "outputs/metrics.json").read_text())
    assert fresh["final_model"] == original["final_model"]
    for name, expected in original["test"].items():
        assert np.isclose(fresh["test"][name], expected, atol=1e-12), name
    sample = pd.read_csv(dest / "outputs/test_predictions.csv")
    scored = score_transactions(sample, model_dir=dest / "models")
    np.testing.assert_allclose(scored.Fraud_Score, sample.Fraud_Score, atol=1e-12)
    np.testing.assert_allclose(scored.Anomaly_Score, sample.Anomaly_Score, atol=1e-12)
    assert (dest / "data/processed/fraud_detection.sqlite").exists()
    assert len(list((dest / "outputs/figures").glob("*.png"))) == 7
    print("PASS: empty-folder ETL, all models, metrics, artifacts, scoring, alerts and figures reproduce.")
