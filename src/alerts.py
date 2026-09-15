"""Offline fraud-alert simulation. No email, credentials, Kafka or external calls."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ROOT


def simulate_alerts(scored, output=None, limit=25):
    candidates = scored.loc[scored.Predicted_Fraud.eq(1)].sort_values("Fraud_Score", ascending=False)
    if limit is not None:
        candidates = candidates.head(limit)
    records = []
    for idx, row in candidates.iterrows():
        records.append({"status": "SIMULATED_ONLY", "action": "human_review_requested",
                        "transaction_id": str(row.get("Transaction_ID", f"ROW-{idx}")),
                        "transaction_date": str(row.Transaction_Date),
                        "amount": None if pd.isna(row.Transaction_Amount) else float(row.Transaction_Amount),
                        "fraud_score": float(row.Fraud_Score),
                        "anomaly_flag": int(row.Anomaly_Flag),
                        "message": "Score exceeded the frozen validation threshold. This is not a confirmed fraud finding."})
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/scored_transactions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fraud_alerts.jsonl")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    records = simulate_alerts(pd.read_csv(args.input), args.output, args.limit)
    print(f"Created {len(records)} simulated alerts -> {args.output}; no messages sent.")
