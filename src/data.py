"""Stateless cleaning, an auditable chronological split, and SQLite ETL."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL, ID_COLUMNS, NUMERIC, RAW_COLUMNS, RAW_DATA, ROOT, TARGET


def parse_dates(values):
    """Accept the supplied day-first format or unambiguous ISO dates."""
    s = pd.Series(values, index=values.index)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    first = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    return first.fillna(iso)


def load_clean(path=RAW_DATA):
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if set(df.columns) != set(RAW_COLUMNS):
        raise ValueError("Unexpected dataset columns; review the schema before training.")
    audit = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "rows_raw": len(df), "columns": df.columns.tolist(),
             "missing_raw": df.isna().sum().astype(int).to_dict(),
             "exact_duplicates_removed": int(df.duplicated().sum())}
    df = df.drop_duplicates().copy()
    for c in [*ID_COLUMNS, *CATEGORICAL, "Suspicious_Keyword"]:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df["Transaction_Date"] = parse_dates(df["Transaction_Date"])
    audit["invalid_numeric_cells"] = {}
    for c in NUMERIC:
        old = df[c].notna()
        v = pd.to_numeric(df[c], errors="coerce")
        invalid = ~np.isfinite(v) | (v < 0)
        if c == "Is_International":
            invalid |= ~v.isin([0, 1])
        if c in ["Previous_Transactions", "Account_Age_Days"]:
            invalid |= v.mod(1).ne(0)
        audit["invalid_numeric_cells"][c] = int((old & invalid).sum())
        df[c] = v.mask(invalid, np.nan)
    labels = pd.to_numeric(df[TARGET], errors="coerce")
    valid = labels.isin([0, 1]) & df["Transaction_Date"].notna() & df[ID_COLUMNS].notna().all(axis=1)
    audit["invalid_key_date_or_label_rows_removed"] = int((~valid).sum())
    df = df.loc[valid].copy()
    df[TARGET] = labels.loc[valid].astype(int)
    if df["Transaction_ID"].duplicated().any():
        raise ValueError("Conflicting or repeated transaction IDs need source review.")
    # Keep extreme and zero amounts: unusual transactions may be the fraud signal.
    df = df.sort_values(["Transaction_Date", "Transaction_ID"], kind="stable").reset_index(drop=True)
    audit.update(rows_clean=len(df), fraud_count=int(df[TARGET].sum()),
                 fraud_rate=float(df[TARGET].mean()),
                 total_amount=float(df["Transaction_Amount"].sum()),
                 fraudulent_amount=float(df.loc[df[TARGET].eq(1), "Transaction_Amount"].sum()),
                 zero_amount_count=int(df.Transaction_Amount.eq(0).sum()),
                 missing_clean=df.isna().sum().astype(int).to_dict(),
                 min_date=str(df.Transaction_Date.min()), max_date=str(df.Transaction_Date.max()),
                 unique_customers=int(df.Customer_ID.nunique()))
    repeated = df.groupby("Customer_ID", sort=False)
    audit["history_caution"] = {
        "account_age_decreases": int(repeated.Account_Age_Days.diff().lt(0).sum()),
        "previous_transactions_decreases": int(repeated.Previous_Transactions.diff().lt(0).sum()),
        "interpretation": "Snapshot definitions and customer-ID consistency are unverified. History inputs must be available before the transaction; no history is reconstructed from future rows."}
    return df, audit


def chronological_split(df):
    """60/20/20 by time; equal timestamps stay in the later partition."""
    if len(df) < 100:
        raise ValueError("At least 100 transactions are required for this protocol.")
    if not df.Transaction_Date.is_monotonic_increasing:
        raise ValueError("Sort cleaned transactions by time before splitting.")
    cut1 = df.Transaction_Date.iloc[int(len(df) * .6)]
    cut2 = df.Transaction_Date.iloc[int(len(df) * .8)]
    parts = {
        "train": df.loc[df.Transaction_Date.lt(cut1)].copy(),
        "validation": df.loc[df.Transaction_Date.ge(cut1) & df.Transaction_Date.lt(cut2)].copy(),
        "test": df.loc[df.Transaction_Date.ge(cut2)].copy(),
    }
    for name, part in parts.items():
        if part[TARGET].nunique() != 2:
            raise ValueError(f"{name} must contain both classes.")
    return parts


def write_etl(df, output_root=ROOT):
    output_root = Path(output_root)
    dest = output_root / "data/processed"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "transactions_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(dest / "fraud_detection.sqlite") as conn:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS transaction_id_idx ON transactions (Transaction_ID)')
        conn.execute('CREATE INDEX IF NOT EXISTS transaction_date_idx ON transactions (Transaction_Date)')
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    if count != len(df):
        raise RuntimeError("SQLite row count does not match cleaned data.")
    return dest / "fraud_detection.sqlite"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    cleaned, audit = load_clean(args.data)
    db = write_etl(cleaned, args.output_dir)
    print(json.dumps(audit, indent=2))
    print(f"ETL complete: {len(cleaned)} transactions -> {db}")
