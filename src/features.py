"""Row-local features. Learned imputers/encoders/scalers live inside each pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, HISTORY, INPUT_COLUMNS, NUMERIC
from .data import parse_dates


TIME_FEATURES = ["Hour_Sin", "Hour_Cos", "Weekday_Sin", "Weekday_Cos", "Is_Weekend"]
DERIVED = ["Log_Amount", "Log_Average_Spend", "Amount_to_Average", "Amount_Minus_Average",
           "Log_Account_Age", "Log_Previous_Transactions"]


class TransactionFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, include_keyword=False, include_history=True):
        self.include_keyword = include_keyword
        self.include_history = include_history

    def fit(self, X, y=None):
        self.transform(X)  # Validate schema without learning statistics.
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Pass a pandas DataFrame with named raw transaction columns.")
        needed = [c for c in INPUT_COLUMNS if self.include_history or c not in HISTORY]
        if self.include_keyword:
            needed.append("Suspicious_Keyword")
        missing = sorted(set(needed) - set(X.columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        result = pd.DataFrame(index=X.index)
        for c in NUMERIC:
            if not self.include_history and c in HISTORY:
                continue
            v = pd.to_numeric(X[c], errors="coerce")
            v = v.mask(~np.isfinite(v) | v.lt(0), np.nan)
            if c == "Is_International":
                v = v.where(v.isin([0, 1]), np.nan)
            if c in ["Account_Age_Days", "Previous_Transactions"]:
                v = v.where(v.mod(1).eq(0), np.nan)
            result[c] = v
        cats = CATEGORICAL + (["Suspicious_Keyword"] if self.include_keyword else [])
        for c in cats:
            values = X[c].astype("string").str.strip().replace("", pd.NA)
            result[c] = values.astype(object).where(values.notna(), np.nan)
        dates = parse_dates(X.Transaction_Date)
        hour = dates.dt.hour + dates.dt.minute / 60
        weekday = dates.dt.dayofweek
        result["Hour_Sin"] = np.sin(2 * np.pi * hour / 24)
        result["Hour_Cos"] = np.cos(2 * np.pi * hour / 24)
        result["Weekday_Sin"] = np.sin(2 * np.pi * weekday / 7)
        result["Weekday_Cos"] = np.cos(2 * np.pi * weekday / 7)
        result["Is_Weekend"] = weekday.ge(5).astype(float).where(dates.notna(), np.nan)
        result["Log_Amount"] = np.log1p(result.Transaction_Amount)
        if self.include_history:
            result["Log_Average_Spend"] = np.log1p(result.Average_Spend)
            result["Amount_to_Average"] = result.Transaction_Amount / result.Average_Spend.where(result.Average_Spend.gt(0))
            result["Amount_Minus_Average"] = result.Transaction_Amount - result.Average_Spend
            result["Log_Account_Age"] = np.log1p(result.Account_Age_Days)
            result["Log_Previous_Transactions"] = np.log1p(result.Previous_Transactions)
        for c in result.select_dtypes(include="number").columns:
            result[c] = result[c].replace([np.inf, -np.inf], np.nan)
        return result


def build_pipeline(estimator, scale=False, include_keyword=False, include_history=True):
    nums = [c for c in NUMERIC if include_history or c not in HISTORY]
    nums += TIME_FEATURES + (DERIVED if include_history else ["Log_Amount"])
    cats = CATEGORICAL + (["Suspicious_Keyword"] if include_keyword else [])
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), nums),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats),
    ], remainder="drop")
    return Pipeline([
        ("features", TransactionFeatures(include_keyword=include_keyword, include_history=include_history)),
        ("preprocess", preprocessing), ("model", estimator),
    ])
