from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data/raw/financial_fraud_detection_dataset.csv"
SEED = 42
TARGET = "Fraudulent"
ID_COLUMNS = ["Transaction_ID", "Customer_ID"]
CATEGORICAL = ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]
NUMERIC = ["Transaction_Amount", "Is_International", "Previous_Transactions",
           "Average_Spend", "Account_Age_Days"]
HISTORY = ["Previous_Transactions", "Average_Spend", "Account_Age_Days"]
INPUT_COLUMNS = ["Transaction_Date", *NUMERIC, *CATEGORICAL]
RAW_COLUMNS = [*ID_COLUMNS, "Transaction_Date", *NUMERIC, *CATEGORICAL,
               "Suspicious_Keyword", TARGET]
EXPECTED_SHA256 = "3f106a0947420fee300785f4493847b6bd9efda5647c512a02cc38017dc3b76e"
