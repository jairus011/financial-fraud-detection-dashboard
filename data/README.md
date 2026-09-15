# Dataset provenance and feature dictionary

`raw/financial_fraud_detection_dataset.csv` is an unchanged byte-for-byte copy of the
user-supplied `financial_fraud_detection_dataset(2).csv`. It is the only training/evaluation
dataset. The `(2)` suffix is an upload filename suffix, not a second experimental dataset.

SHA-256: `3f106a0947420fee300785f4493847b6bd9efda5647c512a02cc38017dc3b76e`.

5,000 rows; 14 columns; 482 fraud labels (9.64%); 3,847 customer IDs; no missing values
or exact duplicate rows in the supplied file. Dates run from 1 January 2023 to 21 February
2024. No external source URL, licence, currency, timezone, label-generation process or
confirmation of real/synthetic provenance was supplied. Do not represent this as verified
bank production data. No broader redistribution licence is granted by this repository.

| Column | Meaning from the supplied name/content | Final model use |
|---|---|---|
| Transaction_ID | Unique transaction reference | Traceability only |
| Customer_ID | Customer reference, sometimes repeated | Audit/split diagnostics only |
| Transaction_Date | Day-first date and minute timestamp | Cyclic hour/weekday and weekend features |
| Transaction_Amount | Nonnegative transaction amount; units unspecified | Raw amount and log(1 + amount) |
| Merchant_Category | Eight merchant categories | One-hot encoded |
| Payment_Method | Credit Card, Debit Card, NetBanking, PayPal, UPI | One-hot encoded |
| Device_Type | Desktop, Mobile, POS | One-hot encoded |
| Location | Seven city labels present in the CSV | One-hot encoded; no invented geography |
| Is_International | Binary 0/1 transaction attribute | Numeric |
| Previous_Transactions | Supplied count snapshot; definition unverified | Excluded; diagnostic sensitivity only |
| Average_Spend | Supplied spending snapshot; definition unverified | Excluded; diagnostic sensitivity only |
| Account_Age_Days | Supplied account-age snapshot | Excluded; diagnostic sensitivity only |
| Suspicious_Keyword | Yes/No flag with unspecified creation timing | Excluded; diagnostic sensitivity only |
| Fraudulent | Binary outcome: 1 fraud, 0 legitimate | Target only |

The audit found 606 decreases in account age and 572 decreases in the previous-transaction
count within customer IDs sorted by timestamp. Different account semantics, inconsistent
identifiers or synthetic construction could explain these patterns; provenance is unknown.
The final pipeline conservatively excludes all three history snapshots.

`processed/transactions_clean.csv` preserves all 5,000 valid rows and uses ISO dates.
`processed/fraud_detection.sqlite` contains the same data in the `transactions` table with
transaction-ID and date indexes. Optional future missing numeric values remain NULL in ETL;
imputation belongs to the fitted model pipeline, not the database.
`prediction_example.csv` contains eight real supplied rows with only required prediction fields.
