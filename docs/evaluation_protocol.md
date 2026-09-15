# Evaluation protocol, fixed before model fitting

Only `data/raw/financial_fraud_detection_dataset.csv` is used. SHA-256:
`3f106a0947420fee300785f4493847b6bd9efda5647c512a02cc38017dc3b76e`.

1. Validate schema, parse day-first timestamps, remove exact duplicates and reject conflicting IDs.
   Invalid dates/keys/labels are excluded and counted. Invalid optional numerics become missing.
   Retain zero and extreme amounts; fraud must not be cleaned away as an outlier.
2. Sort by transaction time and transaction ID. Earliest 60% train, next 20% validation,
   latest 20% test; equal boundary timestamps stay together in the later partition.
3. Exclude target, transaction ID and customer ID from model inputs. Exclude
   `Suspicious_Keyword` because its creation time and relationship to labelling are unknown.
   Exclude the supplied history snapshots too: the source audit found 606 decreases in
   account age and 572 decreases in previous-transaction count within time-sorted customer IDs.
   Their semantics and point-in-time validity are unverified. No customer aggregates are
   computed across the full dataset. Historical inputs are used only in sensitivity analysis.
4. Train Logistic Regression, constrained Decision Tree, Random Forest, and histogram
   gradient boosting, each with and without class weights. No SMOTE or PCA is required.
   Each candidate has its own training-fitted median imputer, one-hot encoder and,
   for Logistic Regression, standard scaler. Fixed model settings keep the comparison small.
5. Select the highest validation average precision (AP). AP is the primary PR summary;
   also report trapezoidal PR-AUC separately to avoid conflating the two definitions.
   Select the chosen model's decision threshold on validation F1 (ties favour recall).
   This is a demo operating point, not a bank-approved cost or capacity policy.
6. Fit Isolation Forest on all unlabelled training features, never validation/test data.
   Use a prespecified 10% training-tail review budget, independent of fraud labels.
   High anomaly scores mean unusualness, not calibrated fraud probability.
7. Freeze model choice and all thresholds before evaluating the test set. Evaluate all
   candidates on the SAME test rows for transparent comparison; never select from test scores.
   Keep the exact train-only fitted artifact used for test evaluation; do not silently refit it.
8. Report precision, recall, F1, ROC-AUC, AP, trapezoidal PR-AUC and confusion matrices.
   Include an all-legitimate dummy baseline and the test prevalence. Accuracy is not a headline.
   Report test performance for seen/unseen customer IDs and customer-group bootstrap intervals.
9. Run validation-only keyword/history sensitivity checks using the selected architecture.
   Diagnostic keyword results do not affect the final pipeline. These checks cannot establish
   source provenance or prove that supplied historical fields are safe in production.
10. Save split IDs and dataset hash. Label scored training rows as in-sample; global dashboard
    KPIs describe the dataset. Only the fixed test partition supports headline model metrics.

This estimates future transactions in this supplied sample, including returning customers.
It does not establish deployment performance, new-institution transfer, currency, verified
real-world provenance, point-in-time history, fairness or calibrated probabilities.

Method references: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
