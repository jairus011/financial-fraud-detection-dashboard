# Reference notebook audit

Reference inspected: `Financial_Fraud_Detection_System_(Final)(2).ipynb` (116 cells).
It was treated as a structural reference, not as a valid source of final results.
It is not copied into the submission, keeping one unambiguous final notebook.

| Observed issue | Evidence from reference code | Final-project resolution |
|---|---|---|
| Different dataset | Cell 3 loads `synthetic_fraud_dataset1.csv`; schema includes Fraud_Label, User_ID, Account_Balance and Card_Type | Use only the final 5,000-row supplied CSV and validate its SHA-256 |
| Resampling leakage | Cell 40 applies SMOTE to X and y before later splits | No synthetic resampling; compare training class weights |
| Global preprocessing | Normalized/resampled arrays are used for subsequent splits | Each model owns a training-fitted preprocessing pipeline |
| Inconsistent evaluation cohorts | Cells 82, 85, 94 use different test fractions (25%, 15%, 30%) | All models share one chronological train/validation/test assignment |
| Potential removal of fraud signal | Cell 36 removes transaction-amount IQR outliers | Retain unusual valid amounts |
| Accuracy focus | Model cells report accuracy prominently | Primary selection uses validation AP; report precision, recall, F1 and ranking metrics |
| Nonportable inputs | Dataset path depends on a hosted-notebook filesystem | Module-relative project paths and portable commands |
| Unnecessary complexity | Multiple SVM, autoencoder and ensemble experiments with inconsistent setup | Four supervised families, weighting comparison and one useful anomaly detector |

No reference notebook metric, figure, label or dataset-specific feature is copied into the
final output. Its errors are described to justify the corrected workflow, not to allege
anything about the original author's intent.
