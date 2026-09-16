# Zidio Submission - Financial Fraud Detection Model with Dashboard

## Submission links

- **GitHub repository:** https://github.com/jairus011/financial-fraud-detection-dashboard
- **Live Streamlit dashboard:** https://financial-fraud-detection-dashboard.onrender.com/

## Core deliverables completed

- End-to-end fraud-detection notebook
- Data audit, cleaning and reproducible SQLite ETL
- Feature engineering and leakage controls
- Logistic Regression, Decision Tree, Random Forest and Histogram Gradient Boosting comparisons
- Isolation Forest anomaly detection
- Validation-based model selection and frozen test evaluation
- Saved model artifacts and scored transaction outputs
- Interactive six-page Streamlit dashboard
- Single-transaction and batch CSV prediction
- Simulated high-risk fraud alerts for human review
- Automated tests and reproducibility checks
- Public GitHub repository
- Live Render deployment
- Technical project report in `reports/project_report.md`

## Final model

**Logistic Regression (unweighted)**, selected using validation Average Precision.

Held-out test results:

| Metric | Result |
|---|---:|
| Precision | 27.85% |
| Recall | 50.00% |
| F1-score | 0.3577 |
| ROC-AUC | 0.8019 |
| Average Precision | 0.3157 |
| PR-AUC | 0.3119 |
| True positives | 44 |
| False positives | 114 |
| False negatives | 44 |
| True negatives | 798 |

## Presentation evidence still to capture

The hosted dashboard is live, but genuine browser screenshots are intentionally not fabricated in the repository. Before final presentation/submission, capture these three views from the live Render dashboard:

1. Executive Overview
2. Model Performance
3. Transaction Prediction after submitting one example transaction

Save them as:

- `outputs/screenshots/executive_overview.png`
- `outputs/screenshots/model_performance.png`
- `outputs/screenshots/transaction_prediction.png`

Then embed them in the root `README.md` if visual evidence is required by the assessor.

## Scope notes

Kafka/Spark streaming, graph-based fraud-network analysis, real external email/SMS alert delivery, adaptive retraining, mobile integration, credit scoring and blockchain are documented as future enhancements. The submitted implementation is an offline historical fraud-screening and analytics prototype with batch scoring and alert simulation.

PCA and SMOTE are not used in the final pipeline because the chosen workflow retains interpretable native features and compares class-weighted/unweighted models directly.
