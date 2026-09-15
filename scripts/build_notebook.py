"""Generate the original final notebook; execute with scripts/execute_notebook.py."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md("""
# Financial Fraud Detection Model with Dashboard
**Zidio final project · Jairus Omondi**

This original notebook follows the supplied project specification and uses **only**
`data/raw/financial_fraud_detection_dataset.csv`. The reference notebook used a different
schema and was not reused for metrics. All results below are recomputed.

Objectives: audit the data, create SQLite ETL, build leakage-safe supervised and unsupervised
pipelines, compare fraud metrics, save reusable models, and connect the same inference code
to the Streamlit dashboard. This is an educational human-review prototype.

Run all cells with Python 3.12 and `requirements.txt` installed. The notebook rebuilds
project artifacts deterministically. It runs from the repository root or its notebook folder.
""")
code("""
from pathlib import Path
import json, sys, sqlite3
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/train.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import RAW_DATA, TARGET
from src.data import load_clean, chronological_split, write_etl
from src.features import TransactionFeatures
from src.predict import PRODUCTION_INPUTS, load_artifacts, score_transactions
from src.evaluation import evaluate
from src.train import run, candidates
print('Project:', ROOT.name)
""")
md("""
## 1. Source data and schema
The final CSV is the uploaded `financial_fraud_detection_dataset(2).csv`, renamed without
changing its bytes. Transaction IDs, customer IDs, dates, amounts, merchant/payment/device/location
categories, international status, three history snapshots, a suspicious-keyword flag and the
`Fraudulent` label are present. Currency, source timezone, label-generation process and
external provenance are unspecified.
""")
code("""
raw = pd.read_csv(RAW_DATA)
display(raw.head())
display(pd.DataFrame({'dtype': raw.dtypes.astype(str), 'missing': raw.isna().sum(), 'unique': raw.nunique()}))
print('Rows, columns:', raw.shape)
""")
md("""
## 2. ETL and data-quality audit
Cleaning trims text, parses explicit day-first or ISO dates, validates labels and transaction
keys, removes exact duplicates, and turns invalid optional numerics into missing values.
Median imputation is learned later, inside training pipelines. Zero and extreme amounts
are retained because they can carry a fraud signal. No class resampling occurs during ETL.
""")
code("""
clean, audit = load_clean()
display(pd.Series({k: v for k, v in audit.items() if not isinstance(v, (list, dict))}))
display(pd.Series(audit['history_caution']))
db_path = write_etl(clean)
with sqlite3.connect(db_path) as conn:
    sql_summary = pd.read_sql_query('SELECT COUNT(*) AS transactions, SUM(Fraudulent) AS frauds, SUM(Transaction_Amount) AS total_amount FROM transactions', conn)
display(sql_summary)
assert sql_summary.transactions.iloc[0] == len(clean)
""")
md("""
## 3. Leakage checks and fixed split
The dataset contains inconsistent history snapshots within time-sorted customer IDs.
The final feature set excludes `Previous_Transactions`, `Average_Spend` and `Account_Age_Days`.
`Suspicious_Keyword` is also excluded because its availability before fraud adjudication is
unknown. IDs and the target are never model features.

Use the earliest 60% for training, the next 20% for validation and the final 20% for testing.
Identical boundary timestamps stay together. Returning customers can occur across partitions,
but identifiers are not predictors. No scaling, encoding, imputation, balancing or model
selection is fitted on validation/test data. See `docs/evaluation_protocol.md`.
""")
code("""
parts = chronological_split(clean)
split_table = pd.DataFrame([{'split': name, 'rows': len(p), 'frauds': int(p.Fraudulent.sum()),
    'fraud_rate': p.Fraudulent.mean(), 'start': p.Transaction_Date.min(), 'end': p.Transaction_Date.max()}
    for name, p in parts.items()])
display(split_table)
assert parts['train'].Transaction_Date.max() < parts['validation'].Transaction_Date.min()
assert parts['validation'].Transaction_Date.max() < parts['test'].Transaction_Date.min()
assert not set(PRODUCTION_INPUTS) & {'Transaction_ID', 'Customer_ID', TARGET, 'Suspicious_Keyword'}
""")
md("""
## 4. Exploratory data analysis
The following full-dataset summaries are descriptive dashboard statistics, not model evaluation.
Inspect class balance, amounts and category rates alongside denominators. Any modelling
interpretation and permutation importance use training or validation data only.
""")
code("""
display(clean.groupby(TARGET).agg(transactions=('Transaction_ID', 'size'), amount=('Transaction_Amount', 'sum')))
display(clean[['Transaction_Amount', 'Is_International']].describe())
for column in ['Merchant_Category', 'Location', 'Device_Type', 'Payment_Method']:
    table = clean.groupby(column).agg(transactions=(TARGET, 'size'), frauds=(TARGET, 'sum'))
    table['fraud_rate'] = table.frauds / table.transactions
    display(table.sort_values('fraud_rate', ascending=False))
""")
md("""
## 5. Feature engineering and preprocessing
The final pipeline uses seven supplied fields. Row-local engineered features are log(1 + amount),
hour sine/cosine, weekday sine/cosine, and a weekend indicator. These require no future rows.
Numeric median imputation and categorical unknown-value imputation are inside the pipeline.
Categoricals use one-hot encoding with unseen-category support. Standard scaling is used
for Logistic Regression and Isolation Forest; tree classifiers do not require it.

History-based ratios and log transforms exist only for a diagnostic experiment.
No PCA is used: this small mixed dataset has directly interpretable features and the required
visualisations do not need a learned projection.
""")
code("""
features = TransactionFeatures(include_history=False).fit_transform(parts['train'][PRODUCTION_INPUTS])
display(features.head())
print('Raw production inputs:', PRODUCTION_INPUTS)
print('Engineered numeric/categorical columns before encoding:', features.shape[1])
""")
md("""
## 6. Supervised comparison and unsupervised detector
Compare Logistic Regression, a constrained Decision Tree, Random Forest and histogram
gradient boosting, each with and without class weights. Histogram gradient boosting is
the suitable gradient-boosting alternative to XGBoost and avoids an additional dependency.

Select by validation **average precision (AP)**, then set the decision threshold by maximum
validation F1. This is a demo threshold; business costs and review capacity are not supplied.
SMOTE is unnecessary for this benchmark: class weights are compared directly and no synthetic
training rows are introduced.

Isolation Forest is fitted to all unlabelled training features. Its review cutoff is the
90th percentile of training anomaly scores, a fixed 10% review-budget assumption. Fraud
labels are used only to evaluate it. Anomaly scores are not fraud probabilities. One-Class
SVM and neural autoencoders would add cost without an established benefit here.
""")
code("""
# This cell actually rebuilds ETL, all models, diagnostics, scores, alerts and figures.
# The model/threshold are frozen on validation before any test model comparison is computed.
results = run(data_path=RAW_DATA, output_root=ROOT)
""")
code("""
comparison = pd.read_csv(ROOT / 'outputs/model_comparison.csv')
columns = ['model','precision','recall','f1','roc_auc','average_precision','pr_auc','threshold']
display(comparison.loc[comparison.split.eq('validation'), columns].sort_values('average_precision', ascending=False))
print('Frozen selected model:', results['final_model'])
""")
md("""
## 7. Held-out test evaluation
Every model uses the same untouched test period. Test results describe the already-frozen
selection; they are not used to switch models. Precision, recall and F1 use the saved
validation threshold. ROC-AUC and PR summaries use continuous scores.

AP is the primary PR summary; trapezoidal PR-AUC is reported separately. The dummy model's
trapezoidal PR-AUC can appear large because it interpolates between just a few endpoints.
Its AP equals prevalence and correctly indicates no useful ranking.
""")
code("""
display(comparison.loc[comparison.split.eq('test'), columns].sort_values('average_precision', ascending=False))
display(pd.Series(results['test']))
m = results['test']
display(pd.DataFrame([[m['tn'],m['fp']],[m['fn'],m['tp']]], index=['Actual legitimate','Actual fraud'], columns=['Predicted legitimate','Predicted review']))
display(Image(filename=str(ROOT / 'outputs/figures/confusion_matrix.png')))
display(Image(filename=str(ROOT / 'outputs/figures/test_curves.png')))
""")
md("""
## 8. Uncertainty, customer overlap and source sensitivity
The customer-group bootstrap resamples test customers with replacement, keeping each
customer's rows together. Its intervals are conditional on the fitted model and this
sample; they do not capture training/selection uncertainty or institutional transfer.
Seen/unseen customer slices help disclose the implications of the chronological split.
The two feature sensitivity experiments use validation only and cannot replace the final model.
""")
code("""
display(pd.DataFrame(results['test_confidence_intervals']['intervals_95'], index=['2.5th percentile','97.5th percentile']).T)
display(pd.DataFrame(results['customer_slices']))
display(pd.read_csv(ROOT / 'outputs/feature_sensitivity.csv'))
display(pd.read_csv(ROOT / 'outputs/feature_importance.csv'))
display(pd.Series(results['isolation_forest_test']))
display(Image(filename=str(ROOT / 'outputs/figures/anomaly_scores.png')))
""")
md("""
## 9. Saved models and scored outputs
The saved joblib bundles contain the entire fitted feature/preprocessing/model pipeline and
the frozen threshold. Load only trusted model files. The dashboard calls the same
`score_transactions` function below. Labels and IDs, even when supplied, cannot affect scores.
The exact train-only fitted model used for test evaluation is retained, without a silent refit.
""")
code("""
model_bundle, anomaly_bundle = load_artifacts()
scored = score_transactions(parts['test'], model_bundle, anomaly_bundle)
saved = pd.read_csv(ROOT / 'outputs/test_predictions.csv')
np.testing.assert_allclose(scored.Fraud_Score, saved.Fraud_Score, atol=1e-12)
recomputed = evaluate(scored.Fraudulent, scored.Fraud_Score, model_bundle['threshold'])
assert np.isclose(recomputed['f1'], results['test']['f1'])
display(scored[['Transaction_ID','Fraudulent','Fraud_Score','Predicted_Fraud','Anomaly_Score','Anomaly_Flag']].head(10))
print('Saved-pipeline predictions and reported metrics match.')
""")
code("""
from src.alerts import simulate_alerts
alerts = simulate_alerts(scored, limit=3)
display(pd.DataFrame(alerts))
print('Simulation only. No credentials or external notifications are used.')
""")
md("""
## 10. Dashboard and reproducibility
Launch from the repository root:
```bash
python -m streamlit run app.py
```
Six pages cover Executive Overview, Fraud Analysis, Transaction Patterns, Risk/Anomaly
Analysis, Model Performance, and Transaction Prediction. Filters update descriptive
transaction views. Performance remains fixed to the test period. The UI supports a
single-transaction form, batch CSV scoring, anomaly plots and a simulated review queue.

Use `python -m pytest` for the pipeline/dashboard checks. Use `python scripts/verify_fresh_run.py`
to rebuild models in an empty temporary output directory, compare results, and check
serialized prediction consistency. Read `README.md` for exact installation commands.
""")
code("""
for figure in ['dataset_overview.png', 'monthly_trends.png', 'training_correlation.png', 'model_comparison.png']:
    display(Image(filename=str(ROOT / 'outputs/figures' / figure)))
""")
md("""
## 11. Conclusions and limitations
The selected model provides useful ranking relative to test prevalence, but precision and
recall remain limited. A review flag is not proof of fraud, and a lower score is not proof
of legitimacy. Scores have not been calibrated. This supplied dataset has only 88 test
frauds, inconsistent histories, and unspecified provenance/currency/timezone; findings
cannot be represented as validated live-bank performance.

Priority improvements are verified point-in-time transaction histories, label/source
documentation, external and rolling-period validation, review-cost thresholding and
calibration. Kafka/Spark ingestion, graph analysis, email notification, adaptive learning,
mobile integration, credit scoring and blockchain are explicitly future work. The current
project provides offline ETL, batch scoring and a simulated alert workflow.

References: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
""")
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
destination = ROOT / "notebooks/financial_fraud_detection_final.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, destination)
print(f"Created {len(cells)} cells -> {destination.name}")
