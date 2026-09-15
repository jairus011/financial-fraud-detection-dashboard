# Dashboard screenshot status

Actual browser screenshots could not be captured in the build environment because the
browser blocks access to the separately isolated local Streamlit server. No image in this
folder is represented as a screenshot without an actual capture. The generated images in
`../figures/` are data/report plots, not dashboard screenshots.

The Streamlit app passed all six page-render tests, the single-prediction form, filters,
alert simulation and a real server health check. For visual submission evidence, run
`python -m streamlit run app.py` and capture:

1. `executive_overview.png`: default full-dataset KPIs and charts.
2. `model_performance.png`: selected model metrics and confusion matrix.
3. `transaction_prediction.png`: a submitted example transaction with its returned scores.

Place genuine screenshots in this folder and reference them from the root README.
These captures remain a documented presentation task, not an executed verification claim.
