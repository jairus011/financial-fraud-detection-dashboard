# Online dashboard deployment

The Streamlit dashboard is deployed as a Render web service and is publicly accessible at:

**https://financial-fraud-detection-dashboard.onrender.com/**

A Streamlit Community Cloud account is not required because the application runs directly on Render.

## Verified service settings

| Setting | Value |
|---|---|
| Service name | `financial-fraud-detection-dashboard` |
| Source | `https://github.com/jairus011/financial-fraud-detection-dashboard` |
| Branch | `main` |
| Runtime | Python 3.12, selected by `.python-version` |
| Build command | `python -m pip install -r requirements.txt` |
| Start command | `python scripts/start_dashboard.py` |
| Compute plan | Free |
| Region | Frankfurt |
| Automatic deploys | Enabled |
| Health endpoint | `/_stcore/health` |
| Required secrets | None |

The startup script binds to `0.0.0.0` and Render's host-provided `PORT`, disables file
watching, and uses the same committed model pipelines as the local dashboard. The deployment
loads saved artifacts and does not retrain models on server startup.

## Deployment verification

Render reports the latest deployment as live. The service build completed successfully,
`python scripts/start_dashboard.py` launched Streamlit, and Render detected the web process on
port 10000 before publishing the primary URL above.

The repository is public, so assessors can inspect the code and open the hosted dashboard
without being added as collaborators.

## Access and data

The service is a publicly accessible educational demonstration. It displays the supplied
historical transaction dataset and allows users to download scored rows. Uploaded CSVs are
processed during the user's app session and returned as downloads; the application does not
persist those uploads to a shared database or send real external fraud alerts.

SQLite remains a committed, reproducible local ETL artifact. No extra hosted database or
persistent disk is required for this read-only historical demonstration.

## Free hosting behavior

Render Free services may sleep after a period without traffic and can take time to wake on the
next visit. Open the app shortly before a presentation. The filesystem is ephemeral; changes
made by a running service are lost when the instance restarts or redeploys.

## Remaining presentation task

The only deployment-related presentation item still outstanding is the capture of genuine
browser screenshots from the live dashboard. Capture and save:

1. `executive_overview.png` - default full-dataset KPIs and charts.
2. `model_performance.png` - selected model metrics and confusion matrix.
3. `transaction_prediction.png` - a submitted example transaction with its returned scores.

Place them under `outputs/screenshots/` and embed them in the root README and final report.
No generated chart should be represented as a dashboard screenshot.

References: [Render Python versions](https://render.com/docs/python-version),
[Render web services and port binding](https://render.com/docs/web-services), and
[Render Free service documentation](https://render.com/docs/free).
