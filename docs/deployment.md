# Online dashboard deployment

The project is prepared for one Render web service. A Streamlit Community Cloud
account is not needed when the Streamlit application runs on Render.

## Prepared service settings

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

The startup script binds to `0.0.0.0` and the host-provided `PORT`, disables file
watching, and uses the same committed model pipelines as the local dashboard.
The deployment loads saved artifacts; it does not retrain models on server startup.
Use `python scripts/verify_launch.py` to check this startup path locally.

## Access and data

The proposed web service is a publicly accessible educational demonstration. It
displays the supplied historical transaction dataset and allows users to download
scored rows. The source repository can remain private, provided Render's GitHub
connection has access to this repository. Uploaded CSVs are processed in the user's
app session and returned as downloads; the application does not save them to a
shared database or send email alerts.

SQLite is a committed, reproducible local ETL artifact. No extra hosted database
or persistent disk is needed for this read-only historical demonstration.

## Free hosting behavior

Render's Free services sleep after 15 minutes without traffic and may take about
a minute to wake on the next visit. Open the app before a presentation. Workspace
usage limits still apply. The filesystem is ephemeral; changes made by a running
service are lost on restart or redeploy. See [Render's Free service documentation](https://render.com/docs/free).

## Final deployment verification

After the workspace is confirmed and the service is created:

1. Confirm Render reports a live deploy and the health endpoint returns `200 / ok`.
2. Open all six pages and check the actual 5,000-row dataset KPIs and test metrics.
3. Submit a prediction and exercise filters and the simulated alert download.
4. Save genuine Executive Overview, Model Performance and Transaction Prediction
   screenshots under `outputs/screenshots/`.
5. Add the verified live URL and screenshot links to the README and project report.

No live URL is claimed until these deployment checks have succeeded.

References: [Render Python versions](https://render.com/docs/python-version),
[Render web services and port binding](https://render.com/docs/web-services).
