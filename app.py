"""Launch from the repository root: python -m streamlit run app.py."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alerts import simulate_alerts
from src.config import CATEGORICAL, ROOT
from src.predict import PRODUCTION_INPUTS, load_artifacts, score_transactions


st.set_page_config(page_title="Zidio | Financial Fraud Intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {padding-top:3.75rem; padding-bottom:2rem; max-width:1550px;}
h1 {font-size:2.25rem !important; letter-spacing:-.06rem;}
h2 {font-size:1.35rem !important;} h3 {font-size:1.1rem !important;}
[data-testid="stMetric"] {background:#fff; border:1px solid #dee7ed; border-radius:12px; padding:16px 20px;}
[data-testid="stMetricValue"] {font-weight:650; font-size:1.8rem;}
[data-testid="stSidebar"] {border-right:1px solid #dee7ed;}
.eyebrow {color:#176a61;font-size:12px;font-weight:700;letter-spacing:2px; margin-bottom:6px;}
.intro {color:#5e7181;font-size:15px;margin-top:-8px;margin-bottom:20px;}
.note {background:#e8f2ef;color:#285d55;padding:12px 16px;border-radius:8px;font-size:13px;}
.footer {font-size:12px;color:#6b7a87;padding-top:24px;border-top:1px solid #dfe6ec;margin-top:20px;}
</style>""", unsafe_allow_html=True)

COLORS = {"Legitimate": "#176A61", "Fraudulent": "#D56552"}
PAGES = ["Executive Overview", "Fraud Analysis", "Transaction Patterns", "Risk/Anomaly Analysis", "Model Performance", "Transaction Prediction"]


@st.cache_resource
def artifacts(version):
    return load_artifacts()


@st.cache_data
def data_assets(version):
    out = ROOT / "outputs"
    return (pd.read_csv(out / "scored_transactions.csv", parse_dates=["Transaction_Date"]),
            json.loads((out / "metrics.json").read_text()),
            pd.read_csv(out / "model_comparison.csv"),
            json.loads((out / "data_audit.json").read_text()),
            json.loads((out / "split_summary.json").read_text()))


def chart(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=35, b=15), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="white", font=dict(color="#263B50", size=12),
                      legend=dict(orientation="h", y=-.2), colorway=["#176A61", "#D56552", "#698EA8", "#D2A552"])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EEF2")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def metrics_row(items):
    for col, (label, value, help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text)


def category_stats(frame, column):
    result = frame.groupby(column, dropna=False).agg(Transactions=("Fraudulent", "size"), Frauds=("Fraudulent", "sum"), Amount=("Transaction_Amount", "sum")).reset_index()
    result["Fraud rate (%)"] = result.Frauds / result.Transactions * 100
    return result


needed = [ROOT / "models/fraud_model.joblib", ROOT / "models/isolation_forest.joblib",
          ROOT / "outputs/scored_transactions.csv", ROOT / "outputs/metrics.json",
          ROOT / "outputs/model_comparison.csv", ROOT / "outputs/data_audit.json", ROOT / "outputs/split_summary.json"]
if not all(p.exists() for p in needed):
    st.error("Project artifacts are missing. From the repository root, run: python -m src.train")
    st.stop()
version = tuple(p.stat().st_mtime_ns for p in needed)
df, performance, comparison, audit, splits = data_assets(version)
model, anomaly_model = artifacts(version)
df["Actual label"] = df.Fraudulent.map({0: "Legitimate", 1: "Fraudulent"})

with st.sidebar:
    st.markdown('<div class="eyebrow">ZIDIO · DATA SCIENCE</div>', unsafe_allow_html=True)
    st.markdown("## Fraud intelligence")
    st.caption("Financial Fraud Detection Model with Dashboard")
    page = st.radio("Workspace", PAGES, key="page")
    st.divider()
    st.markdown("**Explore transactions**")
    scope = st.selectbox("Data partition", ["All transactions", "train", "validation", "test"])
    dates = st.date_input("Date range", value=(df.Transaction_Date.min().date(), df.Transaction_Date.max().date()),
                          min_value=df.Transaction_Date.min().date(), max_value=df.Transaction_Date.max().date())
    location = st.selectbox("Location filter", ["All", *sorted(df.Location.dropna().unique())])
    device = st.selectbox("Device filter", ["All", *sorted(df.Device_Type.dropna().unique())])
    merchant = st.selectbox("Merchant filter", ["All", *sorted(df.Merchant_Category.dropna().unique())])
    st.caption("Filters apply to the four analysis pages. Evaluation stays on the fixed test period.")
    st.divider()
    st.caption("Historical dataset · offline prototype\n\nAmounts are in unspecified dataset units.")

filtered = df.copy()
if scope != "All transactions":
    filtered = filtered.loc[filtered.Split.eq(scope)]
if isinstance(dates, (tuple, list)) and len(dates) == 2:
    filtered = filtered.loc[filtered.Transaction_Date.dt.date.between(dates[0], dates[1])]
for column, value in [("Location", location), ("Device_Type", device), ("Merchant_Category", merchant)]:
    if value != "All":
        filtered = filtered.loc[filtered[column].eq(value)]

st.markdown('<div class="eyebrow">FINANCIAL FRAUD DETECTION MODEL</div>', unsafe_allow_html=True)
st.title(page)
subtitles = {
    "Executive Overview": "A clear view of transaction activity, observed fraud, and model performance.",
    "Fraud Analysis": "Understand where labelled fraud appears and how it compares with transaction volume.",
    "Transaction Patterns": "Explore when transactions happen and how payment activity changes over time.",
    "Risk/Anomaly Analysis": "Separate model review signals from unusual activity and confirmed dataset labels.",
    "Model Performance": "A transparent comparison on the same untouched test period.",
    "Transaction Prediction": "Score a transaction through the exact saved feature and preprocessing pipeline.",
}
st.markdown(f'<div class="intro">{subtitles[page]}</div>', unsafe_allow_html=True)
if page in PAGES[:4]:
    st.caption(f"CURRENT VIEW  ·  {len(filtered):,} of {len(df):,} transactions  ·  {scope}")
    if filtered.empty:
        st.info("No transactions match these filters. Broaden the date range or choose All.")
        st.stop()

if page == "Executive Overview":
    fraud = filtered.loc[filtered.Fraudulent.eq(1)]
    metrics_row([("Total transactions", f"{len(filtered):,}", "Count in the current filtered view."),
                 ("Fraudulent transactions", f"{len(fraud):,}", "Observed Fraudulent = 1 labels, not predictions."),
                 ("Observed fraud rate", f"{filtered.Fraudulent.mean():.2%}", "Fraud labels divided by all filtered rows.")])
    st.write("")
    metrics_row([("Total amount", f"{filtered.Transaction_Amount.sum():,.2f}", "Dataset units; no currency is supplied."),
                 ("Fraudulent amount", f"{fraud.Transaction_Amount.sum():,.2f}", "Amount attached to fraud labels; not verified monetary loss."),
                 ("Model review flags", f"{filtered.Predicted_Fraud.sum():,}", "Scores above the frozen validation-selected threshold; not confirmed fraud.")])
    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Transaction activity over time")
        monthly = filtered.assign(Month=filtered.Transaction_Date.dt.to_period("M").dt.to_timestamp()).groupby(["Month", "Actual label"]).size().reset_index(name="Transactions")
        chart(px.bar(monthly, x="Month", y="Transactions", color="Actual label", color_discrete_map=COLORS), 315)
    with right:
        st.subheader("Fraud exposure by merchant")
        cats = category_stats(filtered, "Merchant_Category").sort_values("Frauds")
        chart(px.bar(cats, x="Frauds", y="Merchant_Category", orientation="h", color_discrete_sequence=["#D56552"], hover_data=["Transactions", "Fraud rate (%)"]), 315)
    st.markdown("### Model snapshot · fixed held-out test")
    m = performance["test"]
    metrics_row([("Precision", f"{m['precision']:.2%}", "Share of model alerts that are labelled fraud."),
                 ("Recall", f"{m['recall']:.2%}", "Share of test frauds detected."),
                 ("Average precision", f"{m['average_precision']:.4f}", "Primary PR summary; test prevalence is 0.088."),
                 ("ROC-AUC", f"{m['roc_auc']:.4f}", "Ranking quality across thresholds.")])
    st.caption(f"{performance['final_model']} · {m['n']:,} test rows · {m['tp']} detected frauds · {m['fp']} false alerts · {m['fn']} missed frauds.")

elif page == "Fraud Analysis":
    dimension = st.selectbox("Compare by", ["Merchant_Category", "Location", "Device_Type", "Payment_Method", "Is_International"])
    stats = category_stats(filtered, dimension).sort_values("Fraud rate (%)", ascending=False)
    left, right = st.columns(2)
    with left:
        st.subheader("Observed fraud rate")
        chart(px.bar(stats, x=dimension, y="Fraud rate (%)", color_discrete_sequence=["#D56552"], hover_data=["Transactions", "Frauds"]))
    with right:
        st.subheader("Transaction volume and fraud counts")
        counts = stats.melt(id_vars=[dimension], value_vars=["Transactions", "Frauds"], var_name="Measure", value_name="Count")
        chart(px.bar(counts, x=dimension, y="Count", color="Measure", barmode="group"))
    st.caption("Compare rates with sample sizes. Associations do not establish causes or justify automatically blocking a category.")
    st.dataframe(stats, hide_index=True, width="stretch")
    st.subheader("Amounts by actual label")
    chart(px.box(filtered, x="Actual label", y="Transaction_Amount", color="Actual label", color_discrete_map=COLORS, points="outliers"))

elif page == "Transaction Patterns":
    frequency = st.selectbox("Time aggregation", ["Week", "Month", "Day"])
    freq = {"Week": "W", "Month": "MS", "Day": "D"}[frequency]
    trend = filtered.set_index("Transaction_Date").resample(freq).agg(Transactions=("Fraudulent", "size"), Frauds=("Fraudulent", "sum"), Amount=("Transaction_Amount", "sum")).reset_index()
    trend["Fraud rate (%)"] = (trend.Frauds / trend.Transactions.replace(0, np.nan) * 100)
    left, right = st.columns(2)
    with left:
        st.subheader("Activity and amount")
        measure = st.selectbox("Trend measure", ["Transactions", "Amount"])
        chart(px.line(trend, x="Transaction_Date", y=measure, markers=True, render_mode="svg"))
    with right:
        st.subheader("Observed fraud rate over time")
        chart(px.line(trend, x="Transaction_Date", y="Fraud rate (%)", markers=True, render_mode="svg", color_discrete_sequence=["#D56552"], hover_data=["Transactions", "Frauds"]))
    st.caption("Boundary periods may be partial; periods with no transactions have no fraud rate.")
    st.subheader("Hour and weekday activity")
    heat = filtered.assign(Hour=filtered.Transaction_Date.dt.hour, Weekday=filtered.Transaction_Date.dt.dayofweek).pivot_table(index="Weekday", columns="Hour", values="Transaction_ID", aggfunc="count", fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
    chart(px.imshow(heat.values, x=list(range(24)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], aspect="auto", color_continuous_scale="Teal", labels={"x": "Hour (source timezone unspecified)", "y": "Weekday", "color": "Transactions"}), 310)

elif page == "Risk/Anomaly Analysis":
    metrics_row([("Model review flags", f"{filtered.Predicted_Fraud.sum():,}", "Supervised score exceeded the validation threshold."),
                 ("Anomaly flags", f"{filtered.Anomaly_Flag.sum():,}", "Isolation Forest score exceeded the training 90th percentile."),
                 ("Both signals", f"{(filtered.Predicted_Fraud.eq(1) & filtered.Anomaly_Flag.eq(1)).sum():,}", "Transactions triggering both detectors.")])
    st.caption("Anomaly means unusual, not necessarily fraudulent. Model scores are not calibrated probabilities. Training rows are in-sample.")
    left, right = st.columns(2)
    with left:
        st.subheader("Supervised score vs anomaly score")
        chart(px.scatter(filtered, x="Anomaly_Score", y="Fraud_Score", color="Actual label", color_discrete_map=COLORS, render_mode="svg",
                         opacity=.55, hover_data=["Transaction_ID", "Transaction_Amount", "Split"]))
    with right:
        st.subheader("Anomaly score distribution")
        figure = px.histogram(filtered, x="Anomaly_Score", color="Actual label", color_discrete_map=COLORS, nbins=40, barmode="overlay")
        figure.add_vline(x=anomaly_model["threshold"], line_dash="dash", annotation_text="Training cutoff")
        chart(figure)
    st.subheader("Review queue · offline alert simulation")
    queue = filtered.loc[filtered.Predicted_Fraud.eq(1) | filtered.Anomaly_Flag.eq(1)].sort_values("Fraud_Score", ascending=False)
    st.dataframe(queue[["Transaction_ID", "Transaction_Date", "Transaction_Amount", "Fraud_Score", "Anomaly_Score", "Review_Status", "Actual label", "Score_Context"]].head(100), hide_index=True, width="stretch")
    if st.button("Simulate high-risk alerts", type="primary"):
        alerts = simulate_alerts(filtered, limit=25)
        st.success(f"Simulated {len(alerts)} review alerts. No email or external messages were sent.")
        st.download_button("Download simulated alerts", "\n".join(json.dumps(a) for a in alerts), file_name="simulated_fraud_alerts.jsonl", mime="application/x-ndjson", on_click="ignore")
        if alerts:
            st.json(alerts[0])
    st.download_button("Download filtered scored transactions", filtered.to_csv(index=False), file_name="filtered_scored_transactions.csv", mime="text/csv", on_click="ignore")

elif page == "Model Performance":
    st.markdown(f"**Selected model: {performance['final_model']}**")
    st.caption(f"Selected by validation average precision. Threshold {model['threshold']:.6f} was chosen on validation F1. The fitted model was not retrained on test rows.")
    m = performance["test"]
    metrics_row([("Precision", f"{m['precision']:.2%}", "TP / (TP + FP)"), ("Recall", f"{m['recall']:.2%}", "TP / (TP + FN)"),
                 ("F1-score", f"{m['f1']:.4f}", "Harmonic mean of precision and recall."), ("ROC-AUC", f"{m['roc_auc']:.4f}", "Threshold-independent ranking metric.")])
    st.write("")
    metrics_row([("Average precision (AP)", f"{m['average_precision']:.4f}", "Primary PR summary: recall-weighted step precision."),
                 ("PR-AUC (trapezoidal)", f"{m['pr_auc']:.4f}", "Trapezoidal area under the PR curve; distinct from AP."),
                 ("Test fraud prevalence", f"{m['prevalence']:.2%}", "No-skill reference for average precision.")])
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Confusion matrix")
        cm = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        chart(px.imshow(cm, x=["Legitimate", "Fraud review"], y=["Legitimate", "Fraudulent"], text_auto=True,
                        color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual"}), 310)
    with right:
        st.subheader("Test average precision comparison")
        test_comparison = comparison.loc[comparison.split.eq("test")].sort_values("average_precision")
        chart(px.bar(test_comparison, x="average_precision", y="model", orientation="h", color="selected", color_discrete_map={True: "#176A61", False: "#9CB4C6"}), 360)
    st.subheader("ROC and precision–recall curves")
    curves = pd.read_csv(ROOT / "outputs/performance_curves.csv")
    choices = st.multiselect("Models on curves", list(comparison.model.unique())[:-2] + ["Isolation Forest"], default=[performance["final_model"], "Isolation Forest"])
    for col, kind in zip(st.columns(2), ["ROC", "PR"]):
        with col:
            selected_curves = curves.loc[curves.model.isin(choices) & curves.curve.eq(kind)]
            figure = px.line(selected_curves, x="x", y="y", color="model", render_mode="svg", labels={"x": "False-positive rate" if kind == "ROC" else "Recall", "y": "Recall" if kind == "ROC" else "Precision"}, title=kind)
            if kind == "ROC":
                figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#99A5AC"))
            else:
                figure.add_hline(y=m["prevalence"], line_dash="dash", annotation_text="Test prevalence")
            chart(figure, 350)
    st.subheader("All models · fixed test partition")
    st.dataframe(test_comparison[["model", "kind", "threshold", "precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc", "tp", "fp", "fn", "tn"]], hide_index=True, width="stretch")
    st.caption("The dummy prior has AP equal to prevalence. Its trapezoidal PR area is misleadingly high because of linear interpolation between sparse endpoints; it is not the model-selection metric.")
    with st.expander("Validation threshold explorer"):
        st.write("This explores validation tradeoffs only. The deployed threshold and reported test metrics stay fixed.")
        t = st.slider("Explore validation threshold", 0.0, 1.0, float(model["threshold"]), .01)
        val = df.loc[df.Split.eq("validation")]
        from src.evaluation import evaluate
        vm = evaluate(val.Fraudulent, val.Fraud_Score, t)
        metrics_row([("Validation precision", f"{vm['precision']:.2%}", None), ("Validation recall", f"{vm['recall']:.2%}", None), ("Validation alerts", str(vm["alert_count"]), None)])
    with st.expander("Evaluation details, uncertainty and feature audit"):
        st.json({k: splits[k] for k in ["train", "validation", "test", "customer_overlap"]})
        ci = performance["test_confidence_intervals"]
        st.dataframe(pd.DataFrame(ci["intervals_95"], index=["2.5th percentile", "97.5th percentile"]).T, width="stretch")
        st.caption(ci["method"])
        st.dataframe(pd.read_csv(ROOT / "outputs/feature_sensitivity.csv"), hide_index=True)
        st.write("History and suspicious-keyword fields are excluded from the final model. These sensitivity results are validation-only diagnostics.")
        st.dataframe(pd.read_csv(ROOT / "outputs/feature_importance.csv"), hide_index=True)
        st.caption("Permutation importance measures validation AP decrease, not a causal effect.")

elif page == "Transaction Prediction":
    st.info("Educational review prototype. Scores are not calibrated fraud probabilities; predictions must not automatically block payments.")
    single, batch = st.tabs(["Single transaction", "Upload transactions"])
    with single:
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                amount = st.number_input("Transaction amount (dataset units)", min_value=0.0, value=125.0, step=10.0)
                category = st.selectbox("Merchant category", sorted(df.Merchant_Category.unique()))
                payment = st.selectbox("Payment method", sorted(df.Payment_Method.unique()))
                international = st.selectbox("International transaction", ["No", "Yes"])
            with right:
                date = st.date_input("Transaction date", value=df.Transaction_Date.max().date(), key="prediction_date")
                time = st.time_input("Transaction time", value=datetime.strptime("14:30", "%H:%M").time())
                input_device = st.selectbox("Device", sorted(df.Device_Type.unique()))
                input_location = st.selectbox("Location", sorted(df.Location.unique()))
            submitted = st.form_submit_button("Score transaction", type="primary")
        st.caption("Uses only transaction-time fields. Customer history and Suspicious_Keyword are not required or used. New dates and values may lie outside the training distribution.")
        if submitted:
            record = {"Transaction_Date": datetime.combine(date, time).isoformat(), "Transaction_Amount": amount,
                      "Merchant_Category": category, "Payment_Method": payment, "Device_Type": input_device,
                      "Location": input_location, "Is_International": int(international == "Yes")}
            prediction = score_transactions(pd.DataFrame([record]), model, anomaly_model).iloc[0]
            st.subheader(str(prediction.Review_Status))
            metrics_row([("Fraud model score", f"{prediction.Fraud_Score:.4f}", "Uncalibrated model score; higher means a stronger fraud signal."),
                         ("Anomaly score", f"{prediction.Anomaly_Score:.4f}", "Higher means more unusual relative to training."),
                         ("Review threshold", f"{model['threshold']:.4f}", "Frozen validation-selected score threshold.")])
            if prediction.Predicted_Fraud:
                st.warning("Model threshold exceeded: route to human review. This is not confirmation of fraud.")
            elif prediction.Anomaly_Flag:
                st.warning("Unusual transaction: the anomaly detector suggests a review.")
            else:
                st.success("Below the configured review cutoffs. A lower score does not guarantee a legitimate transaction.")
            st.download_button("Download prediction", pd.DataFrame([prediction]).to_csv(index=False), file_name="transaction_prediction.csv", mime="text/csv", on_click="ignore")
    with batch:
        st.write("Upload a CSV with the seven columns shown in the template. Blank numeric/categorical cells use training-fitted imputation; invalid dates or values receive a clear error.")
        st.download_button("Download input template", (ROOT / "data/prediction_example.csv").read_bytes(), file_name="prediction_example.csv", mime="text/csv", on_click="ignore")
        upload = st.file_uploader("Transaction CSV", type="csv")
        if upload is not None:
            try:
                incoming = pd.read_csv(upload)
                if len(incoming) > 10000:
                    raise ValueError("Upload at most 10,000 transactions for this local demo.")
                scored_upload = score_transactions(incoming, model, anomaly_model)
                st.success(f"Scored {len(scored_upload):,} transactions using the saved model.")
                st.dataframe(scored_upload.head(100), hide_index=True, width="stretch")
                st.download_button("Download scored CSV", scored_upload.to_csv(index=False), file_name="scored_upload.csv", mime="text/csv", on_click="ignore")
            except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))

st.markdown('<div class="footer">Zidio Financial Fraud Detection · Reproducible ML project · Historical sample, human review, transparent evaluation.</div>', unsafe_allow_html=True)
