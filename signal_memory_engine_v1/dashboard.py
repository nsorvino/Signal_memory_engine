# dashboard.py
import os

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


def show_dashboard():
    st.title("Signal Memory Engine Dashboard")
    mlflow_uri_cred = os.getenv("MLFLOW_TRACKING_URI")
    # allow user to override via Streamlit secrets or sidebar
    mlflow_uri = st.sidebar.text_input("MLflow Tracking URI", mlflow_uri_cred)
    client = MlflowClient(tracking_uri=mlflow_uri)

    exp = client.get_experiment_by_name("SignalMemoryEngine")
    if exp is None:
        st.error("Experiment 'SignalMemoryEngine' not found")
        return

    # fetch the latest 50 runs, sorted by start time descending
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    if not runs:
        st.info("No runs logged yet.")
        return

    # build a dataframe of metrics
    records = []
    for r in runs:
        row = {
            "run_id": r.info.run_id,
            "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
        }
        # flatten all numeric metrics
        for m, v in r.data.metrics.items():
            row[m] = v
        records.append(row)

    df = pd.DataFrame(records).set_index("start_time").sort_index()

    # show endpoint latency
    st.subheader("Endpoint Latency (ms)")
    if "endpoint_latency_ms" in df:
        st.line_chart(df["endpoint_latency_ms"])
    else:
        st.write("_No `endpoint_latency_ms` metric logged yet._")

    # show top similarity score drift
    st.subheader("Top Similarity Score")
    if "top_score" in df:
        st.line_chart(df["top_score"])
    else:
        st.write("_No `top_score` metric logged yet._")

    # optionally show any other metric, e.g. coherence drift
    if "coherence_drift_detected" in df:
        st.subheader("Coherence Drift Detected (0 = stable, 1 = drift)")
        st.line_chart(df["coherence_drift_detected"])
