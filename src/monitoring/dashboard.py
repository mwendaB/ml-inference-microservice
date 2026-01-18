import json
from typing import List

import plotly.express as px
import streamlit as st

from src.monitoring.metrics import MetricsStore


st.set_page_config(page_title="Detection Monitoring", layout="wide")

store = MetricsStore()

st.title("Advanced Multi-Model Monitoring")

latency_values = [float(x) for x in store.fetch("latency_ms") if x]
active_streams_values = [int(float(x)) for x in store.fetch("active_streams") if x]

col1, col2 = st.columns(2)
with col1:
    st.metric("Recent Latency (ms)", f"{latency_values[0] if latency_values else 0:.1f}")
with col2:
    st.metric("Active Streams", active_streams_values[0] if active_streams_values else 0)

if latency_values:
    fig = px.line(y=list(reversed(latency_values)), labels={"y": "Latency (ms)"})
    st.plotly_chart(fig, use_container_width=True)

model_counts = store.fetch("model_used")
if model_counts:
    model_hist = {}
    for model in model_counts:
        model_hist[model] = model_hist.get(model, 0) + 1
    fig = px.bar(x=list(model_hist.keys()), y=list(model_hist.values()), labels={"x": "Model", "y": "Count"})
    st.plotly_chart(fig, use_container_width=True)

backend_counts = store.fetch("backend")
if backend_counts:
    backend_hist = {}
    for backend in backend_counts:
        backend_hist[backend] = backend_hist.get(backend, 0) + 1
    fig = px.pie(names=list(backend_hist.keys()), values=list(backend_hist.values()), title="Backend Usage")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw Metrics (JSON)")
raw = {
    "latency_ms": latency_values[:20],
    "active_streams": active_streams_values[:20],
}
st.code(json.dumps(raw, indent=2))
