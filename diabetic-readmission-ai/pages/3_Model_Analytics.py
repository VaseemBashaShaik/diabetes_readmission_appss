import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt

st.title("Model Analytics Dashboard")

with open("models/model_info.json", "r") as f:
    info = json.load(f)

metrics = info.get("metrics", {})
st.subheader("Model Performance Metrics")
st.write(pd.DataFrame(metrics, index=[0]))

# Example ROC curve (optional)
try:
    roc_img = "assets/roc_curve.png"
    st.image(roc_img, caption="ROC Curve")
except Exception:
    st.info("ROC Curve image not found. You can add it under /assets/")
