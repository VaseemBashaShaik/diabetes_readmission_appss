import streamlit as st

st.title("About This App")

st.markdown("""
### Project: AI-Based Hospital Readmission Predictor

This app uses a **Gradient Boosting Classifier** trained on hospital patient data to predict the probability of readmission within 30 days.  
It helps doctors identify high-risk patients early and take preventive measures.

**Developed using:**
- Python (scikit-learn, pandas, numpy)
- Streamlit for frontend
- Gradient Boosting for classification

**Goal:** Reduce hospital readmissions, save healthcare costs, and improve patient care.
""")
