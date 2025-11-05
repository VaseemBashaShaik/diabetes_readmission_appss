import streamlit as st

st.set_page_config(
    page_title="AI Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI-Powered Hospital Readmission Predictor")
st.markdown("""
Welcome to the **Hospital Readmission Risk Prediction System**.  
Use the sidebar to navigate through the pages:
- Predict Patient Risk  
- Model Analytics  
- About This App
""")
