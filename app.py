import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# Page styling & layouts
st.set_page_config(
    page_title="MedixPort | Clinical Diagnostics Portal",
    page_icon="⚕️",
    layout="wide"
)

# Custom medical styling
st.markdown("""
<style>
    .brand-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0284c7;
        margin-bottom: 0.2rem;
    }
    .brand-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .panel-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .vitals-header {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Synthesize classifier model data (so the ML component is genuine and functional)
@st.cache_resource
def train_diagnostic_model():
    # Features: [Age, HeartRate, O2Saturation, SystolicBP]
    # Label: 0 = Stable, 1 = Elevated Risk, 2 = Critical Urgency
    np.random.seed(42)
    X = np.random.randint(low=[18, 60, 90, 90], high=[85, 120, 100, 160], size=(200, 4))
    y = []
    for row in X:
        age, hr, o2, sbp = row
        # Rules to define target label
        if o2 < 93 or sbp > 150 or hr > 115:
            y.append(2) # Critical
        elif o2 < 95 or sbp > 135 or hr > 95:
            y.append(1) # Elevated
        else:
            y.append(0) # Stable
            
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    return clf

model = train_diagnostic_model()

# Header layout
st.markdown('<div class="brand-title">MedixPort ⚕️</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Automated Telehealth Diagnostics & Biometric Classification Platform</div>', unsafe_allow_html=True)

col_form, col_dash = st.columns([1, 2])

with col_form:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Patient Clinical Profile")
    
    # Demographics
    patient_name = st.text_input("Full Patient Name", value="Eleanor Vance")
    age = st.slider("Patient Age (Years)", min_value=1, max_value=100, value=45)
    gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
    
    st.divider()
    st.subheader("Vital Statistics Intake")
    
    # Vitals Input
    heart_rate = st.slider("Heart Rate (BPM)", min_value=40, max_value=160, value=78)
    o2_sat = st.slider("Oxygen Saturation (SpO2 %)", min_value=80, max_value=100, value=98)
    systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=120, value=80)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_dash:
    st.write("### Clinical Urgency Classification")
    
    # Model inference
    input_data = np.array([[age, heart_rate, o2_sat, systolic_bp]])
    prediction_idx = model.predict(input_data)[0]
    
    # Map predictions
    risk_mapping = {
        0: {"label": "STABLE", "color": "#10b981", "desc": "Patient metrics are within standard physiological boundaries. Routine monitoring recommended."},
        1: {"label": "ELEVATED RISK", "color": "#f59e0b", "desc": "Sub-optimal metrics detected. Check for secondary symptoms and evaluate cardiovascular load."},
        2: {"label": "CRITICAL URGENCY", "color": "#ef4444", "desc": "Significant clinical anomalies detected (severe oxygen deficit or hypertensive warning). Immediate attention recommended."}
    }
    
    risk_details = risk_mapping[prediction_idx]
    
    # Render Risk Banner
    st.markdown(f"""
    <div style="background-color: {risk_details['color']}; padding: 1.5rem; border-radius: 8px; color: #000; font-weight: 800; text-align: center; margin-bottom: 1.5rem;">
        <span style="font-size: 1.8rem; letter-spacing: 2px;">{risk_details['label']}</span>
        <p style="margin-top: 0.5rem; font-weight: 400; font-size: 0.95rem; color: #000;">{risk_details['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Telemetry Visualizations
    st.write("#### Biometric Telemetry Gauges")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        # Heart rate gauge
        fig_hr = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = heart_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pulse (BPM)", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [40, 160]},
                'bar': {'color': "#0284c7"},
                'steps': [
                    {'range': [40, 60], 'color': "#f1f5f9"},
                    {'range': [60, 100], 'color': "#e2e8f0"},
                    {'range': [100, 160], 'color': "#fee2e2"}
                ]
            }
        ))
        fig_hr.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_hr, use_container_width=True)

    with v_col2:
        # Oxygen gauge
        fig_o2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = o2_sat,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "SpO2 (%)", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [80, 100]},
                'bar': {'color': "#10b981"},
                'steps': [
                    {'range': [80, 90], 'color': "#fee2e2"},
                    {'range': [90, 95], 'color': "#fef3c7"},
                    {'range': [95, 100], 'color': "#e2e8f0"}
                ]
            }
        ))
        fig_o2.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_o2, use_container_width=True)

    with v_col3:
        # Systolic BP gauge
        fig_bp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = systolic_bp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Systolic BP (mmHg)", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [80, 200]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [80, 120], 'color': "#e2e8f0"},
                    {'range': [120, 140], 'color': "#fef3c7"},
                    {'range': [140, 200], 'color': "#fee2e2"}
                ]
            }
        ))
        fig_bp.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_bp, use_container_width=True)

    st.divider()
    
    # Generate assessment documentation
    st.write("#### Clinical Summary & Telehealth Records")
    
    report_text = f"""==================================================
MEDIXPORT TELEHEALTH ASSESSMENT REPORT
==================================================
Date: 2026-06-22
Patient Identity: {patient_name}
Age: {age} | Sex: {gender}
--------------------------------------------------
Biometric Vitals Logged:
- Pulse Rate: {heart_rate} BPM
- Oxygen Saturation (SpO2): {o2_sat}%
- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
--------------------------------------------------
Automated Urgency Classification: {risk_details['label']}
Assessment:
{risk_details['desc']}
=================================================="""

    st.code(report_text, language="text")
    
    st.download_button(
        label="Download Clinical Summary (TXT)",
        data=report_text,
        file_name=f"medixport_{patient_name.replace(' ', '_').lower()}_report.txt",
        mime="text/plain",
        use_container_width=True
    )
