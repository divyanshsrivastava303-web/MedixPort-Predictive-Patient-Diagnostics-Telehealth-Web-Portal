# MedixPort: Predictive Patient Diagnostics & Telehealth Portal

MedixPort is an advanced healthcare diagnostics web application designed to assist medical practitioners in analyzing patient vital statistics, classifying clinical risk levels, and generating automated medical report summaries.

## Features
- **Dynamic Diagnostic Intake Form**: Input patient symptoms, demographic data, and biometric markers (blood pressure, heart rate, oxygen saturation).
- **Risk Assessment Core**: Integrates a predictive algorithm to classify patient health urgency into *Critical*, *Elevated*, or *Stable* categories.
- **Interactive Vitals Visualization**: Renders high-fidelity telemetry visualizations mapping patient metrics against standard physiological baselines.
- **Clinical Summary Downloader**: Generates a professional, print-ready text/PDF assessment report for telehealth records.

## Tech Stack
- **Engine**: Python 3.9+
- **Application Portal**: Streamlit
- **Analytics & Plots**: Plotly Express, NumPy
- **Algorithm Backend**: Scikit-Learn (Decision Classifier)

## Setup and Local Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```

---


1. **Upload the code** to a new repository on your GitHub account.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **New app**, select your repository, branch (`main`), and path to the entry file (`app.py`).
4. Click **Deploy!** 
5. Within 1 minute, you will receive a public URL (e.g., `https://medixport.streamlit.app`) that you can screen share live to show it works!
