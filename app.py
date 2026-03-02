import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Lung Cancer Risk Predictor", page_icon="🫁", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }

    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-box h1 { color: white; margin: 0; font-size: 2.2rem; }
    .header-box p  { color: #a8c0e8; margin: 0.4rem 0 0; font-size: 1rem; }

    .section-card {
        background: white;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e9ecef;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    .result-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(255,65,108,0.35);
    }
    .result-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(17,153,142,0.35);
    }

    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #0f3460; }
    .metric-label { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }

    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1.5px solid #dee2e6;
        padding: 0.45rem 0.75rem;
        font-size: 0.9rem;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0f3460;
        box-shadow: 0 0 0 3px rgba(15,52,96,0.1);
    }
    .stRadio > div { gap: 0.5rem; }
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0f3460, #1a6fc4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 14px rgba(15,52,96,0.3) !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(15,52,96,0.4) !important;
    }

    div[data-testid="stAlert"] { border-radius: 10px; }
    .stSuccess { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("lung_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_file = str(e)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🫁 Lung Cancer Risk Predictor</h1>
    <p>AI-powered early risk assessment · Fill in patient details and click Predict</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Model files not found. Ensure `lung_model.pkl`, `scaler.pkl`, and `feature_names.pkl` are in the same directory.\n\n`{missing_file}`")
    st.stop()

st.markdown('<div style="color:#28a745;font-weight:600;margin-bottom:1.2rem;">✅ Model loaded successfully</div>', unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
FIELD_CONFIG = {
    'Age':                      (1,  100, 50),
    'Air Pollution':            (1,  10,  5),
    'Alcohol use':              (1,  10,  5),
    'Dust Allergy':             (1,  10,  5),
    'OccuPational Hazards':     (1,  10,  5),
    'Genetic Risk':             (1,  10,  5),
    'chronic Lung Disease':     (1,  10,  5),
    'Balanced Diet':            (1,  10,  5),
    'Obesity':                  (1,  10,  5),
    'Smoking':                  (1,  10,  5),
    'Passive Smoker':           (1,  10,  5),
    'Chest Pain':               (1,  10,  5),
    'Coughing of Blood':        (1,  10,  5),
    'Fatigue':                  (1,  10,  5),
    'Weight Loss':              (1,  10,  5),
    'Shortness of Breath':      (1,  10,  5),
    'Wheezing':                 (1,  10,  5),
    'Swallowing Difficulty':    (1,  10,  5),
    'Clubbing of Finger Nails': (1,  10,  5),
    'Frequent Cold':            (1,  10,  5),
    'Dry Cough':                (1,  10,  5),
    'Snoring':                  (1,  10,  5),
}

GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}

# Group features into logical sections (excluding Gender)
SECTIONS = {
    "👤 Demographics":         ["Age"],
    "🏭 Environmental & Lifestyle": ["Air Pollution", "Alcohol use", "OccuPational Hazards",
                                      "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Dust Allergy"],
    "🧬 Medical History":      ["Genetic Risk", "chronic Lung Disease"],
    "🩺 Symptoms":             ["Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
                                "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
                                "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"],
}

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    user_input = {}
    errors     = {}

    # Gender card
    st.markdown('<div class="section-card"><div class="section-title">⚧ Gender</div>', unsafe_allow_html=True)
    gender_feat = next((f for f in feature_names if f.lower() == 'gender'), None)
    if gender_feat:
        selected_gender = st.radio(
            "Select Gender",
            options=["Female", "Male", "Other"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )
        user_input[gender_feat] = GENDER_MAP[selected_gender]
    st.markdown('</div>', unsafe_allow_html=True)

    # Sectioned numeric fields
    for section_title, section_fields in SECTIONS.items():
        active_fields = [f for f in section_fields if f in feature_names]
        if not active_fields:
            continue

        st.markdown(f'<div class="section-card"><div class="section-title">{section_title}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, feat in enumerate(active_fields):
            mn, mx, default = FIELD_CONFIG.get(feat, (1, 10, 5))
            raw = cols[i % 3].text_input(
                label=f"{feat}  ({mn}–{mx})",
                value=str(default),
                placeholder=f"{mn}–{mx}",
                key=feat
            )
            try:
                val = float(raw)
                if not (mn <= val <= mx):
                    errors[feat] = f"{feat}: must be {mn}–{mx}"
                else:
                    user_input[feat] = val
            except ValueError:
                errors[feat] = f"{feat}: enter a valid number"
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮  Predict Lung Cancer Risk", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    if errors:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.error("⚠️ Please fix the following errors:")
        for msg in errors.values():
            st.caption(f"• {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Handle duplicate/derived columns not shown in the form ────────────
        # Smoking.1 is a duplicate of Smoking created during preprocessing
        if 'Smoking.1' in feature_names and 'Smoking.1' not in user_input:
            user_input['Smoking.1'] = user_input.get('Smoking', 5)

        # Safety fallback: fill any other missing feature with its default
        for feat in feature_names:
            if feat not in user_input:
                mn, mx, default = FIELD_CONFIG.get(feat, (1, 10, 5))
                user_input[feat] = default

        # Reorder input to match training feature order
        input_df = pd.DataFrame([[user_input[f] for f in feature_names]], columns=feature_names)
        input_sc = scaler.transform(input_df)

        pred     = model.predict(input_sc)[0]
        prob     = model.predict_proba(input_sc)[0][1]
        prob_low = 1 - prob

        st.markdown("---")

        # Result banner
        if pred == 1:
            st.markdown(f'<div class="result-high">⚠️ HIGH RISK &nbsp;|&nbsp; Probability: {prob:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low">✅ LOW RISK &nbsp;|&nbsp; Probability of High Risk: {prob:.2%}</div>', unsafe_allow_html=True)

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{"🔴 High" if pred==1 else "🟢 Low"}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{prob:.2%}</div><div class="metric-label">High Risk Probability</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{prob_low:.2%}</div><div class="metric-label">Low Risk Probability</div></div>', unsafe_allow_html=True)
        with c4:
            confidence = max(prob, prob_low)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence:.2%}</div><div class="metric-label">Model Confidence</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk bar
        st.markdown('<div class="section-card"><div class="section-title">📊 Risk Probability Gauge</div>', unsafe_allow_html=True)
        st.progress(float(prob))
        st.caption(f"🔴 High Risk: {prob:.2%}   ·   🟢 Low Risk: {prob_low:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature contributions (Gender excluded)
        if hasattr(model, 'coef_'):
            st.markdown('<div class="section-card"><div class="section-title">📈 Top Contributing Features</div>', unsafe_allow_html=True)

            non_gender_features = [f for f in feature_names if f.lower() != 'gender']
            non_gender_indices  = [feature_names.index(f) for f in non_gender_features]

            coefs_filtered         = model.coef_[0][non_gender_indices]
            input_sc_filtered      = input_sc[0][non_gender_indices]
            contributions_filtered = pd.Series(coefs_filtered * input_sc_filtered, index=non_gender_features)

            top_contrib = contributions_filtered.abs().sort_values(ascending=False).head(10)
            top_vals    = contributions_filtered[top_contrib.index]

            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8f9fa')

            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_vals.values[::-1]]
            bars = ax.barh(top_vals.index[::-1], top_vals.values[::-1],
                           color=colors, edgecolor='white', linewidth=0.5, height=0.6)

            # Value labels on bars
            for bar, val in zip(bars, top_vals.values[::-1]):
                ax.text(
                    val + (0.005 if val >= 0 else -0.005),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va='center', ha='left' if val >= 0 else 'right',
                    fontsize=8.5, color='#333'
                )

            ax.axvline(0, color='#333', linewidth=1)
            ax.set_xlabel("Contribution to Risk Score", fontsize=10, color='#555')
            ax.set_title(
    "Feature Contributions  ·  Red = Increases Risk   Green = Reduces Risk",
    fontsize=11,
    color='#1a1a2e',
    pad=12
)
            ax.tick_params(axis='y', labelsize=9.5)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)