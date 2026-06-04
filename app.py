# ─── Patient No-Show Risk Predictor ──────────────────────────────────────
# Upskilling in AI: Week 6 project — Phase 1
# Built by Shivani B | Life Sciences domain | Streamlit + Scikit-learn
# ─────────────────────────────────────────────────────────────────────────

import streamlit as st
import pickle
import joblib 
import numpy as np


# ── Load the trained model ────────────────────────────────────────────────
# model.pkl must be in the same folder as app.py
#with open("model.pkl", "rb") as f:
    #model = pickle.load(f)
    model = joblib.load("model.pkl")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient No-Show Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("🏥 Patient No-Show Risk Predictor")
st.caption("Life Sciences · Trained on 110,000 real patient appointments · Built by Shivani using Claude and Scikit-learn"  )
st.markdown("---")
st.markdown("Fill in the patient details below to predict the likelihood of a no-show.")

# ── Input form — two columns ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Details")
    age = st.slider("Patient Age", min_value=0, max_value=100, value=35,
                    help="Patient age in years")
    gender = st.selectbox("Gender", ["Female", "Male"])
    days_wait = st.slider("Days between booking and appointment",
                           min_value=0, max_value=90, value=7,
                           help="Key predictor: longer wait = higher no-show risk")

with col2:
    st.subheader("Medical & Notification Flags")
    sms = st.checkbox("SMS reminder sent", value=True)
    scholarship = st.checkbox("Enrolled in welfare / scholarship programme")
    hypertension = st.checkbox("Has hypertension")
    diabetes = st.checkbox("Has diabetes")
    alcoholism = st.checkbox("Alcoholism flag")

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────
if st.button("🔍 Predict No-Show Risk", use_container_width=True):

    # Build the feature array — must match exact order used during training:
    # ['Age','Gender_enc','Scholarship','Hipertension','Diabetes','Alcoholism','SMS_received','days_wait']
    features = np.array([[
        age,
        1 if gender == "Female" else 0,
        int(scholarship),
        int(hypertension),
        int(diabetes),
        int(alcoholism),
        int(sms),
        days_wait
    ]])

    # Get probability of no-show (class 1)
    prob = model.predict_proba(features)[0][1]
    risk = "HIGH" if prob > 0.35 else "LOW"

    # ── Display result ────────────────────────────────────────────────────
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if risk == "HIGH":
            st.error(f"🔴 No-Show Risk: HIGH\n\nProbability: {prob:.0%}")
        else:
            st.success(f"🟢 No-Show Risk: LOW\n\nProbability: {prob:.0%}")

    with result_col2:
        st.metric(label="No-Show Probability", value=f"{prob:.0%}")

    # ── Recommendation ────────────────────────────────────────────────────
    st.markdown("### Recommended Action")
    if risk == "HIGH":
        st.warning("⚠️ High risk patient. Recommend: Send additional reminder call 48hrs before appointment. Consider overbooking this slot.")
    else:
        st.info("✅ Low risk. Standard appointment management applies.")

    # ── Key insight callout ───────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"💡 Key model insight: Wait time ({days_wait} days) is the strongest predictor of no-show — stronger than SMS reminders alone. Model trained on 110,527 real Brazilian patient appointments.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Shivani · Patient No-Show Predictor · Phase 1 Project · AI Upskilling 2026")