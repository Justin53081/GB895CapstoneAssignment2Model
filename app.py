
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("churn_rf_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

age               = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level      = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education         = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type       = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=5, value=3)
num_active_qtrs = st.number_input("Number of Active 2022 Quarters", min_value=0, max_value=4, value=2)
total_session_length = st.number_input("Total Session Length 2022", min_value=0, max_value=15601, value=1800)
total_num_sessions = st.number_input("Total Number of Sessions 2022", min_value=0, max_value=262, value=40)
num_activity_days = st.number_input("Number of Activity Days 2022", min_value=0, max_value=5, value=2)
subscription_time_2022 = st.number_input("Subscription Time in 2022 (Days)", min_value=1, max_value=364, value=180)
subscription_time_2023 = st.number_input("Subscription Time in 2023 (Days)", min_value=1, max_value=364, value=180)
session_length_per_day = st.number_input("Session Length Per Day 2022", min_value=0.0, max_value=11786.0, value=25.0)
avg_sessions_per_qtr = st.number_input("Avg. Sessions Per Active Quarter", min_value=0.0, max_value=122.0, value=19.0)


if st.button("Predict"):

    # Build categorical DataFrame — column names must match encoder exactly
    # The order of columns here must match the order of categorical_features used during encoder fitting.
    raw = pd.DataFrame([{
        'EDUCATION':    education,
        'INCOME_LEVEL': income_level,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
        'AGE':                age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
        'NUMACTIVE2022QTRS': num_active_qtrs,
        'TOTALSESSIONLENGTH2022': total_session_length,
        'TOTALNUMSESSIONS2022': total_num_sessions,
        'NUMACTIVITYDAYS2022': num_activity_days,
        'SUBSCRIPTIONTIMEIN2022': subscription_time_2022,
        'SUBSCRIPTIONTIMEIN2023': subscription_time_2023,
        'SESSIONLENGTHPERDAY2022': session_length_per_day,
        'AVG_SESSIONSPERACTIVEQTR': avg_sessions_per_qtr
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
