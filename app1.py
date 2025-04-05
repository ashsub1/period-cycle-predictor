import streamlit as st
import pickle
import numpy as np
from datetime import datetime, timedelta

# Load model
model = pickle.load(open("period_model.pkl", "rb"))

st.set_page_config(page_title="Period Cycle Prediction Web", layout="centered")
st.title("🩸 Period Cycle Predictor Web")

st.markdown("""
Enter the details of your last period and cycle information to predict your next period start date.
""")

# User inputs
prev_start = st.text_input("Previous Period Start Date (dd-mm-yyyy)", value="01-02-2024")
cycle_len = st.number_input("Cycle Length (Days)", min_value=15, max_value=60, value=28)
period_dur = st.number_input("Period Duration (Days)", min_value=1, max_value=10, value=5)
days_btwn = st.number_input("Days Between Last Two Periods", min_value=15, max_value=60, value=28)

# Prediction button
if st.button("🔍 Predict Next Period Date"):
    try:
        prev_start_date = datetime.strptime(prev_start, "%d-%m-%Y")
        input_data = np.array([[cycle_len, period_dur, days_btwn]])
        predicted_days = model.predict(input_data)[0]

        next_period_date = prev_start_date + timedelta(days=round(predicted_days))

        st.success(f"📅 Your next period is expected on **{next_period_date.strftime('%d-%m-%Y')}** "
                   f"(in {round(predicted_days)} days).")
    except ValueError:
        st.error("❌ Please enter the date in DD-MM-YYYY format.")
