import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set title
st.set_page_config(page_title="Period Cycle Predictor", layout="centered")
st.title("🩺 Period Cycle Predictor")
st.markdown("Enter your last period start date and cycle details to predict your next period start date.")

# File path
data_file = "Menstruation Tracking Data (1).csv"
model_file = "period_model.pkl"

# Load and train model if not saved
if not os.path.exists(model_file):
    df = pd.read_csv(data_file)
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="%Y-%m-%d")
    df["Predicted Next Start Date"] = pd.to_datetime(df["Predicted Next Start Date"], format="%Y-%m-%d")
    df["Cycle Length (Days)"].fillna(df["Cycle Length (Days)"].median(), inplace=True)
    df["Period Duration (Days)"].fillna(df["Period Duration (Days)"].median(), inplace=True)
    df["Prev Start Date"] = df["Start Date"].shift(1)
    df["Days Between Cycles"] = (df["Start Date"] - df["Prev Start Date"]).dt.days
    df = df.dropna(subset=["Predicted Next Start Date"])

    X = df[["Cycle Length (Days)", "Period Duration (Days)", "Days Between Cycles"]]
    y = (df["Predicted Next Start Date"] - df["Start Date"]).dt.days

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(model_file, "wb") as f:
        pickle.dump(model, f)
else:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

# Input fields
st.markdown("### Enter your details:")
prev_date = st.text_input("Previous Period Start Date (dd-mm-yyyy)", value="01-02-2024")
cycle_length = st.number_input("Cycle Length (Days)", min_value=15, max_value=50, value=28)
period_duration = st.number_input("Period Duration (Days)", min_value=2, max_value=10, value=5)
days_between = st.number_input("Days Between Last Two Periods", min_value=15, max_value=60, value=28)

# Predict button
if st.button("🔍 Predict Next Period Date"):
    try:
        prev_date_dt = datetime.strptime(prev_date, "%d-%m-%Y")
        input_data = np.array([[cycle_length, period_duration, days_between]])
        predicted_days = model.predict(input_data)[0]
        next_period_date = prev_date_dt + timedelta(days=round(predicted_days))

        st.success(f"📅 Your next period is expected on **{next_period_date.strftime('%d-%m-%Y')}** (in {round(predicted_days)} days).")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
