import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model1.pkl')

st.set_page_config(page_title="Prediction App", layout="centered")
st.title("🚀 Predictive Model App")

# Input fields
temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=60.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, step=0.1)
rain = st.number_input("Rain (mm)", min_value=0.0, max_value=500.0, step=0.1)

public_holiday = st.selectbox("Public Holiday", [0, 1])
weekly_holiday = st.selectbox("Weekly Holiday", [0, 1])
festival = st.selectbox("Festival", [0, 1])

low_dev = st.slider("Low Development Area (%)", 0, 100, 0)
med_dev = st.slider("Medium Development Area (%)", 0, 100, 0)
high_dev = st.slider("High Development Area (%)", 0, 100, 0)

year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)

time_in_hours = st.slider("Time in Hours", 0.0, 24.0, 12.0, step=0.5)

red_low = st.checkbox("RED_Low")
red_med = st.checkbox("RED_Medium")

# Convert boolean to int
red_low = int(red_low)
red_med = int(red_med)

if st.button("Predict"):
    input_df = pd.DataFrame([[
        temperature, humidity, wind_speed, rain,
        public_holiday, weekly_holiday, festival,
        low_dev, med_dev, high_dev,
        year, month, day, time_in_hours,
        red_low, red_med
    ]], columns=[
        "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Rain (mm)",
        "Public Holiday", "Weekly Holiday", "Festival",
        "Low Development Area (%)", "Medium Development Area (%)", "High Development Area (%)",
        "year", "month", "day", "Time_in_hours",
        "RED_Low", "RED_Medium"
    ])

    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Predicted Value: {prediction}")
