from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("model1.pkl")

# API key setup
API_KEY = "my-secret-key"  # Change to a secure key
API_KEY_NAME = "access-token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Verify key
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    rain: float
    public_holiday: int
    weekly_holiday: int
    festival: int
    low_dev: int
    med_dev: int
    high_dev: int
    year: int
    month: int
    day: int
    time_in_hours: float
    red_low: int
    red_med: int


@app.post("/predict")
def predict(data: InputData, api_key: str = Depends(verify_api_key)):
    df = pd.DataFrame([[
        data.temperature, data.humidity, data.wind_speed, data.rain,
        data.public_holiday, data.weekly_holiday, data.festival,
        data.low_dev, data.med_dev, data.high_dev,
        data.year, data.month, data.day, data.time_in_hours,
        data.red_low, data.red_med
    ]], columns=[
        "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Rain (mm)",
        "Public Holiday", "Weekly Holiday", "Festival",
        "Low Development Area (%)", "Medium Development Area (%)", "High Development Area (%)",
        "year", "month", "day", "Time_in_hours",
        "RED_Low", "RED_Medium"
    ])

    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
