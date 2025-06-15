

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import numpy as np
import dill

# Security setup
API_KEY = "my-secret-key"
API_KEY_NAME = "access-token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    print(f"🔐 Incoming API key: {api_key}")  # Debugging line
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("model1.pkl", "rb") as f:
    model = dill.load(f)

# Input schema
class InputData(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Rain: float
    Public_Holiday: int
    Weekly_Holiday: int
    Festival: int
    Low_Development: float
    Medium_Development: float
    High_Development: float
    year: int
    month: int
    day: int
    Time_in_hours: int
    RED_Low: bool
    RED_Medium: bool

# Prediction route
@app.post("/predict")
def predict(data: InputData, api_key: str = Depends(verify_api_key)):
    features = np.array([[
        data.Temperature,
        data.Humidity,
        data.Wind_Speed,
        data.Rain,
        data.Public_Holiday,
        data.Weekly_Holiday,
        data.Festival,
        data.Low_Development,
        data.Medium_Development,
        data.High_Development,
        data.year,
        data.month,
        data.day,
        data.Time_in_hours,
        int(data.RED_Low),
        int(data.RED_Medium)
    ]])

    prediction = model.predict(features)
    return {"prediction": prediction[0]}

