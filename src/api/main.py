from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse
import mlflow.pyfunc
import numpy as np
import pandas as pd
import os

app = FastAPI()

# Load the best model from MLflow registry (for now, load from local path)
MODEL_PATH = "mlruns/288870226930422180/models/m-9e119381963b44f89e3eaa6c429cb0c2/artifacts/"
model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert request to DataFrame
    input_data = pd.DataFrame([request.dict()])
    # Predict risk probability (assume binary classification, get proba for class 1)
    proba = model.predict_proba(input_data)[0][1]
    return PredictResponse(risk_probability=float(proba))
