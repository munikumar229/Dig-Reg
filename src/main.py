import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Define the request schema ---
class DigitsFeatures(BaseModel):
    pixel_0: float
    pixel_1: float
    pixel_2: float
    pixel_3: float
    pixel_4: float
    pixel_5: float
    pixel_6: float
    pixel_7: float
    pixel_8: float
    pixel_9: float
    pixel_10: float
    pixel_11: float
    pixel_12: float
    pixel_13: float
    pixel_14: float
    pixel_15: float
    pixel_16: float
    pixel_17: float
    pixel_18: float
    pixel_19: float
    pixel_20: float
    pixel_21: float
    pixel_22: float
    pixel_23: float
    pixel_24: float
    pixel_25: float
    pixel_26: float
    pixel_27: float
    pixel_28: float
    pixel_29: float
    pixel_30: float
    pixel_31: float
    pixel_32: float
    pixel_33: float
    pixel_34: float
    pixel_35: float
    pixel_36: float
    pixel_37: float
    pixel_38: float
    pixel_39: float
    pixel_40: float
    pixel_41: float
    pixel_42: float
    pixel_43: float
    pixel_44: float
    pixel_45: float
    pixel_46: float
    pixel_47: float
    pixel_48: float
    pixel_49: float
    pixel_50: float
    pixel_51: float
    pixel_52: float
    pixel_53: float
    pixel_54: float
    pixel_55: float
    pixel_56: float
    pixel_57: float
    pixel_58: float
    pixel_59: float
    pixel_60: float
    pixel_61: float
    pixel_62: float
    pixel_63: float


# --- Load latest MLflow model ---
def load_latest_model():
    mlflow.set_tracking_uri("http://localhost:5000")

    experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
    if experiment is None:
        raise RuntimeError("Experiment not found. Make sure you trained and logged a model first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )
    if runs.empty:
        raise RuntimeError("No runs found in experiment.")

    latest_run_id = runs.iloc[0]["run_id"]

    # ✅ use the correct artifact path from train.py
    model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
    print(f"Loading model from: {model_uri}")

    return mlflow.sklearn.load_model(model_uri)


# Load once at startup
model = load_latest_model()

# --- FastAPI App ---
app = FastAPI(title="Digits Classification API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Digits Classification API!"}

@app.post("/predict/")
def predict(features: DigitsFeatures):
    """
    Accepts 64 pixel values and returns a digit class prediction.
    """
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)
    return {"predicted_digit": int(prediction[0])}
