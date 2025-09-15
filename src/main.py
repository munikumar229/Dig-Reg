import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Define the request schema ---
class DigitsFeatures(BaseModel):
    pixel_0_0: float
    pixel_0_1: float
    pixel_0_2: float
    pixel_0_3: float
    pixel_0_4: float
    pixel_0_5: float
    pixel_0_6: float
    pixel_0_7: float
    pixel_1_0: float
    pixel_1_1: float
    pixel_1_2: float
    pixel_1_3: float
    pixel_1_4: float
    pixel_1_5: float
    pixel_1_6: float
    pixel_1_7: float
    pixel_2_0: float
    pixel_2_1: float
    pixel_2_2: float
    pixel_2_3: float
    pixel_2_4: float
    pixel_2_5: float
    pixel_2_6: float
    pixel_2_7: float
    pixel_3_0: float
    pixel_3_1: float
    pixel_3_2: float
    pixel_3_3: float
    pixel_3_4: float
    pixel_3_5: float
    pixel_3_6: float
    pixel_3_7: float
    pixel_4_0: float
    pixel_4_1: float
    pixel_4_2: float
    pixel_4_3: float
    pixel_4_4: float
    pixel_4_5: float
    pixel_4_6: float
    pixel_4_7: float
    pixel_5_0: float
    pixel_5_1: float
    pixel_5_2: float
    pixel_5_3: float
    pixel_5_4: float
    pixel_5_5: float
    pixel_5_6: float
    pixel_5_7: float
    pixel_6_0: float
    pixel_6_1: float
    pixel_6_2: float
    pixel_6_3: float
    pixel_6_4: float
    pixel_6_5: float
    pixel_6_6: float
    pixel_6_7: float
    pixel_7_0: float
    pixel_7_1: float
    pixel_7_2: float
    pixel_7_3: float
    pixel_7_4: float
    pixel_7_5: float
    pixel_7_6: float
    pixel_7_7: float


# --- Load latest MLflow model ---
# def load_latest_model():
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")

#     experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
#     if experiment is None:
#         raise RuntimeError("Experiment not found. Make sure you trained and logged a model first.")

#     runs = mlflow.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["attributes.start_time DESC"]
#     )
#     if runs.empty:
#         raise RuntimeError("No runs found in experiment.")

#     latest_run_id = runs.iloc[0]["run_id"]

#     # ✅ use the correct artifact path from train.py
#     model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
#     print(f"Loading model from: {model_uri}")

#     return mlflow.sklearn.load_model(model_uri)
def load_latest_model():
    
# Point to the MLflow tracking server, same as in train.py
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Search for runs in the default experiment (experiment_id='0')
    runs = mlflow.search_runs(experiment_ids=['0'])
    
    # Get the latest run's ID
    latest_run_id = runs.iloc[0]['run_id']
    
    # Construct the model URI
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
