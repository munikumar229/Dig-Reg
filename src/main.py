import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

class DigitsFeatures(BaseModel):
    # 64 pixel fields...
    pixel_0_0: float
    pixel_0_1: float
    # ... all the way to pixel_7_7

def load_latest_model():
    """
    Load the latest MLflow run artifact if Production model/stages are not available.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # or your MLflow server

    # Try to load Production model (if your MLflow supports it)
    try:
        # modern way: load by model name if registered
        model_name = "DigitsClassifier"
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded Production model: {model_uri}")
        return model
    except Exception as e:
        print(f"Production model not found. Loading latest run artifact instead.\nReason: {e}")
        # fallback: find the latest run and load model artifact
        experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
        if experiment is None:
            raise RuntimeError("No experiment found with name 'RandomForest-Digits'")

        runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            raise RuntimeError("No runs found in experiment")

        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
        print(f"Loading latest run artifact: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)

# Load once at startup
model = load_latest_model()

# --- FastAPI ---
app = FastAPI(title="Digits Classification API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Digits Classification API!"}

@app.post("/predict/")
def predict(features: DigitsFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)
    return {"predicted_digit": int(prediction[0])}
