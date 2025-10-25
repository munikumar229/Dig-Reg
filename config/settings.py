"""
Configuration settings for the Dig-Reg application
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{API_PORT}")

# Streamlit Configuration
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENTS = {
    "randomforest": "RandomForest-Digits",
    "mlp": "MLP-Digits"
}

# Model Configuration
AVAILABLE_MODELS = ["randomforest", "mlp"]
DEFAULT_MODEL = "randomforest"

# Data Configuration
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Image Processing
IMAGE_SIZE = (8, 8)  # 8x8 pixels for digit recognition
PIXEL_RANGE = (0, 16)  # Similar to sklearn digits dataset

# Training Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25