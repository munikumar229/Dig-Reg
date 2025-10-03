import streamlit as st
import pandas as pd
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
import mlflow

st.set_page_config(page_title="Digits Classifier", page_icon="🖌️")

# --- Load model (modern MLflow) ---
def load_latest_model():
    """
    Load the latest trained model from MLflow.
    Tries Production model first, else fallback to latest run artifact.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # or your MLflow server

    # Try to load Production model
    try:
        model_name = "DigitsClassifier"
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        st.info(f"Loaded Production model: {model_uri}")
        return model
    except Exception as e:
        st.warning(f"Production model not found. Loading latest run artifact.\nReason: {e}")
        experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
        if experiment is None:
            st.error("No experiment found with name 'RandomForest-Digits'")
            st.stop()

        runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            st.error("No runs found in the experiment")
            st.stop()

        latest_run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
        st.info(f"Loading latest run model: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)

# Load once at startup
model = load_latest_model()

st.title("🖌️ Digits Classification App")
st.write("Draw a digit (0–9) and click Predict!")

# --- Session state for clearing canvas ---
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# --- Drawing canvas ---
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="#222",
    background_color="white",
    height=256,
    width=256,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
    update_streamlit=True,
)

# --- Buttons ---
col1, col2 = st.columns([1,1])
predict_btn = col1.button("Predict")
clear_btn = col2.button("Clear")

# --- Clear canvas ---
if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

# --- Prediction logic ---
if predict_btn:
    if canvas_result.image_data is not None:
        # Convert RGBA to grayscale
        img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        # Resize to 8x8 (like sklearn digits dataset)
        img_resized = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
        img_resized = 255 - img_resized  # invert colors

        # Scale pixels to 0–16
        scaler = MinMaxScaler((0,16))
        img_scaled = scaler.fit_transform(img_resized)
        input_data = img_scaled.flatten().reshape(1,-1)

        # Column names matching training data
        feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Predict
        prediction = model.predict(input_df)
        st.success(f"Predicted Digit: {int(prediction[0])}")
        st.image(img_resized, caption="Processed 8x8 Input", width=150)
    else:
        st.error("Please draw a digit before predicting.")
