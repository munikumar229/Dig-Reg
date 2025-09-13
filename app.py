import streamlit as st
import pandas as pd
import mlflow
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

# --- Load the latest model from MLflow for Digits dataset ---
def load_latest_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
    if experiment is None:
        st.error("MLflow experiment not found.")
        st.stop()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        st.error("No runs found in MLflow experiment.")
        st.stop()
    latest_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{latest_run_id}/sklearn-model"
    return mlflow.sklearn.load_model(model_uri)

model = load_latest_model()

st.title("Digits Classification App")
st.write("Draw a digit (0-9) in the box below. The image will be downsampled to 8x8 and predicted.")

# --- Drawing canvas for user input ---
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Black
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((8, 8), Image.LANCZOS)
    img_arr = np.array(img)
    img_arr = (16 - (img_arr / 255.0) * 16).astype(int)
    flat_pixels = img_arr.flatten()
    st.write("Preview (8x8):")
    st.dataframe(img_arr)
    if st.button("Predict Digit"):
        input_dict = {f"pixel_{i}": flat_pixels[i] for i in range(64)}
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)
        st.success(f"Predicted Digit: {int(prediction[0])}")