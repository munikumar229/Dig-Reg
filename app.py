import streamlit as st
import pandas as pd
import mlflow
import numpy as np
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
import cv2


# --- Load latest model ---
def load_latest_model():
	mlflow.set_tracking_uri("http://localhost:5000")
	experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
	if experiment is None:
		st.error("MLflow experiment 'RandomForest-Digits' not found.")
		st.stop()

	runs = mlflow.search_runs(
		experiment_ids=[experiment.experiment_id],
		order_by=["start_time DESC"]
	)
	if runs.empty:
		st.error("No runs found in MLflow experiment.")
		st.stop()

	latest_run_id = runs.iloc[0]['run_id']
	model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
	return mlflow.sklearn.load_model(model_uri)


# Load the trained MLflow model
model = load_latest_model()

# --- Streamlit UI ---
st.title("🖌️ Digits Classification App")
st.write("Draw a digit (0–9) below and let the model predict it!")

# --- Session state for clearing ---
if "clear" not in st.session_state:
	st.session_state.clear = False

# --- Drawing canvas ---
canvas_result = st_canvas(
	fill_color="white",
	stroke_width=10,
	stroke_color="black",
	background_color="white",
	height=256,
	width=256,
	drawing_mode="freedraw",
	key="canvas",
	update_streamlit=True,
)

col1, col2 = st.columns(2)
with col1:
	predict_btn = st.button("✅ Predict Digit")
with col2:
	clear_btn = st.button("🧹 Clear Drawing")

# --- Clear the canvas ---
if clear_btn:
	st.experimental_rerun()

# --- Prediction logic ---
if predict_btn:
	if canvas_result.image_data is not None:
		# Convert RGBA -> Grayscale
		img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

		# Resize to 8x8 (like sklearn digits dataset)
		img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)

		# Invert colors (model expects black digit on white background)
		img_resized = 255 - img_resized

		# Scale pixel values to 0–16
		scaler = MinMaxScaler((0, 16))
		img_scaled = scaler.fit_transform(img_resized)
		# Flatten into 64 features
		input_data = img_scaled.flatten().reshape(1, -1)

	# Match training column names
		feature_names = [f"pixel_{i}" for i in range(64)]
		input_df = pd.DataFrame(input_data, columns=feature_names)

		# Predict digit
		prediction = model.predict(input_df)
		
		st.success(f"🎯 Predicted Digit: {int(prediction[0])}")

		# Show processed image
		st.image(img_resized, caption="Processed 8x8 Input", width=150)
	else:
		st.error("Please draw a digit before predicting.")
