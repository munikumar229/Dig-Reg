import streamlit as st
import pandas as pd
import mlflow
import numpy as np
# If you see "Import 'streamlit_drawable_canvas' could not be resolved", install it with:
#   pip install streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
import cv2

st.set_page_config(
	page_title="Digits Classification App",
	page_icon="🖌️",
	layout="centered",
	initial_sidebar_state="auto",
	menu_items={
		'Get Help': 'https://docs.streamlit.io/',
		'Report a bug': 'https://github.com/streamlit/streamlit/issues',
		'About': "A beautiful ML digits classifier demo."
	}
)


# --- Load latest model ---
def load_latest_model():
	# mlflow.set_tracking_uri("http://localhost:5000")
	# mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

# --- Custom CSS for better look ---
st.markdown(
	"""
	<style>
	.main {
		background-color: #f7f7fa;
	}
	.stButton > button {
		color: white;
		background: linear-gradient(90deg, #4f8bf9 0%, #235390 100%);
		border-radius: 8px;
		font-size: 1.1em;
		padding: 0.5em 1.5em;
		margin: 0.2em 0.5em;
	}
	.stButton > button:hover {
		background: linear-gradient(90deg, #235390 0%, #4f8bf9 100%);
	}
	.st-cb {
		background: #fff;
		border-radius: 12px;
		box-shadow: 0 2px 8px rgba(0,0,0,0.07);
		padding: 1.5em 2em;
	}
	.stAlert {
		border-radius: 8px;
	}
	</style>
	""",
	unsafe_allow_html=True
)

st.title("🖌️ <span style='color:#235390'>Digits Classification App</span>")
st.markdown(
	"<p style='font-size:1.2em; color:#444;'>Draw a digit (0–9) below and let the model predict it!</p>",
	unsafe_allow_html=True
)

# --- Session state for clearing ---
if "clear" not in st.session_state:
	st.session_state.clear = False

st.markdown("---")

# --- Drawing canvas in a card ---
with st.container():
	st.markdown("<h4 style='color:#4f8bf9;'>Draw Here:</h4>", unsafe_allow_html=True)
	canvas_result = st_canvas(
		fill_color="white",
		stroke_width=12,
		stroke_color="#222",
		background_color="white",
		height=256,
		width=256,
		drawing_mode="freedraw",
		key="canvas",
		update_streamlit=True,
	)

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
	predict_btn = st.button("✅ Predict Digit", use_container_width=True)
with col2:
	clear_btn = st.button("🧹 Clear Drawing", use_container_width=True)

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
		feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
		input_df = pd.DataFrame(input_data, columns=feature_names)

		# Predict digit
		prediction = model.predict(input_df)

		st.success(f"🎯 <span style='font-size:1.3em;'>Predicted Digit: <span style='color:#4f8bf9'>{int(prediction[0])}</span></span>", icon="🎯")

		# Show processed image
		st.image(img_resized, caption="Processed 8x8 Input", width=150)
	else:
		st.error("Please draw a digit before predicting.")
