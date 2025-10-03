import streamlit as st
import pandas as pd
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
import mlflow

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="🖌️ Digits Classifier",
    page_icon="🖌️",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f7fa; }
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
    </style>
    """, unsafe_allow_html=True
)

# -------------------------------
# Load ML Model
# -------------------------------
@st.cache_resource
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.get_experiment_by_name("RandomForest-Digits")
    if experiment is None:
        st.error("MLflow experiment 'RandomForest-Digits' not found.")
        st.stop()
    runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
    latest_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

model = load_latest_model()

# -------------------------------
# App Title & Description
# -------------------------------
st.title("🖌️ Digits Classification App")
st.write("Draw a digit (0–9) to predict the digit!")

# -------------------------------
# Session State
# -------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input Options
# -------------------------------
st.sidebar.header("Input Options")
mode = st.sidebar.radio("Select Input Mode:", ["Draw Digit"])

stroke_width = st.sidebar.slider("Stroke Width", 1, 30, 12)
stroke_color = st.sidebar.color_picker("Stroke Color", "#000000")

# -------------------------------
# Canvas Drawing
# -------------------------------
input_img = None

if mode == "Draw Digit":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="white",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        update_streamlit=True,
    )
    if canvas_result.image_data is not None:
        input_img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

# elif mode == "Upload Image":
#     uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
#     if uploaded_file is not None:
#         file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
#         input_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#         st.image(input_img, caption="Uploaded Image", width=150)

# elif mode == "Webcam":
#     webcam_capture = st.camera_input("Capture a digit with your camera")
#     if webcam_capture is not None:
#         file_bytes = np.frombuffer(webcam_capture.read(), np.uint8)
#         input_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#         st.image(input_img, caption="Captured Image", width=150)

# -------------------------------
# Clear Button
# -------------------------------
col1, col2 = st.columns([1, 1])
predict_btn = col1.button("✅ Predict Digit")
clear_btn = col2.button("🧹 Clear")

if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

# -------------------------------
# Prediction Logic
# -------------------------------
def preprocess_image(img):
    img_resized = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
    img_resized = 255 - img_resized
    scaler = MinMaxScaler((0,16))
    img_scaled = scaler.fit_transform(img_resized)
    input_data = img_scaled.flatten().reshape(1,-1)
    feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
    return pd.DataFrame(input_data, columns=feature_names), img_resized

if predict_btn and input_img is not None:
    input_df, img_resized = preprocess_image(input_img)
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    # Save history
    st.session_state.history.append(int(prediction))

    # Display Prediction
    st.success(f"🎯 Predicted Digit: {int(prediction)}")
    st.image(img_resized, caption="Processed 8x8 Input", width=150)

    # Display Probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probs})
    st.bar_chart(prob_df.set_index("Digit"))

    # Display History
    st.subheader("Prediction History")
    st.table(pd.DataFrame(st.session_state.history, columns=["Predicted Digit"]))

# -------------------------------
# Model Info
# -------------------------------
st.sidebar.subheader("Model Info")
st.sidebar.write("RandomForestClassifier trained on sklearn digits dataset")
st.sidebar.write("Accuracy: ~97.5%")
st.sidebar.write("Input: 8x8 grayscale image (0-16 scale)")
