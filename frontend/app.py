import streamlit as st
import pandas as pd
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler

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
# API Configuration
# -------------------------------
from components.api_client import APIClient, preprocess_image_for_api

# Initialize API client
api_client = APIClient()

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
# Model Selection & User Input Options
# -------------------------------
st.sidebar.header("Model Selection")
selected_model_type = st.sidebar.selectbox(
    "Choose Model Type:",
    ["randomforest", "mlp"],
    help="Select the machine learning model to use for prediction"
)

# Check backend health
if not api_client.check_health():
    st.error(f"⚠️ Backend API is not available at {api_client.base_url}")
    st.error("Please ensure the FastAPI backend is running.")
    st.code("docker-compose up backend")
    st.stop()

# Get available models from backend
available_models = api_client.get_available_models()
st.sidebar.success(f"✅ Connected to API at {api_client.base_url}")

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
predict_btn = col1.button(" Predict Digit")
clear_btn = col2.button("Clear")

if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

# -------------------------------
# Prediction Logic
# -------------------------------
if predict_btn and input_img is not None:
    # Preprocess image for API call
    try:
        pixels, img_resized = preprocess_image_for_api(input_img)
        
        # Make prediction via API
        with st.spinner(f"Making prediction using {selected_model_type.title()} model..."):
            api_result = api_client.predict(pixels, selected_model_type)
        
        prediction = api_result["predicted_digit"]
        confidence = api_result["confidence"]
        probs = api_result["probabilities"]
        model_used = api_result["model_used"]
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.error("Please check if the backend API is running and accessible.")
        st.stop()

    # Save history
    st.session_state.history.append({
        "digit": int(prediction), 
        "model": model_used.title(),
        "confidence": float(confidence)
    })

    # Display Prediction
    st.success(f" Predicted Digit: {int(prediction)} ")
    st.image(img_resized, caption="Processed 8x8 Input", width=150)

    # Display Probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probs})
    st.bar_chart(prob_df.set_index("Digit"))

    # Display History
    st.subheader("Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

# -------------------------------
# Model Info
# -------------------------------
st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Current Model:** {selected_model_type.title()}")

# Display available models from API
st.sidebar.write("**Available Models:**")
for model in available_models:
    st.sidebar.write(f"• {model.title()}: ✅ Available")

st.sidebar.write("**Input:** 8x8 grayscale image (0-16 scale)")
st.sidebar.write("**Backend:** FastAPI + MLflow")

# -------------------------------
# API Information
# -------------------------------
st.sidebar.subheader("API Information")
st.sidebar.write(f"**Backend URL:** {api_client.base_url}")
st.sidebar.write("**Architecture:** Frontend ↔ API ↔ MLflow")

# Command line instructions for training (models are loaded by backend)
st.sidebar.subheader("Model Training")
st.sidebar.write("Models are managed by the backend service.")
st.sidebar.code("# Train models on backend:\npython src/train.py --model randomforest\npython src/train.py --model mlp")
