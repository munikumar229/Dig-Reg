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
def load_model_by_type(model_type):
    import pickle
    import tempfile
    import os
    from sklearn.preprocessing import StandardScaler
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = f"{model_type.title()}-Digits"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        st.error(f"MLflow experiment '{experiment_name}' not found. Please train the model first.")
        return None
    runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        st.error(f"No runs found for experiment '{experiment_name}'. Please train the model first.")
        return None
    latest_run_id = runs.iloc[0]['run_id']
    
    # Determine artifact path based on model type
    if model_type.lower() == "randomforest":
        artifact_path = "random-forest-model"
    elif model_type.lower() == "mlp":
        artifact_path = "mlp-model"
    else:
        st.error(f"Unsupported model type: {model_type}")
        return None
    
    model_uri = f"runs:/{latest_run_id}/{artifact_path}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        run_data = runs.iloc[0]
        
        # Check if this model uses a scaler (MLP models)
        scaler = None
        if model_type.lower() == "mlp" and run_data.get('params.uses_scaler') == 'True':
            try:
                # Download the scaler artifact
                scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{latest_run_id}/scaler.pkl")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load scaler, creating new one: {e}")
                scaler = StandardScaler()
        
        return {"model": model, "scaler": scaler}, run_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

# Load selected model
model_data = load_model_by_type(selected_model_type)
if model_data is None:
    st.error("Please train a model first using the training script.")
    st.code(f"python src/train.py --model {selected_model_type}")
    st.stop()

model_info, run_info = model_data
model = model_info["model"]
scaler = model_info["scaler"]

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
def preprocess_image(img, model_type="randomforest", model_scaler=None):
    img_resized = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
    img_resized = 255 - img_resized
    
    if model_type.lower() == "mlp" and model_scaler is not None:
        # For MLP, flatten first then use the trained scaler
        img_flattened = img_resized.flatten().reshape(1, -1)
        img_scaled = model_scaler.transform(img_flattened)
        # Return as numpy array for MLP (no feature names to avoid warnings)
        return img_scaled, img_resized
    else:
        # For RandomForest, use MinMaxScaler as before
        scaler_rf = MinMaxScaler((0,16))
        img_scaled = scaler_rf.fit_transform(img_resized)
        input_data = img_scaled.flatten().reshape(1,-1)
        feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
        return pd.DataFrame(input_data, columns=feature_names), img_resized

if predict_btn and input_img is not None:
    input_data, img_resized = preprocess_image(input_img, selected_model_type, scaler)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Input data shape: {input_data.shape if hasattr(input_data, 'shape') else 'Unknown'}")
        st.error(f"Model type: {type(model)}")
        st.stop()

    # Save history
    st.session_state.history.append({
        "digit": int(prediction), 
        "model": selected_model_type.title(),
        "confidence": float(max(probs))
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

# Check if models exist
def check_model_exists(model_type):
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        experiment_name = f"{model_type.title()}-Digits"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return False
        runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        return not runs.empty
    except:
        return False

# Display model availability
rf_exists = check_model_exists("randomforest")
mlp_exists = check_model_exists("mlp")

st.sidebar.write("**Model Status:**")
st.sidebar.write(f"• RandomForest: {' Trained' if rf_exists else '❌ Not trained'}")
st.sidebar.write(f"• MLP: {' Trained' if mlp_exists else '❌ Not trained'}")

if run_info is not None:
    st.sidebar.write(f"**Accuracy:** {run_info.get('metrics.accuracy', 'N/A'):.4f}" if 'metrics.accuracy' in run_info else "**Accuracy:** N/A")
    st.sidebar.write(f"**F1 Score:** {run_info.get('metrics.f1_score', 'N/A'):.4f}" if 'metrics.f1_score' in run_info else "**F1 Score:** N/A")

st.sidebar.write("**Input:** 8x8 grayscale image (0-16 scale)")

# Training instructions and buttons
st.sidebar.subheader("Training New Models")

# Check if training data exists
import os
data_exists = (
    os.path.exists("/home/hp/Desktop/Projects/Dig-Reg/data/processed/train.csv") and
    os.path.exists("/home/hp/Desktop/Projects/Dig-Reg/data/processed/val.csv")
)

if not data_exists:
    st.sidebar.warning(" Training data not found! Please run data preprocessing first.")
    if st.sidebar.button("📊 Preprocess Data", help="Run data preprocessing script"):
        try:
            import subprocess
            result = subprocess.run([
                "/home/hp/Desktop/Projects/Dig-Reg/venv/bin/python", 
                "src/process_data.py"
            ], capture_output=True, text=True, cwd="/home/hp/Desktop/Projects/Dig-Reg")
            
            if result.returncode == 0:
                st.sidebar.success(" Data preprocessing completed!")
                st.rerun()
            else:
                st.sidebar.error("❌ Data preprocessing failed!")
                with st.expander("🔍 Error Details"):
                    st.code(result.stderr)
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
    st.sidebar.divider()

# Training buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button(" Train RandomForest", help="Train a new RandomForest model"):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            import subprocess
            import sys
            
            status_text.text("🔄 Starting RandomForest training...")
            progress_bar.progress(25)
            
            result = subprocess.run([
                "/home/hp/Desktop/Projects/Dig-Reg/venv/bin/python", 
                "src/train.py", 
                "--model", 
                "randomforest"
            ], capture_output=True, text=True, cwd="/home/hp/Desktop/Projects/Dig-Reg")
            
            progress_bar.progress(75)
            
            if result.returncode == 0:
                progress_bar.progress(100)
                status_text.text(" RandomForest training completed!")
                st.success(" RandomForest model trained successfully!")
                st.balloons()  # 🎈 Celebration for successful training!
                
                # Show training output in expandable section
                if result.stdout:
                    with st.expander("📋 Training Details"):
                        st.code(result.stdout[-1000:])  # Show last 1000 chars
                
                st.cache_resource.clear()  # Clear cache to reload model
                st.rerun()
            else:
                status_text.text("❌ Training failed!")
                st.error("❌ RandomForest training failed!")
                with st.expander("🔍 Error Details"):
                    st.code(result.stderr)
                    if result.stdout:
                        st.code(result.stdout)
        except Exception as e:
            status_text.text("❌ Error occurred!")
            st.error(f"❌ Error training model: {e}")
        finally:
            # Clean up progress indicators after a delay
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

with col2:
    if st.button(" Train MLP", help="Train a new MLP (Neural Network) model"):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            import subprocess
            import sys
            
            status_text.text("🔄 Starting MLP training...")
            progress_bar.progress(25)
            
            result = subprocess.run([
                "/home/hp/Desktop/Projects/Dig-Reg/venv/bin/python", 
                "src/train.py", 
                "--model", 
                "mlp"
            ], capture_output=True, text=True, cwd="/home/hp/Desktop/Projects/Dig-Reg")
            
            progress_bar.progress(75)
            
            if result.returncode == 0:
                progress_bar.progress(100)
                status_text.text("✅ MLP training completed!")
                st.success("✅ MLP model trained successfully!")
                st.balloons()  # 🎈 Celebration for successful training!
                
                # Show training output in expandable section
                if result.stdout:
                    with st.expander("📋 Training Details"):
                        st.code(result.stdout[-1000:])  # Show last 1000 chars
                
                st.cache_resource.clear()  # Clear cache to reload model
                st.rerun()
            else:
                status_text.text("❌ Training failed!")
                st.error("❌ MLP training failed!")
                with st.expander("🔍 Error Details"):
                    st.code(result.stderr)
                    if result.stdout:
                        st.code(result.stdout)
        except Exception as e:
            status_text.text("❌ Error occurred!")
            st.error(f"❌ Error training model: {e}")
        finally:
            # Clean up progress indicators after a delay
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

# Train both models button
if st.sidebar.button(" Train Both Models", help="Train both RandomForest and MLP models"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        import subprocess
        
        # Train RandomForest
        status_text.text("🔄 Training RandomForest model...")
        progress_bar.progress(10)
        
        rf_result = subprocess.run([
            "/home/hp/Desktop/Projects/Dig-Reg/venv/bin/python", 
            "src/train.py", 
            "--model", 
            "randomforest"
        ], capture_output=True, text=True, cwd="/home/hp/Desktop/Projects/Dig-Reg")
        
        progress_bar.progress(50)
        
        # Train MLP
        status_text.text("🔄 Training MLP model...")
        
        mlp_result = subprocess.run([
            "/home/hp/Desktop/Projects/Dig-Reg/venv/bin/python", 
            "src/train.py", 
            "--model", 
            "mlp"
        ], capture_output=True, text=True, cwd="/home/hp/Desktop/Projects/Dig-Reg")
        
        progress_bar.progress(90)
        
        # Check results
        rf_success = rf_result.returncode == 0
        mlp_success = mlp_result.returncode == 0
        
        progress_bar.progress(100)
        
        if rf_success and mlp_success:
            status_text.text(" Both models trained successfully!")
            st.success(" Both models trained successfully!")
            st.balloons()  # 🎈 Celebration for successful training!
            
            # Show summary of training results
            with st.expander(" Training Summary"):
                st.write("**RandomForest Results:**")
                if rf_result.stdout:
                    st.code(rf_result.stdout[-500:])
                st.write("**MLP Results:**")
                if mlp_result.stdout:
                    st.code(mlp_result.stdout[-500:])
            
            st.cache_resource.clear()  # Clear cache to reload models
            st.rerun()
        else:
            status_text.text("❌ Some models failed to train!")
            if not rf_success:
                st.error("❌ RandomForest training failed!")
                with st.expander("🔍 RandomForest Error Details"):
                    st.code(rf_result.stderr)
            if not mlp_success:
                st.error("❌ MLP training failed!")
                with st.expander("🔍 MLP Error Details"):
                    st.code(mlp_result.stderr)
    except Exception as e:
        status_text.text("❌ Error occurred!")
        st.error(f"❌ Error training models: {e}")
    finally:
        # Clean up progress indicators after a delay
        import time
        time.sleep(3)
        progress_bar.empty()
        status_text.empty()

# Command line instructions (for reference)
st.sidebar.subheader("Command Line Training")
st.sidebar.code("# Train RandomForest\npython3 src/train.py --model randomforest")
st.sidebar.code("# Train MLP\npython3 src/train.py --model mlp")
