# Create: frontend/standalone_app.py
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import os
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="🖌️ Digits Classifier",  
    page_icon="🖌️",
    layout="centered"
)

# Add your existing CSS styling here...
st.markdown("""
    <style>
    .main { background-color: #f7f7fa; }
    .stButton > button {
        color: white;
        background: linear-gradient(90deg, #4f8bf9 0%, #235390 100%);
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Model Loading (Local)
# -------------------------------
@st.cache_resource
def load_models():
    """Load models directly in Streamlit"""
    models = {}
    
    try:
        # Check for models in different locations and naming patterns
        model_paths = {
            'randomforest': [
                'models/best_random_forest.pkl',
                'models/best_randomforest_model.pkl',
                '../models/best_random_forest.pkl',
                '../models/best_randomforest_model.pkl'
            ],
            'mlp': [
                'models/best_mlp.pkl',
                'models/mlp_model.pkl', 
                '../models/best_mlp.pkl',
                '../models/mlp_model.pkl'
            ]
        }
        
        # Try to load existing models from standard paths
        for model_type, paths in model_paths.items():
            for path in paths:
                if os.path.exists(path):
                    try:
                        models[model_type] = joblib.load(path)
                        st.success(f"✅ Loaded {model_type} from {path}")
                        break
                    except Exception as e:
                        st.warning(f"Failed to load {path}: {e}")
                        continue
        
        # Try to load from MLflow artifacts if no models found yet
        if not models:
            st.info("🔍 Searching for MLflow models...")
            try:
                import glob
                mlflow_models = glob.glob('mlruns/*/models/*/artifacts/model.pkl')
                if not mlflow_models:
                    mlflow_models = glob.glob('../mlruns/*/models/*/artifacts/model.pkl')
                
                for i, model_path in enumerate(mlflow_models[:2]):  # Load first 2 models found
                    try:
                        model = joblib.load(model_path)
                        # Try to determine model type by checking the model class
                        model_name = model.__class__.__name__
                        if 'RandomForest' in model_name:
                            models['randomforest'] = model
                            st.success(f"✅ Loaded Random Forest from MLflow: {model_path}")
                        elif 'MLP' in model_name:
                            models['mlp'] = model  
                            st.success(f"✅ Loaded MLP from MLflow: {model_path}")
                        else:
                            # Generic naming if we can't determine type
                            models[f'model_{i+1}'] = model
                            st.success(f"✅ Loaded model_{i+1} from MLflow: {model_path}")
                    except Exception as e:
                        st.warning(f"Could not load MLflow model {model_path}: {e}")
                        continue
            except Exception as e:
                st.warning(f"Error searching MLflow models: {e}")
            
        # If no models found, train basic ones
        if not models:
            st.warning("No pre-trained models found. Training basic models...")
            models = train_basic_models()
        else:
            st.info(f"🎉 Loaded {len(models)} existing trained models!")
            
    except Exception as e:
        st.error(f"Error in model loading process: {e}")
        models = train_basic_models()
    
    return models

def train_basic_models():
    """Train basic models with sample data"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    with st.spinner("🤖 Training models on sklearn digits dataset..."):
        # Load sample data
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        st.info(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train models
        models = {}
        
        # Random Forest
        with st.spinner("Training Random Forest..."):
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
            models['randomforest'] = rf
            st.success(f"✅ Random Forest trained - Accuracy: {rf_accuracy:.3f}")
        
        # MLP
        with st.spinner("Training MLP Neural Network..."):
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            mlp.fit(X_train, y_train)
            mlp_accuracy = accuracy_score(y_test, mlp.predict(X_test))
            models['mlp'] = mlp
            st.success(f"✅ MLP Neural Network trained - Accuracy: {mlp_accuracy:.3f}")
        
        st.success(f"🎉 Training completed! Both models ready for predictions.")
    
    return models

def preprocess_image_standalone(img):
    """Preprocess image for model prediction"""
    # Resize to 8x8 (matching sklearn digits dataset)
    img_resized = cv2.resize(img, (8, 8))
    
    # Convert to match sklearn format (0-16 scale) for model prediction
    img_for_model = 16 - (img_resized * 16 / 255)
    
    # Flatten for model input
    pixels = img_for_model.flatten().reshape(1, -1)
    
    # For display: normalize to [0, 1] range for Streamlit
    img_for_display = img_for_model / 16.0
    
    return pixels, img_for_display

# Load models
models = load_models()

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
# Model Selection
# -------------------------------
st.sidebar.header("Model Selection")
available_models = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Choose Model:",
    available_models,
    format_func=lambda x: x.title()
)

st.sidebar.success(f"✅ {len(available_models)} models loaded")

# -------------------------------
# Drawing Interface
# -------------------------------
st.sidebar.header("Drawing Options")
stroke_width = st.sidebar.slider("Stroke Width", 1, 30, 12)
stroke_color = st.sidebar.color_picker("Stroke Color", "#000000")

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

# -------------------------------
# Buttons
# -------------------------------
col1, col2 = st.columns([1, 1])
predict_btn = col1.button("🔮 Predict Digit")
clear_btn = col2.button("🧹 Clear")

if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

# -------------------------------
# Prediction Logic  
# -------------------------------
if predict_btn and canvas_result.image_data is not None:
    # Get image from canvas
    input_img = cv2.cvtColor(
        canvas_result.image_data.astype("uint8"), 
        cv2.COLOR_RGBA2GRAY
    )
    
    # Preprocess
    pixels, img_resized = preprocess_image_standalone(input_img)
    
    # Make prediction
    model = models[selected_model]
    prediction = model.predict(pixels)[0]
    probabilities = model.predict_proba(pixels)[0]
    confidence = float(max(probabilities))
    
    # Save to history
    st.session_state.history.append({
        "digit": int(prediction),
        "model": selected_model.title(),
        "confidence": confidence
    })
    
    # Display results
    st.success(f"🎯 Predicted Digit: {int(prediction)}")
    st.info(f"Confidence: {confidence:.2%}")
    
    # Show processed image
    col1, col2 = st.columns(2)
    with col1:
        # Normalize original image for display
        display_original = input_img / 255.0 if input_img.max() > 1.0 else input_img
        st.image(display_original, caption="Original Drawing", width=150, clamp=True)
    with col2:
        # img_resized is already normalized for display
        st.image(img_resized, caption="Processed 8x8", width=150, clamp=True)
    
    # Probability chart
    st.subheader("Probability Distribution")
    prob_df = pd.DataFrame({
        "Digit": list(range(10)),
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Digit"))

# -------------------------------
# History
# -------------------------------
if st.session_state.history:
    st.subheader("📊 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, width='stretch')
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# -------------------------------
# Model Info
# -------------------------------
st.sidebar.subheader("📈 Model Information")
for model_name in available_models:
    st.sidebar.write(f"• {model_name.title()}: ✅")

st.sidebar.subheader("ℹ️ Technical Details")
st.sidebar.write("**Input:** 8x8 grayscale (0-16 scale)")
st.sidebar.write("**Models:** Scikit-learn")
st.sidebar.write("**Architecture:** Standalone Streamlit")