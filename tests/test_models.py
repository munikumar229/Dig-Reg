#!/usr/bin/env python3
"""
Test script to verify both RandomForest and MLP models are working correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def test_model_prediction(model_type):
    """Test a model's prediction capability."""
    print(f"\n🧪 Testing {model_type.upper()} Model")
    print("=" * 50)
    
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = f"{model_type.title()}-Digits"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found!")
            return False
            
        runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            print(f"❌ No runs found for experiment '{experiment_name}'!")
            return False
            
        latest_run_id = runs.iloc[0]['run_id']
        print(f"📋 Latest run ID: {latest_run_id}")
        
        # Load model
        if model_type.lower() == "randomforest":
            artifact_path = "random-forest-model"
        elif model_type.lower() == "mlp":
            artifact_path = "mlp-model"
        else:
            print(f"❌ Unsupported model type: {model_type}")
            return False
            
        model_uri = f"runs:/{latest_run_id}/{artifact_path}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully: {type(model)}")
        
        # Load test data (using sklearn digits dataset)
        digits = load_digits()
        # Take first 5 samples for testing
        X_test = digits.data[:5].reshape(5, 8, 8)
        y_true = digits.target[:5]
        
        print(f"🎯 True labels: {y_true}")
        
        # Preprocess data according to model type
        predictions = []
        for i, img in enumerate(X_test):
            if model_type.lower() == "mlp":
                # For MLP, try to load and use the scaler
                try:
                    scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{latest_run_id}/scaler.pkl")
                    import pickle
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                    
                    # Flatten first, then apply scaler
                    img_flattened = img.flatten().reshape(1, -1)
                    input_data = scaler.transform(img_flattened)
                    print(f"📊 MLP input shape: {input_data.shape}")
                    
                except Exception as e:
                    print(f"⚠️  Could not load scaler: {e}")
                    print("   Using StandardScaler instead...")
                    scaler = StandardScaler()
                    img_flattened = img.flatten().reshape(1, -1)
                    input_data = scaler.fit_transform(img_flattened)
            else:
                # For RandomForest, use MinMaxScaler
                scaler = MinMaxScaler((0, 16))
                img_scaled = scaler.fit_transform(img)
                input_data = img_scaled.flatten().reshape(1, -1)
                # Convert to DataFrame for RandomForest
                feature_names = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]
                input_data = pd.DataFrame(input_data, columns=feature_names)
                print(f"📊 RandomForest input shape: {input_data.shape}")
            
            # Make prediction
            try:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0].max()
                predictions.append(pred)
                print(f"   Sample {i+1}: Predicted={pred}, True={y_true[i]}, Confidence={prob:.3f} {'✅' if pred == y_true[i] else '❌'}")
            except Exception as e:
                print(f"   Sample {i+1}: ❌ Prediction failed: {e}")
                return False
        
        # Calculate accuracy
        accuracy = sum(p == t for p, t in zip(predictions, y_true)) / len(y_true)
        print(f"\n📈 Test Accuracy: {accuracy:.2%}")
        print(f"✅ {model_type.upper()} model is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_type} model: {e}")
        return False

def main():
    print("🚀 Testing Digit Classification Models")
    print("=" * 60)
    
    # Test both models
    rf_success = test_model_prediction("randomforest")
    mlp_success = test_model_prediction("mlp")
    
    print(f"\n📋 SUMMARY")
    print("=" * 30)
    print(f"RandomForest Model: {'✅ Working' if rf_success else '❌ Failed'}")
    print(f"MLP Model: {'✅ Working' if mlp_success else '❌ Failed'}")
    
    if rf_success and mlp_success:
        print(f"\n🎉 All models are working correctly!")
        print(f"🖥️  You can now use the Streamlit app: streamlit run app.py")
    else:
        print(f"\n⚠️  Some models have issues. Please retrain them.")
        print(f"🔧 Run: ./train_all_models.sh")

if __name__ == "__main__":
    main()