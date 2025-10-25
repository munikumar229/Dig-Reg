"""
Model loading and management utilities
"""
import mlflow
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ModelLoader:
    """Handles loading and caching of ML models with fallback mechanisms."""
    
    def __init__(self, mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.models_cache = {}
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
    
    def load_model_by_type(self, model_type: str):
        """Load model from MLflow based on type, with robust fallback handling."""
        experiment_name = f"{model_type.title()}-Digits"
        
        # Return cached model if available
        if model_type in self.models_cache:
            print(f"📋 Using cached {model_type} model")
            return self.models_cache[model_type]
        
        # Try MLflow first
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
                if not runs.empty:
                    latest_run_id = runs.iloc[0]["run_id"]
                    
                    if model_type.lower() == "randomforest":
                        model_uri = f"runs:/{latest_run_id}/random-forest-best-model"
                    else:  # mlp
                        model_uri = f"runs:/{latest_run_id}/mlp-best-model"
                        
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✅ Loaded {model_type} model from MLflow: {model_uri}")
                    self.models_cache[model_type] = model
                    return model
        except Exception as e:
            print(f"⚠️ MLflow loading failed: {e}")
        
        # Fallback 1: Try to load from local pickle file
        print(f"🔄 Attempting fallback to local file for {model_type}")
        if model_type.lower() == "randomforest":
            local_path = "models/best_random_forest.pkl"
            if os.path.exists(local_path):
                try:
                    with open(local_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✅ Loaded {model_type} model from local file: {local_path}")
                    self.models_cache[model_type] = model
                    return model
                except Exception as e:
                    print(f"❌ Failed to load local file {local_path}: {e}")
        
        # Fallback 2: Create a simple trained model on the digits dataset
        print(f"🔄 Creating fresh {model_type} model as final fallback")
        try:
            model = self._create_fallback_model(model_type)
            print(f"✅ Created fresh {model_type} model")
            self.models_cache[model_type] = model
            return model
            
        except Exception as e:
            print(f"❌ Failed to create fallback model: {e}")
            raise RuntimeError(f"Could not load or create {model_type} model: {e}")
    
    def _create_fallback_model(self, model_type: str):
        """Create a fresh model trained on digits dataset."""
        # Load digits data
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        # Create and train model
        if model_type.lower() == "randomforest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        else:  # mlp
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.3f}")
        return model
    
    def get_available_models(self):
        """Get list of available model types."""
        return ["randomforest", "mlp"]
    
    def clear_cache(self):
        """Clear the models cache."""
        self.models_cache.clear()