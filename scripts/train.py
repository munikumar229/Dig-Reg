import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import argparse

def train_model(model_type="randomforest"):
    """
    Train model for digit classification and log parameters, metrics, and model to MLflow.
    
    Args:
        model_type (str): Type of model to train. Options: 'randomforest', 'mlp'
    """

    # -------------------------------
    # 1. Set MLflow Tracking
    # -------------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = f"{model_type.title()}-Digits"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # -------------------------------
    # 2. Load Data
    # -------------------------------
    train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join("data", "processed", "val.csv"))

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_val = val_df.drop("target", axis=1)
    y_val = val_df["target"]
    
    # Scale data for MLP (neural networks work better with scaled data)
    scaler = None
    if model_type.lower() == "mlp":
        scaler = StandardScaler()
        # Convert to numpy arrays to avoid feature name issues
        X_train = scaler.fit_transform(X_train.values)
        X_val = scaler.transform(X_val.values)

    # -------------------------------
    # 3. Model setup and Hyperparameter tuning
    # -------------------------------
    if model_type.lower() == "randomforest":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 15]
        }
        model = RandomForestClassifier(random_state=42)
        model_name = "RandomForestClassifier"
        artifact_path = "random-forest-model"
        
    elif model_type.lower() == "mlp":
        param_grid = {
            'hidden_layer_sizes': [(100,), (256, 50), (64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001]
        }
        model = MLPClassifier(random_state=42, max_iter=1000)
        model_name = "MLPClassifier"
        artifact_path = "mlp-model"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'randomforest' or 'mlp'")

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    with mlflow.start_run() as run:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate
        preds = best_model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')

        print(f"Model Type: {model_type.title()}")
        print("Best Hyperparameters:", best_params)
        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        print("Classification Report:\n", classification_report(y_val, preds))

        # Log params, metrics, model
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log model (handle scaler separately for MLP)
        if scaler is not None:
            # For MLP, create a custom model wrapper that includes the scaler
            import pickle
            
            # Save scaler separately
            scaler_path = "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path)
            
            # Log the model
            mlflow.sklearn.log_model(
                best_model, 
                artifact_path=artifact_path, 
                registered_model_name="DigitsClassifier"
            )
            
            # Log scaler info as parameter
            mlflow.log_param("uses_scaler", True)
            mlflow.log_param("scaler_type", "StandardScaler")
            
            # Clean up
            os.remove(scaler_path)
        else:
            mlflow.sklearn.log_model(
                best_model, 
                artifact_path=artifact_path, 
                registered_model_name="DigitsClassifier"
            )
            mlflow.log_param("uses_scaler", False)

        # Save classification report as artifact
        report_path = f"classification_report_{model_type}.txt"
        with open(report_path, "w") as f:
            f.write(f"Model Type: {model_type.title()}\n")
            f.write("Best Hyperparameters: " + str(best_params) + "\n\n")
            f.write(classification_report(y_val, preds))
        mlflow.log_artifact(report_path)

    # -------------------------------
    # 4. Transition model to Production
    # -------------------------------
    try:
        latest_version = client.get_latest_versions("DigitsClassifier")[0].version
        client.transition_model_version_stage(
            name="DigitsClassifier",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {latest_version} transitioned to Production.")
    except Exception as e:
        print(f"Could not transition to Production: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for digit classification")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["randomforest", "mlp"], 
        default="randomforest",
        help="Type of model to train (default: randomforest)"
    )
    
    args = parser.parse_args()
    train_model(model_type=args.model)
