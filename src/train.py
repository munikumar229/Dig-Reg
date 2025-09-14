import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import dagshub

def train_model(use_dagshub=False):
    """
    Train RandomForest model for digit classification
    and log parameters, metrics, and model to MLflow.
    """

    # -------------------------------
    # 1. Set Tracking Backend
    # -------------------------------
    if use_dagshub:
        dagshub.init(
            repo_owner="munikumar229",
            repo_name="MLops-End-to-End-pipeline-for-Crop-Yield-Estiamtion",
            mlflow=True
        )
        print("Tracking MLflow on DagsHub...")
    else:
        # mlflow.set_tracking_uri("http://localhost:5000")  # MLflow server
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local sqlite fallback
        print("Tracking MLflow locally...")

    mlflow.set_experiment("RandomForest-Digits")

    # -------------------------------
    # 2. Load Data
    # -------------------------------
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_val = val_df.drop("target", axis=1)
    y_val = val_df["target"]

    # -------------------------------
    # 3. Training & Logging
    # -------------------------------
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 15]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    with mlflow.start_run():
        print("Starting hyperparameter tuning with GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate
        predictions = best_rf.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average='weighted')

        print("Best Hyperparameters:", best_params)
        print(f"Best Accuracy: {acc}")
        print(f"Best F1 Score: {f1}")
        print("Classification Report:\n", classification_report(y_val, predictions))

        # --- Log to MLflow ---
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # log model with correct artifact path
        mlflow.sklearn.log_model(best_rf, "random-forest-best-model")

        # Save classification report as artifact
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_val, predictions))
        mlflow.log_artifact(report_path)

    # -------------------------------
    # 4. Save best model locally
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, os.path.join("models", "best_random_forest.pkl"))
    print("Best model saved to models/best_random_forest.pkl")

if __name__ == "__main__":
    train_model(use_dagshub=False)
