import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_model():
    """
    Train RandomForest model for digit classification
    and log parameters, metrics, and model to MLflow.
    """

    # -------------------------------
    # 1. Set MLflow Tracking
    # -------------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "RandomForest-Digits"
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

    # -------------------------------
    # 3. Hyperparameter tuning
    # -------------------------------
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 15]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    with mlflow.start_run() as run:
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate
        preds = best_rf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')

        print("Best Hyperparameters:", best_params)
        print(f"Accuracy: {acc}, F1 Score: {f1}")
        print("Classification Report:\n", classification_report(y_val, preds))

        # Log params, metrics, model
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_rf, artifact_path="random-forest-best-model", registered_model_name="DigitsClassifier")

        # Save classification report as artifact
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
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
    train_model()
