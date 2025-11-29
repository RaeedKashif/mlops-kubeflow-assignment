# src/pipeline_components.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import subprocess

# ------------------------------
# 1️⃣ Data Extraction
# ------------------------------
def fetch_data(dvc_path: str = "data/raw_data.csv", remote: str = "localstore") -> str:
    """
    Fetches the dataset using DVC from the remote storage.

    Args:
        dvc_path: Path to the DVC-tracked file.
        remote: Name of the DVC remote.

    Returns:
        Local path to the dataset.
    """
    print(f"Pulling data from DVC remote '{remote}'...")
    subprocess.run(["dvc", "pull", dvc_path, "-r", remote], check=True)
    if not os.path.exists(dvc_path):
        raise FileNotFoundError(f"Dataset not found at {dvc_path}")
    print(f"Data ready at {dvc_path}")
    return dvc_path

# ------------------------------
# 2️⃣ Data Preprocessing
# ------------------------------
def preprocess_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads the CSV dataset, scales features, and splits into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path)
    
    # Assuming the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Save scaler for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("Preprocessing complete: data scaled and split.")
    return X_train, X_test, y_train, y_test

# ------------------------------
# 3️⃣ Model Training
# ------------------------------
def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Trains a RandomForestRegressor and logs the model to MLflow.
    Returns the trained model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_model.joblib"
        joblib.dump(model, model_path)
        
        mlflow.log_artifact(model_path, artifact_path="models")
        
        print("Model training complete and logged to MLflow.")
    
    return model

# ------------------------------
# 4️⃣ Model Evaluation
# ------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and logs metrics to MLflow and a JSON file.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics_path = "models/metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MSE: {mse}\nR2: {r2}\n")
    
    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
    
    print(f"Evaluation complete. MSE={mse:.4f}, R2={r2:.4f}")
    return mse, r2
