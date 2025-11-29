# pipeline_runner.py

from src.pipeline_components import fetch_data, preprocess_data, train_model, evaluate_model
import mlflow

def run_pipeline():
    # Set MLflow tracking
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("ML_Pipeline_Experiment")

    print("=== Pipeline Start ===")

    # Step 1: Fetch data from DVC remote
    data_path = fetch_data(dvc_path="data/raw_data.csv", remote="localstore")

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)

    print("=== Pipeline Finished ===")
    print(f"Metrics: MSE={mse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    run_pipeline()
