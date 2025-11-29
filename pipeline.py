# pipeline.py
from src.pipeline_components import fetch_data, preprocess_data, train_model, evaluate_model

# 1️⃣ Fetch data
data_path = fetch_data(dvc_path="data/raw_data.csv", remote="localstore")

# 2️⃣ Preprocess
X_train, X_test, y_train, y_test = preprocess_data(data_path)

# 3️⃣ Train model
model = train_model(X_train, y_train)

# 4️⃣ Evaluate model
evaluate_model(model, X_test, y_test)
