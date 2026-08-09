import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5002")
mlflow.set_experiment("CatsDogsClassification")

with mlflow.start_run(run_name="test_training_run"):
    mlflow.log_param("epochs", 2)
    mlflow.log_metric("accuracy", 0.90)

print("MLflow logging successful")