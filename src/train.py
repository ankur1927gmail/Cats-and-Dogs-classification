import mlflow
import torch

print("Environment ready")
print("Torch Version:", torch.__version__)

mlflow.set_experiment("CatsDogsClassification")

with mlflow.start_run():
    mlflow.log_param("test_run", True)
    print("MLflow test run created successfully")