# Cats vs Dogs Classification - End-to-End MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) using a Convolutional Neural Network (CNN).

The solution demonstrates the complete machine learning lifecycle including:

- Data Versioning using DVC
- Model Development using PyTorch
- Experiment Tracking using MLflow
- Model Packaging using FastAPI
- Containerization using Docker
- CI Pipeline using GitHub Actions
- CD Pipeline using GitHub Actions and Docker Compose
- Monitoring and Logging
- Smoke Testing

This project was developed as part of the MLOps Assignment. 【1-69fd56】

---

# Project Structure

```text
Cats-and-Dogs-classification/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   └── main.py
│
├── src/
│   ├── model.py
│   ├── preprocess.py
│   └── train_real.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocess.py
│
├── models/
│   └── cat_dog_cnn.pt
│
├── artifacts/
│
├── .dvc/
│
├── Dockerfile
├── docker-compose.yaml
├── deploy.sh
├── smoke_test.py
├── requirements.txt
└── README.md

M1 – Model Development & Experiment Tracking
Data Versioning
Dataset versioning is managed using DVC.
Features:
Dataset Tracking
Reproducibility
Version-Controlled ML Assets
Model Development
Model:
CatDogCNN
Framework:
PyTorch
Saved Model:
models/cat_dog_cnn.pt
Experiment Tracking
MLflow is used to track:
Parameters
Metrics
Training Runs
Model Artifacts
Tracked Metrics:
train_loss
validation_loss
test_loss
train_accuracy
validation_accuracy
test_accuracy
Logged Artifacts:
cat_dog_cnn.pt
confusion_matrix.png
loss_curve.png
M2 – Model Packaging & Containerization
FastAPI Inference Service
Available Endpoints:
GET /health
GET /model-info
GET /metrics
POST /predict
Environment Specification
Dependency management is performed using:
requirements.txt
Docker Containerization
Build Image:
docker build -t catsdogs-api .
Run Container:
docker run -p 8000:8000 catsdogs-api
M3 – Continuous Integration
GitHub Actions is used for CI.
Pipeline Steps:
Checkout Repository
Setup Python
Install Dependencies
Run Unit Tests
Build Docker Image
Publish Docker Image
Automated Tests:
test_model.py
test_preprocess.py
Docker Hub Artifact Publishing
Docker image is published to Docker Hub.
Published Artifact:
catsdogs-api:latest
Docker Hub acts as the container registry for deployment and image distribution.
M4 – Continuous Deployment
Deployment Target
Deployment is implemented using Docker Compose.
Deployment Manifest:
docker-compose.yaml
Start Deployment:
docker compose up -d
Deployment Flow
Code Push
↓
GitHub Actions Trigger
↓
Run Tests
↓
Build Docker Image
↓
Publish Docker Image to Docker Hub
↓
Deploy Stage
↓
Smoke Test
↓
Deployment Complete
Deployment Script
File:
deploy.sh
Deployment Actions:
Stop Existing Deployment
Pull Latest Image
Start Updated Deployment
Verify Running Containers
Deployment Commands:
docker compose down
docker compose pull
docker compose up -d
docker ps
Smoke Testing
File:
smoke_test.py
Validation Process:
Call Health Endpoint
Verify API Availability
Confirm Successful Deployment
Expected Response:
Status Code: 200
Response:
{"status":"healthy"}
M5 – Monitoring & Logging
Logging
Prediction requests are logged.
Example Log Messages:
Prediction request received
Prediction result generated
Monitoring
Metrics Endpoint:
GET /metrics
Example Response:
{"request_count": 5}
Tracked Metric:
Total Prediction Requests
API Usage
Health Check
Endpoint:
GET /health
Example Response:
{"status":"healthy"}
Model Information
Endpoint:
GET /model-info
Example Response:
{ "model": "CatDogCNN", "loaded": true }
Prediction
Endpoint:
POST /predict
Example Response:
{ "prediction": "Dog", "confidence": 0.84 }
Metrics
Endpoint:
GET /metrics
Example Response:
{ "request_count": 5 }
CI/CD Summary
Implemented Components:
✅ Git
✅ DVC
✅ MLflow
✅ PyTorch
✅ FastAPI
✅ Docker
✅ Docker Compose
✅ GitHub Actions
✅ Docker Hub
✅ Smoke Testing
✅ Logging
✅ Monitoring
Screenshots Included
GitHub Repository
DVC Status
MLflow Experiments
MLflow Parameters
MLflow Metrics
MLflow Artifacts
FastAPI Swagger
Prediction Response
Docker Image
Docker Container
GitHub Actions Success
GitHub Actions Workflow
Docker Compose Deployment
Health Endpoint
Metrics Endpoint
Docker Logs
Project Structure
DVC Configuration
Deployment Proof
Monitoring Evidence
Docker Hub Push
Docker Hub Repository
Smoke Test
CI/CD Pipeline
Author
Ankur Sharma
M.Tech Artificial Intelligence & Machine Learning
BITS Pilani
End-to-End MLOps Pipeline for Cats vs Dogs Image Classification