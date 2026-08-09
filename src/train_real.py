import os
import random
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import CatDogCNN
from preprocess import train_transform, test_transform


ImageFile.LOAD_TRUNCATED_IMAGES = True


DATA_DIR = "data/raw"
MODEL_DIR = "models"
ARTIFACT_DIR = "artifacts"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5002"
MLFLOW_EXPERIMENT_NAME = "CatsDogsClassification"

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_IMAGES_PER_CLASS = 1000


class CatDogImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_samples(data_dir, max_images_per_class=1000):
    class_map = {
        "Cat": 0,
        "Dog": 1
    }

    all_samples = []

    for class_name, label in class_map.items():
        class_dir = Path(data_dir) / class_name

        image_files = list(class_dir.glob("*.jpg"))

        valid_images = []

        for image_path in image_files:
            if is_valid_image(image_path):
                valid_images.append(str(image_path))

        random.shuffle(valid_images)
        valid_images = valid_images[:max_images_per_class]

        for image_path in valid_images:
            all_samples.append((image_path, label))

        print(f"{class_name}: {len(valid_images)} images selected")

    random.shuffle(all_samples)

    return all_samples


def split_samples(samples):
    total_count = len(samples)

    train_count = int(total_count * 0.8)
    val_count = int(total_count * 0.1)

    train_samples = samples[:train_count]
    val_samples = samples[train_count:train_count + val_count]
    test_samples = samples[train_count + val_count:]

    return train_samples, val_samples, test_samples


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss = total_loss + loss.item()

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy, all_labels, all_predictions


def save_confusion_matrix(labels, predictions, output_path):
    matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Cat", "Dog"])
    plt.yticks([0, 1], ["Cat", "Dog"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    random.seed(42)
    torch.manual_seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Collecting samples...")

    samples = collect_samples(DATA_DIR, MAX_IMAGES_PER_CLASS)

    train_samples, val_samples, test_samples = split_samples(samples)

    print("Train samples:", len(train_samples))
    print("Validation samples:", len(val_samples))
    print("Test samples:", len(test_samples))

    train_dataset = CatDogImageDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = CatDogImageDataset(
        val_samples,
        transform=test_transform
    )

    test_dataset = CatDogImageDataset(
        test_samples,
        transform=test_transform
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    model = CatDogCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_path = os.path.join(MODEL_DIR, "cat_dog_cnn.pt")
    confusion_matrix_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    loss_curve_path = os.path.join(ARTIFACT_DIR, "loss_curve.png")

    train_losses = []
    val_losses = []

    with mlflow.start_run(run_name="baseline_cnn_training"):
        mlflow.log_param("model_name", "CatDogCNN")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("image_size", "224x224")
        mlflow.log_param("max_images_per_class", MAX_IMAGES_PER_CLASS)
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("validation_samples", len(val_dataset))
        mlflow.log_param("test_samples", len(test_dataset))
        mlflow.log_param("device", str(device))

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, val_accuracy, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", val_accuracy, step=epoch)

            print("Train Loss:", train_loss)
            print("Train Accuracy:", train_accuracy)
            print("Validation Loss:", val_loss)
            print("Validation Accuracy:", val_accuracy)

        test_loss, test_accuracy, test_labels, test_predictions = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

        torch.save(model.state_dict(), model_path)

        save_confusion_matrix(
            test_labels,
            test_predictions,
            confusion_matrix_path
        )

        save_loss_curve(
            train_losses,
            val_losses,
            loss_curve_path
        )

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(loss_curve_path)

        print("Model saved at:", model_path)
        print("Confusion matrix saved at:", confusion_matrix_path)
        print("Loss curve saved at:", loss_curve_path)
        print("Artifacts logged to MLflow")

    print("Training completed successfully")


if __name__ == "__main__":
    main()