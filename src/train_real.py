import random
from pathlib import Path

import mlflow
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from model import CatDogCNN
from preprocess import train_transform, test_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = "data/raw"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5002"
MLFLOW_EXPERIMENT_NAME = "CatsDogsClassification"


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

    val_samples = samples[
        train_count:
        train_count + val_count
    ]

    test_samples = samples[
        train_count + val_count:
    ]

    return train_samples, val_samples, test_samples


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def main():
    random.seed(42)
    torch.manual_seed(42)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Collecting samples...")

    samples = collect_samples(DATA_DIR)

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

    model = CatDogCNN()

    print("CNN model loaded successfully")

    with mlflow.start_run(run_name="dataset_pipeline_test"):

        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("validation_samples", len(val_dataset))
        mlflow.log_param("test_samples", len(test_dataset))

        mlflow.log_param("batch_size", 32)

        mlflow.log_metric("total_images", len(samples))

        print("MLflow logging completed")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()