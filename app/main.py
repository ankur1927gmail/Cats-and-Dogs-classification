from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
import logging

from src.model import CatDogCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

request_count = 0

app = FastAPI()

model = CatDogCNN()

model.load_state_dict(
    torch.load(
        "models/cat_dog_cnn.pt",
        map_location=torch.device("cpu")
    )
)

model.eval()

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model": "CatDogCNN",
        "loaded": True
    }


@app.get("/metrics")
def metrics():
    return {
        "request_count": request_count
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    global request_count
    request_count += 1

    logging.info("Prediction request received")

    image = Image.open(file.file).convert("RGB")

    image = predict_transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    label = "Cat" if predicted.item() == 0 else "Dog"

    logging.info(f"Prediction result: {label}")

    return {
        "prediction": label,
        "confidence": float(confidence.item())
    }