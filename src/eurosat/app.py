import io
import os
import sys
from http import HTTPStatus
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

try:
    from eurosat.model import ModelConfig, create_model
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from eurosat.model import ModelConfig, create_model


app = FastAPI(
    title="EuroSAT Classifier API",
    description="Satellite Image Land-Use Classification",
    version="1.0.0",
)

model = None


@app.on_event("startup")
async def load_model():

    global model
    try:
        cfg = ModelConfig()
        model = create_model(cfg)

        # Model dosya yolu (Proje kök dizininden)
        model_path = "models/model.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"[INFO] Model loaded from {model_path}")
        else:
            print(
                f"[WARNING] Model file not found at {model_path}. Using initialized model with random weights."
            )

        model.eval()
    except Exception as e:
        print(f"[ERROR] Model loading error: {e}")


def process_image(image_bytes):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


@app.get("/")
def read_root():
    return {"message": "EuroSAT API is running correctly."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
 
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        input_tensor = process_image(content)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)

        classes = [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]

        return {
            "filename": file.filename,
            "prediction": classes[pred_idx],
            "confidence": float(conf),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
