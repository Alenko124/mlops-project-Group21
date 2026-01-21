"""FastAPI application for EuroSAT model inference."""

from contextlib import asynccontextmanager
from io import BytesIO

import torch
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import storage
from PIL import Image
from torchvision import transforms


model = None
device = None
transform = None
class_names = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Permanent Crop",
    "Residential",
    "River",
    "Sea Lake",
]


def load_model_from_gcs(bucket_name: str, model_path: str) -> torch.nn.Module:
    """Load model from Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name
        model_path: Path to model file in bucket
    Returns:
        Loaded PyTorch model on appropriate device
    """
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)

    # Download model to memory
    model_bytes = blob.download_as_bytes()
    model_buffer = BytesIO(model_bytes)

    # Create model architecture
    model = timm.create_model(
        "resnet18.a1_in1k",
        pretrained=False,
        num_classes=10,
    )

    # Load weights
    state_dict = torch.load(model_buffer, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def create_transform() -> transforms.Compose:
    """Create image preprocessing pipeline matching training setup."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, transform

    print("Loading model from Google Cloud Storage...")
    try:
        # BUCKET AND MODEL PATH
        bucket_name = "mlops-group21"
        model_path = "models/resnet18_eurosat4.pt"

        model = load_model_from_gcs(bucket_name, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = create_transform()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield

    print("Cleaning up")
    del model, device, transform


app = FastAPI(
    title="EuroSAT Image Classification API",
    description="API for classifying satellite images using ResNet18 trained on EuroSAT",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a satellite image.

    Args:
        file: Image file to classify (JPEG, PNG, etc.)

    Returns:
        Dictionary with predictions, confidence scores, and class names
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_class = predicted_class.item()

        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probabilities[0], k=5)

        return {
            "predicted_class": predicted_class,
            "predicted_class_name": class_names[predicted_class],
            "confidence": round(confidence, 4),
            "top_5_predictions": [
                {
                    "class_id": idx.item(),
                    "class_name": class_names[idx.item()],
                    "confidence": round(prob.item(), 4),
                }
                for prob, idx in zip(top5_probs, top5_indices)
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
