"""FastAPI application for EuroSAT model inference."""

from contextlib import asynccontextmanager
from io import BytesIO

import torch
import timm
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from google.cloud import storage
from PIL import Image
from torchvision import transforms
from prometheus_client import Counter, Histogram, Summary, make_asgi_app, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from eurosat.image_features import extract_image_features
from eurosat.data_logger import CloudPredictionLogger
from eurosat.drift_detector import DriftDetector

model = None
device = None
transform = None
prediction_logger = None
drift_detector = None
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

# Define Prometheus metrics
MY_REGISTRY = CollectorRegistry()
error_counter = Counter('error_counter', 'Error counter', registry=MY_REGISTRY)
request_counter = Counter("prediction_requests", "Number of prediction requests", registry=MY_REGISTRY)
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY)
prediction_by_class = Counter("prediction_by_class_total", "Number of predictions per EuroSAT class", ["class_name"], registry=MY_REGISTRY)


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
    global model, device, transform, prediction_logger, drift_detector

    print("Loading model from Google Cloud Storage...")
    try:
        # BUCKET AND MODEL PATH 
        bucket_name = "mlops-group21" 
        model_path = "models/resnet18_eurosat_latest.pt"

        model = load_model_from_gcs(bucket_name, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = create_transform()
        
        # Initialize prediction logger
        prediction_logger = CloudPredictionLogger(
            bucket_name=bucket_name,
            predictions_folder="predictions/data_logs",
            batch_size=1,
        )
        
        # Initialize drift detector
        drift_detector = DriftDetector(
            bucket_name=bucket_name,
            reference_folder="predictions/reference_data",
            current_folder="predictions/data_logs",
        )
        
        print("Prediction logger initialized successfully")
        print("Drift detector initialized successfully")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield

    print("Cleaning up")
    # Flush any remaining predictions before shutdown
    if prediction_logger:
        prediction_logger.flush()
    del model, device, transform, prediction_logger, drift_detector


app = FastAPI(
    title="EuroSAT Image Classification API",
    description="API for classifying satellite images using ResNet18 trained on EuroSAT",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        generate_latest(MY_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/report")
async def get_drift_report(last_n: int = 100):
    """Get data drift monitoring report.
    
    Args:
        last_n: Number of recent predictions to compare against baseline
        
    Returns:
        HTML report page
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        html_content = drift_detector.generate_drift_report(num_current_records=last_n)
        return Response(content=html_content, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/report/features")
async def get_feature_drift_report(last_n: int = 100):
    """Get focused data drift report on image features.
    
    Args:
        last_n: Number of recent predictions to compare against baseline
        
    Returns:
        HTML report page focused on brightness, contrast, sharpness, confidence
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        html_content = drift_detector.generate_feature_drift_report(
            num_current_records=last_n
        )
        return Response(content=html_content, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/drift-summary")
async def get_drift_summary(last_n: int = 100):
    """Get summary statistics about data drift.
    
    Args:
        last_n: Number of recent predictions to analyze
        
    Returns:
        JSON with drift summary statistics
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    summary = drift_detector.get_drift_summary(num_current_records=last_n)
    return summary

@app.post("/predict")
async def predict(file: UploadFile = File(...), log_to_cloud: bool = True):
    """Classify a satellite image and optionally log to cloud.
    
    Args:
        file: Image file to classify
        log_to_cloud: Whether to log prediction and features to cloud (default: True)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_counter.inc()

    try:
        with request_latency.time():
            contents = await file.read()
            image = Image.open(BytesIO(contents))

            if image.mode != "RGB":
                image = image.convert("RGB")

            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            class_name = class_names[predicted_class]
            prediction_by_class.labels(class_name=class_name).inc()

            top5_probs, top5_indices = torch.topk(probabilities[0], k=5)

            top_5_predictions = [
                {
                    "class_id": idx.item(),
                    "class_name": class_names[idx.item()],
                    "confidence": round(prob.item(), 4),
                }
                for prob, idx in zip(top5_probs, top5_indices)
            ]

            # Extract image features
            image_features = extract_image_features(contents)

            # Log to cloud if enabled
            if log_to_cloud and prediction_logger:
                prediction_logger.log_prediction(
                    predicted_class=predicted_class,
                    predicted_class_name=class_name,
                    confidence=round(confidence, 4),
                    image_features=image_features,
                    top_5_predictions=top_5_predictions,
                    true_label=None,
                )

            return {
                "predicted_class": predicted_class,
                "predicted_class_name": class_name,
                "confidence": round(confidence, 4),
                "top_5_predictions": top_5_predictions,
                "image_features": image_features,
            }

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
