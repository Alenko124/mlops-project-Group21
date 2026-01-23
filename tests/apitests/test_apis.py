from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
from eurosat.api import app


client = TestClient(app)


def test_health_endpoint():
    """Test the /health endpoint returns expected response."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)


def create_test_image() -> BytesIO:
    """Create a test image (64x64 RGB)."""
    img = Image.new("RGB", (64, 64), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def test_predict_endpoint_with_image():
    """Test the /predict endpoint with a test image."""
    with TestClient(app) as client:
        test_image = create_test_image()

        response = client.post("/predict", files={"file": ("test.png", test_image, "image/png")})

        # Will be 503 if model not loaded, 200 if model is loaded
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "predicted_class_name" in data
            assert "confidence" in data
            assert "top_5_predictions" in data
            assert len(data["top_5_predictions"]) == 5
        elif response.status_code == 503:
            assert response.json()["detail"] == "Model not loaded"
