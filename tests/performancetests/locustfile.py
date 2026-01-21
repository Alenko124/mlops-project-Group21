import random
from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


def create_test_image() -> BytesIO:
    # Create random colored image
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new("RGB", (64, 64), color=color)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


class EuroSATUser(HttpUser):
    """Simulates a user interacting with the EuroSAT API.
    
    This user class tests both health checks and image predictions,
    simulating realistic API usage patterns.
    """
    wait_time = between(1, 3) 
    @task(1)
    def health_check(self) -> None:
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(4)
    def predict_image(self) -> None:
        test_image = create_test_image()
        
        with self.client.post(
            "/predict",
            files={"file": ("test.png", test_image, "image/png")},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
