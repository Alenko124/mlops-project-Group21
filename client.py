"""Simple client for sending images to EuroSAT API for prediction and data logging."""

import argparse
import random
import time
from pathlib import Path

import requests


class ImageClassificationClient:
    """Client for sending satellite images to the classification API."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            api_url: Base URL of the API (default: local development)
        """
        self.api_url = api_url.rstrip("/")
        self.endpoint = f"{self.api_url}/predict"

    def get_test_images(self, test_data_dir: str = "data/raw/test") -> list[Path]:
        """Get all test images from directory.

        Args:
            test_data_dir: Path to test data directory

        Returns:
            List of image paths
        """
        data_dir = Path(test_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {data_dir}")

        # Get all JPG files from class subdirectories
        images = list(data_dir.rglob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No images found in {data_dir}")

        return images

    def send_prediction(self, image_path: Path, log_to_cloud: bool = True) -> dict:
        """Send image to API for prediction.

        Args:
            image_path: Path to image file
            log_to_cloud: Whether to log prediction to cloud

        Returns:
            Prediction response from API
        """
        try:
            with open(image_path, "rb") as f:
                files = {"file": f}
                params = {"log_to_cloud": log_to_cloud}

                response = requests.post(
                    self.endpoint,
                    files=files,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending request for {image_path.name}: {e}")
            return None

    def run(
        self,
        test_data_dir: str = "data/raw/test",
        num_requests: int = 10,
        delay: float = 0.5,
        log_to_cloud: bool = True,
        random_sample: bool = True,
    ) -> None:
        """Run the client to send multiple prediction requests.

        Args:
            test_data_dir: Path to test data directory
            num_requests: Number of requests to send
            delay: Delay between requests in seconds
            log_to_cloud: Whether to log predictions to cloud
            random_sample: Whether to randomly sample images
        """
        print("🚀 Starting Image Classification Client")
        print(f"   API URL: {self.api_url}")
        print(f"   Test data: {test_data_dir}")
        print(f"   Num requests: {num_requests}")
        print(f"   Log to cloud: {log_to_cloud}")
        print()

        try:
            images = self.get_test_images(test_data_dir)
            print(f"✅ Found {len(images)} test images")
            print()

            if random_sample:
                images = random.sample(images, min(num_requests, len(images)))
            else:
                images = images[:num_requests]

            successful = 0
            failed = 0

            for i, image_path in enumerate(images, 1):
                print(f"[{i}/{len(images)}] Sending: {image_path.name}")

                result = self.send_prediction(image_path, log_to_cloud)

                if result:
                    print(
                        f"  ✓ Predicted: {result['predicted_class_name']} " f"(confidence: {result['confidence']:.4f})"
                    )

                    # Display extracted image features
                    features = result.get("image_features", {})
                    if features:
                        print(
                            f"  📊 Features - Brightness: {features.get('brightness'):.2f}, "
                            f"Contrast: {features.get('contrast'):.2f}, "
                            f"Sharpness: {features.get('sharpness'):.4f}"
                        )

                    successful += 1
                else:
                    failed += 1

                # Delay between requests
                if i < len(images):
                    time.sleep(delay)

            print()
            print(f"✅ Completed: {successful} successful, {failed} failed")
            if log_to_cloud:
                print("📤 Predictions logged to: gs://mlops-group21/predictions/data_logs/")

        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point for the client."""
    parser = argparse.ArgumentParser(description="Send satellite images to EuroSAT API for prediction and data logging")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/raw/test",
        help="Path to test data directory (default: data/raw/test)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of predictions to send (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Don't log predictions to cloud",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Send images sequentially instead of random sampling",
    )

    args = parser.parse_args()

    client = ImageClassificationClient(api_url=args.url)
    client.run(
        test_data_dir=args.test_dir,
        num_requests=args.num_requests,
        delay=args.delay,
        log_to_cloud=not args.no_cloud,
        random_sample=not args.sequential,
    )


if __name__ == "__main__":
    main()
