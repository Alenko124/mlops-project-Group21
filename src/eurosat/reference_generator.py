"""Generate reference baseline data from training dataset for drift detection."""

import json
from pathlib import Path
from typing import Optional

import torch
import timm
from google.cloud import storage
from io import BytesIO
from PIL import Image
from torchvision import transforms
from loguru import logger

from eurosat.image_features import extract_image_features


class ReferenceDataGenerator:
    """Generate reference baseline data from training images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        bucket_name: str = "mlops-group21",
        reference_folder: str = "predictions/reference_data",
    ):
        """Initialize reference data generator.

        Args:
            model_path: Path to trained model weights. If None, loads from GCS.
            bucket_name: GCS bucket name
            reference_folder: Folder in GCS to save reference data
        """
        self.bucket_name = bucket_name
        self.reference_folder = reference_folder
        self.gcs_client = storage.Client()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._create_transform()
        self.model = self._load_model(model_path)
        self.class_names = [
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

    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing pipeline."""
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

    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        """Load model from local path or GCS.

        Args:
            model_path: Local path to model or None to load from GCS

        Returns:
            Loaded model
        """
        logger.info("Loading model...")

        if model_path:
            # Load from local path
            logger.info(f"Loading model from local path: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
        else:
            # Load from GCS
            logger.info("Loading model from GCS...")
            bucket = self.gcs_client.bucket(self.bucket_name)
            blob = bucket.blob("models/resnet18_eurosat_latest.pt")
            model_bytes = blob.download_as_bytes()
            state_dict = torch.load(BytesIO(model_bytes), map_location=self.device)

        # Create model architecture
        model = timm.create_model(
            "resnet18.a1_in1k",
            pretrained=False,
            num_classes=10,
        )

        # Load weights
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info("Model loaded successfully")
        return model

    def process_image(self, image_path: Path) -> Optional[dict]:
        """Process a single image and return prediction with features.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction data or None if processing failed
        """
        try:
            # Read image
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Extract image features
            image_features = extract_image_features(image_bytes)

            # Make prediction
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            # Extract true label from directory structure (if available)
            # Assumes structure: data/raw/train/CLASS_ID/image.jpg
            true_label = None
            try:
                true_label = int(image_path.parent.name)
            except (ValueError, IndexError):
                pass

            record = {
                "predicted_class": predicted_class,
                "predicted_class_name": self.class_names[predicted_class],
                "confidence": round(confidence, 4),
                "image_features": image_features,
                "true_label": true_label,
            }

            return record

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return None

    def process_directory(
        self, data_dir: Path, sample_every: int = 1
    ) -> list:
        """Process all images in a directory structure.

        Args:
            data_dir: Root directory containing images organized by class
            sample_every: Process every N-th image (for sampling)

        Returns:
            List of prediction records
        """
        records = []
        image_count = 0
        processed_count = 0

        # Find all image files
        for image_path in sorted(data_dir.rglob("*.[jJ][pP][gG]")):
            image_count += 1

            # Sample every N-th image
            if image_count % sample_every != 0:
                continue

            logger.info(f"Processing image {processed_count + 1}: {image_path}")
            record = self.process_image(image_path)

            if record:
                records.append(record)
                processed_count += 1

        logger.info(f"Processed {processed_count} images total")
        return records

    def save_reference_data(self, records: list, filename: str = "reference_baseline.jsonl") -> Optional[str]:
        """Save reference data to GCS as individual JSON files.

        Args:
            records: List of prediction records
            filename: Base filename to save as (used for naming pattern)

        Returns:
            GCS path where data was saved, or None if failed
        """
        if not records:
            logger.warning("No records to save")
            return None

        try:
            base_name = filename.replace(".jsonl", "")
            bucket = self.gcs_client.bucket(self.bucket_name)
            
            # Upload each record as a separate JSON file
            for idx, record in enumerate(records):
                file_name = f"{base_name}_record_{idx}.json"
                blob_path = f"{self.reference_folder}/{file_name}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_string(
                    json.dumps(record, indent=2),
                    content_type="application/json"
                )
                logger.debug(f"Uploaded record {idx} to: {blob_path}")

            first_blob_path = f"{self.reference_folder}/{base_name}_record_0.json"
            logger.info(
                f"Successfully saved {len(records)} reference records to GCS: {self.reference_folder}/{base_name}_*.json"
            )
            return first_blob_path

        except Exception as e:
            logger.error(f"Failed to save reference data: {e}")
            return None

    def generate_baseline(
        self,
        data_dir: Path,
        sample_every: int = 1,
        filename: str = "reference_baseline.jsonl",
    ) -> Optional[str]:
        """Generate and save reference baseline data from training images.

        Args:
            data_dir: Directory containing training images
            sample_every: Process every N-th image (for sampling)
            filename: Filename to save baseline as

        Returns:
            GCS path where baseline was saved, or None if failed
        """
        logger.info(f"Generating reference baseline from: {data_dir}")
        records = self.process_directory(data_dir, sample_every=sample_every)

        if records:
            gcs_path = self.save_reference_data(records, filename)
            return gcs_path
        else:
            logger.error("No records generated")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate reference baseline from training data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to training data directory (e.g., data/raw/train)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model weights (if None, loads from GCS)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every N-th image (for sampling)",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="mlops-group21",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--reference-folder",
        type=str,
        default="predictions/reference_data",
        help="Folder in GCS to save reference data",
    )

    args = parser.parse_args()

    generator = ReferenceDataGenerator(
        model_path=args.model_path,
        bucket_name=args.bucket_name,
        reference_folder=args.reference_folder,
    )

    gcs_path = generator.generate_baseline(
        data_dir=args.data_dir,
        sample_every=args.sample_every,
    )

    if gcs_path:
        print(f"✓ Reference baseline generated successfully!")
        print(f"  Saved to: {gcs_path}")
    else:
        print("✗ Failed to generate reference baseline")
