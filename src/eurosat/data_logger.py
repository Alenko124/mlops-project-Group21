"""Data logging module for saving predictions and image features to cloud."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import storage
from loguru import logger


class CloudPredictionLogger:
    """Logs predictions with image features to Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: str = "mlops-group21",
        predictions_folder: str = "predictions/data_logs",
        batch_size: int = 1,
    ):
        """Initialize cloud prediction logger.

        Args:
            bucket_name: GCS bucket name
            predictions_folder: Folder path in GCS for storing logs
            batch_size: Number of predictions to batch before uploading
        """
        self.bucket_name = bucket_name
        self.predictions_folder = predictions_folder
        self.batch_size = batch_size
        self.buffer: List[Dict[str, Any]] = []
        self.client = storage.Client()
        logger.info(f"Initialized CloudPredictionLogger for bucket: {bucket_name}")

    def log_prediction(
        self,
        predicted_class: int,
        predicted_class_name: str,
        confidence: float,
        image_features: Dict[str, Any],
        top_5_predictions: Optional[List[Dict[str, Any]]] = None,
        true_label: Optional[int] = None,
    ) -> None:
        """Log a prediction with image features to buffer.

        Args:
            predicted_class: Predicted class ID (0-9)
            predicted_class_name: Name of predicted class
            confidence: Model confidence score
            image_features: Dictionary with brightness, contrast, sharpness
            image_bytes: Raw image bytes
            top_5_predictions: Optional list of top 5 predictions
            true_label: Optional ground truth label
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": timestamp,
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "image_features": image_features,
            "true_label": true_label,
        }

        self.buffer.append(record)
        logger.debug(f"Buffered prediction for class: {predicted_class_name}")

        # Auto-flush if buffer reaches batch size
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> Optional[str]:
        """Upload buffered predictions to GCS.

        Returns:
            GCS path where data was saved, or None if buffer was empty
        """
        if not self.buffer:
            logger.debug("Buffer is empty, nothing to flush")
            return None

        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            bucket = self.client.bucket(self.bucket_name)

            # Upload each record as a separate JSON file
            for idx, record in enumerate(self.buffer):
                filename = f"{timestamp}_{len(self.buffer)}_record_{idx}.json"
                blob_path = f"{self.predictions_folder}/{filename}"

                blob = bucket.blob(blob_path)
                blob.upload_from_string(json.dumps(record, indent=2), content_type="application/json")
                logger.debug(f"Uploaded record {idx} to: {blob_path}")

            record_count = len(self.buffer)
            first_blob_path = f"{self.predictions_folder}/{timestamp}_{record_count}_record_0.json"
            self.buffer = []

            logger.info(
                f"Successfully flushed {record_count} predictions to GCS: {self.predictions_folder}/{timestamp}_*.json"
            )
            return first_blob_path

        except Exception as e:
            logger.error(f"Failed to flush predictions to GCS: {e}")
            return None

    def __del__(self):
        """Ensure data is flushed before object is destroyed."""
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Error during cleanup flush: {e}")
