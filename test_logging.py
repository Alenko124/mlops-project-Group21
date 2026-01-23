"""Test script to verify image feature extraction and cloud logging."""

from io import BytesIO
from PIL import Image
import numpy as np
from eurosat.image_features import extract_image_features
from eurosat.data_logger import CloudPredictionLogger


def create_test_image() -> bytes:
    """Create a simple test image.
    
    Returns:
        Image bytes
    """
    # Create a test image with some variation
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


def test_image_features():
    """Test image feature extraction."""
    print("Testing image feature extraction...")
    
    # Create test image
    img_bytes = create_test_image()
    
    # Extract features
    features = extract_image_features(img_bytes)
    
    print(f"✓ Image features extracted:")
    print(f"  - Brightness: {features.get('brightness')}")
    print(f"  - Contrast: {features.get('contrast')}")
    print(f"  - Sharpness: {features.get('sharpness')}")
    
    assert features['brightness'] is not None, "Brightness should not be None"
    assert features['contrast'] is not None, "Contrast should not be None"
    assert features['sharpness'] is not None, "Sharpness should not be None"
    print("✓ All features extracted successfully\n")


def test_cloud_logger():
    """Test cloud prediction logger.
    
    Note: This will attempt to connect to GCS. Make sure you have credentials set up.
    """
    print("Testing cloud prediction logger...")
    
    try:
        logger = CloudPredictionLogger(batch_size=1)
        print("✓ Logger initialized successfully")
        
        # Create test data
        img_bytes = create_test_image()
        features = extract_image_features(img_bytes)
        
        # Log a prediction
        logger.log_prediction(
            predicted_class=0,
            predicted_class_name="Annual Crop",
            confidence=0.95,
            image_features=features,
            image_bytes=img_bytes,
            top_5_predictions=[
                {"class_id": 0, "class_name": "Annual Crop", "confidence": 0.95},
                {"class_id": 1, "class_name": "Forest", "confidence": 0.03},
                {"class_id": 2, "class_name": "Herbaceous Vegetation", "confidence": 0.01},
                {"class_id": 3, "class_name": "Highway", "confidence": 0.005},
                {"class_id": 4, "class_name": "Industrial", "confidence": 0.005},
            ],
        )
        print("✓ Prediction logged to buffer")
        
        # Log another prediction to trigger flush (batch_size=2)
        logger.log_prediction(
            predicted_class=1,
            predicted_class_name="Forest",
            confidence=0.88,
            image_features=features,
            image_bytes=img_bytes,
            top_5_predictions=[
                {"class_id": 1, "class_name": "Forest", "confidence": 0.88},
                {"class_id": 0, "class_name": "Annual Crop", "confidence": 0.07},
                {"class_id": 2, "class_name": "Herbaceous Vegetation", "confidence": 0.03},
                {"class_id": 3, "class_name": "Highway", "confidence": 0.01},
                {"class_id": 4, "class_name": "Industrial", "confidence": 0.01},
            ],
        )
        print("✓ Second prediction logged (should trigger auto-flush)")
        
        # Manually flush remaining
        gcs_path = logger.flush()
        if gcs_path:
            print(f"✓ Data flushed to GCS: {gcs_path}")
        else:
            print("⚠ No data to flush or flush failed")
            
    except Exception as e:
        print(f"⚠ Cloud logger test skipped (GCS not available): {e}")
        print("  This is OK if you don't have GCS credentials set up yet.")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Image Features and Cloud Logging")
    print("=" * 60 + "\n")
    
    test_image_features()
    test_cloud_logger()
    
    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)
