"""Image feature extraction for monitoring data drift."""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any


def extract_image_features(image_bytes: bytes) -> Dict[str, Any]:
    """Extract statistical features from an image.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Dictionary containing image features:
            - brightness: Average pixel intensity (0-255)
            - contrast: Standard deviation of pixel values
            - sharpness: Sharpness metric using Laplacian kernel
    """
    try:
        # Open image
        image = Image.open(BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)

        # Extract features
        brightness = float(np.mean(img_array))
        contrast = float(np.std(img_array))
        sharpness = _calculate_sharpness(img_array)

        return {
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "sharpness": round(sharpness, 4),
        }
    except Exception as e:
        return {
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "error": str(e),
        }


def _calculate_sharpness(img_array: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian kernel.

    Args:
        img_array: Image as numpy array (H, W, C)

    Returns:
        Sharpness score (higher = sharper)
    """
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Apply Laplacian filter
    sharpness_map = _apply_kernel(gray, laplacian_kernel)

    # Return variance of Laplacian (higher = sharper)
    return float(np.var(sharpness_map))


def _apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply convolution kernel to image.

    Args:
        image: Image array (H, W)
        kernel: Convolution kernel (K, K)

    Returns:
        Convolution result
    """
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Pad image
    padded = np.pad(image, pad, mode="edge")

    # Apply convolution
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + kernel_size, j : j + kernel_size]
            result[i, j] = np.sum(region * kernel)

    return result
