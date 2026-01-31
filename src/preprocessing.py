"""Image preprocessing module for MedX-Detect
Handles image loading, denoising, CLAHE enhancement, normalization, and resizing
"""

import cv2
import numpy as np
from pathlib import Path

def load_image(image_path: str, grayscale=True) -> np.ndarray:
    """Load image from file path"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

def denoise_image(image: np.ndarray, method='bilateral') -> np.ndarray:
    """Apply denoising to reduce noise in medical images"""
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    return image

def apply_clahe(image: np.ndarray, clip_limit=2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)

def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Normalize image brightness"""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def resize_image(image: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def preprocess_image(image_path: str, size=(224, 224)) -> np.ndarray:
    """Complete preprocessing pipeline"""
    img = load_image(image_path)
    img = denoise_image(img)
    img = apply_clahe(img)
    img = normalize_brightness(img)
    img = resize_image(img, size)
    return img
