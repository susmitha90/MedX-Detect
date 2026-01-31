"""Unit tests for preprocessing module"""
import unittest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    load_image,
    denoise_image,
    apply_clahe,
    normalize_brightness,
    resize_image,
    preprocess_image
)

class TestPreprocessing(unittest.TestCase):
    """Test cases for image preprocessing functions"""
    
    def setUp(self):
        """Create a test image for testing"""
        # Create a synthetic grayscale test image (100x100)
        self.test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.test_image_path = 'test_image.jpg'
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_load_image(self):
        """Test image loading function"""
        img = load_image(self.test_image_path)
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 2)  # Grayscale image
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
    
    def test_load_image_invalid_path(self):
        """Test loading image with invalid path"""
        with self.assertRaises(ValueError):
            load_image('nonexistent_image.jpg')
    
    def test_denoise_bilateral(self):
        """Test bilateral denoising"""
        denoised = denoise_image(self.test_image, method='bilateral')
        self.assertEqual(denoised.shape, self.test_image.shape)
        self.assertIsInstance(denoised, np.ndarray)
    
    def test_denoise_gaussian(self):
        """Test gaussian denoising"""
        denoised = denoise_image(self.test_image, method='gaussian')
        self.assertEqual(denoised.shape, self.test_image.shape)
        self.assertIsInstance(denoised, np.ndarray)
    
    def test_apply_clahe(self):
        """Test CLAHE contrast enhancement"""
        enhanced = apply_clahe(self.test_image)
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Check if values are in valid range
        self.assertGreaterEqual(enhanced.min(), 0)
        self.assertLessEqual(enhanced.max(), 255)
    
    def test_normalize_brightness(self):
        """Test brightness normalization"""
        normalized = normalize_brightness(self.test_image)
        self.assertEqual(normalized.shape, self.test_image.shape)
        # Check normalization to 0-255 range
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 255)
    
    def test_resize_image(self):
        """Test image resizing"""
        target_size = (224, 224)
        resized = resize_image(self.test_image, size=target_size)
        self.assertEqual(resized.shape, target_size)
    
    def test_resize_different_size(self):
        """Test resizing to different dimensions"""
        target_size = (128, 128)
        resized = resize_image(self.test_image, size=target_size)
        self.assertEqual(resized.shape, target_size)
    
    def test_preprocess_image_pipeline(self):
        """Test complete preprocessing pipeline"""
        processed = preprocess_image(self.test_image_path, size=(224, 224))
        # Check output shape
        self.assertEqual(processed.shape, (224, 224))
        # Check output type
        self.assertIsInstance(processed, np.ndarray)
        # Check value range
        self.assertGreaterEqual(processed.min(), 0)
        self.assertLessEqual(processed.max(), 255)

if __name__ == '__main__':
    unittest.main()
