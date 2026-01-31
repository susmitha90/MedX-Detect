"""Inference script for MedX-Detect - make predictions on new images"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import load_model, predict
from preprocessing import preprocess_image

def predict_tumor(image_path, model_path='models/resnet50_braintumor.h5'):
    """
    Predict whether MRI image contains brain tumor
    
    Args:
        image_path: Path to MRI image
        model_path: Path to trained model
    
    Returns:
        Prediction (0: No tumor, 1: Tumor) and confidence
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    image = np.repeat(image, 3, axis=-1)    # Convert grayscale to RGB
    
    # Load model and make prediction
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    prediction = predict(model, image)
    confidence = float(prediction[0][0])
    predicted_class = 1 if confidence > 0.5 else 0
    
    # Return results
    result = {
        'image': image_path,
        'prediction': 'TUMOR' if predicted_class == 1 else 'NO TUMOR',
        'confidence': confidence,
        'class': predicted_class
    }
    
    return result

def batch_predict(image_dir, model_path='models/resnet50_braintumor.h5'):
    """
    Make predictions on all images in a directory
    
    Args:
        image_dir: Directory containing images
        model_path: Path to trained model
    
    Returns:
        List of predictions
    """
    
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for ext in image_extensions:
        for image_file in Path(image_dir).glob(f'*{ext}'):
            try:
                result = predict_tumor(str(image_file), model_path)
                results.append(result)
                print(f"✓ {result['image']}: {result['prediction']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ {image_file}: {str(e)}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("   or: python inference.py --batch <directory>")
        sys.exit(1)
    
    if sys.argv[1] == '--batch' and len(sys.argv) > 2:
        results = batch_predict(sys.argv[2])
        print(f"\nProcessed {len(results)} images")
    else:
        result = predict_tumor(sys.argv[1])
        print(f"\nResult: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
