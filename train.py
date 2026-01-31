"""Training script for MedX-Detect brain tumor detection model"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import create_resnet50_model
from preprocessing import preprocess_image

def train_model(data_dir, epochs=10, batch_size=16):
    """
    Train ResNet50 model for brain tumor detection
    
    Args:
        data_dir: Path to dataset with train/val/test folders
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    
    # Paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        raise ValueError(f"Dataset directories not found. Expected: train, val, test")
    
    # Create model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    print(model.summary())
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    model_path = 'models/resnet50_braintumor.h5'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, history

if __name__ == "__main__":
    # Default dataset path
    data_dir = "./data/BrainTumor"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found at {data_dir}")
        sys.exit(1)
    
    train_model(data_dir, epochs=10, batch_size=16)
