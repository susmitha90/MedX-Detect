"""ResNet50 Model for brain tumor detection using transfer learning"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create ResNet50 model with transfer learning
    
    Args:
        input_shape: Input image shape (224, 224, 3) for RGB
        num_classes: Number of output classes (2 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    # Load pretrained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model(model_path):
    """Load trained model from file"""
    return keras.models.load_model(model_path)

def predict(model, image_array):
    """Make prediction on preprocessed image"""
    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=0)
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    return model.predict(image_array)
