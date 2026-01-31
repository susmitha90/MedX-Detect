evaluate.py"""Model evaluation script with metrics and visualizations"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_data_dir, batch_size=16, save_dir='evaluation_results'):
    """
    Comprehensive model evaluation with multiple metrics
    
    Args:
        model_path: Path to trained model (.h5 file)
        test_data_dir: Directory containing test data (cancer/noncancer folders)
        batch_size: Batch size for evaluation
        save_dir: Directory to save evaluation results
    
    Returns:
        Dictionary with all evaluation metrics
    """
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Prepare test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\nEvaluating on {test_generator.samples} test samples...")
    
    # Get predictions
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_prob),
        'total_samples': len(y_true),
        'cancer_samples': int(sum(y_true)),
        'non_cancer_samples': int(len(y_true) - sum(y_true))
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"\nTest Samples: {metrics['total_samples']}")
    print(f"  - Cancer: {metrics['cancer_samples']}")
    print(f"  - Non-Cancer: {metrics['non_cancer_samples']}")
    print("="*60)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Cancer', 'Cancer']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # ROC Curve
    plot_roc_curve(y_true, y_pred_prob, metrics['auc_roc'], 
                   save_path=os.path.join(save_dir, 'roc_curve.png'))
    
    # Save metrics to JSON
    metrics_file = os.path.join(save_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Save predictions
    predictions_file = os.path.join(save_dir, 'predictions.csv')
    with open(predictions_file, 'w') as f:
        f.write("image_index,true_label,predicted_label,confidence\n")
        for i, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_pred_prob)):
            f.write(f"{i},{true},{pred},{float(prob[0]):.4f}\n")
    print(f"Predictions saved to {predictions_file}")
    
    return metrics

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Cancer', 'Cancer'],
                yticklabels=['Non-Cancer', 'Cancer'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_pred_prob, auc_score, save_path='roc_curve.png'):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_path> <test_data_dir>")
        print("Example: python evaluate.py models/resnet50_braintumor.h5 data/BrainTumor/test")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_dir = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory not found at {test_data_dir}")
        sys.exit(1)
    
    evaluate_model(model_path, test_data_dir)
