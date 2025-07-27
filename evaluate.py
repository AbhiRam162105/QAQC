"""
Model evaluation script for QAQC heart sound assessment.

Usage:
    python evaluate.py --model_path models/best_model.pkl --data_path data/test_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import extract_comprehensive_features, preprocess_pcg
from train_models import create_feature_columns_from_extraction


def load_model(model_path):
    """Load trained model and components."""
    model_data = joblib.load(model_path)
    return (model_data['model'], model_data['scaler'], 
            model_data['label_encoder'], model_data['feature_columns'])


def evaluate_model(model, scaler, label_encoder, feature_columns, test_data):
    """Evaluate model on test data."""
    
    # Prepare test features
    X_test = test_data[feature_columns].fillna(0)
    y_test = test_data['qaqc']
    
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_test_encoded, y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'true_labels': y_test_encoded
    }


def plot_results(results, label_encoder, output_dir):
    """Create evaluation plots."""
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report heatmap
    report_df = pd.DataFrame(results['classification_report']).iloc[:-1, :].T
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:, :-1].astype(float), 
                annot=True, fmt='.3f', cmap='RdYlBu_r')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate QAQC model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model, scaler, label_encoder, feature_columns = load_model(args.model_path)
    
    print("Loading test data...")
    test_data = pd.read_csv(args.data_path)
    
    print("Evaluating model...")
    results = evaluate_model(model, scaler, label_encoder, feature_columns, test_data)
    
    # Print results
    print(f"\nModel Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for class_name in label_encoder.classes_:
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            print(f"{class_name:12s} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    # Create plots
    plot_results(results, label_encoder, output_dir)
    
    # Save detailed results
    results_summary = {
        'accuracy': results['accuracy'],
        'classification_report': results['classification_report']
    }
    
    import json
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
