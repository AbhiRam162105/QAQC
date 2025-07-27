"""
Modified main script for training QAQC models using existing features in CSV.

Usage:
    python main_train_csv.py --csv_path data/your_data.csv --output_dir results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_models import QAQCModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: str) -> Path:
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'reports').mkdir(exist_ok=True)
    
    return output_path


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the input CSV data."""
    logger.info(f"Loading data from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Check required columns
        required_cols = ['qaqc']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check QAQC label distribution
        qaqc_counts = df['qaqc'].value_counts()
        logger.info("QAQC label distribution:")
        for label, count in qaqc_counts.items():
            logger.info(f"  {label}: {count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def identify_feature_columns(df: pd.DataFrame) -> list:
    """Identify numeric feature columns in the dataframe."""
    
    # Exclude non-feature columns
    exclude_cols = [
        'qaqc', 'dataset', 'subject', 'recording_id', 'recording_path', 
        'segment_id', 'segment_path', 'age_group', 'sex', 'condition', 
        'disease', 'murmur_location', 'systolic_murmur_type', 
        'diastolic_murmur_type', 'pregnancy_status', 'smoking_status',
        'path', 'recording_path.1'
    ]
    
    # Get numeric columns that are not in exclude list
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Identified {len(feature_cols)} feature columns:")
    for col in feature_cols:
        logger.info(f"  - {col}")
    
    return feature_cols


def preprocess_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Preprocess feature data."""
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Fill missing values with median for numeric columns
    for col in feature_columns:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {df_processed[col].isna().sum()} missing values in {col} with median: {median_val:.4f}")
    
    # Remove rows with missing QAQC labels
    df_processed = df_processed.dropna(subset=['qaqc'])
    
    logger.info(f"After preprocessing: {len(df_processed)} samples with {len(feature_columns)} features")
    
    return df_processed


def plot_results(trainer: QAQCModelTrainer, 
                results: dict, 
                output_dir: Path) -> None:
    """Create visualization plots for model results."""
    
    # Model comparison plot
    plt.figure(figsize=(12, 6))
    
    model_names = []
    test_accuracies = []
    cv_scores = []
    cv_stds = []
    
    for model_name, result in results.items():
        model_names.append(model_name.replace('_', ' ').title())
        test_accuracies.append(result['test_results']['accuracy'])
        cv_scores.append(result['cv_results']['mean_cv_score'])
        cv_stds.append(result['cv_results']['std_cv_score'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    plt.errorbar(x + width/2, cv_scores, yerr=cv_stds, fmt='o', 
                capsize=5, label='CV Score ± std', color='red')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance for tree-based models
    for model_name in ['random_forest', 'xgboost']:
        if model_name in results:
            importance = trainer.get_feature_importance(model_name)
            if importance:
                plt.figure(figsize=(10, 8))
                features = list(importance.keys())[:20]  # Top 20 features
                values = list(importance.values())[:20]
                
                plt.barh(range(len(features)), values)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importance - {model_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(output_dir / 'plots' / f'feature_importance_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # Confusion matrices
    for model_name, result in results.items():
        conf_matrix = result['test_results']['confusion_matrix']
        class_names = result['test_results']['class_names']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_dir / 'plots' / f'confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def save_results(results: dict, 
                trainer: QAQCModelTrainer,
                output_dir: Path) -> None:
    """Save results and trained models."""
    
    # Save results summary
    summary = {}
    for model_name, result in results.items():
        summary[model_name] = {
            'test_accuracy': float(result['test_results']['accuracy']),
            'cv_mean': float(result['cv_results']['mean_cv_score']),
            'cv_std': float(result['cv_results']['std_cv_score']),
            'classification_report': result['test_results']['classification_report']
        }
    
    with open(output_dir / 'reports' / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(output_dir / 'reports' / 'detailed_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'test_results': {
                    'accuracy': float(result['test_results']['accuracy']),
                    'classification_report': result['test_results']['classification_report'],
                    'confusion_matrix': result['test_results']['confusion_matrix'].tolist(),
                    'class_names': result['test_results']['class_names'].tolist()
                },
                'cv_results': result['cv_results']
            }
        json.dump(json_results, f, indent=2)
    
    # Save best model
    best_model = max(results.keys(), 
                    key=lambda x: results[x]['test_results']['accuracy'])
    trainer.save_model(best_model, output_dir / 'models' / f'best_model_{best_model}.pkl')
    
    logger.info(f"Best model: {best_model} with accuracy: {results[best_model]['test_results']['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train QAQC models using existing CSV features')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with features and QAQC labels')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results and models')
    parser.add_argument('--sample_limit', type=int, default=None,
                       help='Limit number of samples for testing (optional)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Setup logging to file
    log_file = output_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info("Starting QAQC model training using existing CSV features")
    logger.info(f"Arguments: {args}")
    
    try:
        # Load data
        df = load_and_validate_data(args.csv_path)
        
        # Limit samples if requested
        if args.sample_limit:
            df = df.head(args.sample_limit)
            logger.info(f"Limited to {args.sample_limit} samples")
        
        # Identify feature columns automatically
        feature_columns = identify_feature_columns(df)
        
        if len(feature_columns) == 0:
            raise ValueError("No numeric feature columns found in the data")
        
        # Preprocess features
        df_processed = preprocess_features(df, feature_columns)
        
        logger.info(f"Using {len(df_processed)} samples for training")
        
        if len(df_processed) < 10:
            raise ValueError(f"Not enough samples for training. Found {len(df_processed)}, need at least 10")
        
        # Save processed features
        df_processed.to_csv(output_dir / 'processed_features.csv', index=False)
        
        # Initialize trainer
        trainer = QAQCModelTrainer(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            df_processed, feature_columns, 'qaqc'
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train all models
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Create visualizations
        plot_results(trainer, results, output_dir)
        
        # Save results
        save_results(results, trainer, output_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("QAQC MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset: {args.csv_path}")
        print(f"Samples used: {len(df_processed)}")
        print(f"Features: {len(feature_columns)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print("\nModel Performance:")
        for model_name, result in sorted(results.items(), 
                                       key=lambda x: x[1]['test_results']['accuracy'], 
                                       reverse=True):
            acc = result['test_results']['accuracy']
            cv_mean = result['cv_results']['mean_cv_score']
            cv_std = result['cv_results']['std_cv_score']
            print(f"  {model_name:20s}: {acc:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
        
        best_model = max(results.keys(), key=lambda x: results[x]['test_results']['accuracy'])
        print(f"\nBest Model: {best_model}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
