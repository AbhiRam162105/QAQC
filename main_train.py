"""
Main script for training QAQC models on heart sound data.

Usage:
    python main_train.py --csv_path data/your_data.csv --output_dir results/
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extraction import extract_comprehensive_features, preprocess_pcg, extract_center_segment
from train_models import QAQCModelTrainer, create_feature_columns_from_extraction

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
        required_cols = ['recording_path', 'qaqc']
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


def extract_features_from_data(df: pd.DataFrame, 
                              sample_limit: int = None,
                              target_duration: float = 6.0) -> pd.DataFrame:
    """
    Extract features from audio files in the dataframe.
    
    Args:
        df: DataFrame with recording paths
        sample_limit: Limit number of samples to process (for testing)
        target_duration: Target segment duration in seconds
    
    Returns:
        DataFrame with extracted features
    """
    if sample_limit:
        df = df.head(sample_limit)
        logger.info(f"Processing limited sample of {sample_limit} records")
    
    features_list = []
    feature_columns = create_feature_columns_from_extraction()
    
    logger.info("Extracting features from audio files...")
    
    for idx, row in df.iterrows():
        try:
            # Load audio file
            recording_path = row['recording_path']
            
            # Check if file exists
            if not os.path.exists(recording_path):
                logger.warning(f"File not found: {recording_path}")
                # Create empty feature dict
                features = {col: 0.0 for col in feature_columns}
                features['file_exists'] = False
            else:
                # Use existing parameters from your original code
                import librosa
                source_fs = int(row.get('source_fs', 4000))
                
                # Load full signal
                full_signal, fs = librosa.load(recording_path, sr=source_fs)
                
                # Preprocess
                pcg = preprocess_pcg(full_signal, original_fs=fs, resample_fs=1450, band=(20, 720))
                
                # Extract center segment
                segment, start_time, end_time = extract_center_segment(
                    pcg, fs=1450, target_duration=target_duration
                )
                
                # Extract comprehensive features
                features = extract_comprehensive_features(segment, fs=1450)
                features['file_exists'] = True
                features['center_start_time'] = start_time
                features['center_end_time'] = end_time
            
            # Add metadata
            for col in df.columns:
                if col not in features:
                    features[col] = row[col]
            
            features_list.append(features)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} files")
                
        except Exception as e:
            logger.error(f"Error processing {row['recording_path']}: {e}")
            # Create empty feature dict for failed files
            features = {col: 0.0 for col in feature_columns}
            features['file_exists'] = False
            for col in df.columns:
                if col not in features:
                    features[col] = row[col]
            features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
    
    return features_df


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
    parser = argparse.ArgumentParser(description='Train QAQC models for heart sound quality assessment')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with recording paths and QAQC labels')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results and models')
    parser.add_argument('--sample_limit', type=int, default=None,
                       help='Limit number of samples for testing (optional)')
    parser.add_argument('--target_duration', type=float, default=6.0,
                       help='Target segment duration in seconds')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Setup logging to file
    log_file = output_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info("Starting QAQC model training")
    logger.info(f"Arguments: {args}")
    
    try:
        # Load data
        df = load_and_validate_data(args.csv_path)
        
        # Extract features
        features_df = extract_features_from_data(
            df, 
            sample_limit=args.sample_limit,
            target_duration=args.target_duration
        )
        
        # Save features
        features_df.to_csv(output_dir / 'extracted_features.csv', index=False)
        
        # Prepare for training
        feature_columns = create_feature_columns_from_extraction()
        
        # Filter out missing features and files that don't exist
        valid_features = features_df[features_df['file_exists'] == True]
        logger.info(f"Using {len(valid_features)} valid samples for training")
        
        if len(valid_features) < 10:
            raise ValueError("Not enough valid samples for training")
        
        # Initialize trainer
        trainer = QAQCModelTrainer(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            valid_features, feature_columns, 'qaqc'
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
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
