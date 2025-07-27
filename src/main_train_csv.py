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
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Show first few column names and data types
        logger.info("Dataset overview:")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        
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
        
        # Check for any completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")
        
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
        'path', 'recording_path.1', 'fold'  # Added 'fold' to exclude list
    ]
    
    # Get all columns
    all_cols = df.columns.tolist()
    logger.info(f"Total columns in dataset: {len(all_cols)}")
    
    # Get numeric columns that are not in exclude list
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Numeric columns found: {len(numeric_cols)}")
    logger.info(f"Excluded columns: {len(exclude_cols)}")
    logger.info(f"Identified {len(feature_cols)} feature columns:")
    for col in feature_cols:
        # Check for problematic values
        col_data = df[col]
        inf_count = np.isinf(col_data).sum()
        nan_count = col_data.isna().sum()
        zero_count = (col_data == 0).sum()
        
        logger.info(f"  - {col}: {len(col_data)} values, {nan_count} NaN, {inf_count} inf, {zero_count} zeros")
        
        # Show basic statistics
        if len(col_data.dropna()) > 0:
            logger.debug(f"    Range: [{col_data.min():.4f}, {col_data.max():.4f}], Mean: {col_data.mean():.4f}")
    
    return feature_cols


def preprocess_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Preprocess feature data."""
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    logger.info("Starting feature preprocessing...")
    
    # Clean each feature column
    for col in feature_columns:
        if col in df_processed.columns:
            original_count = len(df_processed)
            
            # Check for infinite values
            inf_count = np.isinf(df_processed[col]).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in {col}")
                # Replace infinite values with NaN
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
            
            # Check for extremely large values (potential overflow)
            large_threshold = 1e10
            large_count = (np.abs(df_processed[col]) > large_threshold).sum()
            if large_count > 0:
                logger.warning(f"Found {large_count} extremely large values in {col}")
                # Cap extremely large values
                df_processed[col] = np.clip(df_processed[col], -large_threshold, large_threshold)
            
            # Handle missing values
            missing_count = df_processed[col].isna().sum()
            if missing_count > 0:
                median_val = df_processed[col].median()
                if pd.isna(median_val):  # If all values are NaN, use 0
                    median_val = 0.0
                df_processed[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {missing_count} missing values in {col} with median: {median_val:.4f}")
            
            # Remove outliers using IQR method
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Only apply if there's variance
                lower_bound = Q1 - 3 * IQR  # Use 3*IQR instead of 1.5 for less aggressive outlier removal
                upper_bound = Q3 + 3 * IQR
                
                outlier_count = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
                if outlier_count > 0:
                    logger.info(f"Capping {outlier_count} outliers in {col} to range [{lower_bound:.4f}, {upper_bound:.4f}]")
                    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
            
            # Final check for any remaining invalid values
            invalid_count = (~np.isfinite(df_processed[col])).sum()
            if invalid_count > 0:
                logger.warning(f"Replacing {invalid_count} remaining invalid values in {col} with 0")
                df_processed[col] = df_processed[col].fillna(0)
                df_processed[col] = np.where(np.isfinite(df_processed[col]), df_processed[col], 0)
    
    # Remove rows with missing QAQC labels
    original_len = len(df_processed)
    df_processed = df_processed.dropna(subset=['qaqc'])
    removed_count = original_len - len(df_processed)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with missing QAQC labels")
    
    logger.info(f"After preprocessing: {len(df_processed)} samples with {len(feature_columns)} features")
    
    # Final data validation
    for col in feature_columns:
        if col in df_processed.columns:
            if not np.all(np.isfinite(df_processed[col])):
                logger.error(f"Column {col} still contains invalid values after preprocessing!")
            else:
                logger.debug(f"Column {col} validation passed")
    
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
