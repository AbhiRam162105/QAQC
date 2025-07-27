"""
QAQC Model Training Module

This module contains classes and functions for training various
machine learning models for heart sound quality assessment.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class QAQCModelTrainer:
    """
    Comprehensive trainer for QAQC models with multiple algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.results = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    feature_columns: List[str],
                    target_column: str = 'qaqc',
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper preprocessing.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_column])
        
        # Extract features and target
        X = df_clean[feature_columns].fillna(0)  # Fill missing features with 0
        y = df_clean[target_column]
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=self.random_state, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def setup_models(self) -> Dict[str, Any]:
        """
        Initialize all models with optimized parameters.
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss'
            ),
            
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=self.random_state,
                probability=True
            ),
            
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                multi_class='ovr'
            )
        }
        
        self.models = models
        return models
    
    def train_model(self, 
                   model_name: str,
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   scale_features: bool = True) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            scale_features: Whether to scale features
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            self.scalers[model_name] = None
        
        # Handle class weights for XGBoost
        if model_name == 'xgboost':
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = np.array([class_weights[i] for i in y_train])
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)
        
        return model
    
    def evaluate_model(self, 
                      model_name: str,
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scale test features if scaler exists
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get class names for reporting
        class_names = self.label_encoder.classes_
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'class_names': class_names
        }
        
        return results
    
    def cross_validate_model(self, 
                           model_name: str,
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Scale features if scaler exists
        if scaler is not None:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Stratified K-Fold for balanced splits
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
        
        results = {
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist()
        }
        
        return results
    
    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        X_test: np.ndarray, 
                        y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate all models.
        
        Returns:
            Dictionary with results for all models
        """
        self.setup_models()
        all_results = {}
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            
            # Train model
            self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            test_results = self.evaluate_model(model_name, X_test, y_test)
            
            # Cross-validation
            cv_results = self.cross_validate_model(model_name, X_train, y_train)
            
            # Combine results
            all_results[model_name] = {
                'test_results': test_results,
                'cv_results': cv_results
            }
            
            print(f"{model_name} - Test Accuracy: {test_results['accuracy']:.4f}, "
                  f"CV Score: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
        
        self.results = all_results
        return all_results
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of feature importances
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return None
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model and its scaler.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers[model_name],
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Dictionary containing model components
        """
        model_data = joblib.load(filepath)
        return model_data


def create_feature_columns_from_extraction() -> List[str]:
    """
    Define the feature columns that will be extracted by feature_extraction.py
    """
    feature_columns = [
        # Original features
        'spectral_flatness',
        'envelope_variance',
        
        # Signal quality
        'snr',
        
        # Spectral features
        'spectral_centroid',
        'spectral_bandwidth', 
        'spectral_rolloff',
        'spectral_contrast_mean',
        'spectral_contrast_std',
        'zero_crossing_rate',
        
        # MFCC features
        'mfcc_mean',
        'mfcc_std',
        'mfcc_skew',
        'mfcc_kurtosis',
        'mfcc_1_mean', 'mfcc_1_std',
        'mfcc_2_mean', 'mfcc_2_std',
        'mfcc_3_mean', 'mfcc_3_std',
        'mfcc_4_mean', 'mfcc_4_std',
        'mfcc_5_mean', 'mfcc_5_std',
        
        # Temporal features
        'signal_mean',
        'signal_std',
        'signal_skew', 
        'signal_kurtosis',
        'rms_energy',
        'peak_amplitude',
        'crest_factor',
        'dynamic_range',
        
        # Frequency domain
        'spectral_entropy',
        'dominant_frequency',
        'energy_band_20_100',
        'energy_band_100_200', 
        'energy_band_200_400',
        'energy_band_400_720',
        
        # Signal properties
        'signal_length',
        'duration_seconds'
    ]
    
    return feature_columns


if __name__ == "__main__":
    print("QAQC Model Trainer loaded successfully!")
    print("Available models: Random Forest, XGBoost, SVM, Logistic Regression")
    print("Use QAQCModelTrainer class for training and evaluation.")
