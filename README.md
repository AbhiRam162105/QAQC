# Heart Sound QAQC Model

A comprehensive Quality Assurance/Quality Control system for heart sound recordings using machine learning and deep learning approaches.

## Project Overview

This project implements multiple approaches for automatically assessing the quality of phonocardiogram (PCG) recordings:

1. **Feature-Based Models**: Traditional ML using extracted audio features
2. **Deep Learning Models**: CNN-based approaches on spectrograms and raw audio
3. **Hybrid Models**: Multi-modal fusion of features and deep learning

## Dataset Structure

The project expects a CSV file with the following key columns:
- `recording_path`: Path to audio files
- `spectral_flatness`: Audio quality metric
- `qaqc`: Quality labels (very bad, bad, ok, good, very good)
- Additional metadata: age, sex, condition, etc.

## Features Extracted

- Spectral flatness
- Envelope variance (Hilbert transform)
- Signal-to-noise ratio
- Zero crossing rate
- Spectral centroid/rolloff
- MFCC features
- Spectral bandwidth/contrast

## Models Implemented

### Traditional ML
- Random Forest
- XGBoost
- SVM with RBF kernel
- Logistic Regression

### Deep Learning
- 1D CNN for raw audio
- 2D CNN for spectrograms
- ResNet-based transfer learning

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: 
   ```python
   python src/data_preprocessing.py --csv_path your_data.csv
   ```

2. **Feature Extraction**:
   ```python
   python src/feature_extraction.py --data_dir processed_data/
   ```

3. **Training Models**:
   ```python
   python src/train_models.py --config configs/baseline.yaml
   ```

4. **Evaluation**:
   ```python
   python src/evaluate.py --model_path models/best_model.pkl
   ```

## Project Structure

```
QAQC/
├── src/                    # Source code
├── data/                   # Data directory
├── models/                 # Trained models
├── configs/                # Configuration files
├── notebooks/              # Jupyter notebooks
├── results/                # Results and plots
└── tests/                  # Unit tests
```

## Performance Metrics

- Accuracy, Precision, Recall, F1-score
- Cohen's Kappa (inter-rater agreement)
- Confusion matrices
- ROC curves for binary classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request
