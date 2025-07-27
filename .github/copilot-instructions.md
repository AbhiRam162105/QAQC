<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Heart Sound QAQC Project Instructions

This is a machine learning project for Quality Assurance/Quality Control of heart sound recordings (phonocardiograms).

## Project Context
- Working with medical audio data (PCG signals)
- Multi-class classification: very bad, bad, ok, good, very good
- Features: spectral_flatness, envelope_variance, signal quality metrics
- Audio processing at 1450Hz sampling rate
- 6-second center segments for analysis

## Code Guidelines
- Use librosa for audio processing and feature extraction
- Implement proper cross-validation for medical data
- Handle class imbalance with appropriate techniques
- Include comprehensive logging and error handling
- Follow scikit-learn conventions for model interfaces
- Use proper medical data validation techniques

## Key Components
- Feature extraction from PCG signals
- Traditional ML models (RF, XGBoost, SVM)
- Deep learning on spectrograms and raw audio
- Model evaluation with medical-appropriate metrics
- Visualization of results and model performance

## Data Handling
- Ensure patient-level splits for validation
- Handle missing values appropriately
- Normalize features consistently
- Implement proper data augmentation for audio
