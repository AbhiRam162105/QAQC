"""
Heart Sound QAQC Feature Extraction Module

This module contains functions for extracting comprehensive features
from phonocardiogram (PCG) signals for quality assessment.
"""

import numpy as np
import librosa
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def preprocess_pcg(signal_data: np.ndarray, 
                   original_fs: int = 4000, 
                   resample_fs: int = 1450, 
                   band: Tuple[int, int] = (20, 720)) -> np.ndarray:
    """
    Preprocess PCG signal with resampling and bandpass filtering.
    
    Args:
        signal_data: Raw audio signal
        original_fs: Original sampling frequency
        resample_fs: Target sampling frequency
        band: Bandpass filter frequency range (low, high)
    
    Returns:
        Preprocessed PCG signal
    """
    # Resample if needed
    if original_fs != resample_fs:
        signal_data = librosa.resample(signal_data, orig_sr=original_fs, target_sr=resample_fs)
    
    # Bandpass filter
    nyquist = resample_fs / 2
    low, high = band
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal


def envelope_variance(signal_data: np.ndarray) -> float:
    """
    Calculate envelope variance using Hilbert transform.
    Higher variance indicates more dynamic signal.
    """
    try:
        analytic = signal.hilbert(signal_data)
        envelope = np.abs(analytic)
        return float(np.var(envelope))
    except:
        return 0.0


def calculate_snr(signal_data: np.ndarray, noise_floor_percentile: int = 10) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    Estimates noise floor from low-amplitude portions.
    """
    try:
        # Estimate noise floor from lower percentile of signal amplitudes
        noise_floor = np.percentile(np.abs(signal_data), noise_floor_percentile)
        signal_power = np.mean(signal_data**2)
        noise_power = noise_floor**2
        
        if noise_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)
    except:
        return 0.0


def spectral_features(signal_data: np.ndarray, fs: int = 1450) -> Dict[str, float]:
    """
    Extract comprehensive spectral features.
    """
    features = {}
    
    try:
        # Basic spectral features
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=signal_data, sr=fs)))
        features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=signal_data, sr=fs)))
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=signal_data, sr=fs)))
        features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=signal_data)))
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=signal_data, sr=fs)
        features['spectral_contrast_mean'] = float(np.mean(contrast))
        features['spectral_contrast_std'] = float(np.std(contrast))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(signal_data)
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
    except Exception as e:
        # Return default values if extraction fails
        for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                   'spectral_flatness', 'spectral_contrast_mean', 'spectral_contrast_std',
                   'zero_crossing_rate']:
            features[key] = 0.0
    
    return features


def mfcc_features(signal_data: np.ndarray, fs: int = 1450, n_mfcc: int = 13) -> Dict[str, float]:
    """
    Extract MFCC features and statistics.
    """
    features = {}
    
    try:
        mfccs = librosa.feature.mfcc(y=signal_data, sr=fs, n_mfcc=n_mfcc)
        
        # Statistical measures of MFCCs
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        features['mfcc_skew'] = float(stats.skew(mfccs.flatten()))
        features['mfcc_kurtosis'] = float(stats.kurtosis(mfccs.flatten()))
        
        # Individual MFCC coefficients (first 5)
        for i in range(min(5, n_mfcc)):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
    except Exception as e:
        # Return default values if extraction fails
        features.update({
            'mfcc_mean': 0.0, 'mfcc_std': 0.0, 'mfcc_skew': 0.0, 'mfcc_kurtosis': 0.0
        })
        for i in range(5):
            features[f'mfcc_{i+1}_mean'] = 0.0
            features[f'mfcc_{i+1}_std'] = 0.0
    
    return features


def temporal_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract temporal domain features.
    """
    features = {}
    
    try:
        # Basic statistics
        features['signal_mean'] = float(np.mean(signal_data))
        features['signal_std'] = float(np.std(signal_data))
        features['signal_skew'] = float(stats.skew(signal_data))
        features['signal_kurtosis'] = float(stats.kurtosis(signal_data))
        
        # Amplitude features
        features['rms_energy'] = float(np.sqrt(np.mean(signal_data**2)))
        features['peak_amplitude'] = float(np.max(np.abs(signal_data)))
        features['crest_factor'] = features['peak_amplitude'] / features['rms_energy'] if features['rms_energy'] > 0 else 0.0
        
        # Dynamic range
        features['dynamic_range'] = float(np.max(signal_data) - np.min(signal_data))
        
    except Exception as e:
        for key in ['signal_mean', 'signal_std', 'signal_skew', 'signal_kurtosis',
                   'rms_energy', 'peak_amplitude', 'crest_factor', 'dynamic_range']:
            features[key] = 0.0
    
    return features


def frequency_domain_features(signal_data: np.ndarray, fs: int = 1450) -> Dict[str, float]:
    """
    Extract frequency domain features using FFT.
    """
    features = {}
    
    try:
        # Compute FFT
        fft_vals = fft(signal_data)
        fft_magnitude = np.abs(fft_vals)
        freqs = fftfreq(len(signal_data), 1/fs)
        
        # Only use positive frequencies
        positive_freq_idx = freqs > 0
        freqs = freqs[positive_freq_idx]
        fft_magnitude = fft_magnitude[positive_freq_idx]
        
        # Spectral entropy
        power_spectrum = fft_magnitude**2
        power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-10))
        features['spectral_entropy'] = float(spectral_entropy)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = float(freqs[dominant_freq_idx])
        
        # Frequency bands energy distribution
        bands = [(20, 100), (100, 200), (200, 400), (400, 720)]
        total_energy = np.sum(power_spectrum)
        
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(power_spectrum[band_mask])
            features[f'energy_band_{low}_{high}'] = float(band_energy / total_energy) if total_energy > 0 else 0.0
        
    except Exception as e:
        for key in ['spectral_entropy', 'dominant_frequency']:
            features[key] = 0.0
        for low, high in [(20, 100), (100, 200), (200, 400), (400, 720)]:
            features[f'energy_band_{low}_{high}'] = 0.0
    
    return features


def extract_comprehensive_features(signal_data: np.ndarray, fs: int = 1450) -> Dict[str, float]:
    """
    Extract all features for QAQC assessment.
    
    Args:
        signal_data: Preprocessed PCG signal
        fs: Sampling frequency
    
    Returns:
        Dictionary containing all extracted features
    """
    all_features = {}
    
    # Envelope variance (from original code)
    all_features['envelope_variance'] = envelope_variance(signal_data)
    
    # Signal-to-noise ratio
    all_features['snr'] = calculate_snr(signal_data)
    
    # Spectral features
    all_features.update(spectral_features(signal_data, fs))
    
    # MFCC features
    all_features.update(mfcc_features(signal_data, fs))
    
    # Temporal features
    all_features.update(temporal_features(signal_data))
    
    # Frequency domain features
    all_features.update(frequency_domain_features(signal_data, fs))
    
    # Signal length and duration
    all_features['signal_length'] = len(signal_data)
    all_features['duration_seconds'] = len(signal_data) / fs
    
    return all_features


def extract_center_segment(signal_data: np.ndarray, 
                          fs: int = 1450, 
                          target_duration: float = 6.0) -> Tuple[np.ndarray, float, float]:
    """
    Extract center segment of specified duration from signal.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        target_duration: Target duration in seconds
    
    Returns:
        Tuple of (segment, start_time, end_time)
    """
    total_len = len(signal_data)
    total_duration = total_len / fs
    
    if total_duration > target_duration:
        center = total_len // 2
        half_len = int(target_duration * fs / 2)
        start = max(0, center - half_len)
        end = min(total_len, center + half_len)
        segment = signal_data[start:end]
        start_time = start / fs
        end_time = end / fs
    else:
        segment = signal_data
        start_time = 0.0
        end_time = total_duration
    
    return segment, start_time, end_time


if __name__ == "__main__":
    # Example usage
    print("Feature extraction module loaded successfully!")
    print("Available functions:")
    print("- preprocess_pcg()")
    print("- extract_comprehensive_features()")
    print("- extract_center_segment()")
