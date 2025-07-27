"""
Heart Sound QAQC Feature Extraction Module

This module contains functions for extracting comprehensive features
from phonocardiogram (PCG) signals for quality assessment.
"""
import os
import pywt
import glob
import pywt
import tqdm
import h5py
import shutil
import librosa
import numpy as np 
import pandas as pd 
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.base import RegressorMixin
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.signal import (filtfilt, sosfilt, medfilt, 
                        butter, hilbert, resample, iirnotch, 
                        periodogram, cheby1, ellip, bessel, sosfiltfilt)



def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='band', output='sos')
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfilt(sos, data)
	return y

def scale_data(arr, x=1):
	"""
	This function is used to scale the sound before storing the data
	"""
	try:
		return x * (2 * (arr - np.min(arr)) / np.ptp(arr) - 1)
	except:
		print("scaling the data failed")

def upsample_array_linear(y, upsample_factor, gain=10000):
	x = np.arange(0, len(y))
	f = interpolate.splrep(x, y)
	
	x_upsampled = np.linspace(min(x), max(x), len(x) * upsample_factor)
	
	up_array = interpolate.splev(x_upsampled, f)
	up_array[np.isnan(up_array)] = 0
	up_array[np.isinf(up_array)] = np.max(up_array)
	up_array = scale_data(up_array, gain)
	
	return up_array

def flexible_filter(signal,
                    fs,
                    mode='bandpass',
                    freq=(20, 800),
                    order=4,
                    filter_type='butter',
                    method='sosfiltfilt',
                    rp=1, rs=40,
                    plot_response=False):
    """
    Flexible signal filter with support for various filter types and methods.

    Parameters:
        signal         : 1D numpy array (input signal)
        fs             : Sampling frequency in Hz
        mode           : 'lowpass', 'highpass', 'bandpass', 'bandstop'
        freq           : Cutoff frequency (single value or tuple in Hz)
        order          : Filter order (default = 4)
        filter_type    : 'butter', 'cheby1', 'ellip', 'bessel'
        method         : 'filtfilt', 'sosfilt', or 'sosfiltfilt'
        rp             : Passband ripple (used in Chebyshev, Elliptic)
        rs             : Stopband attenuation (used in Chebyshev, Elliptic)
        plot_response  : Plot the filter frequency response (True/False)

    Returns:
        filtered_signal : Filtered signal (1D array)
    """
    nyq = 0.5 * fs
    
    if isinstance(freq, (list, tuple, np.ndarray)):
        Wn = [f / nyq for f in freq]
    else:
        Wn = freq / nyq

    # Select filter design
    filter_designs = {
        'butter': butter,
        'cheby1': cheby1,
        'ellip' : ellip,
        'bessel': bessel
    }

    if filter_type not in filter_designs:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    design_func = filter_designs[filter_type]

    if filter_type in ['cheby1', 'ellip']:
        sos = design_func(order, rp, Wn, btype=mode, output='sos')
    else:
        sos = design_func(order, Wn, btype=mode, output='sos')

    # Apply filter
    if method == 'filtfilt':
        from scipy.signal import sos2tf
        b, a = sos2tf(sos)
        filtered_signal = filtfilt(b, a, signal)
    elif method == 'sosfilt':
        filtered_signal = sosfilt(sos, signal)
    elif method == 'sosfiltfilt':
        filtered_signal = sosfiltfilt(sos, signal)
    else:
        raise ValueError(f"Unsupported filtering method: {method}")

    # Optional: Plot frequency response
    if plot_response:
        from scipy.signal import sosfreqz
        w, h = sosfreqz(sos, worN=2000, fs=fs)
        plt.figure(figsize=(8, 3))
        plt.semilogx(w, 20 * np.log10(np.abs(h)), label='Filter Response')
        plt.title(f'{filter_type.capitalize()} {mode} Filter Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True, which='both', ls='--')
        plt.axvline(np.min(freq) if isinstance(freq, (list, tuple)) else freq, color='red', linestyle='--')
        if mode in ['bandpass', 'bandstop']:
            plt.axvline(np.max(freq), color='red', linestyle='--')
        plt.show()

    return filtered_signal

def schmidt_spike_removal(signal: np.ndarray, threshold: Optional[float] = 3.0, window_size: Optional[int] = 11) -> np.ndarray:
	"""
	Perform spike removal using Schmidt method.
 
	Parameters:
	-----------
	signal: np.ndarray
		The input signal.
	threshold: float, optional
		The threshold for spike removal. Default is 3.0.
	window_size: int, optional
		The window size for spike removal. Default is 11.
  
	Returns:
	--------
	np.ndarray
		The signal with outlier spikes removed.
	"""
	window_size = min(window_size, len(signal))
	
	if window_size % 2 == 0:
		window_size -= 1
		
	smoothed_signal = medfilt(signal, kernel_size=window_size)
	
	residuals = np.abs(signal - smoothed_signal)
	
	std_residuals = np.std(residuals)
	
	spike_indices = residuals > (threshold * std_residuals)
	
	signal[spike_indices] = smoothed_signal[spike_indices]
	
	return signal

def auto_notch_filter(signal, fs, freq=50.0, Q=30.0, power_threshold_db=-20):
    """
    Automatically applies a notch filter at the given powerline frequency 
    if a spectral peak exceeds a defined threshold.

    Parameters:
        signal            : 1D numpy array (PCG signal)
        fs                : Sampling frequency
        freq              : Powerline frequency (e.g., 50 or 60 Hz)
        Q                 : Quality factor of the notch filter (default=30)
        power_threshold_db: Threshold in dB to detect significant peak

    Returns:
        cleaned_signal    : Filtered signal (notch applied if needed)
        notch_applied     : Boolean flag indicating whether notch was used
    """
    # Compute power spectral density
    f, Pxx = periodogram(signal, fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # Find power near the target frequency (within ±1 Hz)
    target_band = (f >= freq - 1) & (f <= freq + 1)
    max_power_in_band = np.max(Pxx_db[target_band])

    # Check if it exceeds the threshold
    if max_power_in_band > power_threshold_db:
        # Apply notch filter
        b, a = iirnotch(w0=freq / (0.5 * fs), Q=Q)
        filtered = filtfilt(b, a, signal)
        return filtered, True
    else:
        # Return original signal
        return signal, False

def ica_denoise_singlechannel(signal, delay=20, n_components=5, keep_components=1):
    """
    ICA-based denoising for single-channel PCG using delayed embedding.

    Parameters:
        signal          : 1D numpy array (raw PCG signal)
        delay           : Number of time-lagged versions to stack (embedding dimension)
        n_components    : Number of ICA components to extract
        keep_components : Number of components to retain for reconstruction
    
    Returns:
        cleaned_signal  : 1D denoised PCG signal
    """
    N = len(signal)
    embed_len = N - delay
    if embed_len <= 0:
        raise ValueError("Signal too short for selected delay.")

    # Create delayed embedding (shape: [delay, embed_len])
    embedded = np.stack([signal[i:i+embed_len] for i in range(delay)], axis=0)

    # ICA decomposition
    ica = FastICA(n_components=n_components, random_state=42)
    S_ = ica.fit_transform(embedded.T)  # [embed_len, n_components]
    A_ = ica.mixing_

    # Select top components (by kurtosis, abs to avoid sign flips)
    k_vals = np.abs(kurtosis(S_, axis=0))
    top_indices = np.argsort(k_vals)[-keep_components:]

    # Reconstruct selected components
    S_selected = S_[:, top_indices]
    A_selected = A_[:, top_indices]
    reconstructed = np.dot(S_selected, A_selected.T).T  # shape: [delay, embed_len]

    # Collapse back to 1D (mean across rows)
    denoised = np.mean(reconstructed, axis=0)

    # Pad back to original length
    padded = np.pad(denoised, (delay // 2, N - len(denoised) - delay // 2), mode='edge')

    return padded

def pca_denoise_signal(signal, delay=20, n_components=3):
    """
    PCA-based denoising for single-channel PCG using delayed embedding.

    Parameters:
        signal        : 1D numpy array (raw PCG signal)
        delay         : Delay window size for embedding
        n_components  : Number of principal components to retain

    Returns:
        denoised      : Reconstructed 1D denoised signal
    """
    N = len(signal)
    embed_len = N - delay
    if embed_len <= 0:
        raise ValueError("Signal too short for selected delay.")

    # Step 1: Create delayed embedding (shape: [delay, embed_len])
    embedded = np.stack([signal[i:i+embed_len] for i in range(delay)], axis=0)

    # Step 2: PCA decomposition
    pca = PCA(n_components=delay)
    transformed = pca.fit_transform(embedded.T)  # shape: [embed_len, delay]

    # Step 3: Keep only top components
    transformed[:, n_components:] = 0

    # Step 4: Reconstruct & collapse back to 1D
    reconstructed = pca.inverse_transform(transformed).T  # shape: [delay, embed_len]
    denoised = np.mean(reconstructed, axis=0)

    # Step 5: Pad to original length
    padded = np.pad(denoised, (delay // 2, N - len(denoised) - delay // 2), mode='edge')

    return padded

def pca_denoise_auto(signal, delay=30, target_variance=0.95):
    """
    PCA-based denoising with automatic selection of components 
    based on explained variance threshold.

    Parameters:
        signal           : 1D numpy array (raw PCG signal)
        delay            : Delay window size for embedding
        target_variance  : Proportion of variance to retain (e.g., 0.95)

    Returns:
        denoised         : Reconstructed denoised signal
        n_used           : Number of components used
    """
    N = len(signal)
    embed_len = N - delay
    if embed_len <= 0:
        raise ValueError("Signal too short for selected delay.")

    # Step 1: Create delayed embedding (shape: [delay, embed_len])
    embedded = np.stack([signal[i:i+embed_len] for i in range(delay)], axis=0)

    # Step 2: Full PCA
    pca = PCA(n_components=delay)
    transformed = pca.fit_transform(embedded.T)

    # Step 3: Find number of components needed to reach target variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, target_variance) + 1

    # Step 4: Zero out the rest
    transformed[:, n_components:] = 0

    # Step 5: Reconstruct signal from selected components
    reconstructed = pca.inverse_transform(transformed).T  # shape: [delay, embed_len]
    denoised = np.mean(reconstructed, axis=0)

    # Step 6: Pad to original length
    padded = np.pad(denoised, (delay // 2, N - len(denoised) - delay // 2), mode='edge')

    return padded, n_components

def wavelet_denoise_signal(signal, wavelet='db4', level=4, threshold_scale=0.8, store_step=False):
    """
    Apply wavelet shrinkage denoising to a 1D signal.

    Parameters:
        signal          : 1D numpy array (input signal)
        wavelet         : Wavelet name (e.g., 'db4', 'sym5')
        level           : Decomposition level
        threshold_scale : Multiplier for universal threshold
        store_step      : If True, return coefficients and threshold used

    Returns:
        denoised_signal : Reconstructed 1D signal after denoising
        meta            : (optional) dict with 'coeffs', 'threshold', 'sigma'
    """
    N = len(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(N))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)[:N]

    if store_step:
        return denoised_signal, {'coeffs': coeffs, 'threshold': uthresh, 'sigma': sigma}
    
    return denoised_signal

def fft_boost_band(signal, fs, band=(20, 800), gain=1.5, store_step=False):
    """
    Boost energy in a frequency band using FFT-domain manipulation.

    Parameters:
        signal      : 1D numpy array (input signal)
        fs          : Sampling frequency (Hz)
        band        : Tuple (low, high) band in Hz to boost
        gain        : Multiplicative gain for that band
        store_step  : If True, return spectrum and boost mask

    Returns:
        boosted     : Signal after frequency-domain band boosting
        meta        : (optional) dict with 'spectrum', 'boost_mask', 'freqs'
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    spectrum = np.fft.rfft(signal)

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    spectrum_boosted = np.copy(spectrum)
    spectrum_boosted[band_mask] *= gain

    boosted = np.fft.irfft(spectrum_boosted, n=N)

    if store_step:
        return boosted, {'spectrum': spectrum, 'boost_mask': band_mask, 'freqs': freqs}

    return boosted

def detrend_spectrogram_with_demographics(spectrograms: np.ndarray,
                                          demographics: Union[Dict[str, Union[list, float, int]], np.ndarray],
                                          model_cls=LinearRegression,
                                          standardize=True,
                                          return_predicted=False):
    """
    Remove demographic influence from single or batched spectrograms using regression.

    Parameters:
        spectrograms     : Array of shape [F, T] or [B, F, T]
        demographics     : Dict of scalar or list values, or np.ndarray [B, D]
        model_cls        : sklearn regressor class
        standardize      : Whether to z-score demographic features
        return_predicted : If True, return predicted spectrograms

    Returns:
        residuals        : Detrended spectrogram(s), shape [F, T] or [B, F, T]
        predicted        : (optional) Predicted demographic components
    """
    # Handle single spectrogram case
    single_input = False
    if spectrograms.ndim == 2:
        spectrograms = spectrograms[None, ...]  # [1, F, T]
        single_input = True

    B, F, T = spectrograms.shape
    flat_dim = F * T

    # Prepare demographic matrix
    if isinstance(demographics, dict):
        feature_arrays = []
        for v in demographics.values():
            if isinstance(v, (int, float)):
                feature_arrays.append(np.array([v] * B))  # scalar → broadcast
            else:
                feature_arrays.append(np.asarray(v))

        # Validate shapes
        for i, arr in enumerate(feature_arrays):
            if arr.ndim != 1 or len(arr) != B:
                raise ValueError(f"Demographic feature {list(demographics.keys())[i]} must be scalar or length {B}")
        X = np.stack(feature_arrays, axis=1)  # [B, D]
    else:
        X = np.asarray(demographics).reshape(B, -1)

    if standardize:
        X = StandardScaler().fit_transform(X)

    residuals = np.zeros_like(spectrograms)
    predicted_batch = np.zeros_like(spectrograms) if return_predicted else None

    for i in range(B):
        Y = spectrograms[i].reshape(-1, 1)  # shape [F*T, 1]
        x = X[i].reshape(1, -1)
        x_repeated = np.tile(x, (flat_dim, 1))  # [F*T, D]

        model = model_cls()
        model.fit(x_repeated, Y.ravel())
        Y_hat = model.predict(x_repeated).reshape(F, T)

        residuals[i] = spectrograms[i] - Y_hat
        if return_predicted:
            predicted_batch[i] = Y_hat

    if single_input:
        residuals = residuals[0]
        predicted_batch = predicted_batch[0] if return_predicted else None

    return (residuals, predicted_batch) if return_predicted else residuals


def detrend_signal_with_demographics(signals: np.ndarray,
                                     demographics: Union[Dict[str, Union[list, float, int]], np.ndarray],
                                     model_cls=LinearRegression,
                                     standardize=True,
                                     return_predicted=False):
    """
    Remove demographic influence from 1D or batched signals using regression.

    Parameters:
        signals          : 1D array [T] or 2D array [B, T]
        demographics     : Dict of scalar or list values, or np.ndarray
        model_cls        : sklearn regressor class
        standardize      : Whether to z-score demographic features
        return_predicted : If True, also return predicted component

    Returns:
        residuals        : Detrended signal(s), shape [T] or [B, T]
        predicted_batch  : (optional) Predicted component(s)
    """
    # Handle single signal case
    single_input = False
    if signals.ndim == 1:
        signals = signals[None, :]  # [1, T]
        single_input = True

    B, T = signals.shape

    # Handle demographic dicts
    if isinstance(demographics, dict):
        feature_arrays = []
        for v in demographics.values():
            if isinstance(v, (int, float)):
                feature_arrays.append(np.array([v] * B))  # scalar → broadcast
            else:
                feature_arrays.append(np.asarray(v))

        # Validate all arrays are shape [B]
        for i, arr in enumerate(feature_arrays):
            if arr.ndim != 1 or len(arr) != B:
                raise ValueError(f"Demographic feature {list(demographics.keys())[i]} must be scalar or length {B}")

        X = np.stack(feature_arrays, axis=1)  # [B, D]
    else:
        X = np.asarray(demographics).reshape(B, -1)

    if standardize:
        X = StandardScaler().fit_transform(X)

    residuals = np.zeros_like(signals)
    predicted_batch = np.zeros_like(signals) if return_predicted else None

    for i in range(B):
        y = signals[i].reshape(-1, 1)
        x = X[i].reshape(1, -1)
        x_repeated = np.tile(x, (T, 1))

        model = model_cls()
        model.fit(x_repeated, y.ravel())
        y_hat = model.predict(x_repeated).reshape(-1)

        residuals[i] = y.ravel() - y_hat
        if return_predicted:
            predicted_batch[i] = y_hat

    if single_input:
        residuals = residuals[0]
        predicted_batch = predicted_batch[0] if return_predicted else None

    return (residuals, predicted_batch) if return_predicted else residuals

def preprocess_pcg(signal,
                   original_fs,
                   resample_fs=1000,
                   band=(20, 730),
                   boost_gain=0.5,
                   wavelet='db4',
                   level=4,
                   ica_denoise=False,
                   pca_denoise=False,
                   denoise_delay=10,
                   n_components=3,
                   spike_threshold=3.0,
                   spike_window=11,
                   normalize="maxabs",
                   smooth_kernel=5,
                   log_steps=False):
    """
    Full PCG preprocessing pipeline with FFT boosting, wavelet denoising,
    optional ICA/PCA denoising, and flexible step tracking.

    Returns:
        signal     : Final cleaned 1D signal
        steps_dict : (optional) Dictionary of intermediate outputs
    """
    steps = {} if log_steps else None

    # Step 1: Resample
    if original_fs != resample_fs:
        target_length = int(len(signal) * resample_fs / original_fs)
        signal = resample(signal, target_length)
    if log_steps: steps['resampled'] = signal.copy()

    # Step 2: Bandpass Filter
    signal = flexible_filter(signal, fs=resample_fs, mode='bandpass', freq=band, method='sosfiltfilt')
    if log_steps: steps['bandpassed'] = signal.copy()

    # Step 3: Notch Filters (50 Hz & 60 Hz)
    # for freq in [60.0, 50.0]:
    #     signal, notch_applied = auto_notch_filter(signal, fs=resample_fs, freq=freq)
    #     if notch_applied:
    #         print(f"⚡ Notch filter applied at {freq} Hz.")
    # if log_steps: steps['notched'] = signal.copy()

    # Step 4: Spike Removal
    signal = schmidt_spike_removal(signal, threshold=spike_threshold, window_size=spike_window)
    if log_steps: steps['despiked'] = signal.copy()

    # Step 5: FFT Domain Boosting
    signal = fft_boost_band(signal, fs=resample_fs, band=band, gain=boost_gain, store_step=False)
    if log_steps: steps['fft_boosted'] = signal.copy()

    # Step 6: Wavelet Denoising
    signal = wavelet_denoise_signal(signal, wavelet=wavelet, level=level, threshold_scale=0.8, store_step=False)
    if log_steps: steps['wavelet_denoised'] = signal.copy()

    # Step 7: ICA or PCA Denoising
    if ica_denoise and pca_denoise:
        raise ValueError("❌ Choose only one of ICA or PCA denoising.")
    elif ica_denoise:
        signal = ica_denoise_singlechannel(signal, delay=denoise_delay, n_components=n_components, keep_components=1)
    elif pca_denoise:
        signal = pca_denoise(signal, delay=denoise_delay, n_components=n_components)
    if log_steps: steps['component_denoised'] = signal.copy()

    # Step 8: Normalization
    if normalize == "zscore":
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    elif normalize == "minmax":
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
    elif normalize == "maxabs":
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs
    if log_steps: steps['normalized'] = signal.copy()

    # Step 9: Median Smoothing
    if smooth_kernel and smooth_kernel > 1:
        signal = medfilt(signal, kernel_size=smooth_kernel)
    if log_steps: steps['smoothed'] = signal.copy()

    return (signal, steps) if log_steps else signal

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
