#!/usr/bin/env python3
"""
Feature Extraction Module

Computes multimodal features for machine learning using Neurokit2 and other signal processing libraries:
- HRV features from ECG/RR intervals (time-domain, frequency-domain, non-linear)
- EDA features from electrodermal activity signal (SCL, SCR, tonic/phasic decomposition)
- PPG features from photoplethysmography (later)
- IMU features from accelerometer data (via Tifex, later)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


TIFEX_TOP_FEATURES_50: List[str] = [
    'mean_of_auto_corr_lag_1_to_23',
    'higuchi_fractal_dimensions_k=10',
    'mean',
    'no._of_slope_sign_changes',
    'shape_factor',
    'moment_order_3',
    'spectrum_linear_slope',
    'higuchi_fractal_dimensions_k=20',
    'hjorth_complexity',
    'higuchi_fractal_dimensions_k=5',
    'min',
    'max_of_tkeo',
    'iqr_of_wav_coeffs_lvl_4',
    'no._of_mean_crossings_of_tkeo',
    'no._of_zero_crossings',
    'harmonic_mean_of_abs',
    'min_of_abs',
    'geometric_mean',
    'max',
    'max_of_wav_coeffs_lvl_0',
    'geometric_mean_of_abs',
    'coefficient_of_variation_of_tkeo',
    'min_of_wav_coeffs_lvl_0',
    'svd_entropy',
    'no._of_mean_crossings',
    'rms',
    'no._of_zero_crossings_of_tkeo',
    'hjorth_mobility',
    'permutation_entropy',
    'entropy',
    'no._of_slope_sign_changes_of_tkeo',
    'coefficient_of_variation',
    'spectral_rel_power_band_[0.6, 4]',
    'median',
    'max_of_abs',
    'trimmed_mean_thresh_0.1',
    'higuchi_fractal_dimensions_k=40',
    'max_of_wav_coeffs_lvl_4',
    'skewness',
    'median_of_wav_coeffs_lvl_0',
    'skewness_of_abs',
    'iqr',
    'skewness_of_tkeo',
    'median_of_abs',
    'median_abs_deviation',
    'kurtosis',
    'std_of_abs',
    'kurtosis_of_wav_coeffs_4',
    'skewness_of_wav_coeffs_4',
    'min_of_wav_coeffs_lvl_4',
]


TIFEX_TO_REQUESTED_KEY_MAP: Dict[str, str] = {
    'mean_of_auto_corr_lags': 'mean_of_auto_corr_lag_1_to_23',
    'spectral_slope_linear': 'spectrum_linear_slope',
    'tkeo_max': 'max_of_tkeo',
    'wave_coeffs_lvl_4_iqr': 'iqr_of_wav_coeffs_lvl_4',
    'tkeo_no._of_mean_crossings': 'no._of_mean_crossings_of_tkeo',
    'no._of_zero_crossings': 'no._of_zero_crossings',
    'harmonic_mean_of_abs': 'harmonic_mean_of_abs',
    'min_of_abs': 'min_of_abs',
    'geometric_mean': 'geometric_mean',
    'max': 'max',
    'wave_coeffs_lvl_0_max': 'max_of_wav_coeffs_lvl_0',
    'geometric_mean_of_abs': 'geometric_mean_of_abs',
    'tkeo_spectral_coefficient_of_variation': 'coefficient_of_variation_of_tkeo',
    'wave_coeffs_lvl_0_min': 'min_of_wav_coeffs_lvl_0',
    'svd_entropy': 'svd_entropy',
    'no._of_mean_crossings': 'no._of_mean_crossings',
    'rms': 'rms',
    'tkeo_no._of_zero_crossings': 'no._of_zero_crossings_of_tkeo',
    'hjorth_mobility': 'hjorth_mobility',
    'permutation_entropy': 'permutation_entropy',
    'entropy': 'entropy',
    'tkeo_no._of_slope_sign_changes': 'no._of_slope_sign_changes_of_tkeo',
    'spectral_coefficient_of_variation': 'coefficient_of_variation',
    'relative_band_power_[0.6, 4]': 'spectral_rel_power_band_[0.6, 4]',
    'median': 'median',
    'max_of_abs': 'max_of_abs',
    'trimmed_mean_0.1': 'trimmed_mean_thresh_0.1',
    'higuchi_fractal_dimensions_k=40': 'higuchi_fractal_dimensions_k=40',
    'wave_coeffs_lvl_4_max': 'max_of_wav_coeffs_lvl_4',
    'skewness': 'skewness',
    'wave_coeffs_lvl_0_median': 'median_of_wav_coeffs_lvl_0',
    'skewness_of_abs': 'skewness_of_abs',
    'iqr': 'iqr',
    'tkeo_skewness': 'skewness_of_tkeo',
    'median_of_abs': 'median_of_abs',
    'median_absolute_deviation': 'median_abs_deviation',
    'mean_abs_deviation': 'median_abs_deviation',
    'kurtosis': 'kurtosis',
    'std_of_abs': 'std_of_abs',
    'wave_coeffs_lvl_4_kurtosis': 'kurtosis_of_wav_coeffs_4',
    'wave_coeffs_lvl_4_skewness': 'skewness_of_wav_coeffs_4',
    'wave_coeffs_lvl_4_min': 'min_of_wav_coeffs_lvl_4',
    'higuchi_fractal_dimensions_k=10': 'higuchi_fractal_dimensions_k=10',
    'mean': 'mean',
    'no._of_slope_sign_changes': 'no._of_slope_sign_changes',
    'shape_factor': 'shape_factor',
    'moment_order_3': 'moment_order_3',
    'higuchi_fractal_dimensions_k=20': 'higuchi_fractal_dimensions_k=20',
    'hjorth_complexity': 'hjorth_complexity',
    'higuchi_fractal_dimensions_k=5': 'higuchi_fractal_dimensions_k=5',
    'min': 'min',
}


def extract_hrv_features(rr_intervals_ms: np.ndarray,
                         fs: Optional[float] = None,
                         method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute HRV (Heart Rate Variability) features from RR intervals.
    
    Uses Neurokit2's hrv() and hrv_analyze() to calculate comprehensive HRV features:
    time-domain (SDNN, RMSSD, CVNN, pNN50, pNN20, etc.),
    frequency-domain (LF, HF, VLF, LF/HF ratio, etc.),
    and non-linear (SampEn, ApEn, DFA, etc.).
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds (1D array)
        fs: Sampling frequency (Hz) - used for frequency domain analysis. If None, estimated from RR intervals.
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - Time-domain: SDNN, RMSSD, CVNN, pNN50, pNN20, etc.
        - Frequency-domain: LF, HF, VLF, LF/HF ratio, etc.
        - Non-linear: SampEn, ApEn, DFA, etc.
    """
    if rr_intervals_ms is None or len(rr_intervals_ms) < 10:
        logger.warning(f"Insufficient RR intervals ({len(rr_intervals_ms) if rr_intervals_ms is not None else 0}) for HRV extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Neurokit2 expects RR intervals in seconds
        rr_intervals_sec = rr_intervals_ms / 1000.0
        
        # Estimate sampling frequency if not provided
        if fs is None:
            # Approximate fs for frequency domain analysis
            # Use median RR interval to estimate
            median_rr_sec = np.median(rr_intervals_sec)
            fs_estimate = 1.0 / median_rr_sec if median_rr_sec > 0 else 1.0
        else:
            fs_estimate = fs
        
        # Convert RR intervals to peak indices (samples)
        # nk.hrv() requires peak indices, not RR intervals
        peak_indices = nk.intervals_to_peaks(rr_intervals_sec)
        
        # Compute HRV metrics using Neurokit2's comprehensive hrv() function
        # Pass peak indices and sampling rate for frequency domain analysis
        hrv_metrics = nk.hrv(peak_indices, sampling_rate=int(fs_estimate), show=False)
        
        # Extract features from result
        features = {}
        if isinstance(hrv_metrics, pd.DataFrame):
            if len(hrv_metrics) > 0:
                features = hrv_metrics.iloc[0].to_dict()
        elif isinstance(hrv_metrics, dict):
            features = hrv_metrics
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} HRV features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"HRV feature extraction failed: {str(e)}")
        return {}


def extract_eda_features(eda_signal: np.ndarray,
                        time_array: np.ndarray,
                        fs: float = 1.0,
                        method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute EDA (Electrodermal Activity) features from raw EDA/BioZ signal.
    
    Uses Neurokit2's eda_analyze() to decompose signal into tonic and phasic components,
    and extract features like SCL, SCR amplitude/latency, etc.
    
    Args:
        eda_signal: Raw EDA/BioZ signal (1D array)
        time_array: Time vector corresponding to signal (1D array)
        fs: Sampling frequency (Hz, default 1.0 for pre-sampled data)
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - SCL (Skin Conductance Level) - mean tonic component
        - SCR (Skin Conductance Response) features - phasic component analysis
        - Signal variability metrics
    """
    if eda_signal is None or len(eda_signal) < 10:
        logger.warning(f"Insufficient EDA samples ({len(eda_signal) if eda_signal is not None else 0}) for feature extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(eda_signal)
        if valid_mask.sum() < 10:
            logger.warning(f"Insufficient valid EDA samples after NaN removal")
            return {}
        
        clean_signal = eda_signal[valid_mask]
        
        # Process and analyze EDA signal using Neurokit2
        eda_processed, info = nk.eda_process(clean_signal, sampling_rate=int(fs))
        
        # Use Neurokit2's built-in eda_analyze() for comprehensive feature extraction
        eda_features = nk.eda_analyze(eda_processed, sampling_rate=int(fs))
        
        # Extract features from result (should be a DataFrame)
        features = {}
        if isinstance(eda_features, pd.DataFrame):
            if len(eda_features) > 0:
                features = eda_features.iloc[0].to_dict()
        elif isinstance(eda_features, dict):
            features = eda_features
        
        # Add basic signal statistics if not already present
        if 'EDA_Mean' not in features:
            features['EDA_Mean'] = float(np.mean(clean_signal))
        if 'EDA_Std' not in features:
            features['EDA_Std'] = float(np.std(clean_signal))
        if 'EDA_Min' not in features:
            features['EDA_Min'] = float(np.min(clean_signal))
        if 'EDA_Max' not in features:
            features['EDA_Max'] = float(np.max(clean_signal))
        if 'EDA_Range' not in features:
            features['EDA_Range'] = float(np.max(clean_signal) - np.min(clean_signal))
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} EDA features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"EDA feature extraction failed: {str(e)}")
        return {}


def extract_ppg_features(ppg_signal: np.ndarray,
                         time_array: np.ndarray,
                         fs: float = 1.0,
                         method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute PPG (Photoplethysmography) features from raw PPG signal.
    
    Uses Neurokit2's ppg_process() and ppg_analyze() to extract cardiac and pulse-related features:
    - Heart rate from PPG
    - Pulse features
    - Signal quality metrics
    
    Args:
        ppg_signal: Raw PPG signal (1D array)
        time_array: Time vector corresponding to signal (1D array)
        fs: Sampling frequency (Hz, default 1.0 for pre-sampled data)
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - PPG_Heart_Rate: Heart rate extracted from PPG
        - PPG pulse-related metrics
        - Signal quality indicators
    """
    if ppg_signal is None or len(ppg_signal) < 10:
        logger.warning(f"Insufficient PPG samples ({len(ppg_signal) if ppg_signal is not None else 0}) for feature extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(ppg_signal)
        if valid_mask.sum() < 10:
            logger.warning(f"Insufficient valid PPG samples after NaN removal")
            return {}
        
        clean_signal = ppg_signal[valid_mask]
        
        # Process PPG signal using Neurokit2
        ppg_processed, info = nk.ppg_process(clean_signal, sampling_rate=int(fs))
        
        # Use Neurokit2's built-in ppg_analyze() for comprehensive feature extraction
        ppg_features = nk.ppg_analyze(ppg_processed, sampling_rate=int(fs))
        
        # Extract features from result (should be a DataFrame)
        features = {}
        if isinstance(ppg_features, pd.DataFrame):
            if len(ppg_features) > 0:
                features = ppg_features.iloc[0].to_dict()
        elif isinstance(ppg_features, dict):
            features = ppg_features
        
        # Add basic signal statistics if not already present
        if 'PPG_Mean' not in features:
            features['PPG_Mean'] = float(np.mean(clean_signal))
        if 'PPG_Std' not in features:
            features['PPG_Std'] = float(np.std(clean_signal))
        if 'PPG_Min' not in features:
            features['PPG_Min'] = float(np.min(clean_signal))
        if 'PPG_Max' not in features:
            features['PPG_Max'] = float(np.max(clean_signal))
        if 'PPG_Range' not in features:
            features['PPG_Range'] = float(np.max(clean_signal) - np.min(clean_signal))
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} PPG features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"PPG feature extraction failed: {str(e)}")
        return {}


def extract_activity_hrv_features(rr_intervals_ms: np.ndarray,
                                  activity_dict: Dict,
                                  fs: Optional[float] = None) -> Dict[str, float]:
    """
    Extract HRV features for a single activity window.
    
    Wraps extract_hrv_features and adds activity context.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
        activity_dict: Activity metadata (t_start, t_end, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with HRV features, prefixed 'hrv_'
    """
    features = extract_hrv_features(rr_intervals_ms, fs=fs)
    
    # Prefix all feature names with 'hrv_' for clarity
    hrv_features = {f'hrv_{k}': v for k, v in features.items()}
    
    return hrv_features


def extract_activity_eda_features(eda_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  fs: float = 1.0) -> Dict[str, float]:
    """
    Extract EDA features for a single activity window.
    
    Wraps extract_eda_features and adds activity context.
    
    Args:
        eda_signal: Raw EDA/BioZ signal
        time_array: Time vector
        activity_dict: Activity metadata (t_start, t_end, duration_sec, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with EDA features, prefixed 'eda_'
    """
    features = extract_eda_features(eda_signal, time_array, fs=fs)
    
    # Prefix all feature names with 'eda_' for clarity
    eda_features = {f'eda_{k.lower()}': v for k, v in features.items()}
    
    return eda_features


def extract_activity_ppg_features(ppg_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  fs: float = 1.0) -> Dict[str, float]:
    """
    Extract PPG features for a single activity window.
    
    Wraps extract_ppg_features and adds activity context.
    
    Args:
        ppg_signal: Raw PPG signal
        time_array: Time vector
        activity_dict: Activity metadata (t_start, t_end, duration_sec, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with PPG features, prefixed 'ppg_'
    """
    features = extract_ppg_features(ppg_signal, time_array, fs=fs)
    
    # Prefix all feature names with 'ppg_' for clarity
    ppg_features = {f'ppg_{k.lower()}': v for k, v in features.items()}
    
    return ppg_features


def extract_imu_tifex_top_features(imu_signal: np.ndarray,
                                   fs: float,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract a constrained set of top-performing IMU features using Tifex-Py.

    This function computes only a targeted subset of features (default: top-50
    list) to keep runtime practical.
    """
    if imu_signal is None or len(imu_signal) < 64:
        logger.debug("Insufficient IMU samples for Tifex feature extraction")
        return {}

    selected_features = set(feature_names or TIFEX_TOP_FEATURES_50)

    try:
        from tifex_py.feature_extraction.extraction import (
            calculate_statistical_features,
            calculate_spectral_features,
            calculate_time_frequency_features,
            StatisticalFeatureParams,
            SpectralFeatureParams,
            TimeFrequencyFeatureParams,
        )
    except ImportError:
        logger.error("Tifex-Py not installed. Install via: pip install tifex-py")
        return {}

    try:
        valid_mask = ~np.isnan(imu_signal)
        clean_signal = np.asarray(imu_signal[valid_mask], dtype=float)
        if len(clean_signal) < 64:
            return {}

        window_size = len(clean_signal)
        fs_safe = max(1, int(round(float(fs))))

        stat_params = StatisticalFeatureParams(
            window_size=window_size,
            n_lags_auto_correlation=23,
            moment_orders=[3],
            trimmed_mean_thresholds=[0.1],
            higuchi_k_values=[5, 10, 20, 40],
            calculators=[
                'mean', 'higuchi_fractal_dimensions', 'slope_sign_change',
                'shape_factor', 'higher_order_moments', 'hjorth_mobility_and_complexity',
                'min', 'zero_crossings', 'harmonic_mean_abs', 'min_abs',
                'geometric_mean', 'max', 'geometric_mean_abs', 'coefficient_of_variation',
                'svd_entropy', 'mean_crossing', 'root_mean_square',
                'permutation_entropy', 'entropy', 'median', 'max_abs',
                'trimmed_mean', 'skewness', 'skewness_abs', 'interquartile_range',
                'median_abs', 'median_absolute_deviation', 'kurtosis', 'std_abs',
                'mean_auto_correlation',
            ],
        )

        tkeo_sf_params = StatisticalFeatureParams(
            window_size=window_size,
            calculators=['max', 'mean_crossing', 'zero_crossings', 'slope_sign_change', 'coefficient_of_variation', 'skewness'],
        )
        wavelet_sf_params = StatisticalFeatureParams(
            window_size=window_size,
            calculators=['interquartile_range', 'max', 'min', 'median', 'kurtosis', 'skewness'],
        )
        tf_params = TimeFrequencyFeatureParams(
            window_size=window_size,
            decomposition_level=5,
            tkeo_sf_params=tkeo_sf_params,
            wavelet_sf_params=wavelet_sf_params,
            calculators=['tkeo_features', 'wavelet_features'],
        )

        spec_params = SpectralFeatureParams(
            fs=fs_safe,
            f_bands=[[0.6, 4]],
            calculators=['spectral_slope_linear', 'band_power'],
        )

        stat_df = calculate_statistical_features(clean_signal, params=stat_params)
        tf_df = calculate_time_frequency_features(clean_signal, params=tf_params)
        spec_df = calculate_spectral_features(clean_signal, params=spec_params)

        raw_features: Dict[str, float] = {}
        for feature_df in (stat_df, tf_df, spec_df):
            if isinstance(feature_df, pd.DataFrame) and len(feature_df) > 0:
                raw_features.update(feature_df.iloc[0].to_dict())

        mapped_features: Dict[str, float] = {}
        for tifex_key, requested_key in TIFEX_TO_REQUESTED_KEY_MAP.items():
            if requested_key not in selected_features:
                continue
            if tifex_key not in raw_features:
                continue
            value = raw_features[tifex_key]
            if pd.isna(value):
                continue
            if isinstance(value, (np.integer, np.floating)):
                mapped_features[requested_key] = float(value)
            else:
                mapped_features[requested_key] = value

        return mapped_features

    except Exception as e:
        logger.warning(f"IMU Tifex feature extraction failed: {str(e)}")
        return {}


def extract_activity_imu_features(imu_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  sensor_name: str,
                                  fs: float = 1.0,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Extract selected IMU features for a single activity window and sensor."""
    features = extract_imu_tifex_top_features(imu_signal, fs=fs, feature_names=feature_names)
    sensor_key = str(sensor_name).strip().lower().replace(' ', '_')
    imu_features = {f'imu_{sensor_key}_{k}': v for k, v in features.items()}
    return imu_features


def merge_feature_dicts(*feature_dicts) -> Dict[str, float]:
    """Merge multiple feature dictionaries, handling conflicts by taking first non-None value."""
    result = {}
    for feat_dict in feature_dicts:
        if feat_dict is None:
            continue
        for key, val in feat_dict.items():
            if key not in result and val is not None:
                result[key] = val
    return result
