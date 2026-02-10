#!/usr/bin/env python3
"""
Data Loading Module

Utilities for loading and preparing physiological data from various formats.
Supports PPG, ECG, HR, and ADL (Activities of Daily Living) data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def _load_single_timeseries_file(data_path: Path,
                                 time_col: str = 't_sec',
                                 signal_col: str = 'value',
                                 compression: Optional[str] = 'infer',
                                 verbose: bool = True) -> pd.DataFrame:
    """Load a single timeseries CSV file with flexible column detection."""
    data_path = Path(data_path)

    # Load data
    try:
        df = pd.read_csv(data_path, compression='gzip')
    except Exception as e:
        raise IOError(f"Failed to read {data_path}: {str(e)}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find time column
    time_options = ['t_sec', 'time', 'timestamp', 't']
    time_col_actual = None
    for opt in time_options:
        if opt in df.columns:
            time_col_actual = opt
            break

    if time_col_actual is None:
        raise ValueError(f"No time column found. Columns: {df.columns.tolist()}")

    # Find signal column
    signal_options = ['value', 'signal', 'ppg', 'ecg', 'hr', 'eda', 'imu']
    signal_col_actual = None
    for opt in signal_options:
        if opt in df.columns:
            signal_col_actual = opt
            break

    if signal_col_actual is None:
        # Use first numeric column after time
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != time_col_actual]
        if numeric_cols:
            signal_col_actual = numeric_cols[0]
        else:
            raise ValueError(f"No numeric signal column found. Columns: {df.columns.tolist()}")

    # Extract relevant columns
    result = df[[time_col_actual, signal_col_actual]].copy()
    result.columns = ['t_sec', 'value']

    # Convert to numeric
    result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
    result['value'] = pd.to_numeric(result['value'], errors='coerce')

    # Remove NaN rows
    result = result.dropna()

    # Sort by time
    result = result.sort_values('t_sec').reset_index(drop=True)

    if len(result) == 0:
        raise ValueError(f"No valid data after cleaning: {data_path}")

    if verbose:
        print(f"Loaded {len(result)} samples from {data_path}")
        print(f"  Time range: {result['t_sec'].min():.1f} - {result['t_sec'].max():.1f} seconds")
        print(f"  Value range: {result['value'].min():.2f} - {result['value'].max():.2f}")

    return result


def load_timeseries_data(data_path: Path,
                         time_col: str = 't_sec',
                         signal_col: str = 'value',
                         compression: Optional[str] = 'infer') -> pd.DataFrame:
    """
    Load timeseries data from CSV with flexible column detection.
    
    Args:
        data_path: Path to CSV file
        time_col: Name of time column (will try variations if not found)
        signal_col: Name of signal column (will try variations if not found)
        compression: Compression type ('infer', 'gzip', None, etc.)
        
    Returns:
        DataFrame with [time_col, signal_col] columns, sorted by time
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.is_dir():
        # Load and concatenate all CSV.GZ files in directory (recursively)
        files = sorted(data_path.glob('**/*.csv.gz'))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV.GZ files found in directory: {data_path}")

        frames = []
        for f in files:
            try:
                df_part = _load_single_timeseries_file(f, time_col=time_col, signal_col=signal_col, compression=compression, verbose=False)
                frames.append(df_part)
            except Exception:
                continue

        if len(frames) == 0:
            raise ValueError(f"No valid data after cleaning: {data_path}")

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values('t_sec').reset_index(drop=True)

        print(f"Loaded {len(result)} samples from {data_path} ({len(frames)} files)")
        print(f"  Time range: {result['t_sec'].min():.1f} - {result['t_sec'].max():.1f} seconds")
        print(f"  Value range: {result['value'].min():.2f} - {result['value'].max():.2f}")
        return result

    # Single file path
    return _load_single_timeseries_file(data_path, time_col=time_col, signal_col=signal_col, compression=compression, verbose=True)


def load_hr_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    Load pre-computed HR metrics (RMSSD, SDNN, HR, etc.) from CSV.
    
    Expected columns: t_sec, rmssd, sdnn, mean_hr, stress_index, etc.
    
    Args:
        metrics_path: Path to CSV with HR metrics
        
    Returns:
        DataFrame with HR metrics timeseries
    """
    metrics_path = Path(metrics_path)
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure t_sec column exists
    if 't_sec' not in df.columns:
        if 'time' in df.columns:
            df['t_sec'] = pd.to_numeric(df['time'], errors='coerce')
        else:
            raise ValueError("No time column found in metrics")
    
    # Convert numeric columns
    for col in df.columns:
        if col != 't_sec':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by time
    df = df.sort_values('t_sec').reset_index(drop=True)
    
    print(f"Loaded HR metrics: {len(df)} samples")
    print(f"  Columns: {[c for c in df.columns if c != 't_sec']}")
    
    return df


def extract_window_data(data_df: pd.DataFrame,
                        t_start: float,
                        t_end: float,
                        time_col: str = 't_sec',
                        signal_col: str = 'value',
                        margin_sec: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signal data for a specific time window.
    
    Args:
        data_df: DataFrame with time and signal columns
        t_start: Start time (seconds)
        t_end: End time (seconds)
        time_col: Name of time column
        signal_col: Name of signal column
        margin_sec: Margin to add on both sides of window
        
    Returns:
        Tuple of (signal_values, time_values)
    """
    t_start_adj = t_start - margin_sec
    t_end_adj = t_end + margin_sec
    
    mask = (data_df[time_col] >= t_start_adj) & (data_df[time_col] <= t_end_adj)
    window_data = data_df[mask].copy()
    
    if len(window_data) == 0:
        return np.array([]), np.array([])
    
    return window_data[signal_col].values, window_data[time_col].values


def estimate_sampling_frequency(time_array: np.ndarray) -> float:
    """
    Estimate sampling frequency from time array.
    
    Args:
        time_array: Array of time samples
        
    Returns:
        Estimated sampling frequency (Hz)
    """
    if len(time_array) < 2:
        return 1.0
    
    time_diffs = np.diff(time_array)
    # Remove outliers (might be gaps)
    time_diffs = time_diffs[time_diffs > 0]
    
    if len(time_diffs) == 0:
        return 1.0
    
    # Use median to be robust to outliers
    median_dt = np.median(time_diffs)
    fs = 1.0 / median_dt if median_dt > 0 else 1.0
    
    return fs


def create_data_summary(data_df: pd.DataFrame,
                       time_col: str = 't_sec',
                       signal_col: str = 'value') -> Dict:
    """
    Create summary statistics for a dataset.
    
    Args:
        data_df: Data DataFrame
        time_col: Name of time column
        signal_col: Name of signal column
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        'n_samples': len(data_df),
        'duration_sec': data_df[time_col].max() - data_df[time_col].min(),
        'time_start': data_df[time_col].min(),
        'time_end': data_df[time_col].max(),
        'signal_mean': data_df[signal_col].mean(),
        'signal_std': data_df[signal_col].std(),
        'signal_min': data_df[signal_col].min(),
        'signal_max': data_df[signal_col].max(),
        'signal_range': data_df[signal_col].max() - data_df[signal_col].min(),
        'nan_count': data_df[signal_col].isna().sum(),
    }
    
    # Estimate sampling rate
    summary['estimated_fs_hz'] = estimate_sampling_frequency(data_df[time_col].values)
    
    return summary


def load_ppg_data(subject_path: Path, channel: str = 'green') -> Optional[pd.DataFrame]:
    """
    Load PPG data for a subject.
    
    Args:
        subject_path: Path to subject directory
        channel: PPG channel - 'green', 'infrared' (or 'infra_red'), or 'red'
        
    Returns:
        DataFrame with columns [t_sec, ppg] or None if not found
    """
    subject_path = Path(subject_path)
    
    # Map channel names to directory patterns
    channel_map = {
        'green': 'corsano_wrist_ppg2_green_6',
        'infrared': 'corsano_wrist_ppg2_infra_red_22',
        'red': 'corsano_wrist_ppg2_red_182',
    }
    
    if channel.lower() not in channel_map:
        return None
    
    ppg_dir = subject_path / channel_map[channel.lower()]
    
    if not ppg_dir.exists():
        return None
    
    # Find CSV file in the directory
    csv_files = list(ppg_dir.glob('*.csv.gz'))
    if not csv_files:
        return None
    
    try:
        ppg_path = csv_files[0]
        df = pd.read_csv(ppg_path, compression='gzip')
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Find time and signal columns
        time_col = None
        signal_col = None
        
        for col in df.columns:
            if col in ['t_sec', 'time', 'timestamp', 't']:
                time_col = col
            elif col in ['value', 'ppg', 'signal', 'data']:
                signal_col = col
        
        # If we couldn't find standard column names, use the first two columns
        if time_col is None or signal_col is None:
            if len(df.columns) >= 2:
                time_col = df.columns[0]
                signal_col = df.columns[1]
            else:
                return None
        
        # Extract and rename columns
        result = df[[time_col, signal_col]].copy()
        result.columns = ['t_sec', 'ppg']
        
        # Convert to numeric and remove NaNs
        result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
        result['ppg'] = pd.to_numeric(result['ppg'], errors='coerce')
        result = result.dropna()
        
        # Sort by time
        result = result.sort_values('t_sec').reset_index(drop=True)
        
        return result if len(result) > 0 else None
        
    except Exception as e:
        return None


def load_best_available_signal(subject_path: Path, sensor_priority: list = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load the best available physiological signal (ECG, PPG) for a subject.
    Tries sensors in order of priority.
    
    Args:
        subject_path: Path to subject directory
        sensor_priority: List of sensors to try in order. Default: ['ecg', 'ppg_green', 'ppg_infrared', 'ppg_red']
        
    Returns:
        Tuple of (DataFrame, sensor_name) or (None, None) if no data found
    """
    if sensor_priority is None:
        sensor_priority = ['ecg', 'ppg_green', 'ppg_infrared', 'ppg_red']
    
    subject_path = Path(subject_path)
    
    for sensor in sensor_priority:
        if sensor == 'ecg':
            ecg_dir = subject_path / 'vivalnk_vv330_ecg'
            if ecg_dir.exists():
                # Try direct path first
                direct_ecg = ecg_dir / 'data_1.csv.gz'
                if direct_ecg.exists():
                    try:
                        df = pd.read_csv(direct_ecg, compression='gzip')
                        df.columns = [c.strip().lower() for c in df.columns]
                        if 'time' in df.columns and 'ecg' in df.columns:
                            result = df[['time', 'ecg']].copy()
                            result.columns = ['t_sec', 'signal']
                            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
                            result['signal'] = pd.to_numeric(result['signal'], errors='coerce')
                            result = result.dropna().sort_values('t_sec').reset_index(drop=True)
                            if len(result) > 0:
                                return result, 'ecg'
                    except:
                        pass
                # Try date subfolders
                try:
                    for item in ecg_dir.glob('*/*.csv.gz'):
                        df = pd.read_csv(item, compression='gzip')
                        df.columns = [c.strip().lower() for c in df.columns]
                        if 'time' in df.columns and 'ecg' in df.columns:
                            result = df[['time', 'ecg']].copy()
                            result.columns = ['t_sec', 'signal']
                            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
                            result['signal'] = pd.to_numeric(result['signal'], errors='coerce')
                            result = result.dropna().sort_values('t_sec').reset_index(drop=True)
                            if len(result) > 0:
                                return result, 'ecg'
                        break
                except:
                    pass
        
        elif sensor.startswith('ppg_'):
            channel = sensor.split('_')[1]
            ppg_df = load_ppg_data(subject_path, channel)
            if ppg_df is not None:
                ppg_df.columns = ['t_sec', 'signal']
                return ppg_df, f'ppg_{channel}'
    
    return None, None

