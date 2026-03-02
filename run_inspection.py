#!/usr/bin/env python3
"""
Data Inspection Pipeline - Main Script

Comprehensive pipeline for:
1. Activity extraction (propulsion, resting, etc.)
2. HR metrics computation during activities
3. Baseline-activity comparisons
4. Window overlap and delay analysis
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from activity_extraction import (
    parse_adl_file, identify_activity_intervals,
    extract_propulsion_activities, extract_resting_activities,
    add_baseline_reference, extract_custom_activities
)
from hr_metrics import (
    extract_rr_intervals_from_ecg, compute_hr_metrics_for_window, 
    compute_differential_metrics, extract_hr_metrics_from_timeseries,
    check_signal_quality
)
from feature_extraction import (
    extract_activity_hrv_features, extract_activity_eda_features,
    extract_activity_ppg_features, extract_activity_imu_features,
    merge_feature_dicts
)
from window_overlap_analysis import (
    segment_activity_into_phases, extract_phases_from_data,
    compute_optimal_windows_for_metrics, create_window_overlap_report
)
from data_loading import (
    load_timeseries_data, load_hr_metrics, extract_window_data,
    estimate_sampling_frequency, create_data_summary,
    load_ppg_data, load_imu_sensors, load_eda_bioz_data
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _compute_ecg_segments(t_sec: np.ndarray, gap_factor: float = 10.0):
    """Identify continuous ECG segments based on large time gaps."""
    if t_sec is None or len(t_sec) < 2:
        return []
    t = np.asarray(t_sec)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return [(float(t[0]), float(t[-1]))]

    median_dt = np.median(dt)
    gap_threshold = median_dt * gap_factor
    breaks = np.where(np.diff(t) > gap_threshold)[0]

    segments = []
    start_idx = 0
    for b in breaks:
        end_idx = b
        segments.append((float(t[start_idx]), float(t[end_idx])))
        start_idx = b + 1
    segments.append((float(t[start_idx]), float(t[-1])))
    return segments


def _total_overlap(activities: pd.DataFrame, segments, offset: float) -> float:
    if activities is None or len(activities) == 0 or not segments:
        return 0.0
    total = 0.0
    for _, row in activities.iterrows():
        t_start = row['t_start'] + offset
        t_end = row['t_end'] + offset
        for seg_start, seg_end in segments:
            overlap = max(0.0, min(t_end, seg_end) - max(t_start, seg_start))
            total += overlap
    return total


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> dict:
    """Create default configuration template."""
    return {
        'project': {
            'name': 'data-inspection',
            'output_dir': './output',
        },
        'data': {
            'adl_path': 'D:/ETHZ/Lifelogging/interim/scai-ncgg/sim_elderly_2/scai_app/ADLs_1.csv.gz',  # Path to ADL CSV
            'ecg_path': 'D:/ETHZ/Lifelogging/interim/scai-ncgg/sim_elderly_2/vivalnk_vv330_ecg/data_1.csv.gz',  # Path to PPG CSV
            'imu_paths': None,  # Optional: dict mapping sensor_name -> path, or null for auto-discovery
            'eda_bioz_path': None,  # Optional: corsano_bioz_bioz path or file (auto-discovery if null)
            'hr_metrics_path': None,  # Path to pre-computed HR metrics (optional)
        },
        'activities': {
            'propulsion_keywords': ['level walking', 'walker', 'propulsion'],
            'resting_keywords': ['sitting', 'rest', 'lying'],
            'min_duration_sec': 30.0,
            'baseline_min_duration_sec': 35.0,
            'extra': {
                # Example custom short activity
                'washing_hands': {
                    'keywords': ['wash hands', 'washing hands', 'hand wash'],
                    'min_duration_sec': 15.0,
                }
            },
        },
        'signal': {
            'signal_type': 'ecg',  # One of: ppg, ecg, hr
            'sampling_frequency_hz': 128.0,
        },
        'analysis': {
            'compute_baseline_comparison': True,
            'compute_window_overlap': True,
            'analyze_delays': True,
            'max_delay_sec': 300.0,
            'recovery_window_sec': 300.0,
            'baseline_window_sec': 120.0,
        },
        'visualization': {
            'enable_overlays': False,
            'activities': ['propulsion', 'resting', 'washing_hands'],
            'margin_sec': 30.0,
            'max_windows_per_activity': 5,
            'relative_time': True,
            'output_dir': 'overlays'
        }
    }


def _plot_overlay_window(window: dict,
                         ecg_data: pd.DataFrame,
                         output_dir: Path,
                         subject_label: str,
                         margin_sec: float,
                         fs: float | None,
                         relative_time: bool) -> bool:
    try:
        from visualize_hr_overlays import plot_window_hr
    except Exception as exc:
        logger.warning(f"  Overlay plot import failed: {exc}")
        return False

    if ecg_data is None or len(ecg_data) == 0:
        return False

    signal, time = extract_window_data(
        ecg_data,
        window['t_start'],
        window['t_end'],
        margin_sec=margin_sec
    )

    if len(signal) == 0:
        return False

    data = pd.DataFrame({'t_sec': time, 'value': signal})
    try:
        plot_window_hr(
            window,
            data,
            output_dir,
            subject_label,
            margin_sec,
            fs,
            relative_time=relative_time
        )
    except Exception as exc:
        logger.warning(f"  Overlay plot failed for {window.get('activity', 'unknown')}: {exc}")
        return False

    return True


def run_inspection_pipeline(config_path: str) -> None:
    """
    Main pipeline execution.
    
    Args:
        config_path: Path to YAML configuration file
    """
    logger.info("=" * 80)
    logger.info("Data Inspection Pipeline")
    logger.info("=" * 80)
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Create output directory
    output_dir = Path(cfg['project']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    viz_cfg = cfg.get('visualization', {}) or {}
    viz_enabled = bool(viz_cfg.get('enable_overlays', False))
    viz_activities = viz_cfg.get('activities', ['propulsion', 'resting'])
    if isinstance(viz_activities, str):
        viz_activities = [a.strip() for a in viz_activities.split(',') if a.strip()]
    viz_margin_sec = float(viz_cfg.get('margin_sec', 30.0))
    viz_relative_time = bool(viz_cfg.get('relative_time', True))
    viz_max_windows = viz_cfg.get('max_windows_per_activity', None)
    if viz_max_windows is not None:
        try:
            viz_max_windows = int(viz_max_windows)
        except Exception:
            viz_max_windows = None
    if viz_max_windows is not None and viz_max_windows <= 0:
        viz_max_windows = None

    viz_output_dir = Path(viz_cfg.get('output_dir', 'overlays'))
    if not viz_output_dir.is_absolute():
        viz_output_dir = output_dir / viz_output_dir

    subject_label = cfg.get('project', {}).get('subject_id') or output_dir.name
    viz_counts = {str(a): 0 for a in viz_activities}
    
    # ========================================================================
    # STEP 1: Load and parse ADL data
    # ========================================================================
    logger.info("\n[STEP 1] Loading ADL data...")
    adl_path = Path(cfg['data']['adl_path'])
    adl_df = parse_adl_file(adl_path)
    logger.info(f"  Loaded {len(adl_df)} ADL events")
    
    # Identify activity intervals
    adl_intervals = identify_activity_intervals(adl_df)
    logger.info(f"  Identified {len(adl_intervals)} activity intervals")
    
    # ========================================================================
    # STEP 2: Extract activity types
    # ========================================================================
    logger.info("\n[STEP 2] Extracting activity types...")
    
    propulsion = extract_propulsion_activities(
        adl_intervals,
        min_duration_sec=cfg['activities'].get('min_duration_sec', 30.0),
        keywords=cfg['activities'].get('propulsion_keywords', ['level walking','walking','walker','self propulsion','propulsion','assisted propulsion'])
    )
    logger.info(f"  Propulsion activities: {len(propulsion)}")
    
    resting = extract_resting_activities(
        adl_intervals,
        min_duration_sec=cfg['activities'].get('baseline_min_duration_sec', 40.0),
        keywords=cfg['activities'].get('resting_keywords', ['sitting','rest','lying','seated'])
    )
    logger.info(f"  Resting activities: {len(resting)}")
    
    # Save activity extracts
    propulsion.to_csv(output_dir / 'propulsion_activities.csv', index=False)
    resting.to_csv(output_dir / 'resting_activities.csv', index=False)
    logger.info(f"  Saved activity extracts to output_dir/")

    # Optional: extract and save custom activities
    extra_cfg = cfg['activities'].get('extra', {})
    custom_activities = extract_custom_activities(adl_intervals, extra_cfg)
    for name, df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        df.to_csv(output_dir / f'activity_{safe_name}.csv', index=False)
        logger.info(f"  Saved custom activity '{name}' with {len(df)} intervals")
    
    # ========================================================================
    # STEP 3: Load PPG/HR data
    # ========================================================================
    logger.info("\n[STEP 3] Loading physiological data...")
    
    # Get subject path for fallback PPG loading
    subject_path = Path(cfg['data']['ecg_path']).parent.parent
    
    # Try to load ECG, assess quality, and fallback to PPG if needed
    ecg_data = None
    signal_source = None
    sensor_quality = {}
    
    # Load ECG and check quality
    try:
        ecg_path = Path(cfg['data']['ecg_path'])
        ecg_data = load_timeseries_data(ecg_path)
        
        if ecg_data is not None and len(ecg_data) > 0:
            # Check ECG quality
            ecg_quality = check_signal_quality(ecg_data['value'].values)
            sensor_quality['ecg'] = ecg_quality
            logger.info(f"  ECG loaded: {len(ecg_data)} samples, quality_score={ecg_quality['quality_score']:.3f}")
            
            # If ECG quality is poor (flat signal), try PPG fallback
            if ecg_quality['is_flat']:
                logger.warning(f"  ECG signal is flat (std={ecg_quality['std']:.2e}) - attempting PPG fallback")
                ecg_data = None
            else:
                signal_source = 'ecg'
    except Exception as e:
        logger.warning(f"  Failed to load ECG: {str(e)} - attempting PPG fallback")
        ecg_data = None
    
    # If ECG failed or is poor quality, try PPG sensors
    if ecg_data is None or signal_source is None:
        ppg_channels = ['green', 'infrared', 'red']
        best_ppg = None
        best_ppg_channel = None
        best_ppg_quality = -1
        
        logger.info(f"  Attempting to load PPG data as fallback...")
        for channel in ppg_channels:
            ppg_data = load_ppg_data(subject_path, channel)
            if ppg_data is not None and len(ppg_data) > 0:
                ppg_quality = check_signal_quality(ppg_data['ppg'].values)
                sensor_quality[f'ppg_{channel}'] = ppg_quality
                logger.info(f"    PPG ({channel}): {len(ppg_data)} samples, quality_score={ppg_quality['quality_score']:.3f}")
                
                # Keep track of best PPG channel
                if ppg_quality['quality_score'] > best_ppg_quality:
                    best_ppg = ppg_data
                    best_ppg_channel = channel
                    best_ppg_quality = ppg_quality['quality_score']
        
        if best_ppg is not None:
            logger.info(f"  Using PPG ({best_ppg_channel}) as signal source (quality_score={best_ppg_quality:.3f})")
            # Convert PPG data to match ECG format
            ecg_data = best_ppg.copy()
            ecg_data.columns = ['t_sec', 'value']
            signal_source = f'ppg_{best_ppg_channel}'
        else:
            logger.error(f"  No usable PPG data found - proceeding with empty ECG data")
            ecg_data = pd.DataFrame(columns=['t_sec', 'value'])
            signal_source = 'none'
    
    logger.info(f"  Signal source: {signal_source}")
    logger.info(f"  Sensor quality scores: {sensor_quality}")

    # Load IMU data (multiple sensors, each kept separate)
    imu_sensors = {}  # Dict mapping sensor_name -> DataFrame
    try:
        imu_cfg_paths = cfg.get('data', {}).get('imu_paths', None)
        if imu_cfg_paths is None:
            imu_cfg_paths = {}
        elif isinstance(imu_cfg_paths, str):
            # Support legacy single-path config by wrapping in dict
            imu_cfg_paths = {'imu': imu_cfg_paths}
        
        imu_sensors = load_imu_sensors(subject_path=subject_path, imu_config=imu_cfg_paths)
        
        if len(imu_sensors) > 0:
            logger.info(f"  IMU sensors loaded: {len(imu_sensors)} sensor(s)")
            for sensor_name, sensor_df in imu_sensors.items():
                duration_sec = sensor_df['t_sec'].max() - sensor_df['t_sec'].min()
                n_samples = len(sensor_df)
                logger.info(f"    - {sensor_name}: {n_samples} samples, duration={duration_sec:.1f}s")
        else:
            logger.info("  IMU data not found (skipping IMU for this run)")
    except Exception as e:
        logger.warning(f"  Failed to load IMU data: {str(e)}")
        imu_sensors = {}

    # Load EDA/BioZ data (corsano_bioz_bioz sensor)
    eda_bioz_data = None
    eda_bioz_summary = None
    try:
        eda_bioz_cfg_path = cfg.get('data', {}).get('eda_bioz_path')
        eda_bioz_data = load_eda_bioz_data(subject_path=subject_path, eda_bioz_path=eda_bioz_cfg_path)
        
        if eda_bioz_data is not None and len(eda_bioz_data) > 0:
            eda_bioz_summary = create_data_summary(eda_bioz_data, time_col='t_sec', signal_col='eda_bioz')
            logger.info(
                f"  EDA/BioZ loaded: {len(eda_bioz_data)} samples, "
                f"duration={eda_bioz_summary['duration_sec']:.1f}s, "
                f"estimated_fs={eda_bioz_summary['estimated_fs_hz']:.2f}Hz"
            )
        else:
            logger.info("  EDA/BioZ data not found (skipping for this run)")
            eda_bioz_data = None
            eda_bioz_summary = None
    except Exception as e:
        logger.warning(f"  Failed to load EDA/BioZ data: {str(e)}")
        eda_bioz_data = None
        eda_bioz_summary = None
    
    # Determine sampling frequency
    cfg_fs = cfg.get('signal', {}).get('sampling_frequency_hz', None)
    if cfg_fs is not None and cfg_fs > 0:
        fs = float(cfg_fs)
        logger.info(f"  Using configured sampling frequency: {fs:.2f} Hz")
    elif len(ecg_data) > 0:
        fs = estimate_sampling_frequency(ecg_data['t_sec'].values)
        logger.info(f"  Estimated sampling frequency: {fs:.2f} Hz")
    else:
        fs = 128.0  # Default fallback
        logger.info(f"  Estimated sampling frequency: {fs:.2f} Hz")
    
    # Apply time offset to align ADL times with ECG times
    # If offset is None or 'auto', estimate it from data; otherwise use configured value
    time_offset_sec = cfg['activities'].get('time_offset_sec', None)

    # Preserve raw activity times for offset optimization
    propulsion_raw = propulsion.copy()
    resting_raw = resting.copy()
    
    if time_offset_sec is None or time_offset_sec == 'auto':
        # Auto-estimate offset: align ADL activities to ECG segments based on ADL start time and ECG recording times
        if len(adl_intervals) > 0 and len(ecg_data) > 0:
            adl_min = adl_intervals['t_start'].min()
            ecg_min = ecg_data['t_sec'].min()
            ecg_max = ecg_data['t_sec'].max()

            # Compute ECG segments to avoid aligning activities into gaps
            segments = _compute_ecg_segments(ecg_data['t_sec'].values)

            # Candidate offsets: align ADL start to each segment start
            candidate_offsets = [seg_start - adl_min for seg_start, _ in segments]
            candidate_offsets.append(ecg_min - adl_min)

            best_offset = ecg_min - adl_min
            best_overlap = -1.0
            for off in candidate_offsets:
                overlap = _total_overlap(propulsion_raw, segments, off) + _total_overlap(resting_raw, segments, off)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_offset = off

            time_offset_sec = best_offset
            logger.info(f"  Auto-estimated time offset: {time_offset_sec:.3f} sec ({time_offset_sec/3600:.1f} hours)")
            logger.info(f"    (ADL start: {adl_min:.1f}, ECG range: {ecg_min:.1f} to {ecg_max:.1f})")
        else:
            time_offset_sec = 0.0
            logger.warning("  No activities to offset; using 0.0")
    
    if time_offset_sec != 0.0:
        propulsion['t_start'] = propulsion['t_start'] + time_offset_sec
        propulsion['t_end'] = propulsion['t_end'] + time_offset_sec
        resting['t_start'] = resting['t_start'] + time_offset_sec
        resting['t_end'] = resting['t_end'] + time_offset_sec
        for name, df in custom_activities.items():
            if df is not None and len(df) > 0:
                df['t_start'] = df['t_start'] + time_offset_sec
                df['t_end'] = df['t_end'] + time_offset_sec
        logger.info(f"  Applied time offset: {time_offset_sec:.3f} sec")
    
    # ========================================================================
    # STEP 4: Compute HR metrics from signal if available
    # ========================================================================
    logger.info("\n[STEP 4] Computing HR metrics from signal...")
    
    # Check if pre-computed metrics available
    hr_metrics_path = cfg['data'].get('hr_metrics_path')
    if hr_metrics_path and Path(hr_metrics_path).exists():
        logger.info("  Using pre-computed HR metrics")
        hr_metrics = load_hr_metrics(Path(hr_metrics_path))
    else:
        logger.info("  Computing HR metrics from ECG signal (this may take a while)...")
        # For now, just compute metrics for activities
        hr_metrics = None
    
    # ========================================================================
    # STEP 5: Extract HR metrics for propulsion activities
    # ========================================================================
    logger.info("\n[STEP 5] Extracting HR metrics for propulsion activities...")
    
    propulsion_metrics = []
    skipped_prop = {'insufficient_data': 0, 'outside_range': 0}

    # Diagnostic: log data time range and sample activity intervals
    logger.info(f"  ECG time range: {ecg_data['t_sec'].min():.1f} - {ecg_data['t_sec'].max():.1f}")
    if len(propulsion) > 0:
        logger.info("  Sample propulsion intervals:")
        for i, row in propulsion.head(5).iterrows():
            logger.info(f"    idx={i} t_start={row['t_start']} t_end={row['t_end']} duration={row['duration_sec']}")
    # Report overlap statistics
    prop_min = propulsion['t_start'].min() if len(propulsion)>0 else np.nan
    prop_max = propulsion['t_end'].max() if len(propulsion)>0 else np.nan
    resting_min = resting['t_start'].min() if len(resting)>0 else np.nan
    resting_max = resting['t_end'].max() if len(resting)>0 else np.nan
    logger.info(f"  Propulsion time range: {prop_min} - {prop_max}")
    logger.info(f"  Resting time range: {resting_min} - {resting_max}")

    # Count how many activities fall within ECG time range
    ecg_min = ecg_data['t_sec'].min()
    ecg_max = ecg_data['t_sec'].max()
    prop_in_range = propulsion[(propulsion['t_start'] >= ecg_min) & (propulsion['t_end'] <= ecg_max)]
    rest_in_range = resting[(resting['t_start'] >= ecg_min) & (resting['t_end'] <= ecg_max)]
    logger.info(f"  Propulsion intervals within ECG range: {len(prop_in_range)}/{len(propulsion)}")
    logger.info(f"  Resting intervals within ECG range: {len(rest_in_range)}/{len(resting)}")
    
    for idx, activity in propulsion.iterrows():
        t_start = activity['t_start']
        t_end = activity['t_end']
        
        # Check if activity is within ECG bounds
        if not (t_start >= ecg_min and t_end <= ecg_max):
            logger.warning(f"  Activity {idx}: Outside ECG time range")
            skipped_prop['outside_range'] += 1
            continue
        
        # Extract ECG signal
        signal, time = extract_window_data(ecg_data, t_start, t_end)
        
        if len(signal) < 100:
            logger.warning(f"  Activity {idx}: Insufficient data ({len(signal)} samples) - possible gap in ECG recording")
            skipped_prop['insufficient_data'] += 1
            continue
        
        # Check signal quality
        signal_std = np.std(signal)
        if signal_std < 1e-6:
            logger.warning(f"  Activity {idx}: Signal has no variation (std={signal_std:.2e}) - cannot extract HR metrics")
            skipped_prop['insufficient_data'] += 1
            continue
        
        # Compute HR metrics
        # Determine signal type based on source
        if signal_source and signal_source.startswith('ppg'):
            signal_type = 'ppg'
        else:
            signal_type = cfg['signal'].get('signal_type', 'ecg')
        
        activity_metrics = extract_hr_metrics_from_timeseries(
            signal, time,
            signal_type=signal_type,
            fs=fs
        )
        
        # Check if metrics extraction was successful (n_beats == 0 means no peaks detected)
        if activity_metrics.get('n_beats', 0) == 0:
            logger.warning(f"  Activity {idx}: No peaks detected in {signal_type.upper()} signal (signal_std={signal_std:.4f})")
        
        # =============================
        # Feature extraction for activity
        # =============================
        # Extract HRV features from RR intervals (if available from HR metrics extraction)
        hrv_features = {}
        rr_intervals_available = activity_metrics.get('rr_intervals_ms') is not None
        if rr_intervals_available:
            rr_intervals = np.array(activity_metrics['rr_intervals_ms'])
            if isinstance(rr_intervals, list):
                rr_intervals = np.array(rr_intervals)
            if len(rr_intervals) >= 10:
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                hrv_features = extract_activity_hrv_features(rr_intervals, activity_context, fs=fs)
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Extracted {len(hrv_features)} HRV features from {len(rr_intervals)} RR intervals")
            else:
                logger.debug(f"    Activity {idx}: Insufficient RR intervals ({len(rr_intervals)} < 10) for HRV extraction")
        else:
            if idx % 10 == 0:
                logger.debug(f"    Activity {idx}: No RR intervals in activity metrics (rr_intervals_ms=None)")
        
        # Extract EDA features if EDA/BioZ data available
        eda_features = {}
        if eda_bioz_data is not None and len(eda_bioz_data) > 0:
            eda_signal, eda_time = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
            if len(eda_signal) >= 10:
                eda_fs = estimate_sampling_frequency(eda_bioz_data['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                eda_features = extract_activity_eda_features(eda_signal, eda_time, activity_context, fs=eda_fs)
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Extracted {len(eda_features)} EDA features from {len(eda_signal)} samples")
            else:
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Insufficient EDA samples ({len(eda_signal)} < 10) for EDA extraction")
        else:
            if idx % 10 == 0:
                logger.debug(f"    Activity {idx}: No EDA/BioZ data available for feature extraction")
        
        # Extract PPG features if signal is PPG (or from ECG if PPG data was loaded as fallback)
        ppg_features = {}
        if ecg_data is not None and len(ecg_data) > 0 and signal_source and 'ppg' in signal_source.lower():
            # Extract PPG window from loaded signal
            ppg_signal, ppg_time = extract_window_data(ecg_data, t_start, t_end, signal_col='value')
            if len(ppg_signal) >= 10:
                ppg_fs = estimate_sampling_frequency(ecg_data['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                ppg_features = extract_activity_ppg_features(ppg_signal, ppg_time, activity_context, fs=ppg_fs)
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Extracted {len(ppg_features)} PPG features from {len(ppg_signal)} samples")
            else:
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Insufficient PPG samples ({len(ppg_signal)} < 10) for PPG extraction")
        else:
            if idx % 10 == 0 and signal_source and 'ppg' not in signal_source.lower():
                logger.debug(f"    Activity {idx}: Signal is {signal_source}, not PPG - skipping PPG feature extraction")

        # Extract IMU top features per sensor (magnitude channel only for runtime efficiency)
        imu_features = {}
        if imu_sensors:
            for sensor_name, sensor_df in imu_sensors.items():
                if sensor_df is None or len(sensor_df) == 0 or 'imu_magnitude' not in sensor_df.columns:
                    continue
                imu_signal, imu_time = extract_window_data(sensor_df, t_start, t_end, signal_col='imu_magnitude')
                if len(imu_signal) < 64:
                    continue
                imu_fs = estimate_sampling_frequency(sensor_df['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                sensor_features = extract_activity_imu_features(
                    imu_signal, imu_time, activity_context,
                    sensor_name=sensor_name,
                    fs=imu_fs,
                )
                imu_features.update(sensor_features)
            if idx % 10 == 0:
                logger.debug(f"    Activity {idx}: Extracted {len(imu_features)} IMU features across {len(imu_sensors)} sensor(s)")
        
        # Store with activity info
        result = {
            'activity_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            't_start': t_start,
            't_end': t_end,
            'duration_sec': activity['duration_sec'],
        }
        result.update(activity_metrics)
        result.update(hrv_features)
        result.update(eda_features)
        result.update(ppg_features)
        result.update(imu_features)
        propulsion_metrics.append(result)

        if viz_enabled and 'propulsion' in viz_activities:
            if viz_max_windows is None or viz_counts.get('propulsion', 0) < viz_max_windows:
                window = {
                    'activity': 'propulsion',
                    'activity_idx': idx,
                    't_start': t_start,
                    't_end': t_end,
                    'duration_sec': activity['duration_sec']
                }
                if _plot_overlay_window(window, ecg_data, viz_output_dir, subject_label, viz_margin_sec, fs, viz_relative_time):
                    viz_counts['propulsion'] = viz_counts.get('propulsion', 0) + 1
        
        if (idx + 1) % 5 == 0:
            logger.info(f"  Processed {idx + 1} activities...")
    
    logger.info(f"  Propulsion HR metrics extraction complete:")
    logger.info(f"    Successfully processed: {len(propulsion_metrics)}")
    logger.info(f"    Skipped - outside ECG range: {skipped_prop['outside_range']}")
    logger.info(f"    Skipped - missing ECG data: {skipped_prop['insufficient_data']}")
    
    propulsion_metrics_df = pd.DataFrame(propulsion_metrics)
    # Ensure expected index column exists even if DataFrame is empty
    if 'activity_idx' not in propulsion_metrics_df.columns:
        propulsion_metrics_df['activity_idx'] = pd.Series(dtype='int')
    # Ensure expected time and metric columns exist so later joins/selections don't KeyError
    _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
    for _c in _required_cols:
        if _c not in propulsion_metrics_df.columns:
            propulsion_metrics_df[_c] = pd.Series(dtype='float')
    propulsion_metrics_df.to_csv(output_dir / 'propulsion_hr_metrics.csv', index=False)
    logger.info(f"  Computed HR metrics for {len(propulsion_metrics_df)} propulsion activities")
    
    # ========================================================================
    # STEP 6: Extract HR metrics for resting activities
    # ========================================================================
    logger.info("\n[STEP 6] Extracting HR metrics for resting activities...")
    
    resting_metrics = []
    skipped_rest = {'insufficient_data': 0, 'outside_range': 0}
    
    for idx, activity in resting.iterrows():
        t_start = activity['t_start']
        t_end = activity['t_end']
        
        # Check if activity is within ECG bounds
        if not (t_start >= ecg_min and t_end <= ecg_max):
            logger.warning(f"  Resting {idx}: Outside ECG time range")
            skipped_rest['outside_range'] += 1
            continue
        
        # Extract ECG signal
        signal, time = extract_window_data(ecg_data, t_start, t_end)
        
        if len(signal) < 100:
            logger.warning(f"  Resting {idx}: Insufficient data ({len(signal)} samples) - possible gap in ECG recording")
            skipped_rest['insufficient_data'] += 1
            continue
        
        # Check signal quality
        signal_std = np.std(signal)
        if signal_std < 1e-6:
            logger.warning(f"  Resting {idx}: Signal has no variation (std={signal_std:.2e}) - cannot extract HR metrics")
            skipped_rest['insufficient_data'] += 1
            continue
        
        # Compute HR metrics
        # Determine signal type based on source
        if signal_source and signal_source.startswith('ppg'):
            signal_type = 'ppg'
        else:
            signal_type = cfg['signal'].get('signal_type', 'ecg')
        
        activity_metrics = extract_hr_metrics_from_timeseries(
            signal, time,
            signal_type=signal_type,
            fs=fs
        )
        
        # Check if metrics extraction was successful (n_beats == 0 means no peaks detected)
        if activity_metrics.get('n_beats', 0) == 0:
            logger.warning(f"  Resting {idx}: No peaks detected in {signal_type.upper()} signal (signal_std={signal_std:.4f})")
        
        # Extract HRV features from RR intervals
        hrv_features = {}
        rr_intervals_available = activity_metrics.get('rr_intervals_ms') is not None
        if rr_intervals_available:
            rr_intervals = np.array(activity_metrics['rr_intervals_ms'])
            if isinstance(rr_intervals, list):
                rr_intervals = np.array(rr_intervals)
            if len(rr_intervals) >= 10:
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                hrv_features = extract_activity_hrv_features(rr_intervals, activity_context, fs=fs)
                if idx % 10 == 0:
                    logger.debug(f"    Activity {idx}: Extracted {len(hrv_features)} HRV features from {len(rr_intervals)} RR intervals")
            else:
                logger.debug(f"    Activity {idx}: Insufficient RR intervals ({len(rr_intervals)} < 10) for HRV extraction")
        else:
            if idx % 10 == 0:
                logger.debug(f"    Activity {idx}: No RR intervals in activity metrics (rr_intervals_ms=None)")
        
        # Extract EDA features if EDA/BioZ data available
        eda_features = {}
        if eda_bioz_data is not None and len(eda_bioz_data) > 0:
            eda_signal, eda_time = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
            if idx % 10 == 0 or idx == 0:
                logger.debug(f"    Resting {idx}: t_start={t_start:.1f}, t_end={t_end:.1f}, eda_data_range=[{eda_bioz_data['t_sec'].min():.1f}, {eda_bioz_data['t_sec'].max():.1f}], eda_samples={len(eda_signal)}")
            if len(eda_signal) >= 10:
                eda_fs = estimate_sampling_frequency(eda_bioz_data['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                eda_features = extract_activity_eda_features(eda_signal, eda_time, activity_context, fs=eda_fs)
                if idx % 10 == 0:
                    logger.debug(f"    Resting {idx}: Extracted {len(eda_features)} EDA features from {len(eda_signal)} samples")
            else:
                if idx % 10 == 0 or idx == 0:
                    logger.debug(f"    Resting {idx}: Insufficient EDA samples ({len(eda_signal)} < 10) for EDA extraction, activity duration={activity['duration_sec']:.1f}s")
        else:
            if idx % 10 == 0:
                logger.debug(f"    Resting {idx}: No EDA/BioZ data available for feature extraction")
        
        # Extract PPG features if signal is PPG (or from ECG if PPG data was loaded as fallback)
        ppg_features = {}
        if ecg_data is not None and len(ecg_data) > 0 and signal_source and 'ppg' in signal_source.lower():
            # Extract PPG window from loaded signal
            ppg_signal, ppg_time = extract_window_data(ecg_data, t_start, t_end, signal_col='value')
            if len(ppg_signal) >= 10:
                ppg_fs = estimate_sampling_frequency(ecg_data['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                ppg_features = extract_activity_ppg_features(ppg_signal, ppg_time, activity_context, fs=ppg_fs)
                if idx % 10 == 0:
                    logger.debug(f"    Resting {idx}: Extracted {len(ppg_features)} PPG features from {len(ppg_signal)} samples")
            else:
                if idx % 10 == 0:
                    logger.debug(f"    Resting {idx}: Insufficient PPG samples ({len(ppg_signal)} < 10) for PPG extraction")
        else:
            if idx % 10 == 0 and signal_source and 'ppg' not in signal_source.lower():
                logger.debug(f"    Resting {idx}: Signal is {signal_source}, not PPG - skipping PPG feature extraction")

        # Extract IMU top features per sensor (magnitude channel only for runtime efficiency)
        imu_features = {}
        if imu_sensors:
            for sensor_name, sensor_df in imu_sensors.items():
                if sensor_df is None or len(sensor_df) == 0 or 'imu_magnitude' not in sensor_df.columns:
                    continue
                imu_signal, imu_time = extract_window_data(sensor_df, t_start, t_end, signal_col='imu_magnitude')
                if len(imu_signal) < 64:
                    continue
                imu_fs = estimate_sampling_frequency(sensor_df['t_sec'].values)
                activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                sensor_features = extract_activity_imu_features(
                    imu_signal, imu_time, activity_context,
                    sensor_name=sensor_name,
                    fs=imu_fs,
                )
                imu_features.update(sensor_features)
            if idx % 10 == 0:
                logger.debug(f"    Resting {idx}: Extracted {len(imu_features)} IMU features across {len(imu_sensors)} sensor(s)")
        
        # Store with activity info
        result = {
            'resting_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            't_start': t_start,
            't_end': t_end,
            'duration_sec': activity['duration_sec'],
        }
        result.update(activity_metrics)
        result.update(hrv_features)
        result.update(eda_features)
        result.update(ppg_features)
        result.update(imu_features)
        resting_metrics.append(result)

        if viz_enabled and 'resting' in viz_activities:
            if viz_max_windows is None or viz_counts.get('resting', 0) < viz_max_windows:
                window = {
                    'activity': 'resting',
                    'activity_idx': idx,
                    't_start': t_start,
                    't_end': t_end,
                    'duration_sec': activity['duration_sec']
                }
                if _plot_overlay_window(window, ecg_data, viz_output_dir, subject_label, viz_margin_sec, fs, viz_relative_time):
                    viz_counts['resting'] = viz_counts.get('resting', 0) + 1
        
        if (idx + 1) % 5 == 0:
            logger.info(f"  Processed {idx + 1} resting activities...")
    
    logger.info(f"  Resting HR metrics extraction complete:")
    logger.info(f"    Successfully processed: {len(resting_metrics)}")
    logger.info(f"    Skipped - outside ECG range: {skipped_rest['outside_range']}")
    logger.info(f"    Skipped - missing ECG data: {skipped_rest['insufficient_data']}")
    
    resting_metrics_df = pd.DataFrame(resting_metrics)
    # Ensure expected index column exists even if DataFrame is empty
    if 'resting_idx' not in resting_metrics_df.columns:
        resting_metrics_df['resting_idx'] = pd.Series(dtype='int')
    # Ensure expected time and metric columns exist so later selections don't KeyError
    _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
    for _c in _required_cols:
        if _c not in resting_metrics_df.columns:
            resting_metrics_df[_c] = pd.Series(dtype='float')
    resting_metrics_df.to_csv(output_dir / 'resting_hr_metrics.csv', index=False)
    logger.info(f"  Computed HR metrics for {len(resting_metrics_df)} resting activities")

    # ========================================================================
    # STEP 6B: Extract HR metrics for custom activities
    # ========================================================================
    custom_metrics_dfs = {}
    if custom_activities:
        logger.info("\n[STEP 6B] Extracting HR metrics for custom activities...")

    for name, activities_df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        activity_metrics_list = []
        skipped_custom = {'insufficient_data': 0, 'outside_range': 0}

        for activity_idx, activity in activities_df.reset_index(drop=True).iterrows():
            t_start = activity['t_start']
            t_end = activity['t_end']

            if not (t_start >= ecg_min and t_end <= ecg_max):
                skipped_custom['outside_range'] += 1
                continue

            signal, time = extract_window_data(ecg_data, t_start, t_end)
            if len(signal) < 100:
                skipped_custom['insufficient_data'] += 1
                continue

            signal_std = np.std(signal)
            if signal_std < 1e-6:
                skipped_custom['insufficient_data'] += 1
                continue

            if signal_source and signal_source.startswith('ppg'):
                signal_type = 'ppg'
            else:
                signal_type = cfg['signal'].get('signal_type', 'ecg')

            metrics = extract_hr_metrics_from_timeseries(
                signal, time,
                signal_type=signal_type,
                fs=fs
            )

            # Extract HRV features from RR intervals
            hrv_features = {}
            rr_intervals_available = metrics.get('rr_intervals_ms') is not None
            if rr_intervals_available:
                rr_intervals = np.array(metrics['rr_intervals_ms'])
                if isinstance(rr_intervals, list):
                    rr_intervals = np.array(rr_intervals)
                if len(rr_intervals) >= 10:
                    activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                    hrv_features = extract_activity_hrv_features(rr_intervals, activity_context, fs=fs)
                    if activity_idx % 10 == 0:
                        logger.debug(f"    Custom {name} {activity_idx}: Extracted {len(hrv_features)} HRV features from {len(rr_intervals)} RR intervals")
                else:
                    logger.debug(f"    Custom {name} {activity_idx}: Insufficient RR intervals ({len(rr_intervals)} < 10) for HRV extraction")
            else:
                if activity_idx % 10 == 0:
                    logger.debug(f"    Custom {name} {activity_idx}: No RR intervals in activity metrics (rr_intervals_ms=None)")
            
            # Extract EDA features if EDA/BioZ data available
            eda_features = {}
            if eda_bioz_data is not None and len(eda_bioz_data) > 0:
                eda_signal, eda_time = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
                if len(eda_signal) >= 10:
                    eda_fs = estimate_sampling_frequency(eda_bioz_data['t_sec'].values)
                    activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                    eda_features = extract_activity_eda_features(eda_signal, eda_time, activity_context, fs=eda_fs)
                    if activity_idx % 10 == 0:
                        logger.debug(f"    Custom {name} {activity_idx}: Extracted {len(eda_features)} EDA features from {len(eda_signal)} samples")
                else:
                    if activity_idx % 10 == 0:
                        logger.debug(f"    Custom {name} {activity_idx}: Insufficient EDA samples ({len(eda_signal)} < 10) for EDA extraction")
            else:
                if activity_idx % 10 == 0:
                    logger.debug(f"    Custom {name} {activity_idx}: No EDA/BioZ data available for feature extraction")

            # Extract PPG features if signal is PPG (or from ECG if PPG data was loaded as fallback)
            ppg_features = {}
            if ecg_data is not None and len(ecg_data) > 0 and signal_source and 'ppg' in signal_source.lower():
                # Extract PPG window from loaded signal
                ppg_signal, ppg_time = extract_window_data(ecg_data, t_start, t_end, signal_col='value')
                if len(ppg_signal) >= 10:
                    ppg_fs = estimate_sampling_frequency(ecg_data['t_sec'].values)
                    activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                    ppg_features = extract_activity_ppg_features(ppg_signal, ppg_time, activity_context, fs=ppg_fs)
                    if activity_idx % 10 == 0:
                        logger.debug(f"    Custom {name} {activity_idx}: Extracted {len(ppg_features)} PPG features from {len(ppg_signal)} samples")
                else:
                    if activity_idx % 10 == 0:
                        logger.debug(f"    Custom {name} {activity_idx}: Insufficient PPG samples ({len(ppg_signal)} < 10) for PPG extraction")
            else:
                if activity_idx % 10 == 0 and signal_source and 'ppg' not in signal_source.lower():
                    logger.debug(f"    Custom {name} {activity_idx}: Signal is {signal_source}, not PPG - skipping PPG feature extraction")

            # Extract IMU top features per sensor (magnitude channel only for runtime efficiency)
            imu_features = {}
            if imu_sensors:
                for sensor_name, sensor_df in imu_sensors.items():
                    if sensor_df is None or len(sensor_df) == 0 or 'imu_magnitude' not in sensor_df.columns:
                        continue
                    imu_signal, imu_time = extract_window_data(sensor_df, t_start, t_end, signal_col='imu_magnitude')
                    if len(imu_signal) < 64:
                        continue
                    imu_fs = estimate_sampling_frequency(sensor_df['t_sec'].values)
                    activity_context = {'t_start': t_start, 't_end': t_end, 'duration_sec': activity['duration_sec']}
                    sensor_features = extract_activity_imu_features(
                        imu_signal, imu_time, activity_context,
                        sensor_name=sensor_name,
                        fs=imu_fs,
                    )
                    imu_features.update(sensor_features)
                if activity_idx % 10 == 0:
                    logger.debug(f"    Custom {name} {activity_idx}: Extracted {len(imu_features)} IMU features across {len(imu_sensors)} sensor(s)")

            result = {
                'activity_idx': activity_idx,
                'activity_name': activity.get('activity', str(name)),
                't_start': t_start,
                't_end': t_end,
                'duration_sec': activity['duration_sec'],
            }
            result.update(metrics)
            result.update(hrv_features)
            result.update(eda_features)
            result.update(ppg_features)
            result.update(imu_features)
            activity_metrics_list.append(result)

            if viz_enabled and safe_name in viz_activities:
                if viz_max_windows is None or viz_counts.get(safe_name, 0) < viz_max_windows:
                    window = {
                        'activity': safe_name,
                        'activity_idx': activity_idx,
                        't_start': t_start,
                        't_end': t_end,
                        'duration_sec': activity['duration_sec']
                    }
                    if _plot_overlay_window(window, ecg_data, viz_output_dir, subject_label, viz_margin_sec, fs, viz_relative_time):
                        viz_counts[safe_name] = viz_counts.get(safe_name, 0) + 1

        metrics_df = pd.DataFrame(activity_metrics_list)
        if 'activity_idx' not in metrics_df.columns:
            metrics_df['activity_idx'] = pd.Series(dtype='int')
        _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
        for _c in _required_cols:
            if _c not in metrics_df.columns:
                metrics_df[_c] = pd.Series(dtype='float')

        metrics_df.to_csv(output_dir / f'activity_{safe_name}_hr_metrics.csv', index=False)
        custom_metrics_dfs[name] = metrics_df
        logger.info(
            f"  Custom activity '{name}': {len(metrics_df)} metrics; "
            f"skipped outside range={skipped_custom['outside_range']}, "
            f"insufficient data={skipped_custom['insufficient_data']}"
        )
    
    # ========================================================================
    # STEP 6C: Feature Extraction Summary
    # ========================================================================
    logger.info("\n[STEP 6C] Feature extraction summary...")
    
    # Count HRV features in propulsion metrics
    propulsion_hrv_cols = [c for c in propulsion_metrics_df.columns if c.startswith('hrv_')]
    propulsion_eda_cols = [c for c in propulsion_metrics_df.columns if c.startswith('eda_')]
    propulsion_ppg_cols = [c for c in propulsion_metrics_df.columns if c.startswith('ppg_')]
    propulsion_imu_cols = [c for c in propulsion_metrics_df.columns if c.startswith('imu_')]
    logger.info(f"  Propulsion activities: {len(propulsion_hrv_cols)} HRV, {len(propulsion_eda_cols)} EDA, {len(propulsion_ppg_cols)} PPG, {len(propulsion_imu_cols)} IMU features")
    
    # Count features in resting metrics
    resting_hrv_cols = [c for c in resting_metrics_df.columns if c.startswith('hrv_')]
    resting_eda_cols = [c for c in resting_metrics_df.columns if c.startswith('eda_')]
    resting_ppg_cols = [c for c in resting_metrics_df.columns if c.startswith('ppg_')]
    resting_imu_cols = [c for c in resting_metrics_df.columns if c.startswith('imu_')]
    logger.info(f"  Resting activities: {len(resting_hrv_cols)} HRV, {len(resting_eda_cols)} EDA, {len(resting_ppg_cols)} PPG, {len(resting_imu_cols)} IMU features")
    
    # Count features in custom metrics
    for name, metrics_df in custom_metrics_dfs.items():
        custom_hrv_cols = [c for c in metrics_df.columns if c.startswith('hrv_')]
        custom_eda_cols = [c for c in metrics_df.columns if c.startswith('eda_')]
        custom_ppg_cols = [c for c in metrics_df.columns if c.startswith('ppg_')]
        custom_imu_cols = [c for c in metrics_df.columns if c.startswith('imu_')]
        logger.info(f"  Custom activity '{name}': {len(custom_hrv_cols)} HRV, {len(custom_eda_cols)} EDA, {len(custom_ppg_cols)} PPG, {len(custom_imu_cols)} IMU features")
    
    # ========================================================================
    # STEP 7: Baseline-Activity Comparison
    # ========================================================================
    logger.info("\n[STEP 7] Computing baseline-activity comparisons...")
    
    # Pair propulsion with preceding resting baseline
    propulsion_with_baseline = add_baseline_reference(propulsion, resting)
    
    # Compute differential metrics
    comparisons = []
    for idx, activity in propulsion_with_baseline.iterrows():
        if pd.isna(activity.get('baseline_t_start')):
            continue
        
        # Find corresponding metrics
        activity_metrics_row = propulsion_metrics_df[
            propulsion_metrics_df['activity_idx'] == idx
        ]
        
        baseline_row = resting_metrics_df[
            (resting_metrics_df['t_start'] >= activity['baseline_t_start']) &
            (resting_metrics_df['t_end'] <= activity['baseline_t_end'])
        ]
        
        if len(activity_metrics_row) == 0 or len(baseline_row) == 0:
            continue
        
        activity_metrics = activity_metrics_row.iloc[0].to_dict()
        baseline_metrics = baseline_row.iloc[0].to_dict()
        
        # Compute differentials
        diff_metrics = compute_differential_metrics(activity_metrics, baseline_metrics)
        
        comparison = {
            'activity_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            'propulsion_t_start': activity['t_start'],
            'propulsion_t_end': activity['t_end'],
            'propulsion_duration_sec': activity['duration_sec'],
            'baseline_t_start': activity['baseline_t_start'],
            'baseline_t_end': activity['baseline_t_end'],
            'baseline_duration_sec': activity['baseline_t_end'] - activity['baseline_t_start'],
            'time_gap_sec': activity['baseline_time_before_sec'],
        }
        comparison.update(diff_metrics)
        comparisons.append(comparison)
    
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(output_dir / 'baseline_activity_comparisons.csv', index=False)
    logger.info(f"  Created {len(comparisons_df)} baseline-activity comparisons")

    # Save HR differentials based on baseline comparison pairing
    if len(comparisons_df) > 0 and 'delta_mean_hr' in comparisons_df.columns:
        propulsion_vs_resting_df = comparisons_df[[
            'activity_idx',
            'activity_name',
            'delta_mean_hr',
            'propulsion_t_start',
            'propulsion_t_end',
            'baseline_t_start',
            'baseline_t_end'
        ]].copy()
        propulsion_vs_resting_df.rename(columns={'delta_mean_hr': 'hr_differential'}, inplace=True)
        propulsion_vs_resting_df.to_csv(output_dir / 'propulsion_vs_resting_differential.csv', index=False)
        logger.info(f"  Saved {len(propulsion_vs_resting_df)} propulsion vs resting differentials")
    else:
        pd.DataFrame().to_csv(output_dir / 'propulsion_vs_resting_differential.csv', index=False)
        logger.info("  No propulsion vs resting differentials available")

    # Baseline comparisons for custom activities
    for name, activities_df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        custom_metrics_df = custom_metrics_dfs.get(name, pd.DataFrame())

        if len(activities_df) == 0 or len(custom_metrics_df) == 0:
            pd.DataFrame().to_csv(output_dir / f'activity_{safe_name}_baseline_comparisons.csv', index=False)
            continue

        custom_with_baseline = add_baseline_reference(activities_df.reset_index(drop=True), resting)
        custom_comparisons = []
        for idx, activity in custom_with_baseline.iterrows():
            if pd.isna(activity.get('baseline_t_start')):
                continue

            activity_metrics_row = custom_metrics_df[
                custom_metrics_df['activity_idx'] == idx
            ]
            baseline_row = resting_metrics_df[
                (resting_metrics_df['t_start'] >= activity['baseline_t_start']) &
                (resting_metrics_df['t_end'] <= activity['baseline_t_end'])
            ]

            if len(activity_metrics_row) == 0 or len(baseline_row) == 0:
                continue

            activity_metrics = activity_metrics_row.iloc[0].to_dict()
            baseline_metrics = baseline_row.iloc[0].to_dict()
            diff_metrics = compute_differential_metrics(activity_metrics, baseline_metrics)

            comparison = {
                'activity_idx': idx,
                'activity_type': name,
                'activity_name': activity.get('activity', str(name)),
                'activity_t_start': activity['t_start'],
                'activity_t_end': activity['t_end'],
                'activity_duration_sec': activity['duration_sec'],
                'baseline_t_start': activity['baseline_t_start'],
                'baseline_t_end': activity['baseline_t_end'],
                'baseline_duration_sec': activity['baseline_t_end'] - activity['baseline_t_start'],
                'time_gap_sec': activity['baseline_time_before_sec'],
            }
            comparison.update(diff_metrics)
            custom_comparisons.append(comparison)

        custom_comparisons_df = pd.DataFrame(custom_comparisons)
        custom_comparisons_df.to_csv(output_dir / f'activity_{safe_name}_baseline_comparisons.csv', index=False)
        logger.info(f"  Custom activity '{name}': {len(custom_comparisons_df)} baseline comparisons")
        
        # Save custom activity vs resting differentials based on baseline pairing
        if len(custom_comparisons_df) > 0 and 'delta_mean_hr' in custom_comparisons_df.columns:
            custom_vs_resting_df = custom_comparisons_df[[
                'activity_idx',
                'activity_name',
                'delta_mean_hr',
                'activity_t_start',
                'activity_t_end',
                'baseline_t_start',
                'baseline_t_end'
            ]].copy()
            custom_vs_resting_df.rename(columns={'delta_mean_hr': 'hr_differential'}, inplace=True)
            custom_vs_resting_df.to_csv(output_dir / f'activity_{safe_name}_vs_resting_differential.csv', index=False)
            logger.info(f"  Saved {len(custom_vs_resting_df)} {name} vs resting differentials")
        else:
            pd.DataFrame().to_csv(output_dir / f'activity_{safe_name}_vs_resting_differential.csv', index=False)
            logger.info(f"  No {name} vs resting differentials available")
    
    # ========================================================================
    # STEP 8: Window Overlap and Delay Analysis
    # ========================================================================
    if cfg['analysis'].get('compute_window_overlap', True) and hr_metrics is not None:
        logger.info("\n[STEP 8] Analyzing window overlaps and delays...")
        
        overlap_reports = []
        
        for idx, activity in propulsion.iterrows():
            # Segment into phases
            phases = segment_activity_into_phases(
                (activity['t_start'], activity['t_end']),
                baseline_before_sec=cfg['analysis'].get('baseline_window_sec', 120.0),
                recovery_after_sec=cfg['analysis'].get('recovery_window_sec', 300.0)
            )
            
            # Create overlap report
            activity_dict = activity.to_dict()
            report = create_window_overlap_report(
                activity_dict, phases, hr_metrics,
                hr_metric_col='rmssd'
            )
            overlap_reports.append(report)
        
        if overlap_reports:
            full_overlap_report = pd.concat(overlap_reports, ignore_index=True)
            full_overlap_report.to_csv(output_dir / 'window_overlap_report.csv', index=False)
            logger.info(f"  Created window overlap report with {len(full_overlap_report)} rows")
    
    # ========================================================================
    # STEP 9: Summary Statistics and Reporting
    # ========================================================================
    logger.info("\n[STEP 9] Generating summary report...")
    
    summary = {
        'total_adl_events': len(adl_df),
        'total_activity_intervals': len(adl_intervals),
        'propulsion_count': len(propulsion),
        'resting_count': len(resting),
        'propulsion_with_metrics': len(propulsion_metrics_df),
        'resting_with_metrics': len(resting_metrics_df),
        'baseline_comparisons': len(comparisons_df),
        'ecg_data_samples': len(ecg_data),
        'ecg_data_duration_sec': ecg_data['t_sec'].max() - ecg_data['t_sec'].min(),
        'ecg_estimated_fs_hz': fs,
        'imu_sensors_loaded': len(imu_sensors),
        'eda_bioz_loaded': eda_bioz_data is not None and len(eda_bioz_data) > 0,
    }

    if len(imu_sensors) > 0:
        for sensor_name, sensor_df in imu_sensors.items():
            duration_sec = sensor_df['t_sec'].max() - sensor_df['t_sec'].min()
            fs_estimate = estimate_sampling_frequency(sensor_df['t_sec'].values)
            magnitude_mean = sensor_df['imu_magnitude'].mean()
            magnitude_std = sensor_df['imu_magnitude'].std()
            
            summary[f'imu_{sensor_name}_samples'] = len(sensor_df)
            summary[f'imu_{sensor_name}_duration_sec'] = duration_sec
            summary[f'imu_{sensor_name}_fs_hz'] = fs_estimate
            summary[f'imu_{sensor_name}_magnitude_mean'] = magnitude_mean
            summary[f'imu_{sensor_name}_magnitude_std'] = magnitude_std

    if eda_bioz_summary is not None:
        summary['eda_bioz_samples'] = eda_bioz_summary.get('n_samples', 0)
        summary['eda_bioz_duration_sec'] = eda_bioz_summary.get('duration_sec', np.nan)
        summary['eda_bioz_estimated_fs_hz'] = eda_bioz_summary.get('estimated_fs_hz', np.nan)
        summary['eda_bioz_mean'] = eda_bioz_summary.get('signal_mean', np.nan)
        summary['eda_bioz_std'] = eda_bioz_summary.get('signal_std', np.nan)
    
    # Add propulsion metrics summary
    if len(propulsion_metrics_df) > 0:
        summary['propulsion_mean_hr'] = propulsion_metrics_df['mean_hr'].mean()
        summary['propulsion_mean_rmssd'] = propulsion_metrics_df['rmssd'].mean()
        summary['propulsion_mean_stress_index'] = propulsion_metrics_df['stress_index'].mean()
    
    # Add resting metrics summary
    if len(resting_metrics_df) > 0:
        summary['resting_mean_hr'] = resting_metrics_df['mean_hr'].mean()
        summary['resting_mean_rmssd'] = resting_metrics_df['rmssd'].mean()
        summary['resting_mean_stress_index'] = resting_metrics_df['stress_index'].mean()

    # Add custom activity summaries
    if custom_activities:
        custom_total_count = 0
        custom_total_with_metrics = 0
        for name, activities_df in custom_activities.items():
            safe_name = str(name).strip().lower().replace(' ', '_')
            metrics_df = custom_metrics_dfs.get(name, pd.DataFrame())

            summary[f'custom_{safe_name}_count'] = len(activities_df)
            summary[f'custom_{safe_name}_with_metrics'] = len(metrics_df)

            if len(metrics_df) > 0:
                summary[f'custom_{safe_name}_mean_hr'] = metrics_df['mean_hr'].mean()
                summary[f'custom_{safe_name}_mean_rmssd'] = metrics_df['rmssd'].mean()
                summary[f'custom_{safe_name}_mean_stress_index'] = metrics_df['stress_index'].mean()

            custom_total_count += len(activities_df)
            custom_total_with_metrics += len(metrics_df)

        summary['custom_total_count'] = custom_total_count
        summary['custom_total_with_metrics'] = custom_total_with_metrics
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'pipeline_summary.csv', index=False)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Data Inspection Pipeline for Activity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run with existing config
            python run_inspection.py --config config.yaml
            
            # Create default config template
            python run_inspection.py --create-config config.yaml
            """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default config template'
    )
    
    args = parser.parse_args()
    
    config_path = args.config
    
    if args.create_config:
        # Create default config
        config = create_default_config()
        import os
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created default config template: {config_path}")
        print("Please edit the config file with your data paths and settings.")
        return
    
    # Run pipeline
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    run_inspection_pipeline(config_path)


if __name__ == '__main__':
    main()
