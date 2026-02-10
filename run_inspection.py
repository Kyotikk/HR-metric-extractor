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
from window_overlap_analysis import (
    segment_activity_into_phases, extract_phases_from_data,
    compute_optimal_windows_for_metrics, create_window_overlap_report
)
from data_loading import (
    load_timeseries_data, load_hr_metrics, extract_window_data,
    estimate_sampling_frequency, create_data_summary,
    load_ppg_data
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
            'adl_path': '/path/to/ADLs.csv.gz',  # Path to ADL CSV
            'ecg_path': '/path/to/ecg.csv.gz',  # Path to ECG/PPG CSV
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
        }
    }


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
        # Auto-estimate offset: align ADL activities to ECG segments
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
    # STEP 4: Compute HR metrics from signal
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
        
        # Store with activity info
        result = {
            'activity_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            't_start': t_start,
            't_end': t_end,
            'duration_sec': activity['duration_sec'],
        }
        result.update(activity_metrics)
        propulsion_metrics.append(result)
        
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
        
        # Store with activity info
        result = {
            'resting_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            't_start': t_start,
            't_end': t_end,
            'duration_sec': activity['duration_sec'],
        }
        result.update(activity_metrics)
        resting_metrics.append(result)
        
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

            result = {
                'activity_idx': activity_idx,
                'activity_name': activity.get('activity', str(name)),
                't_start': t_start,
                't_end': t_end,
                'duration_sec': activity['duration_sec'],
            }
            result.update(metrics)
            activity_metrics_list.append(result)

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
    }
    
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
