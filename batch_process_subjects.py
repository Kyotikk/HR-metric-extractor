#!/usr/bin/env python3
"""
Batch Processing Pipeline for Multiple Subjects

Processes all available subjects from the SCAI-NCGG dataset,
extracting HR metrics and generating comparative analysis.
"""

import logging
import os
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths
DATA_BASE_PATH = Path(os.environ.get('DATA_BASE_PATH', './data'))
SUBJECTS = [
    'sim_elderly_1', 'sim_elderly_2', 'sim_elderly_3', 'sim_elderly_4', 'sim_elderly_5',
    'sim_healthy_1', 'sim_healthy_2', 'sim_healthy_3', 'sim_healthy_4', 'sim_healthy_5',
    'sim_severe_1', 'sim_severe_2', 'sim_severe_3', 'sim_severe_4', 'sim_severe_5',
    'sub_0103', 'sub_0301', 'sub_0301_2', 'sub_0302', 'sub_0303', 'sub_0304', 'sub_0305'
]


def check_subject_data(subject_id: str) -> dict:
    """
    Check if subject has required ECG and ADL data.
    
    Returns:
        dict with keys: 'has_ecg', 'has_adl', 'ecg_path', 'adl_path', 'subject_type'
    """
    subject_path = DATA_BASE_PATH / subject_id
    ecg_dir = subject_path / 'vivalnk_vv330_ecg'
    adl_path = None
    has_adl = False
    ecg_path = None
    has_ecg = False
    adl_time_min = None
    adl_time_max = None
    
    # Check for ADL data in scai_app (primary location)
    scai_app_path = subject_path / 'scai_app' / 'ADLs_1.csv.gz'
    if scai_app_path.exists():
        adl_path = str(scai_app_path)
        has_adl = True

    # If ADL exists, load time range to select the best matching ECG file
    if has_adl and adl_path is not None:
        try:
            with gzip.open(adl_path, 'rt', encoding='utf-8', errors='ignore') as f:
                adl_df = pd.read_csv(f)
            adl_df.columns = [c.strip().lower() for c in adl_df.columns]

            if 'time' in adl_df.columns:
                adl_times = pd.to_numeric(adl_df['time'], errors='coerce')
            elif 't_start' in adl_df.columns and 't_end' in adl_df.columns:
                adl_times = pd.to_numeric(adl_df[['t_start', 't_end']].stack(), errors='coerce')
            else:
                adl_times = None

            if adl_times is not None:
                adl_times = adl_times.dropna()
                if len(adl_times) > 0:
                    adl_time_min = float(adl_times.min())
                    adl_time_max = float(adl_times.max())
        except Exception:
            adl_time_min = None
            adl_time_max = None

    # Check for ECG data - could be in root or in a date subfolder
    if ecg_dir.exists():
        # Try direct path first (simulated subjects)
        direct_ecg = ecg_dir / 'data_1.csv.gz'
        if direct_ecg.exists():
            ecg_path = str(direct_ecg)
            has_ecg = True
        else:
            # For real subjects, point to the ECG directory to load all files
            has_any = False
            for item in sorted(ecg_dir.glob('*/*.csv.gz')):
                try:
                    with gzip.open(item, 'rt', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline()
                        second_line = f.readline()
                        if first_line and second_line:
                            has_any = True
                            break
                except Exception:
                    continue
            if has_any:
                ecg_path = str(ecg_dir)
                has_ecg = True
    
    result = {
        'subject_id': subject_id,
        'has_ecg': has_ecg,
        'has_adl': has_adl,
        'ecg_path': ecg_path,
        'adl_path': adl_path,
        'subject_type': 'simulated' if subject_id.startswith('sim_') else 'real'
    }
    return result


def create_subject_config(subject_id: str, output_dir: Path, data_check: dict) -> Path:
    """
    Create a temporary config file for a specific subject.
    
    Args:
        subject_id: Subject identifier
        output_dir: Directory for outputs
        data_check: Result from check_subject_data() with ADL path
        
    Returns:
        Path to created config file (absolute path)
    """
    config = {
        'project': {
            'name': f'multi-subject-analysis-{subject_id}',
            'output_dir': str(output_dir / subject_id)
        },
        'data': {
            'adl_path': data_check['adl_path'],
            'ecg_path': data_check['ecg_path'],
            'hr_metrics_path': None
        },
        'activities': {
            'time_offset_sec': None,  # Auto-estimate
            'propulsion_keywords': ['level walking', 'walking', 'walker', 'self propulsion', 'propulsion', 'assisted propulsion'],
            'resting_keywords': ['sitting', 'rest', 'lying'],
            'min_duration_sec': 30.0,
            'baseline_min_duration_sec': 35.0,
            # additional activities - iterate through entire SENSEI protocol?
            'extra': {
                'washing_hands': {
                    'keywords': ['wash hands', 'washing hands', 'hand wash'],
                    'min_duration_sec': 15.0
                }
            }
        },
        'signal': {
            'signal_type': 'ECG',
            'sampling_frequency_hz': 128.0
        },
        'analysis': {
            'compute_baseline_comparison': True,
            'compute_window_overlap': True,
            'analyze_delays': True,
            'max_delay_sec': 300.0,
            'recovery_window_sec': 300.0,
            'baseline_window_sec': 120.0
        }
    }
    
    # Save to batch directory (not subject-specific directory which may not exist yet)
    config_path = output_dir / f'config_{subject_id}.yaml'
    config_path = config_path.resolve()  # Convert to absolute path
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def process_subject(subject_id: str, batch_output_dir: Path) -> dict:
    """
    Process a single subject through the pipeline.
    
    Returns:
        dict with processing status and metrics
    """
    logger.info(f"Processing subject: {subject_id}")
    
    # Check data availability
    data_check = check_subject_data(subject_id)
    if not data_check['has_ecg'] or not data_check['has_adl']:
        logger.warning(f"  ✗ Missing data - ECG: {data_check['has_ecg']}, ADL: {data_check['has_adl']}")
        return {
            'subject_id': subject_id,
            'status': 'SKIPPED',
            'reason': 'Missing ECG or ADL data',
            'subject_type': data_check['subject_type']
        }
    
    # Create subject-specific config
    subject_config = create_subject_config(subject_id, batch_output_dir, data_check)
    
    # Create subject output directory
    subject_output_dir = batch_output_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the pipeline for this subject
        logger.info(f"  Running pipeline...")
        result = subprocess.run(
            [sys.executable, 'run_inspection.py', '--config', str(subject_config)],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"  ✓ Pipeline completed successfully")
            
            # Try to extract summary metrics
            metrics = extract_subject_metrics(subject_output_dir)
            
            return {
                'subject_id': subject_id,
                'status': 'SUCCESS',
                'subject_type': data_check['subject_type'],
                **metrics
            }
        else:
            logger.error(f"  ✗ Pipeline failed with return code {result.returncode}")
            logger.error(f"  Full error output:\n{result.stderr}")
            return {
                'subject_id': subject_id,
                'status': 'FAILED',
                'reason': 'Pipeline execution error',
                'subject_type': data_check['subject_type']
            }
    
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Pipeline timeout (5 minutes)")
        return {
            'subject_id': subject_id,
            'status': 'TIMEOUT',
            'subject_type': data_check['subject_type']
        }
    
    except Exception as e:
        logger.error(f"  ✗ Unexpected error: {str(e)}")
        return {
            'subject_id': subject_id,
            'status': 'ERROR',
            'reason': str(e),
            'subject_type': data_check['subject_type']
        }
    
    finally:
        # Clean up temporary config
        if subject_config.exists():
            subject_config.unlink()


def extract_subject_metrics(subject_output_dir: Path) -> dict:
    """
    Extract summary metrics from subject's pipeline output.
    
    Returns:
        dict with summary metrics
    """
    metrics = {
        'propulsion_count': 0,
        'resting_count': 0,
        'propulsion_mean_hr': None,
        'resting_mean_hr': None,
        'stress_index_delta': None
    }
    
    try:
        # Read propulsion activities (detected activities, not necessarily with HR metrics)
        prop_act_file = subject_output_dir / 'propulsion_activities.csv'
        if prop_act_file.exists():
            prop_act_df = pd.read_csv(prop_act_file)
            metrics['propulsion_count'] = len(prop_act_df)
        
        # Read propulsion HR metrics (subset with successful HR extraction)
        prop_file = subject_output_dir / 'propulsion_hr_metrics.csv'
        if prop_file.exists():
            prop_df = pd.read_csv(prop_file)
            if len(prop_df) > 0:
                metrics['propulsion_mean_hr'] = prop_df['mean_hr'].mean()
        
        # Read resting activities
        rest_act_file = subject_output_dir / 'resting_activities.csv'
        if rest_act_file.exists():
            rest_act_df = pd.read_csv(rest_act_file)
            metrics['resting_count'] = len(rest_act_df)
        
        # Read resting HR metrics
        rest_file = subject_output_dir / 'resting_hr_metrics.csv'
        if rest_file.exists():
            rest_df = pd.read_csv(rest_file)
            if len(rest_df) > 0:
                metrics['resting_mean_hr'] = rest_df['mean_hr'].mean()
        
        # Calculate delta
        if metrics['propulsion_mean_hr'] is not None and metrics['resting_mean_hr'] is not None:
            metrics['stress_index_delta'] = (
                metrics['propulsion_mean_hr'] - metrics['resting_mean_hr']
            )
    
    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")
    
    return metrics


def main(subject_ids: list = None, max_workers: int = 1):
    """
    Process multiple subjects and generate summary report.
    
    Args:
        subject_ids: List of subject IDs to process. If None, processes all.
        max_workers: Number of parallel workers (currently sequential only)
    """
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING PIPELINE - SCAI-NCGG Dataset")
    logger.info("=" * 80)
    
    # Determine which subjects to process
    if subject_ids is None:
        subject_ids = SUBJECTS
    
    logger.info(f"Subjects to process: {len(subject_ids)}")
    for sid in subject_ids:
        logger.info(f"  - {sid}")
    
    # Create batch output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_output_dir = Path('./output_batch') / f'batch_{timestamp}'
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput directory: {batch_output_dir}")
    
    # Process each subject
    results = []
    for idx, subject_id in enumerate(subject_ids, 1):
        logger.info(f"\n[{idx}/{len(subject_ids)}] {subject_id}")
        result = process_subject(subject_id, batch_output_dir)
        results.append(result)
    
    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    summary_df = pd.DataFrame(results)
    logger.info(f"\nTotal subjects: {len(summary_df)}")
    logger.info(f"Successful: {len(summary_df[summary_df['status'] == 'SUCCESS'])}")
    logger.info(f"Failed: {len(summary_df[summary_df['status'] != 'SUCCESS'])}")
    
    # Status breakdown
    logger.info("\nStatus breakdown:")
    status_counts = summary_df['status'].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Save summary
    summary_path = batch_output_dir / 'batch_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Summary saved: {summary_path}")
    
    # Generate comparative analysis for successful subjects
    successful_df = summary_df[summary_df['status'] == 'SUCCESS']
    if len(successful_df) > 0:
        logger.info("\n" + "-" * 80)
        logger.info("COMPARATIVE METRICS")
        logger.info("-" * 80)
        
        # HR comparison by subject type
        logger.info("\nMean HR by Subject Type:")
        for subject_type in ['simulated', 'real']:
            subset = successful_df[successful_df['subject_type'] == subject_type]
            if len(subset) > 0:
                logger.info(f"\n  {subject_type.upper()}:")
                if 'propulsion_mean_hr' in subset.columns:
                    prop_hrs = subset['propulsion_mean_hr'].dropna()
                    if len(prop_hrs) > 0:
                        logger.info(f"    Propulsion HR: {prop_hrs.mean():.1f} ± {prop_hrs.std():.1f} bpm")
                
                if 'resting_mean_hr' in subset.columns:
                    rest_hrs = subset['resting_mean_hr'].dropna()
                    if len(rest_hrs) > 0:
                        logger.info(f"    Resting HR: {rest_hrs.mean():.1f} ± {rest_hrs.std():.1f} bpm")
                
                if 'stress_index_delta' in subset.columns:
                    deltas = subset['stress_index_delta'].dropna()
                    if len(deltas) > 0:
                        logger.info(f"    HR Delta: {deltas.mean():.1f} ± {deltas.std():.1f} bpm")
    
    logger.info("\n" + "=" * 80)
    logger.info("Batch processing completed!")
    logger.info("=" * 80)
    
    return batch_output_dir, summary_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process multiple subjects')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subject IDs to process')
    parser.add_argument('--subject-type', choices=['simulated', 'real'], default=None,
                       help='Process only simulated or real subjects')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Filter subjects if needed
    subject_ids = args.subjects
    if args.subject_type == 'simulated':
        subject_ids = [s for s in SUBJECTS if s.startswith('sim_')]
    elif args.subject_type == 'real':
        subject_ids = [s for s in SUBJECTS if not s.startswith('sim_')]
    
    # Run batch processing
    batch_dir, summary = main(subject_ids, args.max_workers)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print(f"\nOutput directory: {batch_dir}")
