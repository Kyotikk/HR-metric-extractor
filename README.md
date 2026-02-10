# Data Inspection Pipeline

Pipeline for extracting activity intervals, computing HR/HRV metrics, and generating baseline comparisons from ADL logs plus ECG/PPG/HR signals.

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

1. Copy the template config:

```bash
copy config.example.yaml config.yaml
```

2. Edit `config.yaml` with your data paths and settings.
3. Run the pipeline:

```bash
python run_inspection.py --config config.yaml
```

## Configuration

Key fields in `config.yaml`:
- `data.adl_path` - ADL CSV with start/end events
- `data.ecg_path` - ECG/PPG/HR signal CSV (or folder of CSV.GZ)
- `signal.signal_type` - `ecg`, `ppg`, or `hr`
- `activities.*` - keywords and minimum durations

You can also generate a template via:

```bash
python run_inspection.py --config config.yaml --create-config
```

## Batch processing (optional)

Batch processing expects a dataset folder with subject subdirectories. Set the base path via environment variable:

```bash
set DATA_BASE_PATH=D:\path\to\dataset
python batch_process_subjects.py
```

See [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md) for details.

## Notebooks

- [analysis_notebook.ipynb](analysis_notebook.ipynb) - single-subject exploration
- [multi_subject_analysis.ipynb](multi_subject_analysis.ipynb) - batch analysis

## Outputs

Results are written to `output/` (single run) or `output_batch/` (batch runs). Each run includes:
- `propulsion_activities.csv`, `resting_activities.csv`
- `propulsion_hr_metrics.csv`, `resting_hr_metrics.csv`
- `baseline_activity_comparisons.csv`
- `pipeline_summary.csv`

Optional diagnostics and window overlap reports are created when enabled and data is available.
