import gzip
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from hr_metrics import extract_hr_metrics_from_timeseries, HAS_NEUROKIT

# Enable logging to see fallback messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Path to a sample ECG file (adjust if needed)
ecg_path = Path(r'D:\ETHZ\Lifelogging\interim\scai-ncgg\sim_elderly_1\vivalnk_vv330_ecg\data_1.csv.gz')
print('ECG path exists:', ecg_path.exists())

try:
    with gzip.open(ecg_path, 'rt') as f:
        df = pd.read_csv(f)
    print('Loaded ECG rows:', len(df))
    # Use a 30-second window (approx) depending on sampling
    # Try to infer sampling by time deltas if available
    if 'time' in df.columns:
        # take first 5 seconds of data if timestamps are available
        sig = df['ecg'].values[:4096]
    else:
        sig = df.iloc[:, 1].values[:4096]

    print('Signal segment length:', len(sig))
    res = extract_hr_metrics_from_timeseries(sig, time=None, signal_type='ecg', fs=128.0)
    print('HAS_NEUROKIT:', HAS_NEUROKIT)
    print('HR metrics result:')
    for k, v in res.items():
        print(f'  {k}: {v}')

except Exception as e:
    print('Test failed:', e)
