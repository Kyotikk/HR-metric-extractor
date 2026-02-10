import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


def parse_adl_file(adl_path: Path) -> pd.DataFrame:
    df = pd.read_csv(adl_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'time' in df.columns:
        time_col = 'time'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        raise ValueError('No time column found')

    if 'adls' in df.columns:
        activity_col = 'adls'
    elif 'adl' in df.columns:
        activity_col = 'adl'
    elif 'activity' in df.columns:
        activity_col = 'activity'
    elif 'event' in df.columns:
        activity_col = 'event'
    else:
        raise ValueError('No activity column found')

    df['t_sec'] = pd.to_numeric(df[time_col], errors='coerce')
    df['activity'] = df[activity_col].astype(str).str.strip().str.lower()
    df = df.dropna(subset=['t_sec'])
    return df[['t_sec', 'activity']]


def identify_activity_intervals(adl_df: pd.DataFrame) -> pd.DataFrame:
    events = []
    active = {}
    for _, row in adl_df.iterrows():
        a = row['activity']
        t = row['t_sec']
        if 'start' in a:
            name = a.replace('start', '').strip()
            active[name] = t
        elif 'end' in a:
            name = a.replace('end', '').strip()
            if name in active:
                events.append({'activity': name, 't_start': active[name], 't_end': t, 'duration_sec': t - active[name]})
                del active[name]
    return pd.DataFrame(events)


def extract_propulsion_activities(adl_intervals: pd.DataFrame, min_duration_sec: float = 30.0, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['level walking','walking','walker','self propulsion','propulsion','assisted propulsion']
    mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
    out = adl_intervals[mask].copy()
    out = out[out['duration_sec'] >= min_duration_sec].reset_index(drop=True)
    return out


def extract_resting_activities(adl_intervals: pd.DataFrame, min_duration_sec: float = 60.0, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['sitting','rest','lying','seated']
    mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
    out = adl_intervals[mask].copy()
    out = out[out['duration_sec'] >= min_duration_sec].reset_index(drop=True)
    return out


def extract_custom_activities(adl_intervals: pd.DataFrame, activities_config: Dict) -> Dict[str, pd.DataFrame]:
    """Extract custom activities with per-activity keyword and duration settings.

    activities_config example:
    {
        'washing_hands': {
            'keywords': ['washing hands', 'hand wash'],
            'min_duration_sec': 15.0
        },
        'stairs': {
            'keywords': ['stairs'],
            'min_duration_sec': 20.0
        }
    }
    """
    results: Dict[str, pd.DataFrame] = {}
    if not activities_config:
        return results

    for name, cfg in activities_config.items():
        if not isinstance(cfg, dict):
            continue
        keywords = cfg.get('keywords', [])
        min_duration = float(cfg.get('min_duration_sec', 0.0))
        if not keywords:
            results[name] = pd.DataFrame(columns=adl_intervals.columns)
            continue

        mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
        out = adl_intervals[mask].copy()
        out = out[out['duration_sec'] >= min_duration].reset_index(drop=True)
        results[name] = out

    return results


def add_baseline_reference(activities: pd.DataFrame, baseline_activities: pd.DataFrame) -> pd.DataFrame:
    res = activities.copy()
    res['baseline_t_start'] = np.nan
    res['baseline_t_end'] = np.nan
    res['baseline_time_before_sec'] = np.nan
    for i, row in res.iterrows():
        t_start = row['t_start']
        preceding = baseline_activities[baseline_activities['t_end'] <= t_start]
        if len(preceding) > 0:
            nb = preceding.iloc[-1]
            res.at[i,'baseline_t_start'] = nb['t_start']
            res.at[i,'baseline_t_end'] = nb['t_end']
            res.at[i,'baseline_time_before_sec'] = t_start - nb['t_end']
    return res
