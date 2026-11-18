#!/usr/bin/env python3
# train_models.py
import os
from pathlib import Path
from datetime import datetime
import argparse
import joblib
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EXPECTED = [
    "adm0_id","adm0_name","adm1_id","adm1_name","mkt_id","mkt_name",
    "cm_id","cm_name","cur_id","cur_name","pt_id","pt_name",
    "um_id","um_name","mp_month","mp_year","mp_price","mp_commoditysource"
]

def read_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in EXPECTED:
        if col not in df.columns:
            df[col] = pd.NA
    df['mp_month'] = pd.to_numeric(df['mp_month'], errors='coerce').astype('Int64')
    df['mp_year'] = pd.to_numeric(df['mp_year'], errors='coerce').astype('Int64')
    df['mp_price'] = pd.to_numeric(df['mp_price'], errors='coerce').astype('float')

    def make_date(row):
        try:
            m = int(row['mp_month'])
            y = int(row['mp_year'])
            return datetime(y, m, 1)
        except Exception:
            return pd.NaT
    df['mp_date'] = df.apply(make_date, axis=1)
    df = df.dropna(subset=['cm_id','mkt_id','mp_date','mp_price'])
    df['cm_id'] = df['cm_id'].astype(int)
    df['mkt_id'] = df['mkt_id'].astype(int)
    return df

def train_series(ts, min_points_for_hw=24, min_points_for_lr=6):
    n = len(ts.dropna())
    if n >= min_points_for_hw:
        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
            fit = model.fit(optimized=True, use_brute=True)
            return {'type':'holt', 'model': fit, 'last_date': ts.index[-1].to_pydatetime(), 'n_points': n}
        except Exception:
            pass
    if n >= min_points_for_lr:
        arr = np.array([d.toordinal() for d in ts.index]).reshape(-1,1)
        y = ts.values.reshape(-1,1)
        lr = LinearRegression()
        lr.fit(arr, y)
        coef_flat = np.asarray(lr.coef_).ravel().tolist()
        intercept_flat = np.asarray(lr.intercept_).ravel().tolist()
        intercept_value = intercept_flat[0] if len(intercept_flat) == 1 else intercept_flat
        return {
            'type':'linear',
            'model': lr,
            'last_date': ts.index[-1].to_pydatetime(),
            'n_points': n,
            'meta': {'coef': coef_flat, 'intercept': intercept_value}
        }
    return None

def train_all_models_from_csv(csv_path, overwrite=False, limit_pairs=None, verbose=True):
    df = read_csv(csv_path)
    groups = df.groupby(['cm_id','mkt_id'])
    summary = {'trained': 0, 'skipped': 0, 'total_pairs': 0}
    for (cm_id, mkt_id), group in groups:
        summary['total_pairs'] += 1
        g = group.groupby('mp_date')['mp_price'].mean().sort_index()
        ts = g.asfreq('MS').interpolate(method='time').ffill().bfill()
        model_obj = train_series(ts)
        if model_obj is None:
            summary['skipped'] += 1
            if verbose:
                print(f"Skipping cm_{cm_id}_mkt_{mkt_id}: not enough points ({len(ts.dropna())})")
            continue

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fname = MODELS_DIR / f"cm_{cm_id}_mkt_{mkt_id}.joblib"
        meta_fname = MODELS_DIR / f"cm_{cm_id}_mkt_{mkt_id}.meta.json"

        if fname.exists() and not overwrite:
            if verbose:
                print(f"Skipping existing model {fname} (use --overwrite to replace).")
            continue

        payload = {
            'cm_id': int(cm_id),
            'mkt_id': int(mkt_id),
            'cm_name': group['cm_name'].dropna().unique().tolist(),
            'mkt_name': group['mkt_name'].dropna().unique().tolist(),
            'trained_at': datetime.utcnow().isoformat(),
            'model_type': model_obj['type'],
            'n_points': int(model_obj['n_points']),
            'last_date': model_obj['last_date'].isoformat(),
            'model': model_obj['model']
        }

        joblib.dump(payload, fname)
        meta = {k: payload[k] for k in payload if k not in ('model',)}
        with open(meta_fname, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2, default=str)

        if verbose:
            print(f"Saved model: {fname} (type={payload['model_type']}, points={payload['n_points']})")

        summary['trained'] += 1
        if limit_pairs and summary['trained'] >= limit_pairs:
            break

    return summary

def main():
    parser = argparse.ArgumentParser(description="Train per-(cm_id,mkt_id) models and save to models/")
    parser.add_argument("--csv", default="market_price_dataset.csv", help="Path to CSV file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing models")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to train (for testing)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    res = train_all_models_from_csv(args.csv, overwrite=args.overwrite, limit_pairs=args.limit, verbose=not args.quiet)
    print("Training complete:", res)

if __name__ == "__main__":
    main()
