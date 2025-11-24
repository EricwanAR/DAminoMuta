#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch process mutation scan CSVs: normalize scores, filter, rank, and save results.

Usage: python batch_ranking.py --input_dir ./preds_folder --output_dir ./rankings --threshold 0.8
"""

import argparse
import glob
import os
import pandas as pd


def lowercase_weight(seq: str) -> float:
    """Synthesis cost weight: 1.0 (all uppercase) to 0.0 (all lowercase)"""
    return 1.0 - sum(ch.islower() for ch in seq) / len(seq)


def process_file(infile: str, outdir: str, threshold: float):
    df = pd.read_csv(infile).rename(columns={
        'log_ratio_lower_better': 'reg_ft',
        'prob_higher_better': 'cls_ft'
    })

    base = os.path.basename(infile)

    # Normalize cls_ft to [0,1]
    cls_min, cls_max = df['cls_ft'].min(), df['cls_ft'].max()
    df['cls_ft'] = (df['cls_ft'] - cls_min) / (cls_max - cls_min) if cls_max > cls_min else 0.0
    print(f"[{base}] cls_ft normalized: min→{df['cls_ft'].min():.3f}, max→{df['cls_ft'].max():.3f}")

    # Filter by threshold
    if threshold > 0:
        before = len(df)
        df = df[df['cls_ft'] >= threshold].reset_index(drop=True)
        print(f"[{base}] Filtered {before - len(df)} / {before} rows with cls_ft < {threshold}")

    # Compute weight
    df['weight'] = df['seq'].apply(lowercase_weight)

    # Normalize reg to [0,1]
    reg_min, reg_max = df['reg_ft'].min(), df['reg_ft'].max()
    df['reg_ft'] = (df['reg_ft'] - reg_min) / (reg_max - reg_min) if reg_max > reg_min else 0.0
    print(f"[{base}] reg normalized: min→{df['reg_ft'].min():.3f}, max→{df['reg_ft'].max():.3f}")

    # Compute scores and rankings
    df['raw_score'] = df['cls_ft'] - df['reg_ft']
    df['final_score'] = (1.0 - df['reg_ft']) * df['weight']
    df['ranking_weighted'] = df['final_score'].rank(ascending=False, method='dense').astype(int)
    df['ranking_unweighted'] = df['raw_score'].rank(ascending=False, method='dense').astype(int)

    # Save results
    outfile = os.path.join(outdir, f"{os.path.splitext(base)[0]}_ranking.csv")
    df.sort_values('final_score', ascending=False).to_csv(
        outfile,
        columns=['seq', 'reg_ft', 'cls_ft', 'weight', 'raw_score', 'final_score',
                 'ranking_weighted', 'ranking_unweighted'],
        index=False
    )
    print(f"[{base}] Saved ranking to: {outfile}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute weighted scores and rankings for CSVs in a folder"
    )
    parser.add_argument("--input_dir", "-d", default="predicts",
                        help="Input directory containing .csv files")
    parser.add_argument("--output_dir", "-o", default="rankings",
                        help="Output directory for ranking CSVs")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                        help="Classification threshold for filtering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in {args.input_dir}")
        return

    for filepath in files:
        process_file(filepath, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()