#!/usr/bin/env python3
"""Join D3CODE items + raters + ratings into one CSV for the DiADEM pipeline.

Drops ratings with rating_binary NaN (raw=-1: rater did not understand the message).
Uses binary offensiveness label (0/1) as in the D3CODE data card.
"""
import argparse
from pathlib import Path

import pandas as pd


def build(items_path, raters_path, ratings_path, out_path):
    items = pd.read_csv(items_path).drop_duplicates(subset=["item_id"], keep="first")
    raters = pd.read_csv(raters_path).drop_duplicates(subset=["rater_id"], keep="first")
    ratings = pd.read_csv(ratings_path).drop_duplicates(subset=["item_id", "rater_id"], keep="first")

    print(f"items={items.shape} raters={raters.shape} ratings={ratings.shape}")
    df = ratings.merge(items, on="item_id", how="left")
    df = df.merge(raters, on="rater_id", how="left")

    n_before = len(df)
    df = df[df["rating_binary"].notna()].copy()
    print(f"dropped {n_before - len(df)} rows with rating_binary NaN (raw=-1)")
    if df["item_id"].isna().any() or df["rater_id"].isna().any():
        print(f"warning: missing keys after merge item_na={df['item_id'].isna().sum()} rater_na={df['rater_id'].isna().sum()}")
        df = df.dropna(subset=["item_id", "rater_id"])

    df["rating_binary"] = df["rating_binary"].astype(int)
    # Stable column names for pipeline configs
    df = df.rename(columns={
        "rater_id": "annotator_id",
        "text": "comment_text",
        "rating_binary": "offensive",
    })
    # SES as string category for one-hot
    if "SES" in df.columns:
        df["SES"] = df["SES"].astype("Int64").astype(str).replace("<NA>", "")

    out_cols = [
        "item_id", "annotator_id", "offensive", "comment_text",
        "category", "sub-category", "rating_raw",
        "Age", "Gender", "Region", "Country", "SES",
        "care", "equality", "proportionality", "authority", "loyalty", "purity",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols].reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"  rows={len(df)} items={df['item_id'].nunique()} annotators={df['annotator_id'].nunique()}")
    print(f"  label counts:\n{df['offensive'].value_counts().to_string()}")
    return out_path


def main():
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "datasets" / "d3code"
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(default_dir / "d3-items.csv"))
    ap.add_argument("--raters", default=str(default_dir / "d3-raters.csv"))
    ap.add_argument("--ratings", default=str(default_dir / "d3-ratings.csv"))
    ap.add_argument("--out", default=str(default_dir / "d3code_complete.csv"))
    args = ap.parse_args()
    build(args.items, args.raters, args.ratings, args.out)


if __name__ == "__main__":
    main()
