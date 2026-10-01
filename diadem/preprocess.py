"""
Unified preprocessing for all datasets in cleanedup_diadem.
Writes design matrices to outputs/{dataset}/{split_mode}/processed/
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from diadem.paths import processed_dir
from diadem.registry import load_dataset, split_csv_for
from diadem.utils.metadata_encoding import (
    encode_metadata_onehot,
    extract_categorical_features,
    save_metadata_encoding_config,
)


def _label_map_native(raw: Dict) -> Dict:
    """Convert JSON label_map keys to int where possible."""
    out = {}
    for k, v in raw.items():
        try:
            out[int(k) if str(k).lstrip("-").isdigit() else k] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def build_metadata_dict(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ann_col = cfg["annotator_col"]
    meta_col_map = cfg.get("meta_col_map", {})
    continuous_map = cfg.get("continuous_meta_cols", {}) or {}
    metadata_dict = {}
    for _, row in df.drop_duplicates(subset=[ann_col]).iterrows():
        rid = str(int(row[ann_col])) if pd.api.types.is_numeric_dtype(df[ann_col]) else str(row[ann_col])
        meta = {}
        for dcol, mkey in meta_col_map.items():
            if dcol in row and pd.notna(row[dcol]) and str(row[dcol]).strip():
                val = str(row[dcol]).strip()
                if val.lower() not in ("nan", "none", "null", ""):
                    meta[mkey] = val
        for dcol, mkey in continuous_map.items():
            if dcol in row and pd.notna(row[dcol]):
                try:
                    meta[mkey] = float(row[dcol])
                except (TypeError, ValueError):
                    pass
        metadata_dict[rid] = meta
    return metadata_dict


def _encode_with_continuous(metadata_dict, categorical_values, cat_feature_names, continuous_names, cont_range):
    """One-hot categoricals then append continuous floats (optionally min-max to [0,1])."""
    lo, hi = cont_range
    span = max(hi - lo, 1e-8)
    cat_dim = sum(len(categorical_values[f]) for f in cat_feature_names) if cat_feature_names else 0
    cont_dim = len(continuous_names)
    total_dim = cat_dim + cont_dim
    encoded = {}
    for ann_id, metadata in metadata_dict.items():
        vec = np.zeros(total_dim, dtype=np.float32)
        offset = 0
        for fname in cat_feature_names:
            value = metadata.get(fname, None)
            if value is not None and str(value) in categorical_values[fname]:
                idx = categorical_values[fname].index(str(value))
                vec[offset + idx] = 1.0
            offset += len(categorical_values[fname])
        for i, fname in enumerate(continuous_names):
            raw = metadata.get(fname, None)
            if raw is None:
                continue
            try:
                x = float(raw)
            except (TypeError, ValueError):
                continue
            vec[offset + i] = float(np.clip((x - lo) / span, 0.0, 1.0))
        encoded[ann_id] = vec
    return encoded, total_dim


def get_item_embeddings(df, item_ids, item_col, text_fields, embed_model=None):
    from sentence_transformers import SentenceTransformer

    print(f"[preprocess] loading SentenceTransformer '{embed_model or 'all-mpnet-base-v2'}' ...")
    sys.stdout.flush()
    model = SentenceTransformer(embed_model or "all-mpnet-base-v2")
    texts = []
    for iid in item_ids:
        sub = df[df[item_col] == iid].iloc[0]
        if len(text_fields) == 1:
            msg = str(sub.get(text_fields[0], "")).strip() or "empty"
        else:
            parts = [str(sub.get(f, "")) for f in text_fields]
            msg = " ".join(parts).strip() or "empty"
        texts.append(msg)
    print(f"[preprocess] encoding {len(texts)} items ...")
    sys.stdout.flush()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"[preprocess] embeddings done shape=({len(embs)}, {embs.shape[1] if len(embs) else 0})")
    sys.stdout.flush()
    return {iid: embs[i].astype(np.float32) for i, iid in enumerate(item_ids)}


def run_preprocess(dataset_name: str, split_mode: str, embed_model: str = "all-mpnet-base-v2"):
    print(f"[preprocess] dataset={dataset_name} split_mode={split_mode} embed_model={embed_model}")
    sys.stdout.flush()
    cfg = load_dataset(dataset_name)
    input_csv = split_csv_for(cfg, split_mode)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {input_csv}\n"
            f"Run: python -m diadem.pipeline --dataset {dataset_name} --split-mode {split_mode} --steps split"
        )

    out_dir = processed_dir(dataset_name, split_mode)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[preprocess] input CSV : {input_csv}")
    print(f"[preprocess] output dir: {out_dir}")
    sys.stdout.flush()

    print("[preprocess] reading CSV ...")
    sys.stdout.flush()
    df = pd.read_csv(input_csv)
    print(f"[preprocess] raw rows={len(df)} cols={list(df.columns)[:8]}...")
    for c in cfg["required_columns"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {input_csv}")

    label_map = _label_map_native(cfg["label_map"])
    label_col = cfg["label_col"]
    n_classes = len(cfg["label_names"])

    if df[label_col].dtype in ("int64", "float64", "int32", "float32"):
        df = df[df[label_col].isin(list(label_map.keys()))].copy()
        df["label_idx"] = df[label_col].map(label_map)
    else:
        str_map = {str(k): v for k, v in label_map.items()}
        df = df[df[label_col].astype(str).isin(str_map.keys())].copy()
        df["label_idx"] = df[label_col].astype(str).map(str_map)

    ann_col, item_col = cfg["annotator_col"], cfg["item_col"]
    n_before = len(df)
    df = df.drop_duplicates(subset=[item_col, ann_col], keep="first").reset_index(drop=True)
    if len(df) < n_before:
        print(f"[preprocess] dropped {n_before - len(df)} duplicate (item, annotator) rows")
    print(f"[preprocess] after filter/dedup: {len(df)} rows, {df[item_col].nunique()} items, {df[ann_col].nunique()} annotators")
    sys.stdout.flush()

    print("[preprocess] building metadata dict + one-hot encoding ...")
    sys.stdout.flush()
    metadata_dict = build_metadata_dict(df, cfg)
    # JSON needs serializable values
    with open(out_dir / "annotator_metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    continuous_map = cfg.get("continuous_meta_cols", {}) or {}
    continuous_names = list(continuous_map.values())
    cont_range = cfg.get("continuous_meta_range", [0.0, 1.0])

    # Categorical extraction must ignore continuous float keys
    cat_only = {}
    for rid, meta in metadata_dict.items():
        cat_only[rid] = {k: v for k, v in meta.items() if k not in continuous_names and not isinstance(v, (int, float))}
        # keep string categoricals only; continuous floats excluded above
        for k, v in meta.items():
            if k not in continuous_names and isinstance(v, str):
                cat_only[rid][k] = v

    feature_names, categorical_values, feature_dimensions = extract_categorical_features(cat_only)
    if isinstance(feature_dimensions, dict):
        feature_dims = [feature_dimensions.get(f, 1) for f in feature_names]
    else:
        feature_dims = list(feature_dimensions) if feature_dimensions else [1] * len(feature_names)

    if continuous_names:
        encoded_meta, total_meta_dim = _encode_with_continuous(
            metadata_dict, categorical_values, feature_names, continuous_names, cont_range
        )
        feature_names = feature_names + continuous_names
        feature_dims = feature_dims + [1] * len(continuous_names)
        feature_dimensions = {f: d for f, d in zip(feature_names, feature_dims)}
        print(f"[preprocess] categorical={feature_names[:-len(continuous_names)]} continuous={continuous_names}")
    else:
        encoded_meta, total_meta_dim = encode_metadata_onehot(metadata_dict, categorical_values, feature_names)
        feature_dimensions = feature_dimensions if isinstance(feature_dimensions, dict) else dict(zip(feature_names, feature_dims))
    print(f"[preprocess] features={feature_names} dims={feature_dims} total_meta_dim={total_meta_dim}")
    sys.stdout.flush()

    item_ids = sorted(df[item_col].unique())
    rater_ids = sorted(df[ann_col].unique())
    item_to_idx = {i: idx for idx, i in enumerate(item_ids)}
    rater_to_idx = {r: idx for idx, r in enumerate(rater_ids)}

    if cfg.get("include_annotator_id_onehot"):
        print(f"[preprocess] prepending AnnotatorID one-hot ({len(rater_ids)} dims) ...")
        sys.stdout.flush()
        ann_feat = "AnnotatorID"
        n_raters = len(rater_ids)
        for rid in rater_ids:
            rid_key = str(int(rid)) if isinstance(rid, (int, float, np.integer)) else str(rid)
            ann_onehot = np.zeros(n_raters, dtype=np.float32)
            ann_onehot[rater_to_idx[rid]] = 1.0
            base = encoded_meta.get(rid_key, np.zeros(total_meta_dim, dtype=np.float32))
            encoded_meta[rid_key] = np.concatenate([ann_onehot, base]).astype(np.float32)
        feature_names = [ann_feat] + feature_names
        feature_dims = [n_raters] + feature_dims
        feature_dimensions = {f: d for f, d in zip(feature_names, feature_dims)}
        total_meta_dim = n_raters + total_meta_dim
        cat_with_ann = {ann_feat: [str(int(r)) if isinstance(r, (int, float)) else str(r) for r in rater_ids]}
        cat_with_ann.update(categorical_values)
        save_metadata_encoding_config(feature_names, cat_with_ann, feature_dimensions, str(out_dir / "metadata_encoding_config.json"))
        print(f"[preprocess] total_meta_dim now={total_meta_dim}")
    else:
        save_metadata_encoding_config(feature_names, categorical_values, feature_dimensions, str(out_dir / "metadata_encoding_config.json"))

    with open(out_dir / "metadata_feature_info.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "feature_dimensions": feature_dimensions if isinstance(feature_dimensions, dict) else dict(zip(feature_names, feature_dims)),
            "total_dim": total_meta_dim,
            "continuous_features": continuous_names,
        }, f, indent=2)

    split_column = cfg.get("split_column")
    splits: List = [None]
    if split_column and split_column in df.columns:
        splits = [s for s in df[split_column].unique().tolist() if pd.notna(s)] or [None]
    print(f"[preprocess] splits to build: {splits}")
    sys.stdout.flush()

    prefix = cfg["ail_prefix"]
    for split_name in splits:
        sub = df[df[split_column] == split_name] if split_name is not None else df
        if len(sub) == 0:
            print(f"[preprocess] skip empty split={split_name}")
            continue
        suffix = f"_{split_name}" if split_name else ""
        print(f"[preprocess] --- split={split_name or 'all'} rows={len(sub)} ---")
        sys.stdout.flush()

        split_items = sorted(sub[item_col].unique())
        embeddings = get_item_embeddings(df, split_items, item_col, cfg["text_fields"], embed_model)

        item_labels = sub.groupby(item_col)["label_idx"].apply(
            lambda x: np.bincount(x, minlength=n_classes)
        ).to_dict()
        ann_labels = sub.groupby(ann_col)["label_idx"].apply(
            lambda x: np.bincount(x, minlength=n_classes)
        ).to_dict()

        X_rows, Y_rows, Yi_rows, Ya_rows, I_rows, A_rows, A_base_rows = [], [], [], [], [], [], []
        ail_rows = []

        print(f"[preprocess] building design matrices for {len(sub)} annotations ...")
        sys.stdout.flush()
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Build {split_name or 'all'}"):
            iid, rid, lab = row[item_col], row[ann_col], row["label_idx"]
            emb = embeddings.get(iid)
            if emb is None:
                continue
            rid_key = str(int(rid)) if isinstance(rid, (int, float, np.integer)) else str(rid)
            meta = encoded_meta.get(rid_key, np.zeros(total_meta_dim, dtype=np.float32))

            yi = np.array(item_labels.get(iid, np.zeros(n_classes)), dtype=np.float32)
            yi = yi / (yi.sum() + 1e-8)
            ya = np.array(ann_labels.get(rid, np.zeros(n_classes)), dtype=np.float32)
            ya = ya / (ya.sum() + 1e-8)
            y_onehot = np.zeros(n_classes, dtype=np.float32)
            y_onehot[int(lab)] = 1.0

            ann_idx = rater_to_idx.get(rid, 0)
            A_row = np.concatenate([[float(ann_idx)], meta]).astype(np.float32)
            A_base = np.array([[float(ann_idx)]], dtype=np.float32)

            X_rows.append(emb)
            Y_rows.append(y_onehot)
            Yi_rows.append(yi)
            Ya_rows.append(ya)
            I_rows.append([item_to_idx[iid]])
            A_rows.append(A_row)
            A_base_rows.append(A_base[0])

            y_oh = np.zeros(n_classes, dtype=int)
            y_oh[int(lab)] = 1
            if len(cfg["text_fields"]) == 1:
                msg = str(row.get(cfg["text_fields"][0], "")).strip() or "empty"
            else:
                msg = " ".join(str(row.get(f, "")) for f in cfg["text_fields"]).strip() or "empty"
            ail_rows.append({"item": iid, "annotator": rid, "label": "[" + " ".join(map(str, y_oh)) + "]", "message": msg})

        Xi = np.vstack(X_rows).astype(np.float32)
        print(f"[preprocess] saving npy for {suffix or 'all'} ...")
        sys.stdout.flush()
        np.save(out_dir / f"Xi{suffix}.npy", Xi)
        np.save(out_dir / f"Y{suffix}.npy", np.vstack(Y_rows))
        np.save(out_dir / f"Yi{suffix}.npy", np.vstack(Yi_rows))
        np.save(out_dir / f"Ya{suffix}.npy", np.vstack(Ya_rows))
        np.save(out_dir / f"I{suffix}.npy", np.vstack(I_rows))
        np.save(out_dir / f"A{suffix}.npy", np.vstack(A_rows))
        np.save(out_dir / f"A_baseline{suffix}.npy", np.vstack(A_base_rows).astype(np.float32))

        Ii = np.array([[item_to_idx[i]] for i in split_items], dtype=np.float32)
        np.save(out_dir / f"Ii{suffix}.npy", Ii)

        unique_anns = sub[ann_col].unique()
        ann_meta_rows = []
        for rid in unique_anns:
            rid_key = str(int(rid)) if isinstance(rid, (int, float, np.integer)) else str(rid)
            ann_meta_rows.append(encoded_meta.get(rid_key, np.zeros(total_meta_dim, dtype=np.float32)))
        np.save(out_dir / f"annotator_metadata{suffix}.npy", np.vstack(ann_meta_rows).astype(np.float32))

        pd.DataFrame(ail_rows).to_csv(out_dir / f"{prefix}{suffix}_AIL.csv", index=False, header=False)
        print(f"[preprocess] Saved {suffix or 'all'}: Xi {Xi.shape}, A {np.vstack(A_rows).shape}, meta_dim={total_meta_dim} -> {out_dir}")
        sys.stdout.flush()

    print(f"[preprocess] done -> {out_dir}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dices", "voiced", "d3code"])
    ap.add_argument("--split-mode", required=True, choices=["by_annotator", "by_item"])
    ap.add_argument("--embed-model", default="all-mpnet-base-v2")
    args = ap.parse_args()
    run_preprocess(args.dataset, args.split_mode, args.embed_model)


if __name__ == "__main__":
    main()
