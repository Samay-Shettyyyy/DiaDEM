"""Demographic plurality baseline: majority vote from similar annotators.

For each test (item, annotator):
  1. Find TRAIN annotators who labeled the same item.
  2. Score them by how many demographic attributes match the target
     (Age, Gender, Region, Country, SES, …) plus optional continuous closeness.
  3. Keep the best matches (threshold / top-k) and take a plurality vote.
  4. Fallbacks: all train labels on that item → global train plurality.

Designed for by_annotator cold-start (test annotators unseen in train) where
items fully overlap train. Also works for by_item using train annotations only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

from diadem.data_io import ail_path_for, prepare_run_dir
from diadem.llm_judge import _ece, _jsd_01, _md, _parse_one_hot
from diadem.paths import CONFIGS
from diadem.registry import load_dataset, load_json

RULE_BASELINES = frozenset({"majority_similar"})
DEFAULT_CATEGORICAL = ("Age", "Gender", "Region", "Country", "SES")
DEFAULT_CONTINUOUS = (
    "Care", "Equality", "Proportionality", "Authority", "Loyalty", "Purity",
)


def is_rule_baseline(name: str) -> bool:
    return name in RULE_BASELINES or name.startswith("majority_")


def load_majority_cfg(model_key: str = "majority_similar") -> Dict[str, Any]:
    path = CONFIGS / "majority_similar.json"
    cfg = load_json(path) if path.exists() else {}
    # allow per-key overrides later; single shared config for now
    return {
        "min_cat_matches": int(cfg.get("min_cat_matches", 2)),
        "top_k": int(cfg.get("top_k", 25)),
        "use_continuous": bool(cfg.get("use_continuous", True)),
        "continuous_weight": float(cfg.get("continuous_weight", 1.0)),
        "categorical_features": list(cfg.get("categorical_features") or DEFAULT_CATEGORICAL),
        "continuous_features": list(cfg.get("continuous_features") or DEFAULT_CONTINUOUS),
        "model_key": model_key,
    }


def _majority_vote(labels: Sequence[int]) -> int:
    """Plurality with deterministic tie-break: smallest label."""
    if not labels:
        return 0
    counts = Counter(int(x) for x in labels)
    max_c = max(counts.values())
    return int(min(lab for lab, c in counts.items() if c == max_c))


def _load_ail(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=["item", "annotator", "label", "message"])
    raw["y"] = raw["label"].map(lambda s: int(np.argmax(_parse_one_hot(s))))
    raw["annotator"] = raw["annotator"].astype(str)
    raw["item"] = raw["item"].astype(str)
    return raw


def _meta_lookup(data_dir: Path) -> Dict[str, dict]:
    path = data_dir / "annotator_metadata.json"
    if not path.exists():
        # processed/ sibling
        alt = data_dir.parent / "processed" / "annotator_metadata.json"
        path = alt if alt.exists() else path
    if not path.exists():
        raise FileNotFoundError(
            f"Need annotator_metadata.json in {data_dir} (or processed/) — run preprocess/prepare"
        )
    with open(path) as f:
        raw = json.load(f)
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in raw.items()}


def _cat_match_count(a: dict, b: dict, feats: Sequence[str]) -> int:
    n = 0
    for f in feats:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        if str(va) == str(vb):
            n += 1
    return n


def _cont_closeness(a: dict, b: dict, feats: Sequence[str]) -> float:
    """Mean 1 - |a-b|/range proxy; missing → 0 contribution. Values are typically in ~[1,5] or [0,1]."""
    diffs = []
    for f in feats:
        try:
            va = float(a.get(f))
            vb = float(b.get(f))
        except (TypeError, ValueError):
            continue
        # assume moral foundations ~[1,5] or already [0,1]
        span = 4.0 if max(va, vb) > 1.5 else 1.0
        diffs.append(1.0 - min(1.0, abs(va - vb) / span))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _similarity(
    target: dict,
    other: dict,
    cat_feats: Sequence[str],
    cont_feats: Sequence[str],
    use_continuous: bool,
    continuous_weight: float,
) -> Tuple[int, float]:
    cat = _cat_match_count(target, other, cat_feats)
    cont = _cont_closeness(target, other, cont_feats) if use_continuous else 0.0
    score = float(cat) + continuous_weight * cont
    return cat, score


def run_majority_similar(
    data_dir: Path,
    model_key: str,
    dataset: str,
    split_mode: str = "by_annotator",
) -> dict:
    data_dir = Path(data_dir)
    cfg = load_majority_cfg(model_key)
    ds = load_dataset(dataset)
    label_names = ds["label_names"]
    n_classes = len(label_names)

    train_ail = ail_path_for(data_dir, dataset, "train")
    test_ail = ail_path_for(data_dir, dataset, "test")
    if not train_ail.exists() or not test_ail.exists():
        raise FileNotFoundError(f"Need train+test AIL under {data_dir}")

    print(f"[majority_similar] model={model_key} dataset={dataset} dir={data_dir}")
    print(
        f"[majority_similar] min_cat_matches={cfg['min_cat_matches']} "
        f"top_k={cfg['top_k']} use_continuous={cfg['use_continuous']}"
    )
    sys.stdout.flush()

    train = _load_ail(train_ail)
    test = _load_ail(test_ail)
    meta = _meta_lookup(data_dir)

    # item -> list of (annotator, label)
    train_by_item: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for _, row in train.iterrows():
        train_by_item[str(row["item"])].append((str(row["annotator"]), int(row["y"])))

    global_plurality = _majority_vote(train["y"].tolist())
    item_plurality = {
        item: _majority_vote([y for _, y in pairs])
        for item, pairs in train_by_item.items()
    }

    cat_feats = cfg["categorical_features"]
    cont_feats = cfg["continuous_features"]
    min_m = cfg["min_cat_matches"]
    top_k = max(1, cfg["top_k"])

    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[np.ndarray] = []
    used_similar = 0
    used_item_fb = 0
    used_global_fb = 0
    n_neighbors: List[int] = []

    # For soft item metrics: group by item after loop
    pred_by_row: List[int] = []

    for _, row in tqdm(test.iterrows(), total=len(test), desc=f"majority/{model_key}"):
        item = str(row["item"])
        ann = str(row["annotator"])
        yt = int(row["y"])
        target_meta = meta.get(ann, {})
        candidates = train_by_item.get(item, [])

        scored: List[Tuple[float, int, int]] = []  # (-score, cat, label) for sort
        for cann, cy in candidates:
            if cann == ann:
                continue  # safety for by_item overlap
            om = meta.get(cann, {})
            cat, score = _similarity(
                target_meta, om, cat_feats, cont_feats,
                cfg["use_continuous"], cfg["continuous_weight"],
            )
            scored.append((-score, cat, cy))

        # Prefer neighbors with cat >= min_m; else best available on the item
        strong = [t for t in scored if t[1] >= min_m]
        pool = strong if strong else scored
        pool.sort()  # highest similarity first
        pool = pool[:top_k]
        labels = [lab for _, _, lab in pool]

        if labels:
            yp = _majority_vote(labels)
            used_similar += 1
            n_neighbors.append(len(labels))
        elif item in item_plurality:
            yp = item_plurality[item]
            used_item_fb += 1
            n_neighbors.append(0)
        else:
            yp = global_plurality
            used_global_fb += 1
            n_neighbors.append(0)

        y_true.append(yt)
        y_pred.append(yp)
        pred_by_row.append(yp)
        p = np.zeros(n_classes, dtype=float)
        p[yp] = 1.0
        probs.append(p)

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    probs_a = np.asarray(probs)

    # Item-level soft metrics vs empirical test hist
    emp_hists, pred_dists, maj_true, maj_pred = [], [], [], []
    test = test.reset_index(drop=True)
    for item, idxs in test.groupby("item").groups.items():
        idxs = list(idxs)
        hist = np.zeros(n_classes, dtype=float)
        for i in idxs:
            hist[y_true_a[i]] += 1
        hist = hist / max(hist.sum(), 1.0)
        # predicted distribution: hard votes over annotators on item
        ph = np.zeros(n_classes, dtype=float)
        for i in idxs:
            ph[y_pred_a[i]] += 1
        ph = ph / max(ph.sum(), 1.0)
        emp_hists.append(hist)
        pred_dists.append(ph)
        maj_true.append(int(np.argmax(hist)))
        maj_pred.append(int(np.argmax(ph)))

    jsd_vals = [_jsd_01(e, p) for e, p in zip(emp_hists, pred_dists)]
    md_vals = [_md(e, p) for e, p in zip(emp_hists, pred_dists)]
    er = float(np.mean(y_true_a != y_pred_a))

    summary = {
        "model": model_key,
        "dataset": dataset,
        "split_mode": split_mode,
        "protocol": "demographic_plurality_similar_annotators",
        "n_annotations": int(len(test)),
        "n_items": int(test["item"].nunique()),
        "label_names": label_names,
        "config": {
            "min_cat_matches": cfg["min_cat_matches"],
            "top_k": cfg["top_k"],
            "use_continuous": cfg["use_continuous"],
            "categorical_features": cfg["categorical_features"],
            "continuous_features": cfg["continuous_features"],
        },
        "routing": {
            "used_similar": int(used_similar),
            "used_item_fallback": int(used_item_fb),
            "used_global_fallback": int(used_global_fb),
            "mean_neighbors": float(np.mean(n_neighbors)) if n_neighbors else 0.0,
        },
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "item_majority_accuracy": float(accuracy_score(maj_true, maj_pred)),
        "f1_macro": float(f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true_a, y_pred_a)),
        "mcc": float(matthews_corrcoef(y_true_a, y_pred_a)) if len(np.unique(y_true_a)) > 1 else 0.0,
        "jsd_mean": float(np.mean(jsd_vals)),
        "md_mean": float(np.mean(md_vals)),
        "er_mean": er,
        "ece": _ece(probs_a, y_true_a),
    }

    with open(data_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    err_dir = data_dir / "error_analysis"
    err_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "metrics": {
            "accuracy": summary["accuracy"],
            "f1_macro": summary["f1_macro"],
            "f1_weighted": summary["f1_weighted"],
            "kappa": summary["kappa"],
            "mcc": summary["mcc"],
            "perspectivist_average_er": summary["er_mean"],
            "perspectivist_average_md": summary["md_mean"],
            "mean_jsd": summary["jsd_mean"],
            "ece": summary["ece"],
        },
        "protocol": summary["protocol"],
        "model": model_key,
        "routing": summary["routing"],
        "note": (
            "For each test annotator/item, plurality vote over train annotators "
            "on the same item with the most matching demographics "
            f"(min_cat_matches>={cfg['min_cat_matches']}, top_k={cfg['top_k']})."
        ),
    }
    with open(err_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Persist per-row predictions for debugging
    out_pred = data_dir / "majority_similar_predictions.csv"
    pd.DataFrame(
        {
            "item": test["item"].tolist(),
            "annotator": test["annotator"].tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "n_neighbors": n_neighbors,
        }
    ).to_csv(out_pred, index=False)

    print(json.dumps(summary, indent=2))
    print(f"[majority_similar] wrote {data_dir / 'eval_summary.json'}")
    print(f"[majority_similar] wrote {err_dir / 'analysis_summary.json'}")
    sys.stdout.flush()
    return summary


def prepare_majority_run_dir(dataset: str, split_mode: str, model: str) -> Path:
    return prepare_run_dir(dataset, split_mode, model)
