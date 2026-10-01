"""
Evaluation script for DEM-MoE.

Loads dem_moe_model.pt + dem_moe_config.json, runs per-annotator predictions,
and writes the same output files as diadem/eval.py + error_detailed.py so that
compare.py picks them up automatically:

  eval_summary.json             — basic metrics (KL, accuracy, F1 …)
  error_analysis/
    analysis_summary.json       — full metrics incl. RPS, Wasserstein, EPD …
    error_analysis_detailed.csv — per-annotation CSV
    confusion_matrix.pdf
    disagreement_correlation.pdf
    disagreement_calibration_curve.pdf
    ece_reliability_diagram.pdf
    mean_response_distributions.pdf
"""

import json
import os
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, matthews_corrcoef,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.5)

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from diadem.models.dem_moe import DemMoEClassifier
from diadem.train_dem_moe import parse_metadata_config, build_tensors


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (same as error_detailed.py)
# ──────────────────────────────────────────────────────────────────────────────

def _jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    return float(np.clip((jensenshannon(p, q) ** 2) / np.log(2), 0, 1))

def _rps(p, q):
    K = len(p)
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    return float(np.clip(np.sum((np.cumsum(q)[:-1] - np.cumsum(p)[:-1]) ** 2) / (K-1), 0, 1))

def _wass(p, q):
    K = len(p)
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    vals = np.arange(K, dtype=float)
    return float(wasserstein_distance(vals, vals, p, q))

def _hellinger(p, q):
    K = len(p)
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    return float((1/np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

def _bhattacharyya(p, q):
    K = len(p)
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    bc = float(np.sum(np.sqrt(p * q)))
    return bc, float(-np.log(max(bc, 1e-10)))

def _epd(p):
    p = np.asarray(p, float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    return float(1 - np.sum(p ** 2))

def _dist_entropy(p):
    p = np.asarray(p, float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    return float(-np.sum(p * np.log(p + 1e-10)))

def _dist_variance(p):
    p = np.asarray(p, float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    k = np.arange(len(p))
    mean = np.dot(k, p); var = np.dot(k**2, p) - mean**2
    return float(max(var, 0))

def _ece(all_probs, all_labels, n_bins=10):
    all_probs  = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)
    confs = np.max(all_probs, axis=1)
    preds = np.argmax(all_probs, axis=1)
    correct = (preds == all_labels).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins+1)[:-1], np.linspace(0, 1, n_bins+1)[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            ece += (mask.sum() / len(all_labels)) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(run_dir: Path, device: torch.device):
    cfg_path = run_dir / "dem_moe_config.json"
    wt_path  = run_dir / "dem_moe_model.pt"
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = DemMoEClassifier(
        text_embed_dim   = cfg["text_embed_dim"],
        num_annotators   = cfg["num_annotators"],
        num_cat_per_attr = cfg["num_cat_per_attr"],
        num_cont_feats   = cfg["num_cont_feats"],
        num_classes      = cfg["num_classes"],
        hidden_dim       = cfg.get("hidden_dim", 256),
        num_experts      = cfg.get("num_experts", 8),
        ann_embed_dim    = cfg.get("ann_embed_dim", 128),
        id_embed_dim     = cfg.get("id_embed_dim", 64),
        projected_id_dim = cfg.get("projected_id_dim", 256),
        dropout          = 0.0,  # no dropout at eval
        top_k            = cfg.get("top_k", 3),
    ).to(device)

    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.eval()
    return model, cfg


def _predict_all(model, cfg, Xi, A, device, batch_size=512):
    """
    Run per-row predictions.
    Returns probs: (N, num_classes) numpy array.
    """
    # Rebuild tensors using saved config
    cat_sizes    = cfg["num_cat_per_attr"]
    cont_start   = cfg["cont_start"]
    num_cont     = cfg["num_cont_feats"]
    id_map_str   = cfg["annotator_id_map"]           # str keys
    id_map       = {int(k): v for k, v in id_map_str.items()}

    meta = A[:, 1:].astype(np.float32)
    ann_ids_raw = A[:, 0].astype(np.int64)

    # Map raw annotator IDs; OOV → 0 (first annotator as fallback)
    mapped_ann = np.array([id_map.get(int(x), 0) for x in ann_ids_raw], dtype=np.int64)

    # Reconstruct per-attribute integer indices
    offset = 0
    cat_slices = []
    for sz in cat_sizes:
        cat_slices.append(slice(offset, offset + sz))
        offset += sz

    id_cols = np.stack(
        [np.argmax(meta[:, s], axis=1).astype(np.int64) for s in cat_slices],
        axis=1
    )

    cont_feats = meta[:, cont_start: cont_start + num_cont] if num_cont > 0 else None

    N = len(Xi)
    all_probs = np.zeros((N, cfg["num_classes"]), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bxi  = torch.from_numpy(Xi[start:end]).to(device)
            bann = torch.from_numpy(mapped_ann[start:end]).to(device)
            bid  = torch.from_numpy(id_cols[start:end]).to(device)
            bcont = (torch.from_numpy(cont_feats[start:end]).to(device)
                     if cont_feats is not None else None)

            logits, _, _ = model(bxi, bann, bid, bcont)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs[start:end] = probs

    return all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation + analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    data_dir:       Path,
    model_name:     str  = "dem_moe",
    dataset:        str  = "d3code",
    gpu_id:         int  = 0,
    split_name:     str  = "test",
):
    """Pipeline entry point for 'eval' step."""
    from diadem.data_io  import ail_path_for
    from diadem.registry import load_dataset

    data_dir = Path(data_dir)
    device   = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available()
                            else "cpu")
    print(f"[dem_moe eval] dataset={dataset} split={split_name} device={device}")

    cfg_ds     = load_dataset(dataset)
    label_names = cfg_ds["label_names"]
    ail_path   = ail_path_for(data_dir, dataset, split_name)

    Xi  = np.load(data_dir / f"Xi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    Yi  = np.load(data_dir / f"Yi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    A   = np.load(data_dir / f"A_{split_name}.npy",   allow_pickle=True)
    raw = pd.read_csv(ail_path, header=None, names=["item", "annotator", "label", "message"])

    model, cfg_m = _load_model(data_dir, device)
    print(f"[dem_moe eval] predicting {len(Xi)} rows …")
    all_probs = _predict_all(model, cfg_m, Xi, A, device)

    unique_items = raw["item"].unique()
    # row.item would resolve to the pandas Series.item() method — use groupby instead
    _first = raw.groupby("item", sort=False).apply(lambda g: g.index[0])
    item_to_first = _first.to_dict()

    empirical_labels, predictions, data_to_write = [], [], []
    for uid in tqdm(unique_items, desc="eval items"):
        first_idx = item_to_first[uid]
        empirical_labels.append(Yi[first_idx].tolist())
        pred_dist = all_probs[first_idx].tolist()
        predictions.append(pred_dist)

        item_anns = raw[raw["item"] == uid]
        ann_preds = [int(np.argmax(all_probs[r])) for r in item_anns.index if r < len(all_probs)]
        data_to_write.append({
            "message_id": uid,
            "labels":     pred_dist,
            "annotator_ids": item_anns["annotator"].tolist(),
            "annotator_labels": ann_preds,
        })

    # Basic metrics
    y_true = [int(np.argmax(e)) for e in empirical_labels]
    y_pred = [int(np.argmax(p)) for p in predictions]

    summary = {
        "model":     model_name,
        "dataset":   dataset,
        "split":     split_name,
        "n_items":   len(unique_items),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro":  float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "label_names": label_names,
    }

    with open(data_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(data_to_write).to_excel(data_dir / "preds.xlsx", index=False)
    with open(data_dir / "op_predictions.tsv", "w") as f:
        for row in data_to_write:
            f.write(f"{row['message_id']}\t{row['labels']}\n")

    print(f"[dem_moe eval] accuracy={summary['accuracy']:.4f}  "
          f"F1-macro={summary['f1_macro']:.4f}")
    print(f"[dem_moe eval] saved eval_summary.json")
    return summary


def run_error_detailed(
    data_dir:       Path,
    model_fname:    Path,   # unused — model loaded from dem_moe_model.pt
    empirical_fname: Path,
    split_name:     str = "test",
    gpu_id:         int = 0,
):
    """Pipeline entry point for 'error_detailed' step."""
    data_dir = Path(data_dir)
    out_dir  = data_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available()
                          else "cpu")

    # Load
    Xi  = np.load(data_dir / f"Xi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    Yi  = np.load(data_dir / f"Yi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    A   = np.load(data_dir / f"A_{split_name}.npy",   allow_pickle=True)
    raw = pd.read_csv(empirical_fname, header=None)
    if raw.shape[1] >= 4:
        raw.columns = ["item", "annotator", "label", "message"]
    else:
        raw.columns = ["item", "annotator", "label"]
        raw["message"] = ""
    raw = raw.dropna(subset=["item", "annotator", "label"])

    model, cfg_m = _load_model(data_dir, device)
    print(f"[dem_moe error] predicting {len(Xi)} rows …")
    all_probs = _predict_all(model, cfg_m, Xi, A, device)

    n_classes   = cfg_m["num_classes"]
    label_names = [str(i) for i in range(n_classes)]

    # ── Per-item analysis ──────────────────────────────────────────────────────
    def parse_one_hot(s):
        s = str(s).strip().replace("[","").replace("]","")
        return np.array([int(x) for x in s.split()])

    item_to_first = {}
    for idx, row in raw.iterrows():
        if row["item"] not in item_to_first:
            item_to_first[row["item"]] = idx

    item_results = []
    ann_results  = []

    for uid in tqdm(raw["item"].unique(), desc="error analysis"):
        if uid not in item_to_first:
            continue
        fidx = int(item_to_first[uid])
        if fidx >= len(Xi):
            continue

        item_anns   = raw[raw["item"] == uid]
        row_indices = item_anns.index.tolist()
        actual_lbls = []
        for l in item_anns["label"].tolist():
            oh = parse_one_hot(l)
            actual_lbls.append(int(np.argmax(oh)))

        # Predicted probs per annotator (use each annotator's own row)
        ann_probs = []
        for ri in row_indices:
            if ri < len(all_probs):
                ann_probs.append(all_probs[ri].tolist())
        if not ann_probs:
            continue

        ann_preds = [int(np.argmax(p)) for p in ann_probs]

        hist = np.zeros(n_classes)
        for lbl in actual_lbls:
            if 0 <= lbl < n_classes:
                hist[lbl] += 1

        mean_pred = np.mean(ann_probs, axis=0).tolist()

        # Actual disagreement
        labels    = np.array(actual_lbls)
        lc        = Counter(actual_lbls)
        total     = len(labels)
        act_var   = float(np.var(labels))
        lp        = [c/total for c in lc.values()]
        act_ent   = float(-sum(p*np.log2(p+1e-10) for p in lp))

        # Predicted disagreement
        dist_vars = [_dist_variance(p) for p in ann_probs]
        dist_ents = [_dist_entropy(p)  for p in ann_probs]

        # All metrics
        jsd  = _jsd(hist, mean_pred)
        rps  = _rps(hist, mean_pred)
        wass = _wass(hist, mean_pred)
        hell = _hellinger(hist, mean_pred)
        bc, bd = _bhattacharyya(hist, mean_pred)
        act_epd  = _epd(hist)
        pred_epd = _epd(mean_pred)
        msg = item_anns["message"].iloc[0] if "message" in item_anns.columns else ""

        item_results.append({
            "item_id": uid,
            "actual_labels": actual_lbls,
            "predicted_labels": ann_preds,
            "empirical_hist": hist.tolist(),
            "mean_predicted_dist": mean_pred,
            "actual_disagreement": {"variance": act_var, "entropy": act_ent},
            "predicted_disagreement": {
                "dist_variance_mean": float(np.mean(dist_vars)),
                "dist_entropy_mean":  float(np.mean(dist_ents)),
            },
            "jsd": jsd, "rps": rps, "wasserstein": wass,
            "hellinger": hell, "bhattacharyya_coeff": bc, "bhattacharyya_dist": bd,
            "actual_epd": act_epd, "predicted_epd": pred_epd,
        })

        ann_ids = item_anns["annotator"].tolist()
        for i, aid in enumerate(ann_ids[: len(ann_probs)]):
            ann_results.append({
                "item_id": uid, "annotator_id": aid,
                "actual_label": actual_lbls[i],
                "predicted_label": ann_preds[i],
                "correct": int(actual_lbls[i] == ann_preds[i]),
                "message": str(msg) if pd.notna(msg) else "",
                "predicted_probs": ann_probs[i],
                "item_actual_variance": act_var,
                "item_predicted_variance_dist": float(np.mean(dist_vars)),
                "item_predicted_entropy_dist":  float(np.mean(dist_ents)),
                "item_jsd": jsd, "item_rps": rps, "item_wasserstein": wass,
                "item_hellinger": hell,
                "item_bhattacharyya_coeff": bc, "item_bhattacharyya_dist": bd,
                "item_actual_epd": act_epd, "item_predicted_epd": pred_epd,
            })

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    actual_all = [r["actual_label"]    for r in ann_results]
    pred_all   = [r["predicted_label"] for r in ann_results]
    probs_all  = np.array([r["predicted_probs"] for r in ann_results])

    accuracy   = float(accuracy_score(actual_all, pred_all))
    f1_macro   = float(f1_score(actual_all, pred_all, average="macro",    zero_division=0))
    f1_weighted= float(f1_score(actual_all, pred_all, average="weighted", zero_division=0))
    kappa      = float(cohen_kappa_score(actual_all, pred_all))
    mcc        = float(matthews_corrcoef(actual_all, pred_all))
    cm         = confusion_matrix(actual_all, pred_all, labels=list(range(n_classes)))

    # Soft-label
    mean_jsd  = float(np.mean([r["jsd"]        for r in item_results]))
    mean_rps  = float(np.mean([r["rps"]        for r in item_results]))
    mean_wass = float(np.mean([r["wasserstein"] for r in item_results]))
    mean_hell = float(np.mean([r["hellinger"]  for r in item_results]))
    mean_bc   = float(np.mean([r["bhattacharyya_coeff"] for r in item_results]))
    mean_bd   = float(np.mean([r["bhattacharyya_dist"]  for r in item_results]))
    mean_act_epd  = float(np.mean([r["actual_epd"]    for r in item_results]))
    mean_pred_epd = float(np.mean([r["predicted_epd"] for r in item_results]))

    # MD (perspectivist)
    def _md(targets_soft, pred_soft):
        mds = []
        for t_list, p_list in zip(targets_soft, pred_soft):
            item_mds = []
            for t, p in zip(t_list, p_list):
                t, p = np.asarray(t, float), np.asarray(p, float)
                item_mds.append(float(np.sum(np.abs(t - p))))
            mds.append(np.mean(item_mds) if item_mds else 0.0)
        return float(np.mean(mds)) if mds else 0.0

    targets_soft, preds_soft = [], []
    for r in item_results:
        uid = r["item_id"]
        rows = [x for x in ann_results if x["item_id"] == uid]
        oh_list = []; pr_list = []
        for x in rows:
            oh = np.zeros(n_classes); oh[x["actual_label"]] = 1.0
            oh_list.append(oh.tolist())
            p = list(x["predicted_probs"]); p = (p + [0.0]*n_classes)[:n_classes]
            pr_list.append(p)
        targets_soft.append(oh_list); preds_soft.append(pr_list)
    mean_md = _md(targets_soft, preds_soft)

    ece = _ece(probs_all, np.array(actual_all))

    # Disagreement correlation
    act_var  = [r["actual_disagreement"]["variance"] for r in item_results]
    pred_var = [r["predicted_disagreement"]["dist_variance_mean"] for r in item_results]
    act_ent  = [r["actual_disagreement"]["entropy"]  for r in item_results]
    pred_ent = [r["predicted_disagreement"]["dist_entropy_mean"]  for r in item_results]
    act_epd  = [r["actual_epd"]    for r in item_results]
    pred_epd_l = [r["predicted_epd"] for r in item_results]

    def _corr(x, y, label):
        pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
        print(f"  [{label}] Pearson r={pr:.4f} (p={pp:.2e}) | Spearman ρ={sr:.4f} (p={sp:.2e})")
        return {"pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_r": float(sr), "spearman_p": float(sp)}

    print("\n── DEM-MoE Disagreement Correlation ──")
    var_c = _corr(act_var,  pred_var,    "Variance (dist)")
    ent_c = _corr(act_ent,  pred_ent,    "Entropy  (dist)")
    epd_c = _corr(act_epd,  pred_epd_l,  "EPD (NEW)")

    # ── Build analysis_summary.json ───────────────────────────────────────────
    results = {
        "metrics": {
            "accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
            "kappa": kappa, "mcc": mcc,
            "mean_jsd": mean_jsd, "perspectivist_average_md": mean_md,
            "perspectivist_average_er": 0.0,  # not computed here
            "ece": ece,
            "mean_rps": mean_rps, "mean_wasserstein": mean_wass,
            "mean_hellinger": mean_hell,
            "mean_bhattacharyya_coeff": mean_bc, "mean_bhattacharyya_dist": mean_bd,
            "mean_actual_epd": mean_act_epd, "mean_predicted_epd": mean_pred_epd,
            "confusion_matrix": cm.tolist(),
        },
        "disagreement_correlation": {
            "distribution_level": {"variance": var_c, "entropy": ent_c},
            "epd": epd_c,
        },
    }

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_names[:n_classes])
    disp.plot(include_values=True, cmap=plt.cm.Blues)
    plt.title("DEM-MoE — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Disagreement correlation plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (x, y, title, stats) in zip(axes.flat, [
        (act_var,   pred_var,    "Variance (dist)",  var_c),
        (act_ent,   pred_ent,    "Entropy  (dist)",  ent_c),
        (act_epd,   pred_epd_l, "EPD (NEW)",         epd_c),
    ]):
        ax.scatter(x, y, alpha=0.4, s=15)
        mn, mx = min(x+y), max(x+y)
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlabel(f"Actual {title.split()[0]}")
        ax.set_ylabel(f"Predicted {title.split()[0]}")
        ax.set_title(f"{title}\nr={stats['pearson_r']:.3f}, ρ={stats['spearman_r']:.3f}")
        ax.grid(True, alpha=0.3)
    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "disagreement_correlation.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # ECE reliability diagram
    bins = np.linspace(0, 1, 11)
    confs = np.max(probs_all, axis=1)
    preds_arr = np.argmax(probs_all, axis=1)
    correct = (preds_arr == np.array(actual_all)).astype(float)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() > 0:
            bin_acc.append(correct[m].mean())
            bin_conf.append(confs[m].mean())
        else:
            bin_acc.append(0); bin_conf.append((lo+hi)/2)
    fig, ax = plt.subplots(figsize=(6, 5))
    bc = (bins[:-1]+bins[1:])/2
    ax.bar(bc, bin_acc, width=0.08, alpha=0.7, color="steelblue", edgecolor="white", label="Accuracy")
    ax.plot([0,1],[0,1],"r--",lw=2,label="Perfect")
    ax.set(xlabel="Confidence", ylabel="Accuracy",
           title=f"Reliability Diagram (ECE={ece:.4f})", xlim=(0,1), ylim=(0,1))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ece_reliability_diagram.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Disagreement calibration curve
    act_var_arr  = np.array(act_var)
    pred_var_arr = np.array(pred_var)
    act_ent_arr  = np.array(act_ent)
    pred_ent_arr = np.array(pred_ent)

    def calib_bins(a, p, n=10):
        edges = np.quantile(a, np.linspace(0, 1, n+1))
        ba, bp, bs = [], [], []
        for i in range(n):
            mask = (a >= edges[i]) & (a <= edges[i+1])
            if mask.sum() > 0:
                ba.append(float(a[mask].mean()))
                bp.append(float(p[mask].mean()))
                bs.append(float(p[mask].std()))
        return np.array(ba), np.array(bp), np.array(bs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (a, p, lbl) in zip(axes, [
        (act_var_arr, pred_var_arr, "Variance"),
        (act_ent_arr, pred_ent_arr, "Entropy"),
    ]):
        ba, bp, bs = calib_bins(a, p)
        ax.fill_between(ba, bp-bs, bp+bs, alpha=0.2, color="steelblue", label="±1 std")
        ax.plot(ba, bp, "o-", color="steelblue", lw=2, ms=6, label="Model")
        mn, mx = min(ba.min(), bp.min()), max(ba.max(), bp.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
        ax.set(xlabel=f"Actual {lbl}", ylabel=f"Predicted {lbl}",
               title=f"Disagreement Calibration — {lbl}")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "disagreement_calibration_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save CSV
    rows = []
    for r in ann_results:
        probs = (list(r["predicted_probs"]) + [0.0]*n_classes)[:n_classes]
        row = {k: v for k, v in r.items() if k != "predicted_probs"}
        for i in range(n_classes):
            row[f"predicted_prob_class_{i}"] = float(probs[i])
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(out_dir / "error_analysis_detailed.csv", index=False)

    # Save JSON
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    m = results["metrics"]
    dc = results["disagreement_correlation"]
    print(f"\n{'='*70}")
    print("DEM-MoE SUMMARY")
    print(f"{'='*70}")
    print(f"  Accuracy:              {m['accuracy']:.4f}")
    print(f"  F1 (macro):            {m['f1_macro']:.4f}")
    print(f"  F1 (weighted):         {m['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:         {m['kappa']:.4f}")
    print(f"  MCC:                   {m['mcc']:.4f}")
    print(f"  Mean JSD [0,1]:        {m['mean_jsd']:.5f}")
    print(f"  ECE:                   {m['ece']:.5f}")
    print(f"  Mean RPS [0,1]:        {m['mean_rps']:.5f}  [ORDINAL]")
    print(f"  Mean Wasserstein-1:    {m['mean_wasserstein']:.5f}  [ORDINAL]")
    print(f"  Mean Hellinger [0,1]:  {m['mean_hellinger']:.5f}")
    print(f"  Mean Bhattacharyya BC: {m['mean_bhattacharyya_coeff']:.5f}")
    print(f"  Mean Actual EPD:       {m['mean_actual_epd']:.5f}")
    print(f"  Mean Predicted EPD:    {m['mean_predicted_epd']:.5f}")
    ep = dc.get("epd", {})
    print(f"  EPD Pearson r:         {ep.get('pearson_r',0):.4f}  (p={ep.get('pearson_p',1):.2e})")
    print(f"  EPD Spearman ρ:        {ep.get('spearman_r',0):.4f}  (p={ep.get('spearman_p',1):.2e})")
    print(f"{'='*70}")
    print(f"\nOutputs → {out_dir}/")

    return Path(out_dir) / "analysis_summary.json"
