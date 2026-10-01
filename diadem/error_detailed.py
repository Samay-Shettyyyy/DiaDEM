"""
Comprehensive Error Analysis for DiADEM / DisCo models (VOICED, DICES, D3Code).

Mirrors `dices/error_detailed.py` metrics and outputs:
  - Standard: Accuracy, F1 (macro/weighted), Cohen's Kappa, MCC
  - Soft-label: Mean JSD [0,1] (LeWiDi 2025 primary), Perspectivist MD
  - Perspectivist hard-label ER
  - Disagreement-aware: distribution-level variance/entropy + correlation + calibration curve
  - Calibration: ECE + reliability diagram
  - NEW — Ordinal-aware (respects label ordering):
      * Ranked Probability Score (RPS) — primary ordinal metric
      * Wasserstein-1 / Earth Mover's Distance
  - NEW — Symmetric distribution divergences:
      * Hellinger Distance — bounded proper metric
      * Bhattacharyya Coefficient + Distance
  - NEW — Disagreement probability:
      * Expected Pairwise Disagreement (EPD) = 1 - sum(p_k^2)
        + Pearson/Spearman correlation actual vs predicted EPD

Usage:
  python error_detailed.py --data_dir=./voices/experimental_data/voiced_by_annotator/ \
      --model_fname=./voices/experimental_data/voiced_by_annotator/trained_model.disco \
      --split_name=test --empirical_fname=./voices/experimental_data/voiced_by_annotator/voiced_test_AIL.csv \
      --gpu_id=0
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from diadem.utils.utils import load_object

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.5)


# VOICED is binary in this repo's preprocessing: PERSON_TOXIC ∈ {0,1}
LABEL_NAMES = ["0", "1"]


def parse_one_hot(s):
    s = str(s).strip().replace("[", "").replace("]", "")
    return np.array([int(x) for x in s.split()])


def error_rate(targets, predictions):
    """Perspectivist hard-label ER (lower is better)."""
    match_scores = []
    for target, prediction in zip(targets, predictions):
        errors = sum(abs(t - p) for t, p in zip(target, prediction))
        match_score = round(1 - (len(target) - errors) / len(target), 5) if target else 0.0
        match_scores.append(match_score)
    return float(np.mean(match_scores)) if match_scores else 0.0


def average_MD(target_dists, pred_dists):
    """Average Manhattan Distance between distributions for one item (over annotators)."""
    if not target_dists or not pred_dists:
        return 0.0
    mds = []
    for t, p in zip(target_dists, pred_dists):
        t, p = np.asarray(t, dtype=float), np.asarray(p, dtype=float)
        n = max(len(t), len(p))
        t = np.resize(t, n) if len(t) < n else t[:n]
        p = np.resize(p, n) if len(p) < n else p[:n]
        mds.append(float(np.sum(np.abs(t - p))))
    return float(np.mean(mds))


def multilabel_average_MD(targets, predictions):
    """Soft Perspectivist score (lower is better)."""
    if not targets or not predictions:
        return 0.0
    soft_scores = [average_MD(targets[i], predictions[i]) for i in range(len(targets))]
    return round(float(np.mean(soft_scores)), 5)


def compute_jsd_item(empirical_hist, predicted_dist):
    """
    Jensen-Shannon Divergence between empirical label histogram (normalized)
    and mean predicted distribution for one item.
    Returns JSD in [0,1] (normalized by log(2)). Lower is better.
    """
    p = np.asarray(empirical_hist, dtype=float)
    q = np.asarray(predicted_dist, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    else:
        p = np.ones_like(p) / len(p)
    if q.sum() > 0:
        q = q / q.sum()
    else:
        q = np.ones_like(q) / len(q)
    js_distance = jensenshannon(p, q)  # sqrt(JSD) with natural log base
    jsd = (js_distance ** 2) / np.log(2)
    return float(np.clip(jsd, 0.0, 1.0))


def distribution_variance(prob_dist):
    """Distribution-level variance over class indices."""
    p = np.asarray(prob_dist, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    k = np.arange(len(p))
    mean = np.dot(k, p)
    var = np.dot(k ** 2, p) - mean ** 2
    return float(max(var, 0.0))


def distribution_entropy(prob_dist):
    """Shannon entropy (nats)."""
    p = np.asarray(prob_dist, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def compute_ece(all_probs, all_labels, n_bins=10):
    """Expected Calibration Error (ECE). Lower is better."""
    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    correct = (predictions == all_labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece = 0.0
    n = len(all_labels)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correct[mask].mean())
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            ece += (count / n) * abs(avg_acc - avg_conf)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return {
        "ece": float(ece),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def compute_rps_item(empirical_hist, predicted_dist):
    """
    Ranked Probability Score (RPS) for ordinal labels.
    RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_pred[k] - CDF_empirical[k])^2
    Lower is better. Range [0,1]. Respects label ordering (unlike JSD/KL).
    """
    K = len(empirical_hist)
    p = np.asarray(empirical_hist, dtype=float)
    q = np.asarray(predicted_dist, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    cdf_p, cdf_q = np.cumsum(p), np.cumsum(q)
    return float(np.clip(np.sum((cdf_q[:-1] - cdf_p[:-1]) ** 2) / (K - 1), 0.0, 1.0))


def compute_wasserstein_item(empirical_hist, predicted_dist):
    """
    Wasserstein-1 / Earth Mover's Distance between distributions.
    Uses class indices (0,1,...,K-1) as the ordinal ground metric.
    Lower is better. Ordinal-aware (unlike JSD, Hellinger, Bhattacharyya).
    """
    K = len(empirical_hist)
    p = np.asarray(empirical_hist, dtype=float)
    q = np.asarray(predicted_dist, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    values = np.arange(K, dtype=float)
    return float(wasserstein_distance(values, values, p, q))


def compute_epd_item(prob_dist):
    """
    Expected Pairwise Disagreement (EPD) = 1 - sum_k p_k^2
    Probability that two randomly drawn annotators give different labels.
    Range [0,1]. 0 = all agree, 1 = maximum disagreement.
    Apply to empirical_hist for actual EPD, to predicted_dist for predicted EPD.
    """
    p = np.asarray(prob_dist, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    return float(1.0 - np.sum(p ** 2))


def compute_hellinger_item(empirical_hist, predicted_dist):
    """
    Hellinger Distance between distributions.
    H = (1/sqrt(2)) * ||sqrt(P) - sqrt(Q)||_2. Range [0,1].
    Symmetric proper metric. Does NOT respect ordinal ordering.
    """
    K = len(empirical_hist)
    p = np.asarray(empirical_hist, dtype=float)
    q = np.asarray(predicted_dist, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    return float((1.0 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def compute_bhattacharyya_item(empirical_hist, predicted_dist):
    """
    Bhattacharyya Coefficient (BC) and Distance (BD).
    BC = sum_k sqrt(P_k * Q_k) ∈ [0,1]; 1=identical.
    BD = -log(BC) ∈ [0,∞); 0=identical.
    """
    K = len(empirical_hist)
    p = np.asarray(empirical_hist, dtype=float)
    q = np.asarray(predicted_dist, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(K) / K
    q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
    bc = float(np.sum(np.sqrt(p * q)))
    return bc, float(-np.log(max(bc, 1e-10)))


class VoicedErrorAnalyzer:
    """Comprehensive error analysis for VOICED DisCo model."""

    def __init__(self, data_dir, model_fname, split_name, empirical_fname, gpu_id=-1):
        self.data_dir = data_dir.rstrip("/") + "/"
        self.model_fname = model_fname
        self.split_name = split_name
        self.empirical_fname = empirical_fname
        self.gpu_id = gpu_id
        self.output_dir = os.path.join(self.data_dir, "error_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}
        self.item_level_results = []
        self.annotator_level_results = []
        self.is_disco_baseline = False
        self.rater_id_to_meta = {}

    def load_data(self):
        print("=" * 70)
        print("VOICED Error Analysis - Loading Data")
        print("=" * 70)

        self.Xi = np.load(f"{self.data_dir}Xi_{self.split_name}.npy", allow_pickle=True)
        self.Yi = np.load(f"{self.data_dir}Yi_{self.split_name}.npy", allow_pickle=True)
        self.Ya = np.load(f"{self.data_dir}Ya_{self.split_name}.npy", allow_pickle=True)
        self.Y = np.load(f"{self.data_dir}Y_{self.split_name}.npy", allow_pickle=True)

        # I/A may be stored as Ii/Ai or I/A depending on generator
        if os.path.exists(f"{self.data_dir}I_{self.split_name}.npy"):
            self.I = np.load(f"{self.data_dir}I_{self.split_name}.npy", allow_pickle=True)
        else:
            self.I = np.load(f"{self.data_dir}Ii_{self.split_name}.npy", allow_pickle=True)
        if os.path.exists(f"{self.data_dir}A_{self.split_name}.npy"):
            self.A = np.load(f"{self.data_dir}A_{self.split_name}.npy", allow_pickle=True)
        else:
            self.A = np.load(f"{self.data_dir}Ai_{self.split_name}.npy", allow_pickle=True)

        # AIL: item, annotator, label, message (message present in this repo)
        self.raw_data = pd.read_csv(self.empirical_fname, header=None)
        if self.raw_data.shape[1] >= 4:
            self.raw_data.columns = ["item", "annotator", "label", "message"]
        else:
            self.raw_data.columns = ["item", "annotator", "label"]
            self.raw_data["message"] = ""
        self.raw_data = self.raw_data.dropna(subset=["item", "annotator", "label"])

        # item_id -> first row index
        self.item_id_to_idx = {}
        for idx, row in self.raw_data.iterrows():
            iid = row["item"]
            if iid not in self.item_id_to_idx:
                self.item_id_to_idx[iid] = idx

        # Annotator metadata features are stored in A matrix after first column (id/index)
        meta_dim = (self.A.shape[1] - 1) if self.A.shape[1] > 1 else 0
        self.rater_id_to_meta = {}
        if meta_dim > 0:
            for i in range(min(len(self.raw_data), len(self.A))):
                rater_id = self.raw_data.iloc[i]["annotator"]
                meta = self.A[i, 1:].astype(np.float32)
                self.rater_id_to_meta[rater_id] = meta

        n_classes = int(self.Yi.shape[1])
        class_names = LABEL_NAMES[:n_classes] if len(LABEL_NAMES) >= n_classes else [str(i) for i in range(n_classes)]
        print(f"  Items: {len(self.item_id_to_idx)}, Annotations: {len(self.raw_data)}")
        print(f"  Label classes: {n_classes} ({', '.join(class_names)})")

    def load_model(self):
        gpu_tag = "/GPU:0" if self.gpu_id >= 0 else "/CPU:0"
        if self.gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        with tf.device(gpu_tag):
            self.model = load_object(self.model_fname)
            self.model.drop_p = 0.0
        name = type(self.model).__name__
        self.is_disco_baseline = name.lower() == "disco" and not hasattr(self.model, "W_demographics")
        print(f"Loaded model: {name} (disco_baseline={self.is_disco_baseline})")

    def _predict_yi_for_row(self, x_s, row_idx, aid=None):
        """
        DiADEM: encode(x, meta) -> (z, z_a), decode_yi(z, z_a)
        DisCo baseline: encode(x, a_idx) -> z, decode_yi(z)
        """
        if self.is_disco_baseline:
            if row_idx is not None and row_idx < len(self.A):
                ann_idx = int(np.asarray(self.A[row_idx]).reshape(-1)[0])
            else:
                ann_idx = 0
            # Cold-start / by_annotator: clamp OOV indices to UNK (last Wa row)
            a_dim = int(self.model.Wa.shape[0])
            if ann_idx < 0 or ann_idx >= a_dim:
                ann_idx = a_dim - 1
            a = tf.constant([[ann_idx]], dtype=tf.int32)
            z = self.model.encode(x_s, a)
            yi_prob, _ = self.model.decode_yi(z)
            return yi_prob

        meta_dim = (self.A.shape[1] - 1) if self.A.shape[1] > 1 else 0
        if aid is not None and self.rater_id_to_meta and aid in self.rater_id_to_meta:
            meta = tf.cast(np.expand_dims(self.rater_id_to_meta[aid], 0), tf.float32)
            a_idx_val = 0
        elif row_idx is not None and self.A.shape[1] > 1 and row_idx < len(self.A):
            meta = tf.cast(np.expand_dims(self.A[row_idx, 1:], 0), tf.float32)
            a_idx_val = int(self.A[row_idx, 0])
        else:
            meta = tf.zeros([1, max(meta_dim, 1)], tf.float32)
            a_idx_val = 0
        a_idx = None
        if getattr(self.model, "use_identity_embedding", False):
            n_ann = int(getattr(self.model, "n_annotators", 1))
            if a_idx_val < 0 or a_idx_val >= n_ann:
                a_idx_val = n_ann - 1
            a_idx = tf.constant([a_idx_val], dtype=tf.int32)
        z, z_a = self.model.encode(x_s, meta, a_idx=a_idx)
        yi_prob, _ = self.model.decode_yi(z, z_a)
        return yi_prob

    def calculate_actual_disagreement(self, annotator_labels):
        labels = np.array(annotator_labels)
        variance = np.var(labels)
        label_counts = Counter(labels)
        total = len(labels)
        probs = [c / total for c in label_counts.values()] if total else [1.0]
        entropy = float(-sum(p * np.log2(p + 1e-10) for p in probs))
        majority_label = label_counts.most_common(1)[0][0] if total else 0
        agreement_rate = sum(1 for l in labels if l == majority_label) / total if total else 0.0
        return {
            "variance": float(variance),
            "entropy": float(entropy),
            "agreement_rate": float(agreement_rate),
            "num_annotators": int(total),
        }

    def calculate_predicted_disagreement_from_probs(self, annotator_prob_lists, n_classes):
        if not annotator_prob_lists:
            return {
                "argmax_variance": 0.0,
                "argmax_entropy": 0.0,
                "dist_variance_mean": 0.0,
                "dist_entropy_mean": 0.0,
                "mean_predicted_dist": [1.0 / n_classes] * n_classes,
                "num_annotators": 0,
            }

        preds = [int(np.argmax(p)) for p in annotator_prob_lists]
        labels = np.array(preds)
        argmax_var = float(np.var(labels))
        lc = Counter(preds)
        total = len(preds)
        lp = [c / total for c in lc.values()] if total else [1.0]
        argmax_ent = float(-sum(p * np.log2(p + 1e-10) for p in lp))

        dist_vars = [distribution_variance(p) for p in annotator_prob_lists]
        dist_ents = [distribution_entropy(p) for p in annotator_prob_lists]
        mean_dist = np.mean(np.array(annotator_prob_lists), axis=0).tolist()

        return {
            "argmax_variance": argmax_var,
            "argmax_entropy": argmax_ent,
            "dist_variance_mean": float(np.mean(dist_vars)),
            "dist_entropy_mean": float(np.mean(dist_ents)),
            "mean_predicted_dist": mean_dist,
            "num_annotators": int(total),
        }

    def run_predictions(self):
        print("\n" + "=" * 70)
        print("Running Predictions")
        print("=" * 70)
        print(f"  Mode: {'DisCo baseline (annotator index)' if self.is_disco_baseline else 'DiADEM (demographics)'}")

        n_classes = int(self.Yi.shape[1])
        unique_items = self.raw_data["item"].unique()
        desc = "VOICED baseline error analysis" if self.is_disco_baseline else "VOICED error analysis"

        for item_id in tqdm(unique_items, desc=desc):
            if item_id not in self.item_id_to_idx:
                continue
            idx = int(self.item_id_to_idx[item_id])
            if idx >= len(self.Xi):
                continue

            item_data = self.raw_data[self.raw_data["item"] == item_id]
            annotator_ids = item_data["annotator"].tolist()
            actual_labels = []
            for l in item_data["label"].tolist():
                oh = parse_one_hot(l)
                actual_labels.append(int(np.argmax(oh)))

            msg = item_data["message"].iloc[0] if "message" in item_data.columns else ""
            # Use each annotation's own row when available (needed for baseline A[:,0])
            row_indices = item_data.index.tolist()
            x_s = tf.cast(np.expand_dims(self.Xi[idx, :], axis=0), tf.float32)

            ann_probs = []
            for i, aid in enumerate(annotator_ids):
                row_idx = row_indices[i] if i < len(row_indices) else idx
                if row_idx >= len(self.Xi):
                    continue
                # Prefer this row's Xi when aligned with AIL (baseline & diadem)
                if row_idx < len(self.Xi):
                    x_row = tf.cast(np.expand_dims(self.Xi[row_idx, :], axis=0), tf.float32)
                else:
                    x_row = x_s
                yi_prob = self._predict_yi_for_row(x_row, row_idx, aid=aid)
                ann_probs.append([float(x) for x in yi_prob.numpy()[0].tolist()])

            ann_preds = [int(np.argmax(p)) for p in ann_probs]

            hist = np.zeros(n_classes)
            for lbl in actual_labels:
                if 0 <= lbl < n_classes:
                    hist[lbl] += 1

            actual_disp = self.calculate_actual_disagreement(actual_labels)
            pred_disp = self.calculate_predicted_disagreement_from_probs(ann_probs, n_classes)
            jsd = compute_jsd_item(hist, pred_disp["mean_predicted_dist"])

            # ── NEW metrics ───────────────────────────────────────────────
            rps        = compute_rps_item(hist, pred_disp["mean_predicted_dist"])
            wass       = compute_wasserstein_item(hist, pred_disp["mean_predicted_dist"])
            hellinger  = compute_hellinger_item(hist, pred_disp["mean_predicted_dist"])
            bc, bd     = compute_bhattacharyya_item(hist, pred_disp["mean_predicted_dist"])
            actual_epd    = compute_epd_item(hist)
            predicted_epd = compute_epd_item(pred_disp["mean_predicted_dist"])
            # ─────────────────────────────────────────────────────────────

            self.item_level_results.append(
                {
                    "item_id": item_id,
                    "item_index": idx,
                    "num_annotators": len(annotator_ids),
                    "actual_disagreement": actual_disp,
                    "predicted_disagreement": pred_disp,
                    "jsd": jsd,
                    # NEW
                    "rps": rps,
                    "wasserstein": wass,
                    "hellinger": hellinger,
                    "bhattacharyya_coeff": bc,
                    "bhattacharyya_dist": bd,
                    "actual_epd": actual_epd,
                    "predicted_epd": predicted_epd,
                    # ─────────────────────────────────────────────────────
                    "message": str(msg) if pd.notna(msg) else "",
                    "actual_labels": actual_labels,
                    "predicted_labels": ann_preds,
                    "empirical_hist": hist.tolist(),
                    "mean_predicted_dist": pred_disp["mean_predicted_dist"],
                }
            )

            for i, aid in enumerate(annotator_ids[: len(ann_probs)]):
                self.annotator_level_results.append(
                    {
                        "item_id": item_id,
                        "annotator_id": aid,
                        "actual_label": actual_labels[i],
                        "predicted_label": ann_preds[i],
                        "correct": int(actual_labels[i] == ann_preds[i]),
                        "message": str(msg) if pd.notna(msg) else "",
                        "predicted_probs": ann_probs[i],
                        "item_actual_variance": actual_disp["variance"],
                        "item_predicted_variance_argmax": pred_disp["argmax_variance"],
                        "item_predicted_variance_dist": pred_disp["dist_variance_mean"],
                        "item_predicted_entropy_dist": pred_disp["dist_entropy_mean"],
                        "item_jsd": jsd,
                        # NEW
                        "item_rps": rps,
                        "item_wasserstein": wass,
                        "item_hellinger": hellinger,
                        "item_bhattacharyya_coeff": bc,
                        "item_bhattacharyya_dist": bd,
                        "item_actual_epd": actual_epd,
                        "item_predicted_epd": predicted_epd,
                    }
                )

        print(f"\nProcessed {len(self.item_level_results)} items, {len(self.annotator_level_results)} annotations")
        if self.annotator_level_results:
            acc = sum(r["correct"] for r in self.annotator_level_results) / len(self.annotator_level_results)
            print(f"  Annotator-level accuracy: {acc:.4f}")

    def calculate_metrics(self):
        print("\n" + "=" * 70)
        print("Calculating Metrics")
        print("=" * 70)

        n_classes = int(self.Yi.shape[1])
        class_names = LABEL_NAMES[:n_classes] if len(LABEL_NAMES) >= n_classes else [str(i) for i in range(n_classes)]

        actual = [r["actual_label"] for r in self.annotator_level_results]
        pred = [r["predicted_label"] for r in self.annotator_level_results]
        all_probs = np.array([r["predicted_probs"] for r in self.annotator_level_results])

        accuracy = accuracy_score(actual, pred)
        f1_macro = f1_score(actual, pred, average="macro", zero_division=0)
        f1_weighted = f1_score(actual, pred, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(actual, pred)
        mcc = matthews_corrcoef(actual, pred)
        cm = confusion_matrix(actual, pred, labels=list(range(n_classes)))
        report = classification_report(actual, pred, target_names=class_names, output_dict=True, zero_division=0)

        targets_hard = [r["actual_labels"] for r in self.item_level_results]
        predictions_hard = [r["predicted_labels"] for r in self.item_level_results]
        perspectivist_avg_er = error_rate(targets_hard, predictions_hard)

        targets_soft, predictions_soft = [], []
        for r in self.item_level_results:
            item_id = r["item_id"]
            ann_rows = [x for x in self.annotator_level_results if x["item_id"] == item_id]
            one_hots, probs = [], []
            for x in ann_rows:
                oh = np.zeros(n_classes)
                oh[x["actual_label"]] = 1.0
                one_hots.append(oh.tolist())
                p = list(x["predicted_probs"])
                p = (p + [0.0] * n_classes)[:n_classes]
                probs.append(p)
            targets_soft.append(one_hots)
            predictions_soft.append(probs)
        perspectivist_avg_md = multilabel_average_MD(targets_soft, predictions_soft)

        all_jsd = [r["jsd"] for r in self.item_level_results]
        mean_jsd = float(np.mean(all_jsd)) if all_jsd else 0.0

        # ── NEW: aggregate new metrics ────────────────────────────────────
        mean_rps       = float(np.mean([r["rps"]                for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_wass      = float(np.mean([r["wasserstein"]         for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_hellinger = float(np.mean([r["hellinger"]           for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_bc        = float(np.mean([r["bhattacharyya_coeff"] for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_bd        = float(np.mean([r["bhattacharyya_dist"]  for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_actual_epd    = float(np.mean([r["actual_epd"]    for r in self.item_level_results])) if self.item_level_results else 0.0
        mean_predicted_epd = float(np.mean([r["predicted_epd"] for r in self.item_level_results])) if self.item_level_results else 0.0
        # ─────────────────────────────────────────────────────────────────

        ece_results = compute_ece(all_probs, np.array(actual)) if len(actual) else {"ece": 0.0, "bin_confidences": [], "bin_accuracies": [], "bin_counts": []}

        self.results["metrics"] = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "kappa": float(kappa),
            "mcc": float(mcc),
            "perspectivist_average_er": float(perspectivist_avg_er),
            "perspectivist_average_md": float(perspectivist_avg_md),
            "mean_jsd": float(mean_jsd),
            "ece": float(ece_results["ece"]),
            # NEW
            "mean_rps": mean_rps,
            "mean_wasserstein": mean_wass,
            "mean_hellinger": mean_hellinger,
            "mean_bhattacharyya_coeff": mean_bc,
            "mean_bhattacharyya_dist": mean_bd,
            "mean_actual_epd": mean_actual_epd,
            "mean_predicted_epd": mean_predicted_epd,
            # ─────────────────────────────────────────────────────────────
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        print(f"\n{'─'*50}")
        print(f"  Accuracy:              {accuracy:.4f}")
        print(f"  F1 (macro):            {f1_macro:.4f}")
        print(f"  F1 (weighted):         {f1_weighted:.4f}")
        print(f"  Cohen's Kappa:         {kappa:.4f}")
        print(f"  MCC:                   {mcc:.4f}")
        print(f"{'─'*50}")
        print(f"  Perspectivist ER:      {perspectivist_avg_er:.5f}  (lower=better)")
        print(f"  Perspectivist MD:      {perspectivist_avg_md:.5f}  (lower=better)")
        print(f"  Mean JSD [0,1]:        {mean_jsd:.5f}  (lower=better)")
        print(f"  ECE:                   {ece_results['ece']:.5f}  (lower=better)")
        print(f"{'─'*50}")
        print(f"  ── NEW ordinal-aware ──")
        print(f"  Mean RPS [0,1]:        {mean_rps:.5f}  (lower=better) [ORDINAL]")
        print(f"  Mean Wasserstein-1:    {mean_wass:.5f}  (lower=better) [ORDINAL]")
        print(f"  ── NEW symmetric divergences ──")
        print(f"  Mean Hellinger [0,1]:  {mean_hellinger:.5f}  (lower=better)")
        print(f"  Mean Bhattacharyya BC: {mean_bc:.5f}  (higher=better)")
        print(f"  Mean Bhattacharyya BD: {mean_bd:.5f}  (lower=better)")
        print(f"  ── NEW disagreement probability ──")
        print(f"  Mean Actual EPD:       {mean_actual_epd:.5f}  (actual annotator disagreement prob)")
        print(f"  Mean Predicted EPD:    {mean_predicted_epd:.5f}  (model predicted disagreement prob)")
        print(f"{'─'*50}")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(include_values=True, cmap=plt.cm.Blues)
        plt.title("VOICED DisCo - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        self._plot_ece_diagram(ece_results)

    def _plot_ece_diagram(self, ece_results):
        if not ece_results.get("bin_confidences"):
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.linspace(0, 1, len(ece_results["bin_confidences"]) + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(
            bin_centers,
            ece_results["bin_accuracies"],
            width=0.08,
            alpha=0.7,
            label="Accuracy",
            color="steelblue",
            edgecolor="white",
        )
        ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect calibration")
        ax.set_xlabel("Confidence (max predicted probability)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram  (ECE = {ece_results['ece']:.4f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ece_reliability_diagram.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def analyze_disagreement_correlation(self):
        print("\n" + "=" * 70)
        print("Analyzing Disagreement Correlation")
        print("=" * 70)

        act_var = [r["actual_disagreement"]["variance"] for r in self.item_level_results]
        act_ent = [r["actual_disagreement"]["entropy"] for r in self.item_level_results]
        pred_var_dist = [r["predicted_disagreement"]["dist_variance_mean"] for r in self.item_level_results]
        pred_ent_dist = [r["predicted_disagreement"]["dist_entropy_mean"] for r in self.item_level_results]

        def corr_block(act, pred, label):
            pr, pp = pearsonr(act, pred)
            sr, sp = spearmanr(act, pred)
            print(f"  [{label}] Pearson r={pr:.4f} (p={pp:.2e}) | Spearman ρ={sr:.4f} (p={sp:.2e})")
            return {"pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(sp)}

        var_dist = corr_block(act_var, pred_var_dist, "Variance (dist)")
        ent_dist = corr_block(act_ent, pred_ent_dist, "Entropy  (dist)")

        # ── NEW: EPD correlation ─────────────────────────────────────────
        act_epd  = np.array([r["actual_epd"]    for r in self.item_level_results])
        pred_epd = np.array([r["predicted_epd"] for r in self.item_level_results])
        print("\n── EPD — Expected Pairwise Disagreement (NEW) ──")
        epd_corr = corr_block(act_epd.tolist(), pred_epd.tolist(), "EPD")
        # ─────────────────────────────────────────────────────────────────

        self.results["disagreement_correlation"] = {
            "distribution_level": {"variance": var_dist, "entropy": ent_dist},
            "epd": epd_corr,  # NEW
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        pairs = [
            (act_var, pred_var_dist, "Variance (dist-level)", var_dist),
            (act_ent, pred_ent_dist, "Entropy (dist-level)", ent_dist),
            (act_epd.tolist(), pred_epd.tolist(), "EPD (NEW)", epd_corr),
        ]
        for ax, (x, y, title, stats) in zip(axes.flat, pairs):
            ax.scatter(x, y, alpha=0.4, s=15)
            mn, mx = min(x + y), max(x + y)
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
            ax.set_xlabel(f"Actual {title.split()[0]}")
            ax.set_ylabel(f"Predicted {title.split()[0]}")
            ax.set_title(f"{title}\nr={stats['pearson_r']:.3f}, ρ={stats['spearman_r']:.3f}")
            ax.grid(True, alpha=0.3)
        axes.flat[-1].set_visible(False)  # hide unused 4th subplot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/disagreement_correlation.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_disagreement_calibration_curve(self, n_bins=10):
        print("\n" + "=" * 70)
        print("Plotting Disagreement Calibration Curve")
        print("=" * 70)

        act_var = np.array([r["actual_disagreement"]["variance"] for r in self.item_level_results])
        pred_var = np.array([r["predicted_disagreement"]["dist_variance_mean"] for r in self.item_level_results])
        act_ent = np.array([r["actual_disagreement"]["entropy"] for r in self.item_level_results])
        pred_ent = np.array([r["predicted_disagreement"]["dist_entropy_mean"] for r in self.item_level_results])

        def calibration_bins(actual, predicted, n_bins):
            quantile_edges = np.quantile(actual, np.linspace(0, 1, n_bins + 1))
            bin_actual, bin_pred, bin_std, bin_count = [], [], [], []
            for i in range(n_bins):
                mask = (actual >= quantile_edges[i]) & (actual <= quantile_edges[i + 1])
                if mask.sum() > 0:
                    bin_actual.append(float(actual[mask].mean()))
                    bin_pred.append(float(predicted[mask].mean()))
                    bin_std.append(float(predicted[mask].std()))
                    bin_count.append(int(mask.sum()))
            return np.array(bin_actual), np.array(bin_pred), np.array(bin_std), np.array(bin_count)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (act, pred, label) in zip(axes, [(act_var, pred_var, "Variance"), (act_ent, pred_ent, "Entropy")]):
            ba, bp, bs, _ = calibration_bins(act, pred, n_bins)
            ax.fill_between(ba, bp - bs, bp + bs, alpha=0.2, color="steelblue", label="±1 std")
            ax.plot(ba, bp, "o-", color="steelblue", lw=2, ms=6, label="Model prediction")
            mn, mx = min(ba.min(), bp.min()), max(ba.max(), bp.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect calibration")
            ax.set_xlabel(f"Actual {label} (binned)")
            ax.set_ylabel(f"Mean Predicted {label}")
            ax.set_title(f"Disagreement Calibration — {label}")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Disagreement Calibration Curve", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/disagreement_calibration_curve.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        ba_v, bp_v, _, _ = calibration_bins(act_var, pred_var, n_bins)
        ba_e, bp_e, _, _ = calibration_bins(act_ent, pred_ent, n_bins)
        self.results["disagreement_calibration"] = {
            "variance": {"actual_bins": ba_v.tolist(), "predicted_bins": bp_v.tolist()},
            "entropy": {"actual_bins": ba_e.tolist(), "predicted_bins": bp_e.tolist()},
        }

    def analyze_mean_response_distributions(self):
        print("\n" + "=" * 70)
        print("Analyzing Mean Response Distributions")
        print("=" * 70)

        label_groups = defaultdict(list)
        for r in self.annotator_level_results:
            label_groups[r["actual_label"]].append(r)

        n_classes = int(self.Yi.shape[1])
        class_names = LABEL_NAMES[:n_classes] if len(LABEL_NAMES) >= n_classes else [str(i) for i in range(n_classes)]
        disco_dists, annotator_dists = {}, {}

        for true_label in sorted(label_groups.keys()):
            rows = label_groups[true_label]
            disco_probs = np.array([r["predicted_probs"] for r in rows])
            disco_mean = np.mean(disco_probs, axis=0)
            disco_std = np.std(disco_probs, axis=0)
            actual_onehot = np.zeros((len(rows), n_classes))
            for i, r in enumerate(rows):
                actual_onehot[i, r["actual_label"]] = 1.0
            ann_mean = np.mean(actual_onehot, axis=0)
            ann_std = np.std(actual_onehot, axis=0)
            disco_dists[true_label] = {"mean": disco_mean.tolist(), "std": disco_std.tolist(), "count": len(rows)}
            annotator_dists[true_label] = {"mean": ann_mean.tolist(), "std": ann_std.tolist(), "count": len(rows)}
            print(
                f"  True={class_names[true_label]} (n={len(rows)}): "
                f"pred_mean={np.round(disco_mean,3)}, actual_mean={np.round(ann_mean,3)}"
            )

        self.results["mean_response_distributions"] = {"disco": disco_dists, "annotator": annotator_dists}
        num_labels = len(disco_dists)
        fig, axes = plt.subplots(1, max(1, num_labels), figsize=(6 * max(1, num_labels), 5))
        if num_labels == 1:
            axes = [axes]
        for idx, (true_label, dd) in enumerate(sorted(disco_dists.items())):
            ad = annotator_dists[true_label]
            x = np.arange(n_classes)
            w = 0.35
            axes[idx].bar(x - w / 2, dd["mean"], w, label="DisCo Predicted", alpha=0.8, yerr=dd["std"], capsize=5)
            axes[idx].bar(x + w / 2, ad["mean"], w, label="Actual Annotators", alpha=0.8, yerr=ad["std"], capsize=5)
            axes[idx].set_xlabel("Class")
            axes[idx].set_ylabel("Mean Probability")
            axes[idx].set_title(f"True: {class_names[true_label]} (n={dd['count']})")
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(class_names)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/mean_response_distributions.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def save_error_analysis_csv(self):
        if not self.annotator_level_results:
            return
        n_classes = int(self.Yi.shape[1])
        rows = []
        for r in self.annotator_level_results:
            probs = (list(r["predicted_probs"]) + [0.0] * n_classes)[:n_classes]
            row = {
                "item_id": r["item_id"],
                "annotator_id": r["annotator_id"],
                "actual_label": r["actual_label"],
                "predicted_label": r["predicted_label"],
                "correct": r["correct"],
                "message": r["message"],
                "item_actual_variance": r["item_actual_variance"],
                "item_predicted_variance_argmax": r["item_predicted_variance_argmax"],
                "item_predicted_variance_dist": r["item_predicted_variance_dist"],
                "item_predicted_entropy_dist": r["item_predicted_entropy_dist"],
                "item_jsd": r["item_jsd"],
                # NEW
                "item_rps": r.get("item_rps", float("nan")),
                "item_wasserstein": r.get("item_wasserstein", float("nan")),
                "item_hellinger": r.get("item_hellinger", float("nan")),
                "item_bhattacharyya_coeff": r.get("item_bhattacharyya_coeff", float("nan")),
                "item_bhattacharyya_dist": r.get("item_bhattacharyya_dist", float("nan")),
                "item_actual_epd": r.get("item_actual_epd", float("nan")),
                "item_predicted_epd": r.get("item_predicted_epd", float("nan")),
            }
            for i in range(n_classes):
                row[f"predicted_prob_class_{i}"] = float(probs[i])
                row[f"actual_label_class_{i}"] = 1.0 if i == r["actual_label"] else 0.0
            rows.append(row)
        df = pd.DataFrame(rows).drop_duplicates()
        df = df.sort_values(["correct", "item_id"], ascending=[True, True])
        df.to_csv(f"{self.output_dir}/error_analysis_detailed.csv", index=False)
        print(f"\nSaved {len(df)} rows → {self.output_dir}/error_analysis_detailed.csv")
        print(f"  Correct: {df['correct'].sum()} | Accuracy: {df['correct'].mean():.4f}")

    def print_summary(self):
        m  = self.results.get("metrics", {})
        dc = self.results.get("disagreement_correlation", {})
        dl = dc.get("distribution_level", {})
        ep = dc.get("epd", {})

        def _f(v, fmt=".4f"):
            return format(v, fmt) if isinstance(v, float) else str(v)

        print("\n" + "=" * 70)
        print("SUMMARY — copy-paste ready for paper table")
        print("=" * 70)
        print(f"  ─── Standard ───")
        print(f"  Accuracy:              {_f(m.get('accuracy', 0.0))}")
        print(f"  F1 (macro):            {_f(m.get('f1_macro', 0.0))}")
        print(f"  F1 (weighted):         {_f(m.get('f1_weighted', 0.0))}")
        print(f"  Cohen's Kappa:         {_f(m.get('kappa', 0.0))}")
        print(f"  MCC:                   {_f(m.get('mcc', 0.0))}")
        print(f"  ─── Soft label ───")
        print(f"  Mean JSD [0,1]:        {_f(m.get('mean_jsd', 0.0), '.5f')}  ← LeWiDi2025 primary")
        print(f"  Perspectivist ER:      {_f(m.get('perspectivist_average_er', 0.0), '.5f')}")
        print(f"  Perspectivist MD:      {_f(m.get('perspectivist_average_md', 0.0), '.5f')}")
        print(f"  ECE:                   {_f(m.get('ece', 0.0), '.5f')}")
        print(f"  ─── NEW — Ordinal-aware ───")
        print(f"  Mean RPS [0,1]:        {_f(m.get('mean_rps', 0.0), '.5f')}  (lower=better)")
        print(f"  Mean Wasserstein-1:    {_f(m.get('mean_wasserstein', 0.0), '.5f')}  (lower=better)")
        print(f"  ─── NEW — Symmetric divergences ───")
        print(f"  Mean Hellinger [0,1]:  {_f(m.get('mean_hellinger', 0.0), '.5f')}  (lower=better)")
        print(f"  Mean Bhattacharyya BC: {_f(m.get('mean_bhattacharyya_coeff', 0.0), '.5f')}  (higher=better)")
        print(f"  Mean Bhattacharyya BD: {_f(m.get('mean_bhattacharyya_dist', 0.0), '.5f')}  (lower=better)")
        print(f"  ─── NEW — Disagreement probability (EPD) ───")
        print(f"  Mean Actual EPD:       {_f(m.get('mean_actual_epd', 0.0), '.5f')}")
        print(f"  Mean Predicted EPD:    {_f(m.get('mean_predicted_epd', 0.0), '.5f')}")
        print(f"  ─── Disagreement tracking ───")
        if dl:
            vd = dl.get("variance", {}); ed = dl.get("entropy", {})
            print(f"  Variance Pearson r:    {_f(vd.get('pearson_r', 0.0))}  (p={_f(vd.get('pearson_p', 1.0), '.2e')})")
            print(f"  Variance Spearman ρ:   {_f(vd.get('spearman_r', 0.0))}  (p={_f(vd.get('spearman_p', 1.0), '.2e')})")
            print(f"  Entropy  Pearson r:    {_f(ed.get('pearson_r', 0.0))}  (p={_f(ed.get('pearson_p', 1.0), '.2e')})")
            print(f"  Entropy  Spearman ρ:   {_f(ed.get('spearman_r', 0.0))}  (p={_f(ed.get('spearman_p', 1.0), '.2e')})")
        if ep:
            print(f"  EPD      Pearson r:    {_f(ep.get('pearson_r', 0.0))}  (p={_f(ep.get('pearson_p', 1.0), '.2e')})  ← NEW")
            print(f"  EPD      Spearman ρ:   {_f(ep.get('spearman_r', 0.0))}  (p={_f(ep.get('spearman_p', 1.0), '.2e')})  ← NEW")
        print("=" * 70)

    def run(self):
        print("=" * 70)
        print("VOICED Error Analysis — DICES-style detailed metrics")
        print("=" * 70)
        self.load_data()
        self.load_model()
        self.run_predictions()
        self.calculate_metrics()
        self.analyze_disagreement_correlation()
        self.plot_disagreement_calibration_curve()
        self.analyze_mean_response_distributions()
        self.save_error_analysis_csv()
        self.print_summary()
        with open(f"{self.output_dir}/analysis_summary.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nAll outputs saved to: {self.output_dir}/")


def run_error_detailed(data_dir, model_fname, empirical_fname, split_name="test", gpu_id=0):
    """Pipeline entry point matching voices/error_detailed.py."""
    analyzer = VoicedErrorAnalyzer(
        str(data_dir), str(model_fname), split_name, str(empirical_fname), gpu_id
    )
    analyzer.run()
    return Path(data_dir) / "error_analysis" / "analysis_summary.json"


def main():
    ap = argparse.ArgumentParser(description="VOICED DisCo error analysis (DICES-style)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_fname", required=True)
    ap.add_argument("--split_name", default="test")
    ap.add_argument("--empirical_fname", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    analyzer = VoicedErrorAnalyzer(
        args.data_dir, args.model_fname, args.split_name, args.empirical_fname, args.gpu_id
    )
    analyzer.run()


if __name__ == "__main__":
    main()

