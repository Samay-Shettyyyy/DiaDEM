"""
In-depth perspectivist error analysis for VOICED with confidence intervals.

Adds to the existing detailed analysis:
  - robust row-level alignment checks
  - proper probabilistic scores (NLL, Brier)
  - bootstrap 95% CIs for key metrics
  - disagreement tracking with bootstrap CIs

Usage:
  python error_detailed_in_depth.py \
    --data_dir ./experimental_data/voiced_by_item/ \
    --model_fname ./experimental_data/voiced_by_item/trained_model.disco \
    --split_name test \
    --empirical_fname ./experimental_data/voiced_by_item/voiced_test_AIL.csv \
    --gpu_id 0
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef
from tqdm import tqdm

from diadem.utils.utils import load_object


EPS = 1e-12


def parse_one_hot(s):
    s = str(s).strip().replace("[", "").replace("]", "")
    return np.array([int(x) for x in s.split()])


def normalize_prob(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    z = p.sum()
    if z <= 0:
        return np.ones_like(p) / len(p)
    return p / z


def jsd_01(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    d = jensenshannon(p, q)
    return float(np.clip((d ** 2) / np.log(2), 0.0, 1.0))


def md_l1(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(np.sum(np.abs(p - q)))


def md_perspectivist_one_item(labels, probs, n_classes):
    """
    Perspectivist MD for one item:
    average L1 distance between each annotator one-hot target and that
    annotator's predicted probability distribution.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0 or len(probs) == 0:
        return 0.0

    mds = []
    n = min(len(labels), len(probs))
    for i in range(n):
        y = int(labels[i])
        p = normalize_prob(probs[i])
        oh = np.zeros(n_classes, dtype=float)
        if 0 <= y < n_classes:
            oh[y] = 1.0
        mds.append(float(np.sum(np.abs(oh - p))))
    return float(np.mean(mds)) if mds else 0.0


def error_rate_binary_or_multiclass(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true != y_pred))


def ece_multiclass(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    corr = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    out = {"bin_confidences": [], "bin_accuracies": [], "bin_counts": [], "ece": 0.0}
    n = len(labels)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        c = int(mask.sum())
        out["bin_counts"].append(c)
        if c == 0:
            out["bin_confidences"].append(0.0)
            out["bin_accuracies"].append(0.0)
            continue
        bconf = float(conf[mask].mean())
        bacc = float(corr[mask].mean())
        out["bin_confidences"].append(bconf)
        out["bin_accuracies"].append(bacc)
        ece += (c / n) * abs(bacc - bconf)
    out["ece"] = float(ece)
    return out


def nll_multiclass(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    idx = np.arange(len(labels))
    p_true = np.clip(probs[idx, labels], EPS, 1.0)
    return float(np.mean(-np.log(p_true)))


def brier_multiclass(probs, labels, n_classes):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    y = np.zeros((len(labels), n_classes), dtype=float)
    y[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def distribution_entropy(prob_dist):
    p = normalize_prob(prob_dist)
    return float(-np.sum(p * np.log2(p + EPS)))


def distribution_variance(prob_dist):
    p = normalize_prob(prob_dist)
    x = np.arange(len(p))
    m = np.dot(x, p)
    return float(max(np.dot(x ** 2, p) - m ** 2, 0.0))


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=69):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"low": None, "high": None, "mean": None}
    rng = np.random.default_rng(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(np.mean(vals[idx]))
    low = float(np.percentile(means, 100 * (alpha / 2)))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"low": low, "high": high, "mean": float(np.mean(vals))}


def bootstrap_correlation_ci(x, y, corr_type="spearman", n_boot=1000, alpha=0.05, seed=69):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"r": None, "p": None, "ci_low": None, "ci_high": None}
    if corr_type == "pearson":
        r, p = pearsonr(x, y)
        corr_fn = pearsonr
    else:
        r, p = spearmanr(x, y)
        corr_fn = spearmanr
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            br, _ = corr_fn(x[idx], y[idx])
            if np.isfinite(br):
                boots.append(float(br))
        except Exception:
            continue
    if not boots:
        return {"r": float(r), "p": float(p), "ci_low": None, "ci_high": None}
    low = float(np.percentile(boots, 100 * (alpha / 2)))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"r": float(r), "p": float(p), "ci_low": low, "ci_high": high}


class InDepthAnalyzer:
    def __init__(self, data_dir, model_fname, split_name, empirical_fname, gpu_id=0, n_boot=1000):
        if data_dir is None:
            raise ValueError("data_dir is required and cannot be None.")
        data_dir = str(data_dir).strip()
        if not data_dir:
            raise ValueError("data_dir is empty. Pass a valid experiment directory path.")
        resolved_data_dir = os.path.abspath(data_dir)
        if resolved_data_dir == "/":
            raise ValueError(
                "Resolved data_dir is '/'. This usually means an unset shell variable. "
                "Pass an explicit experiment directory, e.g. voices/experimental_data/voiced_by_item/."
            )
        self.data_dir = resolved_data_dir.rstrip("/") + "/"
        self.model_fname = model_fname
        self.split_name = split_name
        self.empirical_fname = empirical_fname
        self.gpu_id = gpu_id
        self.n_boot = n_boot
        self.output_dir = os.path.join(self.data_dir, "error_analysis_in_depth")
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        self.Xi = np.load(f"{self.data_dir}Xi_{self.split_name}.npy", allow_pickle=True)
        self.Y = np.load(f"{self.data_dir}Y_{self.split_name}.npy", allow_pickle=True)
        self.A = np.load(f"{self.data_dir}A_{self.split_name}.npy", allow_pickle=True)
        self.raw = pd.read_csv(self.empirical_fname, header=None)
        if self.raw.shape[1] >= 4:
            self.raw.columns = ["item", "annotator", "label", "message"]
        else:
            self.raw.columns = ["item", "annotator", "label"]
            self.raw["message"] = ""
        self.raw = self.raw.dropna(subset=["item", "annotator", "label"]).reset_index(drop=True)
        n = min(len(self.raw), len(self.Xi), len(self.A), len(self.Y))
        if n < len(self.raw) or n < len(self.Xi):
            print(
                f"[WARN] Length mismatch: raw={len(self.raw)} Xi={len(self.Xi)} A={len(self.A)} Y={len(self.Y)}. "
                f"Truncating to n={n} for safe row-level alignment."
            )
        self.raw = self.raw.iloc[:n].copy()
        self.Xi = self.Xi[:n]
        self.A = self.A[:n]
        self.Y = self.Y[:n]
        self.n_classes = int(self.Y.shape[1])

    def load_model(self):
        if self.gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            dev = "/GPU:0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            dev = "/CPU:0"
        with tf.device(dev):
            self.model = load_object(self.model_fname)
            self.model.drop_p = 0.0

    def _annotator_input_for_model(self, row_idx):
        """
        Returns (a_input, a_idx_or_None).
        - DisCo baseline: (index tensor, None) — encode(x, a)
        - DiADEM: (meta tensor, a_idx tensor|None) — encode(x, meta, a_idx=...)
        """
        model_name = type(self.model).__name__.lower()
        is_disco_baseline = model_name == "disco" and not hasattr(self.model, "W_demographics")

        if is_disco_baseline:
            if self.A.shape[1] >= 1:
                ann_idx = int(self.A[row_idx, 0])
            else:
                ann_idx = 0
            a_dim = int(self.model.Wa.shape[0])
            if ann_idx < 0 or ann_idx >= a_dim:
                ann_idx = a_dim - 1  # UNK
            return tf.cast(np.array([[ann_idx]], dtype=np.int32), tf.int32), None

        if self.A.shape[1] > 1:
            meta = tf.cast(np.expand_dims(self.A[row_idx, 1:], 0), tf.float32)
            a_idx = None
            if getattr(self.model, "use_identity_embedding", False):
                ann_idx = int(self.A[row_idx, 0])
                n_ann = int(getattr(self.model, "n_annotators", 1))
                if ann_idx < 0 or ann_idx >= n_ann:
                    ann_idx = n_ann - 1
                a_idx = tf.constant([ann_idx], dtype=tf.int32)
            return meta, a_idx
        return tf.cast(np.zeros((1, 0), dtype=np.float32), tf.float32), None

    def predict(self):
        rows = []
        for i in tqdm(range(len(self.raw)), desc="Predict rows"):
            x = tf.cast(np.expand_dims(self.Xi[i], 0), tf.float32)
            a, a_idx = self._annotator_input_for_model(i)

            # Support both model signatures:
            # - DiADEM-style: encode(x, a, a_idx=?) -> (z, z_a), decode_yi(z, z_a)
            # - DisCo-style:  encode(x, a) -> z,        decode_yi(z)
            if a_idx is not None:
                z_out = self.model.encode(x, a, a_idx=a_idx)
            else:
                z_out = self.model.encode(x, a)
            if isinstance(z_out, tuple):
                z, z_a = z_out
                yi_prob, _ = self.model.decode_yi(z, z_a)
            else:
                z = z_out
                yi_prob, _ = self.model.decode_yi(z)
            p = normalize_prob(yi_prob.numpy()[0])
            y_true = int(np.argmax(parse_one_hot(self.raw.iloc[i]["label"])))
            y_pred = int(np.argmax(p))
            rows.append(
                {
                    "row_idx": i,
                    "item_id": self.raw.iloc[i]["item"],
                    "annotator_id": self.raw.iloc[i]["annotator"],
                    "actual_label": y_true,
                    "predicted_label": y_pred,
                    "predicted_probs": p.tolist(),
                    "message": self.raw.iloc[i]["message"],
                }
            )
        self.row_results = pd.DataFrame(rows)

    def aggregate_items(self):
        item_results = []
        grouped = self.row_results.groupby("item_id", sort=False)
        for item_id, g in grouped:
            labels = g["actual_label"].to_numpy(dtype=int)
            probs = np.vstack(g["predicted_probs"].to_numpy())
            pred_labels = g["predicted_label"].to_numpy(dtype=int)
            hist = np.bincount(labels, minlength=self.n_classes).astype(float)
            p_emp = normalize_prob(hist)
            p_pred = normalize_prob(np.mean(probs, axis=0))
            item_results.append(
                {
                    "item_id": item_id,
                    "n_annotators": int(len(g)),
                    "actual_disagreement_entropy": distribution_entropy(p_emp),
                    "actual_disagreement_variance": distribution_variance(p_emp),
                    "predicted_disagreement_entropy": float(np.mean([distribution_entropy(pp) for pp in probs])),
                    "predicted_disagreement_variance": float(np.mean([distribution_variance(pp) for pp in probs])),
                    "item_jsd": jsd_01(p_emp, p_pred),
                    "item_md": md_l1(p_emp, p_pred),
                    "item_er": error_rate_binary_or_multiclass(labels, pred_labels),
                }
            )
        self.item_results = pd.DataFrame(item_results)

    def compute_metrics(self):
        y_true = self.row_results["actual_label"].to_numpy(dtype=int)
        y_pred = self.row_results["predicted_label"].to_numpy(dtype=int)
        probs = np.vstack(self.row_results["predicted_probs"].to_numpy())

        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

        metrics["nll"] = nll_multiclass(probs, y_true)
        metrics["brier"] = brier_multiclass(probs, y_true, self.n_classes)
        metrics["ece"] = ece_multiclass(probs, y_true, n_bins=10)["ece"]

        metrics["mean_jsd"] = float(self.item_results["item_jsd"].mean())
        metrics["perspectivist_md"] = float(self.item_results["item_md"].mean())
        metrics["perspectivist_er"] = float(self.item_results["item_er"].mean())

        act_var = self.item_results["actual_disagreement_variance"].to_numpy(dtype=float)
        pred_var = self.item_results["predicted_disagreement_variance"].to_numpy(dtype=float)
        act_ent = self.item_results["actual_disagreement_entropy"].to_numpy(dtype=float)
        pred_ent = self.item_results["predicted_disagreement_entropy"].to_numpy(dtype=float)

        metrics["disagreement_corr_variance_spearman"] = bootstrap_correlation_ci(
            act_var, pred_var, corr_type="spearman", n_boot=self.n_boot
        )
        metrics["disagreement_corr_variance_pearson"] = bootstrap_correlation_ci(
            act_var, pred_var, corr_type="pearson", n_boot=self.n_boot
        )
        metrics["disagreement_corr_entropy_spearman"] = bootstrap_correlation_ci(
            act_ent, pred_ent, corr_type="spearman", n_boot=self.n_boot
        )
        metrics["disagreement_corr_entropy_pearson"] = bootstrap_correlation_ci(
            act_ent, pred_ent, corr_type="pearson", n_boot=self.n_boot
        )

        per_item = {
            "mean_jsd": self.item_results["item_jsd"].to_numpy(),
            "perspectivist_md": self.item_results["item_md"].to_numpy(),
            "perspectivist_er": self.item_results["item_er"].to_numpy(),
        }
        per_row = {
            "nll_row": -np.log(np.clip(probs[np.arange(len(y_true)), y_true], EPS, 1.0)),
            "brier_row": np.sum((probs - np.eye(self.n_classes)[y_true]) ** 2, axis=1),
            "correct_row": (y_true == y_pred).astype(float),
        }

        ci = {}
        for k, arr in per_item.items():
            ci[k] = bootstrap_ci(arr, n_boot=self.n_boot)
        for k, arr in per_row.items():
            ci[k] = bootstrap_ci(arr, n_boot=self.n_boot)

        self.results = {
            "n_rows": int(len(self.row_results)),
            "n_items": int(len(self.item_results)),
            "n_classes": int(self.n_classes),
            "metrics": metrics,
            "bootstrap_ci_95": ci,
            "notes": {
                "alignment": "Row-level alignment between AIL CSV and Xi/A/Y arrays after safe truncation to min length.",
                "soft_scores": "NLL/Brier use annotator-level predicted probabilities against one-hot ground truth.",
                "perspectivist_scores": "JSD/MD are computed at item level using empirical vs mean predicted distributions; ER is computed from per-annotator hard predictions.",
            },
        }

    def save(self):
        self.row_results.to_csv(os.path.join(self.output_dir, "annotator_level_predictions.csv"), index=False)
        self.item_results.to_csv(os.path.join(self.output_dir, "item_level_metrics.csv"), index=False)
        with open(os.path.join(self.output_dir, "analysis_summary_in_depth.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nSaved in-depth outputs to: {self.output_dir}")

    def print_summary(self):
        m = self.results["metrics"]
        ci = self.results["bootstrap_ci_95"]
        print("\n" + "=" * 72)
        print("IN-DEPTH SUMMARY (with 95% bootstrap CI)")
        print("=" * 72)
        print(f"Rows={self.results['n_rows']}  Items={self.results['n_items']}  Classes={self.results['n_classes']}")
        print(f"Accuracy          : {m['accuracy']:.4f}  [CI {ci['correct_row']['low']:.4f}, {ci['correct_row']['high']:.4f}]")
        print(f"F1 macro          : {m['f1_macro']:.4f}")
        print(f"F1 weighted       : {m['f1_weighted']:.4f}")
        print(f"Kappa             : {m['kappa']:.4f}")
        print(f"MCC               : {m['mcc']:.4f}")
        print("-" * 72)
        print(f"Mean JSD [0,1]    : {m['mean_jsd']:.5f}  [CI {ci['mean_jsd']['low']:.5f}, {ci['mean_jsd']['high']:.5f}]")
        print(f"Perspectivist MD  : {m['perspectivist_md']:.5f}  [CI {ci['perspectivist_md']['low']:.5f}, {ci['perspectivist_md']['high']:.5f}]")
        print(f"Perspectivist ER  : {m['perspectivist_er']:.5f}  [CI {ci['perspectivist_er']['low']:.5f}, {ci['perspectivist_er']['high']:.5f}]")
        print(f"ECE               : {m['ece']:.5f}")
        print(f"NLL               : {m['nll']:.5f}  [CI {ci['nll_row']['low']:.5f}, {ci['nll_row']['high']:.5f}]")
        print(f"Brier             : {m['brier']:.5f}  [CI {ci['brier_row']['low']:.5f}, {ci['brier_row']['high']:.5f}]")
        print("-" * 72)
        vsp = m["disagreement_corr_variance_spearman"]
        esp = m["disagreement_corr_entropy_spearman"]
        print(
            f"Disagreement Spearman (variance): r={vsp['r']:.4f}, p={vsp['p']:.2e}, "
            f"CI=[{vsp['ci_low']:.4f}, {vsp['ci_high']:.4f}]"
        )
        print(
            f"Disagreement Spearman (entropy) : r={esp['r']:.4f}, p={esp['p']:.2e}, "
            f"CI=[{esp['ci_low']:.4f}, {esp['ci_high']:.4f}]"
        )
        print("=" * 72)

    def run(self):
        self.load()
        self.load_model()
        self.predict()
        self.aggregate_items()
        self.compute_metrics()
        self.save()
        self.print_summary()


def run_error_in_depth(data_dir, model_fname, empirical_fname, split_name="test", gpu_id=0, n_boot=1000):
    """Pipeline entry point matching voices/error_detailed_in_depth.py."""
    analyzer = InDepthAnalyzer(
        data_dir=str(data_dir),
        model_fname=str(model_fname),
        split_name=split_name,
        empirical_fname=str(empirical_fname),
        gpu_id=gpu_id,
        n_boot=n_boot,
    )
    analyzer.run()
    return Path(data_dir) / "error_analysis_in_depth" / "analysis_summary_in_depth.json"


def main():
    ap = argparse.ArgumentParser(description="In-depth VOICED error analysis with CI")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_fname", required=True)
    ap.add_argument("--split_name", default="test")
    ap.add_argument("--empirical_fname", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    analyzer = InDepthAnalyzer(
        data_dir=args.data_dir,
        model_fname=args.model_fname,
        split_name=args.split_name,
        empirical_fname=args.empirical_fname,
        gpu_id=args.gpu_id,
        n_boot=args.n_boot,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
