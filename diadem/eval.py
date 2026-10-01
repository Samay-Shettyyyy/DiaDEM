"""Item-level ensemble eval — aligned with voices/eval_voiced.py."""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from diadem.data_io import ail_path_for
from diadem.registry import load_dataset
from diadem.utils.utils import load_object

matplotlib.use("Agg")


def _convert_to_majority_array(labels):
    out = []
    for lab in labels:
        lab = np.array(lab)
        z = np.zeros(len(lab))
        z[np.argmax(lab)] = 1
        out.append(z)
    return out


def _convert_to_majority_index(labels):
    return [int(np.argmax(np.array(lab))) for lab in labels]


def _kl_divergence(p, q):
    eps = 1e-5
    p = np.array(p) + eps
    q = np.array(q) + eps
    return float(np.sum(p * np.log(p / q)))


def _kl_empirical2pred(empirical_pcts, prediction_proba):
    kl_sum = []
    for emp, pred in zip(empirical_pcts, prediction_proba):
        kl = _kl_divergence(emp, np.asarray(pred))
        kl_sum.append(0.0 if math.isnan(kl) else kl)
    return float(np.mean(kl_sum))


def _measure_accuracy(y_test, y_pred):
    y_test_arr = _convert_to_majority_array(y_test)
    y_pred_arr = _convert_to_majority_array(y_pred)
    matched = sum(accuracy_score(t, p) for t, p in zip(y_test_arr, y_pred_arr))
    return float(matched / len(y_test)) if y_test else 0.0


def _measure_f1(y_test, y_pred):
    y_test_idx = _convert_to_majority_index(y_test)
    y_pred_idx = _convert_to_majority_index(y_pred)
    report = classification_report(y_test_idx, y_pred_idx, digits=3, output_dict=True, zero_division=0)
    return (
        f1_score(y_test_idx, y_pred_idx, average="macro", zero_division=0),
        f1_score(y_test_idx, y_pred_idx, average="micro", zero_division=0),
        f1_score(y_test_idx, y_pred_idx, average="weighted", zero_division=0),
        {"macro": report["macro avg"]["precision"], "weighted": report["weighted avg"]["precision"]},
        {"macro": report["macro avg"]["recall"], "weighted": report["weighted avg"]["recall"]},
    )


def _confusion_matrix_plot(labels_test, labels_preds, out_path, display_labels):
    n_classes = len(labels_test[0]) if labels_test else 0
    display_labels = display_labels or [str(i) for i in range(n_classes)]
    if len(display_labels) < n_classes:
        display_labels = [str(i) for i in range(n_classes)]
    y_test = _convert_to_majority_index(labels_test)
    y_pred = _convert_to_majority_index(labels_preds)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels[:n_classes])
    disp.plot(include_values=True, cmap=plt.cm.Blues)
    plt.title("Eval - Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def run_eval(data_dir: Path, model_name: str, dataset: str, gpu_id: int = 0, split_name: str = "test"):
    """Item-level ensemble evaluation matching voices/eval_voiced.py."""
    data_dir = Path(data_dir)
    print(f"[eval] model={model_name} dataset={dataset} split={split_name} gpu_id={gpu_id}")
    print(f"[eval] data_dir={data_dir}")
    sys.stdout.flush()
    cfg = load_dataset(dataset)
    label_names = cfg["label_names"]
    ail_path = ail_path_for(data_dir, dataset, split_name)
    model_path = data_dir / "trained_model.disco"
    print(f"[eval] model_file={model_path}")
    print(f"[eval] AIL={ail_path}")
    sys.stdout.flush()

    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_tag = "/GPU:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_tag = "/CPU:0"

    print(f"[eval] loading npy matrices ...")
    sys.stdout.flush()
    Xi = np.load(data_dir / f"Xi_{split_name}.npy", allow_pickle=True)
    Yi = np.load(data_dir / f"Yi_{split_name}.npy", allow_pickle=True)
    A = np.load(data_dir / f"A_{split_name}.npy", allow_pickle=True)
    if model_name != "diadem":
        # Clamp cold-start annotator indices to trained UNK (last row of Wa)
        A_train = np.load(data_dir / "A_train.npy", allow_pickle=True)
        from diadem.data_io import remap_disco_oov_annotators
        A, n_a, n_oov = remap_disco_oov_annotators(A_train, A)
        print(f"[eval] disco OOV remap: {n_oov}/{len(A)} rows → UNK (n_a expect ≥{n_a})")
    raw_ail = pd.read_csv(ail_path, header=None, names=["item", "annotator", "label", "message"])
    unique_items = raw_ail["item"].unique()
    print(f"[eval] Xi={Xi.shape} Yi={Yi.shape} A={A.shape} n_items={len(unique_items)}")
    sys.stdout.flush()

    metadata_mapping = None
    all_annotator_metadata = None
    all_annotator_ids = None
    aid_to_idx = None
    if model_name == "diadem" and A.shape[1] > 1:
        from diadem.data_io import remap_disco_oov_annotators
        A_train = np.load(data_dir / "A_train.npy", allow_pickle=True)
        A, n_ann, n_oov = remap_disco_oov_annotators(A_train, A)
        print(f"[eval] diadem identity OOV remap: {n_oov}/{len(A)} rows → UNK (n_ann≥{n_ann})")
        annotator_ids_from_A = A[:, 0].astype(int)
        annotator_metadata_from_A = A[:, 1:]
        unique_ann_ids = np.unique(annotator_ids_from_A)
        metadata_mapping = {}
        aid_to_idx = {}
        for aid in unique_ann_ids:
            idx = np.where(annotator_ids_from_A == aid)[0]
            if len(idx) > 0:
                metadata_mapping[aid] = annotator_metadata_from_A[idx[0]]
                aid_to_idx[aid] = int(aid)
        all_annotator_ids = np.array(list(metadata_mapping.keys()), dtype=np.int32)
        all_annotator_metadata = np.array(list(metadata_mapping.values()), dtype=np.float32)
        print(f"[eval] {len(metadata_mapping)} unique annotators with metadata (ensemble)")
        sys.stdout.flush()

    print(f"[eval] loading model on {gpu_tag} ...")
    sys.stdout.flush()
    with tf.device(gpu_tag):
        model = load_object(str(model_path))
        model.drop_p = 0.0
    print(f"[eval] loaded {type(model).__name__}")
    sys.stdout.flush()

    print(f"[eval] predicting over {len(unique_items)} items ...")
    sys.stdout.flush()
    empirical_labels, predictions, data_to_write = [], [], []
    for unique_item in tqdm(unique_items, desc="eval"):
        first_idx = raw_ail.index[raw_ail["item"] == unique_item][0]
        empirical_label = Yi[first_idx]
        empirical_labels.append(empirical_label)

        item_anns = raw_ail[raw_ail["item"] == unique_item]
        annotator_ids = item_anns["annotator"].tolist()
        x_s = tf.cast(np.expand_dims(Xi[first_idx, :], axis=0), tf.float32)

        if model_name == "diadem":
            py, _ = model.decode_y_ensemble(
                x_s,
                all_annotator_metadata=all_annotator_metadata,
                all_annotator_ids=all_annotator_ids,
            )
            predicted_dist = py.numpy()[0].tolist()
            ann_preds = []
            use_id = getattr(model, "use_identity_embedding", False)
            for row_i in item_anns.index:
                ri = int(row_i)
                if ri < len(A) and A.shape[1] > 1:
                    meta = tf.cast(np.expand_dims(A[ri, 1:], axis=0), tf.float32)
                    a_idx = tf.constant([int(A[ri, 0])], dtype=tf.int32) if use_id else None
                else:
                    meta = tf.zeros([1, max(A.shape[1] - 1, 1)], dtype=tf.float32)
                    a_idx = tf.constant([0], dtype=tf.int32) if use_id else None
                z, z_a = model.encode(x_s, meta, a_idx=a_idx)
                yi_prob, _ = model.decode_yi(z, z_a)
                ann_preds.append(int(tf.argmax(yi_prob, axis=1).numpy()[0]))
        else:
            z = model.encode(x_s, tf.cast(A[first_idx:first_idx + 1, :], tf.float32))
            py, _ = model.decode_y(z)
            predicted_dist = py.numpy()[0].tolist()
            z = model.encode(x_s, tf.cast(A[first_idx:first_idx + 1, :], tf.float32))
            yi_prob, _ = model.decode_yi(z)
            ann_preds = [int(tf.argmax(yi_prob, axis=1).numpy()[0])] * len(annotator_ids)

        predictions.append(predicted_dist)
        data_to_write.append({
            "message_id": unique_item,
            "labels": predicted_dist,
            "annotator_ids": annotator_ids,
            "annotator_labels": ann_preds,
        })

    print("[eval] computing metrics ...")
    sys.stdout.flush()
    f1_macro, f1_micro, f1_weighted, prec, rec = _measure_f1(empirical_labels, predictions)
    summary = {
        "model": model_name,
        "dataset": dataset,
        "split": split_name,
        "n_items": len(unique_items),
        "KL": _kl_empirical2pred(empirical_labels, predictions),
        "accuracy": _measure_accuracy(empirical_labels, predictions),
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision": prec,
        "recall": rec,
        "label_names": label_names,
    }

    print("[eval] writing outputs (cmatrix, preds.xlsx, tsv, eval_summary.json) ...")
    sys.stdout.flush()
    _confusion_matrix_plot(empirical_labels, predictions, data_dir / "cmatrix.pdf", label_names)
    pd.DataFrame(data_to_write).to_excel(data_dir / "preds.xlsx", index=False)
    with open(data_dir / "op_predictions.tsv", "w") as f:
        for row in data_to_write:
            f.write(f"{row['message_id']}\t{row['labels']}\n")
    with open(data_dir / "op_ann.tsv", "w") as f:
        for row in data_to_write:
            f.write(f"{row['message_id']}\t{row['annotator_labels']}\n")

    out = data_dir / "eval_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[eval] Saved: {data_dir}/cmatrix.pdf, preds.xlsx, op_predictions.tsv, op_ann.tsv")
    print(f"[eval] Saved: {out}")
    sys.stdout.flush()
    return summary
