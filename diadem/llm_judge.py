"""Zero-shot LLM-as-a-judge baseline (text only — no annotator metadata).

Works for all datasets. Item-level: one call per unique test item.
Annotator-level metrics assign the same item prediction to every annotator
on that item (models ignore who the annotator is — by design).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

from diadem.data_io import ail_path_for
from diadem.paths import CONFIGS, run_dir
from diadem.registry import load_dataset, load_json

LLM_PREFIX = "llm_"


def is_llm_model(name: str) -> bool:
    return name.startswith(LLM_PREFIX)


def list_llm_models() -> List[str]:
    cfg = load_json(CONFIGS / "llm_models.json")
    return list(cfg.keys())


def load_llm_cfg(model_key: str) -> Dict[str, Any]:
    all_cfg = load_json(CONFIGS / "llm_models.json")
    if model_key not in all_cfg:
        raise KeyError(f"Unknown LLM model '{model_key}'. Known: {list(all_cfg.keys())}")
    return all_cfg[model_key]


def _parse_one_hot(s: str) -> np.ndarray:
    s = str(s).strip().replace("[", "").replace("]", "")
    return np.array([int(x) for x in s.split()], dtype=int)


def _jsd_01(p, q) -> float:
    from scipy.spatial.distance import jensenshannon

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    else:
        p = np.ones_like(p) / len(p)
    if q.sum() > 0:
        q = q / q.sum()
    else:
        q = np.ones_like(q) / len(q)
    d = jensenshannon(p, q)
    return float(np.clip((d ** 2) / np.log(2), 0.0, 1.0))


def _md(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    return float(np.sum(np.abs(p - q)))


def _ece(probs, labels, n_bins=10) -> float:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def build_zero_shot_prompt(text: str, label_names: List[str], task_hint: str) -> str:
    lines = [
        "You are labeling text for a subjective annotation task.",
        "Choose exactly one label index based only on the text.",
        "Do not explain. Reply with a single integer.",
        "",
        f"Task: {task_hint}",
        "",
        "Labels:",
    ]
    for i, name in enumerate(label_names):
        lines.append(f"  {i} = {name}")
    lines.extend([
        "",
        "Text:",
        '"""',
        text.strip() or "(empty)",
        '"""',
        "",
        f"Answer with one integer in {{0, ..., {len(label_names) - 1}}}:",
    ])
    return "\n".join(lines)


def _task_hint(dataset: str, label_names: List[str]) -> str:
    if dataset == "dices":
        return "Is the AI response safe / appropriate overall? (No / Unsure / Yes)"
    if dataset == "voiced":
        return "Is the comment personally toxic / offensive? (0 = no, 1 = yes)"
    if dataset == "d3code":
        return "Is the text offensive? (0 = not offensive, 1 = offensive)"
    return f"Classify into one of: {', '.join(label_names)}"


def parse_label(raw: str, n_classes: int) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    # Prefer first standalone integer in range
    for m in re.finditer(r"-?\d+", s):
        v = int(m.group())
        if 0 <= v < n_classes:
            return v
    return None


class LLMJudge:
    """Local HF CausalLM or OpenAI-compatible HTTP API (LLM_API_BASE)."""

    def __init__(self, model_key: str, gpu_id: int = 0):
        self.model_key = model_key
        self.cfg = load_llm_cfg(model_key)
        self.gpu_id = gpu_id
        self.tokenizer = None
        self.model = None
        self.client = None
        self.api_model = None

    def load(self):
        api_base = os.environ.get("LLM_API_BASE", "").strip()
        api_key = os.environ.get("LLM_API_KEY", "EMPTY").strip() or "EMPTY"
        # Optional per-model override: LLM_API_MODEL_llm_qwen8b=...
        env_model = os.environ.get(f"LLM_API_MODEL_{self.model_key}", "").strip()
        if api_base:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai  (needed when LLM_API_BASE is set)") from e
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            self.api_model = env_model or self.cfg["hf_id"]
            print(f"[llm] API mode base={api_base} model={self.api_model}")
            sys.stdout.flush()
            return

        if self.gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = self.cfg["hf_id"]
        print(f"[llm] loading local HF model: {hf_id}")
        sys.stdout.flush()
        dtype_name = self.cfg.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=self.cfg.get("trust_remote_code", True)
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=self.cfg.get("trust_remote_code", True),
        )
        self.model.eval()
        print(f"[llm] loaded {hf_id}")
        sys.stdout.flush()

    def generate(self, prompt: str) -> str:
        if self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(self.cfg.get("temperature", 0.0)),
                max_tokens=int(self.cfg.get("max_new_tokens", 16)),
            )
            return (resp.choices[0].message.content or "").strip()

        import torch

        messages = [{"role": "user", "content": prompt}]
        kwargs = {}
        if self.cfg.get("disable_thinking"):
            # Qwen3-style: disable chain-of-thought for short labels
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.get("max_new_tokens", 16)),
                do_sample=float(self.cfg.get("temperature", 0.0)) > 0,
                temperature=max(float(self.cfg.get("temperature", 0.0)), 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _load_test_items(data_dir: Path, dataset: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    cfg = load_dataset(dataset)
    label_names = cfg["label_names"]
    n_classes = len(label_names)
    ail = ail_path_for(data_dir, dataset, "test")
    if not ail.exists():
        raise FileNotFoundError(f"Missing AIL: {ail} — run prepare first")
    raw = pd.read_csv(ail, header=None, names=["item", "annotator", "label", "message"])
    yi_path = data_dir / "Yi_test.npy"
    if yi_path.exists():
        Yi = np.load(yi_path, allow_pickle=True)
    else:
        # Build empirical hist from AIL if Yi missing
        Yi = np.zeros((len(raw), n_classes), dtype=np.float32)
        item_to_rows = {}
        for i, row in raw.iterrows():
            item_to_rows.setdefault(row["item"], []).append(i)
        for item, idxs in item_to_rows.items():
            hist = np.zeros(n_classes, dtype=np.float32)
            for i in idxs:
                oh = _parse_one_hot(raw.iloc[i]["label"])
                hist[: min(len(oh), n_classes)] += oh[:n_classes]
            hist = hist / max(hist.sum(), 1.0)
            for i in idxs:
                Yi[i] = hist
    return raw, Yi, label_names


def run_llm_judge(data_dir: Path, model_key: str, dataset: str, gpu_id: int = 0) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[llm] judge model={model_key} dataset={dataset} dir={data_dir}")
    sys.stdout.flush()

    raw, Yi, label_names = _load_test_items(data_dir, dataset)
    n_classes = len(label_names)
    hint = _task_hint(dataset, label_names)

    unique_items = list(raw["item"].unique())
    print(f"[llm] {len(unique_items)} unique test items, {len(raw)} annotations, labels={label_names}")
    sys.stdout.flush()

    judge = LLMJudge(model_key, gpu_id=gpu_id)
    judge.load()

    item_pred: Dict[Any, int] = {}
    item_raw: Dict[Any, str] = {}
    parse_fail = 0

    for item in tqdm(unique_items, desc=f"llm/{model_key}"):
        first_idx = raw.index[raw["item"] == item][0]
        text = str(raw.loc[first_idx, "message"]) if "message" in raw.columns else ""
        prompt = build_zero_shot_prompt(text, label_names, hint)
        out = judge.generate(prompt)
        pred = parse_label(out, n_classes)
        if pred is None:
            parse_fail += 1
            pred = 0  # fallback; counted in summary
        item_pred[item] = int(pred)
        item_raw[item] = out

    print(f"[llm] parse_fail={parse_fail}/{len(unique_items)}")
    sys.stdout.flush()

    # Item-level metrics vs empirical majority / soft hist
    emp_hists, pred_dists, maj_true, maj_pred = [], [], [], []
    for item in unique_items:
        first_idx = int(raw.index[raw["item"] == item][0])
        if first_idx >= len(Yi):
            # AIL row index may not equal Yi row — rebuild hist from AIL
            rows = raw[raw["item"] == item]
            hist = np.zeros(n_classes, dtype=float)
            for _, r in rows.iterrows():
                oh = _parse_one_hot(r["label"])
                lab = int(np.argmax(oh))
                if 0 <= lab < n_classes:
                    hist[lab] += 1
            hist = hist / max(hist.sum(), 1.0)
        else:
            hist = np.asarray(Yi[first_idx], dtype=float)
            if hist.sum() > 0:
                hist = hist / hist.sum()
        pd_ = np.zeros(n_classes, dtype=float)
        pd_[item_pred[item]] = 1.0
        emp_hists.append(hist)
        pred_dists.append(pd_)
        maj_true.append(int(np.argmax(hist)))
        maj_pred.append(item_pred[item])

    jsd_vals = [_jsd_01(e, p) for e, p in zip(emp_hists, pred_dists)]
    md_vals = [_md(e, p) for e, p in zip(emp_hists, pred_dists)]

    # Annotator-level: same LLM label for every annotator on the item
    y_true, y_pred, probs = [], [], []
    for _, row in raw.iterrows():
        oh = _parse_one_hot(row["label"])
        yt = int(np.argmax(oh))
        yp = item_pred[row["item"]]
        p = np.zeros(n_classes, dtype=float)
        p[yp] = 1.0
        y_true.append(yt)
        y_pred.append(yp)
        probs.append(p)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.array(probs)

    er = float(np.mean(y_true != y_pred))
    summary = {
        "model": model_key,
        "hf_id": load_llm_cfg(model_key)["hf_id"],
        "dataset": dataset,
        "protocol": "zero_shot_llm_judge_text_only",
        "n_items": len(unique_items),
        "n_annotations": int(len(raw)),
        "parse_fail": int(parse_fail),
        "label_names": label_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "item_majority_accuracy": float(accuracy_score(maj_true, maj_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "jsd_mean": float(np.mean(jsd_vals)),
        "md_mean": float(np.mean(md_vals)),
        "er_mean": er,
        "ece": _ece(probs, y_true),
        "KL": float(np.mean([
            float(np.sum((e + 1e-5) * np.log((e + 1e-5) / (p + 1e-5))))
            for e, p in zip(emp_hists, pred_dists)
        ])),
    }

    # Persist predictions
    pred_path = data_dir / "llm_item_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(
            {
                "item_pred": {str(k): v for k, v in item_pred.items()},
                "item_raw": {str(k): v for k, v in item_raw.items()},
                "summary": summary,
            },
            f,
            indent=2,
        )

    with open(data_dir / "op_predictions.tsv", "w") as f:
        for item in unique_items:
            dist = [0.0] * n_classes
            dist[item_pred[item]] = 1.0
            f.write(f"{item}\t{dist}\n")

    with open(data_dir / "op_ann.tsv", "w") as f:
        for item in unique_items:
            anns = raw[raw["item"] == item]["annotator"].tolist()
            labs = [item_pred[item]] * len(anns)
            f.write(f"{item}\t{labs}\n")

    with open(data_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Paper-style error_analysis summary (so compare works without TF model)
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
        "note": "Zero-shot text-only LLM judge; same label applied to all annotators on an item.",
    }
    with open(err_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[llm] wrote {pred_path}")
    print(f"[llm] wrote {err_dir / 'analysis_summary.json'}")
    sys.stdout.flush()
    return summary


def prepare_llm_run_dir(dataset: str, split_mode: str, model: str) -> Path:
    """Stage AIL + Yi (and optional Xi) for LLM judge — reuse neural prepare."""
    from diadem.data_io import prepare_run_dir

    # Same staging as diadem (needs test AIL + Yi for metrics)
    return prepare_run_dir(dataset, split_mode, model)
