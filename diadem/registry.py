"""Load dataset and model JSON configs."""
import json
from pathlib import Path
from typing import Any, Dict

from diadem.paths import CONFIGS, DATASETS_CFG, ROOT, resolve


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_dataset(name: str) -> Dict[str, Any]:
    cfg = load_json(DATASETS_CFG / f"{name}.json")
    cfg["_root"] = str(ROOT)
    if "source_csv" in cfg:
        cfg["source_csv_path"] = str(resolve(ROOT, cfg["source_csv"]))
    if "split_csv_pattern" in cfg:
        cfg["split_csv_pattern_resolved"] = str(resolve(ROOT, cfg["split_csv_pattern"]))
    # Ensure label_names always present
    if "label_names" not in cfg:
        cfg["label_names"] = []
    return cfg


def load_model_cfg(model: str, dataset: str = None) -> Dict[str, Any]:
    if model == "diadem":
        return load_json(CONFIGS / "diadem.json")
    if dataset == "voiced":
        voiced_baseline = CONFIGS / "disco_baseline_voiced.json"
        if voiced_baseline.exists():
            return load_json(voiced_baseline)
    return load_json(CONFIGS / "disco_baseline.json")


def split_csv_for(dataset_cfg: Dict[str, Any], split_mode: str) -> Path:
    pattern = dataset_cfg["split_csv_pattern_resolved"].format(split_mode=split_mode)
    return Path(pattern)


def split_col_for(dataset_cfg: Dict[str, Any], split_mode: str) -> str:
    if split_mode == "by_annotator":
        return dataset_cfg["split_by_annotator"]
    if split_mode == "by_item":
        return dataset_cfg["split_by_item"]
    raise ValueError(f"Unknown split_mode: {split_mode}")
