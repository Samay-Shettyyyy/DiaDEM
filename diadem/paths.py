"""Central path helpers — all paths relative to cleanedup_diadem root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATASETS_CFG = CONFIGS / "datasets"
OUTPUTS = ROOT / "outputs"


def dataset_cfg_path(name: str) -> Path:
    return DATASETS_CFG / f"{name}.json"


def processed_dir(dataset: str, split_mode: str) -> Path:
    return OUTPUTS / dataset / split_mode / "processed"


def run_dir(dataset: str, split_mode: str, model: str) -> Path:
    return OUTPUTS / dataset / split_mode / model


def resolve(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    return p
