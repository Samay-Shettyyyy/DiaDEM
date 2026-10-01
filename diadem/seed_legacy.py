"""Copy existing processed npy from legacy repo into cleanedup outputs (one-time bootstrap)."""
import ast
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diadem.paths import processed_dir

LEGACY = {
    ("dices",     "by_annotator"): Path("../dices/datasets/dices/processed_by_annotator"),
    ("dices",     "by_item"):      Path("../dices/datasets/dices/processed_by_item"),
    # VOICED npy live in experimental_data (gen_disco_dataset output)
    ("voiced",    "by_annotator"): Path("../voices/experimental_data/voiced_by_annotator"),
    ("voiced",    "by_item"):      Path("../voices/experimental_data/voiced_by_item"),
    # MultiPICo has a single split (no by_item); map both keys to same source
    ("multipico", "by_annotator"): Path("../Mutlipico/experimental_data/mp"),
    ("multipico", "by_item"):      Path("../Mutlipico/experimental_data/mp"),
}


def _fix_multipico_ail(src_path: Path, dst_path: Path):
    """
    MultiPICo AIL labels are stored as '[1 0]' one-hot strings.
    Convert to integer (argmax) so the rest of the pipeline works unchanged.
    Also renames 'dev' split files to 'test' (pipeline expects train/test).
    """
    for split_in, split_out in [("train", "train"), ("dev", "test")]:
        src_ail = src_path / f"mp_{split_in}_AIL.csv"
        if not src_ail.exists():
            continue
        df = pd.read_csv(src_ail, header=None,
                         names=["item", "annotator", "label", "message"])

        def _to_int(v):
            try:
                arr = np.array(ast.literal_eval(str(v)))
                return int(np.argmax(arr))
            except Exception:
                return int(v)

        df["label"] = df["label"].apply(_to_int)
        out_name = f"multipico_{split_out}_AIL.csv"
        df.to_csv(dst_path / out_name, index=False, header=False)
        print(f"[seed]   converted AIL {src_ail.name} -> {out_name} (labels: one-hot -> int)")

    # Also copy dev npy as test npy (pipeline expects Xi_test, Yi_test, etc.)
    for f in sorted(src_path.iterdir()):
        if f.suffix == ".npy" and "_dev" in f.name:
            new_name = f.name.replace("_dev", "_test")
            if not (dst_path / new_name).exists():
                shutil.copy2(f, dst_path / new_name)
                print(f"[seed]   aliased {f.name} -> {new_name}")


def seed_from_legacy(dataset: str, split_mode: str, root: Path):
    key = (dataset, split_mode)
    if key not in LEGACY:
        raise ValueError(f"No legacy mapping for {key}. "
                         f"Registered: {list(LEGACY.keys())}")
    src = (root / LEGACY[key]).resolve()
    dst = processed_dir(dataset, split_mode)
    print(f"[seed] legacy source : {src}")
    print(f"[seed] destination   : {dst}")
    sys.stdout.flush()
    if not src.exists():
        raise FileNotFoundError(f"Legacy processed dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for f in sorted(src.iterdir()):
        if f.suffix in (".npy", ".json", ".csv"):
            shutil.copy2(f, dst / f.name)
            n_copied += 1
            print(f"[seed]   copied {f.name}")
            sys.stdout.flush()

    # Dataset-specific post-processing
    if dataset == "multipico":
        _fix_multipico_ail(src, dst)

    print(f"[seed] done — copied {n_copied} files -> {dst}")
    sys.stdout.flush()


def main():
    import argparse
    from diadem.paths import ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["dices", "voiced", "multipico"])
    ap.add_argument("--split-mode", required=True,
                    choices=["by_annotator", "by_item"])
    args = ap.parse_args()
    seed_from_legacy(args.dataset, args.split_mode, ROOT)


if __name__ == "__main__":
    main()
