"""Copy processed matrices into model-specific run directory."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from diadem.paths import processed_dir, run_dir


def _load_split(data_dir: Path, suffix: str) -> dict:
    print(f"[data_io] loading split{suffix} from {data_dir}")
    sys.stdout.flush()
    return {
        "Xi": np.load(data_dir / f"Xi{suffix}.npy"),
        "Y": np.load(data_dir / f"Y{suffix}.npy"),
        "Yi": np.load(data_dir / f"Yi{suffix}.npy"),
        "Ya": np.load(data_dir / f"Ya{suffix}.npy"),
        "I": np.load(data_dir / f"I{suffix}.npy"),
        "A": np.load(data_dir / f"A{suffix}.npy"),
    }


def _unique_annotator_metadata(A: np.ndarray) -> np.ndarray:
    """Match voices/train_disco_demographic.read_data_voiced ensemble metadata."""
    ids, metas = _unique_annotator_ids_and_meta(A)
    return metas


def _unique_annotator_ids_and_meta(A: np.ndarray):
    """Return (ids [N], metas [N, meta_dim]) in first-seen order. ids from A[:,0]."""
    if A is None or A.shape[1] <= 1:
        return None, None
    annotator_ids = A[:, 0].astype(int)
    ann_to_meta = {}
    for i, aid in enumerate(annotator_ids):
        if aid not in ann_to_meta:
            ann_to_meta[int(aid)] = A[i, 1:]
    ids = np.array(list(ann_to_meta.keys()), dtype=np.int32)
    metas = np.array(list(ann_to_meta.values()), dtype=np.float32)
    return ids, metas


def remap_disco_oov_annotators(A_train: np.ndarray, A_other: np.ndarray):
    """Map test/dev annotator indices unseen in train to a single UNK slot.

    DisCo uses tf.nn.embedding_lookup(Wa, a_idx). On by_annotator splits, test
    annotators are cold-start and can be out of range / never trained. Remap
    them to UNK = max(train_idx)+1 and size Wa with n_a = UNK+1.
    Returns (A_other_remapped, n_a, n_oov_rows).
    """
    train_ids = np.unique(A_train[:, 0].astype(int))
    train_id_set = set(train_ids.tolist())
    train_max = int(train_ids.max()) if len(train_ids) else 0
    unk_id = train_max + 1
    out = np.array(A_other, copy=True)
    col = out[:, 0].astype(int)
    oov = np.array([int(i) not in train_id_set for i in col], dtype=bool)
    n_oov = int(oov.sum())
    if n_oov:
        col = col.copy()
        col[oov] = unk_id
        out[:, 0] = col.astype(out.dtype)
    n_a = unk_id + 1
    return out, n_a, n_oov


def read_data(data_dir: Path, model_name: str) -> dict:
    """Load train + dev (test) matrices — aligned with voices/train_disco_demographic.py."""
    data_dir = Path(data_dir)
    print(f"[data_io] read_data model={model_name} dir={data_dir}")
    sys.stdout.flush()
    data = _load_split(data_dir, "_train")

    data["n_xi"] = data["Xi"].shape[1]
    data["yi_dim"] = data["Yi"].shape[1]
    data["ya_dim"] = data["Ya"].shape[1]
    data["y_dim"] = data["Y"].shape[1]
    data["n_i"] = int(np.max(data["I"])) + 1

    if model_name == "diadem":
        data["annotator_ids"] = data["A"][:, 0].astype(int)
        data["meta_dim"] = data["A"].shape[1] - 1
        data["n_a"] = data["meta_dim"]  # meta feature dim (legacy name)

        info_path = data_dir / "metadata_feature_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            data["metadata_feature_names"] = info.get("feature_names", [])
            fd = info.get("feature_dimensions", {})
            if isinstance(fd, dict):
                data["metadata_feature_names"] = data["metadata_feature_names"] or list(fd.keys())
                data["metadata_feature_dims"] = [fd.get(f, 1) for f in data["metadata_feature_names"]]
            else:
                data["metadata_feature_dims"] = list(fd) if fd else []
            print(f"[data_io] demographics: {data['metadata_feature_names']} dims={data['metadata_feature_dims']}")
        else:
            data["metadata_feature_names"] = ["metadata"]
            data["metadata_feature_dims"] = [data["meta_dim"]]
            print(f"[data_io] no metadata_feature_info.json — using meta_dim={data['meta_dim']}")

        # Identity table size = train max id + 1 (+ UNK after remapping test)
        data["n_annotators"] = int(np.max(data["A"][:, 0])) + 1
        ids, metas = _unique_annotator_ids_and_meta(data["A"])
        data["annotator_id_list"] = ids
        data["annotator_metadata"] = metas
        if metas is not None:
            print(f"[data_io] unique annotators for ensemble: {len(metas)} (n_annotators table={data['n_annotators']})")
    else:
        data["n_a"] = int(np.max(data["A"])) + 1
        data["n_annotators"] = data["n_a"]
        data["metadata_feature_names"] = []
        data["metadata_feature_dims"] = []
        data["annotator_metadata"] = None
        data["annotator_id_list"] = None
        print(f"[data_io] disco baseline: n_a (annotator count)={data['n_a']}")

    # Dev set = test split (matches voices_demographic.cfg dev_* = Xi_test.npy)
    test_path = data_dir / "Xi_test.npy"
    if test_path.exists():
        print("[data_io] loading dev/test split ...")
        sys.stdout.flush()
        dev = _load_split(data_dir, "_test")
        data["dev_Xi"] = dev["Xi"]
        data["dev_Y"] = dev["Y"]
        data["dev_Yi"] = dev["Yi"]
        data["dev_Ya"] = dev["Ya"]
        data["dev_I"] = dev["I"]
        data["dev_A"] = dev["A"]
        if model_name == "diadem":
            # Remap cold-start / unseen annotator IDs → UNK for identity embedding
            data["dev_A"], data["n_annotators"], n_oov = remap_disco_oov_annotators(
                data["A"], data["dev_A"]
            )
            print(
                f"[data_io] diadem identity OOV remap: {n_oov}/{len(data['dev_A'])} "
                f"test rows → UNK; n_annotators={data['n_annotators']}"
            )
            ids, metas = _unique_annotator_ids_and_meta(data["dev_A"])
            data["dev_annotator_id_list"] = ids
            data["dev_annotator_metadata"] = metas
            if metas is not None:
                print(f"[data_io] dev unique annotators: {len(metas)}")
        else:
            data["dev_annotator_metadata"] = None
            data["dev_annotator_id_list"] = None
            data["dev_A"], data["n_a"], n_oov = remap_disco_oov_annotators(data["A"], data["dev_A"])
            data["n_annotators"] = data["n_a"]
            print(
                f"[data_io] disco OOV remap: {n_oov}/{len(data['dev_A'])} test rows → UNK; "
                f"n_a={data['n_a']}"
            )
        print(f"[data_io] dev Xi {data['dev_Xi'].shape} Y {data['dev_Y'].shape}")
    else:
        print("[data_io] no Xi_test.npy — skipping dev set")
        data["dev_Xi"] = data["dev_Y"] = data["dev_Yi"] = data["dev_Ya"] = None
        data["dev_I"] = data["dev_A"] = data["dev_annotator_metadata"] = None
        data["dev_annotator_id_list"] = None

    print(f"[data_io] train ready: Xi {data['Xi'].shape} Y {data['Y'].shape} A {data['A'].shape} n_a={data['n_a']}")
    sys.stdout.flush()
    return data


def ail_path_for(data_dir: Path, dataset: str, split: str = "test") -> Path:
    prefix = {"voiced": "voiced", "dices": "dices", "d3code": "d3code",
              "multipico": "multipico"}.get(dataset, dataset)
    return Path(data_dir) / f"{prefix}_{split}_AIL.csv"


def prepare_run_dir(dataset: str, split_mode: str, model: str) -> Path:
    src = processed_dir(dataset, split_mode)
    dst = run_dir(dataset, split_mode, model)
    print(f"[prepare] src={src}")
    print(f"[prepare] dst={dst}")
    print(f"[prepare] model={model} (disco uses A_baseline as A)")
    sys.stdout.flush()
    if not src.exists():
        raise FileNotFoundError(f"Processed dir missing: {src} — run seed or preprocess first")
    dst.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for split in ("train", "test", "dev"):
        suffix = f"_{split}"
        for name in ("Xi", "Y", "Yi", "Ya", "I", "Ii", "annotator_metadata"):
            s = src / f"{name}{suffix}.npy"
            if s.exists():
                shutil.copy2(s, dst / f"{name}{suffix}.npy")
                n_copied += 1

        if model == "disco":
            ab = src / f"A_baseline{suffix}.npy"
            if ab.exists():
                shutil.copy2(ab, dst / f"A{suffix}.npy")
                print(f"[prepare]   {split}: A <- A_baseline{suffix}.npy")
                n_copied += 1
            else:
                full = src / f"A{suffix}.npy"
                if full.exists():
                    a = np.load(full)
                    np.save(dst / f"A{suffix}.npy", a[:, :1].astype(np.float32))
                    print(f"[prepare]   {split}: A <- first column of A{suffix}.npy (shape {a[:, :1].shape})")
                    n_copied += 1
        else:
            s = src / f"A{suffix}.npy"
            if s.exists():
                shutil.copy2(s, dst / f"A{suffix}.npy")
                n_copied += 1

    # Always refresh metadata sidecars (needed by majority_similar)
    for fname in ("metadata_encoding_config.json", "metadata_feature_info.json", "annotator_metadata.json"):
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)
            print(f"[prepare]   copied {fname}")
            n_copied += 1

    prefix = {"voiced": "voiced", "dices": "dices", "d3code": "d3code"}.get(dataset, dataset)
    for split in ("train", "test"):
        ail = src / f"{prefix}_{split}_AIL.csv"
        if ail.exists():
            shutil.copy2(ail, dst / f"{prefix}_{split}_AIL.csv")
            print(f"[prepare]   copied {prefix}_{split}_AIL.csv")
            n_copied += 1

    print(f"[prepare] done — {n_copied} files staged -> {dst}")
    sys.stdout.flush()
    return dst
