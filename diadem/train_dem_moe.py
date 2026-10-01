"""
Training script for DEM-MoE on pre-extracted .npy data.

Reads:
  Xi_train.npy         — (N, text_dim) sentence embeddings
  Yi_train.npy         — (N, num_classes) soft label distributions
  A_train.npy          — (N, 1 + meta_dim) [annotator_id | one-hot features | cont features]
  metadata_encoding_config.json — tells us which features are categorical vs continuous

Three-phase training schedule (Xu et al. 2026):
  Phase 1 (A): main KL + KL-reg + load balance           (epochs 1..phase1_epochs)
  Phase 2 (B): + orthogonality + variance losses          (epochs phase1..phase2_epochs)
  Phase 3 (C): main KL + KL-reg only, gating frozen      (remaining epochs)

Saves:
  dem_moe_model.pt     — PyTorch state dict
  dem_moe_config.json  — hyperparams + data metadata needed by eval
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── resolve local imports ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from diadem.models.dem_moe import DemMoEClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_metadata_config(config_path: Path):
    """
    Parse metadata_encoding_config.json to understand the A-matrix layout.

    Returns
    -------
    cat_feature_names : list[str]   — categorical attribute names
    cat_sizes         : list[int]   — vocab size per attribute
    cat_slices        : list[slice] — slice in A[:,1:] for each attribute's one-hot block
    cont_start        : int         — index in A[:,1:] where continuous features begin
    num_cont          : int         — number of continuous features
    """
    with open(config_path) as f:
        cfg = json.load(f)

    feature_names = cfg.get("feature_names", [])
    cat_values    = cfg.get("categorical_values", {})

    cat_feature_names, cat_sizes, cat_slices = [], [], []
    cont_feature_names = []
    offset = 0

    for fn in feature_names:
        vals = cat_values.get(fn, [])
        if vals:
            n = len(vals)
            cat_feature_names.append(fn)
            cat_sizes.append(n)
            cat_slices.append(slice(offset, offset + n))
            offset += n
        else:
            cont_feature_names.append(fn)

    cont_start = offset
    num_cont   = len(cont_feature_names)

    return cat_feature_names, cat_sizes, cat_slices, cont_start, num_cont


def build_tensors(A: np.ndarray, cat_slices: list, cont_start: int, num_cont: int):
    """
    Convert A matrix (N, 1 + meta_dim) into:
      annotator_ids : (N,)               int64
      identity_ids  : (N, num_cat_attrs) int64 (argmax of each one-hot block)
      cont_feats    : (N, num_cont)      float32  or None
    """
    meta = A[:, 1:].astype(np.float32)

    annotator_ids = A[:, 0].astype(np.int64)
    # Re-index annotators to 0..num_unique-1
    unique_ids    = np.unique(annotator_ids)
    id_map        = {v: i for i, v in enumerate(unique_ids)}
    annotator_ids = np.array([id_map[x] for x in annotator_ids], dtype=np.int64)

    identity_ids = np.stack(
        [np.argmax(meta[:, s], axis=1).astype(np.int64) for s in cat_slices],
        axis=1
    )  # (N, num_cat_attrs)

    if num_cont > 0:
        cont_feats = meta[:, cont_start: cont_start + num_cont]
    else:
        cont_feats = None

    return annotator_ids, identity_ids, cont_feats, id_map


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:       Path,
    out_dir:        Path,
    model_cfg:      dict,
    gpu_id:         int  = 0,
    split_name:     str  = "train",
):
    """
    Full training pipeline.  Called from pipeline.py or directly.
    Writes dem_moe_model.pt + dem_moe_config.json to out_dir.
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available()
                          else "cpu")
    print(f"[dem_moe train] device={device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[dem_moe train] Loading data …")
    Xi = np.load(data_dir / f"Xi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    Yi = np.load(data_dir / f"Yi_{split_name}.npy",  allow_pickle=True).astype(np.float32)
    A  = np.load(data_dir / f"A_{split_name}.npy",   allow_pickle=True)

    # Load metadata config — try several possible locations
    cfg_path = None
    for candidate in [
        data_dir / "metadata_encoding_config.json",
        data_dir.parent / "processed" / "metadata_encoding_config.json",
    ]:
        if candidate.exists():
            cfg_path = candidate
            break
    if cfg_path is None:
        raise FileNotFoundError(
            "metadata_encoding_config.json not found in data_dir or processed/. "
            "Run the prepare step first."
        )

    cat_names, cat_sizes, cat_slices, cont_start, num_cont = parse_metadata_config(cfg_path)
    print(f"[dem_moe train] categorical attrs: {cat_names} sizes={cat_sizes}")
    print(f"[dem_moe train] continuous feats:  {num_cont}")

    ann_ids, identity_ids, cont_feats, id_map = build_tensors(A, cat_slices, cont_start, num_cont)

    num_annotators = len(id_map)
    num_classes    = Yi.shape[1]
    text_dim       = Xi.shape[1]

    print(f"[dem_moe train] N={len(Xi)} text_dim={text_dim} "
          f"num_classes={num_classes} num_annotators={num_annotators}")

    # ── Build tensors ──────────────────────────────────────────────────────────
    t_xi   = torch.from_numpy(Xi)
    t_yi   = torch.from_numpy(Yi)
    t_ann  = torch.from_numpy(ann_ids)
    t_id   = torch.from_numpy(identity_ids)
    t_cont = torch.from_numpy(cont_feats).float() if cont_feats is not None else None

    tensors = [t_xi, t_yi, t_ann, t_id]
    if t_cont is not None:
        tensors.append(t_cont)

    dataset    = TensorDataset(*tensors)
    batch_size = int(model_cfg.get("batch_size", 256))
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Build model ────────────────────────────────────────────────────────────
    num_experts  = int(model_cfg.get("num_experts",    8))
    hidden_dim   = int(model_cfg.get("hidden_dim",     256))
    ann_emb_dim  = int(model_cfg.get("ann_embed_dim",  128))
    id_emb_dim   = int(model_cfg.get("id_embed_dim",   64))
    proj_id_dim  = int(model_cfg.get("projected_id_dim", 256))
    dropout      = float(model_cfg.get("dropout",      0.3))
    top_k        = int(model_cfg.get("top_k",          3))
    T_init       = float(model_cfg.get("T_init",       6.0))
    T_decay      = float(model_cfg.get("T_decay",      0.92))
    T_min        = float(model_cfg.get("T_min",        4.0))
    T_warmup     = int(model_cfg.get("T_warmup",       25))

    n_epochs     = int(model_cfg.get("n_epoch",        30))
    lr           = float(model_cfg.get("lr", model_cfg.get("learning_rate", 5e-4)))
    kl_weight    = float(model_cfg.get("kl_weight",    1e-4))
    ortho_weight = float(model_cfg.get("ortho_weight", 0.01))
    var_weight   = float(model_cfg.get("var_weight",   0.01))
    load_weight  = float(model_cfg.get("load_weight",  0.01))

    # Phase boundaries: explicit epochs > fractional fallback
    phase_a = int(model_cfg.get("phase_a_epochs", 0))
    phase_b = int(model_cfg.get("phase_b_epochs", 0))
    if phase_a > 0 and phase_b > 0:
        phase1_end = phase_a
        phase2_end = phase_a + phase_b
    else:
        phase1_end = int(model_cfg.get("phase1_end", math.floor(n_epochs * 0.4)))
        phase2_end = int(model_cfg.get("phase2_end", math.floor(n_epochs * 0.8)))

    # Class weights to fight ordinal-middle collapse (e.g. DICES "Unsure")
    use_class_weights = bool(model_cfg.get("use_class_weights", False))
    class_weight_tensor = None
    if use_class_weights and num_classes > 2:
        # Compute inverse-frequency weights from training Yi
        yi_counts = Yi.sum(axis=0).astype(float)   # shape [num_classes]
        yi_counts = np.maximum(yi_counts, 1.0)
        inv_freq  = 1.0 / yi_counts
        inv_freq  = inv_freq / inv_freq.sum() * num_classes   # normalize to sum=K
        class_weight_tensor = torch.tensor(inv_freq, dtype=torch.float32, device=device)
        print(f"[dem_moe train] class weights (inv-freq): {inv_freq.tolist()}")

    model = DemMoEClassifier(
        text_embed_dim   = text_dim,
        num_annotators   = num_annotators,
        num_cat_per_attr = cat_sizes,
        num_cont_feats   = num_cont,
        num_classes      = num_classes,
        hidden_dim       = hidden_dim,
        num_experts      = num_experts,
        ann_embed_dim    = ann_emb_dim,
        id_embed_dim     = id_emb_dim,
        projected_id_dim = proj_id_dim,
        dropout          = dropout,
        top_k            = top_k,
        T_init           = T_init,
        T_decay          = T_decay,
        T_min            = T_min,
        T_warmup         = T_warmup,
    ).to(device)

    print(f"[dem_moe train] model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss   = float("inf")
    best_state  = None

    for epoch in range(1, n_epochs + 1):
        # Determine phase
        if epoch <= phase1_end:
            phase = 1
        elif epoch <= phase2_end:
            phase = 2
        else:
            phase = 3

        # Phase 3: freeze gating network
        for p in model.gate.parameters():
            p.requires_grad = (phase < 3)

        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs} [phase {phase}]",
                          leave=False):
            if t_cont is not None:
                bxi, byi, bann, bid, bcont = [b.to(device) for b in batch]
            else:
                bxi, byi, bann, bid = [b.to(device) for b in batch]
                bcont = None

            optimizer.zero_grad()

            logits, kl_ann, kl_id, topk_idx, topk_sc = model(
                bxi, bann, bid, bcont,
                epoch=epoch,
                return_routing=True,
            )

            loss, comps = model.compute_loss(
                logits, byi, kl_ann, kl_id,
                topk_indices=topk_idx,
                topk_scores=topk_sc,
                phase=phase,
                kl_weight=kl_weight,
                ortho_weight=ortho_weight,
                var_weight=var_weight,
                load_weight=load_weight,
                class_weights=class_weight_tensor,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update expert load-balancing biases (outside gradient graph)
            # topk_idx: [B, top_k]  topk_sc: [B, top_k]
            # Scatter scores back onto expert dimension to get per-expert avg load
            with torch.no_grad():
                expert_load = torch.zeros(model.num_experts, device=topk_idx.device)
                expert_load.scatter_add_(0, topk_idx.view(-1), topk_sc.view(-1).float())
                avg_load = expert_load / max(topk_idx.shape[0], 1)   # [num_experts]
                ideal    = 1.0 / model.num_experts
                model.expert_biases -= model.bias_lr * (avg_load - ideal)

            epoch_loss += comps["total"]
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        print(f"  Epoch {epoch:3d}/{n_epochs}  phase={phase}  loss={avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ───────────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "dem_moe_model.pt")

    save_cfg = {
        "text_embed_dim":    text_dim,
        "num_annotators":    num_annotators,
        "num_cat_per_attr":  cat_sizes,
        "cat_feature_names": cat_names,
        "cont_start":        cont_start,
        "num_cont_feats":    num_cont,
        "num_classes":       num_classes,
        "hidden_dim":        hidden_dim,
        "num_experts":       num_experts,
        "ann_embed_dim":     ann_emb_dim,
        "id_embed_dim":      id_emb_dim,
        "projected_id_dim":  proj_id_dim,
        "dropout":           dropout,
        "top_k":             top_k,
        "T_init":            T_init,
        "T_decay":           T_decay,
        "T_min":             T_min,
        "T_warmup":          T_warmup,
        "annotator_id_map":  {str(k): v for k, v in id_map.items()},
    }
    with open(out_dir / "dem_moe_config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)

    print(f"[dem_moe train] Saved model → {out_dir}/dem_moe_model.pt")
    print(f"[dem_moe train] Saved config → {out_dir}/dem_moe_config.json")
    return model, save_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def train_model(data_dir: Path, model_cfg: dict, gpu_id: int = 0):
    """Called by pipeline.py train step for model_name='dem_moe'."""
    train(data_dir=data_dir, out_dir=data_dir, model_cfg=model_cfg, gpu_id=gpu_id)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train DEM-MoE on .npy data")
    ap.add_argument("--data_dir",  required=True, help="Directory with Xi/Yi/A npy files")
    ap.add_argument("--out_dir",   default=None,  help="Output directory (default: data_dir)")
    ap.add_argument("--gpu_id",    type=int, default=0)
    ap.add_argument("--n_epoch",   type=int, default=30)
    ap.add_argument("--batch_size",type=int, default=256)
    ap.add_argument("--lr",        type=float, default=5e-4)
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--top_k",     type=int, default=3)
    args = ap.parse_args()

    cfg = {
        "n_epoch":       args.n_epoch,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "num_experts":   args.num_experts,
        "top_k":         args.top_k,
    }
    train(
        data_dir  = Path(args.data_dir),
        out_dir   = Path(args.out_dir or args.data_dir),
        model_cfg = cfg,
        gpu_id    = args.gpu_id,
    )


if __name__ == "__main__":
    main()
