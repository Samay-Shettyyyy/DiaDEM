"""Side-by-side comparison table for paper (Markdown + LaTeX)."""
import json
from pathlib import Path
from typing import List

from diadem.paths import run_dir


METRICS = [
    # ── Standard ──
    ("accuracy",    "Acc."),
    ("f1_macro",    "F1$_{mac}$"),
    ("f1_weighted", "F1$_{wt}$"),
    ("kappa",       "$\\kappa$"),
    ("mcc",         "MCC"),
    # ── Soft-label ──
    ("jsd_mean",    "JSD$\\downarrow$"),
    ("md_mean",     "MD$\\downarrow$"),
    ("er_mean",     "ER$\\downarrow$"),
    ("ece",         "ECE$\\downarrow$"),
    # ── NEW — Ordinal-aware ──
    ("rps",         "RPS$\\downarrow$"),
    ("wasserstein", "Wass$\\downarrow$"),
    # ── NEW — Symmetric divergences ──
    ("hellinger",   "Hell$\\downarrow$"),
    ("bhat_bc",     "BC$\\uparrow$"),
    ("bhat_bd",     "BD$\\downarrow$"),
    # ── NEW — Disagreement probability ──
    ("actual_epd",    "EPD$_{act}$"),
    ("predicted_epd", "EPD$_{pred}$"),
    # ── NEW — Disagreement tracking (EPD correlation) ──
    ("epd_pearson_r",  "EPD-$r$"),
    ("epd_spearman_r", "EPD-$\\rho$"),
]

# Keys where lower is better (used for bolding the best)
_LOWER_IS_BETTER = {
    "jsd_mean", "md_mean", "er_mean", "ece",
    "rps", "wasserstein", "hellinger", "bhat_bd",
}
# Keys where higher is better
_HIGHER_IS_BETTER = {
    "accuracy", "f1_macro", "f1_weighted", "kappa", "mcc",
    "bhat_bc", "epd_pearson_r", "epd_spearman_r",
}
# EPD actual/predicted: no clear winner direction — don't bold
_NO_BOLD = {"actual_epd", "predicted_epd"}


def _load_model_summary(dataset: str, split_mode: str, model: str) -> dict:
    base = run_dir(dataset, split_mode, model)
    err_path = base / "error_analysis" / "analysis_summary.json"
    if err_path.exists():
        with open(err_path) as f:
            raw = json.load(f)
        m   = raw.get("metrics", raw)
        dc  = raw.get("disagreement_correlation", {})
        epd = dc.get("epd", {})
        return {
            "model": model,
            # Standard
            "accuracy":    m.get("accuracy"),
            "f1_macro":    m.get("f1_macro"),
            "f1_weighted": m.get("f1_weighted"),
            "kappa":       m.get("kappa"),
            "mcc":         m.get("mcc"),
            # Soft-label
            "jsd_mean": m.get("mean_jsd"),
            "md_mean":  m.get("perspectivist_average_md"),
            "er_mean":  m.get("perspectivist_average_er"),
            "ece":      m.get("ece"),
            # NEW — ordinal-aware
            "rps":         m.get("mean_rps"),
            "wasserstein": m.get("mean_wasserstein"),
            # NEW — symmetric divergences
            "hellinger": m.get("mean_hellinger"),
            "bhat_bc":   m.get("mean_bhattacharyya_coeff"),
            "bhat_bd":   m.get("mean_bhattacharyya_dist"),
            # NEW — disagreement probability
            "actual_epd":    m.get("mean_actual_epd"),
            "predicted_epd": m.get("mean_predicted_epd"),
            # NEW — EPD correlation
            "epd_pearson_r":  epd.get("pearson_r"),
            "epd_spearman_r": epd.get("spearman_r"),
        }

    eval_path = base / "eval_summary.json"
    if eval_path.exists():
        with open(eval_path) as f:
            row = json.load(f)
        row["model"] = model
        if "KL" in row and "jsd_mean" not in row:
            row["jsd_mean"] = row["KL"]
        return row
    return {}


def load_summaries(dataset: str, split_mode: str, models: List[str]) -> List[dict]:
    rows = []
    for m in models:
        row = _load_model_summary(dataset, split_mode, m)
        if row:
            rows.append(row)
    return rows


def to_markdown(dataset: str, split_mode: str, models: List[str]) -> str:
    rows = load_summaries(dataset, split_mode, models)
    if not rows:
        return "No eval_summary.json or error_analysis/analysis_summary.json found. Run eval/error_detailed first."

    lines = [
        f"## {dataset.upper()} ({split_mode})",
        "",
        "| Model | " + " | ".join(l for _, l in METRICS) + " |",
        "| --- | " + " | ".join(["---"] * len(METRICS)) + " |",
    ]
    best = {}
    for key, _ in METRICS:
        if key in _NO_BOLD:
            continue
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        if key in _LOWER_IS_BETTER:
            best[key] = min(vals)
        elif key in _HIGHER_IS_BETTER:
            best[key] = max(vals)

    for r in rows:
        cells = [r["model"]]
        for key, _ in METRICS:
            v = r.get(key)
            if v is None:
                cells.append("-")
            else:
                s = f"{v:.4f}"
                if key in best and abs(v - best[key]) < 1e-9:
                    s = f"**{s}**"
                cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_comparison(dataset: str, split_mode: str, models: List[str], out_path: Path):
    md = to_markdown(dataset, split_mode, models)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote comparison -> {out_path}")
    print(md)
