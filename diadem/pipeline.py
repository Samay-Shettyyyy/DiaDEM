"""Orchestrate split → preprocess → prepare → train → eval → compare (+ LLM judges)."""
import argparse
import subprocess
import sys
from pathlib import Path

from diadem.compare import write_comparison
from diadem.data_io import ail_path_for, prepare_run_dir
from diadem.error_detailed import run_error_detailed
from diadem.error_detailed_in_depth import run_error_in_depth
from diadem.eval import run_eval
from diadem.llm_judge import is_llm_model, prepare_llm_run_dir, run_llm_judge
from diadem.majority_similar import (
    is_rule_baseline,
    prepare_majority_run_dir,
    run_majority_similar,
)
from diadem.paths import ROOT, run_dir
from diadem.preprocess import run_preprocess
from diadem.registry import load_dataset, split_col_for, split_csv_for
from diadem.seed_legacy import seed_from_legacy
from diadem.train import train_model

# DEM-MoE (PyTorch) — lazy import to avoid TF/torch conflicts at startup
def _is_dem_moe(model_name: str) -> bool:
    return model_name == "dem_moe"


def _banner(title: str):
    print("")
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    sys.stdout.flush()


def run_split(dataset: str, split_mode: str):
    cfg = load_dataset(dataset)
    source = Path(cfg["source_csv_path"])
    if not source.exists():
        raise FileNotFoundError(f"Source CSV missing: {source}")
    out_csv = split_csv_for(cfg, split_mode)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_col = split_col_for(cfg, split_mode)
    script = ROOT / "scripts" / "split_per_group.py"
    print(f"[split] source CSV : {source}")
    print(f"[split] group col  : {split_col}")
    print(f"[split] output CSV : {out_csv}")
    print(f"[split] running {script.name} ...")
    sys.stdout.flush()
    subprocess.check_call([
        sys.executable, str(script),
        split_col, str(source), str(out_csv), "--group_level",
    ])
    print(f"[split] done -> {out_csv}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="DiADEM cleaned-up pipeline")
    ap.add_argument("--dataset", default="voiced", choices=["dices", "voiced", "d3code"])
    ap.add_argument("--split-mode", default="by_annotator", choices=["by_annotator", "by_item"])
    ap.add_argument(
        "--models",
        default="diadem,disco",
        help="Comma-separated: diadem,disco,dem_moe,majority_similar,llm_gemma4,llm_gpt_oss,llm_qwen8b,llm_deepseek",
    )
    ap.add_argument(
        "--steps",
        default="seed,prepare,train,eval,error_detailed,error_in_depth,compare",
        help="split,preprocess,seed,prepare,train,eval,error_detailed,error_in_depth,compare",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "d3code":
        steps_tmp = [s.strip() for s in args.steps.split(",") if s.strip() and s.strip() != "seed"]
        if "seed" in [s.strip() for s in args.steps.split(",") if s.strip()]:
            print("[warn] d3code has no legacy seed — removing 'seed'")
        if "prepare" in steps_tmp and "preprocess" not in steps_tmp:
            insert_at = 0
            for i, s in enumerate(steps_tmp):
                if s == "prepare":
                    insert_at = i
                    break
            front = []
            if "split" not in steps_tmp:
                front.append("split")
            front.append("preprocess")
            steps_tmp = steps_tmp[:insert_at] + front + steps_tmp[insert_at:]
            print(f"[warn] d3code auto-added steps: {front}")
        args.steps = ",".join(steps_tmp) if steps_tmp else (
            "split,preprocess,prepare,train,eval,error_detailed,error_in_depth,compare"
        )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    llm_models    = [m for m in models if is_llm_model(m)]
    rule_models   = [m for m in models if is_rule_baseline(m)]
    dem_moe_models= [m for m in models if _is_dem_moe(m)]
    neural_models = [m for m in models if not is_llm_model(m) and not is_rule_baseline(m)
                     and not _is_dem_moe(m)]

    print("")
    print("*" * 60)
    print("  cleanedup_diadem pipeline starting")
    print(f"  dataset    = {args.dataset}")
    print(f"  split_mode = {args.split_mode}")
    print(f"  models     = {models}")
    print(f"  neural     = {neural_models}")
    print(f"  dem_moe    = {dem_moe_models}")
    print(f"  rule_base  = {rule_models}")
    print(f"  llm_judge  = {llm_models}")
    print(f"  steps      = {steps}")
    print(f"  gpu_id     = {args.gpu_id}")
    print(f"  root       = {ROOT}")
    print("*" * 60)
    sys.stdout.flush()

    if "split" in steps:
        _banner("STEP: split")
        run_split(args.dataset, args.split_mode)

    if "seed" in steps:
        _banner("STEP: seed (copy legacy npy -> processed/)")
        seed_from_legacy(args.dataset, args.split_mode, ROOT)

    if "preprocess" in steps:
        _banner("STEP: preprocess (CSV -> embeddings + npy)")
        run_preprocess(args.dataset, args.split_mode)

    if "prepare" in steps:
        _banner("STEP: prepare (processed/ -> per-model dirs)")
        for m in models:
            print(f"[prepare] staging files for model={m} ...")
            sys.stdout.flush()
            if is_llm_model(m):
                d = prepare_llm_run_dir(args.dataset, args.split_mode, m)
            elif is_rule_baseline(m):
                d = prepare_majority_run_dir(args.dataset, args.split_mode, m)
            else:
                d = prepare_run_dir(args.dataset, args.split_mode, m)
            print(f"[prepare] {m} ready -> {d}")
            sys.stdout.flush()

    if "train" in steps:
        _banner("STEP: train (neural only; LLM / majority baselines skip)")
        for m in neural_models:
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[train] >>> starting model={m}")
            print(f"[train]     data_dir={d}")
            sys.stdout.flush()
            train_model(d, m, dataset=args.dataset, gpu_id=args.gpu_id)
            print(f"[train] <<< finished model={m}")
            sys.stdout.flush()
        for m in dem_moe_models:
            from diadem.train_dem_moe import train_model as train_dem_moe
            from diadem.registry import load_json
            from diadem.paths import CONFIGS
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[train] >>> DEM-MoE model={m}  data_dir={d}")
            sys.stdout.flush()
            model_cfg = load_json(CONFIGS / "dem_moe.json")
            train_dem_moe(data_dir=d, model_cfg=model_cfg, gpu_id=args.gpu_id)
            print(f"[train] <<< DEM-MoE finished model={m}")
            sys.stdout.flush()
        for m in rule_models:
            print(f"[train] skip {m} (demographic plurality — no training)")
            sys.stdout.flush()
        for m in llm_models:
            print(f"[train] skip {m} (zero-shot LLM judge — no training)")
            sys.stdout.flush()

    if "eval" in steps:
        _banner("STEP: eval")
        for m in neural_models:
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[eval] >>> neural model={m} dir={d}")
            sys.stdout.flush()
            run_eval(d, m, args.dataset, gpu_id=args.gpu_id)
            print(f"[eval] <<< finished model={m}")
            sys.stdout.flush()
        for m in dem_moe_models:
            from diadem.eval_dem_moe import run_eval as run_eval_dem_moe
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[eval] >>> DEM-MoE model={m} dir={d}")
            sys.stdout.flush()
            run_eval_dem_moe(d, m, args.dataset, gpu_id=args.gpu_id)
            print(f"[eval] <<< DEM-MoE finished model={m}")
            sys.stdout.flush()
        for m in rule_models:
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[eval] >>> rule baseline={m} dir={d}")
            sys.stdout.flush()
            run_majority_similar(d, m, args.dataset, split_mode=args.split_mode)
            print(f"[eval] <<< finished {m}")
            sys.stdout.flush()
        for m in llm_models:
            d = run_dir(args.dataset, args.split_mode, m)
            print(f"[eval] >>> LLM judge={m} dir={d}")
            sys.stdout.flush()
            run_llm_judge(d, m, args.dataset, gpu_id=args.gpu_id)
            print(f"[eval] <<< finished LLM={m}")
            sys.stdout.flush()

    if "error_detailed" in steps:
        _banner("STEP: error_detailed")
        for m in neural_models:
            d = run_dir(args.dataset, args.split_mode, m)
            ail = ail_path_for(d, args.dataset, "test")
            print(f"[error_detailed] >>> model={m}")
            sys.stdout.flush()
            run_error_detailed(
                d, d / "trained_model.disco", ail,
                split_name="test", gpu_id=args.gpu_id,
            )
            print(f"[error_detailed] <<< finished model={m}")
            sys.stdout.flush()
        for m in dem_moe_models:
            from diadem.eval_dem_moe import run_error_detailed as run_ed_dem_moe
            d = run_dir(args.dataset, args.split_mode, m)
            ail = ail_path_for(d, args.dataset, "test")
            print(f"[error_detailed] >>> DEM-MoE model={m}")
            sys.stdout.flush()
            run_ed_dem_moe(d, d / "dem_moe_model.pt", ail,
                           split_name="test", gpu_id=args.gpu_id)
            print(f"[error_detailed] <<< DEM-MoE finished model={m}")
            sys.stdout.flush()
        for m in rule_models:
            d = run_dir(args.dataset, args.split_mode, m)
            summary_path = d / "error_analysis" / "analysis_summary.json"
            if summary_path.exists():
                print(f"[error_detailed] skip {m} — already wrote {summary_path} during eval")
            else:
                print(f"[error_detailed] {m}: re-running majority_similar ...")
                sys.stdout.flush()
                run_majority_similar(d, m, args.dataset, split_mode=args.split_mode)
            sys.stdout.flush()
        for m in llm_models:
            d = run_dir(args.dataset, args.split_mode, m)
            summary_path = d / "error_analysis" / "analysis_summary.json"
            if summary_path.exists():
                print(f"[error_detailed] skip {m} — already wrote {summary_path} during eval")
            else:
                print(f"[error_detailed] {m}: re-running LLM judge to produce paper metrics ...")
                sys.stdout.flush()
                run_llm_judge(d, m, args.dataset, gpu_id=args.gpu_id)
            sys.stdout.flush()

    if "error_in_depth" in steps:
        _banner("STEP: error_in_depth (neural only)")
        for m in neural_models:
            d = run_dir(args.dataset, args.split_mode, m)
            ail = ail_path_for(d, args.dataset, "test")
            print(f"[error_in_depth] >>> model={m}")
            sys.stdout.flush()
            run_error_in_depth(
                d, d / "trained_model.disco", ail,
                split_name="test", gpu_id=args.gpu_id,
            )
            print(f"[error_in_depth] <<< finished model={m}")
            sys.stdout.flush()
        for m in rule_models + llm_models:
            print(f"[error_in_depth] skip {m} (uses eval/error_analysis metrics)")
            sys.stdout.flush()

    if "compare" in steps:
        _banner("STEP: compare (side-by-side table)")
        out = run_dir(args.dataset, args.split_mode, "comparison") / "results_table.md"
        print(f"[compare] writing table for models={models}")
        print(f"[compare] output -> {out}")
        sys.stdout.flush()
        write_comparison(args.dataset, args.split_mode, models, out)

    print("")
    print("*" * 60)
    print("  pipeline finished successfully")
    print("*" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
