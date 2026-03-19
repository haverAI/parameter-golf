"""Hyperparameter sweep runner for Parameter Golf.

Generates and runs experiments systematically on H100s.
Each run takes ~10 min, so plan budget accordingly.

Usage:
    # Dry run — just print configs
    python sweep.py --dry-run

    # Run on 1xH100 (quick iteration)
    python sweep.py --gpus 1 --max-runs 10

    # Full sweep on 8xH100 (competition mode)
    python sweep.py --gpus 8 --max-runs 50
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# SWEEP CONFIGURATIONS
# ============================================================
# Each dict is a hyperparameter axis to sweep.
# We do a grid search over the most impactful params first.

# TIER 1: Architecture (highest impact)
ARCH_CONFIGS = [
    # Baseline
    {"name": "baseline", "NUM_LAYERS": 9, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 1},
    # Depth recurrence variants
    {"name": "7x2", "NUM_LAYERS": 7, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2},
    {"name": "6x3", "NUM_LAYERS": 6, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 3},
    {"name": "5x3", "NUM_LAYERS": 5, "MODEL_DIM": 576, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 3},
    {"name": "4x4", "NUM_LAYERS": 4, "MODEL_DIM": 640, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 4},
    # Wider models (no recurrence)
    {"name": "wide640", "NUM_LAYERS": 6, "MODEL_DIM": 640, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 1},
    # Budget-maxing recurrence configs (use ALL 16MB)
    {"name": "8x2_512", "NUM_LAYERS": 8, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2},
    {"name": "7x2_576", "NUM_LAYERS": 7, "MODEL_DIM": 576, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2},
    {"name": "6x2_640", "NUM_LAYERS": 6, "MODEL_DIM": 640, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2},
    {"name": "5x3_576", "NUM_LAYERS": 5, "MODEL_DIM": 576, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 3},
    # MLP multiplier variants
    {"name": "9x1_512_mlp3", "NUM_LAYERS": 9, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 1, "MLP_MULT": 3},
    {"name": "6x2_512_mlp3", "NUM_LAYERS": 6, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2, "MLP_MULT": 3},
    # SwiGLU variants (3 matrices instead of 2, but 2/3 hidden for same param count)
    {"name": "7x2_swiglu", "NUM_LAYERS": 7, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2, "USE_SWIGLU": 1},
    {"name": "8x2_swiglu", "NUM_LAYERS": 8, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 2, "USE_SWIGLU": 1},
    {"name": "baseline_swiglu", "NUM_LAYERS": 9, "MODEL_DIM": 512, "NUM_HEADS": 8, "NUM_KV_HEADS": 4, "NUM_RECURRENCE": 1, "USE_SWIGLU": 1},
]

# QAT sweep (run after finding best architecture)
QAT_CONFIGS = [
    {"name": "qat_off", "QAT_START_FRAC": 0.0},
    {"name": "qat_50", "QAT_START_FRAC": 0.5},
    {"name": "qat_60", "QAT_START_FRAC": 0.6},
    {"name": "qat_70", "QAT_START_FRAC": 0.7},
]

# TIER 2: Learning rates (Muon + Adam)
LR_CONFIGS = [
    # Baseline LRs
    {"name": "lr_base", "MATRIX_LR": 0.04, "EMBED_LR": 0.6, "TIED_EMBED_LR": 0.05},
    # Higher Muon LR
    {"name": "lr_high", "MATRIX_LR": 0.06, "EMBED_LR": 0.8, "TIED_EMBED_LR": 0.07},
    # Lower Muon LR
    {"name": "lr_low", "MATRIX_LR": 0.025, "EMBED_LR": 0.4, "TIED_EMBED_LR": 0.035},
    # Much higher (aggressive)
    {"name": "lr_aggro", "MATRIX_LR": 0.08, "EMBED_LR": 1.0, "TIED_EMBED_LR": 0.1},
]

# TIER 3: Training recipe
RECIPE_CONFIGS = [
    # Baseline recipe
    {"name": "recipe_base", "WARMUP_STEPS": 20, "WARMDOWN_ITERS": 1200, "MUON_MOMENTUM": 0.95},
    # Longer warmup (helps deep models)
    {"name": "recipe_warmup", "WARMUP_STEPS": 100, "WARMDOWN_ITERS": 1200, "MUON_MOMENTUM": 0.95},
    # Longer warmdown
    {"name": "recipe_warmdown", "WARMUP_STEPS": 20, "WARMDOWN_ITERS": 2000, "MUON_MOMENTUM": 0.95},
    # Lower momentum
    {"name": "recipe_lowmom", "WARMUP_STEPS": 20, "WARMDOWN_ITERS": 1200, "MUON_MOMENTUM": 0.90},
    # Higher momentum
    {"name": "recipe_himom", "WARMUP_STEPS": 20, "WARMDOWN_ITERS": 1200, "MUON_MOMENTUM": 0.975},
]

# TIER 4: Other knobs
OTHER_CONFIGS = [
    {"name": "other_base", "LOGIT_SOFTCAP": 30.0, "QK_GAIN_INIT": 1.5, "ROPE_BASE": 10000.0},
    {"name": "other_softcap20", "LOGIT_SOFTCAP": 20.0, "QK_GAIN_INIT": 1.5, "ROPE_BASE": 10000.0},
    {"name": "other_softcap50", "LOGIT_SOFTCAP": 50.0, "QK_GAIN_INIT": 1.5, "ROPE_BASE": 10000.0},
    {"name": "other_qkgain2", "LOGIT_SOFTCAP": 30.0, "QK_GAIN_INIT": 2.0, "ROPE_BASE": 10000.0},
]


def estimate_params(config):
    """Quick param count estimate."""
    dim = config.get("MODEL_DIM", 512)
    layers = config.get("NUM_LAYERS", 9)
    heads = config.get("NUM_HEADS", 8)
    kv_heads = config.get("NUM_KV_HEADS", 4)
    vocab = config.get("VOCAB_SIZE", 1024)
    mlp_mult = config.get("MLP_MULT", 2)
    
    head_dim = dim // heads
    kv_dim = head_dim * kv_heads
    
    # Per block
    attn_params = dim * dim + dim * kv_dim * 2 + dim * dim + dim  # Q, K, V, O + bias approx
    mlp_hidden = dim * mlp_mult
    mlp_params = dim * mlp_hidden + mlp_hidden * dim  # up + down
    block_params = attn_params + mlp_params
    
    # Total
    embed_params = vocab * dim  # tied
    total = layers * block_params + embed_params
    
    # Estimate compressed size (int8 + zlib ≈ 93% of int8)
    compressed = int(total * 0.93)
    
    return total, compressed


def generate_sweep_plan(tiers="all", max_runs=None):
    """Generate experiment configs."""
    experiments = []
    
    if tiers == "arch":
        # Just architecture sweep with baseline LR/recipe
        for arch in ARCH_CONFIGS:
            config = {**arch, **LR_CONFIGS[0], **OTHER_CONFIGS[0]}
            config["run_name"] = f"arch_{arch['name']}"
            experiments.append(config)
    
    elif tiers == "lr":
        # LR sweep with best architecture (run after arch sweep)
        best_arch = ARCH_CONFIGS[1]  # Default to 7x2, update after arch results
        for lr in LR_CONFIGS:
            config = {**best_arch, **lr, **OTHER_CONFIGS[0]}
            config["run_name"] = f"lr_{lr['name']}"
            experiments.append(config)
    
    elif tiers == "recipe":
        # Recipe sweep with best arch + LR
        best_arch = ARCH_CONFIGS[1]
        best_lr = LR_CONFIGS[0]
        for recipe in RECIPE_CONFIGS:
            config = {**best_arch, **best_lr, **recipe, **OTHER_CONFIGS[0]}
            config["run_name"] = f"recipe_{recipe['name']}"
            experiments.append(config)
    
    elif tiers == "all":
        # Full sweep: arch × lr (most impactful axes)
        for arch in ARCH_CONFIGS:
            for lr in LR_CONFIGS[:2]:  # Just base and high LR
                config = {**arch, **lr, **RECIPE_CONFIGS[0], **OTHER_CONFIGS[0]}
                config["run_name"] = f"{arch['name']}_{lr['name']}"
                experiments.append(config)
    
    elif tiers == "fine":
        # Fine-grained sweep around best config
        # Fill in after initial results
        pass
    
    # Filter by size
    valid = []
    for exp in experiments:
        params, compressed = estimate_params(exp)
        exp["est_params"] = params
        exp["est_compressed"] = compressed
        exp["fits_16mb"] = compressed <= 16_000_000
        if exp["fits_16mb"]:
            valid.append(exp)
        else:
            print(f"SKIP {exp['run_name']}: {compressed:,} bytes (over 16MB)")
    
    if max_runs and len(valid) > max_runs:
        valid = valid[:max_runs]
    
    return valid


def run_experiment(config, gpus=1, script_path="train_gpt.py", dry_run=False):
    """Run a single training experiment."""
    run_name = config.pop("run_name")
    est_params = config.pop("est_params", 0)
    est_compressed = config.pop("est_compressed", 0)
    fits = config.pop("fits_16mb", True)
    
    # Build env vars
    env = os.environ.copy()
    env["RUN_ID"] = run_name
    
    # Remove non-env keys
    skip_keys = {"name"}
    for key, value in config.items():
        if key not in skip_keys:
            env[key] = str(value)
    
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}",
        script_path
    ]
    
    print(f"\n{'='*60}")
    print(f"RUN: {run_name}")
    print(f"  Params: ~{est_params:,} | Compressed: ~{est_compressed:,} bytes")
    print(f"  Config: {json.dumps({k:v for k,v in config.items() if k != 'name'}, indent=2)}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        return {"run_name": run_name, "status": "dry_run", "config": config}
    
    # Run training
    log_path = Path(f"sweep_logs/{run_name}.log")
    log_path.parent.mkdir(exist_ok=True)
    
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            timeout=900  # 15 min timeout (10 min training + overhead)
        )
    
    # Parse results
    result = {
        "run_name": run_name,
        "status": "success" if proc.returncode == 0 else "failed",
        "config": config,
        "est_params": est_params,
    }
    
    # Extract val_bpb from log
    try:
        log_text = log_path.read_text()
        for line in log_text.split("\n"):
            if "val_bpb" in line:
                # Parse the last val_bpb line
                parts = line.split()
                for part in parts:
                    if part.startswith("val_bpb:") or part.startswith("val_bpb="):
                        result["val_bpb"] = float(part.split(":")[-1].split("=")[-1])
            if "bytes_total" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("bytes_total:") or part.startswith("bytes_total="):
                        result["bytes_total"] = int(part.split(":")[-1].split("=")[-1])
    except Exception as e:
        result["parse_error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Hyperparameter Sweep")
    parser.add_argument("--tiers", default="arch", choices=["arch", "lr", "recipe", "all", "fine"],
                       help="Which tier of hyperparams to sweep")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs")
    parser.add_argument("--dry-run", action="store_true", help="Just print configs, don't run")
    parser.add_argument("--script", default="train_gpt.py", help="Training script path")
    args = parser.parse_args()
    
    experiments = generate_sweep_plan(args.tiers, args.max_runs)
    
    print(f"\n{'='*60}")
    print(f"PARAMETER GOLF SWEEP")
    print(f"  Tier: {args.tiers}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Est. time: {len(experiments) * 12} minutes ({len(experiments) * 12 / 60:.1f} hours)")
    print(f"  Est. cost: ~${len(experiments) * 12 / 60 * 20:.0f} (at $20/hr for 8xH100)")
    print(f"{'='*60}")
    
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}]")
        result = run_experiment(exp.copy(), gpus=args.gpus, script_path=args.script, dry_run=args.dry_run)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    
    # Sort by val_bpb
    scored = [r for r in results if "val_bpb" in r]
    scored.sort(key=lambda r: r["val_bpb"])
    
    if scored:
        print(f"\n{'Run':<30} {'BPB':>10} {'Bytes':>12} {'Params':>12}")
        print("-" * 66)
        for r in scored:
            print(f"{r['run_name']:<30} {r['val_bpb']:>10.6f} {r.get('bytes_total', 'N/A'):>12} {r.get('est_params', 'N/A'):>12}")
        
        best = scored[0]
        print(f"\nBEST: {best['run_name']} — {best['val_bpb']:.6f} BPB")
        print(f"Improvement over baseline: {1.2244 - best['val_bpb']:.6f} BPB")
    else:
        for r in results:
            print(f"  {r['run_name']}: {r['status']}")
    
    # Save results
    results_path = Path(f"sweep_logs/sweep_{args.tiers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
