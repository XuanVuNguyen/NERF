#!/bin/bash
# Build NERF cache JSONs (wraps scripts/build_nerf_cache.py), in the same format as
# the other models' caches used by plot_ood_combined_nerf.ipynb. Reads the per-temperature
# dumps <dump_dir>/<split>_pertemp.pickle (produced by topk_eval.sh with dump_dir=<dump_dir>)
# plus the retained-index alignment files, and writes <out_dir>/<model_name>_<split>.json.
# CPU only (rdkit + pyarrow, no torch/GPU) -- safe to run while training occupies the GPU.
# Run from repo root:  bash scripts/build_nerf_cache.sh
set -e

# ---- config (edit these) ----
dump_dir=/data/vu/retromech/baselines/nerf/results/nerf_bs128_uspto480k_unified_low_lr                # dir with <split>_pertemp.pickle (topk_eval.sh dump_dir)
out_dir=results/nerf_cache_low_lr     # where to write <model_name>_<split>.json
retained_dir=results            # dir with <split>_retained_hf_idx.pickle (dataset-level, run-independent)
model_name=nerf                 # 'model' label in the cache + filename prefix (e.g. nerf, nerf_low_lr)

# Example for the low-LR run's dumps -> a separate cache dir & label:
#   dump_dir=/data/vu/retromech/baselines/nerf/results/nerf_bs128_uspto480k_unified_low_lr
#   out_dir=results/nerf_cache_low_lr
#   model_name=nerf_low_lr

# rdkit + pyarrow env (NOT the nerf/torch env). Defaults to the chrimp env used to build
# the committed caches; override with: PYTHON=/path/to/python bash scripts/build_nerf_cache.sh
PYTHON="${PYTHON:-/home/vu/miniconda3/envs/chrimp/bin/python}"

"$PYTHON" scripts/build_nerf_cache.py \
  --dump-dir "$dump_dir" \
  --out-dir "$out_dir" \
  --retained-dir "$retained_dir" \
  --model-name "$model_name"
