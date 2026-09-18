#!/bin/bash
# OOD test launcher for NERF. Runs the epoch-3 checkpoint over each OOD test split
# sequentially (greedy top-1) and renames each result pickle so they don't clobber
# one another (main.py hardcodes results/<temperature>_<seed>.pickle).
# Run from repo root with the nerf env active:  bash scripts/test_ood.sh
set -e

export LD_LIBRARY_PATH=
# ---- config ----
name=nerf_bs128_uspto480k_unified                    # checkpoint namespace (--name)
save_path=/data/vu/retromech/baselines/nerf/runs     # parent dir of <name>/
checkpoint=epoch-3-loss-1.4836149771539195
temperature=0.0                                      # greedy top-1
seed=2019
n_gpu=1
port=16018

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# OOD splits to test (scaffold_B intentionally excluded). Each entry is the --prefix,
# which makes load_data read data/<prefix>_test.pickle.
prefixes="uspto480k_unified_ood_ester uspto480k_unified_ood_scaffold_A uspto480k_unified_ood_mass"

# results/ symlink -> shared results dir (idempotent)
results_dir=/data/vu/retromech/baselines/nerf/results
mkdir -p "$results_dir"
if [ ! -e results ]; then ln -s "$results_dir" results; fi

for prefix in $prefixes; do
  echo "=================================================="
  echo "OOD test: $prefix"
  echo "=================================================="
  python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py \
    --world_size $n_gpu --test \
    --vae \
    --batch_size 256 --dropout 0.1 --depth 6 --dim 256 \
    --prefix $prefix --name $name --save_path $save_path \
    --checkpoint $checkpoint \
    --temperature $temperature
  # preserve this split's result (else the next run overwrites it)
  mv "results/${temperature}_${seed}.pickle" "results/${prefix}_${temperature}_${seed}.pickle"
  echo "saved results/${prefix}_${temperature}_${seed}.pickle"
done
