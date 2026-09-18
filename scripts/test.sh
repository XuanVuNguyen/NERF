#!/bin/bash
# Test / top-k sampling launcher for NERF. Run from the repo root:  bash scripts/test.sh
set -e

export LD_LIBRARY_PATH=
# ---- config (edit these) ----
name=nerf_bs128_uspto480k_unified   # must match the trained model's --name (loads from <save_path>/<name>/)
prefix=uspto480k_unified            # load_data reads data/<prefix>_<split>.pickle
save_path=/data/vu/retromech/baselines/nerf/runs   # parent dir of <name>/ (matches train.sh ckpt_dir)
n_gpu=1
tb_port=6018
port=1$tb_port

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Checkpoints to load. Passing several averages their weights (loss had
# plateaued, so averaging the last few epochs is a cheap, effective trick).
checkpoints="epoch-3-loss-1.4836149771539195"

# Top-k sampling protocol: temperature = 0.7 * 1.3**n. 0.0 is greedy (top-1).
# Pass multiple values for a sweep; each writes results/<temperature>_<seed>.pickle.
# Quick greedy top-1 check of the pre-explosion epoch-3 checkpoint.
temperatures="0.0"

# Where to write result pickles. main.py hardcodes the relative path "results/",
# so we point that at results_dir via a symlink (keeps main.py untouched).
results_dir=/data/vu/retromech/baselines/nerf/results   # e.g. /data/experiments/nerf_results

# ---- output dir + data symlink ----
mkdir -p "$results_dir"
if [ "$results_dir" != "results" ] && [ ! -e results ]; then ln -s "$results_dir" results; fi
src="data/test_data.pickle"; dst="data/${prefix}_test.pickle"
if [ -f "$src" ] && [ ! -e "$dst" ]; then ln -s "test_data.pickle" "$dst"; fi

# ---- test ----
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py \
  --world_size $n_gpu --test \
  --vae \
  --batch_size 256 --dropout 0.1 --depth 6 --dim 256 \
  --prefix $prefix --name $name --save_path $save_path \
  --checkpoint $checkpoints \
  --temperature $temperatures
