#!/bin/bash
# Test / top-k sampling launcher for NERF. Run from the repo root:  bash scripts/test.sh
set -e

export LD_LIBRARY_PATH=
# ---- config (edit these) ----
name=nerf            # must match the trained model's --name (loads from CKPT/<name>/)
prefix=data          # load_data reads data/<prefix>_<name>.pickle
n_gpu=1
tb_port=6018
port=1$tb_port

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Checkpoints to load. Passing several averages their weights (loss had
# plateaued, so averaging the last few epochs is a cheap, effective trick).
checkpoints="epoch-96-loss-25.889057971519595 \
             epoch-97-loss-25.88925590102511 \
             epoch-98-loss-25.889436682333187 \
             epoch-99-loss-25.88953116231997"

# Top-k sampling protocol: temperature = 0.7 * 1.3**n. 0.0 is greedy (top-1).
# Pass multiple values for a sweep; each writes results/<temperature>_<seed>.pickle.
temperatures="0.0 0.7 0.91 1.183"

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
  --prefix $prefix --name $name \
  --checkpoint $checkpoints \
  --temperature $temperatures
