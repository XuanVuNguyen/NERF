#!/bin/bash
# Training launcher for NERF. Run from the repo root:  bash scripts/train.sh
set -e

export LD_LIBRARY_PATH=
# ---- config (edit these) ----
name=nerf_bs128      # namespaces CKPT/<name>/ and log/<name>/ (fresh name: the old
                     # 'nerf' run diverged/collapsed — keep its checkpoints/logs separate)
prefix=data          # load_data reads data/<prefix>_<name>.pickle
n_gpu=1              # processes; model is always DDP-wrapped even for 1 GPU
ckpt_dir=/data/vu/retromech/baselines/nerf/runs     # checkpoints go to <ckpt_dir>/<name>/ ; e.g. /data/vu/.../ckpt
tb_port=6018
port=1$tb_port       # master_port for torch.distributed

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# ---- output dirs ----
# main.py uses os.mkdir (single level), so the <ckpt_dir>/<name>/ parent must exist first.
mkdir -p "$ckpt_dir" log

# ---- data symlinks ----
# preprocess.py writes data/<name>_data.pickle, but load_data expects
# data/<prefix>_<name>.pickle. Bridge the two naming conventions (idempotent).
for split in train valid test; do
  src="data/${split}_data.pickle"
  dst="data/${prefix}_${split}.pickle"
  if [ -f "$src" ] && [ ! -e "$dst" ]; then
    ln -s "${split}_data.pickle" "$dst"
  fi
done

# ---- train ----
# --save is required to write checkpoints to CKPT/<name>/. --beta is
# auto-annealed during training. (--eval is omitted: its per-epoch TensorBoard
# image logging calls rdkit Draw, which crashes on modern Pillow — see below.
# Train accuracy is still sampled every 500 steps; run full eval via test.sh.)
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py \
  --world_size $n_gpu --train \
  --vae \
  --batch_size 128 --dropout 0.1 --depth 6 --dim 256 \
  --prefix $prefix --name $name --epochs 250 \
  --save_path $ckpt_dir --save
