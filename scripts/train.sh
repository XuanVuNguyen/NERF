#!/bin/bash
# Training launcher for NERF. Run from the repo root:  bash scripts/train.sh
set -e

export LD_LIBRARY_PATH=
# ---- config (edit these) ----
name=nerf_bs128_uspto480k_no_reactant_mask      # namespaces CKPT/<name>/ and log/<name>/ (fresh name: the old
                     # 'nerf' run diverged/collapsed — keep its checkpoints/logs separate)
prefix=uspto480k_original   # data/ subdir; load_data picks data/<prefix>/*<split>.pickle
no_reactant_flag="--no_reactant_flag"  # leakage-free: drop the reactant/spectator tag. Set to
                     # "" to keep the legacy flag. Must match test.sh/test_ood.sh at eval time.
n_gpu=1              # processes; model is always DDP-wrapped even for 1 GPU
ckpt_dir=/data/vu/retromech/baselines/nerf/runs     # checkpoints go to <ckpt_dir>/<name>/ ; e.g. /data/vu/.../ckpt
tb_port=6018
port=1$tb_port       # master_port for torch.distributed

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# ---- output dirs ----
# main.py uses os.mkdir (single level), so the <ckpt_dir>/<name>/ parent must exist first.
mkdir -p "$ckpt_dir" log

# ---- data check ----
# load_data picks the single data/<prefix>/*<split>.pickle. Fail early with a clear
# message if preprocessing hasn't been run for this prefix.
for split in train valid test; do
  if ! ls data/${prefix}/*${split}.pickle >/dev/null 2>&1; then
    echo "ERROR: no *${split}.pickle in data/${prefix}/. Run preprocess.py first (in the nerf env)." >&2
    exit 1
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
  $no_reactant_flag \
  --lr 2e-4 \
  --batch_size 128 --dropout 0.1 --depth 6 --dim 256 \
  --prefix $prefix --name $name --epochs 250 \
  --save_path $ckpt_dir --save
