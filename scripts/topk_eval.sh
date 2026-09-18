#!/bin/bash
# Top-k accuracy launcher for NERF (wraps scripts/topk_eval.py).
# Runs the README multi-temperature sampling protocol: greedy T=0 (rank 1) plus
# the ladder T = 0.7*1.3**n, accumulating distinct valid candidates per reaction
# to score top-1/3/5/10 by main-product containment.
# Run from repo root WITH THE NERF ENV ACTIVE:  conda activate nerf && bash scripts/topk_eval.sh
set -e

export LD_LIBRARY_PATH=
# ---- config (edit these) ----
name=nerf_bs128_uspto480k_unified                    # checkpoint namespace (--name used at train time)
save_path=/data/vu/retromech/baselines/nerf/runs     # parent dir of <name>/ (matches train.sh ckpt_dir)
checkpoint=epoch-3-loss-1.4836149771539195           # checkpoint file inside <save_path>/<name>/
splits="iid ood_ester ood_mass"                      # subset of: iid ood_ester ood_mass
n_ladder=11                                          # ladder temps T=0.7*1.3**n for n=0..n_ladder-1 (+ greedy T=0)
batch_size=256
topk="1 3 5 10"
# Optional: DIRECTORY to save per-reaction per-temperature predictions for EVERY split,
# written as <dump_dir>/<split>_pertemp.pickle. Leave empty to skip. e.g. dump_dir=results
dump_dir=

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# ---- run ----
extra=""
[ -n "$dump_dir" ] && extra="--dump $dump_dir"
python scripts/topk_eval.py \
  --checkpoint "$save_path/$name/$checkpoint" \
  --splits $splits \
  --n_ladder $n_ladder \
  --batch_size $batch_size \
  --topk $topk \
  $extra
