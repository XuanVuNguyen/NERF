#!/bin/bash
# Create the `nerf` conda environment used to run this repo.
# Reconstructed from the provenance of the working env at
# /home/vu/miniconda3/envs/nerf (see `conda list -n nerf`).
#
# Pins (must match for reproducibility — see README.md):
#   python 3.7.6-ish (3.7.x), numpy 1.18.5, torch 1.8.1, rdkit 2018.03.4
#
# Usage:  bash scripts/install_env.sh
set -euo pipefail

ENV_NAME=nerf

# ── locate conda ─────────────────────────────────────────────────────
# `conda activate` needs the shell hook; sourcing conda.sh provides it in
# non-interactive shells (login nodes, SLURM, CI).
if [ -z "${CONDA_EXE:-}" ]; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH." >&2; exit 1; }

# ── (re)create the env ───────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}\s|/${ENV_NAME}\$"; then
  echo "Env '${ENV_NAME}' already exists. Remove it first with:"
  echo "  conda env remove -n ${ENV_NAME}"
  exit 1
fi

# python + rdkit from conda-forge (rdkit 2018.03.4 only builds against py37).
conda create -y -n "${ENV_NAME}" -c conda-forge \
  python=3.7 \
  rdkit=2018.03.4

# pytorch 1.8.1 with the CUDA 11.1 build from the pytorch channel.
# (For a CPU-only box, drop cudatoolkit and add `cpuonly`.)
conda install -y -n "${ENV_NAME}" -c pytorch -c conda-forge \
  pytorch=1.8.1 \
  cudatoolkit=11.1

# numpy was installed via pip in the reference env.
conda run -n "${ENV_NAME}" pip install "numpy==1.18.5"

# ── verify ───────────────────────────────────────────────────────────
# LD_LIBRARY_PATH is cleared here for the same reason train.sh does it: a
# stray system CUDA on the path makes `import torch` fail with
#   libcublas.so.11: undefined symbol: free_gemm_select
echo "── verifying ─────────────────────────────────────────"
env LD_LIBRARY_PATH= conda run -n "${ENV_NAME}" python -c \
  "import sys,numpy,torch,rdkit; \
   print('python', sys.version.split()[0]); \
   print('numpy ', numpy.__version__); \
   print('torch ', torch.__version__, 'cuda', torch.version.cuda); \
   print('rdkit ', rdkit.__version__)"

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "(remember to 'export LD_LIBRARY_PATH=' before running main.py)"
