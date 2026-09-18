"""Build per-split cache JSONs for NERF in the same format as the other models'
caches in retromech/octavian_investigation/.../.cache (model, dataset_path,
num_samples, first_rank, mw_values, top1_correct).

first_rank = 1-based rank of the first distinct valid candidate (temperature-ladder
order) whose fragment set contains the TRUE main product (ref_product / product_main),
or null if never. This is the matched, main-product-containment criterion — directly
comparable to the other models' first_rank. mw_values = HF product_mw of the target.

Run in an env with rdkit + pyarrow (e.g. chrimp):
    python scripts/build_nerf_cache.py \
        --dump-dir results --out-dir results/nerf_cache --model-name nerf
"""
import argparse
import json
import os
import pickle

import pyarrow as pa
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

HF = "/data/share/vu/retromech/uspto_480k_unified"

# split -> (per-temp dump basename, retained-hf-idx basename or None for identity, HF split dir)
SPLITS = {
    "iid":       ("iid_pertemp.pickle",       "iid_retained_hf_idx.pickle",       "test_iid"),
    "ood_ester": ("ood_ester_pertemp.pickle", None,                               "test_ood_ester"),
    "ood_mass":  ("ood_mass_pertemp.pickle",  "ood_mass_retained_hf_idx.pickle",  "test_ood_mass"),
}


def canon(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m else None


def build(split, dump_path, retained_path, hf_split, out_dir, model_name):
    rows = pickle.load(open(dump_path, "rb"))["rows"]
    retained = (pickle.load(open(retained_path, "rb")) if retained_path
                else list(range(len(rows))))          # identity when 0 rows dropped
    assert len(rows) == len(retained), (split, len(rows), len(retained))

    with pa.memory_map(f"{HF}/{hf_split}/data-00000-of-00001.arrow", "r") as s:
        t = pa.ipc.open_stream(s).read_all()
    hf_main = t.column("product_main").to_pylist()
    hf_mw = t.column("product_mw").to_pylist()

    first_rank, mw_values, top1_correct = [], [], []
    for row, hf_i in zip(rows, retained):
        target = canon(hf_main[hf_i])
        mw_values.append(float(hf_mw[hf_i]))
        # temperature-ordered distinct valid candidate fragment-sets
        seen, rank_hit = set(), None
        r = 0
        for p in row["preds"]:
            if p is None:
                continue
            fs = frozenset(filter(None, (canon(f) for f in p.split("."))))
            if not fs or fs in seen:
                continue
            seen.add(fs)
            r += 1
            if target is not None and target in fs:
                rank_hit = r
                break
        first_rank.append(rank_hit)
        top1_correct.append(1 if rank_hit == 1 else 0)

    cache = {
        "model": model_name,
        "dataset_path": f"{out_dir}/{hf_split}",   # 'test_iid'/'test_ood_ester'/'test_ood_mass' substring for classify
        "align_key": None,
        "num_samples": len(rows),
        "first_rank": first_rank,
        "mw_values": mw_values,
        "top1_correct": top1_correct,
    }
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{model_name}_{split}.json"
    json.dump(cache, open(out, "w"))
    hit = sum(1 for x in first_rank if x == 1)
    print(f"{split}: n={len(rows)}  top1={hit/len(rows):.4f}  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", default="results",
                    help="dir containing <split>_pertemp.pickle (topk_eval.sh dump_dir)")
    ap.add_argument("--out-dir", default="results/nerf_cache",
                    help="dir to write <model_name>_<split>.json cache files")
    ap.add_argument("--retained-dir", default="results",
                    help="dir containing <split>_retained_hf_idx.pickle (dataset-level, run-independent)")
    ap.add_argument("--model-name", default="nerf",
                    help="'model' label in the cache + cache filename prefix (e.g. nerf, nerf_low_lr)")
    args = ap.parse_args()

    for split, (dump_base, ret_base, hf_split) in SPLITS.items():
        dump_path = os.path.join(args.dump_dir, dump_base)
        if not os.path.exists(dump_path):
            print(f"{split}: SKIP (missing {dump_path})")
            continue
        retained_path = os.path.join(args.retained_dir, ret_base) if ret_base else None
        build(split, dump_path, retained_path, hf_split, args.out_dir, args.model_name)


if __name__ == "__main__":
    main()
