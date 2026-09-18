"""Build a CLEAN MEGAN cache directly from the raw predictions, bypassing the
octavian arrow export bug (which duplicated/dropped rows on the large splits).

Raw source: /data/share/vu/retromech/uspto_unified_v3_predictions/megan/
  pred_test_<split>_10_16.txt : per reaction, a header line
      "<megan_source_idx> <reactant> <ref_product> <mapped_state>"
      followed by ranked candidate lines "<rank> <mapped_reactant> <PRODUCT> {atoms} (edits)"
  idx_map_test_<split>.csv    : megan_source_idx -> ds_row_idx (HF row); blank when unmapped

Alignment: use idx_map's ds_row_idx; for blanks, fall back to canonical-reactant match
against the HF split. first_rank = 1-based RAW rank of the first candidate whose products
contain the true product_main (matches MEGAN's own rescore comparator). Missing-as-wrong
is NOT applied — covered reactions only (num_samples reported).

Writes results/megan_cache_clean/megan_<split>.json in the notebook cache format.
Run in rdkit+pyarrow env (chrimp):  python scripts/build_megan_clean_cache.py --split ood_ester
"""
import argparse
import csv
import json
import os
import re

import numpy as np
import pyarrow as pa
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

RAW = "/data/share/vu/retromech/uspto_unified_v3_predictions/megan"
HF = "/data/share/vu/retromech/uspto_480k_unified"
OUT = "/home/vu/repos/NERF/results/megan_cache_clean"
MAPS = re.compile(r":\d+")
# our split label -> (HF dir / raw-file infix, pred-file token)
SPLITS = {"iid": "test_iid", "ood_ester": "test_ood_ester", "ood_mass": "test_ood_mass"}


def canon(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


def canon_nomap(s):
    m = Chem.MolFromSmiles(MAPS.sub("", s)) if s else None
    return Chem.MolToSmiles(m) if m else None


def arrow(path, cols):
    with pa.memory_map(path, "r") as s:
        t = pa.ipc.open_stream(s).read_all()
    return {c: t.column(c).to_pylist() for c in cols}


def build(split):
    d = SPLITS[split]
    # idx_map
    rows = list(csv.DictReader(open(f"{RAW}/idx_map_{d}.csv")))
    msi_set = set(int(r["megan_source_idx"]) for r in rows)
    msi2ds = {int(r["megan_source_idx"]): (int(r["ds_row_idx"]) if r["ds_row_idx"] not in ("", None) else None)
              for r in rows}

    # parse pred file: msi -> [reactant_header, ranked candidate products]
    blocks = {}
    cur = None
    for line in open(f"{RAW}/pred_{d}_10_16.txt"):
        tok = line.split()
        if not tok:
            continue
        try:
            first = int(tok[0])
        except ValueError:
            continue
        if first in msi_set:
            cur = first
            blocks[cur] = [tok[1] if len(tok) > 1 else "", []]
        elif cur is not None and len(tok) >= 3:
            blocks[cur][1].append(tok[2])

    # HF
    hf = arrow(f"{HF}/{d}/data-00000-of-00001.arrow", ["product_main", "product_mw", "reactant"])
    hf_main = [canon(x) for x in hf["product_main"]]
    hf_mw = hf["product_mw"]
    react2row = {}
    for j, rc in enumerate(hf["reactant"]):
        react2row.setdefault(canon_nomap(rc), j)

    per_ds = {}   # ds_row_idx -> (first_rank, mw, top1)
    recovered = unresolved = 0
    for msi, (react, cands) in blocks.items():
        ds = msi2ds.get(msi)
        if ds is None:
            ds = react2row.get(canon_nomap(react))
            if ds is None:
                unresolved += 1
                continue
            recovered += 1
        if ds in per_ds:
            continue
        target = hf_main[ds]
        rank = None
        for r, c in enumerate(cands, start=1):          # RAW rank order (matches rescore)
            frags = frozenset(filter(None, (canon(f) for f in c.split("."))))
            if target is not None and target in frags:
                rank = r
                break
        per_ds[ds] = (rank, float(hf_mw[ds]), 1 if rank == 1 else 0)

    ds_sorted = sorted(per_ds)
    first_rank = [per_ds[i][0] for i in ds_sorted]
    mw_values = [per_ds[i][1] for i in ds_sorted]
    top1 = [per_ds[i][2] for i in ds_sorted]
    cache = {"model": "megan", "dataset_path": f"{OUT}/{d}", "align_key": None,
             "num_samples": len(ds_sorted), "first_rank": first_rank,
             "mw_values": mw_values, "top1_correct": top1}
    os.makedirs(OUT, exist_ok=True)
    json.dump(cache, open(f"{OUT}/megan_{split}.json", "w"))

    n = len(ds_sorted)
    fr = np.array([r if r is not None else np.inf for r in first_rank], float)
    hf_n = len(hf["product_main"])
    print(f"{split}: HF={hf_n}  covered={n} (mapped+recovered; fallback_recovered={recovered}, unresolved={unresolved})")
    print("  cumulative top-k: " + "  ".join(f"top{k}={float(np.mean(fr <= k)):.4f}" for k in (1, 2, 3, 5, 10)))
    print(f"  -> {OUT}/megan_{split}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default=None, help="one split; default = all")
    args = ap.parse_args()
    for sp in ([args.split] if args.split else list(SPLITS)):
        build(sp)
