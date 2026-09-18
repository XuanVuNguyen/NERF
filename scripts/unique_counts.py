"""Compute the number of distinct valid predictions per reaction, per model per split,
and summarise mean/std/n -> results/unique_counts.json (consumed by the notebook table).

A "prediction" is whatever the model emits (full product set for nerf/g2s, main product
for mt/megan/maelle); we count distinct canonical valid ones per reaction. Sources:
  g2s/mt/megan : ranked `pred` list (arrow)
  maelle       : `counts` dict keys (arrow)
  nerf/nerf_low_lr : per-temperature dump rows' preds
Run in an rdkit+pyarrow env (chrimp).
"""
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

BASE = "/home/vu/repos/retromech/octavian_investigation/uspto_480k_unified"
LOWLR = "/data/vu/retromech/baselines/nerf/results/nerf_bs128_uspto480k_unified_low_lr"
NERF = "/home/vu/repos/NERF/results"
SPLITS = [("iid", "test_iid"), ("ood_ester", "test_ood_ester"), ("ood_mass", "test_ood_mass")]


def n_unique_valid(preds):
    seen = set()
    for p in preds:
        if not p:
            continue
        m = Chem.MolFromSmiles(p)
        if m is not None:
            seen.add(Chem.MolToSmiles(m))
    return len(seen)


def arrow_col(path, col):
    with pa.memory_map(path, "r") as s:
        t = pa.ipc.open_stream(s).read_all()
    return t.column(col).to_pylist()


def summarize(pred_lists, pool):
    counts = list(pool.map(n_unique_valid, pred_lists, chunksize=64))
    a = np.array(counts, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}


def main():
    pool = ProcessPoolExecutor(12)
    out = {}

    for model, sub in [("g2s", "g2s/results"), ("mt", "mt/predictions"), ("megan", "megan")]:
        out[model] = {}
        for label, d in SPLITS:
            preds = arrow_col(f"{BASE}/{sub}/{d}/data-00000-of-00001.arrow", "pred")
            out[model][label] = summarize(preds, pool)
            print(model, label, out[model][label])

    out["maelle"] = {}
    for label, d in SPLITS:
        cnts = arrow_col(f"{BASE}/maelle/inference_59450/{d}_8_steps_t_pois_06_t_cat_1_do_sampling/data-00000-of-00001.arrow", "counts")
        lists = [list(json.loads(c).keys()) for c in cnts]
        out["maelle"][label] = summarize(lists, pool)
        print("maelle", label, out["maelle"][label])

    for model, root in [("nerf", NERF), ("nerf_low_lr", LOWLR)]:
        out[model] = {}
        for label, _ in SPLITS:
            rows = pickle.load(open(f"{root}/{label}_pertemp.pickle", "rb"))["rows"]
            lists = [[p for p in r["preds"] if p] for r in rows]
            out[model][label] = summarize(lists, pool)
            print(model, label, out[model][label])

    pool.shutdown()
    json.dump(out, open(f"{NERF}/unique_counts.json", "w"), indent=1)
    print("wrote results/unique_counts.json")


if __name__ == "__main__":
    main()
