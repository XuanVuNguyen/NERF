"""Accuracy + McNemar bolding for the notebook tables, computed on ONE common,
idx-aligned, de-duplicated reaction set shared by all models — so displayed numbers
and the significance test are mutually consistent.

Why: `source_idx` is not unique and MEGAN's predictions contain duplicate / partially
different rows, so per-model full-set numbers aren't directly comparable. We align by
HF row index (`idx`; NERF via its retained-HF-idx map), de-duplicate (keep first),
intersect across models, and compute everything on that common set.

Bold = best model per column/bin + any model not significantly worse (McNemar, alpha=0.05).
Writes results/mcnemar_bold.json (accuracies included; the notebook displays them).
Run in an rdkit/pyarrow/scipy env (chrimp).
"""
import glob
import json
import pickle

import numpy as np
import pyarrow as pa
from scipy.stats import chi2, binomtest

BASE = "/home/vu/repos/retromech/octavian_investigation/uspto_480k_unified"
CACHE_DIRS = [f"{BASE}/.cache", "results/nerf_cache_low_lr"]
HF = "/data/share/vu/retromech/uspto_480k_unified"
SPLIT_DIR = {"iid": "test_iid", "ood_ester": "test_ood_ester", "ood_mass": "test_ood_mass"}
RAW = {"g2s": "g2s/results/{d}", "mt": "mt/predictions/{d}", "megan": "megan/{d}",
       "maelle": "maelle/inference_59450/{d}_8_steps_t_pois_06_t_cat_1_do_sampling"}
RETAINED = {"iid": "results/iid_retained_hf_idx.pickle", "ood_ester": None,
            "ood_mass": "results/ood_mass_retained_hf_idx.pickle"}
MODELS = ["maelle", "g2s", "mt", "megan", "nerf"]
KS = [1, 3, 5, 10]
ALPHA = 0.05
MW_BINS = [(-np.inf, 475), (475, 500), (500, 525), (525, 550), (550, 575), (575, 600),
           (600, 625), (625, 650), (650, 675), (675, 700), (700, 725), (725, np.inf)]


def arrow(path, cols):
    with pa.memory_map(path, "r") as s:
        t = pa.ipc.open_stream(s).read_all()
    return {c: t.column(c).to_pylist() for c in cols}


cache = {}
for cdir in CACHE_DIRS:
    for p in glob.glob(cdir + "/*.json"):
        d = json.load(open(p))
        dp, m = d["dataset_path"], d["model"]
        split = ("iid" if "test_iid" in dp else "ood_ester" if "test_ood_ester" in dp
                 else "ood_mass" if "test_ood_mass" in dp else None)
        if split is None:
            continue
        cache[(m, split)] = np.array([r if r is not None else np.inf for r in d["first_rank"]], float)

hf_mw = {s: np.array(arrow(f"{HF}/{SPLIT_DIR[s]}/data-00000-of-00001.arrow", ["product_mw"])["product_mw"], float)
         for s in SPLIT_DIR}


def idx_for(model, split):
    if model == "nerf":
        ret = RETAINED[split]
        return pickle.load(open(ret, "rb")) if ret else list(range(len(cache[(model, split)])))
    return arrow(f"{BASE}/{RAW[model].format(d=SPLIT_DIR[split])}/data-00000-of-00001.arrow", ["idx"])["idx"]


def key2fr(model, split):
    d = {}
    for i, r in zip(idx_for(model, split), cache[(model, split)]):
        if i not in d:                       # de-dup MEGAN's duplicate rows (keep first)
            d[i] = r
    return d


def mcnemar_p(a, b):
    a, b = a.astype(bool), b.astype(bool)
    b01, c01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = b01 + c01
    if n == 0:
        return 1.0
    if n < 25:
        return float(binomtest(min(b01, c01), n, 0.5).pvalue)
    stat = (abs(b01 - c01) - 1) ** 2 / n
    return float(chi2.sf(stat, 1))


def analyse(fr, idxs, k):
    """acc per model + best + McNemar bold set on the reaction indices `idxs`."""
    idxs = list(idxs)
    correct = {m: np.array([fr[m][i] <= k for i in idxs]) for m in fr}
    acc = {m: float(correct[m].mean()) for m in fr}
    best = max(acc, key=acc.get)
    bold = [best] + [m for m in fr if m != best and mcnemar_p(correct[best], correct[m]) >= ALPHA]
    return {"acc": {m: round(acc[m], 4) for m in acc}, "best": best, "bold": bold}


result = {}
frs, commons = {}, {}
for split in SPLIT_DIR:
    fr = {m: key2fr(m, split) for m in MODELS if (m, split) in cache}
    common = set.intersection(*[set(fr[m]) for m in fr])
    frs[split], commons[split] = fr, common
    result[split] = {"n_common": len(common)}
    for k in KS:
        result[split][str(k)] = analyse(fr, common, k)

# MW-by-bin (OOD mass, top-1), same common set, binned by HF product_mw
split = "ood_mass"
fr, common, mw = frs[split], commons[split], hf_mw[split]
labels = []
for lo, hi in MW_BINS:
    labels.append(f"≤{int(hi)}" if np.isneginf(lo) else f">{int(lo)}" if np.isposinf(hi) else str(int(hi)))
result["mw"] = {"n_common": len(common), "labels": labels}
for bi, (lo, hi) in enumerate(MW_BINS):
    bin_idx = [i for i in common if lo < mw[i] <= hi]
    r = analyse(fr, bin_idx, 1) if bin_idx else {"acc": {}, "best": None, "bold": []}
    r["count"] = len(bin_idx)
    result["mw"][str(bi)] = r

json.dump(result, open("results/mcnemar_bold.json", "w"), indent=1)
for split in SPLIT_DIR:
    print(f"\n{split} (n_common={result[split]['n_common']}):")
    for k in KS:
        r = result[split][str(k)]
        print(f"  top{k}: best={r['best']}  bold={r['bold']}  acc={r['acc']}")
print(f"\nMW bins (ood_mass top-1, n_common={result['mw']['n_common']}):")
for bi in range(len(MW_BINS)):
    r = result["mw"][str(bi)]
    print(f"  {result['mw']['labels'][bi]:>5} (n={r['count']:5d}): best={r['best']}  bold={r['bold']}")
