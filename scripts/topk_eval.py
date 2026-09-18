"""Top-k accuracy for NERF via the README sampling protocol.

Nothing in the repo computes top-k, and the result pickles from main.py don't store
ground truth, so this is a standalone evaluator. It loads a checkpoint once and, per
split, derives the ground-truth product (result2mol on the target features) and samples
the model across the temperature ladder T = 0.7*1.3**n (plus greedy T=0 as rank 1).
Distinct valid predictions are accumulated per reaction in ascending-temperature order;
a reaction is correct@k if any of its first k distinct valid candidates contains all
ground-truth fragments (same containment test as main.py:test).

Run from repo root with the nerf env active:
    LD_LIBRARY_PATH= python scripts/topk_eval.py \
        --checkpoint /data/vu/retromech/baselines/nerf/runs/nerf_bs128_uspto480k_unified/epoch-3-loss-1.4836149771539195
"""
import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import RDLogger

from model import MoleculeVAE
from dataset import TransformerDataset
from utils import result2mol

RDLogger.DisableLog('rdApp.*')

NTOKEN = 100  # matches main.py Trainer.ntoken

# split label -> processed test pickle (data/<prefix>_test.pickle style)
DEFAULT_SPLITS = {
    "iid":       "data/uspto480k_unified_test.pickle",
    "ood_ester": "data/uspto480k_unified_ood_ester_test.pickle",
    "ood_mass":  "data/uspto480k_unified_ood_mass_test.pickle",
}


def frag_key(smiles_list):
    """Order-independent identity of a predicted product (set of canonical fragments)."""
    return tuple(sorted(smiles_list))


def eval_split(model, data, temps, pool, batch_size, topks, device, dump_out=None):
    ds = TransformerDataset(False, data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=TransformerDataset.collate_fn)

    gt_sets = []           # per reaction: set of ground-truth fragments
    cand_lists = []        # per reaction: ordered list of distinct valid candidate frag-keys

    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader):
            b = batch['element'].shape[0]
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            element, src_mask = batch['element'], batch['src_mask']

            # ground truth (once per batch)
            gt_args = [(element[i], batch['tgt_mask'][i], batch['tgt_bond'][i],
                        batch['tgt_aroma'][i], batch['tgt_charge'][i], None) for i in range(b)]
            gt = list(pool.map(result2mol, gt_args, chunksize=64))
            batch_gt = [set(item[1].split(".")) for item in gt]

            # candidates: greedy first, then ascending temperature
            batch_cands = [[] for _ in range(b)]      # ordered distinct valid frag-keys
            seen = [set() for _ in range(b)]
            # raw prediction per temperature (canonical SMILES, or None if invalid) -- only if dumping
            per_temp = [[None] * len(temps) for _ in range(b)] if dump_out is not None else None
            for t, temp in enumerate(temps):
                out = model('sample', batch_gpu, temp)
                pb, pa, pc = out['bond'].cpu(), out['aroma'].cpu(), out['charge'].cpu()
                pred_args = [(element[j], src_mask[j], pb[j], pa[j], pc[j], None) for j in range(b)]
                res = list(pool.map(result2mol, pred_args, chunksize=64))
                for j in range(b):
                    _, smi, valid = res[j]
                    if not valid:
                        continue
                    if per_temp is not None:
                        per_temp[j][t] = smi
                    key = frag_key(smi.split("."))
                    if key not in seen[j]:
                        seen[j].add(key)
                        batch_cands[j].append(key)

            gt_sets.extend(batch_gt)
            cand_lists.extend(batch_cands)
            if dump_out is not None:
                for j in range(b):
                    dump_out.append({'gt': sorted(batch_gt[j]), 'preds': per_temp[j]})

    # score
    n = len(gt_sets)
    hits = {k: 0 for k in topks}
    any_valid = 0
    n_distinct = 0
    for gt_set, cands in zip(gt_sets, cand_lists):
        if cands:
            any_valid += 1
        n_distinct += len(cands)
        # rank of first candidate that contains all gt fragments
        correct_rank = None
        for rank, key in enumerate(cands, start=1):
            if gt_set.issubset(set(key)):
                correct_rank = rank
                break
        if correct_rank is not None:
            for k in topks:
                if correct_rank <= k:
                    hits[k] += 1
    return {
        "n": n,
        "topk": {k: hits[k] / n for k in topks},
        "valid": any_valid / n,
        "mean_distinct": n_distinct / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="full path to a checkpoint file")
    ap.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS.keys()),
                    help="subset of: " + ", ".join(DEFAULT_SPLITS))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_ladder", type=int, default=11,
                    help="number of ladder temps T=0.7*1.3**n, n=0..n_ladder-1 (plus greedy T=0)")
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dump", default=None,
                    help="if set, a DIRECTORY to save per-reaction per-temperature predictions, "
                         "one file per split as <dump>/<split>_pertemp.pickle "
                         "({'temps': [...], 'rows': [{'gt': [...], 'preds': [smiles_or_None per temp]}, ...]}).")
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    temps = [0.0] + [0.7 * 1.3 ** n for n in range(args.n_ladder)]

    model_args = argparse.Namespace(local_rank=0, vae=True, beta=0.1)
    model = MoleculeVAE(model_args, NTOKEN, args.dim, args.depth).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("loaded %s (epoch %s, step %s)" % (args.checkpoint, ckpt.get('epoch'), ckpt.get('step')))
    print("temperatures (%d):" % len(temps), [round(t, 3) for t in temps])

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    pool = ProcessPoolExecutor(args.workers)
    results = {}
    for label in args.splits:
        path = DEFAULT_SPLITS[label]
        print("\n=== %s (%s) ===" % (label, path))
        data = pickle.load(open(path, "rb"))
        dump_out = [] if args.dump else None
        results[label] = eval_split(model, data, temps, pool, args.batch_size, args.topk, device, dump_out)
        if dump_out is not None:
            out_path = os.path.join(args.dump, "%s_pertemp.pickle" % label)
            pickle.dump({'temps': temps, 'split': label, 'rows': dump_out}, open(out_path, "wb"))
            print("  dumped %d per-temperature predictions -> %s" % (len(dump_out), out_path))
        r = results[label]
        print("  n=%d  valid=%.4f  mean_distinct=%.2f" % (r["n"], r["valid"], r["mean_distinct"]))
        print("  " + "  ".join("top%d=%.4f" % (k, r["topk"][k]) for k in args.topk))
    pool.shutdown()

    # summary table
    print("\n================ SUMMARY ================")
    hdr = "%-12s %8s %8s" % ("split", "n", "valid") + "".join("%9s" % ("top%d" % k) for k in args.topk)
    print(hdr)
    for label in args.splits:
        r = results[label]
        row = "%-12s %8d %8.4f" % (label, r["n"], r["valid"]) + "".join("%9.4f" % r["topk"][k] for k in args.topk)
        print(row)


if __name__ == "__main__":
    main()
