"""Convert the ABSynth (absynth_maelle_test) Arrow dataset into a NERF test cache.

Unlike scripts/convert_maelle_to_nerf.py (which targets the fully atom-mapped
uspto_480k_unified and rounds through preprocess.py), this dataset has
**atom-mapped reactants but UNMAPPED products**. NERF's preprocess.py indexes
target atoms by `GetAtomMapNum()-1`, so it cannot build target features here and
its whole electron-redistribution target is undefined.

For a *test* set we don't need the target features: at sample time the model uses
only the source encoding + the N(0,I) prior (model.py `mode=='sample'`), and
accuracy is a SMILES-containment test. So this script:

  1. Builds NERF source features directly from the mapped reactants (reusing
     preprocess.molecule), with the `reactant` flag set to 1 for every fragment
     -- i.e. reagents and reactants are all treated as inputs (no spectator
     distinction, since without product maps we can't derive one).
  2. Emits DUMMY target features (copies of the source) so DataLoader collation
     and the sample-mode forward have the tensors they expect; they are never
     read during sampling.
  3. Writes the canonical product SMILES to a separate ground-truth sidecar,
     aligned row-for-row with the feature list, for exact-SMILES scoring in
     scripts/topk_eval.py (--gt-pickle).

Reactant maps in this dataset are contiguous 1..N (verified), which molecule()
requires. Rows whose reactant fails to parse or violates MAX_BONDS are dropped;
the sidecar stays aligned because both are appended in the same loop.

Usage (from repo root, in the nerf env):
    LD_LIBRARY_PATH= python scripts/convert_absynth_to_nerf.py \
        --src /data/vu/absynth/absynth_maelle_test \
        --out data/absynth_maelle_test.pickle \
        --gt-out data/absynth_maelle_test.products.pickle
"""
import argparse
import os
import pickle
import sys

import pyarrow as pa
from rdkit import Chem
from rdkit import RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import molecule  # noqa: E402  (reuse the canonical feature builder)


def canon(smi):
    """RDKit-canonical, map-free, STEREO-FREE SMILES.

    NERF has no stereochemistry: result2mol rebuilds molecules from
    element/bond/charge/aroma only, so predicted SMILES are always flat. To score
    fairly we flatten the ground truth too (isomericSmiles=False) -- otherwise
    every stereo-bearing product (common in ABSynth) is marked wrong even when the
    predicted connectivity is exactly right.
    """
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def read_arrow(split_dir):
    """Return (reactant, product, source_idx) columns from a datasets-saved arrow dir."""
    path = os.path.join(split_dir, "data-00000-of-00001.arrow")
    with pa.memory_map(path, "r") as src:
        table = pa.ipc.open_stream(src).read_all()
    reactants = table.column("reactant").to_pylist()
    products = table.column("product").to_pylist()
    source_idx = (table.column("source_idx").to_pylist()
                  if "source_idx" in table.column_names else list(range(len(reactants))))
    return reactants, products, source_idx


def build_source_features(reactant):
    """Build NERF source features from an atom-mapped reactant SMILES.

    Returns the molecule() feature dict, or None if the reactant is unparseable,
    has non-contiguous atom maps, or violates MAX_BONDS.
    """
    whole = Chem.MolFromSmiles(reactant)
    if whole is None:
        return None
    n = whole.GetNumAtoms()
    if sorted(a.GetAtomMapNum() for a in whole.GetAtoms()) != list(range(1, n + 1)):
        return None  # molecule() indexes by GetAtomMapNum()-1; needs contiguous 1..N

    frags = [Chem.MolFromSmiles(f) for f in reactant.split(".")]
    if any(m is None for m in frags):
        return None
    reactant_mask = [True] * len(frags)  # merge all reagents+reactants as inputs
    return molecule(frags, n, reactant_mask, None)  # None on MAX_BONDS overflow


def convert(src_dir, out_path, gt_path):
    reactants, products, source_idx = read_arrow(src_dir)
    total = len(reactants)

    dataset = []
    gt_smiles = []
    gt_source_idx = []
    skipped_src = 0
    skipped_prod = 0
    for reactant, product, sidx in zip(reactants, products, source_idx):
        src = build_source_features(reactant)
        if src is None:
            skipped_src += 1
            continue
        gt = canon(product)
        if gt is None:
            skipped_prod += 1
            continue

        # source features + dummy target features (copies; unused in sample mode)
        item = {"element": src["element"], "reactant": src["reactant"]}
        for key in ("bond", "charge", "aroma", "mask", "segment"):
            item["src_" + key] = src[key]
            item["tgt_" + key] = src[key]
        dataset.append(item)
        gt_smiles.append(gt)
        gt_source_idx.append(sidx)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    with open(gt_path, "wb") as f:
        pickle.dump({"product_smiles": gt_smiles, "source_idx": gt_source_idx}, f)

    print("wrote %d / %d rows  (skipped %d unparseable/oversized reactants, %d bad products)"
          % (len(dataset), total, skipped_src, skipped_prod))
    print("  features -> %s" % out_path)
    print("  ground truth (%d canonical product SMILES) -> %s" % (len(gt_smiles), gt_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/data/vu/absynth/absynth_maelle_test",
                    help="the datasets-saved arrow directory")
    ap.add_argument("--out", default="data/absynth_maelle_test.pickle",
                    help="output NERF feature pickle (list of feature dicts)")
    ap.add_argument("--gt-out", default="data/absynth_maelle_test.products.pickle",
                    help="output ground-truth sidecar (canonical product SMILES, aligned)")
    args = ap.parse_args()

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog("rdApp.*")

    convert(args.src, args.out, args.gt_out)


if __name__ == "__main__":
    main()
