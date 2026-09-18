"""Convert the uspto_480k_unified HuggingFace/Arrow dataset into NERF's .txt format.

NERF expects one `reactants>>products` SMILES per line, with every atom carrying
an atom-map number, indexed downstream by `GetAtomMapNum()-1` (see preprocess.py).

This dataset stores atom-mapped `reactant` / `product` SMILES in Arrow files. The
only cleanup needed is stripping *fully-unmapped* fragments (reagents/solvents that
carry no map number) from the reactant -- an unmapped atom would index to -1 and
silently corrupt the last atom slot in preprocess.py.

Reads the .arrow files directly with pyarrow (no `datasets` library needed), so it
runs in the pinned `nerf` env (python 3.7 / rdkit 2018.03.4 / pyarrow 12).

Usage (from repo root, in the nerf env):
    LD_LIBRARY_PATH= python scripts/convert_maelle_to_nerf.py \
        --src /data/share/vu/retromech/uspto_480k_unified \
        --out-dir data/uspto480k_unified
Add --ood to also emit the OOD test splits.
"""
import argparse
import os

import pyarrow as pa
from rdkit import Chem
from rdkit import RDLogger

# split_dir_name -> nerf output basename. NERF's preprocess.py __main__ looks for
# train/valid/test, so `val` and `test_iid` are renamed accordingly.
CORE_SPLITS = {"train": "train", "val": "valid", "test_iid": "test"}
OOD_SPLITS = {
    "test_ood_ester": "test_ood_ester",
    "test_ood_mass": "test_ood_mass",
    "test_ood_scaffold_A": "test_ood_scaffold_A",
    "test_ood_scaffold_B": "test_ood_scaffold_B",
}


def read_arrow(split_dir):
    """Yield the 'reactant' and 'product' columns from a datasets-saved arrow dir."""
    path = os.path.join(split_dir, "data-00000-of-00001.arrow")
    with pa.memory_map(path, "r") as src:
        table = pa.ipc.open_stream(src).read_all()
    reactants = table.column("reactant").to_pylist()
    products = table.column("product").to_pylist()
    return reactants, products


def strip_unmapped_fragments(smiles):
    """Drop dot-separated fragments in which no atom carries a map number.

    Returns (cleaned_smiles, ok). ok is False if any kept fragment fails to parse
    (so the caller can skip the row -- preprocess.py has no None-guard on parsing).
    """
    kept = []
    for frag in smiles.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return None, False
        if all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms()):
            continue  # fully-unmapped reagent/solvent -> drop
        kept.append(frag)
    if not kept:
        return None, False
    return ".".join(kept), True


def maps_are_clean(reactant, product):
    """NERF invariants: reactant maps are contiguous 1..N and product maps subset."""
    r_mol = Chem.MolFromSmiles(reactant)
    p_mol = Chem.MolFromSmiles(product)
    if r_mol is None or p_mol is None:
        return False
    r_maps = [a.GetAtomMapNum() for a in r_mol.GetAtoms()]
    p_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms()}
    n = r_mol.GetNumAtoms()
    if sorted(r_maps) != list(range(1, n + 1)):
        return False  # gaps, dups, or unmapped atoms remain
    if not p_maps.issubset(set(r_maps)):
        return False  # a product atom not present in the reactant
    return True


def convert_split(split_dir, out_path):
    reactants, products = read_arrow(split_dir)
    written = 0
    skipped_parse = 0
    skipped_dirty = 0
    with open(out_path, "w") as out:
        for reactant, product in zip(reactants, products):
            cleaned, ok = strip_unmapped_fragments(reactant)
            if not ok:
                skipped_parse += 1
                continue
            if not maps_are_clean(cleaned, product):
                skipped_dirty += 1
                continue
            out.write("%s>>%s\n" % (cleaned, product))
            written += 1
    total = len(reactants)
    print("  %s: wrote %d / %d  (skipped %d unparseable, %d map-dirty)"
          % (os.path.basename(out_path), written, total, skipped_parse, skipped_dirty))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/data/share/vu/retromech/uspto_480k_unified",
                        help="root of the uspto_480k_unified dataset")
    parser.add_argument("--out-dir", default="data/uspto480k_unified",
                        help="where to write the NERF .txt files")
    parser.add_argument("--ood", action="store_true",
                        help="also convert the OOD test splits")
    args = parser.parse_args()

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog("rdApp.*")

    os.makedirs(args.out_dir, exist_ok=True)
    splits = dict(CORE_SPLITS)
    if args.ood:
        splits.update(OOD_SPLITS)

    for split_name, out_name in splits.items():
        split_dir = os.path.join(args.src, split_name)
        if not os.path.isdir(split_dir):
            print("  %s: MISSING (%s) -- skipped" % (split_name, split_dir))
            continue
        out_path = os.path.join(args.out_dir, out_name + ".txt")
        print("converting %s ->" % split_name)
        convert_split(split_dir, out_path)


if __name__ == "__main__":
    main()
