#!/usr/bin/env python3
"""Mount preference + base instructions into the 0526 pref data tree.

For every leaf under <ROOT>/<cat>/ (taskA, flat) and <ROOT>/<cat>/taskB/
(taskB, nested), copy two instruction sets from the collection-side trees:

    <PREF_SRC>/<leaf>/<leaf>/instructions/  ->  <leaf>/instructions/        (rich, preference inline)
    <BASE_SRC>/<leaf>/<leaf>/instructions/  ->  <leaf>/instructions_base/   (preference stripped)

Then make each mounted set's episode indices match the leaf's data/ exactly
(trim extras, duplicate from the lowest existing index to fill gaps) -- the
collection-side trees ship 100 episode files per leaf, taskB data is 50 ep.

Idempotent: re-running overwrites both instruction dirs.

Usage:
    python r-preference/mount_instructions_0526.py \
        --root      /mnt/localssd/$USER/pref/data/0526 \
        --pref-src  /mnt/localssd/$USER/pref/data/.instr_src_0526/preference \
        --base-src  /mnt/localssd/$USER/pref/data/.instr_src_0526/preference_base \
        [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

CATS = ("contact", "height", "hvlv", "orient", "place")


def episode_indices(folder: Path, ext: str) -> set[int]:
    out: set[int] = set()
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        n = p.name
        if n.startswith("episode") and n.endswith(f".{ext}"):
            try:
                out.add(int(n[len("episode"):-len(f".{ext}")]))
            except ValueError:
                pass
    return out


def sync_to_data(instr_dir: Path, data_dir: Path, dry: bool) -> tuple[int, int]:
    """Match instr_dir json indices to data_dir hdf5 indices. Returns (added, removed)."""
    data_idx = episode_indices(data_dir, "hdf5")
    instr_idx = episode_indices(instr_dir, "json")
    if not data_idx:
        return (0, 0)
    to_remove = instr_idx - data_idx
    to_add = data_idx - instr_idx
    template = None
    if instr_idx:
        seed = 0 if 0 in instr_idx else min(instr_idx)
        template = (instr_dir / f"episode{seed}.json").read_bytes()
    if not dry:
        for i in to_remove:
            (instr_dir / f"episode{i}.json").unlink()
        if to_add and template is not None:
            for i in to_add:
                (instr_dir / f"episode{i}.json").write_bytes(template)
    return (len(to_add), len(to_remove))


def mount_one(leaf_dir: Path, src_tree: Path, dest_name: str, dry: bool) -> str:
    """Copy <src_tree>/<leaf>/<leaf>/instructions -> <leaf_dir>/<dest_name>, sync to data/."""
    leaf = leaf_dir.name
    src = src_tree / leaf / leaf / "instructions"
    if not src.is_dir():
        return f"MISSING-SRC ({dest_name})"
    dst = leaf_dir / dest_name
    if not dry:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    n_src = len(list(src.iterdir()))
    n_add, n_rm = sync_to_data(dst, leaf_dir / "data", dry)
    n_data = len(episode_indices(leaf_dir / "data", "hdf5"))
    return f"{dest_name}: src={n_src} +{n_add} -{n_rm} -> {n_data}"


def iter_leaves(root: Path):
    """Yield leaf_dir Path for taskA (flat) and taskB (nested) across all cats."""
    for cat in CATS:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for d in sorted(cat_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "taskB":
                for tb in sorted(d.iterdir()):
                    if tb.is_dir() and (tb / "data").is_dir():
                        yield cat, "taskB", tb
                continue
            if (d / "data").is_dir():
                yield cat, "taskA", d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pref-src", type=Path, required=True)
    ap.add_argument("--base-src", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (args.root, args.pref_src, args.base_src):
        if not p.is_dir():
            sys.exit(f"error: not a dir: {p}")

    leaves = list(iter_leaves(args.root))
    if not leaves:
        sys.exit(f"error: no leaves under {args.root}")

    print(f"{'(dry-run) ' if args.dry_run else ''}mounting {len(leaves)} leaves under {args.root}")
    print(f"  pref-src: {args.pref_src}")
    print(f"  base-src: {args.base_src}\n")

    n_ok = n_miss = 0
    for cat, kind, leaf_dir in leaves:
        r_pref = mount_one(leaf_dir, args.pref_src, "instructions", args.dry_run)
        r_base = mount_one(leaf_dir, args.base_src, "instructions_base", args.dry_run)
        flag = "" if ("MISSING" not in r_pref and "MISSING" not in r_base) else "   <-- CHECK"
        if "MISSING" in r_pref or "MISSING" in r_base:
            n_miss += 1
        else:
            n_ok += 1
        print(f"  [{cat}/{kind}] {leaf_dir.name}")
        print(f"      {r_pref}")
        print(f"      {r_base}{flag}")

    print(f"\nsummary: leaves={len(leaves)}  ok={n_ok}  missing-src={n_miss}")


if __name__ == "__main__":
    main()
