#!/usr/bin/env python3
"""Strip preference-grasp clauses from Task B (`put_*_plate_*`) instructions.

Task B's source JSON files on the 5090 have the form
    "Place the boxdrink onto the plate, grasping on the bottom of the boxdrink."
where the clause after the first comma encodes a grasp-height preference
(_25 → bottom, _50 → middle, _75 → top). Task B is meant to be
preference-free, so we cut each sentence at the first comma, restore the
trailing period, and dedupe (preference stripping collapses many variants
to the same base sentence).

Input layout (kempner-style nested):
    SRC/<task>/<task>/instructions/episode{0..N-1}.json
    each JSON: {"seen": [...], "unseen": [...]}

Default: rewrite in-place under SRC (only the 12 Task B dirs are touched).
Pass --out DIR to mirror the cleaned tree into a separate directory instead.

Usage:
    # in-place rewrite (default)
    python r-preference/strip_preference_taskb.py \\
        /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj

    # mirror to a parallel tree (raw kept intact)
    python r-preference/strip_preference_taskb.py \\
        /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj \\
        --out /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj_clean

    # preview only
    python r-preference/strip_preference_taskb.py <SRC> --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

TASK_B_TASKS = [
    f"put_{obj}_plate_{pct}"
    for obj in ("boxdrink", "callbell", "fork", "screwdriver")
    for pct in (25, 50, 75)
]


def strip_one(sentence: str) -> str:
    """Cut at first comma, restore period. Leave alone if no comma."""
    idx = sentence.find(",")
    if idx < 0:
        return sentence
    base = sentence[:idx].rstrip()
    return base if base.endswith(".") else base + "."


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe."""
    seen, out = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def process_json(src: Path, dst: Path, dry_run: bool) -> tuple[int, int, int]:
    """Returns (n_seen_in, n_seen_out, n_unseen_out) for reporting."""
    d = json.loads(src.read_text())
    n_in_seen = len(d.get("seen", []))
    new_seen   = dedupe([strip_one(s) for s in d.get("seen", [])])
    new_unseen = dedupe([strip_one(s) for s in d.get("unseen", [])])
    out = {"seen": new_seen, "unseen": new_unseen}
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2))
    return (n_in_seen, len(new_seen), len(new_unseen))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("src", type=Path, help="instructions source root (kempner-style)")
    ap.add_argument("--out", type=Path, default=None,
                    help="mirror cleaned tree here (default: rewrite in place)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print actions without writing files")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"error: src does not exist: {args.src}")

    print(f"stripping preference clauses for {len(TASK_B_TASKS)} Task B tasks "
          f"under {args.src}")
    if args.out:
        print(f"writing cleaned tree to {args.out}")
    else:
        print("rewriting in place")
    if args.dry_run:
        print("(dry-run — no files will be written)")
    print()

    n_tasks_done = n_missing = 0
    for task in TASK_B_TASKS:
        instr_dir = args.src / task / task / "instructions"
        if not instr_dir.is_dir():
            print(f"  [skip] {task}: source dir missing ({instr_dir})")
            n_missing += 1
            continue
        files = sorted(instr_dir.glob("episode*.json"))
        if not files:
            print(f"  [skip] {task}: no episode*.json under {instr_dir}")
            n_missing += 1
            continue

        # representative sample for visual sanity-check
        sample_in  = json.loads(files[0].read_text()).get("seen", [None])[0]
        sample_out = strip_one(sample_in) if sample_in else None

        seen_sizes_out = []
        for f in files:
            if args.out:
                rel = f.relative_to(args.src)
                dst = args.out / rel
            else:
                dst = f
            _, n_out_seen, _ = process_json(f, dst, args.dry_run)
            seen_sizes_out.append(n_out_seen)

        min_out, max_out = min(seen_sizes_out), max(seen_sizes_out)
        marker = " (dry-run)" if args.dry_run else ""
        print(f"  [strip{marker}] {task}: {len(files)} json; "
              f"seen 100 → {min_out}..{max_out} after dedupe")
        print(f"    e.g. {sample_in!r}")
        print(f"      → {sample_out!r}")
        n_tasks_done += 1

    print()
    print(f"summary: tasks_processed={n_tasks_done}  missing={n_missing}  "
          f"total={len(TASK_B_TASKS)}")


if __name__ == "__main__":
    main()
