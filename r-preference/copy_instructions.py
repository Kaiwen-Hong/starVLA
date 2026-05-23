#!/usr/bin/env python3
"""Copy pre-generated per-task instructions into a collected-data tree.

For each task folder under DATA_ROOT, look up the matching instruction
folder under INSTRUCTIONS_ROOT and copy it as DATA_ROOT/<task>/instructions/.

Source layout (kempner-style nested):
    INSTRUCTIONS_ROOT/<task>/<task>/instructions/episode{0..99}.json

Destination layout (flat alongside data/, video/, scene_info.json, seed.txt):
    DATA_ROOT/<task>/instructions/episode{0..99}.json

Conflict policy: if the destination already has an instructions/ folder,
prompt interactively per task with [s]kip / [o]verwrite / [S]kip-all /
[O]verwrite-all. Missing source: warn and skip.

Usage:
    python script/copy_instructions.py \\
        /home/kaiwen/Desktop/research/ar-research_exp/data/preference \\
        /home/kaiwen/Desktop/research/ar-research_exp/data/0520-collected

    # non-interactive: pre-commit to a policy
    python script/copy_instructions.py <SRC> <DST> --on-conflict overwrite
    python script/copy_instructions.py <SRC> <DST> --on-conflict skip

    # see what would happen without touching anything
    python script/copy_instructions.py <SRC> <DST> --dry-run
"""
import argparse
import shutil
import sys
from pathlib import Path


def find_source(instructions_root: Path, task: str) -> Path | None:
    """Return the source instructions/ folder for `task`, or None if absent."""
    candidate = instructions_root / task / task / "instructions"
    return candidate if candidate.is_dir() else None


def episode_indices(folder: Path, ext: str) -> set[int]:
    """Set of episode indices from filenames like `episode<N>.<ext>`."""
    out = set()
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        name = p.name
        if name.startswith("episode") and name.endswith(f".{ext}"):
            try:
                out.add(int(name[len("episode"):-len(f".{ext}")]))
            except ValueError:
                pass
    return out


def sync_to_data(instructions_dir: Path, data_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Make instructions_dir match data_dir's episode indices exactly.

    - Deletes instruction files whose data counterpart is missing.
    - Duplicates an existing instruction file (preferring episode0.json) to
      cover data files that lack one.

    Returns (n_added, n_removed).
    """
    data_idx = episode_indices(data_dir, "hdf5")
    instr_idx = episode_indices(instructions_dir, "json")
    if not data_idx:
        return (0, 0)  # no data → leave instructions alone

    to_remove = instr_idx - data_idx
    to_add    = data_idx - instr_idx

    # source payload: prefer episode0.json, else any existing
    template = None
    if instr_idx:
        seed = 0 if 0 in instr_idx else min(instr_idx)
        template = (instructions_dir / f"episode{seed}.json").read_bytes()

    if not dry_run:
        for i in to_remove:
            (instructions_dir / f"episode{i}.json").unlink()
        if to_add and template is not None:
            for i in to_add:
                (instructions_dir / f"episode{i}.json").write_bytes(template)

    return (len(to_add), len(to_remove))


def list_task_folders(data_root: Path) -> list[str]:
    """Direct subdirs of DATA_ROOT — each treated as a task name."""
    return sorted(p.name for p in data_root.iterdir() if p.is_dir())


def prompt_conflict(task: str, dst: Path) -> str:
    """Return one of: 'skip', 'overwrite', 'skip-all', 'overwrite-all'."""
    print(f"\n  [conflict] {dst} already exists.")
    while True:
        choice = input(
            "    [s]kip this / [o]verwrite this / "
            "[S]kip all remaining / [O]verwrite all remaining: "
        ).strip()
        if choice == "s": return "skip"
        if choice == "o": return "overwrite"
        if choice == "S": return "skip-all"
        if choice == "O": return "overwrite-all"
        print("    invalid — enter s, o, S, or O.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "instructions_root",
        type=Path,
        help="path to the instructions tree (e.g. data/preference)",
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="path to the collected data tree (e.g. data/0520-collected)",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["ask", "skip", "overwrite"],
        default="ask",
        help="what to do when destination instructions/ already exists "
             "(default: ask interactively)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print actions without copying anything",
    )
    args = parser.parse_args()

    if not args.instructions_root.is_dir():
        sys.exit(f"error: instructions_root does not exist: {args.instructions_root}")
    if not args.data_root.is_dir():
        sys.exit(f"error: data_root does not exist: {args.data_root}")

    tasks = list_task_folders(args.data_root)
    if not tasks:
        sys.exit(f"error: no task subfolders found under {args.data_root}")

    print(f"scanning {len(tasks)} task folders under {args.data_root}")
    print(f"looking up instructions from {args.instructions_root}")
    if args.dry_run:
        print("(dry-run — no files will be copied)")
    print()

    sticky_policy: str | None = None  # 'skip-all' or 'overwrite-all' once set
    if args.on_conflict in ("skip", "overwrite"):
        sticky_policy = f"{args.on_conflict}-all"

    n_copied = n_skipped = n_overwritten = n_missing = 0
    for task in tasks:
        src = find_source(args.instructions_root, task)
        dst = args.data_root / task / "instructions"

        if src is None:
            print(f"  [warn] {task}: no source instructions/ — skipping")
            n_missing += 1
            continue

        if dst.exists():
            if sticky_policy == "skip-all":
                action = "skip"
            elif sticky_policy == "overwrite-all":
                action = "overwrite"
            else:
                choice = prompt_conflict(task, dst)
                if choice in ("skip-all", "overwrite-all"):
                    sticky_policy = choice
                    action = choice.replace("-all", "")
                else:
                    action = choice

            if action == "skip":
                print(f"  [skip] {task}: kept existing instructions/")
                n_skipped += 1
                continue
            else:  # overwrite
                if not args.dry_run:
                    shutil.rmtree(dst)
                print(f"  [overwrite] {task}: removed existing instructions/")
                n_overwritten += 1

        if not args.dry_run:
            shutil.copytree(src, dst)
        n_copied += 1
        n_files = len(list(src.iterdir()))
        marker = " (dry-run)" if args.dry_run else ""
        print(f"  [copy{marker}] {task}: {n_files} files → {dst}")

        # Sync instruction count to actual data episodes
        data_dir = args.data_root / task / "data"
        n_data = len(episode_indices(data_dir, "hdf5"))
        if n_data == 0:
            print(f"    [warn] {task}: no data/episode*.hdf5 — leaving "
                  f"instructions at {n_files}")
        elif n_data != n_files:
            n_add, n_rm = sync_to_data(dst, data_dir, args.dry_run)
            print(f"    [sync{marker}] {task}: data has {n_data} episodes; "
                  f"instructions +{n_add} −{n_rm} → {n_data}")

    print()
    print(f"summary: copied={n_copied}  overwritten={n_overwritten}  "
          f"skipped={n_skipped}  missing-source={n_missing}  "
          f"total-tasks={len(tasks)}")


if __name__ == "__main__":
    main()
