#!/usr/bin/env python3
"""
Fix tasks.jsonl in pickandplace-real-0307: replace experiment names with actual task description.

The LeRobot dataset stores task strings in meta/tasks.jsonl. When the task column
contains experiment names (e.g. "pickandplace-vla- 0307") instead of the human-readable
task description, training receives wrong instructions.

Usage:
    python realworld/fix_pickandplace_tasks_jsonl.py
    python realworld/fix_pickandplace_tasks_jsonl.py --dataset playground/Datasets/FastUMI/pickandplace-real-0307
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "playground/Datasets/FastUMI/pickandplace-real-0307"
TASK_DESCRIPTION = "pick up the red building block"

# Strings to replace (experiment/run names)
EXPERIMENT_NAMES = {
    "pickandplace-vla- 0307",
    "pickandplace-vla-0307",
    "pickandplace-real-0307",
}


def fix_tasks_jsonl(dataset_path: Path, dry_run: bool = False) -> int:
    tasks_path = dataset_path / "meta/tasks.jsonl"
    if not tasks_path.exists():
        print(f"ERROR: {tasks_path} not found")
        return 1

    with open(tasks_path, "r") as f:
        lines = f.readlines()

    changes = 0
    new_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            new_lines.append(line + "\n")
            continue
        obj = json.loads(line)
        old_task = obj.get("task", "")
        if str(old_task).strip() in EXPERIMENT_NAMES or "pickandplace" in str(old_task).lower():
            obj["task"] = TASK_DESCRIPTION
            changes += 1
            print(f"  Line {i+1}: '{old_task}' -> '{TASK_DESCRIPTION}'")
        new_lines.append(json.dumps(obj) + "\n")

    if changes == 0:
        print("No changes needed.")
        return 0

    if dry_run:
        print(f"[DRY RUN] Would update {changes} task(s)")
        return 0

    with open(tasks_path, "w") as f:
        f.writelines(new_lines)
    print(f"Updated {tasks_path}: {changes} task(s) replaced")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fix task descriptions in tasks.jsonl")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset directory (containing meta/tasks.jsonl)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()
    print(f"Dataset: {args.dataset}")
    return fix_tasks_jsonl(args.dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    exit(main())
