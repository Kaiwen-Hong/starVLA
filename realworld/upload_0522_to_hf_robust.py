"""
Robust uploader for the 0522 FastUMI runs.

Replaces the previous bash uploads that relied on huggingface-cli `--include`
globs containing slashes (e.g. "checkpoints/steps_20000_pytorch_model.pt").
That pattern combined with a non-"." `path_in_repo` silently only matched
`summary.jsonl`, so the actual .pt + config.yaml + dataset_statistics.json
never made it to the Hub.

This script uploads each file explicitly via HfApi.upload_file.

Destinations (each run):
  1. outsider86/<run_id>                          (standalone repo)
  2. outsider86/DiscreteRTC, path_in_repo=<run_id>/  (consolidated repo)

Files uploaded per run:
  - checkpoints/steps_20000_pytorch_model.pt   (~12 GB)
  - config.yaml
  - dataset_statistics.json
  - summary.jsonl
"""
from __future__ import annotations

from pathlib import Path
import sys

from huggingface_hub import HfApi, create_repo, HfFolder
from huggingface_hub.utils import HfHubHTTPError
import os

CHECKPOINTS_ROOT = Path("/scratch/wangpc/starVLA/results/Checkpoints")
DISCRETE_RTC_REPO = "outsider86/DiscreteRTC"

RUN_IDS = [
    "fastumi_airhockey_bounce_qwenPI_0522_DiT-S",
    "fastumi_pool_qwenPI_0522_DiT-S",
]

FILES = [
    "checkpoints/steps_20000_pytorch_model.pt",
    "config.yaml",
    "dataset_statistics.json",
    "summary.jsonl",
]


def ensure_repo(api: HfApi, repo_id: str) -> None:
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=api.token)
    except HfHubHTTPError as e:
        print(f"[warn] create_repo({repo_id}) raised: {e}")


def upload(api: HfApi, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    if not local_path.exists():
        print(f"  MISSING  {local_path}")
        return
    print(f"  -> {repo_id}/{path_in_repo}  ({local_path.stat().st_size / 1e9:.2f} GB)")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def main() -> int:
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        cache_path = Path.home() / ".cache" / "huggingface" / "token"
        if cache_path.exists():
            token = cache_path.read_text().strip()
    if not token:
        print("ERROR: not logged in. Run `huggingface-cli login` first.", file=sys.stderr)
        return 1
    api = HfApi(token=token)

    ensure_repo(api, DISCRETE_RTC_REPO)
    for run_id in RUN_IDS:
        ensure_repo(api, f"outsider86/{run_id}")

    for run_id in RUN_IDS:
        run_dir = CHECKPOINTS_ROOT / run_id
        print(f"\n=== {run_id} ===")
        for rel in FILES:
            local = run_dir / rel
            # 1. standalone repo: outsider86/<run_id>/<rel>
            upload(api, f"outsider86/{run_id}", local, rel)
            # 2. consolidated: outsider86/DiscreteRTC/<run_id>/<rel>
            upload(api, DISCRETE_RTC_REPO, local, f"{run_id}/{rel}")

    print("\nDone.")
    print(f"  https://huggingface.co/outsider86/{RUN_IDS[0]}")
    print(f"  https://huggingface.co/outsider86/{RUN_IDS[1]}")
    print(f"  https://huggingface.co/{DISCRETE_RTC_REPO}/tree/main/{RUN_IDS[0]}")
    print(f"  https://huggingface.co/{DISCRETE_RTC_REPO}/tree/main/{RUN_IDS[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
