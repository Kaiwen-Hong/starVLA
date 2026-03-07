# Project Context

Shared context for working in this repo. Read this first before any task.

## Who

- User: Kaiwen
- Shared lab account — multiple people use the same home directory `~`
- Kaiwen's personal shell config: `~/.bashrc-kaiwen` (not `~/.bashrc`)

## Where

- Lab storage root: `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/`
- Repo root: `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA`
- Platform: HPC cluster (FASRC / Slurm), Rocky Linux 8

## Storage Rules

- Home directory `~` has **very limited quota**. Never write caches, downloads, conda envs, or large files there.
- **All** downloads, caches, and outputs go under `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/`.
- This applies to: HuggingFace cache, pip cache, conda packages, model weights, datasets, training checkpoints, WandB logs, Triton cache.
- Every `huggingface-cli download` must include `--cache-dir $HF_HUB_CACHE`.
- Shell config goes in `~/.bashrc-kaiwen`, never `~/.bashrc`.
