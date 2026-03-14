# chenserver-files

Additive files from the `chenserver` branch, merged into starVLA main.
Path structure mirrors the repo; copy into project root to integrate.

## Contents

- **0312.md**, **LOCAL-DEPLOYMENT.md** — docs
- **ref_dd_mode.py** — discrete diffusion reference
- **scripts/setup_local_env.sh** — local setup
- **realworld/0312-train-fastumi-pickandplace-discrete-diffusion.sh** — training script
- **starVLA/config/training/** — discrete diffusion configs
- **starVLA/model/framework/QwenDiscreteDiffusion.py** — framework
- **starVLA/model/modules/action_model/** — DiscreteDiffusion headers + discrete_diffusion/ package
- **examples/Robotwin/** — Robotwin eval/train scripts

## Excluded

- `.whl` binaries
- `examples/Robotwin/eval_files/Untitled`
