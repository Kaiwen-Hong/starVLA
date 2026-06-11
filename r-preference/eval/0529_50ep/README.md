# 0529 Stage-B paired controllability eval

Snapshot of the 2026-05-29 controllability evaluation on the 4×RTX5090 box, archived so future
re-runs (same ckpt/env dirs, which get `rm -rf`'d) cannot overwrite it. See
`r-preference/doc/0529-5090-eval-and-stageb-controllability.md` for the full setup.

## What this is
Paired same-seed two-prompt controllability: one fixed taskB env, the prompt's preference label
varied between its two values; for each rollout `pref_metric` measures the policy's actual
preference-relevant quantity (decoupled from `check_success`, which is recipe-bound). A clean
result = the two prompts produce a separated quantity in the correct direction.

- **height, orient**: 50 paired seeds × 2 prompts (the headline). `ours` = `pref_stageb_main_{height,orient}` (token-label Stage-B, steps_1500).
- **place, contact, hvlv**: 5 paired seeds × 2 prompts (sample). `ours` = `pref_stageb_main_{place,contact,hvlv}_geom` (geometric-pseudo-label Stage-B, steps_1500).

Metrics aligned with the training geom pseudo-labels (`examples/preference/stage_b/pseudo_label_geom.py`):
place = EE.xy@release→receptacle dist; hvlv = max perpendicular detour of the active-arm EE path off
its 25–85% transport chord (pure EE); contact = EE.z−obj.z at grasp (height_fraction proxy);
height = release_z−grasp_z; orient = EE local-x tilt from vertical at grasp.

## Results
See `summary.md` / `summary.json`. Headline:

| cat | scale | separation | follow | task success | verdict |
|---|---|---|---|---|---|
| height | 50ep | +0.071 m | 81% (high 67 / low 94) | high 50/50, low 41/50 | ✅ controllable |
| orient | 50ep | +47.9° | 81% (0=100 / 90=61) | 0: 48/50, 90: 43/50 | ✅ controllable (90 capped by IK-marginal spawns) |
| place | 5ep | +0.046 m | 88% (corner 100 / center 75) | center 4/5, corner 0/5* | ✅ controllable |
| contact | 5ep | +0.021 m | 75% | both 4/5 | 🟡 correct direction, weak |
| hvlv | 5ep | −0.001 m | 50% | both 0/5 | ❌ not demonstrated (task transfer fails) |

\* **place corner success = 0/5 is a recipe artifact, NOT a failure.** The env is the `_center` recipe,
so `check_success` only passes when the soap is placed at center. Under the *corner* prompt the policy
correctly places at the corner (metric 0.067 vs center 0.020), which the center-recipe success check
counts as a miss. The metric is exactly what disambiguates this — controllability holds.

**hvlv open issue**: stamp_seal6 (taskB) scores 0/5 success under BOTH prompts, so the detour signal is
noise (50%). Most likely the policy cannot perform the never-trained "stamp" motion at all (hard taskB
transfer, like orient's move_can5_away taskB was 0% in earlier rollout tests), rather than a
controllability failure per se. Needs a transfer-vs-env diagnosis (see videos in r-preference/debug/0529_videos/hvlv/).

## Reproducibility
- All runs `--seed 0` → `st_seed = 100000*(1+seed) = 100000`; eval walks seeds upward, runs the expert
  on each, and only expert-solvable seeds are used for the policy (rest skipped) until `test_num` valid.
- Both prompts of a task use the **same** seed list (paired). See `seeds.json` for the exact 50-seed
  lists (height: 100002..100060; orient: 100001..100057 — gaps = expert-unsolvable seeds, env-specific).
- To run a baseline on identical scenes: same env + `--seed 0` + same `test_num`. For a fresh
  non-overlapping set use `--seed 1` (→ st_seed 200000).

## Files
- `per_episode/<ckpt_run>/<env>__prompt_<pk>/<task>_epNNN.json` — raw per-episode metric records
- `summary.json`, `summary.md` — aggregated separation / follow / success
- `seeds.json` — exact evaluated seed lists + seed scheme
- `build_summary.py` — regenerates the summary from per_episode/
- videos: `r-preference/debug/0529_videos/<cat>/{success,failure}/` (head_camera, not committed to git)
