# 0611/0612 overnight hvlv closed-loop diagnosis — archive

Full narrative: `r-preference/doc/0612-hvlv-diagnosis.md`.
ROOT CAUSE: every hvlv eval to date rendered the default `aloha-agilex` head camera
(53° pitch) while ALL hvlv data (taskA + taskB) was collected with
`aloha-agilex-topdown` (73°; collection-box `doc/0528-hvlv-topdown-camera.md`).
Secondary global bug (real but non-binding for hvlv success): collection hdf5 JPEGs are
R/B-swapped (`cv2.imencode` on RGB) — the bridge gained an env-gated `STARVLA_SWAP_RB=1`.

All runs: env `stamp_seal6_hv`, paired prompts, `test_num 6, seed 0` (same expert-filtered
seed walk per arm), chunk 50/50, new 2×5090 box (169.40.1.214). N−1 pref-JSON quirk: 5
records per 6-episode arm. Success predicate: seal within ±2 cm of pad + grippers open +
obstacle (coke can) unmoved.

## Results matrix

| arm (dir) | ckpt | camera | swap | hv | lv |
|---|---|---|---|---|---|
| 0529 original (old box; see eval/0529_50ep) | main_geom@1500 | default | off | 0/5 | 0/5 |
| `pref_stageb_main_hvlv_geom_defaultcam_swap/` | main_geom@1500 | default | on | 0/6 | 0/6 |
| `pref_stageb_main_hvlv_geom_topdown_swap/` | main_geom@1500 | topdown | on | **3/6** | **3/6** |
| `pref_stageb_main_hvlv_geom_topdown_noswap/` | main_geom@1500 | topdown | off | 3/6 | not run |
| `pref_stageb_b0_hvlv_topdown_swap/` | b0@1500 | topdown | on | 0/6 | 0/6 |
| `pref_stageb_main_hvlv_geom2500_topdown_swap/` | main_geom@2500 | topdown | on | **4/6** | **2/6** |

Key readings:
- Camera fix recovers the stamp MOTION in every arm (grasp≈t60-85 → release≈t194-243,
  matching the demo script); residual failures are ±2 cm placement near-misses.
- SPT@1500 (main_geom) 6/12 vs Naive-FT (b0) 0/12 pooled task success — Fisher 1-sided p≈0.005.
- Detour separation: @1500 −0.004 m (none) but **@2500 +0.075 m** (hv 0.145 / lv 0.070,
  expert-like; follow 90% per-run midpoint, 70% under the expert-midpoint threshold) —
  hvlv conditioning EXISTS at 2500 Stage-B steps. Recommended hvlv SPT ckpt: steps_2500.
- Color swap: no closed-loop success effect at n=6 (camera-only arm identical);
  offline TF cost: main 0.0228→0.0286 MAE, b0 0.0302→0.0656.

## Videos (`hvlv_videos/`)
- `swap_hv_ep0_new.mp4` — default-cam control ep0: wanders, descends at the CAN, never
  grasps the seal (the seal spawned at the near edge, out of the default head view).
- `topdown_swap_hv_ep0_SUCCESS.mp4` — first fixed-camera success (same seed 100003).
- `td_hv/`, `td_lv/` — fix-run episode videos + `_result.txt`.
- `td_b0_hv/`, `td_b0_lv/` — b0 arm videos (all near-miss timeouts).
- `td_noswap_hv/` — camera-only control videos.
- `td_2500_hv/`, `td_2500_lv/` — steps_2500 arm videos.

## Teacher forcing (offline, demo stamp_seal6_hv/ep0 fed to the box server)

| ckpt | as-stored (training-like) MAE rad | true-color (old-deploy) MAE rad |
|---|---|---|
| main_hvlv_geom@1500 | 0.0228 | 0.0286 |
| b0_hvlv@1500 | 0.0302 | 0.0656 |

Expert detour reference (50 demos/side): whole-traj-window hv 0.109 / lv 0.036;
grasp→release-segment window hv 0.206 / lv 0.083 (metric upgrade recommended for layer-2).

## Logs
`hvlv_logs/` — driver scripts + driver logs + per-prompt eval logs (note per-run logs
overwrite on the box; the driver logs carry the final success lines of every arm).
