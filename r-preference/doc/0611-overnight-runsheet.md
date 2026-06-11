# 0611 Overnight Runsheet (live — supervision passes update this file)

> Purpose: survives context compaction. Each ~30-min supervision pass: check every
> in-flight row, unstick per the standing authorizations, update Status/Notes here,
> reschedule the next wakeup. Full context: `0611-paper-gap-analysis.md`.
> User away; returns in the morning. I am responsible for results.

## Standing authorizations (user, 2026-06-11 night)
1. contact experiment ladder: EF(早帧加权) → LR×3 → 偏好段加权loss (方案3, code change OK). Each: own run_id, early-frame probe + Tier-1 gate, never overwrite.
2. hvlv: if Agent A diagnoses stamp_seal6 unfixable → AUTHORIZED to reconstruct target
   (hold out place_hamburg_plate_{hv,lv} from source, retrain hvlv Stage-A ~2h excl. that
   pair, Stage-B, eval). Document fully; reversible.
3. Commits: topical, local opd, NO push. Paper tex: DO NOT edit — produce diff-suggestion doc only.
4. Fixed source-stats follow thresholds = official convention (done, committed).

## In-flight (update每轮)
| # | What | Where | Watch | Status @09:55Z |
|---|---|---|---|---|
| 1 | contact_geom_ef training v2 (2500 steps, save 500) | H100 8GPU | queue log `stageb_queue_*.log`; started 09:43:47Z | RUNNING |
| 2 | EF chain: wait steps_2500 → early-frame probe + Tier-1 | H100 bg task bx8hrco1v | outputs `earlyframe_contact_ef.json`, `ctrl_contact_ef_2500.json`; log `chain_contact_ef_0611.log` payload | ARMED |
| 3 | Agent A: hvlv deep-dive | 5090 box (drives it) | footprints: box `/root/drive_*`, `/root/eval_pref_ctrl/*hvlv*`, H100 `r-preference/eval/0611_ctrl/`, doc `0612-hvlv-diagnosis.md` | RUNNING |
| 4 | Agent B: privilege-free labelers | H100 (1 GPU windows) | footprints: `examples/preference/stage_b/pseudo_label_{ee,vision}.py`, caches `*_B_{ee,vision}.json`, `r-preference/debug/v5/`, doc `0612-privilege-free-labelers.md` | RUNNING (ee files appeared) |

## Queue (run in order as box/GPUs free; do NOT collide with Agent A on the box)
1. After Agent A done: orient 50ep both arms (orient_geom + b0_orient, full ~50-seed list, prompts 0/90) — Table-2 official numbers.
2. place 50ep: ship pref_stageb_main_place_geom + pref_stageb_b0_place steps_1500 (+sidecars) to box (GCS relay, setup doc §6); paired 50ep both arms.
3. contact closed-loop redo (fixed metric; if contact-points still unavailable, add EE-fallback source threshold first).
4. contact ladder step 2/3 if EF fails its gates (compare vs geom@1500 eff 0.0029/follow 0.625, b0 0.0009/0.417; success bar: clearly higher effect + follow, then closed-loop on box).

## Key facts for post-compaction me
- Box: ssh -p 17278 root@169.40.1.214 (2 usable GPUs; pkill needs [.]-trick; test_num=K+1 quirk;
  track_step obs-fix applied #13; robomme shares box — never touch, yield).
- Disk: H100 SSD cleaned to 84% (940G free); deletion log `logs/disk_cleanup_0611.log`; keep-set = evaluated ckpts only.
- HF full backup exists (memory `project_hf_backup_0611.md`): code GitHub opd@backup-0611, data/ckpts/workspace on HF — restores possible.
- All numbers/evidence: `0611-paper-gap-analysis.md` (§4b 早帧方法, §4c FT closed-loop, §4d official table).
- Morning deliverables promised: hvlv diagnosis(+fix/reconstruction), contact ladder对照, privilege-free labeler成绩, orient/place 50ep, paper diff doc, 晨报.

## Supervision log
- 09:55Z pass 0: runsheet created; EF v2 training healthy (started 09:43); agents running; disk 84%.
