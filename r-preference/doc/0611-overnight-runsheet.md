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
- 10:25Z pass: Agent B DONE (place vision 1.000, contact EE 1.000 — committed 47c6a60; gap doc §1 updated).
  EF@2500 result: early-frame peak_eff 0.0221 (7.6x geom, 24x b0), follow 0.667 balanced — PARTIAL SUCCESS, trend rising.
  Ladder-2 launched: ef5k (5000 steps) + ef_lr3 (EF+qwen_vl LR x3), queue started 10:19:07Z, probe chain armed (bv91rliz8).
  Agent A still running (writing 0612-hvlv-diagnosis.md; defaultcam_swap variant dir spotted). Disk 39%.
- 10:42Z pass: ef5k 2349/5000 (~23min left), probe chain armed. Agent A deep in camera-view hypothesis
  (3 variant evals archived: defaultcam_swap / topdown_swap / topdown_noswap; doc at 247 lines; box now
  running stamp hv arm on :10099). Disk H100 39%, box 68% (41G). No new completions to fold/commit.
- 10:58Z pass (event): Agent A DONE — hvlv ROOT CAUSE = eval camera embodiment mismatch (topdown 73° data
  vs default 53° eval); fixed: SPT succ 6/12 vs FT 0/12 (p~.005); @2500 FOLLOWS pref (sep +0.075). Also
  global R/B-swap finding (STARVLA_SWAP_RB toggle). hvlv@2500 topdown 50ep launched on box (watcher b84aih80l,
  ~4-5h). Gap doc §4c.3 added. Next box slot after 50ep: orient 50ep.
- 11:14Z pass: **ef5k@5000 early-frame BREAKTHROUGH: peak_eff 0.1064 (37x geom), follow 0.958 (25:0.917/75:1.0)**
  — contact conditioning solved offline by EF + longer schedule. ef_lr3 training 579/2500 (probe will follow).
  ef5k@5000 ckpt shipping to box (background). Box queue updated: hvlv50 (running, ep001+) -> contact ef5k
  closed-loop paired (test_num 11 first) -> orient 50ep -> place 50ep. Disk: H100 40%, box 68%.
- 11:38Z pass (event): ladder final — eflr3 HURT (0.0082/0.625), winner ef5k@5000 (0.1064/0.958); lever =
  early-frame weighting x schedule length, NOT LR. ef5k ckpt on box md5-verified (d79fbac1). hvlv early-frame
  probe @2500/@1500/b0 all ~chance — probe structurally blind to trajectory-shape prefs (detour accumulates
  across chunks); hvlv arbitration = closed-loop only (50ep in flight, ep001+). H100 GPUs free.
- 11:48Z pass: hvlv50 FAST (hv arm 50/50 done ~45min, lv arm running — full run likely done ~12:15Z).
  Box ckpts main_height/b0_height/main_orient already pruned by Agent A; disk 32G (robomme grew) — place
  pre-ship deferred until after hvlv50. Paper diff doc written (0612-paper-diff-suggestions.md). H100 idle.
- 12:18Z pass: hvlv50 hv arm DONE succ 17/51=33.3% (large-n regression from 4/6 sample; SPT 33% vs FT 0% holds);
  lv arm 45/50, ~10min out. Watcher will archive+compute; then box queue: contact ef5k closed-loop -> orient 50ep.
  Disk box 32G / H100 40%. H100 idle.
- 12:25Z pass (event): hvlv50 DONE — sep +0.0375, follow 0.59 fixed / 0.61 midpoint, succ 33.3%/19.6%
  (large-n regression vs 5ep sample; FT-success contrast holds, paper 75/82 not met). Archived 100 json.
  Contact ef5k@5000 closed-loop LAUNCHED on box (driver pid 3658752, watcher b29sd0i6a). Gap doc updated.
- 12:50Z pass: contact ef5k 25-arm in progress (server 12:28Z, first eps running). orient50 driver pre-staged
  on box (/root/drive_orient50_0611.sh: geom then b0, 51eps each). Box 32G / H100 idle 40%.
- 13:32Z pass: contact ef5k FIRST RUN WAS STUCK — box missing asset 068_boxdrink (same class as Agent A's
  100_seal find); eval spun 1h erroring per seed. Fixed: relayed 068_boxdrink+003_plate+107_soap from
  collection kempner assets via H100; contact ef5k RESTARTED 13:30Z; watcher v2 armed (b534j3lvr, 3h budget;
  old watcher b29sd0i6a may fire early on partial — ignore if so). place assets now pre-positioned too.
- 13:42Z pass (event): contact ef5k closed-loop DONE in 8min post-asset-fix — succ 81.8%/81.8%, BUT
  sep +0.0157m / midpoint-follow 0.65 (~geom@1500): plan-level EF conditioning does not survive
  per-chunk re-query execution. Gap doc §4c.4. orient50 LAUNCHED (geom then b0, watcher b2i69xxsi).
- 13:59Z pass: orient50 geom 0-arm 37/50 (fast, full done ~14:45 est). place ckpt pair shipping via GCS
  (background) + place50 driver staged (geom then b0, 51eps each). Box 32G — monitor when place lands (+19.6G).
- 14:31Z pass: orient50 GEOM arm DONE — succ 98.0%/96.1% (50/51, 49/51), 50 metric eps/arm; b0 arm running.
  place pair on box md5-verified. Box disk pruned (4 evaluated ckpts freed) 14G->50G. place50 fires after orient b0.
- 14:38Z pass (event): orient50 GEOM OFFICIAL — follow 1.00/1.00 @n=50 (90-arm 50/50 flipped), succ 98.0%/96.1%.
  b0 arm failed to start (ckpt pruned by Agent A) — reshipping (bg) + b0orient50 driver staged for after place50.
  place50 LAUNCHED (geom then b0). Box 50G.
- 15:02Z pass: place50 geom center-arm 50/50 done, corner-arm 6/50 running; b0_orient reshipped md5-OK.
  Chain bs80mr5gb owns place50->b0orient50->archive+compute. Box 41G. ETA: place50 ~16:10, b0orient50 ~16:45, 晨报 ~17:00.
- 15:34Z pass: place50 geom center succ 88.2% (45/51), corner 47/50 near done; b0_place arms next, then
  b0orient50. Chain ETA ~16:40, 晨报 after. Box 41G healthy.
- 16:06Z pass: place50 geom DONE (center succ 88.2%; corner succ 2.0% = known center-recipe artifact, metric
  disambiguates). b0_place center 44/50; then corner, then b0orient50. Chain ETA ~17:00. Box 41G.
- 16:37Z pass: b0_place center arm DONE succ 49.0% (25/51) vs geom 88.2% — FT weaker even on success for place.
  corner arm + b0orient50 remain (~50min). One more cycle then 晨报.
- 17:09Z pass: place50 OFFICIAL folded+committed — SPT 0.94 vs FT 0.48 (n=50, showcase axis). b0orient50
  90-arm finishing (~10min); 晨报 on chain completion.
- 17:32Z FINAL: b0orient50 DONE (follow 1.00/1.00 n=50, succ 98.0/84.3). ALL QUEUE ITEMS COMPLETE.
  晨报 written to gap doc §6. NIGHT CLOSED — no further wakeups will reschedule.
