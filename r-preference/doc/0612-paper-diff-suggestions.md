# Paper diff suggestions (v7 → v8) — 2026-06-12 morning

> Per your choice I did NOT touch the tex. Apply these to `paper_draft-v7.tex` (saved in
> repo). Items are ordered: mechanical fixes → numbers now backed by files → decisions
> with ready-made text for each option. Evidence pointers: `0611-paper-gap-analysis.md`
> (§4b–4d), `0612-hvlv-diagnosis.md`, `0612-privilege-free-labelers.md`,
> `eval/0529_50ep/summary.md`, `eval/0611_ctrl/`, `eval/source_follow_thresholds.json`.

## 1. Mechanical (apply as-is)

1. **Residual half-sentence** (Implementation details, end): delete
   `"  and the $\phi$ and $w(\cdot)$ vocabularies."` or restore the original
   `"The appendix lists all hyperparameters and the $\phi$ and $w(\cdot)$ vocabularies."`
2. **`\eg` macro** (Methods/Architecture): `(\eg, ``grasp low'')` → `(e.g., ``grasp low'')`
   unless corl_2026.sty defines `\eg` (v6 used plain "e.g."; compile will fail otherwise).
3. **Units of improvement**: abstract/intro/contribs say "improves … by 25.4\%" and
   "about 40 \%" — these are percentage **points**: revert to v6's "25.4 points" /
   "40 points" (or "from 59.4\% to 84.8\%").
4. **Bib**: ensure `nasiriany2024robocasa`, `mees2022calvin` exist in the .bib
   (newly cited in intro).

## 2. Numbers now backed by measurement (safe updates)

- **Table 2 height row**: claimed FT 53 / SPT 81 — measured closed-loop under the paper's
  own convention (fixed source-stats threshold 0.1245): **FT 0.55 (10 ep) / SPT 0.84 (49 ep)**;
  success FT 100%, SPT 91%. Keep claims or update to measured (recommend measured).
- **Metrics paragraph**: you already say "thresholded from source statistics" — we have the
  literal per-axis thresholds now (`source_follow_thresholds.json`: drop 0.1245 m, ee_x tilt
  41.2°, height-fraction 0.484, place-offset 0.0536 m, detour 0.1501 m). Consider citing
  them in the appendix. NB the archived per-run-midpoint numbers differ for weak arms
  (b0-height 0.80 midpoint vs 0.55 fixed) — the fixed convention is the defensible one.
- **hvlv row** (after tonight's 50ep finishes): following @2500 ckpt was 90% midpoint /
  70% expert-thr at n=5; success **SPT 6/12 vs Naive-FT 0/12** (Fisher p≈0.005) after the
  eval-camera fix. The 50-ep paired numbers land in `eval/0611_ctrl/hvlv_2500_topdown_50ep/`.
  hvlv SPT ckpt = steps_2500.

## 3. Decisions (text provided per option)

### 3.1 Recognition story (Table 1 + Methods) — RECOMMEND option A
**A. Privilege-free set** {contact 100 (EE), height 100 (VQA), hvlv 92 (EE), orient 100 (EE),
place 100 (vision)} → avg **98.4**. Method add-on sentence:
> "The recognition branch reads the preference through the VLM language head for the
> proprioceptive axes; for the relational axes we read it from features computed with the
> robot's own sensing — the end-effector trajectory, and for placement the receptacle pose
> estimated from the head camera by visual grounding and back-projection — with thresholds
> fit on the labeled source only. Low-margin episodes are abstained from and excluded by
> the Stage-B filter."
Also FIX the limitation sentence "the predictor never abstains" — the new labelers DO
abstain (reject bands; coverage 96–98%, post-filter 1.00).
**B. Keep v7 numbers** {…, place 90} → then the place Stage-B ckpt reported must be the
token-label-trained `pref_stageb_main_place` (it exists, offline follow 1.00) for
consistency, and method must still disclose the geom/feature labelers for contact/hvlv.

### 3.2 Orient row — measured FT=1.00 at n=10 on move_can5_away (b0 fully retains source
conditioning; orient_geom SPT also 1.00). Options: (i) await the 50-ep extension (queued)
and report measured; (ii) reframe: "on the discrete grasp-orientation axis both FT and SPT
reach ceiling — source conditioning survives naive finetuning for motor-primitive-like
preferences (place < height < orient gradient) — and SPT's advantage concentrates on the
relational axes and on label quality"; (iii) different orient target task (collection
effort, cherry-pick risk). RECOMMEND (i)+(ii).

### 3.3 New ablation row (strong, all files on disk): **label quality → controllability**
place: labels 0.61 (vanilla VLM) → follow 0.75; 0.90 (token) → 1.00 offline; 1.00 (geom)
→ 1.00. orient: labels 0.95 → 0.75 closed-loop vs 1.00 → 1.00 closed-loop (same seeds).
This directly supports "following tracks recognition" with a causal experiment.

### 3.4 Contact row — pending tonight's closed-loop of `contact_geom_ef5k@5000`
(early-frame-weighted Stage-B: prompt effect ×37, early-frame follow 0.958). If closed-loop
confirms: add one method sentence:
> "On axes where mid-episode observations already reveal the executed preference, we
> over-sample pre-commitment frames during Stage-B so the action loss is dominated by
> states where only the prompt disambiguates the preference."
If not confirmed: keep v7's 90 as target and flag.

## 4. New facts worth a sentence in the paper

- **Eval-camera embodiment for hvlv** (appendix/setup): hvlv uses the top-down head camera
  embodiment for BOTH data and evaluation (this was the root cause of earlier hvlv failures).
- **R/B channel swap**: training images are R/B-swapped by the collection encoder across
  all axes; deployment matches the training distribution via a channel-swap at the bridge
  (and should be fixed in the collector). One honesty sentence in the appendix.
