# DiscreteRTC → StarVLA: Codebase Cleaning & PR Preparation Plan

This is the operational checklist for turning our research fork
(`Kaiwen-Hong/starVLA @ starVLA`) into a series of clean, reviewable PRs
against `starVLA/starVLA`.

It is intentionally pragmatic: we keep our research branch intact, build a
**separate** clean branch on top of the latest upstream, and hand-port only
what is needed for the contribution described in
`DOC-pr/starvla_discretertc_integration_summary.md`.

> **Sources of truth for upstream rules** (read these before opening
> anything):
> - `starVLA/starVLA :: docs/PR_readme.md` (PR template, `make check`,
>   commit format, framework-addition requirements)
> - `starVLA/starVLA :: docs/branching_strategy.md` (target branch,
>   branch naming, file-isolation rule)
> - The README §"Contributing" + the Cooperation Form
>   (`forms.gle/R4VvgiVveULibTCCA`)

---

## 0. Repo state at the time of writing

```
origin    https://github.com/Kaiwen-Hong/starVLA.git   (our private fork)
upstream  https://github.com/starVLA/starVLA.git       (target repo)

research branch : starVLA           (off origin/starVLA, HEAD 9228d95 "running_fm")
clean branch    : feat/discrete-rtc (off upstream/starVLA_dev, HEAD b71b48c)
```

`git diff --stat upstream/starVLA_dev...starVLA` shows **~644 changed
files, ~132k insertions**, of which only ~28 files (~4.6k insertions) sit
inside the `starVLA/` Python package. The rest is research scaffolding
(real-world UR5 scripts, internal docs, analysis notebooks, model
artifacts, `chenserver-files/`). **None of that goes upstream.**

---

## 1. Upstream rules that constrain our plan

These are non-negotiable upstream guidelines we discovered after reading
`starVLA/starVLA :: docs/PR_readme.md` and `docs/branching_strategy.md`:

1. **Target branch is `starVLA_dev`**, never `starVLA`. The stable
   `starVLA` branch only receives squash-merged promotions.
2. **Branch naming**: `feat/`, `fix/`, `docs/`, `refactor/`, `exp/`,
   `hotfix/`. Lowercase, hyphenated. Our branch is `feat/discrete-rtc`.
3. **PR size budget**: "ideally under 500 lines, one logical change per
   PR." Our combined contribution (head + runtime) is ~1.5k LOC, so we
   **must split** (see §6).
4. **`make check`** runs `black --check .` and `ruff check
   --show-source .`. Required to pass on the files our PR touches (the
   repo has historical lint backlog, so we don't fix unrelated files).
5. **Commit message format**: Conventional Commits, ≤72 char subject,
   `Closes #<issue>` reference. PRs are **squash-merged**, so the PR
   title becomes the final commit subject.
6. **File isolation**: "new frameworks / benchmarks / datasets should be
   self-contained." Modifying shared modules requires explicit
   justification. → keep all our changes inside new files where possible.
7. **Open an Issue first.** Quoted from `docs/PR_readme.md`:
   *"Open an Issue first → branch from `starVLA_dev` → submit PR with
   clear description → address review feedback."* Issue precedes PR.
8. **New-framework PRs require all of the following** (per
   `docs/PR_readme.md` §"Special Requirements for Framework/Dataset
   Additions"):
   - Register in framework config system (we already do this via
     `FRAMEWORK_REGISTRY.register("QwenDiscreteDiffusion")`).
   - Default config dataclass / YAML.
   - Example YAML under `examples/<benchmark>/train_files/`.
   - Update `docs/model_zoo.md`.
   - **Provide benchmark results + a public HuggingFace checkpoint.**
     This is mandatory, not optional.
9. **No secrets, API keys, or large binary files** in any PR.
10. **Inactive PRs (>14 days) may be closed** — keep momentum after
    opening.

---

## 2. Branching strategy

```
upstream/starVLA_dev  ─── (latest, b71b48c)
                              └── feat/discrete-rtc           ← clean base for our work (already created)
                                    ├── feat/discrete-rtc-utils      (PR 1a: discrete_diffusion/ utilities)
                                    ├── feat/discrete-rtc-head       (PR 1b: action head + framework, depends on 1a)
                                    ├── feat/discrete-rtc-runtime    (PR 2: RTC runtime, depends on 1b)
                                    └── feat/continuous-rtc          (PR 3, optional: FM RTC)

origin/starVLA        ─── our research history (do NOT delete, do NOT rebase, do NOT PR)
```

Concrete state already on disk:

```bash
$ git branch
  starVLA              # research branch, untouched
* feat/discrete-rtc    # off upstream/starVLA_dev @ b71b48c, no tracking set
```

Tracking is intentionally **unset** on `feat/discrete-rtc` so a stray
`git push` cannot target either remote. We bind it to `origin` only when
we explicitly push (`git push -u origin feat/discrete-rtc`), and we
**never** push branches to `upstream`.

> **Why not cherry-pick from `starVLA` directly?** Our commits on
> `starVLA` are interleaved (`running_fm`, `update changes`, `fixing_fm`,
> …) and touch unrelated research code. Cherry-picking would drag in
> noise and break atomicity. A hand-ported clean branch with a small
> number of well-scoped commits is much easier to review.

---

## 3. What ships, what doesn't

### 3a. Ship (PR-relevant) — code

| File | Status | Notes |
|---|---|---|
| `starVLA/model/modules/action_model/discrete_diffusion/__init__.py` | new | re-exports |
| `starVLA/model/modules/action_model/discrete_diffusion/action_binning.py` | new | 172 LOC, per-dim quantizer |
| `starVLA/model/modules/action_model/discrete_diffusion/mask_git_schedule.py` | new | 87 LOC, cosine + topk masking |
| `starVLA/model/modules/action_model/discrete_diffusion/models.py` | new | 106 LOC, DiscreteDiT wrapper |
| `starVLA/model/modules/action_model/LayerwiseDiscreteDiffusion_ActionHeader.py` | new | **production head**, 493 LOC, includes `predict_action_realtime` |
| `starVLA/model/framework/QwenDiscreteDiffusion.py` | new | 321 LOC, registers via `FRAMEWORK_REGISTRY` |
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | modified | only the `predict_action_realtime` block (≈+249 LOC) — for the optional ContinuousRTC PR |
| `starVLA/model/framework/QwenPI.py` | modified | only the `predict_action_realtime` block (+75 LOC) — for the optional ContinuousRTC PR |
| `examples/<benchmark>/train_files/starvla_train_discrete_diffusion.yaml` | new | example train config (path renamed to match upstream's `examples/<benchmark>/train_files/` convention required by `docs/PR_readme.md`) |
| `docs/model_zoo.md` | modified | required: add the QwenDiscreteDiffusion row + HF checkpoint link |

> **Decision:** `DiscreteDiffusion_ActionHeader.py` (the 509-LOC
> non-layerwise reference impl) is **dropped** for the upstream PR.
> Justification: 500-LOC PR budget, file-isolation rule, and we always
> have it on the research branch. We can revisit as a follow-up.

### 3b. Ship (PR-relevant) — runtime to be **extracted**

The actual RTC executor logic is currently embedded in our real-world
deployment scripts:

```
ur5n/dd/closedloop_rtc_v6.py        ← DiscreteRTC, threaded servo + inference
ur5n/fm/closedloop_rtc_v6.py        ← ContinuousRTC version
```

These files are 1300–1400 LOC each and **80%+ is UR5e / Robotiq /
matplotlib / `modular_policy` glue that does not belong upstream**. The
strategy is:

1. Extract the **runtime-agnostic core** (state machine: `prev_chunk`,
   `inference_delay d`, `execution_horizon s`, prefix copy, mask
   construction, `predict_action_realtime` call, action-buffer update)
   into a new module:

   ```
   starVLA/runtime/rtc/
     __init__.py
     base_executor.py        # BaseRTCBackend / RTCExecutor interface
     discrete_backend.py     # masked-token inpainting via DD head
     continuous_backend.py   # ΠGDM-style FM inpainting (optional PR)
   ```

   Hardware-side glue (servo loop, gripper, matplotlib, cv2, scipy
   transforms) **stays in our private repo**, not in the PR.

2. Add a tiny standalone smoke-test mirroring StarVLA's
   `python starVLA/model/framework/QwenGR00T.py` convention:

   ```
   starVLA/runtime/rtc/__main__.py    # fake-data smoke test
   ```

3. Provide one optional integration example under `examples/RTC/`:

   ```
   examples/RTC/README.md
   examples/RTC/eval_files/run_rtc_demo.py     # uses fake or LeRobot replay
   ```

   No real-robot drivers, no UR5e-specific imports.

> **Caveat re: file-isolation rule.** A new top-level
> `starVLA/runtime/rtc/` directory is technically a new module, not a
> change to a shared one — but it's not a `framework/`, `dataloader/`,
> or `examples/` either. We need to confirm in the design Issue that
> the maintainers accept this location, or relocate to
> `starVLA/model/framework/rtc/` if they prefer.

### 3c. Do not ship — top-level research scaffolding

Everything below stays on `starVLA` (our research branch) and never
touches the PR branch:

```
ur5/, ur5n/, realworld/, deployment/         # real-robot drivers + 1300-LOC closed-loop scripts
analysis/, visualization/, playground/        # plotting + checkpoints
chenserver-files/                             # mirror of files for our cluster
r-preference/                                 # internal training campaign logs
DOC-data/, DOC-debug/, DOC-reference/,
  DOC-setup/, DOC-training/, DOC-context.md,
  DOC-pr/                                     # internal research docs (this file lives here too)
0130-train-libero.sh, 0402-lesson.md,
  2506.07339v2.pdf, important.md,
  TEMP-solve-merge-issues.md,
  TEMP-upload-checkpoint-hf-starvla.sh        # ad-hoc artifacts
examples/Franka/, examples/FastUMI/...        # other tracks, unrelated to RTC
```

### 3d. Do not ship — incidental package-level changes

These are real diffs vs upstream but unrelated to the contribution. The
file-isolation rule reinforces dropping these — each is a touch on a
shared file that needs its own justification. Either drop or open as
**separate** small PRs later.

| File | What it is | Action for this PR |
|---|---|---|
| `starVLA/dataloader/gr00t_lerobot/datasets.py` (+555 LOC) | action-mode cache, FastUMI/RobotwinEE plumbing, stats cache versioning | drop; not needed for DD training (the head owns its own quantization) |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` (+183 LOC) | FastUMIDataConfig + RobotwinEEDataConfig | drop |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` (+276 LOC) | unrelated mixture configs | drop |
| `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` (+45 LOC) | new normalization modes | verify whether DD config references one of them; if no, drop |
| `starVLA/dataloader/__init__.py`, `starVLA/training/train_starvla.py` (`dist.is_initialized()` guards) | small robustness fixes | drop or split into a tiny "robustness fixes" PR — not in scope here |
| `starVLA/training/train_starvla.py` (HF auto-upload, secrets scrubber, `--no_deepspeed`) | our own convenience tooling | drop |
| `starVLA/model/framework/QwenOFT.py` (+232 LOC, attention capture) | env-var-controlled attention-map dumper | drop |
| `starVLA/model/framework/ABot_M0.py`, `LangForce.py`, `M1.py` mods | other tracks | drop |
| `starVLA/model/modules/action_model/AML_ActionHeader.py` | unrelated head | drop |
| `starVLA/model/modules/vlm/*.py` (small mods) | minor compatibility shims | verify; if needed by DD framework, include the **minimal** subset |
| `starVLA/model/framework/base_framework.py` (+64 LOC), `starVLA/model/tools.py` (+168 LOC) | utility expansions | include only the lines actually called by `QwenDiscreteDiffusion` / `LayerwiseDiscreteDiffusion_ActionHeader`; drop the rest |

### 3e. Do not ship — secrets

We must double-check none of the following leak into the PR branch:

- `hf_token`, `wandb_*` keys in any YAML or env file
- Cluster paths (`/n/holyscratch01/...`, `chenserver`, etc.)
- Personal HuggingFace org names beyond what is already documented
- Real-world checkpoints (.pt / .safetensors)

Grep before pushing:

```bash
git grep -nE 'hf_token|HF_TOKEN|wandb.*api|/n/holyscratch|chenserver' feat/discrete-rtc -- 'starVLA/**' 'examples/RTC/**' 'docs/**'
```

---

## 4. Hand-port procedure (per-file, per-commit)

For each file in §3a / §3b:

1. `git checkout feat/discrete-rtc-utils` (or whichever sub-branch we're
   currently filling).
2. `git checkout starVLA -- <file>` to copy the file from our research
   branch.
3. Open it and **scrub research-only references**:
   - imports of `modular_policy`, `robotiq_gripper`, `ur5*`,
     `chenserver-files`, `realworld`, `deployment.*` not in upstream;
   - hardcoded paths;
   - `_save_attention`-style env-var debug branches that bleed in from
     other PRs;
   - dead `ref_dd_mode.py` comments.
4. Run `black --check` and `ruff check --show-source` on the file.
5. Run the standalone smoke test the file is supposed to support
   (`python starVLA/model/framework/QwenDiscreteDiffusion.py` etc.) on a
   single GPU with fake data — this is StarVLA's documented contract for
   each framework file (README §"Smoke test any submodule").
6. Stage with a Conventional-Commits-style message:
   - `feat(action_model): add discrete diffusion utilities (binning, mask schedule, DiT)`
   - `feat(action_model): add LayerwiseDiscreteDiffusion action head`
   - `feat(framework): add QwenDiscreteDiffusion`
   - `feat(runtime): add RTC executor and discrete backend`
   - `feat(runtime): add continuous (flow-matching) RTC backend`
   - `docs(model_zoo): add QwenDiscreteDiffusion row`
   - `docs(rtc): add example and README`

Squash-merge means our internal commit history disappears at merge time,
but reviewers still read it during review — so commits should still
compile and pass lint individually.

---

## 5. Open scoping decisions to resolve in the design Issue

We surface these in the Issue (see `cooperation_form_and_email.md` §3)
**before** opening any PR:

1. **`runtime/rtc/` location** — accept new `starVLA/runtime/` top-level
   directory? Or relocate under `starVLA/model/framework/rtc/`?
2. **`DiscreteDiffusion_ActionHeader` (509 LOC, naive non-layerwise)** —
   recommendation: drop for first PR; resurrect later if asked.
3. **DiscreteRTC partial-token preservation** — ship simple
   committed-prefix mode only; advanced natural-guidance mode as a
   follow-up.
4. **ContinuousRTC inclusion** — recommendation: separate PR (PR 3).
5. **Tokenization ownership** — keep quantization inside the action
   head; no dataloader changes.
6. **HF checkpoint scope** — does upstream want Bridge-RT-1, LIBERO, or
   both? The ask in `docs/PR_readme.md` is "at least one benchmark
   result + public HF checkpoint."

---

## 6. Pre-PR checklist

Run on each clean sub-branch before opening:

- [ ] Branch is off the latest `upstream/starVLA_dev`, not `starVLA`.
- [ ] Branch name matches `feat/...` per
      `docs/branching_strategy.md`.
- [ ] `git diff --stat upstream/starVLA_dev...HEAD` is **≤500 LOC**
      (upstream's stated rule). PRs over budget go in a comment in the
      Issue with explicit justification.
- [ ] `git grep -nE 'TODO\(kaiwen\)|FIXME|XXX|chenserver|ur5n|realworld'`
      returns nothing.
- [ ] No `.pt`, `.safetensors`, `.pdf`, `.npy`, `.zip` tracked.
- [ ] `make check` passes on the touched files
      (`black --check <files>` and `ruff check --show-source <files>`).
- [ ] Smoke tests run on 1 GPU with fake data:
  - `python starVLA/model/framework/QwenDiscreteDiffusion.py`
  - `python -m starVLA.runtime.rtc` (fake-data RTC smoke test)
- [ ] One end-to-end LIBERO training step runs with the new YAML.
- [ ] Secret-scrub grep (§3e) is clean.
- [ ] Every new file carries StarVLA's standard MIT header.
- [ ] **HF checkpoint published** (PR 1b only):
      `StarVLA/Qwen-DiscreteDiffusion-<benchmark>` is public, weights
      load via `from_pretrained`, README on the model card includes the
      eval table from the integration summary.
- [ ] `docs/model_zoo.md` row added (PR 1b only).
- [ ] PR description follows `docs/PR_readme.md` template:
      Motivation / Changes / Testing / Breaking Changes / Screenshots.

---

## 7. PR breakdown (revised — per upstream's ≤500-LOC rule)

| # | Branch | Scope | Approx LOC | Depends on |
|---|---|---|---|---|
| 1a | `feat/discrete-rtc-utils` | `starVLA/model/modules/action_model/discrete_diffusion/` (binning + mask schedule + DiscreteDiT) | ~387 | — |
| 1b | `feat/discrete-rtc-head` | `LayerwiseDiscreteDiffusion_ActionHeader.py` + `QwenDiscreteDiffusion.py` + example YAML + `docs/model_zoo.md` row + HF checkpoint link | head:493 + framework:321 + yaml ≈ 850 (over budget; flag in Issue) | 1a |
| 2  | `feat/discrete-rtc-runtime` | `starVLA/runtime/rtc/` (base + discrete) + smoke test + `examples/RTC/` | ~400 | 1b |
| 3  | `feat/continuous-rtc` (optional) | `predict_action_realtime` on FM head + QwenPI + `runtime/rtc/continuous_backend.py` | ~500 | 2 |
| 4  | `docs/discrete-rtc` (only if maintainers want it separately) | README section, demo notes, GIFs | small | 3 |

**On the 850-LOC PR 1b:** that exceeds the ≤500-LOC rule. Two options:

- **Option A (preferred):** flag this in the design Issue, ask for an
  exception because action-head + framework should be reviewed together
  to be evaluable. This is a one-paragraph ask, not a fight.
- **Option B (fallback):** further split into
  `feat/discrete-rtc-head-only` (action head, lands first, smoke-tested
  with a stub framework script) and
  `feat/discrete-rtc-framework` (QwenDiscreteDiffusion wraps the head).
  More PRs, more review cycles, but each <500 LOC.

We default to A; if maintainers push back during the Issue stage, we
fall back to B before opening anything.

---

## 8. After the PR is open

- Mirror the PR description on our `Kaiwen-Hong/starVLA` README so
  external collaborators land on the right page.
- Keep each `feat/discrete-rtc-*` rebased onto `upstream/starVLA_dev`
  weekly — it moves fast (Qwen3.5 backbone, Bridge-RT-1, LIBERO-plus,
  RoboTwin updates, recent QwenPI_v3 in #293, RoboCasa365 in #300).
- Address review feedback within 14 days to avoid auto-close per
  `docs/PR_readme.md`.
- Once merged, mark `DOC-pr/` as historical and remove `TEMP-*` files
  from the research branch in a separate housekeeping commit.
