# StarVLA Cooperation: Form Responses + Outreach Email

The StarVLA contribution flow is documented in three places, all of which
say the same thing: **align scope first, then PR**.

- README §"Contributing":
  > "If you have ideas to improve StarVLA, feel free to open a PR. To
  > make sure we will accept your effect, please align scope and design
  > first via an Issue or by booking a short sync with this Cooperation
  > Form."
- `docs/PR_readme.md` (on `starVLA_dev`):
  > "Open an Issue first → branch from `starVLA_dev` → submit PR with
  > clear description → address review feedback."
- `docs/branching_strategy.md` (on `starVLA_dev`):
  > "All contributions start as feature branches forked from
  > `starVLA_dev`."

So the right ordering is:

1. **Submit the Cooperation Form** with the proposal in §1 below.
2. **Open a tracking GitHub Issue** the same day (template in §3) on
   `starVLA/starVLA`, linking to the form submission. The Issue is
   what `docs/PR_readme.md` requires before any PR — **it is not
   optional**.
3. **Optionally** send the email in §4 to a maintainer if we have a
   direct contact (e.g. the names on the README's Calvin announcement)
   and want to land a sync before the next Friday office hours.
4. **Only after the design Issue gets a green light**, branch
   `feat/discrete-rtc-utils` off `upstream/starVLA_dev` and open PR 1a.

Do not open the PR before steps 1–3. Do not target `starVLA` (the stable
branch). Always target `starVLA_dev`.

---

## 1. Cooperation Form — copy/paste-ready answers

The Google form fields will likely be a subset of the questions below. We
keep one short answer per field plus one longer reusable paragraph.

### Field: *Your name & affiliation*

```
Kaiwen Hong (kaiwenh.17@gmail.com), [your affiliation here].
Co-authors on the DiscreteRTC research: [list co-authors / advisors].
```

### Field: *What would you like to contribute / discuss?*

```
We would like to contribute an asynchronous-execution extension to
StarVLA, centered on two modular components:

  (1) a discrete diffusion (MaskGIT-style) action head that performs
      non-autoregressive masked-token action generation, and
  (2) a native DiscreteRTC inference runtime that reuses the head's
      inpainting capability for asynchronous real-time chunk execution.

Optionally, we are also prepared to contribute a ContinuousRTC backend
for the existing flow-matching policies (QwenPI / QwenGR00T) so that
StarVLA can offer a unified RTC interface across continuous and discrete
action heads.

This is a modular contribution, not a request to merge an entire research
codebase. The discrete head fits StarVLA's existing action_model
abstraction, the framework registers via FRAMEWORK_REGISTRY just like
QwenPI / QwenGR00T, and the RTC runtime is an inference-time wrapper
around predict_action / predict_action_realtime — no required changes to
training infrastructure or dataloaders. This is consistent with the
file-isolation rule in docs/branching_strategy.md.
```

### Field: *Why do you think this fits StarVLA?*

```
StarVLA already supports MLP regression (QwenOFT), autoregressive token
decoding (QwenFast), and flow-matching (QwenPI, QwenGR00T) as
action-generation paradigms. A discrete diffusion head is the missing
fourth quadrant — non-autoregressive masked-token decoding — and it
slots in cleanly under starVLA/model/modules/action_model/ with the same
forward / generate contract.

For deployment, StarVLA's docs note that predict_action works directly
on raw inputs and that frameworks expose a standard inference surface.
RTC fits that surface as a thin executor: it copies the committed prefix
from the previous chunk into the next inference call's token sequence,
masks the rest, and runs standard unmasking — so async execution becomes
a runtime concern, not a model-architecture concern.

Discrete diffusion is structurally well-suited for RTC because masked-
token inpainting is its native operation: no ΠGDM-style guidance, no
RTC-specific fine-tuning, no inference-time correction schedule.
```

### Field: *What is the scope?*

```
Per docs/PR_readme.md (≤500 lines per PR, one logical change per PR),
we plan to split the contribution into a sequence of small PRs against
starVLA_dev:

  PR 1a (feat/discrete-rtc-utils):
    Discrete diffusion utilities (per-dim binning, MaskGIT cosine mask
    schedule, DiscreteDiT wrapper). ~390 LOC, no framework change.

  PR 1b (feat/discrete-rtc-head):
    LayerwiseDiscreteDiffusion action head + QwenDiscreteDiffusion
    framework + example YAML under examples/<benchmark>/train_files/ +
    docs/model_zoo.md row + public HF checkpoint. ~850 LOC.

    Note on size: action head and framework are tightly coupled and we
    think they review better as one unit. If you'd prefer two separate
    PRs we'll split into "head only" + "framework wraps head" — flagged
    here so we can decide together.

  PR 2 (feat/discrete-rtc-runtime):
    DiscreteRTC inference runtime: starVLA/runtime/rtc/ with a
    BaseRTCBackend interface and a DiscreteRTCBackend, plus a fake-data
    smoke test and an example. ~400 LOC.

  PR 3 (feat/continuous-rtc, optional, gated on your interest):
    ContinuousRTC backend for QwenPI / QwenGR00T (ΠGDM-style prefix-
    conditioned flow-matching inpainting), so users can compare
    synchronous, naive async, ContinuousRTC, and DiscreteRTC under one
    deployment stack. ~500 LOC.

Total expected diff against upstream/starVLA_dev for PR 1a + 1b + 2:
~25–30 files, ~1.6k insertions, all under starVLA/, examples/RTC/, and
one row in docs/model_zoo.md. No dataloader or training-script changes.
```

### Field: *Have you tested it?*

```
Yes. The DiscreteRTC paper reports:

  - Simulated benchmark (Kinetix dynamic control): DiscreteRTC
    consistently outperforms ContinuousRTC, bidirectional decoding, and
    naive async execution across inference delays.

  - Real-world (UR5e + Robotiq, wrist RGB, Qwen2.5-VL-3B-Instruct
    backbone, layerwise cross-attention DiT head, single RTX 4090,
    20 Hz control / 100 Hz servo via 5x interpolation):

      Method            Dyn. Place   Dyn. Pick    Inference Time
      Continuous Sync   0%           0%           151 ms
      Discrete Sync     0%           0%           303 ms
      Continuous RTC    90%          45%          256 ms
      DiscreteRTC       100%         95%          206 ms

We have a working implementation in our private fork
(github.com/Kaiwen-Hong/starVLA, branch starVLA) and have already set up
a clean feat/discrete-rtc base off upstream/starVLA_dev to hand-port the
upstream-relevant subset.

For the mandatory HF checkpoint asked for in docs/PR_readme.md, we will
publish StarVLA/Qwen-DiscreteDiffusion-<benchmark> with weights, model
card, and the eval table above before opening PR 1b.
```

### Field: *Open questions for the maintainers*

```
Before we open the PR, we'd like to confirm:

  1. Where should the RTC runtime live? Our default is a new
     starVLA/runtime/rtc/ module with a BaseRTCBackend interface and
     per-backend subclasses, but we are happy to put it under
     starVLA/model/framework/rtc/ if that better matches your file-
     isolation conventions.

  2. The PR-1b diff (action head + framework + example YAML + HF
     checkpoint) is ~850 LOC, slightly over the docs/PR_readme.md
     500-line guideline. We think head+framework should review together;
     we can split into two PRs if you prefer. Which do you want?

  3. Are you open to a separate optional PR adding ContinuousRTC for
     QwenPI / QwenGR00T (PR 3), or should we keep the contribution
     focused on the discrete head only?

  4. For the example, would you prefer a fake-data smoke test only, or
     a LIBERO / RoboCasa replay-based demo? We avoid bundling real-
     robot drivers (UR5e, Robotiq) in the PR.

  5. For the mandatory HF checkpoint required by docs/PR_readme.md,
     which benchmark would you like us to publish first — Bridge-RT-1
     (matches your existing Qwen-FAST/OFT/PI/GR00T-Bridge-RT-1
     checkpoints), or LIBERO?

  6. Does Friday office hours work for a 30-minute design review of
     PR 1a/1b before we open them?
```

### Field: *Anything else?*

```
- We will keep our PR branches rebased onto upstream/starVLA_dev so
  review can proceed in parallel with your active work (Qwen3.5
  backbone, QwenPI_v3, RoboCasa365, LIBERO-plus, RoboTwin 2.0).
- We can provide a 1–2 minute video of the real-world dynamic-pick
  demo on request.
- License: MIT, matching StarVLA's LICENSE.
```

---

## 2. The one-paragraph abstract (for any field that wants a short blurb)

```
We propose adding RTC-style asynchronous execution to StarVLA via two
modular components: a MaskGIT-style discrete diffusion action head and a
native DiscreteRTC inference runtime that reuses the head's masked-token
inpainting for prefix-conditioned chunk continuation. The runtime copies
the committed prefix from the previous chunk, masks the suffix, and runs
standard unmasking — no ΠGDM correction, no RTC-specific fine-tuning. We
will optionally also contribute a ContinuousRTC backend for existing
flow-matching policies so users can compare sync, naive async, flow-RTC,
and DiscreteRTC under one deployment stack. Everything fits StarVLA's
modular framework / action_head / predict_action interfaces and respects
the file-isolation rule in docs/branching_strategy.md.
```

---

## 3. Tracking GitHub Issue (template, target = `starVLA/starVLA`)

Open this on `starVLA/starVLA` the same day the form is submitted. This
is the Issue that `docs/PR_readme.md` requires before any PR.

**Title**

```
[Proposal] Discrete diffusion action head + DiscreteRTC asynchronous
inference runtime
```

**Body**

```markdown
## Summary

We submitted a Cooperation Form proposing a modular contribution against
`starVLA_dev`:

1. A discrete diffusion (MaskGIT-style) action head, fitting under
   `starVLA/model/modules/action_model/`.
2. A native DiscreteRTC asynchronous inference runtime, exposed via a
   new `starVLA/runtime/rtc/` module.
3. (Optional) A ContinuousRTC backend for existing flow-matching
   policies (QwenPI / QwenGR00T), so StarVLA can offer a unified async-
   execution interface.

## Why

Action-chunking VLA policies hit an asynchronous-execution problem on
real robots: model latency typically exceeds the control period, so the
robot must act while the next chunk is being generated. RTC handles
chunk transitions as inpainting: freeze the committed prefix, generate
the suffix conditioned on it.

For flow-matching, RTC needs ΠGDM-style guidance and additional fine-
tuning. For discrete diffusion, masked-token inpainting is the native
operation — no extra training, no inference-time correction.

Real-world results (UR5e, single RTX 4090):

| Method            | Dyn. Place | Dyn. Pick | Inference |
|-------------------|-----------:|----------:|----------:|
| Continuous Sync   |  0%        |  0%       | 151 ms    |
| Discrete Sync     |  0%        |  0%       | 303 ms    |
| Continuous RTC    | 90%        | 45%       | 256 ms    |
| DiscreteRTC       | 100%       | 95%       | 206 ms    |

## Proposed PR breakdown (matches docs/PR_readme.md ≤500-LOC rule)

- **PR 1a** `feat/discrete-rtc-utils` — discrete diffusion utilities
  (binning, mask schedule, DiscreteDiT). ~390 LOC. No framework change.
- **PR 1b** `feat/discrete-rtc-head` — LayerwiseDiscreteDiffusion
  action head + QwenDiscreteDiffusion framework + example YAML +
  `docs/model_zoo.md` row + public HF checkpoint. ~850 LOC (size note
  in §"Open questions").
- **PR 2** `feat/discrete-rtc-runtime` — `starVLA/runtime/rtc/` (base +
  discrete backend) + fake-data smoke test + `examples/RTC/`. ~400 LOC.
- **PR 3 (optional)** `feat/continuous-rtc` — ContinuousRTC backend
  for QwenPI / QwenGR00T. ~500 LOC.

Expected total diff against `upstream/starVLA_dev` for PR 1a+1b+2:
~25–30 files, ~1.6k insertions, all under `starVLA/`, `examples/RTC/`,
and one row in `docs/model_zoo.md`. No dataloader or training-script
changes are required.

All branches are off `upstream/starVLA_dev`, named per
`docs/branching_strategy.md`. We will run `make check` (black + ruff) on
modified files only, per `docs/PR_readme.md`.

## Open questions before we open PR 1a

1. RTC runtime location: `starVLA/runtime/rtc/` (default) or
   `starVLA/model/framework/rtc/`?
2. Is the PR-1b 850-LOC size acceptable as one PR (head+framework
   reviewed together), or should we split into two ≤500-LOC PRs?
3. Should we include the optional ContinuousRTC PR (PR 3) in scope?
4. Demo style: fake-data smoke test only, or LIBERO/RoboCasa replay?
5. HF checkpoint benchmark target: Bridge-RT-1 or LIBERO?
6. Can we book a 30-min design review at the next Friday office hours
   before opening PR 1a?

## Status

Working implementation lives on our private fork
(`github.com/Kaiwen-Hong/starVLA`, research branch `starVLA`). A clean
hand-port branch (`feat/discrete-rtc`) is already set up off
`upstream/starVLA_dev`. We will not open any PR until the questions
above are answered.

## License

MIT, matching StarVLA's LICENSE.

cc: @starVLA-maintainers
```

---

## 4. Outreach email (optional, only if we have a direct contact)

Use this only if we have a name + email from the README (e.g. the Calvin
contacts: Zhijie Song <1600013008@pku.edu.cn> or Feng Yan
<bphengyan@163.com>). Otherwise the form + Issue is enough.

**Subject**

```
StarVLA contribution proposal: discrete diffusion action head + DiscreteRTC
```

**Body**

```
Hi <Name>,

I'm Kaiwen Hong (kaiwenh.17@gmail.com). I work on asynchronous execution
for action-chunking VLA policies, and I'd like to propose a modular
contribution to StarVLA.

The contribution has two parts (plus one optional):

  1. A MaskGIT-style discrete diffusion action head, slotting in under
     starVLA/model/modules/action_model/ alongside the existing MLP /
     fast / flow-matching heads.
  2. A native DiscreteRTC inference runtime: an executor that copies
     the committed prefix from the previous action chunk into the next
     inference call's token sequence, masks the suffix, and runs
     standard unmasking. No ΠGDM correction, no RTC-specific fine-
     tuning — masked-token inpainting is the head's native operation.
  3. (Optional) A ContinuousRTC backend for QwenPI / QwenGR00T, so
     StarVLA can support a unified async-execution interface for both
     discrete and flow-matching policies.

We've validated this on a real UR5e with a wrist camera and Qwen2.5-VL-
3B-Instruct on a single RTX 4090. On dynamic-pick / dynamic-place tasks
the synchronous baselines fail (0%); ContinuousRTC reaches 45% / 90%;
DiscreteRTC reaches 95% / 100% while running ~50 ms faster than
ContinuousRTC per inference.

I've already submitted the Cooperation Form and opened a tracking issue
on the repo: <issue link>. The plan respects docs/PR_readme.md
(≤500-LOC PRs, branching off starVLA_dev, make check on touched files,
mandatory HF checkpoint for new frameworks) and is split into:

  - PR 1a: discrete diffusion utilities (~390 LOC)
  - PR 1b: action head + framework + HF checkpoint (~850 LOC, size
    flagged in the issue for discussion)
  - PR 2:  RTC runtime under starVLA/runtime/rtc/ (~400 LOC)
  - PR 3 (optional): ContinuousRTC backend (~500 LOC)

Before opening any PR, I'd like to align on the design choices listed
in the issue (RTC runtime location, PR-1b size, scope of the optional
ContinuousRTC PR, demo style, HF checkpoint benchmark target).

Would the next Friday office hours work for a 30-minute design review?
If a different time works better I'm happy to fit your schedule.

Two longer reference documents are linked below for context:

  - Integration summary (proposal + design choices):
    DOC-pr/starvla_discretertc_integration_summary.md
  - DiscreteRTC paper: <arXiv link>

Thanks for considering it,
Kaiwen
```

---

## 5. Internal pre-flight before sending anything

- [ ] All three sources (paper, integration summary, this file) cite
      the same numbers for the real-world table.
- [ ] No private email addresses in the issue body (only ours).
- [ ] No internal repo paths (`/n/holyscratch01/...`,
      `chenserver-files`) in the issue body or the email.
- [ ] Affiliation line at the top of §1 is filled in.
- [ ] Co-author list at the top of §1 is filled in and matches the
      paper.
- [ ] arXiv link is added to the email §4 before sending.
- [ ] Cooperation Form URL still resolves
      (`forms.gle/R4VvgiVveULibTCCA`).
- [ ] `docs/PR_readme.md` and `docs/branching_strategy.md` on
      `starVLA_dev` haven't changed since this draft was written
      (re-read both before sending).
- [ ] If the form has a file-upload field, attach
      `DOC-pr/starvla_discretertc_integration_summary.md` (rename to
      drop the `DOC-pr/` prefix).
