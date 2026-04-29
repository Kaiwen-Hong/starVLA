# StarVLA Cooperation / PR Integration Summary

## Purpose

This document summarizes the intended cooperation / PR request for integrating modules from the DiscreteRTC codebase into the StarVLA codebase. It is written as a technical handoff for implementation planning, so that a coding agent can inspect the StarVLA repository and produce a more concrete implementation plan with exact file paths, class names, APIs, and PR diffs.

The high-level goal is to contribute an asynchronous execution extension to StarVLA, centered on:

1. A **Discrete Diffusion Action Head**.
2. A **DiscreteRTC asynchronous inference runtime** for discrete diffusion policies.
3. Optionally, a **ContinuousRTC / flow-matching RTC implementation** for existing flow-matching policies.

The main cooperation request should be framed as a modular integration proposal, not as a request to merge an entire research codebase wholesale.

---

## Background: What DiscreteRTC Contributes

The paper/codebase behind this request is **DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors**.

The core observation is:

> Real-time chunking (RTC) for robot action chunks is fundamentally an inpainting problem. Discrete diffusion policies are naturally trained to perform inpainting through masked-token unmasking, so they are structurally well-suited for asynchronous action execution.

In action-chunking VLA policies, the policy predicts a future action chunk:

```text
A_t = (a_t, a_{t+1}, ..., a_{t+H-1})
```

For real-world robot deployment, model inference latency is often longer than the low-level control period. Therefore, the robot cannot wait for a full new action chunk before moving. It must **act while the next action chunk is being generated**.

This creates the asynchronous execution problem:

- The robot is executing actions from the previous chunk.
- A new chunk is being inferred in the background.
- When the new chunk is ready, part of its early actions have already become "committed" because the robot has moved during inference.
- The new chunk must be consistent with these committed actions.

RTC solves this by treating chunk transition as **inpainting**:

- Freeze / copy the already committed prefix.
- Generate the remaining suffix conditioned on that prefix.

For flow-matching policies, RTC requires additional inference-time correction such as ΠGDM-style guidance. This introduces:

- Inpainting patterns not seen during pretraining.
- Additional fine-tuning requirements.
- Heuristic guidance schedules.
- Extra inference-time cost.

Discrete diffusion policies avoid these issues because masked-token inpainting is their native operation.

---

## Main Cooperation Request

The main request is to discuss whether the StarVLA maintainers would be open to accepting a contribution that adds:

1. **A discrete diffusion action head**.
2. **A native DiscreteRTC asynchronous inference module for that head**.
3. Optionally, **a ContinuousRTC implementation for existing flow-matching policies**.

Recommended phrasing:

```text
We would like to contribute an asynchronous execution extension to StarVLA, centered around two modules: a discrete diffusion action head and its native DiscreteRTC inference runtime. In addition, we are open to contributing a flow-matching RTC implementation so that StarVLA can support a unified real-time chunking interface across both discrete-diffusion and continuous flow-matching policies.
```

The proposal should emphasize that the contribution is modular, minimally invasive, and aligned with StarVLA's "lego-like" design philosophy.

---

## Why This Fits StarVLA

StarVLA is designed as a modular VLA development codebase. Based on the docs, the relevant properties are:

- It supports multiple action output strategies.
- It uses framework-level model abstractions.
- A framework exposes key APIs such as `forward()` and `predict_action()`.
- Dataloaders return raw, model-agnostic dictionaries such as images, language, state, and action.
- Training and evaluation are configured through YAML / OmegaConf-style configs.
- Evaluation/deployment uses a client-server policy interface, where the policy server calls `Framework.predict_action()`.

This architecture is well-suited for DiscreteRTC because:

- The discrete diffusion action head can be implemented as a new action head or framework.
- Action tokenization and masking can remain inside the action head / framework.
- RTC async inference can be implemented as an inference-time wrapper around `predict_action()` or as a runtime module used by `predict_action()`.
- Dataloaders do not need to be changed significantly.
- Existing training/evaluation pipelines can be reused with new config entries.

---

## Proposed PR Scope

The recommended PR scope should be split into core and optional contributions.

### Core Contribution 1: Discrete Diffusion Action Head

Add a new discrete diffusion action head compatible with the StarVLA framework interface.

Expected functionality:

- Quantize continuous actions into discrete tokens.
- Support a special `[MASK]` token.
- Embed action tokens.
- Predict logits over action bins for each action token.
- Support iterative MaskGIT-style unmasking.
- Support cosine mask schedule / decode schedule.
- Support CE loss on masked positions.
- Optionally support auxiliary L1 reconstruction loss.
- Support configurable number of bins, action dimension, chunk length, and number of unmasking steps.
- Support deterministic or stochastic decoding through decode temperature / choice temperature.
- Integrate with Qwen-VL / DiT-style VLA backbone if available.

Possible config fields:

```yaml
framework:
  name: QwenDiscreteRTC

  action_model:
    type: discrete_diffusion_dit
    action_dim: 10
    chunk_length: 16
    num_bins: 256
    mask_token_id: 256
    embedding_dim: 2048
    unmask_steps: 8
    mask_schedule: cosine
    decode_schedule: cosine
    decode_temperature: 0.0
    choice_temperature: 0.1
    loss:
      ce_weight: 1.0
      l1_weight: 0.1
```

Potential implementation locations to verify in the StarVLA repo:

```text
starVLA/model/framework/
starVLA/model/action_head/
starVLA/model/action_model/
starVLA/config/
starVLA/train/
```

Exact paths should be verified by inspecting the current StarVLA repository.

---

### Core Contribution 2: DiscreteRTC Async Inference Runtime

Add asynchronous RTC inference logic for discrete diffusion policies.

Expected functionality:

- Maintain previous generated action chunk.
- Track inference delay `d`.
- Track execution horizon `s`.
- Track action chunk length `H`.
- Copy the valid tail of the previous chunk into the prefix of the next chunk.
- Convert the copied prefix into discrete action tokens.
- Construct a partially masked action-token sequence:
  - committed prefix = copied/unmasked tokens
  - remaining suffix = `[MASK]`
- Run native discrete diffusion unmasking to inpaint the remaining tokens.
- Optionally support early stopping once the next `s` executable actions are fully unmasked.
- Decode the resulting tokens back into continuous actions.
- Return the next executable actions to the controller.
- Preserve partially unmasked future tokens if the implementation supports natural guidance.

Conceptual pseudocode:

```python
class DiscreteRTCExecutor:
    def __init__(self, policy, chunk_length, action_dim, delay, exec_horizon):
        self.policy = policy
        self.H = chunk_length
        self.action_dim = action_dim
        self.d = delay
        self.s = exec_horizon
        self.prev_chunk = None
        self.prev_tokens = None

    def predict_action_async(self, obs):
        if self.prev_tokens is None:
            # first call: generate from all masks
            token_input = full_mask_tokens(self.H, self.action_dim)
        else:
            # copy committed prefix from previous chunk
            prefix_tokens = get_committed_prefix(self.prev_tokens, self.d)
            token_input = full_mask_tokens(self.H, self.action_dim)
            token_input[:self.d] = prefix_tokens

        # native discrete diffusion inpainting
        new_tokens = self.policy.unmask(
            obs=obs,
            token_input=token_input,
            stop_when_prefix_ready=self.d + self.s,
        )

        new_chunk = self.policy.decode_tokens(new_tokens)

        self.prev_tokens = new_tokens
        self.prev_chunk = new_chunk

        return new_chunk[self.d:self.d + self.s]
```

This pseudocode is conceptual only. Exact indexing must match the StarVLA control loop and action buffer semantics.

Important RTC variables:

```text
H = action chunk length
d = inference delay measured in control steps
s = execution horizon, i.e., how many newly generated actions are executed before starting/using the next inference
```

In the paper's real-world setting:

```text
H = 16
d = 4
s = 4
control frequency = 20 Hz
servo frequency = 100 Hz using 5x interpolation
discrete unmasking steps = 8
action dimension = 10
action representation = [Δx, Δy, Δz, rot6d(6), gripper]
quantization = 256 bins per action dimension
```

---

### Optional Contribution 3: Flow-Matching RTC / ContinuousRTC

Optionally add RTC support for existing flow-matching policies.

Expected functionality:

- Use the same high-level RTC interface as DiscreteRTC.
- Maintain action chunk buffer and committed prefix.
- For flow-matching action heads, inpaint the remaining suffix conditioned on the prefix.
- Implement or expose ΠGDM-style guidance or another prefix-conditioned inpainting mechanism.
- Support soft mask / guidance weights.
- Provide a fair baseline against DiscreteRTC.
- Allow StarVLA users to compare:
  - synchronous execution
  - naive async execution
  - ContinuousRTC
  - DiscreteRTC

Why this optional module is useful:

- It gives StarVLA a unified async inference stack across continuous and discrete action heads.
- It helps users evaluate whether discrete diffusion provides practical gains over flow-matching RTC.
- It makes the PR more generally useful to existing StarVLA users who already use flow-matching policies.

However, the cooperation request should make clear that this is optional. The main request should remain focused on the discrete diffusion action head and DiscreteRTC runtime.

---

## Recommended Architecture

A minimally invasive design would separate the model/action-head logic from the runtime/executor logic.

### Layer 1: Action Head

Responsible for:

- Training loss.
- Action tokenization / detokenization.
- Masked token modeling.
- Iterative unmasking.
- Action decoding.

Possible interface:

```python
class DiscreteDiffusionActionHead(nn.Module):
    def forward(self, features, actions=None, state=None, **kwargs):
        # training path
        # quantize actions
        # randomly mask tokens
        # predict logits
        # compute CE + optional L1 loss
        pass

    @torch.no_grad()
    def generate(self, features, token_input=None, num_steps=None, **kwargs):
        # inference path
        # if token_input is None, start from all masks
        # otherwise continue from partially unmasked tokens
        # return action tokens
        pass

    def encode_actions(self, actions):
        # continuous actions -> discrete tokens
        pass

    def decode_tokens(self, tokens):
        # discrete tokens -> continuous actions
        pass
```

### Layer 2: Framework

Responsible for StarVLA integration.

Possible interface:

```python
class QwenDiscreteRTCFramework(nn.Module):
    def forward(self, batch):
        # extract images, language, state, actions
        # run VLM backbone
        # run discrete diffusion action head
        # return loss dict
        pass

    @torch.no_grad()
    def predict_action(self, obs, **kwargs):
        # run VLM backbone
        # call action_head.generate()
        # decode tokens to actions
        pass
```

### Layer 3: RTC Runtime / Executor

Responsible for asynchronous execution logic.

Possible interface:

```python
class RTCExecutor:
    def reset(self):
        pass

    def step(self, obs):
        # return executable action(s)
        pass

    def update_previous_chunk(self, chunk):
        pass
```

Discrete and continuous RTC can share an abstract base interface:

```python
class BaseRTCBackend:
    def prepare_condition(self, previous_chunk, delay, exec_horizon):
        raise NotImplementedError

    def inpaint(self, obs, condition):
        raise NotImplementedError

class DiscreteRTCBackend(BaseRTCBackend):
    def prepare_condition(...):
        # produce partially masked token sequence
        pass

    def inpaint(...):
        # call discrete diffusion unmasking
        pass

class ContinuousRTCBackend(BaseRTCBackend):
    def prepare_condition(...):
        # produce continuous prefix and soft mask
        pass

    def inpaint(...):
        # call flow-matching inpainting / ΠGDM
        pass
```

---

## Suggested Config Design

A unified config might look like this:

```yaml
framework:
  name: qwen_discrete_rtc

  backbone:
    name: qwen2_5_vl_3b_instruct

  action_head:
    name: discrete_diffusion
    action_dim: 10
    chunk_length: 16
    num_bins: 256
    mask_token_id: 256
    embedding_dim: 2048
    unmask_steps: 8
    mask_schedule: cosine
    decode_schedule: cosine
    decode_temperature: 0.0
    choice_temperature: 0.1
    loss:
      ce_weight: 1.0
      l1_weight: 0.1

  rtc:
    enabled: true
    backend: discrete
    inference_delay: 4
    execution_horizon: 4
    early_stop: true
    preserve_partial_tokens: true
```

For flow-matching RTC:

```yaml
framework:
  rtc:
    enabled: true
    backend: continuous
    inference_delay: 4
    execution_horizon: 4
    guidance: pigdm
    beta: 5.0
    soft_mask: exponential
```

Exact names should be adapted to StarVLA's existing config style.

---

## Training Behavior

For the discrete diffusion action head, training should follow masked-token prediction.

Expected training process:

1. Read continuous action chunk from batch.
2. Normalize actions using StarVLA's existing action normalization if applicable.
3. Quantize action values into discrete bins.
4. Sample a random masking ratio according to a cosine mask schedule.
5. Replace selected action tokens with `[MASK]`.
6. Run the VLA backbone and action head.
7. Predict token logits for masked positions.
8. Compute cross-entropy loss on masked tokens.
9. Optionally decode predicted tokens and compute auxiliary L1 reconstruction loss against continuous actions.
10. Return StarVLA-compatible loss dict.

Possible loss format:

```python
loss_dict = {
    "loss": total_loss,
    "loss_ce": ce_loss,
    "loss_l1": l1_loss,
}
```

---

## Inference Behavior

### Standard Discrete Diffusion Inference

Without RTC:

1. Construct all-mask action-token sequence.
2. Run iterative unmasking for `K` steps.
3. Decode tokens into continuous action chunk.
4. Return the full action chunk or the first action(s).

### DiscreteRTC Inference

With RTC:

1. Use the previous chunk or previous token sequence.
2. Determine the committed prefix based on inference delay `d`.
3. Construct a partially masked sequence:
   - prefix copied from previous chunk/tokens
   - suffix masked
4. Run iterative unmasking.
5. Stop early once actions up to `d + s` are unmasked if early stopping is enabled.
6. Decode tokens into continuous action chunk.
7. Return actions `[d : d + s]` or otherwise match StarVLA's action queue convention.
8. Store current chunk/tokens for the next inference cycle.

---

## Expected Benefits for StarVLA

The cooperation form should emphasize the following benefits:

### 1. New Action Generation Paradigm

StarVLA would gain a discrete diffusion action head, complementing existing autoregressive-token, MLP-regression, and flow-matching action heads.

### 2. Native Async Execution

StarVLA would gain an RTC-style asynchronous inference pathway, allowing policies to act while computing the next chunk.

### 3. Better Dynamic Manipulation Support

The modules are especially relevant for:

- moving-object pick
- moving-platform place
- dynamic tracking
- real-time manipulation
- tasks where inference latency exceeds control period

### 4. Minimal Disruption

The proposed implementation can be built around StarVLA's existing framework/action-head abstraction and `predict_action()` API.

### 5. Fair Baselines

If ContinuousRTC is included, users can compare synchronous execution, naive async, flow-matching RTC, and DiscreteRTC in one codebase.

---

## Experimental Evidence to Mention

From the DiscreteRTC paper:

### Simulated Benchmark

- Benchmark: Kinetix dynamic control tasks.
- DiscreteRTC consistently outperforms:
  - ContinuousRTC
  - Bidirectional decoding
  - naive async execution
- Improvements hold across different inference delays.

### Real-World Robot Setup

- Robot: UR5e arm.
- Gripper: Robotiq.
- Camera: wrist-mounted RGB camera.
- Backbone: Qwen2.5-VL-3B-Instruct.
- Action head: layerwise cross-attention DiT.
- Deployment GPU: single RTX 4090.
- Control frequency: 20 Hz.
- Servo frequency: 100 Hz with 5x interpolation.
- Tasks:
  - Dynamic Pick
  - Dynamic Place

### Real-World Results

| Method | Dynamic Place | Dynamic Pick | Inference Time |
|---|---:|---:|---:|
| Continuous Sync | 0% | 0% | 151 ms |
| Discrete Sync | 0% | 0% | 303 ms |
| Continuous RTC | 90% | 45% | 256 ms |
| DiscreteRTC | 100% | 95% | 206 ms |

Key interpretation:

- Synchronous baselines fail on dynamic tasks.
- ContinuousRTC improves over sync but adds inference overhead.
- DiscreteRTC is both more successful and faster than ContinuousRTC in the tested real-world setting.

---

## Recommended Cooperation Form Positioning

The form should not sound like:

```text
Please merge our entire codebase into StarVLA.
```

It should sound like:

```text
We would like to contribute modular StarVLA-compatible components that add discrete diffusion action decoding and RTC-style asynchronous inference. These components can be reviewed independently, integrated with existing StarVLA framework abstractions, and accompanied by configs, documentation, and real-world demo evidence.
```

Recommended paragraph:

```text
Our main request is to discuss whether the StarVLA team would be open to accepting a contribution that adds: (1) a discrete diffusion action head, (2) a native DiscreteRTC asynchronous inference module for that head, and optionally (3) a ContinuousRTC implementation for existing flow-matching policies. We believe these modules would extend StarVLA's action-generation and deployment capabilities while preserving its modular design philosophy.
```

---

## Possible PR Breakdown

If the StarVLA team prefers smaller PRs, propose the following sequence:

### PR 1: Discrete Diffusion Action Head

Includes:

- Action quantization utilities.
- Mask token embedding.
- Discrete diffusion / MaskGIT-style action head.
- Training loss.
- Basic inference from all-mask sequence.
- Config examples.

### PR 2: DiscreteRTC Inference Runtime

Includes:

- Async action buffer.
- Previous chunk cache.
- Committed-prefix copy.
- Partially masked token construction.
- Native inpainting via discrete diffusion unmasking.
- Early stopping support.
- Deployment/eval config example.

### PR 3: ContinuousRTC Baseline

Includes:

- Flow-matching RTC backend.
- Prefix-conditioned inpainting.
- ΠGDM-style guidance if appropriate.
- Shared RTC backend interface.
- Comparison scripts.

### PR 4: Documentation and Demo

Includes:

- Docs page or README section.
- Example training config.
- Example evaluation/deployment config.
- Dynamic manipulation demo notes.
- Videos/GIFs if accepted by maintainers.

---

## Implementation Questions for Claude Code / Repo Inspection

When inspecting the StarVLA repo, answer these questions:

1. Where are action heads currently implemented?
2. Is there already an abstract action head interface?
3. Where are framework classes registered?
4. How does `framework.name` map to Python class construction?
5. What is the exact signature of `forward()`?
6. What is the exact signature of `predict_action()`?
7. Does `predict_action()` return a full chunk or a single action?
8. Where is action normalization handled?
9. Where are action post-processing and denormalization handled?
10. Where is deployment/evaluation action queue logic implemented?
11. Is there an existing asynchronous inference thread or action buffer?
12. Where should RTC state be stored: framework, policy server, client, or separate executor?
13. How are configs organized?
14. How are losses returned and logged?
15. How are model components registered?
16. Is StarVLA PyTorch-only, or does it support other backends?
17. Are there existing DiT action heads that can be reused?
18. Are Qwen-GR00T / Qwen-PI implementations good templates?
19. Is there existing flow-matching inference code suitable for ContinuousRTC?
20. Where should docs and examples be added?

---

## Likely Files to Inspect First

These file paths are guesses based on documentation and should be verified:

```text
starVLA/model/framework/
starVLA/model/framework/qwen_pi.py
starVLA/model/framework/qwen_fast.py
starVLA/model/framework/qwen_oft.py
starVLA/model/framework/qwen_groot.py

starVLA/model/action_head/
starVLA/model/action_model/
starVLA/model/backbone/

starVLA/train/
starVLA/eval/
starVLA/serve/
starVLA/deploy/

configs/
configs/model/
configs/training/
configs/eval/
```

Look especially for:

```text
forward
predict_action
action_model
action_head
flow_matching
DiT
chunk
horizon
normalize
denormalize
policy_server
websocket
action_queue
```

---

## Open Design Choices

The implementation should resolve the following design choices after repo inspection.

### 1. Framework vs Action Head

Should the contribution add:

- a new full framework such as `QwenDiscreteRTC`, or
- a reusable action head class that plugs into existing frameworks?

Preferred if possible:

```text
Add a reusable discrete diffusion action head, then expose it through a framework config.
```

### 2. RTC Location

Should RTC logic live in:

- `predict_action()`,
- a deployment wrapper,
- a policy server,
- a client-side action queue,
- or a separate `RTCExecutor` class?

Preferred if possible:

```text
Implement RTC as a separate inference-time executor/wrapper, so it can support both discrete and continuous policies.
```

### 3. Tokenization Ownership

Should action quantization live in:

- dataloader,
- action head,
- framework,
- preprocessing transform?

Preferred:

```text
Keep tokenization inside the action head/framework so dataloaders remain model-agnostic.
```

### 4. Partial Token Preservation

Should DiscreteRTC preserve partially unmasked future tokens across inference cycles?

Options:

- simple version: preserve only fully decoded previous chunk
- advanced version: preserve partially unmasked tokens as natural guidance

Preferred staged implementation:

```text
Start with simple committed-prefix RTC. Add partial-token natural guidance as an optional advanced mode.
```

### 5. ContinuousRTC Scope

Should ContinuousRTC be included in the same PR?

Preferred:

```text
Make ContinuousRTC optional or a follow-up PR to keep the first integration focused.
```

---

## Recommended Final Cooperation Form Answer

A concise version suitable for a cooperation form:

```text
We would like to discuss a potential contribution to StarVLA that adds RTC-style asynchronous execution support for action-chunking VLA policies. The core contribution consists of two modular components: (1) a discrete diffusion action head for non-autoregressive masked-token action generation, and (2) a native DiscreteRTC inference runtime that reuses the discrete diffusion head's inpainting capability for asynchronous real-time chunk execution. The runtime copies committed actions from the previous chunk as a prefix, masks the remaining action tokens, and generates the continuation through standard unmasking, without requiring flow-style ΠGDM correction or RTC-specific fine-tuning.

Optionally, we can also contribute a ContinuousRTC backend for existing flow-matching policies, so that StarVLA can provide a unified RTC interface across both continuous and discrete action heads. This would allow users to compare synchronous execution, naive async execution, flow-matching RTC, and DiscreteRTC under the same deployment stack.

We believe this contribution aligns with StarVLA's modular design philosophy. The discrete diffusion head can be implemented as a new action head/framework module, while the RTC logic can be implemented as an inference-time wrapper around the existing predict_action interface. The integration should require minimal changes to dataloading and training infrastructure, and we can provide configs, documentation, and real-world dynamic manipulation demos as supporting examples.
```

---

## One-Sentence Summary

```text
The proposed contribution adds a discrete diffusion action head and an RTC-style asynchronous inference runtime to StarVLA, enabling native inpainting-based real-time chunk execution for dynamic robot manipulation, with optional flow-matching RTC support for a unified async execution interface.
```
