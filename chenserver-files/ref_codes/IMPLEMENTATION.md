# Training-Time RTC for QwenPI — Implementation Notes

This doc records how the JAX reference (`model.py`, `trainingtimeRTC.py`) was
ported into the QwenPI stack as an **additive** finetune. The goal is to
finetune an existing QwenPI checkpoint so that, at deployment, it can be
served via the already-existing
`predict_action_realtime(mode="simulated_delay")` path with substantially
better consistency to the previous chunk's un-executed prefix.

## 1. Reference behavior (recap)

The training-time RTC loss in `model.py:294-316` differs from vanilla
flow matching in four ways:

1. Sample a **prefix delay** per batch item from an exp-decay
   distribution over `[0, simulated_delay)`. Small delays dominate;
   `delay == 0` means "behave like standard FM training."
2. Build a **per-position time vector** `(B, H)`: prefix positions
   `[0..delay-1]` are set to `time = 1.0` (fully clean), suffix positions
   keep the per-batch base time `t ~ U(0,1)` (in our codebase: Beta).
3. Build `x_t = (1 - time) * noise + time * action`. Prefix slots are
   exactly the clean target; suffix slots are the usual noisy
   interpolation.
4. Mask the velocity MSE so only suffix positions contribute (prefix
   slots carry no gradient because they already equal the target).

At inference, `realtime_action(simulated_delay is not None)` simply
**replaces the prefix slots with the previous chunk and sets their time
to 1.0**, then denoises the suffix — no ΠGDM correction needed. The
training distribution matches this inference distribution by construction.

## 2. Where the changes landed

All edits are confined to one file and one new launcher script:

| File | Change |
|---|---|
| `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | Added `simulated_delay` config + exp-decay buffer in `__init__`; added one-line dispatch at the top of `forward` and a new `_forward_rtc` method. |
| `realworld/0403-dynamic-0403-0-pick_to_moved-rtc-ft.sh` | New launcher mirroring the existing QwenPI block, with `--framework.action_model.simulated_delay 4` and a distinct `run_id`. |

**Untouched**: `Qwen_PI.forward`, `Qwen_PI.predict_action`,
`Qwen_PI.predict_action_realtime`, the DiT block, `AdaLayerNorm`,
`ActionEncoder` parameters, `MLP` state encoder, `future_tokens`,
`position_embedding`, all existing inference paths
(`predict_action`, `predict_action_realtime` with either `pigdm` or
`simulated_delay` mode).

## 3. Additivity guarantees

- **No new `nn.Parameter`**. The exp-decay weight tensor is registered
  as a non-persistent buffer, so checkpoint loading is unaffected.
- **No layer architecture change**. DiT layers, AdaLN, action encoder,
  action decoder, state encoder, future tokens, and position embedding
  are bit-identical.
- **No behavior change when `simulated_delay` is unset**. The dispatch
  in `forward` is a single guarded early-return; the original loop
  follows in the `else` and is byte-identical to the prior
  implementation.
- **Existing YAMLs and launch scripts run unchanged.**
- **Inference companion already exists**. The
  `LayerwiseFlowmatchingActionHead.predict_action_realtime(mode="simulated_delay")`
  path was already in the code (lines 564–632) and is exactly the
  distribution this training targets.

## 4. The new training branch

`LayerwiseFlowmatchingActionHead._forward_rtc` performs the following:

```
B, H, D       = actions.shape
noise         = randn_like(actions)
t_scalar      = sample_time(B)                              # (B,), Beta-derived
velocity      = actions - noise

# Sample per-batch prefix delay from exp-decay over [0, d_max).
weights       = exp(arange(d_max-1, -1, -1)); weights /= sum
delay         = multinomial(weights, B, replacement=True)   # (B,)

prefix_mask   = arange(H)[None,:] < delay[:,None]           # (B, H)

time_per_pos  = where(prefix_mask, 1.0, t_scalar[:,None])   # (B, H)
x_t           = (1 - time)*noise + time*actions             # prefix == clean

# Per-position discretized bucket time for the action encoder,
# inlined because ActionEncoder.forward rejects (B, H) inputs.
t_buckets_pos = (time_per_pos * num_timestep_buckets).long().clamp(0, B-1)
a_emb         = action_encoder.layer1(x_t)
tau_emb       = action_encoder.pos_encoding(t_buckets_pos)
x_enc         = swish(action_encoder.layer2(cat([a_emb, tau_emb], -1)))
action_features = action_encoder.layer3(x_enc)

# Add position embedding, concat (state) + future_tokens + actions
# exactly as in the original path.

# Per-batch temb for AdaLN, matches the inference-time
# predict_action_realtime(mode="simulated_delay") path that uses one
# t_bucket per batch item for AdaLN conditioning.
t_bucket_batch = (t_scalar * num_timestep_buckets).long()
temb           = model.timestep_encoder(t_bucket_batch)

# Layerwise cross-attention with vl_embs_list[layer_idx], unchanged.

pred_actions   = action_decoder(...)[:, -H:]
loss_mask      = (~prefix_mask)[:, :, None]
loss           = sum((pred_actions - velocity)^2 * loss_mask)
                 / (loss_mask.sum() * D + 1e-8)
```

### Why per-position time only in the action encoder

The JAX reference also makes AdaLN scales per-position via per-position
`time_emb`. Our DiT's `AdaLayerNorm` consumes a single `(B, embedding_dim)`
`temb` vector and applies it identically to every token; making it
per-token would require changing the DiT shape contracts and
re-initializing checkpoint-incompatible projection layers. Crucially,
the **existing inference path** (`predict_action_realtime` mode
`"simulated_delay"`, line 612) already uses a per-batch `temb_tensor`,
so this training matches that inference path exactly. Per-position
information still flows in through the action encoder's
`SinusoidalPositionalEncoding`, which is the part of the model that
actually sees individual action tokens.

### Why delay=0 dominates

Exp-decay weights `[exp(d-1), exp(d-2), ..., exp(0)]` after normalization
put the largest mass on `delay == 0`. That draw recovers exactly the
original FM loss, so the finetune preserves vanilla performance while
adding the prefix-conditioned suffix denoising skill.

## 5. Inference

### 5.1 Outer API (recommended)

`Qwen_PI.predict_action_realtime` now accepts the realtime knobs as
first-class arguments and forwards them to the action head. Call it with
the explicit mode that matches the trained behavior:

```python
out = model.predict_action_realtime(
    examples=examples,
    prev_action_chunk_normalized=prev_actions,   # (B, H, action_dim), normalized
    inference_delay=4,
    mode="simulated_delay",   # the path this finetune is trained for
)
normalized_actions = out["normalized_actions"]
```

If you omit `mode` (or pass `mode=None`), it falls back to the head's
`default_realtime_mode` property:

- finetuned with `simulated_delay`>0 → `"simulated_delay"`
- otherwise                          → `"pigdm"`

This way the call site is self-documenting: `mode="simulated_delay"` is
the obvious tag for code review and grep, but a non-finetuned checkpoint
still "just works" without changes.

### 5.2 Direct head call

```python
pred = action_head.predict_action_realtime(
    vl_embs_list,
    state_t,
    prev_action_chunk=prev_chunk_t,
    inference_delay=4,
    mode="simulated_delay",   # explicit; see action_head.default_realtime_mode
)
```

### 5.3 Falling back to ΠGDM

`mode="pigdm"` continues to work against the finetuned weights (it
requires no special training) and remains the recommended fallback for
callers that can pay the ~2x VJP cost per step. Pass
`prefix_attention_schedule`, `suffix_length`, `max_guidance_weight`
through `Qwen_PI.predict_action_realtime` to tune it.

## 6. Config knob

A single new optional field under `framework.action_model`:

```yaml
framework:
  action_model:
    simulated_delay: 4   # int; null / 0 / unset -> original FM training
```

Default `None` ⇒ no behavior change. Recommended finetune value: `4`
(picked to slightly exceed realistic `inference_delay` so the model sees
prefix lengths up to and beyond what it will be asked to inpaint).

## 7. Launching the finetune

`bash realworld/0403-dynamic-0403-0-pick_to_moved-rtc-ft.sh`

This launches the QwenPI training pipeline with `simulated_delay=4`,
resumes from the previous checkpoint (`--trainer.is_resume true`), and
writes to `fastumi_pickandplace_qwenPI_0403_0_pick_to_moved_rtc_ft` so
the original RTC-free checkpoint is preserved for A/B comparison.

## 8. Mapping JAX ref → PyTorch port

| Concept | `model.py` (JAX) | This port (PyTorch) |
|---|---|---|
| Base time | `jax.random.uniform(time_rng, (B,))` | `self.sample_time(B, ...)` (Beta dist, kept) |
| Delay sampling | `jax.random.choice(delay_rng, simulated_delay, (B,), p=w)` with `w = exp(arange(d-1,-1,-1))/Z` | `torch.multinomial(self._simulated_delay_weights, B, replacement=True)` |
| Per-position time | `time = where(mask, 1.0, time[:,None])` → `(B, H)` | Same shape `(B, H)`, then bucketized for the discrete action encoder |
| `x_t` construction | `(1-time)*noise + time*action` with `time` shape `(B, H, 1)` | Same |
| Model call | `self(obs, x_t, time)` — `time` is `(B, H)` and feeds both `posemb_sincos` and AdaLN | Action encoder gets `(B, H)` time; AdaLN/temb gets per-batch `t_scalar` (matches existing inference path) |
| Loss mask | `loss_mask = ~mask[:,:,None]` | Same, `(B, H, 1)`, divisor includes `D` because we compute over the action-dim axis |
| Inference twin | `realtime_action(simulated_delay is not None)` — replace prefix, set time=1, denoise suffix | `predict_action_realtime(mode="simulated_delay")` — already implemented prior to this change |

## 9. Validation checklist

- [ ] With `simulated_delay` unset, training-loss curve matches a known
      QwenPI run within numerical noise.
- [ ] With `simulated_delay=4`, training loss stays comparable in
      magnitude (mask normalization makes per-suffix-position MSE the
      reported quantity).
- [ ] Inference: `predict_action_realtime(mode="simulated_delay",
      inference_delay=k)` for `k ∈ {1,2,3,4}` produces actions whose
      first `k` positions are close to `prev_action_chunk_normalized`.
- [ ] Rollouts under realistic inference latency show fewer
      discontinuities at chunk boundaries than the un-finetuned
      checkpoint served with the same realtime path.

## 10. Out-of-scope / possible follow-ups

- Per-position AdaLN conditioning (closer to the JAX ref). Would require
  reshaping DiT's `temb` to `(B, seq_len, D)` and re-initializing
  `proj_out_1`; not done here to preserve checkpoint compatibility.
- A BID-style multi-sample selection wrapper around `predict_action` for
  rejection-sampling-based prefix consistency (see `bid_action` in
  `model.py:189`). Independent of this finetune.
