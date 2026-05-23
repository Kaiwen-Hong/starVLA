"""
Single-GPU benchmark for the preference-conditioned VLA Stage A baseline.

Measures: ms/step (data + fw+bw+opt), peak GPU memory, and extrapolates 50k-step
wall clock. Uses SGD (no optimizer state) so the benchmark fits in 80 GB without
Zero2 sharding -- AdamW would need ~87 GB just for fp32 master + (m, v).

Per-step fw+bw cost dominates training wall time; opt.step() is small compared
to model forward. Multi-GPU adds an all-reduce per backward but otherwise
scales linearly, so the single-GPU number is a decent lower bound on multi-GPU
per-step wall time (multi-GPU will be slightly slower due to comm).

Run:
  python -m examples.preference.tests.benchmark
  python -m examples.preference.tests.benchmark --per_device_batch 16 --no_grad_ckpt
  python -m examples.preference.tests.benchmark --num_workers 8
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="Disable gradient checkpointing on Qwen backbone")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=15)
    p.add_argument("--attn", default=None, choices=[None, "sdpa", "flash_attention_2", "eager"],
                   help="Override attn_implementation; default uses YAML setting")
    args = p.parse_args()

    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("examples/preference/train_files/starvla_pref_stage_a_baseline.yaml")
    cfg.output_dir = "/tmp/pref_benchmark_output"
    cfg.datasets.vla_data.per_device_batch_size = args.per_device_batch
    cfg.datasets.vla_data.num_workers = args.num_workers
    if args.attn:
        cfg.framework.qwenvl.attn_implementation = args.attn

    print(f"Config: per_device_batch={args.per_device_batch}, num_workers={args.num_workers}, "
          f"grad_ckpt={'OFF' if args.no_grad_ckpt else 'ON'}, "
          f"attn={cfg.framework.qwenvl.attn_implementation}")

    from starVLA.model.framework.QwenPI import Qwen_PI
    print("Loading Qwen_PI...")
    t0 = time.perf_counter()
    model = Qwen_PI(cfg).to("cuda:0")
    model.train()
    if not args.no_grad_ckpt:
        if hasattr(model.qwen_vl_interface.model, "gradient_checkpointing_enable"):
            model.qwen_vl_interface.model.gradient_checkpointing_enable()
            print("  gradient_checkpointing: ON (on Qwen3-VL backbone)")
    print(f"Model loaded in {time.perf_counter()-t0:.1f} s; "
          f"params total = {sum(p.numel() for p in model.parameters())/1e9:.2f} B; "
          f"trainable = {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f} B")

    from starVLA.dataloader import build_dataloader
    dl = build_dataloader(cfg, dataset_py="pref_hdf5")
    print(f"Dataloader: {len(dl)} batches/epoch at per_device_batch={args.per_device_batch}")
    it = iter(dl)

    # SGD: no optimizer state -> fits without Zero2.
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)

    print(f"\nWarmup ({args.n_warmup} steps)...")
    for i in range(args.n_warmup):
        batch = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out["action_loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    print(f"Timed ({args.n_steps} steps)...")
    data_times = []
    step_times = []
    losses = []
    for i in range(args.n_steps):
        t_d0 = time.perf_counter()
        batch = next(it)
        torch.cuda.synchronize()
        t_d1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out["action_loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        data_times.append((t_d1 - t_d0) * 1000)
        step_times.append((t_s - t_d1) * 1000)
        losses.append(loss.item())

    ms_data = st.median(data_times)
    ms_step = st.median(step_times)
    ms_total = ms_data + ms_step
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    free_gb_after = torch.cuda.mem_get_info(0)[0] / 1e9

    print()
    print(f"Per-step medians (over {args.n_steps} steps):")
    print(f"  data:       {ms_data:7.1f} ms")
    print(f"  fw+bw+opt:  {ms_step:7.1f} ms")
    print(f"  total:      {ms_total:7.1f} ms = {ms_total/1000:.3f} s")
    print(f"Loss (first/last): {losses[0]:.3f} / {losses[-1]:.3f}")
    print(f"Peak GPU mem allocated: {peak_gb:.1f} GB; free after run: {free_gb_after:.1f} GB")
    print()
    eff_batch_8gpu = args.per_device_batch * 8
    wall_8gpu_h = 50000 * ms_total / 1000 / 3600
    print(f"Extrapolation (single-GPU step time as proxy for multi-GPU):")
    print(f"  50k steps × {ms_total/1000:.3f} s/step ≈ {wall_8gpu_h:.1f} h wall (8 GPU eff batch {eff_batch_8gpu})")
    print(f"  (Multi-GPU adds all-reduce overhead; real wall typically 1.1-1.3× this.)")


if __name__ == "__main__":
    main()
