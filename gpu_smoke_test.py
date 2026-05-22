"""
GPU smoke test for 8x NVIDIA RTX PRO 6000 Blackwell.

Two modes:
  python gpu_smoke_test.py single   # per-GPU compute correctness + memory
  torchrun --nproc_per_node=8 gpu_smoke_test.py ddp   # NCCL + DDP training
"""
import os
import sys
import time
import torch


def single_gpu_test():
    n = torch.cuda.device_count()
    print(f"\n=== Per-GPU smoke test ({n} GPUs) ===\n")
    matsize = 8192
    failures = []

    for i in range(n):
        torch.cuda.set_device(i)
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"[GPU {i}] {name}  sm_{cap[0]}{cap[1]}  free {free/1e9:.1f}/{total/1e9:.1f} GB")

        try:
            # Disable TF32 for the correctness check so GPU and CPU compute the same thing
            torch.backends.cuda.matmul.allow_tf32 = False
            a = torch.randn(matsize, matsize, device=f"cuda:{i}", dtype=torch.float32)
            b = torch.randn(matsize, matsize, device=f"cuda:{i}", dtype=torch.float32)
            torch.cuda.synchronize(i)
            t0 = time.time()
            for _ in range(5):
                c = a @ b
            torch.cuda.synchronize(i)
            dt = (time.time() - t0) / 5
            tflops = 2 * matsize**3 / dt / 1e12
            # correctness: do a small full matmul on CPU and compare to corresponding block of GPU result
            a_cpu = a[:128, :].cpu()
            b_cpu = b[:, :128].cpu()
            c_ref = a_cpu @ b_cpu  # this is the [:128,:128] block of the full a @ b
            err = (c[:128, :128].cpu() - c_ref).abs().max().item()
            rel_err = err / c_ref.abs().mean().item()
            ok = rel_err < 1e-3
            print(f"         matmul {matsize}^3 fp32: {dt*1000:.1f} ms, {tflops:.1f} TFLOPS, rel-err={rel_err:.2e} {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append((i, f"correctness error {err}"))

            # bf16 test (Blackwell shines here)
            a16 = a.to(torch.bfloat16)
            b16 = b.to(torch.bfloat16)
            torch.cuda.synchronize(i)
            t0 = time.time()
            for _ in range(10):
                c16 = a16 @ b16
            torch.cuda.synchronize(i)
            dt = (time.time() - t0) / 10
            tflops = 2 * matsize**3 / dt / 1e12
            print(f"         matmul {matsize}^3 bf16: {dt*1000:.1f} ms, {tflops:.1f} TFLOPS")

            # memory stress: allocate 80% of free
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(i)
            target_bytes = int(free * 0.8)
            n_elts = target_bytes // 4  # float32
            big = torch.empty(n_elts, device=f"cuda:{i}", dtype=torch.float32)
            big.fill_(1.0)
            s = big.sum().item()
            expected = float(n_elts)
            mem_ok = abs(s - expected) / expected < 1e-3
            print(f"         alloc {target_bytes/1e9:.1f} GB, sum={s:.3e}, expected={expected:.3e} {'OK' if mem_ok else 'FAIL'}")
            if not mem_ok:
                failures.append((i, f"memory check sum mismatch"))
            del a, b, c, a16, b16, c16, big
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"         FAILED: {e}")
            failures.append((i, str(e)))

    print()
    if failures:
        print(f"=== {len(failures)} GPU(s) FAILED ===")
        for i, msg in failures:
            print(f"  GPU {i}: {msg}")
        sys.exit(1)
    print("=== ALL 8 GPUs PASSED single-GPU tests ===")


def ddp_test():
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    if rank == 0:
        print(f"\n=== DDP smoke test: world={world}, NCCL ===\n")

    # Test 1: all_reduce correctness
    t = torch.full((1024, 1024), float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world + 1))
    err = (t - expected).abs().max().item()
    if rank == 0:
        print(f"[all_reduce] sum 1..{world} = {expected}, got {t[0,0].item():.0f}, max-err={err:.2e} {'OK' if err < 1e-3 else 'FAIL'}")

    # Test 2: bandwidth
    sz = 256 * 1024 * 1024  # 256M floats = 1 GB
    buf = torch.randn(sz, device=f"cuda:{local_rank}")
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    dt = (time.time() - t0) / 5
    bytes_xfer = sz * 4 * 2 * (world - 1) / world  # ring all-reduce
    bw = bytes_xfer / dt / 1e9
    if rank == 0:
        print(f"[all_reduce] 1GB tensor: {dt*1000:.1f} ms, ~{bw:.1f} GB/s/GPU (algo bw)")

    # Test 3: tiny DDP training
    torch.manual_seed(42 + rank)
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 10),
    ).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for step in range(20):
        x = torch.randn(64, 1024, device=f"cuda:{local_rank}")
        y = torch.randint(0, 10, (64,), device=f"cuda:{local_rank}")
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # reduce final loss across ranks to confirm sync
    final = torch.tensor([losses[-1]], device=f"cuda:{local_rank}")
    dist.all_reduce(final, op=dist.ReduceOp.SUM)
    final_avg = final.item() / world
    if rank == 0:
        print(f"[DDP train] 20 steps, loss {losses[0]:.3f} -> {losses[-1]:.3f} (avg across ranks: {final_avg:.3f})")
        if losses[-1] < losses[0]:
            print("[DDP train] loss decreased -- OK")
        else:
            print("[DDP train] loss did NOT decrease -- WARN (could be noise)")
        print("\n=== DDP test complete ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        single_gpu_test()
    elif mode == "ddp":
        ddp_test()
    else:
        print(f"Unknown mode: {mode}. Use 'single' or 'ddp'.")
        sys.exit(1)
