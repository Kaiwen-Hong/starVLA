"""
Dummy job to occupy ~75% GPU memory on 4 H100s and keep them busy.
Usage: python gpu_occupy.py
Press Ctrl+C to stop.
"""

import torch
import torch.multiprocessing as mp


def occupy_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_mem, _ = torch.cuda.mem_get_info(device)

    # Target: ensure at least 75% of total GPU memory is occupied.
    # Account for memory already used by other processes.
    already_used = total_mem - free_mem
    target_bytes = max(int(total_mem * 0.75) - already_used, 0)

    if target_bytes == 0:
        print(f"GPU {gpu_id}: already at >=75% usage ({already_used / 1e9:.1f} GB used)")
        while True:
            pass

    # Each float32 element = 4 bytes. Allocate 3 square matrices (a, b, c)
    # so that matmul output goes into pre-allocated c (no extra alloc).
    # Use 90% of target to leave headroom for CUDA context.
    elems_per_matrix = int(target_bytes * 0.9) // (3 * 4)
    dim = int(elems_per_matrix ** 0.5)

    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    c = torch.empty(dim, dim, device=device, dtype=torch.float32)

    alloc = torch.cuda.memory_allocated(device)
    total_used = already_used + alloc
    print(f"GPU {gpu_id}: this process {alloc / 1e9:.1f} GB + others {already_used / 1e9:.1f} GB "
          f"= {total_used / 1e9:.1f} GB / {total_mem / 1e9:.1f} GB "
          f"({total_used / total_mem * 100:.0f}%)")

    # Continuously do matmul into pre-allocated output to keep utilization high
    while True:
        torch.mm(a, b, out=c)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    num_gpus = min(4, torch.cuda.device_count())
    print(f"Launching on {num_gpus} GPUs")

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=occupy_gpu, args=(i,))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in processes:
            p.terminate()
