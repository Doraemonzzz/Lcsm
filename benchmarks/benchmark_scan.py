import os

import numpy as np
import torch
import torch.nn.functional as F
import triton

from lcsm_pytorch.ops import (
    expand_and_shrink,
    pscan,
    pscan_block,
    pscan_cuda_fn,
    pscan_torch,
)

b, n, k, d = 6, 512, 32, 128
b, n, k, d = 4, 512, 128, 128
b, n, k, d = 32, 512, 128, 128
device = "cuda:0"
speed_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 13)],
        line_arg="provider",
        # line_vals=["origin", "pscan", "pscan_torch", "pscan_block"],
        # line_names=["origin", "pscan", "pscan_torch", "pscan_block"],
        line_vals=["pscan_block", "pscan_cuda"],
        line_names=["pscan_block", "pscan_cuda"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        ylabel="ms",
        plot_name=f"mnet-batch{b}-k{k}-d{d}-{mode}",
        args={
            "b": b,
            "k": k,
            "d": d,
            "dtype": torch.float16,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
]


@triton.testing.perf_report(speed_configs)
def bench_mnet_speed(b, n, k, d, mode, provider, dtype=torch.bfloat16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    i = (
        torch.empty((b, n, d), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    e = (
        torch.empty((b, n, k), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    f = F.sigmoid(
        torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    ).requires_grad_()
    s = (
        torch.empty((b, n, k), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    if provider == "origin":
        fun = expand_and_shrink
    elif provider == "pscan":
        fun = pscan
    elif provider == "pscan_torch":
        fun = pscan_torch
    elif provider == "pscan_block":
        fun = pscan_block
    elif provider == "pscan_cuda":
        fun = pscan_cuda_fn
    fn = lambda: fun(i, e, f, s)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


def get_memory(device):
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used


memory_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 13)],
        line_arg="provider",
        # line_vals=["origin", "pscan", "pscan_torch", "pscan_block"],
        # line_names=["origin", "pscan", "pscan_torch", "pscan_block"],
        line_vals=["pscan_block", "pscan_cuda"],
        line_names=["pscan_block", "pscan_cuda"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        ylabel="ms",
        plot_name=f"mnet-batch{b}-k{k}-d{d}-{mode}",
        args={
            "b": b,
            "k": k,
            "d": d,
            "dtype": torch.float16,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
]


@triton.testing.perf_report(memory_configs)
def bench_mnet_memory(b, n, k, d, mode, provider, dtype=torch.bfloat16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    rep = 5
    i = (
        torch.empty((b, n, d), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    e = (
        torch.empty((b, n, k), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    f = F.sigmoid(
        torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    ).requires_grad_()
    s = (
        torch.empty((b, n, k), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    if provider == "origin":
        fun = expand_and_shrink
    elif provider == "pscan":
        fun = pscan
    elif provider == "pscan_torch":
        fun = pscan_torch
    elif provider == "pscan_block":
        fun = pscan_block
    elif provider == "pscan_cuda":
        fun = pscan_cuda_fn
    fn = lambda: fun(i, e, f, s)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    try:
        torch.cuda.reset_peak_memory_stats(device)
        mb_arr = []
        for _ in range(rep):
            fn()
            mb_arr.append(get_memory(device))
        mb = np.mean(mb_arr)
    except:
        mb = -1

    return mb


save_path = "stat"
os.makedirs(save_path, exist_ok=True)
# only works on post-Ampere GPUs right now
bench_mnet_speed.run(save_path=save_path, print_data=True)
bench_mnet_memory.run(save_path=save_path, print_data=True)
