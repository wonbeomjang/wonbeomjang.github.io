"""공용 벤치마크/검증 유틸 — FA1·FA2 모듈에서 재사용한다."""

from __future__ import annotations

import os
import time
from typing import Callable

import torch


def get_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("CUDA 사용 불가")
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cc = torch.cuda.get_device_capability(idx)
    total_mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f"GPU: {name} (cc={cc[0]}.{cc[1]}, {total_mem:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton

        print(f"Triton: {triton.__version__}")
    except ImportError:
        pass


def compare_results(
    out: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    label: str = "",
) -> bool:
    diff = (out - ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status} {label:<28} max={max_diff:.4e}  mean={mean_diff:.4e}")
    return ok


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 10,
    repeat: int = 11,
    drop_first: int = 1,
    **kwargs,
) -> float:
    """fn(*args, **kwargs) 평균 실행 시간을 ms 단위로 반환.

    `repeat` 회를 측정한 뒤 앞쪽 `drop_first` 회는 버린다. autotune 캐시가
    재컴파일되거나 cuBLAS 컨텍스트 초기화로 첫 회가 튀는 케이스를 방어한다.
    """
    assert repeat > drop_first, "repeat must be greater than drop_first"
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    kept = times[drop_first:]
    return sum(kept) / len(kept)


def print_benchmark_table(sizes, triton_ms, torch_ms, size_label: str = "Size") -> None:
    header = f"| {size_label:>10} | {'Triton (ms)':>12} | {'PyTorch (ms)':>13} | {'Speedup':>8} |"
    sep = "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 15 + "|" + "-" * 10 + "|"
    print(header)
    print(sep)
    for n, t, p in zip(sizes, triton_ms, torch_ms):
        speed = p / t if t > 0 else float("inf")
        print(f"| {n:>10} | {t:>12.4f} | {p:>13.4f} | {speed:>7.2f}x |")


def plot_benchmark(sizes, triton_ms, torch_ms, title: str, save_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib 없음 — '{save_path}' 저장 생략")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(sizes)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], triton_ms, width, label="Triton", color="#e74c3c")
    ax.bar([i + width / 2 for i in x], torch_ms, width, label="PyTorch", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Size")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  그래프 저장: {save_path}")
