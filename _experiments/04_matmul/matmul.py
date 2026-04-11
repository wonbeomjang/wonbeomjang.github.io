"""
04. Matrix Multiplication — 2D 타일링과 Autotune
딥러닝의 핵심 연산인 GEMM을 Triton으로 구현합니다.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import triton
import triton.language as tl
from common.benchmark_utils import (
    get_gpu_info,
    compare_results,
    benchmark_fn,
    print_benchmark_table,
    plot_benchmark,
)


# ============================================================
# Autotune 설정: 다양한 블록 크기 조합을 자동 탐색
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # 행렬 포인터
    a_ptr, b_ptr, c_ptr,
    # 행렬 크기
    M, N, K,
    # stride (행/열 간 메모리 간격)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 블록 크기 (autotune이 결정)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # --- 프로그램 ID에서 출력 타일 좌표 계산 ---
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # L2 캐시 최적화를 위한 Swizzling
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- A, B 블록의 시작 포인터 계산 ---
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # --- K 차원을 따라 블록 행렬 곱 누적 ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # K 차원 경계 마스크
        k_mask = offs_k < K - k * BLOCK_SIZE_K

        # A, B 블록 로드
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # 블록 행렬 곱 (텐서 코어 활용!)
        accumulator += tl.dot(a, b)

        # 다음 K 블록으로 포인터 이동
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # --- 결과 저장 ---
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# 래퍼 함수
# ============================================================

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], f"크기 불일치: {a.shape} x {b.shape}"
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================
# 메인: 정확도 검증 + 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("04. Matrix Multiplication — 2D 타일링과 Autotune")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)

    for size in [512, 1024, 2048]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        triton_out = triton_matmul(a, b)
        torch_out = torch.matmul(a, b)
        compare_results(triton_out, torch_out, atol=1.0, rtol=1e-2, label=f"{size}x{size}")

    # --- 벤치마크 ---
    print("\n--- 벤치마크 ---")
    print("(autotune 중... 첫 실행 시 시간이 걸릴 수 있습니다)\n")

    sizes = [512, 1024, 2048, 4096]
    triton_ms_list = []
    torch_ms_list = []

    for size in sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        t_ms = benchmark_fn(triton_matmul, a, b)
        p_ms = benchmark_fn(torch.matmul, a, b)

        triton_ms_list.append(t_ms)
        torch_ms_list.append(p_ms)

    print_benchmark_table(sizes, triton_ms_list, torch_ms_list, size_label="Matrix Size")

    # cuBLAS 대비 비율 출력
    print("cuBLAS 대비 Triton 성능:")
    for size, t_ms, p_ms in zip(sizes, triton_ms_list, torch_ms_list):
        ratio = p_ms / t_ms * 100 if t_ms > 0 else 0
        print(f"  {size}x{size}: {ratio:.1f}%")
    print()

    plot_benchmark(
        sizes,
        triton_ms_list,
        torch_ms_list,
        title="MatMul: Triton vs PyTorch (cuBLAS)",
        save_path="04_matmul/benchmark.png",
    )


if __name__ == "__main__":
    main()
