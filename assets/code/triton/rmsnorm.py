"""
03. RMSNorm — LLM에서 쓰이는 실전 커널
LLaMA, Mistral 등에서 사용하는 RMSNorm을 Triton으로 구현합니다.
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
# PyTorch 참조 구현 (비교용)
# ============================================================

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


# ============================================================
# Triton 커널 정의
# ============================================================

@triton.jit
def rmsnorm_kernel(
    input_ptr,      # 입력 텐서 포인터
    weight_ptr,     # 가중치(γ) 포인터
    output_ptr,     # 출력 텐서 포인터
    stride,         # 행 간 stride
    n_cols,         # 열의 수 (hidden_size)
    eps,            # epsilon (0으로 나누기 방지)
    BLOCK_SIZE: tl.constexpr,
):
    # 각 프로그램이 하나의 행을 처리
    row_idx = tl.program_id(axis=0)

    # 현재 행의 시작 포인터
    row_start = input_ptr + row_idx * stride
    out_start = output_ptr + row_idx * stride

    # 열 오프셋
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 행 데이터 로드
    row = tl.load(row_start + col_offsets, mask=mask, other=0.0)

    # 가중치(γ) 로드
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)

    # --- RMSNorm 계산 (SRAM에서 전부 처리) ---

    # 1단계: 제곱합
    sq_sum = tl.sum(row * row, axis=0)

    # 2단계: RMS 계산
    mean_sq = sq_sum / n_cols
    rms = tl.sqrt(mean_sq + eps)

    # 3단계: 정규화 + 스케일링
    normed = row / rms
    output = normed * weight

    # 결과 저장
    tl.store(out_start + col_offsets, output, mask=mask)


# ============================================================
# 래퍼 함수
# ============================================================

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda, "입력 텐서가 GPU에 있어야 합니다"
    assert x.shape[-1] == weight.shape[0], "마지막 차원과 weight 크기가 같아야 합니다"

    # 2D로 reshape (배치 처리)
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape
    output = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    rmsnorm_kernel[grid](
        x_2d, weight, output,
        x_2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(orig_shape)


# ============================================================
# 메인: 정확도 검증 + 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("03. RMSNorm — LLM 실전 커널")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)

    for hidden_size in [768, 1024, 2048, 4096]:
        x = torch.randn(32, 128, hidden_size, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

        triton_out = triton_rmsnorm(x, weight)
        torch_out = pytorch_rmsnorm(x, weight)
        compare_results(triton_out, torch_out, atol=1e-4, label=f"hidden={hidden_size}")

    # --- 벤치마크 ---
    print("\n--- 벤치마크 ---")
    batch_seq = 4096  # batch_size * seq_len
    hidden_sizes = [768, 1024, 2048, 4096, 8192]
    triton_ms_list = []
    torch_ms_list = []

    for hidden_size in hidden_sizes:
        x = torch.randn(batch_seq, hidden_size, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

        t_ms = benchmark_fn(triton_rmsnorm, x, weight)
        p_ms = benchmark_fn(pytorch_rmsnorm, x, weight)

        triton_ms_list.append(t_ms)
        torch_ms_list.append(p_ms)

    print_benchmark_table(hidden_sizes, triton_ms_list, torch_ms_list, size_label="Hidden Size")

    plot_benchmark(
        hidden_sizes,
        triton_ms_list,
        torch_ms_list,
        title="RMSNorm: Triton vs PyTorch",
        save_path="03_rmsnorm/benchmark.png",
    )


if __name__ == "__main__":
    main()
