"""
02. Fused Softmax — 커널 퓨전과 Reduction
여러 연산을 하나의 커널로 합쳐 메모리 접근을 최소화합니다.
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
# Triton 커널 정의
# ============================================================

@triton.jit
def fused_softmax_kernel(
    input_ptr,      # 입력 행렬 포인터
    output_ptr,     # 출력 행렬 포인터
    input_row_stride,   # 행 간 stride (다음 행까지의 원소 수)
    output_row_stride,
    n_cols,         # 열의 수
    BLOCK_SIZE: tl.constexpr,  # 한 번에 처리할 열의 수
):
    # 각 프로그램이 하나의 행을 처리
    row_idx = tl.program_id(axis=0)

    # 현재 행의 시작 포인터
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 열 오프셋
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 행 데이터 로드 (범위 밖은 -inf로 채움)
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # --- Fused Softmax 계산 (SRAM에서 전부 처리) ---

    # 1단계: 수치 안정성을 위해 최대값 빼기
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # 2단계: exp 계산
    numerator = tl.exp(row)

    # 3단계: 합계 계산
    denominator = tl.sum(numerator, axis=0)

    # 4단계: 정규화
    softmax_output = numerator / denominator

    # 결과 저장
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# ============================================================
# 래퍼 함수
# ============================================================

def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "입력 텐서가 GPU에 있어야 합니다"
    assert x.ndim == 2, "2D 텐서만 지원합니다"

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE는 n_cols 이상의 2의 거듭제곱
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 그리드: 행 수만큼 프로그램 생성
    grid = (n_rows,)

    fused_softmax_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# ============================================================
# 메인: 정확도 검증 + 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("02. Fused Softmax — 커널 퓨전과 Reduction")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)
    x = torch.randn(1024, 2048, device="cuda", dtype=torch.float32)

    triton_out = fused_softmax(x)
    torch_out = torch.softmax(x, dim=-1)
    compare_results(triton_out, torch_out, atol=1e-4)

    # 다양한 크기로 추가 검증
    for n_cols in [127, 256, 1000, 4096]:
        x = torch.randn(64, n_cols, device="cuda", dtype=torch.float32)
        triton_out = fused_softmax(x)
        torch_out = torch.softmax(x, dim=-1)
        compare_results(triton_out, torch_out, atol=1e-4, label=f"cols={n_cols}")

    # --- 나이브 softmax (퓨전 안 된 버전) ---
    def naive_softmax(x):
        """각 단계가 별도 커널로 실행되는 나이브 구현"""
        row_max = x.max(dim=-1, keepdim=True).values   # 커널 1: max
        safe_x = x - row_max                           # 커널 2: 빼기
        numerator = torch.exp(safe_x)                  # 커널 3: exp
        denominator = numerator.sum(dim=-1, keepdim=True)  # 커널 4: sum
        return numerator / denominator                 # 커널 5: 나누기

    # 나이브 정확도 확인
    x = torch.randn(64, 2048, device="cuda", dtype=torch.float32)
    compare_results(naive_softmax(x), torch.softmax(x, dim=-1), atol=1e-4, label="naive")

    # --- 벤치마크: Triton vs 나이브 vs PyTorch ---
    print("\n--- 벤치마크: Triton (fused) vs 나이브 (unfused) vs PyTorch ---")
    n_rows = 4096
    col_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    triton_ms_list = []
    naive_ms_list = []
    torch_ms_list = []

    for n_cols in col_sizes:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

        t_ms = benchmark_fn(fused_softmax, x)
        n_ms = benchmark_fn(naive_softmax, x)
        p_ms = benchmark_fn(torch.softmax, x, dim=-1)

        triton_ms_list.append(t_ms)
        naive_ms_list.append(n_ms)
        torch_ms_list.append(p_ms)

    # 테이블 출력
    from tabulate import tabulate as tabulate_fn
    headers = ["Columns", "Triton (ms)", "Naive (ms)", "PyTorch (ms)", "Triton vs Naive"]
    rows = []
    for i, n_cols in enumerate(col_sizes):
        speedup = naive_ms_list[i] / triton_ms_list[i] if triton_ms_list[i] > 0 else 0
        rows.append([
            n_cols,
            f"{triton_ms_list[i]:.4f}",
            f"{naive_ms_list[i]:.4f}",
            f"{torch_ms_list[i]:.4f}",
            f"{speedup:.2f}x",
        ])
    print()
    print(tabulate_fn(rows, headers=headers, tablefmt="github"))
    print()
    print("→ Triton(fused)과 PyTorch는 둘 다 퓨전되어 있어 비슷합니다.")
    print("→ 나이브(unfused)는 Global Memory를 5번 왕복하므로 느립니다.")
    print("→ 이것이 커널 퓨전의 효과입니다!")

    plot_benchmark(
        col_sizes,
        triton_ms_list,
        naive_ms_list,
        title=f"Fused (Triton) vs Unfused (Naive) Softmax (rows={n_rows})",
        save_path="02_fused_softmax/benchmark.png",
    )


if __name__ == "__main__":
    main()
