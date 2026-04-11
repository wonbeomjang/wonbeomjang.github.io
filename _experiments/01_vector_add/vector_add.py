"""
01. Vector Addition — Triton 커널 기초
가장 간단한 Triton 커널로 GPU 프로그래밍의 기본을 학습합니다.
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
def vector_add_kernel(
    x_ptr,          # 입력 벡터 x의 포인터
    y_ptr,          # 입력 벡터 y의 포인터
    output_ptr,     # 출력 벡터의 포인터
    n_elements,     # 벡터의 총 원소 수
    BLOCK_SIZE: tl.constexpr,  # 각 프로그램이 처리할 원소 수 (컴파일 타임 상수)
):
    # 현재 프로그램의 ID (0부터 시작)
    pid = tl.program_id(axis=0)

    # 이 프로그램이 처리할 원소들의 인덱스 계산
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 경계 마스크: 벡터 범위를 벗어나는 인덱스는 처리하지 않음
    mask = offsets < n_elements

    # Global Memory에서 데이터 로드
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 덧셈 연산
    output = x + y

    # 결과를 Global Memory에 저장
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================
# 래퍼 함수 (PyTorch 텐서를 받아 Triton 커널 호출)
# ============================================================

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "입력 텐서가 GPU에 있어야 합니다"
    assert x.shape == y.shape, "입력 텐서의 shape이 같아야 합니다"

    output = torch.empty_like(x)
    n_elements = output.numel()

    # 그리드 크기: 필요한 프로그램 인스턴스 수
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 커널 실행
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


# ============================================================
# 메인: 정확도 검증 + 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("01. Vector Addition — Triton 커널 기초")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)
    size = 100_003  # 일부러 BLOCK_SIZE의 배수가 아닌 값 사용
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)

    triton_out = vector_add(x, y)
    torch_out = x + y
    compare_results(triton_out, torch_out, atol=1e-6)

    # --- 벤치마크 ---
    print("\n--- 벤치마크 ---")
    sizes = [2**i for i in range(12, 26)]
    triton_ms_list = []
    torch_ms_list = []

    for size in sizes:
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        y = torch.randn(size, device="cuda", dtype=torch.float32)

        t_ms = benchmark_fn(vector_add, x, y)
        p_ms = benchmark_fn(torch.add, x, y)

        triton_ms_list.append(t_ms)
        torch_ms_list.append(p_ms)

    size_labels = [f"2^{i}" for i in range(12, 26)]
    print_benchmark_table(size_labels, triton_ms_list, torch_ms_list, size_label="Size")

    plot_benchmark(
        size_labels,
        triton_ms_list,
        torch_ms_list,
        title="Vector Add: Triton vs PyTorch",
        save_path="01_vector_add/benchmark.png",
    )


if __name__ == "__main__":
    main()
