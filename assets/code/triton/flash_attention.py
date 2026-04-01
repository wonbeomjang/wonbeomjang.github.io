"""
05. Flash Attention — 종합 프로젝트
지금까지 배운 모든 기법을 종합하여 Flash Attention을 구현합니다.
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
# PyTorch 참조 구현 (Standard Attention)
# ============================================================

def pytorch_attention(q, k, v, causal=False):
    """나이브 PyTorch attention (비교용)"""
    scale = q.shape[-1] ** -0.5
    s = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o


# ============================================================
# Triton Flash Attention 커널
# ============================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    # strides for Q (batch, head, seq, dim)
    stride_qb, stride_qh, stride_qm, stride_qk,
    # strides for K
    stride_kb, stride_kh, stride_kn, stride_kk,
    # strides for V
    stride_vb, stride_vh, stride_vn, stride_vk,
    # strides for O
    stride_ob, stride_oh, stride_om, stride_ok,
    # 크기
    seq_len,
    head_dim,
    scale,
    # causal masking
    IS_CAUSAL: tl.constexpr,
    # 블록 크기
    BLOCK_M: tl.constexpr,    # Q 블록 크기
    BLOCK_N: tl.constexpr,    # K/V 블록 크기
    BLOCK_D: tl.constexpr,    # head_dim (2의 거듭제곱으로 패딩)
):
    # 프로그램 ID: 어떤 Q 블록을 처리할지
    pid_m = tl.program_id(0)    # Q 블록 인덱스
    pid_bh = tl.program_id(1)   # batch * head 인덱스

    # batch, head 인덱스 분리
    # (간단히 하기 위해 num_heads를 사용하지 않고 직접 stride로 처리)
    off_b = pid_bh // tl.load(k_ptr + 0) if False else 0  # placeholder

    # Q, K, V, O의 base 포인터 (현재 batch*head에 해당)
    q_base = q_ptr + pid_bh * stride_qh
    k_base = k_ptr + pid_bh * stride_kh
    v_base = v_ptr + pid_bh * stride_vh
    o_base = o_ptr + pid_bh * stride_oh

    # Q 블록의 행 오프셋
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # head_dim 오프셋
    offs_d = tl.arange(0, BLOCK_D)

    # Q 블록 로드 (BLOCK_M × BLOCK_D)
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=q_mask,
        other=0.0,
    )

    # Online Softmax 변수 초기화
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)           # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # output accumulator

    # Causal masking: Q 블록 위치에 따라 K/V 순회 범위 결정
    if IS_CAUSAL:
        k_range = pid_m * BLOCK_M + BLOCK_M
        k_range = min(k_range, seq_len)
    else:
        k_range = seq_len

    # --- 내부 루프: K/V 블록 순회 ---
    for start_n in range(0, k_range, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K 블록 로드 (BLOCK_N × BLOCK_D)
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=k_mask,
            other=0.0,
        )

        # S = Q @ K^T * scale (BLOCK_M × BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * scale

        # Causal mask 적용
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # 범위 밖 마스크
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        # --- Online Softmax 업데이트 ---

        # 현재 블록의 행별 최대값
        m_ij = tl.max(s, axis=1)

        # 새로운 전체 최대값
        m_new = tl.maximum(m_i, m_ij)

        # 보정 계수
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        # sum 업데이트
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Output 보정 및 누적
        acc = acc * alpha[:, None]

        # V 블록 로드 (BLOCK_N × BLOCK_D)
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=v_mask,
            other=0.0,
        )

        # P @ V 누적
        acc += tl.dot(p.to(v.dtype), v)

        # max 업데이트
        m_i = m_new

    # 최종 정규화: O = acc / l
    acc = acc / l_i[:, None]

    # 결과 저장
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(tl.float16),
        mask=o_mask,
    )


# ============================================================
# 래퍼 함수
# ============================================================

def flash_attention(q, k, v, causal=False):
    """
    Flash Attention 래퍼
    입력: q, k, v — (batch*num_heads, seq_len, head_dim) 또는 (num_heads, seq_len, head_dim)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16

    # 3D 입력 처리: (batch_heads, seq_len, head_dim)
    if q.ndim == 3:
        bh, seq_len, head_dim = q.shape
    elif q.ndim == 4:
        b, h, seq_len, head_dim = q.shape
        bh = b * h
        q = q.reshape(bh, seq_len, head_dim)
        k = k.reshape(bh, seq_len, head_dim)
        v = v.reshape(bh, seq_len, head_dim)
    else:
        raise ValueError("q는 3D 또는 4D 텐서여야 합니다")

    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (triton.cdiv(seq_len, BLOCK_M), bh)

    flash_attention_kernel[grid](
        q, k, v, o,
        # Q strides (batch_heads 차원을 head stride로 사용)
        q.stride(0), q.stride(0), q.stride(1), q.stride(2),
        # K strides
        k.stride(0), k.stride(0), k.stride(1), k.stride(2),
        # V strides
        v.stride(0), v.stride(0), v.stride(1), v.stride(2),
        # O strides
        o.stride(0), o.stride(0), o.stride(1), o.stride(2),
        seq_len,
        head_dim,
        head_dim ** -0.5,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return o


# ============================================================
# 메인: 정확도 검증 + 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("05. Flash Attention — 종합 프로젝트")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)

    num_heads = 8
    head_dim = 64

    for seq_len in [128, 256, 512, 1024]:
        q = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        # Non-causal
        triton_out = flash_attention(q, k, v, causal=False)
        torch_out = pytorch_attention(q, k, v, causal=False)
        compare_results(triton_out, torch_out, atol=1e-1, rtol=1e-2, label=f"seq={seq_len}")

        # Causal
        triton_out_c = flash_attention(q, k, v, causal=True)
        torch_out_c = pytorch_attention(q, k, v, causal=True)
        compare_results(triton_out_c, torch_out_c, atol=1e-1, rtol=1e-2, label=f"seq={seq_len} causal")

    # --- 벤치마크 ---
    print("\n--- 벤치마크 ---")
    num_heads = 16
    head_dim = 64
    seq_lengths = [256, 512, 1024, 2048, 4096]
    triton_ms_list = []
    torch_ms_list = []

    for seq_len in seq_lengths:
        q = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        t_ms = benchmark_fn(flash_attention, q, k, v, causal=False)
        p_ms = benchmark_fn(pytorch_attention, q, k, v, causal=False)

        triton_ms_list.append(t_ms)
        torch_ms_list.append(p_ms)

    print_benchmark_table(seq_lengths, triton_ms_list, torch_ms_list, size_label="Seq Length")

    # --- 메모리 사용량 비교 ---
    print("--- 메모리 사용량 비교 ---")
    for seq_len in [1024, 2048, 4096]:
        q = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        # Standard attention 메모리
        torch.cuda.reset_peak_memory_stats()
        _ = pytorch_attention(q, k, v)
        std_mem = torch.cuda.max_memory_allocated() / 1024**2

        # Flash attention 메모리
        torch.cuda.reset_peak_memory_stats()
        _ = flash_attention(q, k, v)
        flash_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  seq_len={seq_len}: Standard={std_mem:.1f}MB, Flash={flash_mem:.1f}MB, "
              f"절약={std_mem - flash_mem:.1f}MB")

    print()

    plot_benchmark(
        seq_lengths,
        triton_ms_list,
        torch_ms_list,
        title="Flash Attention vs Standard Attention",
        save_path="05_flash_attention/benchmark.png",
    )


if __name__ == "__main__":
    main()
