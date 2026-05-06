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
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    # 래퍼에서 (b, h, n, d) → (bh, n, d)로 reshape하므로 stride는 3D 기준
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    stride_lh, stride_lm,                  # LSE: (bh, seq)
    seq_len,
    head_dim,
    scale,
    IS_CAUSAL: tl.constexpr,               # causal masking
    SAVE_LSE: tl.constexpr,                # backward용 LSE 저장 여부
    BLOCK_M: tl.constexpr,                 # Q 블록 크기
    BLOCK_N: tl.constexpr,                 # K/V 블록 크기
    BLOCK_D: tl.constexpr,                 # head_dim (2의 거듭제곱)
):
    pid_m = tl.program_id(0)               # Q 블록 인덱스
    pid_bh = tl.program_id(1)              # batch * head 인덱스

    # 현재 batch*head에 해당하는 base 포인터
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
        k_range = tl.minimum(pid_m * BLOCK_M + BLOCK_M, seq_len)
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

        # 이전 누적값 보정 계수
        alpha = tl.exp(m_i - m_new)

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

    # 결과 저장 (출력 dtype은 입력과 동일 — fp16/bf16 모두 지원)
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(o_ptr.type.element_ty),
        mask=o_mask,
    )

    # backward용 logsumexp 저장 (자연로그)
    # exp 사용했으므로 자연로그 그대로: L = m + log(ℓ)
    if SAVE_LSE:
        lse = m_i + tl.log(l_i)
        tl.store(
            lse_ptr + pid_bh * stride_lh + offs_m * stride_lm,
            lse, mask=offs_m < seq_len,
        )


# ============================================================
# 래퍼 함수
# ============================================================

def flash_attention(q, k, v, causal=False, return_lse=False):
    """
    Flash Attention 래퍼 (forward only).
    입력: q, k, v — (batch*num_heads, seq_len, head_dim) 또는 (num_heads, seq_len, head_dim)
                     또는 (batch, num_heads, seq_len, head_dim)
    return_lse: backward 또는 외부용 logsumexp (자연로그) 반환
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)

    orig_shape = q.shape
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

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    lse = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32) if return_lse else None
    # SAVE_LSE=False여도 jit 호출 호환을 위해 더미 텐서 전달
    lse_buf = lse if lse is not None else torch.empty(1, device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (triton.cdiv(seq_len, BLOCK_M), bh)

    flash_attention_kernel[grid](
        q, k, v, o, lse_buf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse_buf.stride(0) if lse is not None else 0,
        lse_buf.stride(1) if lse is not None else 0,
        seq_len,
        head_dim,
        head_dim ** -0.5,
        IS_CAUSAL=causal,
        SAVE_LSE=return_lse,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    o = o.reshape(orig_shape)
    if return_lse:
        if len(orig_shape) == 4:
            lse = lse.reshape(orig_shape[0], orig_shape[1], orig_shape[2])
        return o, lse
    return o


# ============================================================
# Backward — 두 커널 분리 방식 (논문 §3.1.2 Algorithm 2 기반)
#
# FA1과 FA2의 backward 알고리즘은 동일합니다 (둘 다 같은 chain rule).
# 차이는 forward의 누적 방식뿐. 이 구현은 FA1 스타일에 맞춰
#   - 자연로그 LSE (exp 사용, base-2 트릭 X)
#   - autotune X (학습용 단순 버전)
# 으로 작성. 06_flash_attention_v2의 backward와 비교 학습 가능.
# ============================================================


@triton.jit
def _fa1_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    stride_oh, stride_om, stride_od,
    stride_dh, stride_dm,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """D_i = rowsum(dO_i ∘ O_i)."""
    pid_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)

    o = tl.load(
        o_ptr + off_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask, other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + off_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask, other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(
        delta_ptr + off_bh * stride_dh + offs_m * stride_dm,
        delta, mask=offs_m < seq_len,
    )


@triton.jit
def _fa1_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr,
    dk_ptr, dv_ptr,
    sm_scale,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_doh, stride_dom, stride_dod,
    stride_lh, stride_lm,
    stride_deltah, stride_deltam,
    stride_dkh, stride_dkn, stride_dkd,
    stride_dvh, stride_dvn, stride_dvd,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dK_j, dV_j 계산. 외부 = K/V (j) 고정, 내부 = Q (i) 순회."""
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    kv_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)

    k = tl.load(
        k_ptr + off_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=kv_mask, other=0.0,
    )
    v = tl.load(
        v_ptr + off_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=kv_mask, other=0.0,
    )

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)

        q = tl.load(
            q_ptr + off_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=q_mask, other=0.0,
        )
        do = tl.load(
            do_ptr + off_bh * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
            mask=q_mask, other=0.0,
        )
        l_i = tl.load(
            lse_ptr + off_bh * stride_lh + offs_m * stride_lm,
            mask=offs_m < seq_len, other=0.0,
        )
        d_i = tl.load(
            delta_ptr + off_bh * stride_deltah + offs_m * stride_deltam,
            mask=offs_m < seq_len, other=0.0,
        )

        # FA1 스타일: 자연로그 LSE + tl.exp 사용
        qk = tl.dot(q, tl.trans(k))
        p = tl.exp(qk * sm_scale - l_i[:, None])

        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & \
                    (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        else:
            valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, p, 0.0)

        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None]) * sm_scale

        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)

    tl.store(
        dk_ptr + off_bh * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.type.element_ty), mask=kv_mask,
    )
    tl.store(
        dv_ptr + off_bh * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.type.element_ty), mask=kv_mask,
    )


@triton.jit
def _fa1_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr,
    dq_ptr,
    sm_scale,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_doh, stride_dom, stride_dod,
    stride_lh, stride_lm,
    stride_deltah, stride_deltam,
    stride_dqh, stride_dqm, stride_dqd,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dQ_i 계산. 외부 = Q (i) 고정, 내부 = K/V (j) 순회."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)

    q = tl.load(
        q_ptr + off_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    )
    do = tl.load(
        do_ptr + off_bh * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=q_mask, other=0.0,
    )
    l_i = tl.load(
        lse_ptr + off_bh * stride_lh + offs_m * stride_lm,
        mask=offs_m < seq_len, other=0.0,
    )
    d_i = tl.load(
        delta_ptr + off_bh * stride_deltah + offs_m * stride_deltam,
        mask=offs_m < seq_len, other=0.0,
    )

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
    else:
        hi = seq_len

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)

        k = tl.load(
            k_ptr + off_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask, other=0.0,
        )
        v = tl.load(
            v_ptr + off_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask, other=0.0,
        )

        qk = tl.dot(q, tl.trans(k))
        p = tl.exp(qk * sm_scale - l_i[:, None])

        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & \
                    (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        else:
            valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None]) * sm_scale

        dq = tl.dot(ds.to(k.dtype), k, dq)

    tl.store(
        dq_ptr + off_bh * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.type.element_ty), mask=q_mask,
    )


def _fa1_backward(do, q, k, v, o, lse, causal, sm_scale):
    """FA1 backward 래퍼. lse는 자연로그 LSE."""
    bh, n, d = q.shape
    BLOCK_D = triton.next_power_of_2(d)
    do = do.contiguous()

    delta = torch.empty((bh, n), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    BLOCK_M_PRE = 64
    grid_pre = (triton.cdiv(n, BLOCK_M_PRE), bh)
    _fa1_bwd_preprocess_kernel[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1),
        n, HEAD_DIM=d,
        BLOCK_M=BLOCK_M_PRE, BLOCK_D=BLOCK_D,
    )

    BLOCK_M_BWD = 64
    BLOCK_N_BWD = 64
    grid_dkv = (triton.cdiv(n, BLOCK_N_BWD), bh)
    _fa1_bwd_dkdv_kernel[grid_dkv](
        q, k, v, do, lse, delta, dk, dv, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        n, HEAD_DIM=d, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD,
        num_warps=4, num_stages=2,
    )

    grid_dq = (triton.cdiv(n, BLOCK_M_BWD), bh)
    _fa1_bwd_dq_kernel[grid_dq](
        q, k, v, do, lse, delta, dq, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        n, HEAD_DIM=d, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD,
        num_warps=4, num_stages=2,
    )

    return dq, dk, dv


class FlashAttentionV1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        orig_shape = q.shape
        if q.ndim == 4:
            b, h, n, d = q.shape
            bh = b * h
            q3 = q.reshape(bh, n, d).contiguous()
            k3 = k.reshape(bh, n, d).contiguous()
            v3 = v.reshape(bh, n, d).contiguous()
        else:
            q3 = q.contiguous()
            k3 = k.contiguous()
            v3 = v.contiguous()
            bh, n, d = q3.shape

        o3 = torch.empty_like(q3)
        lse = torch.empty((bh, n), device=q3.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = triton.next_power_of_2(d)
        grid = (triton.cdiv(n, BLOCK_M), bh)

        flash_attention_kernel[grid](
            q3, k3, v3, o3, lse,
            q3.stride(0), q3.stride(1), q3.stride(2),
            k3.stride(0), k3.stride(1), k3.stride(2),
            v3.stride(0), v3.stride(1), v3.stride(2),
            o3.stride(0), o3.stride(1), o3.stride(2),
            lse.stride(0), lse.stride(1),
            n, d, d ** -0.5,
            IS_CAUSAL=causal, SAVE_LSE=True,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q3, k3, v3, o3, lse)
        ctx.causal = causal
        ctx.sm_scale = d ** -0.5
        ctx.orig_shape = orig_shape
        return o3.reshape(orig_shape)

    @staticmethod
    def backward(ctx, do):
        q3, k3, v3, o3, lse = ctx.saved_tensors
        orig_shape = ctx.orig_shape

        if do.ndim == 4:
            b, h, n, d = do.shape
            do3 = do.reshape(b * h, n, d).contiguous()
        else:
            do3 = do.contiguous()

        dq, dk, dv = _fa1_backward(do3, q3, k3, v3, o3, lse, ctx.causal, ctx.sm_scale)
        return (
            dq.reshape(orig_shape),
            dk.reshape(orig_shape),
            dv.reshape(orig_shape),
            None,
        )


def flash_attention_autograd(q, k, v, causal=False):
    """학습 가능한 FA1 (autograd 지원, forward + backward)"""
    return FlashAttentionV1Function.apply(q, k, v, causal)


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
        save_path=os.path.join(os.path.dirname(__file__), "benchmark.png"),
    )

    # --- Backward 정확도 검증 ---
    print("--- Backward 정확도 검증 (FA1 autograd vs PyTorch) ---")
    torch.manual_seed(0)
    for n in [128, 256, 512, 1024]:
        for causal in [False, True]:
            q = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            k = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            v = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            do = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16)

            out_ref = pytorch_attention(q, k, v, causal=causal)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), do)

            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            out_tri = flash_attention_autograd(q2, k2, v2, causal=causal)
            dq_tri, dk_tri, dv_tri = torch.autograd.grad(out_tri, (q2, k2, v2), do)

            tag = f"seq={n}{' causal' if causal else ''}"
            compare_results(out_tri, out_ref, atol=1e-1, rtol=1e-2, label=f"{tag} fwd ")
            compare_results(dq_tri, dq_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dQ  ")
            compare_results(dk_tri, dk_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dK  ")
            compare_results(dv_tri, dv_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dV  ")

    # --- LSE 저장 검증 ---
    print()
    print("--- logsumexp 저장 검증 ---")
    q = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    out, lse = flash_attention(q, k, v, causal=False, return_lse=True)
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (64 ** -0.5)
    lse_ref = torch.logsumexp(s, dim=-1)
    lse_diff = (lse - lse_ref).abs().max().item()
    status = "✓" if lse_diff < 1e-2 else "✗"
    print(f"  {status} LSE max diff: {lse_diff:.6e}")


if __name__ == "__main__":
    main()
