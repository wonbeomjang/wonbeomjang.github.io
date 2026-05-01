"""
09. Flash Attention 2 — FA1 대비 최적화

논문: Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning" (arXiv:2307.08691, 2023.07)

FA1 (05_flash_attention) 대비 핵심 개선:
  1. Algorithm 1 (논문 §3.1.1): "un-scaled" Õ를 누적 → 마지막에 1회만 정규화
     - FA1: 매 step마다 1/ℓ 곱셈 (non-matmul 연산 ↑)
     - FA2: ℓ로 나누는 건 마지막 한 번 → non-matmul FLOPs 감소
  2. exp → exp2 (논문 §3.1): A100/Ada는 exp2를 직접 지원, exp는 exp2·log2(e)로 구현됨.
     qk_scale에 log2(e)를 미리 곱해두면 매 step의 곱셈 1번 + exp 1번을 합쳐 절약.
  3. Causal mask 2단계 분할 (논문 §3.1.1, 6쪽):
     - STAGE 1: 대각선 이전 블록(행 > 열) — 마스크 분기 자체를 생략
     - STAGE 2: 대각선 블록 — 마스크 적용
     - 대각선 위쪽 블록은 루프 자체를 돌지 않음 → ~1.7-1.8× 속도↑
  4. tl.dot의 accumulator 인자 (P @ V를 한 명령으로 누적): tl.dot(p, v, acc)
  5. autotune (논문 §3.3): BLOCK_M/BLOCK_N/num_warps/num_stages 자동 탐색
  6. (옵션) logsumexp L = m + log(ℓ) 저장 → backward용 (논문 §3.1.1 Tweak 2)

병렬화 구조 (논문 §3.2):
  - 외부 루프 = Q (sequence length 차원), 내부 루프 = K/V
  - grid = (seq_len/BLOCK_M, batch*heads) → seq가 길면 SM 점유율 ↑
  - FA1 (논문 원본)이 KV-outer였던 것을 Phil Tillet가 Triton에서 Q-outer로 바꾼 것이
    FA2의 표준이 됨. 본 구현도 동일.
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
    scale = q.shape[-1] ** -0.5
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


# ============================================================
# 내부 함수: K/V 블록 순회 (STAGE별로 분기)
# ============================================================

@triton.jit
def _fa2_inner(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    start_m, qk_scale, seq_len,
    offs_m, offs_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    K/V 블록을 순회하며 online softmax 통계와 출력 누적기 업데이트.

    STAGE 1 (causal, 대각선 이전): offs_m > offs_n이 보장되는 블록만 → 마스크 X
    STAGE 2 (causal, 대각선 블록): 마스크 적용 (offs_m >= offs_n)
    STAGE 3 (non-causal): 전체 K/V 순회
    """
    # 단계별 K/V 순회 범위 (BLOCK_N | BLOCK_M 가정 → STAGE 1의 lo/hi가 BLOCK_N의 배수)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
    else:  # STAGE == 3 (non-causal)
        lo, hi = 0, seq_len

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K 로드 (BLOCK_N × BLOCK_D)
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=k_mask, other=0.0,
        )

        # S = Q K^T (matmul — 가장 비싼 연산이자 우리가 최대한 시간을 쓰고 싶은 곳)
        qk = tl.dot(q, tl.trans(k))

        # FA2 트릭: scale을 곱한 후 mask를 큰 음수로 더해 한 번에 처리
        # qk는 fp32 누적이므로 안전
        if STAGE == 2:
            valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seq_len)
            qk = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)
        elif STAGE == 3:
            valid = offs_n[None, :] < seq_len
            qk = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)
        else:  # STAGE == 1: 마스크 자체 불필요 (큰 절약)
            qk = qk * qk_scale

        # online softmax: 새 max
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        # FA2: exp 대신 exp2 (qk_scale에 log2(e) 미리 포함)
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # 보정 계수 (이전 누적값을 새 max 기준으로 스케일)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # FA2 핵심: acc는 "un-scaled" 상태로 유지 (1/ℓ 곱셈은 마지막 한 번만)
        acc = acc * alpha[:, None]

        # V 로드 (BLOCK_N × BLOCK_D)
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=v_mask, other=0.0,
        )
        # P @ V 누적 — accumulator 인자로 한 번에 (FA2: 명령 1회)
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij

    return acc, l_i, m_i


# ============================================================
# Autotune 설정 (논문 §3.3: BLOCK 크기는 head_dim·SRAM에 따라 적응)
# ============================================================

_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=4, num_warps=8),
]


# ============================================================
# 메인 커널 (Algorithm 1)
# ============================================================

@triton.autotune(configs=_CONFIGS, key=["seq_len", "HEAD_DIM", "IS_CAUSAL"])
@triton.jit
def flash_attention_v2_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    stride_lh, stride_lm,
    seq_len,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SAVE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 외부 루프 (Q 블록 — 병렬, §3.2)
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    # batch*head 오프셋
    q_base = q_ptr + off_bh * stride_qh
    k_base = k_ptr + off_bh * stride_kh
    v_base = v_ptr + off_bh * stride_vh
    o_base = o_ptr + off_bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Q 로드 (외부 루프에서 한 번만 — 내부 루프 동안 SRAM 상주)
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    )

    # online softmax 통계 + 출력 누적기 초기화
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i = 1.0으로 초깃값 (첫 iteration에서 alpha = exp2(-inf - m) = 0이 되어 자동 소거)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # FA2: scale에 log2(e)를 미리 곱해 exp2와 결합 (exp(x) = exp2(x · log2(e)))
    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * LOG2E

    if IS_CAUSAL:
        # STAGE 1: 마스크 불필요한 블록들 (대부분의 일을 여기서 처리)
        acc, l_i, m_i = _fa2_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=1,
        )
        # STAGE 2: 대각선 블록 (마스크 적용)
        acc, l_i, m_i = _fa2_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=2,
        )
    else:
        acc, l_i, m_i = _fa2_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=3,
        )

    # FA2: 마지막에 한 번만 1/ℓ 곱셈 (Algorithm 1, line 12)
    acc = acc / l_i[:, None]

    # logsumexp 저장 (옵션, backward용 — Algorithm 1, line 13)
    # base-2 형태로 저장: L_2 = m + log2(ℓ). backward에서 exp2와 직접 결합.
    # 외부 사용자에게 자연로그로 노출할 때는 wrapper에서 ln(2) 곱셈.
    if SAVE_LSE:
        lse = m_i + tl.math.log2(l_i)
        lse_mask = offs_m < seq_len
        tl.store(
            lse_ptr + off_bh * stride_lh + offs_m * stride_lm,
            lse, mask=lse_mask,
        )

    # 출력 저장
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(o_ptr.type.element_ty),
        mask=o_mask,
    )


# ============================================================
# 래퍼 함수
# ============================================================

def flash_attention_v2(q, k, v, causal=False, return_lse=False):
    """
    Flash Attention 2 forward.

    Args:
        q, k, v: (batch, heads, seq, dim) 또는 (heads, seq, dim) 또는 (bh, seq, dim)
                 dtype은 float16 또는 bfloat16
        causal: causal masking (autoregressive 모델용)
        return_lse: logsumexp 반환 여부 (backward 구현 시 필요)

    Returns:
        out: q와 같은 shape
        (옵션) lse: (bh, seq) shape의 logsumexp
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.shape == k.shape == v.shape

    orig_shape = q.shape
    if q.ndim == 4:
        b, h, n, d = q.shape
        bh = b * h
        q = q.reshape(bh, n, d)
        k = k.reshape(bh, n, d)
        v = v.reshape(bh, n, d)
    elif q.ndim == 3:
        bh, n, d = q.shape
    else:
        raise ValueError("q는 3D 또는 4D 텐서여야 합니다")

    # 연속 메모리 보장 (stride 처리 단순화)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    lse = torch.empty((bh, n), device=q.device, dtype=torch.float32) if return_lse else None
    # autotune jit 호환을 위해 lse_ptr 자리에 항상 텐서를 넘김 (SAVE_LSE=False면 무시)
    lse_buf = lse if lse is not None else torch.empty(1, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(d)
    sm_scale = d ** -0.5

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]), bh)

    flash_attention_v2_kernel[grid](
        q, k, v, o, lse_buf,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse_buf.stride(0) if lse is not None else 0,
        lse_buf.stride(1) if lse is not None else 0,
        n,
        sm_scale,
        HEAD_DIM=d,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        SAVE_LSE=return_lse,
    )

    o = o.reshape(orig_shape)

    if return_lse:
        # 커널은 base-2 LSE 저장. 외부 출력은 자연로그로 변환.
        LN2 = 0.6931471805599453
        lse = lse * LN2
        if len(orig_shape) == 4:
            lse = lse.reshape(orig_shape[0], orig_shape[1], orig_shape[2])
        return o, lse
    return o


# ============================================================
# Backward — Algorithm 2 (논문 §3.1.2)
#
# 두 개의 분리된 커널 + preprocess로 atomic 없이 처리:
#   1) preprocess: D_i = rowsum(dO_i ∘ O_i)  ← (bh, N) 벡터
#   2) dKV kernel: outer = K/V 블록(j), inner = Q 블록(i)
#                  각 K_j, V_j 블록에 대해 dK_j, dV_j 누적
#   3) dQ kernel:  outer = Q 블록(i), inner = K/V 블록(j)
#                  각 Q_i 블록에 대해 dQ_i 누적
#
# 두 번 traverse하지만 atomic 경합/race 없음. 공식 FA2 Triton 구현 방식.
#
# 수식 (논문 §2.2 + §3.1.2):
#   D_i = rowsum(dO_i ∘ O_i)          ∈ R^{B_r}
#   S_ij = Q_i K_j^T                   (matmul 1)
#   P_ij = exp(S_ij · α - L_i)          ← α = 1/√d, L_i는 자연로그 LSE
#         = exp2(S_ij · α · log2(e) - L_i_log2)   ← 본 구현 (base-2)
#   dV_j += P_ij^T · dO_i              (matmul 2)
#   dP_ij = dO_i · V_j^T               (matmul 3)
#   dS_ij = α · P_ij ∘ (dP_ij - D_i)
#   dQ_i += dS_ij · K_j                (matmul 4)
#   dK_j += dS_ij^T · Q_i              (matmul 5)
# ============================================================

@triton.jit
def _fa2_bwd_preprocess_kernel(
    o_ptr, do_ptr, delta_ptr,
    stride_oh, stride_om, stride_od,
    stride_dh, stride_dm,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """D_i = rowsum(dO_i ∘ O_i) ∈ R^{N} 계산. backward의 모든 i, j에서 재사용."""
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
def _fa2_bwd_dkdv_kernel(
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
    BLOCK_M: tl.constexpr,    # Q (inner)
    BLOCK_N: tl.constexpr,    # K/V (outer)
):
    """dK_j, dV_j 계산. 외부=K/V(j) 고정, 내부=Q(i) 순회."""
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # K_j, V_j는 외부 루프 동안 고정 (SRAM 상주)
    kv_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    k = tl.load(
        k_ptr + off_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=kv_mask, other=0.0,
    )
    v = tl.load(
        v_ptr + off_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=kv_mask, other=0.0,
    )

    # dK_j, dV_j 누적기 (FP32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # base-2 트릭: qk_scale에 log2(e) 미리 포함
    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * LOG2E

    # Causal: K_j는 Q_i (i*BLOCK_M >= j*BLOCK_N에 해당하는 i)와만 attention
    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    # Q 블록 순회
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

        # S_ij = Q_i K_j^T — matmul 1
        qk = tl.dot(q, tl.trans(k))

        # P_ij = exp2(qk · qk_scale - L_i)   (L_i는 base-2 LSE)
        p = tl.math.exp2(qk * qk_scale - l_i[:, None])

        # 마스킹
        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & \
                    (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        else:
            valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, p, 0.0)

        # dV_j += P_ij^T @ dO_i — matmul 2
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)

        # dP_ij = dO_i V_j^T — matmul 3
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        # dS_ij = α · P_ij ∘ (dP_ij - D_i)
        ds = p * (dp - d_i[:, None]) * sm_scale

        # dK_j += dS_ij^T @ Q_i — matmul 5
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)

    # dK_j, dV_j 저장
    tl.store(
        dk_ptr + off_bh * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.type.element_ty), mask=kv_mask,
    )
    tl.store(
        dv_ptr + off_bh * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.type.element_ty), mask=kv_mask,
    )


@triton.jit
def _fa2_bwd_dq_kernel(
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
    BLOCK_M: tl.constexpr,    # Q (outer)
    BLOCK_N: tl.constexpr,    # K/V (inner)
):
    """dQ_i 계산. 외부=Q(i) 고정, 내부=K/V(j) 순회."""
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

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * LOG2E

    # Causal: Q_i는 K_j (j*BLOCK_N <= (i+1)*BLOCK_M - 1)와만 attention
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
        p = tl.math.exp2(qk * qk_scale - l_i[:, None])

        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & \
                    (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        else:
            valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - d_i[:, None]) * sm_scale

        # dQ_i += dS_ij @ K_j — matmul 4
        dq = tl.dot(ds.to(k.dtype), k, dq)

    tl.store(
        dq_ptr + off_bh * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.type.element_ty), mask=q_mask,
    )


def _fa2_backward(do, q, k, v, o, lse_log2, causal, sm_scale):
    """
    Backward 래퍼. lse_log2는 forward에서 저장한 base-2 LSE.

    Args:
        do, q, k, v, o: (bh, seq, dim)
        lse_log2: (bh, seq), base-2 LSE
    Returns:
        dq, dk, dv: q와 같은 shape
    """
    bh, n, d = q.shape
    BLOCK_D = triton.next_power_of_2(d)

    do = do.contiguous()
    delta = torch.empty((bh, n), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # 1. Preprocess
    BLOCK_M_PRE = 64
    grid_pre = (triton.cdiv(n, BLOCK_M_PRE), bh)
    _fa2_bwd_preprocess_kernel[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        delta.stride(0), delta.stride(1),
        n, HEAD_DIM=d,
        BLOCK_M=BLOCK_M_PRE, BLOCK_D=BLOCK_D,
    )

    # 2. dKV
    BLOCK_M_BWD = 64
    BLOCK_N_BWD = 64
    grid_dkv = (triton.cdiv(n, BLOCK_N_BWD), bh)
    _fa2_bwd_dkdv_kernel[grid_dkv](
        q, k, v, do, lse_log2, delta, dk, dv,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        lse_log2.stride(0), lse_log2.stride(1),
        delta.stride(0), delta.stride(1),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        n, HEAD_DIM=d, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD,
        num_warps=4, num_stages=2,
    )

    # 3. dQ
    grid_dq = (triton.cdiv(n, BLOCK_M_BWD), bh)
    _fa2_bwd_dq_kernel[grid_dq](
        q, k, v, do, lse_log2, delta, dq,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        lse_log2.stride(0), lse_log2.stride(1),
        delta.stride(0), delta.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        n, HEAD_DIM=d, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M_BWD, BLOCK_N=BLOCK_N_BWD,
        num_warps=4, num_stages=2,
    )

    return dq, dk, dv


# ============================================================
# torch.autograd.Function — 학습 가능 attention layer
# ============================================================

class FlashAttentionV2Function(torch.autograd.Function):
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

        BLOCK_D = triton.next_power_of_2(d)
        sm_scale = d ** -0.5

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]), bh)
        flash_attention_v2_kernel[grid](
            q3, k3, v3, o3, lse,
            q3.stride(0), q3.stride(1), q3.stride(2),
            k3.stride(0), k3.stride(1), k3.stride(2),
            v3.stride(0), v3.stride(1), v3.stride(2),
            o3.stride(0), o3.stride(1), o3.stride(2),
            lse.stride(0), lse.stride(1),
            n, sm_scale,
            HEAD_DIM=d, BLOCK_D=BLOCK_D,
            IS_CAUSAL=causal, SAVE_LSE=True,
        )

        ctx.save_for_backward(q3, k3, v3, o3, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
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

        dq, dk, dv = _fa2_backward(
            do3, q3, k3, v3, o3, lse, ctx.causal, ctx.sm_scale,
        )

        return (
            dq.reshape(orig_shape),
            dk.reshape(orig_shape),
            dv.reshape(orig_shape),
            None,  # causal에 대한 gradient는 없음
        )


def flash_attention_v2_autograd(q, k, v, causal=False):
    """학습 가능한 FA2 (autograd 지원, forward + backward)"""
    return FlashAttentionV2Function.apply(q, k, v, causal)


# ============================================================
# 메인: 정확도 검증 + FA1과의 벤치마크 비교
# ============================================================

def main():
    print("=" * 60)
    print("09. Flash Attention 2 — FA1 대비 최적화")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 (3D, num_heads × seq × head_dim) ---")
    torch.manual_seed(42)
    num_heads, head_dim = 8, 64

    for n in [128, 256, 512, 1024]:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        out = flash_attention_v2(q, k, v, causal=False)
        ref = pytorch_attention(q, k, v, causal=False)
        compare_results(out, ref, atol=1e-1, rtol=1e-2, label=f"seq={n}")

        out_c = flash_attention_v2(q, k, v, causal=True)
        ref_c = pytorch_attention(q, k, v, causal=True)
        compare_results(out_c, ref_c, atol=1e-1, rtol=1e-2, label=f"seq={n} causal")

    # --- 4D 입력 ---
    print()
    print("--- 4D (batch, head, seq, dim) 입력 ---")
    q4 = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    k4 = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    v4 = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    out4 = flash_attention_v2(q4, k4, v4, causal=True)
    ref4 = pytorch_attention(q4, k4, v4, causal=True)
    compare_results(out4, ref4, atol=1e-1, rtol=1e-2, label="4D causal")

    # --- head_dim 128 (대형 모델) ---
    print()
    print("--- head_dim=128 (Llama/Qwen 표준) ---")
    q128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    k128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    v128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    out128 = flash_attention_v2(q128, k128, v128, causal=True)
    ref128 = pytorch_attention(q128, k128, v128, causal=True)
    compare_results(out128, ref128, atol=1e-1, rtol=1e-2, label="head_dim=128 causal")

    # --- LSE 저장 검증 ---
    print()
    print("--- logsumexp 저장 검증 ---")
    q = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    out, lse = flash_attention_v2(q, k, v, causal=False, return_lse=True)
    # 참조 LSE: log(sum(exp(QK^T/sqrt(d)), dim=-1))
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (64 ** -0.5)
    lse_ref = torch.logsumexp(s, dim=-1)
    lse_diff = (lse - lse_ref).abs().max().item()
    status = "✓" if lse_diff < 1e-2 else "✗"
    print(f"  {status} LSE max diff: {lse_diff:.6e}")

    # --- FA1과 비교 벤치마크 ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_flash_attention"))
    from flash_attention import flash_attention as flash_attention_v1

    print()
    print("--- 벤치마크: FA1 vs FA2 vs PyTorch (non-causal) ---")
    num_heads, head_dim = 16, 64
    seq_lengths = [256, 512, 1024, 2048, 4096]

    fa1_ms, fa2_ms, torch_ms = [], [], []
    for n in seq_lengths:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        fa1_ms.append(benchmark_fn(flash_attention_v1, q, k, v, causal=False))
        fa2_ms.append(benchmark_fn(flash_attention_v2, q, k, v, causal=False))
        torch_ms.append(benchmark_fn(pytorch_attention, q, k, v, causal=False))

    from tabulate import tabulate as tabulate_fn
    rows = []
    for n, t1, t2, tp in zip(seq_lengths, fa1_ms, fa2_ms, torch_ms):
        rows.append([n, f"{t1:.4f}", f"{t2:.4f}", f"{tp:.4f}",
                     f"{t1/t2:.2f}x", f"{tp/t2:.2f}x"])
    print()
    print(tabulate_fn(
        rows,
        headers=["Seq Len", "FA1 (ms)", "FA2 (ms)", "PyTorch (ms)",
                 "FA2/FA1", "FA2/PyTorch"],
        tablefmt="github",
    ))
    print()

    # --- causal 벤치마크 (FA2의 STAGE 분할 효과 확인) ---
    print("--- 벤치마크: FA1 vs FA2 vs PyTorch (causal) ---")
    fa1_ms_c, fa2_ms_c, torch_ms_c = [], [], []
    for n in seq_lengths:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        fa1_ms_c.append(benchmark_fn(flash_attention_v1, q, k, v, causal=True))
        fa2_ms_c.append(benchmark_fn(flash_attention_v2, q, k, v, causal=True))
        torch_ms_c.append(benchmark_fn(pytorch_attention, q, k, v, causal=True))

    rows = []
    for n, t1, t2, tp in zip(seq_lengths, fa1_ms_c, fa2_ms_c, torch_ms_c):
        rows.append([n, f"{t1:.4f}", f"{t2:.4f}", f"{tp:.4f}",
                     f"{t1/t2:.2f}x", f"{tp/t2:.2f}x"])
    print(tabulate_fn(
        rows,
        headers=["Seq Len", "FA1 (ms)", "FA2 (ms)", "PyTorch (ms)",
                 "FA2/FA1", "FA2/PyTorch"],
        tablefmt="github",
    ))
    print()

    # --- 메모리 비교 ---
    print("--- 메모리 사용량 (Standard vs FA2) ---")
    for n in [1024, 2048, 4096, 8192]:
        try:
            q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

            torch.cuda.reset_peak_memory_stats()
            _ = pytorch_attention(q, k, v)
            std_mem = torch.cuda.max_memory_allocated() / 1024**2

            torch.cuda.reset_peak_memory_stats()
            _ = flash_attention_v2(q, k, v)
            fa2_mem = torch.cuda.max_memory_allocated() / 1024**2

            print(f"  seq={n}: Standard={std_mem:.1f}MB, FA2={fa2_mem:.1f}MB, "
                  f"절약={std_mem - fa2_mem:.1f}MB ({std_mem/fa2_mem:.1f}x)")
        except torch.cuda.OutOfMemoryError:
            print(f"  seq={n}: Standard OOM, FA2는 가능")
    print()

    # --- Backward 정확도 검증 ---
    print("--- Backward 정확도 검증 (FA2 autograd vs PyTorch) ---")
    torch.manual_seed(0)
    for n in [128, 256, 512, 1024]:
        for causal in [False, True]:
            q = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            k = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            v = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
            do = torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16)

            # 참조: PyTorch autograd
            out_ref = pytorch_attention(q, k, v, causal=causal)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), do)

            # FA2 autograd
            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            out_tri = flash_attention_v2_autograd(q2, k2, v2, causal=causal)
            dq_tri, dk_tri, dv_tri = torch.autograd.grad(out_tri, (q2, k2, v2), do)

            tag = f"seq={n}{' causal' if causal else ''}"
            compare_results(out_tri, out_ref, atol=1e-1, rtol=1e-2, label=f"{tag} fwd ")
            compare_results(dq_tri, dq_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dQ  ")
            compare_results(dk_tri, dk_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dK  ")
            compare_results(dv_tri, dv_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dV  ")

    # --- Forward + Backward 벤치마크 (causal) ---
    print()
    print("--- Forward+Backward 벤치마크 (causal, num_heads=16, head_dim=64) ---")

    def fa2_fwd_bwd(q, k, v, do, causal):
        out = flash_attention_v2_autograd(q, k, v, causal=causal)
        torch.autograd.grad(out, (q, k, v), do, retain_graph=False)

    def torch_fwd_bwd(q, k, v, do, causal):
        out = pytorch_attention(q, k, v, causal=causal)
        torch.autograd.grad(out, (q, k, v), do, retain_graph=False)

    fa2_fb_ms, torch_fb_ms = [], []
    rows = []
    for n in seq_lengths:
        q = torch.randn(num_heads // 2, 2, n, head_dim, device="cuda",
                        dtype=torch.float16, requires_grad=True)
        k = torch.randn(num_heads // 2, 2, n, head_dim, device="cuda",
                        dtype=torch.float16, requires_grad=True)
        v = torch.randn(num_heads // 2, 2, n, head_dim, device="cuda",
                        dtype=torch.float16, requires_grad=True)
        do = torch.randn_like(q)

        t_fa = benchmark_fn(fa2_fwd_bwd, q, k, v, do, True)
        t_pt = benchmark_fn(torch_fwd_bwd, q, k, v, do, True)
        fa2_fb_ms.append(t_fa)
        torch_fb_ms.append(t_pt)
        rows.append([n, f"{t_fa:.4f}", f"{t_pt:.4f}", f"{t_pt/t_fa:.2f}x"])

    print(tabulate_fn(
        rows,
        headers=["Seq Len", "FA2 fwd+bwd (ms)", "PyTorch fwd+bwd (ms)", "Speedup"],
        tablefmt="github",
    ))
    print()

    # --- 그래프: forward / forward+backward 두 subplot ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = list(range(len(seq_lengths)))
        width = 0.35

        # 왼쪽: forward only (causal)
        axes[0].bar([i - width/2 for i in x], fa2_ms_c, width,
                    label="FA2", color="#e74c3c")
        axes[0].bar([i + width/2 for i in x], torch_ms_c, width,
                    label="PyTorch", color="#3498db")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(s) for s in seq_lengths])
        axes[0].set_xlabel("Sequence Length")
        axes[0].set_ylabel("Time (ms, log scale)")
        axes[0].set_title("Forward (causal)")
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, which="both")

        # 오른쪽: forward + backward (causal)
        axes[1].bar([i - width/2 for i in x], fa2_fb_ms, width,
                    label="FA2 fwd+bwd", color="#e74c3c")
        axes[1].bar([i + width/2 for i in x], torch_fb_ms, width,
                    label="PyTorch fwd+bwd", color="#3498db")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(s) for s in seq_lengths])
        axes[1].set_xlabel("Sequence Length")
        axes[1].set_ylabel("Time (ms, log scale)")
        axes[1].set_title("Forward + Backward (causal)")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which="both")

        fig.suptitle(f"FA2 vs PyTorch on RTX 4080 (num_heads={num_heads}, head_dim={head_dim})")
        plt.tight_layout()
        # 어디서 실행해도 동작하도록 이 파일 기준 절대경로
        save_path = os.path.join(os.path.dirname(__file__), "benchmark.png")
        plt.savefig(save_path, dpi=150)
        print(f"그래프 저장: {save_path}")
    except ImportError:
        print("matplotlib 없음 — 그래프 생략")


if __name__ == "__main__":
    main()
