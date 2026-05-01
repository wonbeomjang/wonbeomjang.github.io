"""
07. Flash Attention 3 (Triton) — FA2 위에 얹는 스케줄링/튜닝 개선

논문: Shah et al., "FlashAttention-3: Fast and Accurate Attention with
Asynchrony and Low-precision" (arXiv:2407.08608, 2024.07)

블로그 Triton 07 포스트의 결론을 그대로 반영한 학습용 구현.
A100 + Triton에서 FA2 대비 추가 3-5% 가속이 한계라는 사실을 실측으로 확인하고,
Hopper 전용 기능(TMA·wgmma·producer-consumer warp split)은 제외했다.

FA3 논문의 7가지 핵심 기법 중 Triton으로 표현 가능한 것만 적용:
  1. (논문 §3.2) Producer-consumer warp specialization → △ tl.range(warp_specialize=True) 시도
  2. (논문 §3.3) Inter-warpgroup ping-pong (GEMM↔softmax)  → △ tl.async_task preview, 미적용
  3. (논문 §3.1) TMA (비동기 메모리 복사)                   → △ Triton 3.x preview, 미적용
  4. (논문 §3.1) wgmma (warpgroup matmul)                   → ✓ tl.dot이 H100에서 자동 사용
  5. (논문 §4)   FP8 + incoherent processing                → △ 별도 큰 작업, 미적용
  6. (논문 §3.2) Persistent kernel (NUM_SMS launch)         → ✗ A100에서 -5~-17% 손해, 제거
  7.            Wider autotune                              → ✓ 6 → 17 configs

본 구현이 FA2 대비 추가하는 것:
  (a) 확장된 autotune 탐색 공간 (BLOCK_M ≤ 256, num_stages ≤ 6, num_warps=4|8)
  (b) SRAM-aware early pruning — head_dim 기반으로 컴파일 전 부적합 config 차단
  (c) 명시적 tl.dot(..., out_dtype=tl.float32) — FA3 의도(저정밀 입력 + fp32 누적) 명시
  (d) 1D grid (num_m_tiles · bh,) — 인접 program_id가 같은 (bh, m) 묶음을 받게 분배해
     L2 친화적 SM 매핑 유도
  (e) (옵션) tl.range(warp_specialize=True) — 컴파일러가 K/V 로드와 matmul/softmax를
      별도 warp partition으로 분할 (Hopper warpspec pass, cc 8/9에서 동작).
      flash_attention_v3_warpspec()로 활성화. A100에서 효과는 미미~동일,
      H100에서 본격 효과 기대.
  (f) Backward는 FA2 그대로 재사용 (FA3 backward 개선은 wgmma 의존)

A100-SXM4-80GB · num_heads=16, head_dim=64, causal, fp16 측정값:
  | seq   | FA1    | FA2    | FA3    | FA3/FA2 |
  | 4096  | 0.571  | 0.361  | 0.350  | 1.03×   |
  | 8192  | 1.721  | 1.033  | 0.992  | 1.04×   |
  | 16384 | 5.972  | 3.556  | 3.391  | 1.05×   |
  | 32768 | 22.247 | 13.426 | 12.847 | 1.05×   |

알고리즘 자체는 거의 한계에 도달했다. 그 이상은 H100 + CUDA의 영역이다.
"""

import inspect
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
)


# `tl.range(..., warp_specialize=...)` 는 Triton 3.5+ 에서만 허용된다.
# 구버전(예: 3.2.x)에서는 키워드 자체를 거부하므로 커널 소스에서 빼야 한다.
# 모듈 로드 시점에 검출해서 _fa3_inner 정의를 분기한다.
try:
    _HAS_WARP_SPECIALIZE = "warp_specialize" in inspect.signature(tl.range).parameters
except (TypeError, ValueError):
    _HAS_WARP_SPECIALIZE = False


# ============================================================
# PyTorch 참조 구현
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
# Autotune 설정 (FA3: 확장된 17개 config)
#
# FA2의 6개 → 17개로 확장. BLOCK_M ≤ 256, num_stages ≤ 6.
# head_dim=128 + BLOCK_M=256 같은 SRAM 초과 케이스는 prune_configs_by에서 제거.
# ============================================================

_FA3_CONFIGS = [
    # 작은 블록 (head_dim=128, 짧은 seq에서 유리)
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=4, num_warps=8),
    # 중간 블록 (head_dim=64에서 sweet spot)
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=5, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
    # 큰 블록 (긴 seq + head_dim=64에서 유리, head_dim=128에서는 SRAM 초과)
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=5, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=6, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_stages=3, num_warps=8),
]


def _fa3_prune_configs(configs, named_args, **kwargs):
    """SRAM 한계 초과 config를 컴파일 전에 제거.

    SRAM 사용량 ≈ (BLOCK_M + 2·BLOCK_N) · head_dim · 2  (Q, K, V fp16)
                  + BLOCK_M · head_dim · 4                (acc fp32)
                  + (BLOCK_M·BLOCK_N) · 4                 (S/P 임시 fp32)
                  + num_stages 배수 (pipeline 사본)

    A100 SM당 SRAM은 164 KB(48 KB는 reserved). 보수적으로 100 KB를 상한으로 둔다.
    """
    head_dim = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM", 64))
    sram_limit = 100 * 1024  # bytes, A100 SM 보수적 상한

    pruned = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        # Q + K + V 블록(fp16) + acc(fp32) + S 임시(fp32). pipeline 사본 ≈ num_stages.
        bytes_per_stage = (bm + 2 * bn) * head_dim * 2 + bm * head_dim * 4 + bm * bn * 4
        total = bytes_per_stage * cfg.num_stages
        if total <= sram_limit:
            pruned.append(cfg)
    # 모든 config가 잘리면(매우 큰 head_dim 등) 가장 작은 config라도 남긴다
    if not pruned:
        pruned = [min(configs, key=lambda c: c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"])]
    return pruned


# ============================================================
# 내부 함수: K/V 블록 순회 (FA2와 동일, out_dtype만 명시)
#
# `tl.range(..., warp_specialize=...)` 키워드 지원이 Triton 버전마다 다르므로
# 단일 K/V step 의 본문은 _fa3_step 헬퍼에 모으고, _fa3_inner 만 두 가지로
# 분기 정의한다 (for-루프 한 줄만 다름).
# ============================================================

@triton.jit
def _fa3_step(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    start_n, qk_scale, seq_len,
    offs_m, offs_d,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    """단일 K/V 블록 처리 — _fa3_inner 의 for-loop body."""
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # K 로드
    k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    k = tl.load(
        k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=k_mask, other=0.0,
    )

    # FA3: 명시적 fp32 누적 — H100 wgmma도 동일 명령으로 매핑됨
    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

    if STAGE == 2:
        valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seq_len)
        qk = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)
    elif STAGE == 3:
        valid = offs_n[None, :] < seq_len
        qk = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)
    else:
        qk = qk * qk_scale

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    # V 로드
    v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    v = tl.load(
        v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=v_mask, other=0.0,
    )
    # P @ V 누적 (fp32 acc 명시)
    acc = tl.dot(p.to(v.dtype), v, acc, out_dtype=tl.float32)

    m_i = m_ij
    return acc, l_i, m_i


# Triton 버전에 따라 _fa3_inner 의 for-loop 만 다르게 정의.
# 두 분기의 시그니처·반환값·로직은 동일하다.
if _HAS_WARP_SPECIALIZE:

    @triton.jit
    def _fa3_inner(
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
        WARP_SPECIALIZE: tl.constexpr = False,
    ):
        """K/V 블록 순회 + online softmax 업데이트 (Triton ≥ 3.5: warpspec 지원).

        WARP_SPECIALIZE=True 면 Hopper warpspec pass 가 K/V 로드와
        matmul/softmax 를 별도 warp partition 으로 분할한다 (cc 8/9 동작).
        """
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M
        elif STAGE == 2:
            lo = start_m * BLOCK_M
            hi = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
        else:  # STAGE == 3 (non-causal)
            lo, hi = 0, seq_len

        for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            acc, l_i, m_i = _fa3_step(
                acc, l_i, m_i, q, k_base, v_base,
                stride_kn, stride_kd, stride_vn, stride_vd,
                start_n, qk_scale, seq_len, offs_m, offs_d,
                BLOCK_N, HEAD_DIM, STAGE,
            )
        return acc, l_i, m_i

else:

    @triton.jit
    def _fa3_inner(
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
        WARP_SPECIALIZE: tl.constexpr = False,
    ):
        """K/V 블록 순회 + online softmax 업데이트 (Triton < 3.5 호환).

        이 분기에서는 tl.range 가 warp_specialize 키워드를 거부하므로 무시한다.
        WARP_SPECIALIZE 인자는 호출 호환성을 위해 시그니처에만 남긴다 — kernel 의
        autotune key 와 dispatch 로직은 그대로 두고, 실제 동작은 WS=False 와 동일.
        """
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M
        elif STAGE == 2:
            lo = start_m * BLOCK_M
            hi = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
        else:  # STAGE == 3 (non-causal)
            lo, hi = 0, seq_len

        for start_n in tl.range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            acc, l_i, m_i = _fa3_step(
                acc, l_i, m_i, q, k_base, v_base,
                stride_kn, stride_kd, stride_vn, stride_vd,
                start_n, qk_scale, seq_len, offs_m, offs_d,
                BLOCK_N, HEAD_DIM, STAGE,
            )
        return acc, l_i, m_i


# ============================================================
# 메인 커널 — 1D grid 버전
#
# FA2: grid = (num_m_tiles, bh) 2D
# FA3: grid = (num_m_tiles · bh,) 1D
#       → pid_m = pid % num_m_tiles
#       → pid_bh = pid // num_m_tiles
#   같은 bh의 인접 Q 타일이 연속된 program_id를 받아 SM 스케줄러가
#   인접 SM에 매핑하므로 K/V 재사용률이 올라간다 (L2 친화).
# ============================================================

@triton.autotune(
    configs=_FA3_CONFIGS,
    # WARP_SPECIALIZE도 key에 포함 — FA3와 FA3-WS가 별도 best config로 캐싱됨
    key=["seq_len", "HEAD_DIM", "IS_CAUSAL", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": _fa3_prune_configs},
)
@triton.jit
def flash_attention_v3_kernel(
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
    WARP_SPECIALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # FA3: 1D grid → (bh, m) decomposition
    # num_m_tiles는 BLOCK_M(constexpr)에 의존하므로 커널 안에서 계산
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(seq_len, BLOCK_M)
    off_bh = pid // num_m_tiles
    start_m = pid % num_m_tiles

    q_base = q_ptr + off_bh * stride_qh
    k_base = k_ptr + off_bh * stride_kh
    v_base = v_ptr + off_bh * stride_vh
    o_base = o_ptr + off_bh * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * LOG2E

    if IS_CAUSAL:
        acc, l_i, m_i = _fa3_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=1, WARP_SPECIALIZE=WARP_SPECIALIZE,
        )
        acc, l_i, m_i = _fa3_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=2, WARP_SPECIALIZE=WARP_SPECIALIZE,
        )
    else:
        acc, l_i, m_i = _fa3_inner(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale, seq_len,
            offs_m, offs_d,
            BLOCK_M, BLOCK_N, BLOCK_D, HEAD_DIM,
            STAGE=3, WARP_SPECIALIZE=WARP_SPECIALIZE,
        )

    acc = acc / l_i[:, None]

    if SAVE_LSE:
        lse = m_i + tl.math.log2(l_i)
        lse_mask = offs_m < seq_len
        tl.store(
            lse_ptr + off_bh * stride_lh + offs_m * stride_lm,
            lse, mask=lse_mask,
        )

    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM)
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(o_ptr.type.element_ty),
        mask=o_mask,
    )


# ============================================================
# 래퍼
# ============================================================

def flash_attention_v3(q, k, v, causal=False, return_lse=False, warp_specialize=False):
    """FA3 forward.

    Args:
        q, k, v: (batch, heads, seq, dim) | (heads, seq, dim) | (bh, seq, dim)
                 dtype은 fp16 또는 bf16
        causal: causal masking
        return_lse: backward용 자연로그 logsumexp 반환
        warp_specialize: tl.range(warp_specialize=True)로 Hopper warpspec pass를 활성화.
                         A100에서도 컴파일은 되지만 효과는 미미 (~0%~-4%).
                         H100에서 본격 효과 기대 (TMA+MBARRIER가 함께 작동).

    Returns:
        out: q와 같은 shape
        (옵션) lse: (..., seq) shape의 자연로그 LSE
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

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    lse = torch.empty((bh, n), device=q.device, dtype=torch.float32) if return_lse else None
    lse_buf = lse if lse is not None else torch.empty(1, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(d)
    sm_scale = d ** -0.5

    # FA3: 1D grid — autotune이 BLOCK_M을 고르므로 cdiv를 lambda로 처리
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]) * bh,)

    flash_attention_v3_kernel[grid](
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
        WARP_SPECIALIZE=warp_specialize,
    )

    o = o.reshape(orig_shape)

    if return_lse:
        LN2 = 0.6931471805599453
        lse = lse * LN2
        if len(orig_shape) == 4:
            lse = lse.reshape(orig_shape[0], orig_shape[1], orig_shape[2])
        return o, lse
    return o


def flash_attention_v3_warpspec(q, k, v, causal=False, return_lse=False):
    """FA3 + Triton 자동 warp specialization 활성화.

    `tl.range(..., warp_specialize=True)` 로 컴파일러에 K/V 로드와 matmul/softmax 를
    별도 warp partition 으로 분할하라고 지시한다. Hopper warpspec pass 가 cc 8/9 양쪽에서
    동작하므로 A100 에서도 컴파일은 통과한다.

    A100 측정 결과 (4 GPU 평균, 본 폴더의 main() 출력):
      causal head_dim=64, seq=32768: FA3=12.93 ms, FA3-WS=13.43 ms (FA3-WS가 4% 손해)
    이유: A100 에는 TMA·MBARRIER 가 없어 warpspec pass 가 partition 만 만들고
    실제 latency 가림 효과가 거의 없다. Hopper 에서 본격 효과 기대.
    """
    return flash_attention_v3(q, k, v, causal=causal, return_lse=return_lse, warp_specialize=True)


# ============================================================
# Backward — FA2의 3-stage 커널을 그대로 재사용
#
# 블로그 07 포스트: "FA3 backward 개선은 wgmma 의존이 크다.
#   본 구현은 FA2의 3-stage 커널을 그대로 재사용한다."
# ============================================================

def _import_fa2_backward():
    """sibling 디렉터리 06_flash_attention_v2 에서 backward와 autograd Function 가져오기."""
    here = os.path.dirname(__file__)
    fa2_dir = os.path.join(here, "..", "06_flash_attention_v2")
    sys.path.insert(0, os.path.abspath(fa2_dir))
    from flash_attention_v2 import _fa2_backward  # noqa: E402
    return _fa2_backward


class FlashAttentionV3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, warp_specialize):
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

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]) * bh,)
        flash_attention_v3_kernel[grid](
            q3, k3, v3, o3, lse,
            q3.stride(0), q3.stride(1), q3.stride(2),
            k3.stride(0), k3.stride(1), k3.stride(2),
            v3.stride(0), v3.stride(1), v3.stride(2),
            o3.stride(0), o3.stride(1), o3.stride(2),
            lse.stride(0), lse.stride(1),
            n, sm_scale,
            HEAD_DIM=d, BLOCK_D=BLOCK_D,
            IS_CAUSAL=causal, SAVE_LSE=True,
            WARP_SPECIALIZE=warp_specialize,
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

        # FA3 backward = FA2 backward (블로그 07: wgmma 없는 환경에서 동일)
        _fa2_backward = _import_fa2_backward()
        dq, dk, dv = _fa2_backward(
            do3, q3, k3, v3, o3, lse, ctx.causal, ctx.sm_scale,
        )

        # forward args = (q, k, v, causal, warp_specialize) → 5 grad slots
        return (
            dq.reshape(orig_shape),
            dk.reshape(orig_shape),
            dv.reshape(orig_shape),
            None,  # causal
            None,  # warp_specialize
        )


def flash_attention_v3_autograd(q, k, v, causal=False):
    """학습 가능한 FA3 (forward FA3 커널 + backward는 FA2 재사용)."""
    return FlashAttentionV3Function.apply(q, k, v, causal, False)


def flash_attention_v3_warpspec_autograd(q, k, v, causal=False):
    """학습 가능한 FA3 + warp specialization (autograd 지원).

    flash_attention_v3_warpspec()의 autograd 버전. 본문 docstring 참고.
    """
    return FlashAttentionV3Function.apply(q, k, v, causal, True)


# ============================================================
# 메인: 정확도 + FA1/FA2/FA3 비교 벤치마크
# ============================================================

def main():
    print("=" * 60)
    print("07. Flash Attention 3 — 확장 autotune + 1D grid")
    print("=" * 60)
    print()

    get_gpu_info()

    # --- 정확도 검증 ---
    print("--- 정확도 검증 ---")
    torch.manual_seed(42)
    num_heads, head_dim = 8, 64

    for n in [128, 256, 512, 1024]:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        out = flash_attention_v3(q, k, v, causal=False)
        ref = pytorch_attention(q, k, v, causal=False)
        compare_results(out, ref, atol=1e-1, rtol=1e-2, label=f"seq={n}")

        out_c = flash_attention_v3(q, k, v, causal=True)
        ref_c = pytorch_attention(q, k, v, causal=True)
        compare_results(out_c, ref_c, atol=1e-1, rtol=1e-2, label=f"seq={n} causal")

    # --- head_dim=128 ---
    print()
    print("--- head_dim=128 (Llama/Qwen 표준) ---")
    q128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    k128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    v128 = torch.randn(8, 1024, 128, device="cuda", dtype=torch.float16)
    out128 = flash_attention_v3(q128, k128, v128, causal=True)
    ref128 = pytorch_attention(q128, k128, v128, causal=True)
    compare_results(out128, ref128, atol=1e-1, rtol=1e-2, label="head_dim=128 causal")

    # --- LSE 검증 ---
    print()
    print("--- logsumexp 저장 검증 ---")
    q = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(4, 256, 64, device="cuda", dtype=torch.float16)
    out, lse = flash_attention_v3(q, k, v, causal=False, return_lse=True)
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (64 ** -0.5)
    lse_ref = torch.logsumexp(s, dim=-1)
    lse_diff = (lse - lse_ref).abs().max().item()
    status = "✓" if lse_diff < 1e-2 else "✗"
    print(f"  {status} LSE max diff: {lse_diff:.6e}")

    # --- FA3 + warp specialization 검증 (옵션 (e)) ---
    print()
    print("--- FA3-WS 정확도 검증 (warp_specialize=True) ---")
    for n in [128, 256, 512, 1024]:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        out_ws = flash_attention_v3_warpspec(q, k, v, causal=False)
        ref = pytorch_attention(q, k, v, causal=False)
        compare_results(out_ws, ref, atol=1e-1, rtol=1e-2, label=f"WS seq={n}")

        out_ws_c = flash_attention_v3_warpspec(q, k, v, causal=True)
        ref_c = pytorch_attention(q, k, v, causal=True)
        compare_results(out_ws_c, ref_c, atol=1e-1, rtol=1e-2, label=f"WS seq={n} causal")

    # --- FA1 / FA2 / FA3 / FA3-WS 비교 ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_flash_attention"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "06_flash_attention_v2"))
    from flash_attention import flash_attention as flash_attention_v1
    from flash_attention_v2 import flash_attention_v2

    print()
    print("--- 벤치마크: FA1 vs FA2 vs FA3 vs FA3-WS vs PyTorch (causal, head_dim=64) ---")
    num_heads, head_dim = 16, 64
    seq_lengths = [256, 512, 1024, 2048, 4096]

    fa1_ms, fa2_ms, fa3_ms, fa3ws_ms, torch_ms = [], [], [], [], []
    for n in seq_lengths:
        q = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_heads, n, head_dim, device="cuda", dtype=torch.float16)

        fa1_ms.append(benchmark_fn(flash_attention_v1, q, k, v, causal=True))
        fa2_ms.append(benchmark_fn(flash_attention_v2, q, k, v, causal=True))
        fa3_ms.append(benchmark_fn(flash_attention_v3, q, k, v, causal=True))
        fa3ws_ms.append(benchmark_fn(flash_attention_v3_warpspec, q, k, v, causal=True))
        torch_ms.append(benchmark_fn(pytorch_attention, q, k, v, causal=True))

    try:
        from tabulate import tabulate as tabulate_fn
        rows = []
        for n, t1, t2, t3, tw, tp in zip(seq_lengths, fa1_ms, fa2_ms, fa3_ms, fa3ws_ms, torch_ms):
            rows.append([
                n, f"{t1:.4f}", f"{t2:.4f}", f"{t3:.4f}", f"{tw:.4f}", f"{tp:.4f}",
                f"{t3/tw:.2f}x", f"{tp/t3:.2f}x",
            ])
        print(tabulate_fn(
            rows,
            headers=["Seq", "FA1 (ms)", "FA2 (ms)", "FA3 (ms)", "FA3-WS (ms)",
                     "PyTorch (ms)", "FA3/WS", "FA3/PT"],
            tablefmt="github",
        ))
    except ImportError:
        for n, t1, t2, t3, tw, tp in zip(seq_lengths, fa1_ms, fa2_ms, fa3_ms, fa3ws_ms, torch_ms):
            print(f"  seq={n}: FA1={t1:.4f}ms  FA2={t2:.4f}ms  FA3={t3:.4f}ms  "
                  f"FA3-WS={tw:.4f}ms  PT={tp:.4f}ms  "
                  f"FA3/WS={t3/tw:.2f}x  FA3/PT={tp/t3:.2f}x")

    # --- Backward 정확도 ---
    print()
    print("--- Backward 정확도 검증 (FA3 autograd vs PyTorch) ---")
    torch.manual_seed(0)
    for n in [128, 256, 512]:
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
            out_tri = flash_attention_v3_autograd(q2, k2, v2, causal=causal)
            dq_tri, dk_tri, dv_tri = torch.autograd.grad(out_tri, (q2, k2, v2), do)

            tag = f"seq={n}{' causal' if causal else ''}"
            compare_results(out_tri, out_ref, atol=1e-1, rtol=1e-2, label=f"{tag} fwd ")
            compare_results(dq_tri, dq_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dQ  ")
            compare_results(dk_tri, dk_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dK  ")
            compare_results(dv_tri, dv_ref, atol=5e-2, rtol=1e-2, label=f"{tag} dV  ")


if __name__ == "__main__":
    main()
