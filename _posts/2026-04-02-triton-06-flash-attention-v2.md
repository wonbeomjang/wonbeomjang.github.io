---
layout: post
title: "Triton 06: Flash Attention 2 — FA1 대비 5가지 최적화"
date: 2026-04-02 01:00:00 +0900
description: "Flash Attention 2를 Triton으로 구현한다 — un-scaled 누적, exp2 트릭, Causal 2-stage, tl.dot accumulator, autotune"
categories: [triton]
tags: [triton, gpu, flash-attention, llm, attention, optimization]
giscus_comments: true
related_posts: true
featured: true
---

[Triton 05: Flash Attention](/blog/2026/triton-05-flash-attention/)에서 FA1 기반 구현을 다뤘다. 이번 포스트에서는 [FlashAttention-2 논문](/blog/2023/flashattention-2/)의 핵심 알고리즘 개선을 Triton으로 구현한다.

> FA2 논문과 알고리즘의 원리가 궁금하다면 [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)를 먼저 읽어보는 것을 추천한다.

---

## FA1 대비 개선 요약

| #   | 개선 항목   | FA1                   | FA2                     | 효과                      |
| --- | ----------- | --------------------- | ----------------------- | ------------------------- |
| 1   | acc 정규화  | 매 K/V 블록마다 `/ℓ`  | **마지막에 1회만**      | non-matmul FLOPs ↓        |
| 2   | exp 함수    | `tl.exp`              | **`tl.math.exp2`**      | 하드웨어 명령어 직접 사용 |
| 3   | Causal mask | 루프 내 조건 분기     | **STAGE 1/2 분리**      | 마스크 분기 비용 제거     |
| 4   | P @ V 누적  | `acc += tl.dot(p, v)` | **`tl.dot(p, v, acc)`** | 명령어 1회로 단축         |
| 5   | Block 크기  | 64/64 고정            | **`@triton.autotune`**  | GPU별 최적값 자동 탐색    |

---

## 개선 1: Un-scaled 누적

FA1에서는 매 K/V 블록을 처리할 때마다 `acc`를 정규화된 상태로 유지했다.

**FA1 방식 (매 블록마다 `/ℓ` 포함):**

$$O_i^{(t)} = \text{diag}(l_i^{(t)})^{-1}\!\left(e^{m_i^{(t-1)} - m_i^{(t)}} \cdot \text{diag}(l_i^{(t-1)}) \cdot O_i^{(t-1)} + e^{S_i^{(t)} - m_i^{(t)}} V^{(t)}\right)$$

여기서 `diag(l)^{-1}` 나눗셈이 매 블록마다 실행된다. GPU에서 나눗셈은 non-matmul 연산이고, A100 기준 matmul 대비 **16배** 느리다.

**FA2 방식 (마지막에 1회만):**

$$\tilde{O}_i^{(t)} = e^{m_i^{(t-1)} - m_i^{(t)}} \tilde{O}_i^{(t-1)} + e^{S_i^{(t)} - m_i^{(t)}} V^{(t)}$$

$$O_i = \text{diag}(l_i)^{-1} \tilde{O}_i \quad \leftarrow \text{루프 종료 후 1회}$$

`acc`는 un-scaled 상태로 유지하고, 보정 계수 `alpha`로 이전 max 기준만 맞춰두다가 마지막에 한 번만 나눈다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet01_unscaled_accumulation.py"></script>

나눗셈 횟수: FA1은 `T_c`번(K/V 블록 수만큼), FA2는 **1번**이다. `T_c = N / BLOCK_N`이므로 시퀀스가 길수록 절감 효과가 크다.

---

## 개선 2: `exp` → `exp2` 트릭

A100/Ada/Hopper GPU는 `exp2`(밑이 2인 지수함수)를 하드웨어 명령어로 직접 지원한다. `tl.exp`는 내부적으로 `exp2(x × log₂e)`로 변환되어 곱셈이 1회 추가된다.

`qk_scale`에 `log₂e = 1.4427`을 **커널 진입 전 한 번만** 곱해두면 이 오버헤드를 제거할 수 있다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet02_exp2_trick.py"></script>

LSE도 base-2 형태로 저장하고, 래퍼 함수에서 자연로그로 변환해 반환한다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet03_base2_lse.py"></script>

`l_i` 초기화도 FA2에서 바뀐다. FA1은 `l_i = 0.0`으로 시작했지만 FA2는 `l_i = 1.0`으로 시작한다. 첫 번째 iteration에서 `alpha = exp2(-inf - m_ij) = 0`이 되어 `l_i * alpha = 1.0 * 0 = 0`으로 자동 소거되기 때문이다. 조건 분기 없이 초기화를 처리하는 트릭이다.

---

## 개선 3: Causal Mask 2-Stage 분기

FA1에서는 모든 K/V 블록을 순회하면서 매번 causal mask 조건을 체크했다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet04_causal_mask_fa1.py"></script>

Q 블록 `start_m`을 처리할 때, K/V 블록 `j`가 완전히 대각선 왼쪽에 있다면(`j * BLOCK_N < start_m * BLOCK_M`), 해당 블록의 **모든** 위치가 `offs_m >= offs_n`을 만족하므로 마스크 자체가 불필요하다.

FA2는 이를 STAGE로 명시적으로 분리한다.

```
Q 블록 start_m = 3 (BLOCK_M = 64, seq = 512) 기준:

K/V 블록:  j=0    j=1    j=2    j=3      j=4, 5, ...
          ┌────┬────┬────┬────────┐
STAGE 1:  │ ✓  │ ✓  │ ✓  │        │  → 전부 과거, 마스크 불필요
          └────┴────┴────┴────────┘
STAGE 2:               │  대각선  │  → 일부만 과거, 마스크 적용
                        └──────────┘
미순회:                           ...  → 전부 미래, 루프 자체 안 돎
```

`_fa2_inner`는 STAGE를 `tl.constexpr`로 받아 컴파일 타임에 분기를 제거한다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet05_fa2_inner_stage.py"></script>

메인 커널에서는 causal 여부에 따라 다르게 호출한다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet06_stage_dispatch.py"></script>

시퀀스가 길어질수록 STAGE 1이 담당하는 블록 수가 증가한다. seq=4096, BLOCK_M=BLOCK_N=64이면 Q 블록 하나당 STAGE 1이 63개, STAGE 2가 1개를 처리하므로 **마스크 체크 비용이 1/64로 줄어든다**.

---

## 개선 4: `tl.dot` Accumulator 인자

FA1에서는 행렬곱 결과를 누적기에 별도로 더했다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet07a_tldot_fa1.py"></script>

FA2에서는 `tl.dot`의 세 번째 인자에 누적기를 바로 넘긴다. 내부적으로 단일 FMA(Fused Multiply-Add)로 처리되어 중간 레지스터 할당이 줄어든다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet07b_tldot_fa2.py"></script>

Backward에서도 동일하게 적용한다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet07c_tldot_backward.py"></script>

---

## 개선 5: Autotune

FA1은 `BLOCK_M=64, BLOCK_N=64`로 고정됐다. GPU마다 SRAM 크기와 SM 수가 달라 최적 블록 크기가 다르기 때문에, FA2는 `@triton.autotune`으로 자동 탐색한다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet08_autotune.py"></script>

- `key`: 이 값의 조합마다 별도로 프로파일링하고 결과를 캐싱한다
- `num_stages`: software pipeline 깊이 (다음 블록 데이터를 미리 로드)
- `num_warps`: SM당 동시 실행 warp 수 (SRAM 공유 범위)

**GPU별 autotune 결과 예시:**

| GPU      | 선택 가능성 높은 config                              |
| -------- | ---------------------------------------------------- |
| RTX 4080 | `BLOCK_M=64, BLOCK_N=64, num_stages=3, num_warps=4`  |
| A100     | `BLOCK_M=128, BLOCK_N=64, num_stages=4, num_warps=4` |
| H100     | `BLOCK_M=128, BLOCK_N=64, num_stages=4, num_warps=8` |

---

## Backward 구현

구조는 [Triton 05: Flash Attention](/blog/2026/triton-05-flash-attention/#backward-구현)에서 다룬 3단계 커널(Preprocess → dKV → dQ)과 동일하다. 차이는 exp2 + base-2 LSE를 사용하는 것뿐이다.

<script src="https://gist.github.com/wonbeomjang/5880faa2b9aa8d0ab1bd1dd0ad31baa9.js?file=09_flash_attention_v2_snippet09_backward_exp2.py"></script>

### Forward/Backward 전체 흐름 요약

```
Forward:
  Q 블록 외부 루프, K/V 블록 내부 루프
  ├─ STAGE 1 (causal): 마스크 없이 K/V 순회
  ├─ STAGE 2 (causal): 대각선 블록만 마스크 적용
  └─ STAGE 3 (non-causal): 전체 K/V 순회
  마지막에: acc /= l_i,  LSE = m + log2(l)  저장

Backward:
  [Preprocess]  Δ_i = rowsum(dO_i ⊙ O_i)
  [dKV 커널]   외부=K/V(j), 내부=Q(i)
               P_ij = exp2(QK^T · scale · log₂e − L_i)
               dV_j += P_ij^T @ dO_i
               dK_j += (P_ij ⊙ (dO_i @ V_j^T − Δ_i) · scale)^T @ Q_i
  [dQ 커널]    외부=Q(i), 내부=K/V(j)
               dQ_i += (P_ij ⊙ (dO_i @ V_j^T − Δ_i) · scale) @ K_j
```

---

## 벤치마크 결과

RTX 4080 기준 FA1 vs FA2 vs PyTorch 비교:

{% include figure.liquid loading="lazy" path="assets/img/triton/06_flash_attention_v2/benchmark.png" class="img-fluid rounded z-depth-1" alt="FA1 vs FA2 vs PyTorch 성능 비교" %}

- **non-causal**: FA2가 FA1 대비 ~1.2–1.4× 빠름 (un-scaled 누적 + exp2 + accumulator 효과)
- **causal**: FA2가 FA1 대비 ~1.7–1.8× 빠름 (2-stage 마스크 제거 효과 추가)
- **seq_len이 길수록**: FA2 우위가 커짐 — STAGE 1 비율이 높아져 마스크 분기 비용이 희석됨
- **head_dim=128 (Llama/Qwen 표준)**: autotune이 더 큰 BLOCK_M을 선택하므로 FA2가 더 유리

---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=flash_attention_v2.py"></script>

---

> Flash Attention 1의 Triton 구현이 궁금하다면 [Triton 05: Flash Attention](/blog/2026/triton-05-flash-attention/)을, 알고리즘 원리가 궁금하다면 [FlashAttention 논문 리뷰](/blog/2023/fastattention/)와 [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)를, Hopper GPU 최적화가 궁금하다면 [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)를 참고하자.

# 참고 문헌

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton Fused Attention 공식 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention 논문 리뷰](/blog/2023/fastattention/)
- [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)
- [Triton 05: Flash Attention — 종합 프로젝트](/blog/2026/triton-05-flash-attention/)
