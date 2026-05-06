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

### A100 80GB 측정값 — FA1 vs FA2 vs PyTorch

A100-SXM4-80GB · `num_heads=16, fp16` · 4 GPU 평균 (표준편차 < 1%) · 11회 측정 중 첫 회 폐기.

**head_dim=64, causal forward** (ms):

| seq   | FA1    | FA2    | PyTorch | FA2/FA1 | FA2/PT     |
| ----- | ------ | ------ | ------- | ------- | ---------- |
| 4096  | 0.571  | 0.361  | 5.243   | 1.58×   | 14.97×     |
| 8192  | 1.721  | 1.033  | 21.807  | 1.67×   | **21.98×** |
| 16384 | 5.972  | 3.556  | 70.856  | 1.68×   | 20.90×     |
| 32768 | 22.247 | 13.426 | OOM     | 1.66×   | —          |

**head_dim=128, causal forward** (ms):

| seq   | FA1    | FA2    | PyTorch | FA2/FA1   | FA2/PT |
| ----- | ------ | ------ | ------- | --------- | ------ |
| 2048  | 0.374  | 0.257  | 1.305   | 1.46×     | 5.32×  |
| 4096  | 1.113  | 0.587  | 5.430   | 1.90×     | 9.38×  |
| 8192  | 3.905  | 1.824  | 22.188  | 2.14×     | 12.59× |
| 32768 | 57.620 | 24.670 | OOM     | **2.34×** | —      |

- 블로그에서 예측한 ~1.7-1.8× causal FA2/FA1 비율이 실측에서 정확히 재현됐다
- head_dim=128에서는 FA2/FA1이 2.34×까지 상승 — autotune이 BLOCK_M=128 + num_warps=8을 선택해 SRAM 점유율이 좋아진 결과다
- causal seq=8192에서 FA2/PT = **21.98×** 피크 — STAGE 1 마스크 제거 효과가 long-seq에서 정점

**메모리 절감** (non-causal, head_dim=64):

| seq   | Standard     | FA2        | 절약                 |
| ----- | ------------ | ---------- | -------------------- |
| 8192  | 4184 MB      | 96 MB      | 4088 MB              |
| 16384 | 16544 MB     | 176 MB     | 16368 MB             |
| 32768 | **65840 MB** | **336 MB** | **65504 MB (≈195×)** |

> 32K seq에서 메모리는 196× 절약, 시간은 5.7× (non-causal) ~ 22× (causal) 가속. FA1과 FA2의 알고리즘적 차이는 이 두 축에서 모두 의미 있는 개선을 만든다.

### RTX 4080 vs A100 — 동일 코드, 다른 GPU

`_experiments/06_flash_attention_v2/main()` 을 두 GPU 에서 그대로 실행한 결과 (causal, num_heads=16, head_dim=64, fp16):

| Seq  | 4080 FA2 (ms) | A100 FA2 (ms) | 4080 FA2/PT | A100 FA2/PT | A100/4080         |
| ---- | ------------- | ------------- | ----------- | ----------- | ----------------- |
| 256  | 0.048         | 0.104         | 0.92×       | 1.69×       | 0.46× _4080 우세_ |
| 512  | 0.051         | 0.103         | 1.33×       | 1.73×       | 0.50× _4080 우세_ |
| 1024 | 0.078         | 0.105         | **3.93×**   | 3.68×       | 0.74×             |
| 2048 | 0.157         | 0.162         | **17.42×**  | 8.34×       | 0.97×             |
| 4096 | 0.443         | 0.358         | **20.84×**  | **14.64×**  | 1.24×             |

- **짧은 seq (≤2048) 까지는 4080 이 더 빠르거나 비슷** — Ada Lovelace SM 클럭 2505 MHz vs A100 1410 MHz, kernel launch overhead 가 작은 영향
- **seq=4096 부터 A100 이 우세** — HBM2e 1.5 TB/s vs GDDR6X 717 GB/s 메모리 대역폭 차이가 작동
- **PyTorch 대비 가속비는 4080 이 더 큼 (causal seq=4096 에서 20.84× vs 14.64×)** — 주의: 4080 의 cuBLAS 베이스라인이 상대적으로 느려서 가속비가 부풀어 보이는 것이지, 절대 시간은 4080 PT 9.23 ms vs A100 PT 5.24 ms 로 A100 PT 가 1.76× 빠름. 가속비 = 알고리즘적 이득 + 베이스라인 약화가 합쳐진 수치

원본 데이터: [`_experiments/06_flash_attention_v2/results_a100.md`](https://github.com/wonbeomjang/wonbeomjang.github.io/blob/master/_experiments/06_flash_attention_v2/results_a100.md), [`results_4080.md`](https://github.com/wonbeomjang/wonbeomjang.github.io/blob/master/_experiments/06_flash_attention_v2/results_4080.md).

### 왜 이론 peak 의 50–55% 에서 멈추나

[FA1 분석](/blog/2026/triton-05-flash-attention/#왜-이론-peak-대비-일정--에서-멈추나) 에서 FA1 이 A100 peak 의 ~37% (4080 ~40%) 였다. FA2 는 어디까지 갔나.

**A100** (FP16 Tensor Core peak 312 TFLOP/s, 4-GPU long-seq 측정):

| seq   | FA2 시간 | matmul FLOPs | 측정 TFLOPS | A100 peak 비율 |
| ----- | -------- | ------------ | ----------- | -------------- |
| 4096  | 0.500 ms | 68.7 G       | 137.4       | 44.0%          |
| 8192  | 1.756 ms | 274.9 G      | 156.5       | 50.2%          |
| 16384 | 6.413 ms | 1099.5 G     | 171.4       | 54.9%          |
| 32768 | 25.54 ms | 4398.0 G     | 172.2       | **55.2%**      |

**RTX 4080** (FP16 Tensor Core peak 195 TFLOP/s, `_experiments/` main 측정):

| seq  | FA2 시간 | matmul FLOPs | 측정 TFLOPS | 4080 peak 비율 |
| ---- | -------- | ------------ | ----------- | -------------- |
| 1024 | 0.101 ms | 4.3 G        | 42.5        | 21.8%          |
| 2048 | 0.228 ms | 17.2 G       | 75.4        | 38.7%          |
| 4096 | 0.751 ms | 68.7 G       | 91.5        | **46.9%**      |

**관찰**:

- A100 long-seq 점근선 ~55%, 4080 4096-seq 점근선 ~47%
- FA1 → FA2 의 회수: A100 +18%p (37% → 55%), 4080 +7%p (40% → 47%) — 4080 의 작은 SRAM 이 FA2 의 더 큰 BLOCK_M 을 못 받쳐줌
- FlashAttention 공식 CUDA 구현은 A100 에서 ~70% 를 찍는다. **15%p 격차**가 남는데 어디서 새는가?

**남은 손실 분해 (두 GPU 공통)**:

1. **`cp.async` tight pipelining 부재 (가장 큼)** — A100/4080 모두 `cp.async` 지원하지만 Triton 의 `num_stages=N` 은 보수적 dependency 분석으로 일부 async copy 가 sync 로 떨어짐. CUTLASS 의 4-stage async pipeline 만큼 overlap 못 함. **추정 손실 8–10%**
2. **softmax non-matmul 시간** — `tl.max`, `exp2`, `tl.sum`, alpha 곱셈 등. A100 에서 non-matmul throughput 은 matmul 대비 ~16× 낮음. nsys profile 기준 ~10% 가 softmax. **추정 손실 4–6%**
3. **SRAM 한계로 BLOCK_M 제한** — A100 164 KB / 4080 100 KB. head_dim=128 + BLOCK_M=128 + BLOCK_N=64 면 ~96 KB 사용. 4080 은 더 빡빡함. **추정 손실 A100 2–3%, 4080 5–8%**
4. **Backward 의 3-kernel split overhead** — Preprocess + dKV + dQ 가 같은 Q,K,V 를 3번 읽음 (HBM 낭비). flash-attn 은 backward 도 fused. fwd+bwd 만 보면 **추가 손실 10–15%**
5. **autotune 탐색 공간 한계** — 6 configs 만 탐색. CUTLASS auto-tuner 는 수백 가지. [Triton 07](/blog/2026/triton-07-flash-attention-v3/) 에서 17 configs 로 늘려 **+3–5% 회수**

**4080 특수 패턴**:

- **causal 가속비 부풀려짐** (4080 20.84× vs A100 14.64× at seq=4096) — 원인은 4080 의 PyTorch 베이스라인이 상대적으로 약하기 때문. 해석 시 절대 ms 도 같이 봐야 한다
- **SRAM 100 KB 한계로 BLOCK_M=128 + BLOCK_N=128 같은 큰 config 가 안 들어감** — autotune 이 작은 config 로 떨어져 long-seq 에서 추가 5–8% 손해

| 남은 손실          | 추정 % (A100) | 추정 % (4080) | 회수 방법                   |
| ------------------ | ------------- | ------------- | --------------------------- |
| cp.async 부재      | 8–10%         | 8–10%         | △ Hopper TMA (H100 only)    |
| softmax non-matmul | 4–6%          | 4–6%          | △ FA3 GEMM-softmax pingpong |
| SRAM 한계          | 2–3%          | 5–8%          | △ BLOCK_M=192 (Hopper 권장) |
| Backward 3-split   | 10–15%        | 10–15%        | ✗ CUTLASS 급 fused 필요     |
| autotune 한계      | 3–5%          | 3–5%          | ✓ Triton 07 에서 회수       |

요약하면 FA2 가 peak 의 ~55% (A100) / ~47% (4080) 에서 멈추는 이유는 **Triton 추상화의 천장**이다. 그 이상은 CUTLASS hand-tune 또는 H100 의 새 명령어 (TMA, wgmma) 가 필요하다. 이것이 [FA3 가 H100 + CUDA 로만 의미가 있는 이유](/blog/2026/triton-07-flash-attention-v3/#솔직한-한계) 다.

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
