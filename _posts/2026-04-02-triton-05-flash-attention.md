---
layout: post
title: "Triton 05: Flash Attention — 종합 프로젝트"
date: 2026-04-02 00:00:00 +0900
description: Flash Attention을 Triton으로 구현한다 — Forward/Backward 전체 구현과 RTX 4080·A100·H100·B200 아키텍처별 최적화 포인트
categories: [triton]
tags: [triton, gpu, flash-attention, llm, attention, optimization]
giscus_comments: true
related_posts: true
featured: true
---

## 개요

지금까지 배운 모든 기법을 종합하여 Flash Attention을 구현한다.
LLM 추론/학습에서 가장 중요한 최적화 기법 중 하나다.

> Flash Attention의 원리와 논문 내용이 궁금하다면 [FlashAttention 논문 리뷰](/blog/2023/fastattention/)를 먼저 읽어보는 것을 추천한다.

---

## 핵심 개념

### Attention 수식

$$O = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V$$

- $$Q, K, V$$: Query, Key, Value 행렬 (각각 $$N \times d$$)
- $$\sqrt{d}$$: head dimension의 제곱근으로 나눠서 스케일링
- $$\text{softmax}$$: 행(row) 단위로 적용 → 확률 분포로 변환

### Standard Attention의 문제

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet01_Standard_Attention%EC%9D%98_%EB%AC%B8%EC%A0%9C.py"></script>

시퀀스 길이 N=4096, float16이면:

- S 행렬 크기: 4096 × 4096 × 2 bytes = **32MB**
- N=16384이면: **512MB** — 시퀀스가 길어질수록 VRAM 폭발

### Flash Attention의 핵심 아이디어

**S 행렬을 전체 생성하지 않는다!**

타일 단위로 Q, K, V를 처리하면서 결과를 점진적으로 누적한다.
이를 위해 **Online Softmax** 알고리즘이 필요하다.

### Online Softmax

데이터를 청크(블록) 단위로 받으면서 **점진적으로 업데이트**한다.

**청크 1 처리 후** ($$S_1$$ = 첫 번째 K 블록과의 attention score):

$$m^{(1)} = \max(S_1)$$

$$l^{(1)} = \sum_j e^{S_{1,j} - m^{(1)}}$$

$$O^{(1)} = \text{diag}(l^{(1)})^{-1} \cdot e^{S_1 - m^{(1)}} \cdot V_1$$

**청크 2 처리 후** — 보정 계수 (핵심!):

$$\alpha = e^{m^{(1)} - m^{(2)}}$$

이전 결과를 새로운 max 기준으로 보정:

$$l^{(2)} = l^{(1)} \cdot \alpha + \sum_j e^{S_{2,j} - m^{(2)}}$$

$$O^{(2)} = O^{(1)} \cdot \alpha + e^{S_2 - m^{(2)}} \cdot V_2$$

#### 왜 보정 계수 $$\alpha$$가 필요한가?

max가 바뀌면 이전에 계산한 `exp` 값들이 틀어집니다:

```
청크 1: max=5,  exp(3-5) = exp(-2) = 0.135
청크 2: max=10, exp(3-5)는 틀림! exp(3-10) = exp(-7) = 0.0009여야 함

보정: 0.135 × exp(5-10) = 0.135 × exp(-5) ≈ 0.0009  ✓
                α = exp(m_old - m_new)
```

### 메모리 복잡도

| 방식     | 메모리 | RTX 4080 (16GB)에서 최대 seq_len |
| -------- | ------ | -------------------------------- |
| Standard | O(N²)  | ~8K (float16)                    |
| Flash    | O(N)   | 수십만+                          |

---

## 커널 동작 원리

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/flash_attention_flow.png" class="img-fluid rounded z-depth-1" alt="FlashAttention 타일링 및 연산 흐름" %}

### 단계별 의사코드

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet02_%EB%8B%A8%EA%B3%84%EB%B3%84_%EC%9D%98%EC%82%AC%EC%BD%94%EB%93%9C.py"></script>

---

## Causal Masking

Autoregressive 모델(GPT 등)에서는 미래 토큰을 볼 수 없다:

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/causal_mask.png" class="img-fluid rounded z-depth-1" alt="Causal 마스크 적용 예시" %}

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet03_Causal_Masking.py"></script>

---

## 코드 라인별 설명 (Forward)

### Online Softmax 변수 초기화

- `m_i`: 행별 최대값 추적 (처음엔 -inf → 점점 커짐)
- `l_i`: 행별 softmax 분모 추적 (처음엔 0 → 점점 커짐)
- `acc`: 최종 출력 누적기 (처음엔 0 → P@V 결과가 점점 누적)
- 이 세 변수가 **Online Softmax의 핵심** — 전체 S 행렬 없이 softmax 계산

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet04_Online_Softmax_%EB%B3%80%EC%88%98_%EC%B4%88%EA%B8%B0%ED%99%94.py"></script>

### 내부 루프 — Online Softmax 업데이트 (핵심!)

각 K/V 블록에 대해 다음을 수행한다:

1. **K 블록 로드** → `S = Q @ K^T * scale` 계산 (attention score 타일)
2. **Causal mask 적용** → 미래 토큰 차단 (`-inf`로 마스킹)
3. **Online Softmax 업데이트**:
   - `m_new = max(m_old, max(S))` — 전체 최대값 갱신
   - `alpha = exp(m_old - m_new)` — **이전 결과 보정 계수** (max가 바뀌면 이전 exp 값이 틀어지므로)
   - `l_i = l_i * alpha + sum(exp(S - m_new))` — 분모 업데이트
   - `acc = acc * alpha` — 이전 출력 보정
4. **V 블록 로드** → `acc += P @ V` 누적
5. `p.to(v.dtype)`: FP32 → FP16 변환 (`tl.dot`은 같은 타입 필요)

매 반복마다 `acc`에 결과가 누적되므로 **S 전체를 저장할 필요가 없다.**

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet05_%EB%82%B4%EB%B6%80_%EB%A3%A8%ED%94%84___Online_Softmax_%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8__%ED%95%B5.py"></script>

### 최종 정규화

- `l_i`: 각 행의 softmax 분모 (Σ exp) → 마지막에 한 번만 나눔
- FP32 → FP16 변환 후 저장

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet06_%EC%B5%9C%EC%A2%85_%EC%A0%95%EA%B7%9C%ED%99%94.py"></script>

---

## Backward 구현

Forward만으로는 학습이 불가능하다. Backward에서 $$dQ, dK, dV$$를 계산해야 하는데, Standard attention의 backward는 $$S = QK^T$$와 $$P = \text{softmax}(S)$$를 필요로 한다. 이를 저장하면 $$O(N^2)$$ 메모리가 필요하다.

Flash Attention은 **forward에서 $$S, P$$를 저장하지 않고 logsumexp(LSE)만 저장**한 뒤, backward에서 $$P$$를 재계산한다.

### LSE로 P 재계산 (Recomputation)

Forward에서 저장하는 값:

$$L_i = m_i + \log(\ell_i) \in \mathbb{R}^N$$

Backward에서 $$P$$ 복원:

$$P_{ij} = \exp\!\left(\frac{Q_i K_j^T}{\sqrt{d}} - L_i\right)$$

$$L_i$$는 $$O(N)$$만 차지한다. $$S_{ij}$$를 다시 계산하는 FLOPs는 늘지만, $$O(N^2)$$ HBM 접근이 사라지므로 실제 속도는 더 빠르다.

### Softmax Gradient

Chain rule로 softmax의 gradient를 정리하면:

$$dS_{ij} = P_{ij}\!\left(dP_{ij} - D_i\right) \cdot \text{scale}$$

여기서 $$D_i = \text{rowsum}(dO_i \odot O_i) \in \mathbb{R}^N$$이다. $$D_i$$를 미리 계산해두면 모든 $$j$$ 반복에서 재사용할 수 있다.

### 3단계 커널 구조

Backward는 세 개의 독립 커널로 나뉜다.

```
[1단계] Preprocess 커널
  입력:  O, dO          (각 (bh, N, d))
  출력:  Δ ∈ ℝ^(bh×N)  (Δ_i = rowsum(dO_i ⊙ O_i))

[2단계] dKV 커널  (외부 루프 = K/V 블록 j, 내부 루프 = Q 블록 i)
  입력:  Q, K, V, dO, LSE, Δ
  출력:  dK, dV
  → K_j, V_j를 SRAM에 고정하고 모든 Q_i를 순회

[3단계] dQ 커널   (외부 루프 = Q 블록 i, 내부 루프 = K/V 블록 j)
  입력:  Q, K, V, dO, LSE, Δ
  출력:  dQ
  → Q_i를 SRAM에 고정하고 모든 K_j, V_j를 순회
```

2·3단계를 분리하는 이유: dKV 커널은 j 인덱스로 grid를 잡아 각 thread block이 `dK_j, dV_j`를 독립적으로 누적하고, dQ 커널은 i 인덱스로 grid를 잡아 `dQ_i`를 독립적으로 누적한다. **atomic 연산 없이** 올바른 결과를 얻을 수 있다.

#### dKV 커널 수식 (j 고정, i 순회)

$$S_{ij} = Q_i K_j^T, \quad P_{ij} = \exp(S_{ij} \cdot \text{scale} - L_i)$$

$$dV_j \mathrel{+}= P_{ij}^T \cdot dO_i \quad \text{(matmul 1)}$$

$$dP_{ij} = dO_i \cdot V_j^T \quad \text{(matmul 2)}$$

$$dS_{ij} = P_{ij} \odot (dP_{ij} - \Delta_i) \cdot \text{scale} \quad \text{(softmax gradient)}$$

$$dK_j \mathrel{+}= dS_{ij}^T \cdot Q_i \quad \text{(matmul 3)}$$

#### dQ 커널 수식 (i 고정, j 순회)

$$dQ_i \mathrel{+}= dS_{ij} \cdot K_j \quad \text{(matmul 4)}$$

### torch.autograd.Function 래핑

`torch.autograd.Function`으로 래핑하면 PyTorch `.backward()` API를 그대로 쓸 수 있다.

```python
class FlashAttentionV1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, lse = flash_attention(q, k, v, causal, return_lse=True)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = q.shape[-1] ** -0.5
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _fa1_backward(
            do, q, k, v, o, lse, ctx.causal, ctx.sm_scale
        )
        return dq, dk, dv, None  # causal에 대한 gradient 없음
```

`ctx`에는 $$Q, K, V, O, L$$만 저장된다 — 모두 $$O(Nd)$$ 또는 $$O(N)$$ 크기다. $$S, P \in O(N^2)$$는 저장하지 않는다.

---

## GPU 아키텍처별 최적화

이 구현은 **RTX 4080 (Ada Lovelace)** 기준으로 작성됐다. 다른 GPU에서도 동작하지만 최적 성능을 내려면 아키텍처별 특성에 맞게 파라미터를 조정해야 한다.

### 아키텍처별 특성 비교

| GPU           | 아키텍처     | SM당 SRAM | 권장 BLOCK_M | 권장 BLOCK_N | 주요 하드웨어 기능     |
| ------------- | ------------ | --------- | ------------ | ------------ | ---------------------- |
| RTX 4080/4090 | Ada Lovelace | ~100 KB   | 64           | 64           | 현재 구현 기준         |
| A100          | Ampere       | 192 KB    | 128          | 64           | HBM2e, 더 큰 블록 가능 |
| H100          | Hopper       | 228 KB    | 128          | 64–128       | TMA, wgmma             |
| B200          | Blackwell    | 232 KB+   | 192+         | 128+         | FP4/FP8 matmul         |

### 1. BLOCK_M, BLOCK_N 튜닝

Block 크기는 SRAM 용량에 직접 제약된다. SRAM에 동시에 올려야 하는 데이터 (fp16, head_dim=64 기준):

$$\text{SRAM 사용량} \approx \underbrace{(\text{BLOCK\_M} + 2 \times \text{BLOCK\_N}) \times d \times 2}_{\text{Q, K, V 블록 (fp16)}} + \underbrace{\text{BLOCK\_M} \times d \times 4}_{\text{acc (fp32)}} \text{ bytes}$$

| GPU                | BLOCK_M | BLOCK_N | SRAM 사용량 | SM당 동시 thread block |
| ------------------ | ------- | ------- | ----------- | ---------------------- |
| RTX 4080 (~100 KB) | 64      | 64      | ~40 KB      | 2                      |
| A100 (192 KB)      | 128     | 64      | ~80 KB      | 2                      |
| H100 (228 KB)      | 128     | 128     | ~128 KB     | 1                      |

블록이 클수록 HBM 접근 횟수가 줄어든다. **A100에서 BLOCK_M을 64→128로 바꾸는 것만으로 ~15–20% 향상**이 기대된다.

### 2. `exp` → `exp2` 트릭

현재 구현은 `tl.exp`를 사용한다:

```python
p = tl.exp(s - m_new[:, None])
alpha = tl.exp(m_i - m_new)
```

A100/Ada/Hopper는 `exp2`(밑이 2인 지수)를 하드웨어 명령어로 직접 지원하지만, `exp`는 내부적으로 `exp2(x × log₂e)`로 변환되어 곱셈이 1회 추가된다. `qk_scale`에 `log₂e = 1.4427`을 미리 곱해두면 이 오버헤드를 제거할 수 있다:

```python
LOG2E: tl.constexpr = 1.4426950408889634
qk_scale = scale * LOG2E            # 커널 진입 전 1회만 계산

p = tl.math.exp2(s - m_new[:, None])   # 하드웨어 명령어 직접 사용
alpha = tl.math.exp2(m_i - m_new)
```

단, LSE도 base-2 형태(`m + log₂ℓ`)로 저장해야 backward와 일관성이 유지된다. 이 최적화는 [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)의 구현에 적용되어 있다.

### 3. Autotune으로 자동 최적화

현재 구현은 BLOCK_M=64, BLOCK_N=64로 고정되어 있다. `@triton.autotune`을 사용하면 GPU마다 최적 설정을 자동으로 탐색한다:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_stages=4, num_warps=8),
    ],
    key=["seq_len", "HEAD_DIM", "IS_CAUSAL"],
)
@triton.jit
def flash_attention_kernel(...):
    ...
```

첫 실행 시 모든 config를 프로파일링하고 결과를 캐싱한다. `seq_len`, `HEAD_DIM`, `IS_CAUSAL` 조합마다 별도로 최적화된다. 이 방식이 FA2 Triton 구현의 표준이 됐다.

### 4. H100, B200에서의 추가 기법

H100(Hopper)부터는 Triton만으로 활용하기 어려운 전용 하드웨어 기능이 추가됐다.

**TMA (Tensor Memory Accelerator)**
비동기 메모리 복사 전용 유닛. 행렬 곱 연산과 데이터 로딩을 겹쳐 메모리 레이턴시를 숨긴다. Flash Attention 3의 핵심 최적화 중 하나다.

**wgmma (Warpgroup Matrix Multiply-Accumulate)**
기존 `mma` 대비 더 큰 타일(최대 64×256)을 한 명령에 처리한다. 레지스터 사용량이 줄고 SM 점유율(occupancy)이 높아진다.

**GEMM–softmax 파이프라이닝**
GEMM(matmul)과 softmax(non-matmul)를 비동기로 겹쳐 실행한다. FA2까지는 두 연산이 순차적이었지만, FA3에서 비로소 겹쳐진다. A100에서 non-matmul이 matmul 대비 16배 느린 것을 이 파이프라이닝으로 상당 부분 숨길 수 있다.

| 최적화 기법       | Triton 적용 가능 여부    | 해당 FA 버전 |
| ----------------- | ------------------------ | ------------ |
| BLOCK 크기 튜닝   | ✓ (`@autotune`)          | FA1, FA2     |
| exp → exp2        | ✓ (`tl.math.exp2`)       | FA2          |
| Pipeline prefetch | △ (`num_stages` 로 근사) | FA2          |
| TMA (비동기 복사) | △ (Triton 3.x 실험적)    | FA3          |
| wgmma             | ✗ (CUDA 전용)            | FA3          |
| FP8 matmul        | ✗ (CUDA 전용)            | FA4          |

Triton은 이식성이 높지만 하드웨어 특화 최적화에는 한계가 있다. 프로덕션 수준의 Flash Attention은 CUDA로 작성된 FA3/FA4가 더 적합하다. **Triton 구현은 알고리즘 이해와 프로토타이핑에 최적**이다.

---

## 전체 튜토리얼과의 연결

| 개념                | 어디서 배웠나 | Flash Attention에서의 역할      |
| ------------------- | ------------- | ------------------------------- |
| `tl.load`, mask     | 01 Vector Add | Q, K, V 블록 로드               |
| reduction, `tl.exp` | 02 Softmax    | Online Softmax의 max, sum, exp  |
| stride, 다중 포인터 | 03 RMSNorm    | batch, head, seq, dim 차원 접근 |
| `tl.dot`, 2D 타일링 | 04 MatMul     | S = Q@K^T, O += P@V             |
| K 차원 루프         | 04 MatMul     | K/V 블록 순회 (내부 루프)       |
| **Online Softmax**  | **신규**      | SRAM 제한 극복의 핵심           |
| **Backward 커널**   | **신규**      | LSE 재사용으로 O(N) backward    |

---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/benchmark.png" class="img-fluid rounded z-depth-1" alt="FlashAttention 성능 벤치마크 결과" %}

- **정확도**: PyTorch standard attention과 거의 동일한 결과
- **속도**: 시퀀스 길이가 길수록 (1024+) 큰 속도 향상
- **메모리**: O(N²) → O(N)으로 극적인 메모리 절약

> 이 벤치마크는 RTX 4080 기준이다. A100에서는 BLOCK_M=128 + exp2 트릭 + autotune을 적용하면 추가 향상을 기대할 수 있다.

### A100 80GB 측정값

A100-SXM4-80GB · `num_heads=16, head_dim=64, fp16` · 4 GPU 평균 (표준편차 < 1%) · 11회 측정 중 첫 회 폐기.

**Non-causal forward** (ms):

| seq   | FA1 (ms) | PyTorch (ms) | FA1/PT |
| ----- | -------- | ------------ | ------ |
| 4096  | 0.755    | 2.967        | 3.93×  |
| 8192  | 2.591    | 12.758       | 4.92×  |
| 16384 | 9.570    | 35.787       | 3.74×  |
| 32768 | 37.610   | 146.601      | 3.90×  |

**Causal forward** (ms):

| seq   | FA1 (ms) | PyTorch (ms) | FA1/PT |
| ----- | -------- | ------------ | ------ |
| 4096  | 0.571    | 5.243        | 9.18×  |
| 8192  | 1.721    | 21.807       | 12.67× |
| 16384 | 5.972    | 70.856       | 11.86× |
| 32768 | 22.247   | OOM          | —      |

- 시퀀스가 길어질수록 PyTorch의 S 행렬(32K · 32K · 2B = 32GB)이 OOM에 진입한다
- FA1 단독으로도 long-seq에서 4–13× 가속을 확보한다
- 다만 BLOCK_M=64 고정·exp2 미적용으로 FA2 대비 ~1.5× 손해를 본다 (다음 포스트의 [A100 측정값](/blog/2026/triton-06-flash-attention-v2/#a100-80gb-측정값--fa1-vs-fa2-vs-pytorch) 참고)

### RTX 4080 vs A100 — 동일 코드, 다른 GPU

`_experiments/05_flash_attention/main()` 을 두 GPU 에서 그대로 실행한 결과 (non-causal, num_heads=16, head_dim=64, fp16):

| Seq  | 4080 Triton (ms) | A100 Triton (ms) | 4080 vs PT | A100 vs PT | A100/4080         |
| ---- | ---------------- | ---------------- | ---------- | ---------- | ----------------- |
| 256  | 0.038            | 0.087            | 0.69×      | 1.26×      | 0.44× _4080 우세_ |
| 512  | 0.046            | 0.083            | 1.05×      | 1.37×      | 0.55× _4080 우세_ |
| 1024 | 0.094            | 0.112            | 1.97×      | 2.36×      | 0.84×             |
| 2048 | 0.246            | 0.234            | **6.74×**  | **3.81×**  | 1.05×             |
| 4096 | 0.876            | 0.746            | **6.15×**  | **4.58×**  | 1.17×             |

- **짧은 seq (≤1024) 에서는 RTX 4080 이 더 빠르다** — Ada Lovelace SM 클럭 2505 MHz vs A100 1410 MHz, kernel launch overhead 가 작은 영향
- **긴 seq (≥2048) 부터 A100 우세** — HBM2e 1.5 TB/s vs GDDR6X 717 GB/s 메모리 대역폭 차이가 작동
- **PyTorch 대비 가속비는 4080 이 더 큼** — 4080 의 cuBLAS 베이스라인이 상대적으로 느려서 가속비가 부풀어 보인다. 절대 시간은 long-seq 에서 A100 이 더 빠름

원본 데이터는 [`_experiments/05_flash_attention/results_a100.md`](https://github.com/wonbeomjang/wonbeomjang.github.io/blob/master/_experiments/05_flash_attention/results_a100.md), [`results_4080.md`](https://github.com/wonbeomjang/wonbeomjang.github.io/blob/master/_experiments/05_flash_attention/results_4080.md) 에 있다.

### 왜 이론 peak 대비 일정 % 에서 멈추나

각 GPU 의 FP16 Tensor Core 이론 peak: **A100 312 TFLOP/s, RTX 4080 195 TFLOP/s**. Attention forward 의 matmul FLOPs 는 $$4 \cdot BH \cdot N^2 \cdot d$$ (S = QK^T 와 O = PV 두 번의 matmul). 측정 시간으로 나누면 실제 throughput.

**A100** (위의 4-GPU 평균 측정값 기준, long-seq):

| seq   | FA1 시간 | matmul FLOPs | 측정 TFLOPS | A100 peak 비율 |
| ----- | -------- | ------------ | ----------- | -------------- |
| 4096  | 0.755 ms | 68.7 G       | 91.0        | 29.2%          |
| 8192  | 2.591 ms | 274.9 G      | 106.1       | 34.0%          |
| 16384 | 9.570 ms | 1099.5 G     | 114.9       | 36.8%          |
| 32768 | 37.61 ms | 4398.0 G     | 116.9       | **37.5%**      |

**RTX 4080** (`_experiments/` main() 측정값):

| seq  | FA1 시간 | matmul FLOPs | 측정 TFLOPS | 4080 peak 비율 |
| ---- | -------- | ------------ | ----------- | -------------- |
| 1024 | 0.094 ms | 4.3 G        | 45.7        | 23.4%          |
| 2048 | 0.246 ms | 17.2 G       | 69.9        | 35.8%          |
| 4096 | 0.876 ms | 68.7 G       | 78.4        | **40.2%**      |

**관찰**:

- A100 점근선 ~37%, 4080 점근선 ~40% — **4080 이 % 기준으로는 더 잘 saturate** 한다 (작은 GPU 라 saturate 가 쉬움)
- 절대 throughput 은 A100 이 1.5–1.6× 빠름 (peak 이 1.6× 크기 때문)
- FlashAttention 공식 CUDA 구현은 같은 A100 에서 ~70% 를 찍는다 — Triton 구현은 그 절반 수준

**원인 분해 (두 GPU 공통)**:

1. **매 블록마다 `acc /= l_i`** (Tc회 division)
   - N=4096, BLOCK_N=64 면 Tc = 64. 매 K/V 블록 처리 후 BLOCK_M 개 fp32 division 이 들어가 matmul throughput 을 끊는다
   - GPU 의 fp32 division 은 SFU(special function unit) 에서 처리되어 Tensor Core dispatch 와 충돌
   - **추정 손실 10–15%**. FA2 의 가장 큰 개선 항목이 이것
2. **`tl.exp` 의 숨은 곱셈** — 컴파일러가 `exp2(x · log₂e)` 로 변환, BLOCK_M × BLOCK_N 개 추가 fp32 multiply 누적. **추정 손실 3–5%** (FA2 의 pre-multiply 로 회수)
3. **causal mask 매 iteration 분기** — `if IS_CAUSAL` 가 inner loop 안에 있어 mask 계산이 매 iteration 발생. **추정 손실 5–10%** (causal 한해, FA2 STAGE 1/2 분리로 회수)
4. **fp16 ↔ fp32 round-trip 캐스팅** — register pressure 증가로 occupancy 감소. **추정 손실 2–3%**
5. **Triton vs CUTLASS PTX 격차** — Triton 은 LLVM → PTX 를 거쳐 register allocation·instruction scheduling 에서 **~5–10% 손해**

**4080 특수 제약**:

- **SRAM 100 KB** (A100 164 KB) — BLOCK_M=128 이상 config 가 컴파일 안 되거나 occupancy 손해. autotune 이 작은 BLOCK_M 으로 떨어져 long-seq tail 효율이 낮아짐
- **GDDR6X 717 GB/s** — Long-seq 에서 HBM bound 됐을 때 A100 의 1.5 TB/s 보다 일찍 saturate
- **Cache size 작음** — L2 64 MB (A100) vs 4080 64 MB 으로 비슷하지만, K/V reuse 패턴이 SM 수 (4080 76 vs A100 108) 에 의존하므로 차이 발생

| 손실 원인        | 추정 % | FA2 에서 회수? |
| ---------------- | ------ | -------------- |
| 매 블록 division | 10–15% | ✓ (un-scaled)  |
| exp 숨은 곱셈    | 3–5%   | ✓ (exp2 trick) |
| causal 분기      | 5–10%  | ✓ (STAGE 분리) |
| fp16↔fp32 캐스트 | 2–3%   | 동일           |
| Triton PTX gap   | 5–10%  | 동일           |
| 4080 SRAM 한계   | 5–8%   | 일부           |
| **A100 합계**    | 25–43% | -              |
| **4080 합계**    | 30–51% | -              |

FA1 에서 잃은 이론 peak 의 30–50% 중 절반 이상은 FA2 에서 회수한다 (다음 포스트 참고). 나머지 10–15% 는 "Triton 추상화의 천장" — 더 짜내려면 CUTLASS 또는 Hopper 신명령어가 필요하다.

---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=flash_attention.py"></script>

---

> FlashAttention의 원리가 궁금하다면 [FlashAttention 논문 리뷰](/blog/2023/fastattention/)를, 개선점이 궁금하다면 [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)를, Hopper GPU 최적화가 궁금하다면 [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)를, Blackwell GPU 최적화가 궁금하다면 [FlashAttention-4 논문 리뷰](/blog/2026/flashattention-4/)를, FA2 알고리즘을 Triton으로 구현하고 싶다면 [Triton 06: Flash Attention 2](/blog/2026/triton-06-flash-attention-v2/)를 참고하자.

# 참고 문헌

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://triton-lang.org/main/index.html)
- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
