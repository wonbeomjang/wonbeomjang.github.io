---
layout: post
title: "Triton 05: Flash Attention — 종합 프로젝트"
date: 2026-04-02 00:00:00 +0900
description: 지금까지 배운 모든 기법을 종합하여 Flash Attention을 Triton으로 구현합니다.
categories: [triton]
tags: [triton, gpu, flash-attention, llm, attention]
giscus_comments: true
related_posts: true
featured: true
---

## 개요

지금까지 배운 모든 기법을 종합하여 Flash Attention을 구현합니다.
LLM 추론/학습에서 가장 중요한 최적화 기법 중 하나입니다.

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

타일 단위로 Q, K, V를 처리하면서 결과를 점진적으로 누적합니다.
이를 위해 **Online Softmax** 알고리즘이 필요합니다.

### Online Softmax

데이터를 청크(블록) 단위로 받으면서 **점진적으로 업데이트**합니다.

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

Autoregressive 모델(GPT 등)에서는 미래 토큰을 볼 수 없습니다:

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/causal_mask.png" class="img-fluid rounded z-depth-1" alt="Causal 마스크 적용 예시" %}

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet03_Causal_Masking.py"></script>

---

## 코드 라인별 설명

### Online Softmax 변수 초기화

- `m_i`: 행별 최대값 추적 (처음엔 -inf → 점점 커짐)
- `l_i`: 행별 softmax 분모 추적 (처음엔 0 → 점점 커짐)
- `acc`: 최종 출력 누적기 (처음엔 0 → P@V 결과가 점점 누적)
- 이 세 변수가 **Online Softmax의 핵심** — 전체 S 행렬 없이 softmax 계산

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet04_Online_Softmax_%EB%B3%80%EC%88%98_%EC%B4%88%EA%B8%B0%ED%99%94.py"></script>

### 내부 루프 — Online Softmax 업데이트 (핵심!)

각 K/V 블록에 대해 다음을 수행합니다:

1. **K 블록 로드** → `S = Q @ K^T * scale` 계산 (attention score 타일)
2. **Causal mask 적용** → 미래 토큰 차단 (`-inf`로 마스킹)
3. **Online Softmax 업데이트**:
   - `m_new = max(m_old, max(S))` — 전체 최대값 갱신
   - `alpha = exp(m_old - m_new)` — **이전 결과 보정 계수** (max가 바뀌면 이전 exp 값이 틀어지므로)
   - `l_i = l_i * alpha + sum(exp(S - m_new))` — 분모 업데이트
   - `acc = acc * alpha` — 이전 출력 보정
4. **V 블록 로드** → `acc += P @ V` 누적
5. `p.to(v.dtype)`: FP32 → FP16 변환 (`tl.dot`은 같은 타입 필요)

매 반복마다 `acc`에 결과가 누적되므로 **S 전체를 저장할 필요가 없습니다.**

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet05_%EB%82%B4%EB%B6%80_%EB%A3%A8%ED%94%84___Online_Softmax_%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8__%ED%95%B5.py"></script>

### 최종 정규화

- `l_i`: 각 행의 softmax 분모 (Σ exp) → 마지막에 한 번만 나눔
- FP32 → FP16 변환 후 저장

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=05_flash_attention_snippet06_%EC%B5%9C%EC%A2%85_%EC%A0%95%EA%B7%9C%ED%99%94.py"></script>

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

---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/benchmark.png" class="img-fluid rounded z-depth-1" alt="FlashAttention 성능 벤치마크 결과" %}

- **정확도**: PyTorch standard attention과 거의 동일한 결과
- **속도**: 시퀀스 길이가 길수록 (1024+) 큰 속도 향상
- **메모리**: O(N²) → O(N)으로 극적인 메모리 절약

---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=flash_attention.py"></script>
