---
layout: post
title: "Triton 05: Flash Attention — 종합 프로젝트"
date: 2025-04-01 05:00:00 +0900
description: 지금까지 배운 모든 기법을 종합하여 Flash Attention을 Triton으로 구현합니다.
categories: [triton, gpu]
tags: [triton, gpu, flash-attention, llm, attention]
giscus_comments: true
related_posts: true
---

## 개요

지금까지 배운 모든 기법을 종합하여 Flash Attention을 구현합니다.
LLM 추론/학습에서 가장 중요한 최적화 기법 중 하나입니다.


---

## 핵심 개념

### Attention 수식

$$O = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V$$

- $$Q, K, V$$: Query, Key, Value 행렬 (각각 $$N \times d$$)
- $$\sqrt{d}$$: head dimension의 제곱근으로 나눠서 스케일링
- $$\text{softmax}$$: 행(row) 단위로 적용 → 확률 분포로 변환

### Standard Attention의 문제

```python
S = Q @ K.T / sqrt(d)   # (N, N) — O(N²) 메모리!
P = softmax(S)           # (N, N)
O = P @ V               # (N, d)
```

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

| 방식 | 메모리 | RTX 4080 (16GB)에서 최대 seq_len |
|------|--------|----------------------------------|
| Standard | O(N²) | ~8K (float16) |
| Flash | O(N) | 수십만+ |


---

## 커널 동작 원리

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/flash_attention_flow.png" class="img-fluid rounded z-depth-1" %}

### 단계별 의사코드

```python
for q_block in Q_blocks:          # 각 프로그램
    m = -inf, l = 0, O = 0

    for k_block, v_block in zip(K_blocks, V_blocks):  # 내부 루프
        S = q_block @ k_block.T * scale

        # Online softmax 업데이트
        m_new = max(m, rowmax(S))
        correction = exp(m - m_new)
        P = exp(S - m_new)

        l = l * correction + rowsum(P)
        O = O * correction + P @ v_block
        m = m_new

    O = O / l  # 최종 정규화
```


---

## Causal Masking

Autoregressive 모델(GPT 등)에서는 미래 토큰을 볼 수 없습니다:

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/causal_mask.png" class="img-fluid rounded z-depth-1" %}

```python
if IS_CAUSAL:
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    s = tl.where(causal_mask, s, float("-inf"))
```


---

## 코드 라인별 설명

### Online Softmax 변수 초기화

```python
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)           # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # 출력 누적기
```

### 내부 루프 — Online Softmax 업데이트 (핵심!)

```python
    for start_n in range(0, k_range, BLOCK_N):
        # K 블록 로드 후 S = Q @ K^T * scale
        k = tl.load(k_base + ...)
        s = tl.dot(q, tl.trans(k)) * scale

        # Causal mask 적용
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online Softmax 업데이트
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)            # 보정 계수
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)  # 분모 업데이트
        acc = acc * alpha[:, None]              # 이전 출력 보정

        # P @ V 누적
        v = tl.load(v_base + ...)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new
```

### 최종 정규화

```python
    acc = acc / l_i[:, None]
    tl.store(o_base + ..., acc.to(tl.float16), mask=o_mask)
```


---

## 전체 튜토리얼과의 연결

| 개념 | 어디서 배웠나 | Flash Attention에서의 역할 |
|---|---|---|
| `tl.load`, mask | 01 Vector Add | Q, K, V 블록 로드 |
| reduction, `tl.exp` | 02 Softmax | Online Softmax의 max, sum, exp |
| stride, 다중 포인터 | 03 RMSNorm | batch, head, seq, dim 차원 접근 |
| `tl.dot`, 2D 타일링 | 04 MatMul | S = Q@K^T, O += P@V |
| K 차원 루프 | 04 MatMul | K/V 블록 순회 (내부 루프) |
| **Online Softmax** | **신규** | SRAM 제한 극복의 핵심 |


---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/05_flash_attention/benchmark.png" class="img-fluid rounded z-depth-1" %}

- **정확도**: PyTorch standard attention과 거의 동일한 결과
- **속도**: 시퀀스 길이가 길수록 (1024+) 큰 속도 향상
- **메모리**: O(N²) → O(N)으로 극적인 메모리 절약
