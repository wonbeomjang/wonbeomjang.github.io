---
layout: post
title: "Triton 03: RMSNorm — LLM에서 쓰이는 실전 커널"
date: 2026-04-02 03:00:00 +0900
description: LLaMA, Mistral, Gemma 등 최신 LLM에서 사용하는 RMSNorm을 Triton으로 구현합니다.
categories: [triton, gpu]
tags: [triton, gpu, rmsnorm, llm]
giscus_comments: true
related_posts: true
---

## 개요

LLaMA, Mistral, Gemma 등 최신 LLM에서 사용하는 RMSNorm을 Triton으로 구현합니다.
Softmax와 유사한 패턴이지만, 학습 가능한 가중치(gamma)가 추가됩니다.


---

## 핵심 개념

### LayerNorm vs RMSNorm

```
LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
RMSNorm:    y = x / sqrt(mean(x²) + ε) * γ
```

RMSNorm이 LLM에서 선호되는 이유:
- mean 계산이 필요 없음 → 연산량 감소
- bias(β) 없음 → 파라미터 수 감소
- 실험적으로 LayerNorm과 성능이 비슷

### 수식 분해

```
1. 제곱합:    sum_sq = Σ(x_i²)
2. RMS:       rms = sqrt(sum_sq / n + ε)
3. 정규화:    x_norm = x / rms
4. 스케일링:  y = x_norm * γ
```


---

## 커널 동작 원리

{% include figure.liquid loading="lazy" path="assets/img/triton/03_rmsnorm/rmsnorm_flow.png" class="img-fluid rounded z-depth-1" %}


---

## 코드 라인별 설명

### PyTorch 참조 구현

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=03_rmsnorm_snippet01_PyTorch_%EC%B0%B8%EC%A1%B0_%EA%B5%AC%ED%98%84.py"></script>

### 커널 함수

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=03_rmsnorm_snippet02_%EC%BB%A4%EB%84%90_%ED%95%A8%EC%88%98.py"></script>

### 래퍼 함수

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=03_rmsnorm_snippet03_%EB%9E%98%ED%8D%BC_%ED%95%A8%EC%88%98.py"></script>


---

## 02 Fused Softmax와의 차이점

| | 02 Softmax | 03 RMSNorm |
|---|---|---|
| reduction | `max` + `sum` (2번) | `sum` (1번) |
| 수치 안정성 | max 빼기 | eps 더하기 |
| 범위 밖 채움 | `-inf` | `0.0` |
| 추가 입력 | 없음 | 가중치 γ |
| 입력 shape | 2D만 | 3D/4D → 2D 변환 |


---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/03_rmsnorm/benchmark.png" class="img-fluid rounded z-depth-1" %}

PyTorch의 수동 RMSNorm 구현 대비 커널 퓨전으로 인한 성능 향상이 나타납니다.
hidden_size가 클수록(2048, 4096 등) 차이가 명확합니다.


---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=rmsnorm.py"></script>
