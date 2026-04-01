---
layout: post
title: "Triton 04: Matrix Multiplication — 2D 타일링과 Autotune"
date: 2026-04-02 04:00:00 +0900
description: 딥러닝의 핵심 연산인 행렬 곱셈을 Triton으로 구현하며 2D 타일링, tl.dot, autotune을 학습합니다.
categories: [triton, gpu]
tags: [triton, gpu, matmul, tensor-core]
giscus_comments: true
related_posts: true
---

## 개요

딥러닝의 핵심 연산인 행렬 곱셈(GEMM)을 Triton으로 구현합니다.
2D 타일링, `tl.dot`, `triton.autotune` 등 고급 기능을 학습합니다.


---

## 핵심 개념

### 행렬 곱셈이 왜 중요한가

딥러닝의 거의 모든 연산이 행렬 곱셈:
- Linear layer: `y = xW + b`
- Attention: `QK^T`, `PV`
- MLP: 모든 Feed-Forward 블록

### 나이브 vs 타일링

**나이브**: 출력의 각 원소마다 Global Memory에서 행/열 전체를 읽음 → 같은 데이터를 반복 로드

**타일링**: 행렬을 작은 블록으로 나누어 SRAM에 올리고, 블록 단위로 계산

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/tiling.png" class="img-fluid rounded z-depth-1" %}


---

## 커널 동작 원리

### 2D 그리드

이전 튜토리얼은 1D 그리드(행 단위)였지만, MatMul은 2D 그리드를 사용합니다:

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/2d_grid.png" class="img-fluid rounded z-depth-1" %}

### K 차원 루프

행렬 곱셈 `C = A × B`에서 A(M×K), B(K×N)일 때, K가 크면 한 번에 SRAM에 못 올립니다.
그래서 K를 `BLOCK_SIZE_K`씩 잘라서 반복하며, 부분 결과를 누적합니다.

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=04_matmul_snippet01_K_%EC%B0%A8%EC%9B%90_%EB%A3%A8%ED%94%84.py"></script>

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/k_loop_pointer.png" class="img-fluid rounded z-depth-1" %}

### L2 캐시 최적화 (Swizzling)

**Swizzling = "같은 B 블록을 쓰는 프로그램들을 묶어서 실행"**

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/group_concept.png" class="img-fluid rounded z-depth-1" %}

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/swizzling_detail.png" class="img-fluid rounded z-depth-1" %}

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/swizzling.png" class="img-fluid rounded z-depth-1" %}

### `triton.autotune` 이란?

블록 크기에 따라 성능이 크게 달라집니다. Autotune은 여러 설정을 실행해보고 가장 빠른 것을 선택합니다:

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=04_matmul_snippet02_triton_autotune__%EC%9D%B4%EB%9E%80.py"></script>


---

## 코드 라인별 설명

### K 차원 루프 (핵심)

<script src="https://gist.github.com/wonbeomjang/42cd2b629a46d83e348bc15c5aa83a17.js?file=04_matmul_snippet03_K_%EC%B0%A8%EC%9B%90_%EB%A3%A8%ED%94%84__%ED%95%B5%EC%8B%AC.py"></script>

### 이전 튜토리얼과의 차이점

| | 01~03 | 04 MatMul |
|---|---|---|
| 그리드 | 1D (행 수) | 1D (M타일 × N타일) |
| 데이터 | 1D 벡터/행 | 2D 블록 (타일) |
| 루프 | 없음 | K 차원 루프 |
| 핵심 연산 | `+`, `exp`, `sum` | `tl.dot` (텐서 코어) |
| 파라미터 튜닝 | 수동 BLOCK_SIZE | `triton.autotune` |


---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/04_matmul/benchmark.png" class="img-fluid rounded z-depth-1" %}

cuBLAS(`torch.matmul`)는 수십 년간 최적화된 라이브러리입니다.
Triton으로 cuBLAS의 **80~90%** 성능에 도달하는 것이 목표입니다.


---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=matmul.py"></script>
