---
layout: post
title: "Triton 02: Fused Softmax — 커널 퓨전과 Reduction"
date: 2025-04-01 02:00:00 +0900
description: Softmax를 하나의 커널로 퓨전하여 메모리 접근을 최소화하는 방법을 학습합니다.
categories: [triton, gpu]
tags: [triton, gpu, softmax, kernel-fusion]
giscus_comments: true
related_posts: true
---

## 개요

Softmax를 하나의 커널로 퓨전(fusion)하여 메모리 접근을 최소화합니다.
커널 퓨전이 왜 중요한지, reduction 연산을 어떻게 처리하는지 학습합니다.


---

## 핵심 개념

### Softmax 수식

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

`max(x)`를 빼는 이유: `exp`는 큰 값에서 오버플로우가 발생합니다.
최대값을 빼면 모든 지수가 0 이하가 되어 안정적으로 계산됩니다.

### 왜 커널 퓨전인가?

{% include figure.liquid loading="lazy" path="assets/img/triton/02_fused_softmax/kernel_fusion.png" class="img-fluid rounded z-depth-1" %}

### Reduction 연산

전체 데이터에서 하나의 값을 계산하는 연산:
- `max`: 최대값
- `sum`: 합계
- `mean`: 평균

Triton에서는 `tl.max(x, axis=0)`, `tl.sum(x, axis=0)` 으로 간단하게 수행합니다.


---

## 커널 동작 원리

입력 행렬의 각 **행(row)** 을 하나의 프로그램이 처리합니다.

{% include figure.liquid loading="lazy" path="assets/img/triton/02_fused_softmax/row_processing.png" class="img-fluid rounded z-depth-1" %}


---

## 코드 라인별 설명

### 커널 함수

```python
@triton.jit
def fused_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 한 행 전체를 SRAM에 로드
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Fused Softmax 계산 (전부 SRAM에서)
    row_max = tl.max(row, axis=0)       # 1. 최대값
    row = row - row_max                 # 2. 수치 안정성
    numerator = tl.exp(row)             # 3. 지수 함수
    denominator = tl.sum(numerator, axis=0)  # 4. 합계
    softmax_output = numerator / denominator  # 5. 나누기

    # 결과를 1번만 Global Memory에 저장
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

핵심: **max → exp → sum → 나누기**를 전부 SRAM 안에서 처리.
PyTorch는 이 4단계를 각각 별도 커널로 실행하므로 매번 Global Memory를 왕복합니다.

### 래퍼 함수

```python
def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    fused_softmax_kernel[grid](
        x, output, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```


---

## 01 Vector Add와의 차이점

| | 01 Vector Add | 02 Fused Softmax |
|---|---|---|
| 처리 단위 | 1D 벡터의 청크 | 2D 행렬의 행 |
| 프로그램당 연산 | 덧셈 1번 | max+exp+sum+나누기 |
| 퓨전 효과 | 없음 (연산이 1개) | 4개 연산을 1커널로 |
| 새로운 기능 | - | `tl.max`, `tl.sum`, `tl.exp`, stride |


---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/02_fused_softmax/benchmark.png" class="img-fluid rounded z-depth-1" %}

커널 퓨전 덕분에 메모리 대역폭을 절약하여,
특히 열(column) 수가 클수록 PyTorch 대비 성능 향상이 눈에 띕니다.
