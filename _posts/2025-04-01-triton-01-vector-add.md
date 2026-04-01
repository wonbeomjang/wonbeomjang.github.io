---
layout: post
title: "Triton 01: Vector Addition — Triton 커널 기초"
date: 2025-04-01 01:00:00 +0900
description: 가장 간단한 GPU 커널인 벡터 덧셈을 Triton으로 구현하며 핵심 개념을 학습합니다.
categories: [triton, gpu]
tags: [triton, gpu, cuda, deep-learning]
giscus_comments: true
related_posts: true
---

## 개요

가장 간단한 GPU 커널인 벡터 덧셈을 구현합니다.
이 튜토리얼에서 Triton의 핵심 개념을 모두 배울 수 있습니다.


---

## 핵심 개념

### GPU 병렬 프로그래밍

CPU는 순차적으로 빠르게 처리하고, GPU는 수천 개의 코어로 동시에 처리합니다.

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/cpu_vs_gpu_parallel.png" class="img-fluid rounded z-depth-1" %}

### CUDA vs Triton

| 구분 | CUDA | Triton |
|------|------|--------|
| 언어 | C/C++ | Python |
| 메모리 관리 | 수동 (shared memory 직접 관리) | 자동 (컴파일러가 처리) |
| 스레드 관리 | warp/thread 단위 | block(프로그램) 단위 |
| 난이도 | 높음 | 낮음 |
| 성능 | 최고 | CUDA의 90%+ 달성 가능 |

### Triton 핵심 용어

- **커널(Kernel)**: GPU에서 실행되는 함수
- **프로그램(Program)**: 커널의 하나의 인스턴스 (CUDA의 thread block에 해당)
- **그리드(Grid)**: 프로그램 인스턴스의 총 개수
- **BLOCK_SIZE**: 각 프로그램이 처리하는 데이터 크기


---

## 커널 동작 원리

길이 N인 벡터를 `BLOCK_SIZE` 크기의 청크로 나누고,
각 프로그램이 하나의 청크를 담당합니다.

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/vector_chunking.png" class="img-fluid rounded z-depth-1" %}

### 단계별 분석

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 현재 프로그램의 ID 얻기
    pid = tl.program_id(axis=0)

    # 2. 이 프로그램이 처리할 데이터의 오프셋 계산
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 경계 처리 (벡터 끝부분에서 범위 초과 방지)
    mask = offsets < n_elements

    # 4. Global Memory에서 데이터 로드
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. 연산 수행
    output = x + y

    # 6. 결과를 Global Memory에 저장
    tl.store(output_ptr + offsets, output, mask=mask)
```


---

## 사용된 Triton 기능

| 기능 | 설명 |
|------|------|
| `@triton.jit` | 함수를 Triton 커널로 컴파일 |
| `tl.program_id(axis)` | 현재 프로그램의 ID (어떤 청크를 처리할지 결정) |
| `tl.arange(start, end)` | 연속 정수 벡터 생성 (numpy의 arange와 유사) |
| `tl.load(ptr, mask)` | Global Memory에서 데이터 읽기 |
| `tl.store(ptr, value, mask)` | Global Memory에 데이터 쓰기 |
| `tl.constexpr` | 컴파일 타임 상수 (BLOCK_SIZE처럼 컴파일 시 결정되는 값) |


---

## 마스크(Mask)란?

벡터 길이가 BLOCK_SIZE의 배수가 아닐 때 경계 처리가 필요합니다.

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/mask_explanation.png" class="img-fluid rounded z-depth-1" %}


---

## 그리드 설정

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/grid_launch.png" class="img-fluid rounded z-depth-1" %}


---

## 래퍼 함수

```python
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```


---

## 포인터(Pointer)란?

C/CUDA 경험이 없으면 포인터가 낯설 수 있습니다:

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/pointer_explanation.png" class="img-fluid rounded z-depth-1" %}


---

## 벤치마크 결과

{% include figure.liquid loading="lazy" path="assets/img/triton/01_vector_add/benchmark.png" class="img-fluid rounded z-depth-1" %}

Vector Add는 **메모리 대역폭 바운드(memory-bound)** 연산입니다.
연산량이 적고 데이터 이동이 대부분이라, Triton과 PyTorch의 성능 차이가 크지 않습니다.
하지만 이 패턴은 이후 모든 커널의 기초가 됩니다.
