---
layout: post
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date: 2023-03-29 00:00:00 +0900
description: optimize transformer on gpu device
categories: [optimization]
tags: [paper, attention, efficient-transformer]
giscus_comments: true
related_posts: true
---

# Introduction

현재 NLP와 Vision 분야에서 transformer는 활발히 사용되고 있다.  
하지만 transformer는 메모리를 많이 잡아먹는 모듈이었고 이를 해결하기 위해 sparse-approximation, low-rank approximation 등을 제안했다.  
하지만 이들은 이론과 달리 computational speed를 증가시켜주지 못하는 경우가 많았다.  
저자는 GPU에서 빠른 SRAM으로 연산을 수행할 수 있는 IO-aware 알고리즘을 제시했다.

## Hardware performance

### GPU Memory Hierarchy

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-gpu-hierchy.png" width="50%">
</p>

GPU는 CPU와 마찬가지로 메모리 계층을 가진다. DRAM이 가장 느리고 용량이 크며, SRAM이 가장 빠르고 용량이 작다.  
GPU는 병렬 연산 시 데이터를 HBM에서 가져온 후 SRAM에 올려놓고 연산을 한다. 이후 다른 데이터를 읽어들이면 SRAM에 있는 정보는 다시 HBM에 저장된다.

### Performance characteristics

퍼포먼스를 고려할 때 연산량과 메모리 접근의 관점으로 두 가지를 나눌 수 있다.

1. Compute-bound: 연산량이 메모리 접근보다 많은 경우이다. ex) MatMul
2. Memory-bound: 메모리 접근이 연산량보다 많은 경우이다. ex) softmax, batchnorm

### Kernel fusion

Memory-bound 연산을 가속하는 데 많이 사용되는 방법은 kernel fusion이다.  
만약 같은 input에 대해 여러 연산을 한다고 하면, 컴파일러는 자동으로 많은 elementwise operation을 fusion한다.

## Standard Attention Implementation

Sequence length $$N$$과 head dimension $$d$$에 대하여 attention은 input sequence $$Q,K,V \in \mathbb{R}^{N \times d}$$를 이용하여  
$$O \in \mathbb{R}^{M \times d}$$를 구한다. 그에 대한 식은 다음과 같다.

$$
S=QK^\top \in \mathbb{R}^{N \times N}, P=softmax(S) \in \mathbb{R}^{N \times N}, O = PV \in \mathbb{R}^{N \times d}
$$

이때 softmax는 row-wise operation이다. 보통의 attention은 $$O(N^2)$$의 memory cost를 사용하는데, 대다수의 경우에는 $$N \gg d$$를 만족한다(GPT-2, N=1024 and d=64).

<p align="center">
    <img src="/assets/post/image/legacy/standard-attention-algorithm.png" width="80%">
</p>

# FlashAttention

FlashAttention은 **Tiling**과 **Recomputation**을 사용하여 Attention을 가속화한다.

### Tiling

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-tiling.png" width="50%">
</p>

기존의 softmax 연산은 다음과 같은 과정을 거친다.

$$
m(x):=\underset{i}{max}(x_i), f(x):=[e^{x_1-m(x)} ... e^{x_B-m(x)}],
$$

$$
l(x):=\sum_i f(x)_i, softmax(x):= \frac{f(x)}{l(x)}
$$

vector $$x^{(1)}, x^{(2)} \in \mathbb{R}^B$$일 때 vector의 concatenation $$x=[x^{(1)} x^{(2)}]$$에 대해 softmax는 다음과 같이 decomposition할 수 있다.

$$
m(x)=m([x^{(1)} x^{(2)}])=max(m(x^{(1)}),m(x^{(2)})),
$$

$$
f(x):=[e^{m(x^{(1)})-m(x)}f(x^{(1)}) ... e^{m(x^{(2)})-m(x)}f(x^{(2)})],
$$

$$
l(x)=l([x^{(1)} x^{(2)}])=e^{m(x^{(1)})-m(x)}l(x^{(1)}) + e^{m(x^{(2)})-m(x)}l(x^{(2)}),
$$

$$
softmax(x):= \frac{f(x)}{l(x)}
$$

즉, softmax를 block 단위로 쪼개서 계산할 수 있다는 것이다.

### Recomputation

저자는 backward 때 $$O(N^2)$$의 memory를 저장하지 않기 위해 softmax normalization statistics $$(m,l)$$을 저장한 후 backward 때 다시 구성한다.  
이로 인해 FLOPs는 증가하지만 HBM에서 데이터를 읽는 횟수가 줄어들어 속도가 향상된다.

### Kernel Fusion

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-kernel-fusion.png" width="50%">
</p>

Tiling을 통해 한 번의 HBM load에서 matrix multiply, softmax, optionally masking and dropout, matrix multiply를 한 후 HBM에 저장할 수 있게 되었다.  
이는 반복적인 IO operation을 줄여준다.

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-algorithm.png" width="80%">
</p>

> **Theorem 1**. Algorithm 1 returns $$O=softmax(QK^\top)V$$ with $$O(N^2d)$$ FLOPs and requires additional memory beyond inputs and output.

## Analysis: IO Complexity of FlashAttention

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-coomplexity.png" width="80%">
</p>

FlashAttention은 standard보다 GFLOPs는 많지만, HBM read and write가 적어 runtime이 개선되었다.

> **Theorem 2.** Let $$N$$ be the sequence length, $$d$$ be the head dimension, and $$M$$ be the size of SRAM with $$d \leq M \leq Nd$$. Standard attention (Algorithm 0) requires $$\Theta(Nd+N^2)$$ HBM accesses, while FlashAttention (Algorithm 1) requires $$\Theta(N^2d^2M^{-1})$$ HBM accesses.

> **Proposition 3.** Let $$N$$ be the sequence length, $$d$$ be the head dimension, and $$M$$ be the size of SRAM with $$d \leq M \leq Nd$$. There does not exist an algorithm to compute exact attention with $$\Theta(N^2d^2M^{-1})$$ HBM accesses for all $$M$$ in the range $$[d, Nd]$$.

# Extension

Block-sparse attention을 응용하여 block-sparse flashattention을 만들기도 했다.

$$
S=QK^\top \in \mathbb{R}^{N \times N}, P=softmax(S \odot \mathbb{1}_{\tilde{M}}) \in \mathbb{R}^{N \times N}, O=PV \in \mathbb{R}^{N \times d}
$$

> **Proposition 4.** Let $$N$$ be the sequence length, $$d$$ be the head dimension, and $$M$$ be the size of SRAM with $$d \leq M \leq Nd$$. Block-sparse FlashAttention (Algorithm 5) requires $$\Theta(N^2d^2M^{-1})$$ HBM accesses where 𝑠 is the fraction of nonzero blocks in the block-sparsity mask.

## Experiment

FlashAttention은 tiling을 통해 속도가 빠르고, recomputation을 통해 메모리가 줄어들었다.  
이를 이용하여 sequence length를 늘릴 수 있었고, 이는 추가적인 성능 향상을 가져왔다.

### BERT

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-bert-performance.png" width="80%">
</p>

BERT 학습 시 MLPerf 1.1 기준 학습 시간이 15% 개선되었다.

### GPT-2

GPT-2는 Huggingface, Megatron-LM과 비교했는데 각각 3배, 1.7배의 speed up이 발생했다.

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-gpt-2-performace.png" width="80%">
</p>

### Long-range Arena

LRA에서도 기존 대비 2.4배의 speed up을 보였으며, 다른 attention method보다 성능도 좋았다.

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-long-reange-arena-performace.png" width="80%">
</p>

## Better Models with Longer Sequences

### Language Modeling with Long Context.

Recomputing으로 메모리 사용량이 줄어들면서 더 긴 input sequence를 다룰 수 있게 되었다. 이를 통해 추가적인 성능 향상을 가져왔다.

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-bert-with-long-sequence.png" width="80%">
</p>

### Long Document Classification

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-long-document-classification.png" width="80%">
</p>

### Path-X and Path-256

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-path-x.png" width="80%">
</p>

Path-X와 Path-256은 long context로 기존의 모델들은 random한 결과와 비슷하게 나왔다.  
FlashAttention은 해당 데이터셋에 random 이상의 결과를 가져온 첫 번째 모델이다.

## Benchmarking Attention

<p align="center">
    <img src="/assets/post/image/legacy/fastattention-banchmarking.png" width="80%">
</p>

Standard Attention은 메모리 사용량이 $$O(N^2)$$이고, FlashAttention은 recomputation을 통해 $$O(N)$$으로 줄였다. Approximate attention(sparse attention) 역시 $$O(N)$$이다.  
다만 sequence length가 매우 길어지면 approximate attention이 속도 면에서 유리할 수 있지만, exact attention인 FlashAttention이 정확도에서는 더 우수하다.

# Limitation

FlashAttention은 CUDA kernel을 사용해야 하므로 엔지니어링이 필요하다.  
그리고 GPU마다 컴파일이 필요하며 확장성에 문제가 있다.  
또한 현재는 single GPU를 기준으로 만들어졌으므로, multi-GPU를 위한 알고리즘도 제작해야 한다.

> FlashAttention-2의 개선점이 궁금하다면 [FlashAttention-2 논문 리뷰](/blog/2023/flashattention-2/)를, Hopper GPU에서의 최적화가 궁금하다면 [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)를, Triton으로 직접 구현해보고 싶다면 [Triton 05: Flash Attention — 종합 프로젝트](/blog/2026/triton-05-flash-attention/)를 참고하자.

---

**추가글...**  
아앗... 포스트를 올리고 4개월 만에 URL이 틀렸다는 것을 깨달았다... 하지만 어쩔 수 없다... 그냥 간다...  
fastattentiondl 아니라 flashattention으로 해야 하는데....
