---
layout: post
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
date: 2023-08-07 00:00:00 +0900
description: FlashAttention-2 논문 리뷰와 성능 분석
categories: [attention, hardware-optimization, paper]
tags: [attention, hardware-optimization, paper]
giscus_comments: true
related_posts: true
featured: true
---

# Introduction

GPT부터 시작해서 ViT 등 여러 분야에서 attention layer를 많이 쓰고 있다. 그런데 이 attention layer는 dimension의 제곱에 비례해서 계산 비용이 커서 모델의 병목이 될 수 있다. 그래서 attention layer를 효율적으로 만드는 여러 시도가 있는데, 그 중 하나가 FlashAttention이다. FlashAttention은 tiling과 kernel fusion을 사용해서 기존 attention layer보다 2.4배 더 빠르게 동작한다. 하지만 FlashAttention도 GPU의 이론적 성능에 비해 25~40%밖에 성능을 내지 못한다고 한다.

이런 문제를 해결하기 위해 저자는 FlashAttention을 분석하면서 thread block 간 work partitioning이 비효율적이라는 점을 발견했다. 이로 인해 GPU에서 low-occupancy와 불필요한 memory IO가 발생한다고 느꼈다. 그래서 저자는 이를 개선하기 위해 세 가지 방법을 제안했다.

1. Output을 바꾸지 않고 non-matmul operation의 FLOPS를 줄인다.
2. Single head attention이라도 병렬 처리를 하도록 연산 순서를 바꾼다.
3. Thread block 내에서 warps 간 통신을 줄인다.

저자는 이 세 가지 방법을 통해 기존 FlashAttention보다 2배 빠른 성능을 달성하고, GPU 이론적 성능의 50~73%까지 성능을 끌어올렸다.

# Background

하드웨어 최적화 관련 논문은 익숙하지 않아서 배경 부분을 꼼꼼하게 읽어보자.

## Hardware characteristics

### GPU performance characteristics

GPU는 compute element와 memory hierarchy를 가지고 있다. 예를 들어, Nvidia의 tensor core는 FP16/BF16 같은 저정밀도 연산을 matmul에 최적화하고 있다. 하지만 non-matmul 연산은 최적화가 부족해서 matmul보다 최대 16배 느리다.

메모리 계층 구조에 대해 보면, GPU는 기본적으로 high bandwidth memory(HBM)와 on-chip SRAM(공유 메모리)을 갖고 있다. A100 기준으로 40~80GB의 HBM은 1.5~2.0TB/s의 대역폭을 가지며, 108개의 stream multiprocessor는 각각 192KB의 on-chip SRAM을 갖고 있어 19TB/s의 대역폭을 제공한다. L2 캐시도 있지만, 이는 사용자가 컨트롤할 수 없어서 논의에서는 제외한다.

### Execution Model

GPU는 수많은 thread로 구성되며, 이 thread들이 모여 thread block을 구성한다. 각 thread block은 stream multiprocessor(SM)에서 실행된다. thread block 내에서 thread는 warp이라는 단위로 묶이며, 이 warp들은 공유 메모리를 통해 서로 통신한다.

## Standard Attention Implementation

기존의 attention은 query, key, value들 간의 연산으로 구성된다. 시퀀스 길이를 N, head dimension을 d라고 하자. Input sequence $$Q, K, V \in \mathbb{R}^{N\times d}$$에 대해 attention output $$O \in \mathbb{R}^{N \times d}$$를 계산하는 방식은 아래와 같다.

$$S=QK^{\intercal}\in \mathbb{R}^{N\times N}$$

$$P=\text{softmax}(S)\in\mathbb{R}^{N\times N}$$

$$O=PV\in \mathbb{R}^{N\times d}$$

여기서 softmax는 row-wise로 적용된다.
Backward pass는 아래 과정을 거친다.

$$dV=P^{\intercal}dO\in\mathbb{R}^{N\times d}$$

$$dP=dOV^{\intercal}\in\mathbb{R}^{N\times N}$$

$$dS=\text{dsoftmax}(dP)\in\mathbb{R}^{N\times N}$$

$$dQ=dSK\in\mathbb{R}^{N\times d}$$

$$dK=QdS^\intercal\in\mathbb{R}^{N\times d}$$

FlashAttention에 대해 더 자세한 내용은 다른 포스트에서 확인할 수 있다.

## FlashAttention

FlashAttention의 구체적인 내용은 이전에 다뤘던 [FlashAttention 1 포스트](https://www.wonbeomjang.kr/blog/2023/fastattention/)에서 참고할 수 있다.

### Forward pass

FlashAttention은 K와 V를 tiling하여 병렬적으로 계산한 뒤, on-line softmax를 통해 병렬적으로 softmax를 적용한다. 그 후 tiling한 Q를 불러와 on-chip 연산을 한다. 이를 통해 연산을 fusion하고, Q, K, V는 HBM에서 불러와 연산을 마친 후 다시 HBM에 저장한다. 연산 과정은 아래와 같다. 여기서 $$S$$는 $$S=QK^T$$이다.

$$m^{(1)}=\text{rowmax}(S^{(1)})\in\mathbb{R}^{B_r}$$

$$l^{(1)}=\text{rowsum}(e^{S^{(1)}-m^{(1)}})\in\mathbb{R}^{B_r\times B_c}$$

$$\tilde{P}^{(1)}=\text{diag}(l^{(1)})^{-1}e^{S^{(1)}-m^{(1)}}\in\mathbb{R}^{B_r\times B_c}$$

$$O^{(1)}=\tilde{P}^{(1)}V^{(1)}=\text{diag}(l^{(1)})^{-1}e^{S^{(1)}-m^{(1)}}V^{(1)}\in\mathbb{R}^{B_r\times d}$$

$$m^{(2)}=\text{max}(m^{(1)},\text{rowmax}(S^{(2)}))=m$$

$$l^{(2)}=e^{m^{(1)}-m^{(2)}}l^{(1)}+\text{rowsum}(e^{S^{(2)}-m})=\text{rowsum}(e^{S^{(1)}-m})+\text{rowsum}(e^{S^{(2)}-m})=l$$

$$\tilde{P}^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m^{(2)}}$$

$$O^{(2)}=\text{diag}(l^{(1)}/l^{(2)})^{-1}O^{(1)}+\tilde{P}^{(2)}V^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(1)}-m}V^{(1)}+\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m}V^{(2)}=O$$

이 과정에서 vector를 쪼개고 합치는 방식으로 memory IO를 줄여서 속도를 높였다.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig1.png" width="80%">
</p>

### Backward Pass

Backward pass에서는 attention 연산 중에 계산된 $$m$$과 $$l$$을 사용해서 다시 연산을 재구성할 수 있다.

# FlashAttention-2

FlashAttention-2는 기존 FlashAttention보다 non-matmul FLOPs를 줄인다. 예를 들어, Nvidia의 A100 GPU는 FP16/BF16 matmul 연산에서 이론적으로 312 TFLOPs/s의 성능을 보이지만, non-matmul 연산은 19.5 TFLOPs/s로 훨씬 느리다. 그래서 non-matmul 연산이 전체 연산에서 차지하는 비중이 크면, 이를 최적화하는 것이 중요하다.

## Forward pass

FlashAttention에서는 on-line softmax를 먼저 주목하고, 이를 개선할 수 있는 방법을 제시했다.

### Recalling

기존에는 $$\text{diag}(l^{(2)})^{-1}$$를 두 번 rescaling했으나, FlashAttention-2에서는 마지막 결과 $$\tilde{O}^{(last)}$$를 계산하고, 한 번에 $$\text{diag}(l^{(last)})^{-1}$$으로 rescaling을 한다.

$$\tilde{O}^{(2)}=\text{diag}(l^{(1)})^{-1}O^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)}$$

$$O^{(2)}=\tilde{O}^{(2)}\text{diag}(l^{(2)})^{-1}$$

### Memorization

Backward pass를 위해 $$m$$과 $$l$$을 저장할 필요 없이 $$L^{(j)}=m^{(j)}+\text{log}(l^{(j)})$$을 저장해도 같은 결과를 얻을 수 있다. 그래서 $$m$$과 $$l$$ 대신 $$L$$을 저장한다.

### Result

결과적으로 FlashAttention-2는 다음과 같은 방법으로 attention을 구현한다.

$$m^{(1)}=\text{rowmax}(S^{(1)})\in\mathbb{R}^{B_r}$$

$$l^{(1)}=\text{rowsum}(e^{S^{(1)}-m^{(1)}})\in\mathbb{R}^{B_r\times B_c}$$

$$\tilde{O}^{(1)}=e^{S^{(1)}-m^{(1)}}V^{(1)}\in\mathbb{R}^{B_r\times d}$$

$$m^{(2)}=\text{max}(m^{(1)},\text{rowmax}(S^{(2)}))=m$$

$$l^{(2)}=e^{m^{(1)}-m^{(2)}}l^{(1)}+\text{rowsum}(e^{S^{(2)}-m})=\text{rowsum}(e^{S^{(1)}-m})+\text{rowsum}(e^{S^{(2)}-m})=l$$

$$\tilde{P}^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m^{(2)}}$$

$$\tilde{O}^{(2)}=\text{diag}(e^{m^{(1)}-m^{(2)}})^{-1}\tilde{O}^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)}=e^{S^{(1)}-m}V^{(1)}+e^{S^{(2)}-m}V^{(2)}$$

$$O^{(2)}=\text{diag}(l^{(2)})^{-1}\tilde{O}^{(2)}=O$$

FlashAttention-2에서는 기존과 달리 term 자체가 줄어들었다. Forward pass 알고리즘을 정리하면 다음과 같다.

<p align="center">
    <img src="/assets/post/image/flashattention2/alg1.png" width="100%">
</p>

## Backward

<p align="center">
    <img src="/assets/post/image/flashattention2/alg2.png" width="100%">
</p>

Backward는 $$L$$을 사용한다는 것 말고는 다른 차이점은 없다.

## Parallelism

기본적으로 GPU는 병렬 처리가 가능하다. 각 GPU thread block마다 1개의 attention module이 들어가고, 보통 batch size와 self-attention head 수에 맞춰 thread block을 구성한다. 만약 sequence 길이가 길어지거나, batch size가 작거나, self-attention head가 적으면 병렬 처리가 잘 이루어지지 않는다. 그래서 저자는 sequence length dimension에 따라 병렬 처리를 하도록 했다.

**Forward pass**
저자는 sequence length dimension으로 병렬처리를 하도록 했으며, 이는 한 sequence 내에서 독립적으로 처리되게 구성했다. 물론 기존처럼 batch와 multi-head 간 병렬처리는 계속 유지된다.

**Backward pass**
Algorithm 2에 따르면, backward pass는 column block 간 병렬처리가 이루어진다. 또한 sequence length dimension을 병렬 처리할 수 있도록 추가적인 방법을 사용한다.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig2.png" width="80%">
</p>

결과적으로 worker마다 병렬 처리가 잘 되도록 구성되었다.

## Work Partitioning Between Warp

### Forward

<p align="center">
    <img src="/assets/post/image/flashattention2/fig3.png" width="100%">
</p>

기존 FlashAttention에서는 $$K$$와 $$V$$를 각각의 warp에 나눠서 partition하고, $$Q$$는 모든 warp이 접근할 수 있도록 했다. 이를 "split-K"라고 부른다. 하지만 이 방식은 $$QK^T$$와 $$V$$의 연산이 partition된 후 중간 계산 결과를 저장하고, 읽고, 동기화를 자주 해야 해서 IO에서 병목이 생긴다. 그래서 $$Q$$를 partition하고, $$K$$와 $$Q$$는 공유하게 해서 IO를 줄여 속도를 높였다.

## Backward

"split-K"를 지양한다는 것 밖에 이해를 못했다.

### Tuning block sizes

Block size를 늘리면 memory IO가 줄어든다. 하지만 block 수가 많아지면 registers 수가 늘어나고, total shared memory 크기가 커져 비효율적이 될 수 있다. 너무 많은 registers는 속도를 저하시킬 수 있고, shared memory가 너무 커지면 GPU 메모리가 부족해질 수 있다. 그래서 GPU마다 적절한 block size를 조정해야 한다.

# Empirical Validation

결과적으로 FlashAttention-2는 기존 FlashAttention, xFormer보다 2배 이상의 성능 향상을 보였고, 특히 A100 GPU에서 2.7배까지 성능 향상이 일어났다. 전체적으로 FlashAttention-2는 GPU의 이론적인 성능에 가까운 성능을 보여주었다.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig4.png" width="80%">
</p>

## Conclusion

FlashAttention-2는 기존 FlashAttention을 개선한 방법으로, 다양한 최적화를 통해 성능을 2배 이상 향상시켰다. GPU의 이론적 성능에 근접한 성능을 내면서, low-occupancy와 memory IO를 줄였다. 이 논문은 attention을 최적화하고 성능을 향상시키기 위한 여러 기술적 접근을 다뤘다.
