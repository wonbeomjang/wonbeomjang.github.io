---
layout: post
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
date: 2023-08-07 00:00:00 +0900
description:
categories: [attention, hardware-optimization, paper]
tags: [attention, hardware-optimization, paper]
giscus_comments: true
related_posts: true
---

# Introduction

현재 GPT부터 시작해서 ViT 등 여러 분야에서 attention layer를 사용하고 있다. 하지만 attention layer는 dimension의 제곱에 비례하여 cost가 들어 모델의 병목인 부분이기도 하다. 이에 따라 attention layer를 효율적으로 만드는 시도가 많이 있는데, 그중 하나가 FlashAttention이다. FlashAttention은 tiling과 kernel fusion으로 기존 attention layer 대비 2.4배 속도가 향상되었다. 하지만 FlashAttention 또한 기존 GPU의 이론적 성능에 25~40% 정도의 속도밖에 내지 못한다.

저자는 FlashAttention을 분석하던 중 thread block 간 work를 partitioning할 때 비효율성을 발견했고, 이로 인해 GPU에서 low-occupancy와 불필요한 memory IO가 일어나는 것을 깨달았다. 따라서 저자는 이를 해결하기 위해 3가지를 제안했다.

1. Output을 바꾸지 않고 non-matmul operation의 FLOPS를 줄인다.
2. Single head attention일지라도 병렬처리를 하도록 연산 순서를 변경한다.
3. Thread block 내에 warps 간 통신을 줄인다.

저자는 위 3가지를 통해 기존 FlashAttention 대비 2배 빠른 속도를 달성하고 GPU의 이론적 성능의 50~73%까지 성능을 끌어올렸다.

# Background

하드웨어 최적화에 관한 논문은 익숙하지 않으니 background까지 꼼꼼하게 읽어보자.

## Hardware characteristics

### GPU performance characteristics

GPU는 compute element와 memory hierarchy를 가지고 있다. Nvidia tensor core와 같은 최신 GPU compute element는 FP16/BF16과 같은 low-precision에서 matmul operation을 최적화하고 있다. 반면에 non-matmul operation은 최적화가 되어있지 않아 matmul operation보다 최대 16배가 느리다.

Memory hierarchy에 관해서는 기본적으로 GPU는 high bandwidth memory (HBM)과 on-chip SRAM (shared memory)를 가지고 있다. A100 기준 40~80GB의 HBM은 1.5~2.0TB/s의 bandwidth를 가지고 있고, 108개의 stream multiprocessor는 각각 192KB의 on-chip SRAM을 가지고 있으며 이는 19TB/s의 bandwidth를 가지고 있다. L2 cache도 있으나, 이것은 사용자가 컨트롤할 수 없으므로 논의에서 제외하도록 하자.

### Execution Model

GPU는 수많은 thread로 구성되어 있으며, thread가 모여서 thread block을 구성한다. 이 thread block은 stream multiprocessor (SMs)를 통해 실행된다. Thread block 내에서 thread는 warp이라는 단위로 묶이게 되는데, 이 warp들은 공유 메모리를 통해 communication을 한다.

## Standard Attention Implementation

기존 attention은 query, key, value들 간의 연산으로 구성된다. Sequence length를 N, head dimension을 d라고 하자. Input sequence $$Q, K, V \in \mathbb{R}^{N\times d}$$에 대해 attention output $$O \in \mathbb{R}^{N \times d}$$를 계산하기 위해 아래의 식을 이용한다.

$$S=QK^{\intercal}\in \mathbb{R}^{N\times N}$$

$$P=\text{softmax}(S)\in\mathbb{R}^{N\times N}$$

$$O=PV\in \mathbb{R}^{N\times d}$$
이때 softmax는 row-wise로 적용하게 된다.
Backward pass는 다음과 같은 과정을 거친다.

$$dV=P^{\intercal}dO\in\mathbb{R}^{N\times d}$$

$$dP=dOV^{\intercal}\in\mathbb{R}^{N\times N}$$

$$dS=\text{dsoftmax}(dP)\in\mathbb{R}^{N\times N}$$

$$dQ=dSK\in\mathbb{R}^{N\times d}$$

$$dK=QdS^\intercal\in\mathbb{R}^{N\times d}$$

더 자세한 것은 FlashAttention 설명을 참고하면 된다.

## FlashAttention

자세한 것은 FlashAttention 설명을 참고하기 바란다. [FlashAttention 1 포스트](https://www.wonbeomjang.kr/blog/2023/fastattention/)

### Forward pass

간단하게 이야기하자면, K, V를 tiling하여 병렬적으로 계산 후 on-line softmax를 통해 병렬적으로 softmax를 적용한다. 이후에 tiling한 Q를 불러와 on-chip 연산으로 만든다. 또한 이를 통해 연산을 fusion할 수 있으며 Q, K, V는 HBM에서 load한 이후 모든 연산을 수행한 후 HBM에 저장하게 된다. 연산은 다음과 같고, 아래서 표시한 $$S$$는 $$S=QK^T$$이다.

$$m^{(1)}=\text{rowmax}(S^{(1)})\in\mathbb{R}^{B_r}$$

$$l^{(1)}=\text{rowsum}(e^{S^{(1)}-m^{(1)}})\in\mathbb{R}^{B_r\times B_c}$$

$$\tilde{P}^{(1)}=\text{diag}(l^{(1)})^{-1}e^{S^{(1)}-m^{(1)}}\in\mathbb{R}^{B_r\times B_c}$$

$$O^{(1)}=\tilde{P}^{(1)}V^{(1)}=\text{diag}(l^{(1)})^{-1}e^{S^{(1)}-m^{(1)}}V^{(1)}\in\mathbb{R}^{B_r\times d}$$

$$m^{(2)}=\text{max}(m^{(1)},\text{rowmax}(S^{(2)}))=m$$

$$l^{(2)}=e^{m^{(1)}-m^{(2)}}l^{(1)}+\text{rowsum}(e^{S^{(2)}-m})=\text{rowsum}(e^{S^{(1)}-m})+\text{rowsum}(e^{S^{(2)}-m})=l$$

$$\tilde{P}^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m^{(2)}}$$

$$O^{(2)}=\text{diag}(l^{(1)}/l^{(2)})^{-1}O^{(1)}+\tilde{P}^{(2)}V^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(1)}-m}V^{(1)}+\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m}V^{(2)}=O$$

즉, figure 1처럼 vector를 쪼개고, 합치는 과정을 통해 memory IO를 줄여 연산 속도를 빠르게 만들었다.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig1.png" width="80%">
</p>

### Backward Pass

Backward pass는 attention 연산을 하는 과정에서 $$m, l$$이 계산되는데 이를 이용하면 다시 연산을 recompute할 수 있다.

# 3. FlashAttention-2

FlashAttention은 기본적으로 non-matmul FLOPs를 줄인다. 예를 들어 Nvidia의 A100 GPU는 FP16/BF16의 matmul 연산은 이론적으로 312 TFLOPs/s의 연산량을 가지지만, non-matmul 연산은 19.5 TFLOPs/s의 연산량을 가진다. 즉 non-matmul 연산이 matmul 연산보다 16배 느려 non-matmul 연산이 전체 연산의 일부를 차지하더라도 이를 최적화시켜야 한다.

## Forward pass

저자는 FlashAttention에서 on-line softmax를 먼저 주목했다.

### Recaling

기존에는 $$\text{diag}(l^{(2)})^{-1}$$를 두 항 모두 rescaling했다.

$$O^{(2)}=\text{diag}(l^{(1)}/l^{(2)})^{-1}O^{(1)}+\tilde{P}^{(2)}V^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(1)}-m}V^{(1)}+\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m}V^{(2)}=O$$

이렇게 한다면 두 텀을 각각 읽어 각각 나눠야 하기 때문에 memory IO가 많아진다. 따라서 마지막 결과 $$\tilde{O}^{(last)}$$를 계산 후에 한꺼번에 $$\text{diag}(l^{(last)})^{-1}$$으로 rescaling 한다.

$$\tilde{O}^{(2)}=\text{diag}(l^{(1)})^{-1}O^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)}$$

$$O^{(2)}=\tilde{O}^{(2)}\text{diag}(l^{(2)})^{-1}$$

### Memorization

Backward에 사용하기 위해서 $$m, l$$을 저장한 후 재구성한다고 했다. 각각을 저장하는 대신 $$L^{(j)}=m^{(j)}+\text{log}(l^{(j)})$$를 저장해도 똑같이 backward를 재구성할 수 있어 $$m, l$$ 대신 $$L$$을 저장하게 된다.

### Result

결론적으로 FlashAttention 2에서는 다음과 같은 방법으로 attention을 구현하게 된다.

$$m^{(1)}=\text{rowmax}(S^{(1)})\in\mathbb{R}^{B_r}$$

$$l^{(1)}=\text{rowsum}(e^{S^{(1)}-m^{(1)}})\in\mathbb{R}^{B_r\times B_c}$$

$$\tilde{O}^{(1)}=e^{S^{(1)}-m^{(1)}}V^{(1)}\in\mathbb{R}^{B_r\times d}$$

$$m^{(2)}=\text{max}(m^{(1)},\text{rowmax}(S^{(2)}))=m$$

$$l^{(2)}=e^{m^{(1)}-m^{(2)}}l^{(1)}+\text{rowsum}(e^{S^{(2)}-m})=\text{rowsum}(e^{S^{(1)}-m})+\text{rowsum}(e^{S^{(2)}-m})=l$$

$$\tilde{P}^{(2)}=\text{diag}(l^{(2)})^{-1}e^{S^{(2)}-m^{(2)}}$$

$$\tilde{O}^{(2)}=\text{diag}(e^{m^{(1)}-m^{(2)}})^{-1}\tilde{O}^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)}=e^{S^{(1)}-m}V^{(1)}+e^{S^{(2)}-m}V^{(2)}$$

$$O^{(2)}=\text{diag}(l^{(2)})^{-1}\tilde{O}^{(2)}=O$$

기존 FlashAttention과 다르게 term 자체가 줄어들었다. Forward pass에 관한 알고리즘을 정리하자면 다음과 같다.

<p align="center">
    <img src="/assets/post/image/flashattention2/alg1.png" width="100%">
</p>

## Backward

<p align="center">
    <img src="/assets/post/image/flashattention2/alg2.png" width="100%">
</p>

Backward 자체는 $$L$$을 사용한다는 것 말고는 별다른 이야기는 없다.

## Parallelism

기본적으로 GPU는 병렬처리가 가능하다. 각각의 GPU thread block마다 1개의 attention module이 들어간다. 따라서 보통 # batch size x # self-attention head로 thread block을 구성하게 되고 이를 stream multiprocessor가 나눠 가진다. 그래서 만약 sequence 길이가 길어 small batch size나 small number of self-attention head를 가지게 된다면 병렬처리를 잘 활용하지 못한다. 따라서 저자는 sequence length dimension에 따른 병렬처리를 하게 된다.

**Forward pass**
저자는 sequence length dimension으로 병렬처리를 한다. 하지만 이는 한 sequence 내에서는 독립적으로 처리되어야 함으로 다른 sequence와 통신을 하지 못하도록 구성했다. 물론 이전과 마찬가지로 batch, multi-head 간 병렬처리는 유지한다.

**Backward pass**
Algorithm 2에 의하면 column block 간에 병렬처리만 한다. 위의 경우와 같이 sequence length dimension으로도 병렬처리가 가능하여 추가하게 된다.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig2.png" width="80%">
</p>

결과적으로 worker마다 병렬처리가 잘 되게 된다.

## Work Partitioning Between Warp

### Forward

<p align="center">
    <img src="/assets/post/image/flashattention2/fig3.png" width="100%">
</p>

기존의 FlashAttention은 $$K$$와 $$V$$를 각각의 warp에 K개로 partitioning 했고, $$Q$$는 모든 warp이 접근 가능하도록 했다. 그리고 이를 "split-K"라고 한다. 하지만 이러한 방법은 partition된 $$QK^T$$를 partition된 $$V$$에 곱하게 된다. 따라서 중간 계산 결과를 저장하고, 읽고, 동기화를 많이 해야 해서 IO에서 속도가 느려진다. 따라서 $$Q$$를 partition하고, $$K, Q$$를 공유하게 해 이런 IO를 줄여 속도를 높이게 된다.

## Backward

"split-K"를 지양한다는 것 밖에 이해를 못했다.

### Tuning block sizes

Block size를 늘리면 memory IO의 수가 줄어든다. 하지만 block 수가 많아지면서 registers의 수가 늘어나고, total shared memory 크기가 커져 비효율성이 늘어난다. 많은 registers는 프로그램 속도를 느리게 만들고, total shared memory의 크기가 너무 커지면 GPU memory가 부족하다. 따라서 GPU마다 적절한 block size를 조정한다.

# Empirical Validation

이제 속도를 보자.

<p align="center">
    <img src="/assets/post/image/flashattention2/fig4.png" width="100%">
</p>

<p align="center">
    <img src="/assets/post/image/flashattention2/fig5.png" width="100%">
</p>

<p align="center">
    <img src="/assets/post/image/flashattention2/fig6.png" width="100%">
</p>

<p align="center">
    <img src="/assets/post/image/flashattention2/fig7.png" width="100%">
</p>

FlashAttention-2는 기존 FlashAttention, xFormer 대비 2배의 속도를 보여줬고, Triton으로 구현된 FlashAttention보다 1.3~1.5배 빨라진 속도를 보여줬다. 놀라운 것은 PyTorch에서 naive하게 구현한 것 대비 10배의 속도 차이를 보여준다. 이로 인해 기존의 large model에서도 더 빠른 연산 속도를 보여준다.

<p align="center">
    <img src="/assets/post/image/flashattention2/table1.png" width="100%">
</p>
