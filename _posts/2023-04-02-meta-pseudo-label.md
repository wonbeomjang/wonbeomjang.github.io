---
layout: post
title: "Meta Pseudo Labels"
date: 2023-04-02 00:00:00 +0900
description: Meta Pseudo Labels 논문 리뷰 — ImageNet SOTA 반지도 학습 기법
categories: [paper]
tags: [paper, semi-supervised, pseudo-label]
giscus_comments: true
related_posts: true
---

> [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)

# Introduction

<p align="center">
    <img src="/assets/post/image/legacy/mpl-psudo-label.png" width="50%">
</p>

Semi-supervised learning 방법은 여러가지 있는데 그 중에 한 가지는 pseudo labeling 방법이다.
Pseudo labeling은 잘 학습된 teacher network와 student network가 존재하는데
teacher model은 unlabeled data의 pseudo label을 제작하고 student는 그 label을 학습하는 것으로 진행된다.

하지만 이 방법의 문제점은 teacher model의 pseudo label이 정확하지 않다면 student model은 teacher model의 확증편향으로 인하여 잘못된 방향으로 학습을 진행한다.
따라서 저자는 이러한 문제를 해결하기 위해 teacher model이 labeled data에 대한 student model의 성능을 확인하면서 bias를 수정하는 과정을 제안한다.

# Meta Pseudo Label

들어가기에 앞서 notation에 대하여 설명하겠다.

- $$T(x_u,\theta_T)$$: Soft prediction of **teacher** model on unlabeld data
- $$S(x_u,\theta_S)$$: Soft prediction of **student** model on **unlabeld** data
- $$S(x_l,\theta_S)$$: Soft prediction of student model on **labeled** data
- $$CE(q,p)$$: cross-entropy

<p align="center">
    <img src="/assets/post/image/legacy/mpl.png" width="50%">
</p>

Meta pseudo label은 다음과 같은 과정을 거친다.

1. Teacher model이 unlabeld data의 pseudo label을 생성한다.
2. Student model이 이를 학습한다.
3. Student이 labeled data를 이용하여 성능을 평가한다.
4. Teacher model은 3번을 이용하여 더 좋은 pseudo label을 만들도록 학습한다.

더 이해하기쉽게 식으로 살펴보자.

Student의 optimal parameter는 teacher model이 제작한 pseudo-label과 student model의 예측값의 cross entropy가 낮은 parameter이다.

$$
\theta_S^{PL}=\underset{\theta_S}{\operatorname{argmin}} \mathbb{E}[CE(T(x_u;\theta_T), S(x_u;\theta_S)]


$$

un-labeled dataloss는 다음과 같이 정의할 수 있다.

$$
\mathbb{E}[CE(T(x_u;\theta_T), S(x_u;\theta_S)] := \mathcal{L}_u(\theta_T, \theta_S)
$$

그리고 teacher 모델이 사용하는 labeled data loss는 다음과 같이 정의할 수 있다.

$$
\mathbb{E}_{x_l,y_l}[CE(y_l,S(x_l;\theta_S^{PL}))] := \mathcal{L}_l(\theta_S^{PL})
$$

이 때 student model은 teacher model의 pseudo-label을 이용하여 학습하기 때문에 성능은 teacher model에 의존적인 것을 알 수 있다.
이러한 이유로 meta pseudo label이라고 명명하였고 이를 나타내기 위해서 Notation을 다음과 같이 작성한다.

$$
\theta_S^{PL} \rightarrow \theta_S^{PL}(\theta_T)
$$

또한 label data loss는 다음과 같이 나타낼 수 있다.

$$
\mathcal{L}_l(\theta_S^{PL}) \rightarrow \mathcal{L}_l(\theta_S^{PL}(\theta_T))
$$

techer model은 확증편향을 수정하기 위해 $$\mathcal{L}_l(\theta_S^{PL}(\theta_T))$$을 이용하여 parameter를 update한다.
하지만 이를 직접 계산하는 것은 힘들기 때문에 approximation을 한다.

$$
\theta_S^{PL}(\theta_T) \approx \theta_S - \eta_S \cdot \triangledown_{\theta_S}\mathcal{L}_u(\theta_{T}, \theta_{S})
$$

따라서 teacher model은 다음을 최소화하는 것을 목표로한다.

$$
\underset{\theta_T}{\operatorname{min}} \mathcal{L}_l (\theta_S - \eta_S \cdot \triangledown_{\theta_S}\mathcal{L}_u(\theta_{T}, \theta_{S}))
$$

## 정리

Student model은 SGD로 pseudo-label에 대하여 optimize한다.

$$
\theta^{\prime}_S = \theta_S - \eta_S \triangledown_{\theta_S} \mathcal{L}(\theta_T, \theta_S)
$$

Teacher model은 student의 gradient update를 재사용하여 SGD로 optimize한다.

$$
\theta^{\prime}_T=\theta_T - \eta_T \triangledown_{\theta_T} \mathcal{L}_l (\theta_S - \eta_S \cdot \triangledown_{\theta_S}\mathcal{L}_u(\theta_{T}, \theta_{S}))
$$

## Auxiliary Loss

이외에도 loss를 추가하면 성능향상에 기여한다. 따라서 teacher model은 두 가지 loss를 추가했다.

1. Supervised: train on label
2. Semi-supervised: UDA objective on unlabeld data

Student는 오직 Meta Pseudo label로만 학습을 진행하였다. 이후에는 task에 맞도록 finetuning을 하였다.

# Experiment

## Two moon dataset

<p align="center">
    <img src="/assets/post/image/legacy/mpl-two-moon.png" width="50%">
</p>

각각의 class마다 unlabled data 1000개씩, label data 3개씩 추출하였다.
Supervised 방법은 label data를 잘 분류하지만 다른 데이터는 오류를 보이고 있다.
Supervised 방법으로 학습 된 teacher model을 통하여 pseudo label을 만들었을때는 label data 조차 정확하게 판별을 못했다.
하지만 Meta Pseudo Label에서는 정확하게 두 class를 분리하였다.

## Small Model

EfficientNet과 같은 large model을 실험하기 전에 small model로 실험을 진행하였다.
세 가지 dataset을 사용하였는데 CIFAR-10-4K, SVHN은 WideResNet28-2를 사용하였고 imagenet은 resnet50을 사용하였다.

<p align="center">
    <img src="/assets/post/image/legacy/small-model.png" width="50%">
</p>

다른 SOTA method들 보다 더 좋은 성능을 보였다.

## Semi-supervised learning

### 다른 기법들

ImageNet supervised learning에서 사용했던 기법들과 비교를 해보았을 때 성능이 좋았다.

<p align="center">
    <img src="/assets/post/image/legacy/mpl-supervised.png" width="50%">
</p>

## ImageNet

ImageNet을 labeled data, JFT를 unlabeled data로 사용하여 semi-supervised learning을 한 후 imagenet으로 finetuning한 결과 SOTA를 찍었고
supervised learning보다 성능이 좋았다.

<p align="center">
    <img src="/assets/post/image/legacy/mlp-imagenet.png" width="50%">
</p>
