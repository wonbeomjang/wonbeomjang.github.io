---
layout: post
title: "Invariant Representation for Unsupervised Image Restoration"
date: 2023-04-30 00:00:00 +0900
description: 비지도 이미지 복원 논문 리뷰 — 라벨 없이 이미지를 복원하는 방법
categories: [paper]
tags: [paper, image-restoration, unsupervised]
giscus_comments: true
related_posts: true
---

> [Invariant Representation for Unsupervised Image Restoration](https://arxiv.org/abs/2106.06927)

# Introduction

Image restoration은 대부분 pair한 dataset이 필요하다. 하지만 이를 구하는 것은 어려워서 CycleGAN을 이용하여 Image Restoration을 진행하는 경우가 있다. 하지만 CycleGAN과 같은 Image to Image translation과 DIRT와 같은 unsupervised domain adaptation은 다음과 같은 단점을 가지고 있다.

### Indistint Domain Boundary

Horse to zebra와 같이 image translation은 분명한 domain boundary가 존재한다. 하지만 image restoration은 noise level과 복잡한 background가 domain boundary를 희미하게 만들어서 이미지 퀄리티가 낮아진다.

### Weak Representation

Unsupervised Domain Adaptation은 high-level representation만 추출한다. 이는 domain shift problem을 야기하여 low-quality reconstruction을 만들어낸다.

### Poor Generalization

One-to-one image translation은 semantic representation과 texture representation을 분리하여 잡아내기 힘들다.

따라서 이 논문에서 위의 문제를 해결하며 다음의 contribution을 남겼다.

1. Image restoration분야에서 unsuperviesd representation learning method를 제안했다.
2. Dual domain constraint를 통해 semantic representation과 texture representation; 두 개의 representation을 분리했다.
3. Domain transfer를 통해 unsupervised image restoration을 제안했다.

# The Proposed Method

<p align="center">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled.png" width="80%">
</p>

먼저 문제에 대한 정의를 하겠다. $$\mathcal{X}$$를 Noisy Image Domain, $$\mathcal{Y}$$를 Clean Image Domain이라고 하겠다. Encoder는 각각의 도메인을 같은 vector space인 shared-latent space $$\mathcal{Z}$$로 projection 시킨다. 따라서 vector space에 대하여 다음의 식이 성립한다.

$$
z=E_\mathcal{X}(x)=E_\mathcal{Y}(y)
$$

Generator는 shared-latent space $$\mathcal{Z}$$에서 image를 만들어낸다. 따라서 다음과 같은 식이 성립한다.

$$
x=G_\mathcal{X}(z),y=G_\mathcal{Y}(z)
$$

이 때 각각의 도메인에대해 Encoder와 Generator는 $$\{E_\mathcal{X}, G_\mathcal{X}\}, \{E_\mathcal{Y}, G_\mathcal{Y}\}$$ 각각 존재한다. 각각의 encoder가 shared-latent space $$\mathcal{Z}$$로 projection을 시킨다고 하더라도 각각의 latent vector는 다르다. 따라서 latent vector를 구분하여 적어주겠다.

$$
z_\mathcal{X}=E_\mathcal{X}(x), z_\mathcal{Y}=E_\mathcal{Y}(y)
$$

따라서 우리가 하고싶어하는 Image restoration과정은 다음과 같다.

$$
F^{\mathcal{X}\rightarrow\mathcal{Y}}(x)=G_\mathcal{Y}(z_\mathcal{X})
$$

## Discrete Representation Learning

먼저 one-to-one image translation의 poor generalization 문제를 해결하기 위해 semantic representation과 texture(noise) representation을 분리시켜야한다. 따라서 저자는 다음 4가지의 방법론을 제시했다.

1. Detangling Representation
2. Forward Cross Translation
3. Backward Cross Reconstruction
4. Adversarial Domain Adaptation

### Detangling Representation

<p align="center">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%201.png" width="80%">
</p>

먼저 extra noise encoder($$E_\mathcal{X}^N$$)을 도입을 한다. $$E_\mathcal{X}^N$$은 noise를 나타내는 texture latent vector를 뽑아내는 역할로 이를 도입해서 semantic representation과 texture representation을 분리했다. 이를 통해 $$z_\mathcal{X}$$와 $$z_\mathcal{Y}$$는 같은 distribution을 가지게 된다. 만약 noise image를 self-reconstruction하려면 $$x=G_\mathcal{X}(z_\mathcal{X}, z_\mathcal{X}^N)$$을 통하여 같이 reconstruction하면 된다.

### Forward Cross Translation

CycleGAN처럼 noise image에서 clean image 변환과 clean image에서 noise이미지의 변환이 되어야한다. 따라서 다음과 같은 방법으로 이미지 변환을 한다. 이 때 $$\mathcal{Y}$$에 $$\mathcal{X}$$의 noise를 추가하기 위해 $$z_\mathcal{X}^N$$를 이용한다.

1. Noise image → clean image: $$\tilde{x}^{\mathcal{X}\rightarrow\mathcal{Y}} =G_\mathcal{Y}(z_\mathcal{X})$$
2. Clean image → noise image: $$\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}} =G_\mathcal{X}(z_\mathcal{Y}\oplus z_\mathcal{X}^N)$$

### Backward Cross Translation

Forward cross translation을 했으니 backward cross translation을 할 수 있다. 이 때 $$\mathcal{X}$$에 $$\mathcal{Y}$$의 noise를 추가하기 위해 $$E_\mathcal{X}^N(\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}})$$를 이용한다.

1. Noise image → clean image: $$\hat{x}=G_\mathcal{X}(E_\mathcal{Y}(\tilde{x}^{\mathcal{X}\rightarrow\mathcal{Y}})\oplus E_\mathcal{X}^N(\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}}))$$
2. Clean image → noise image: $$\hat{y}=G_\mathcal{Y}(E_\mathcal{X}(\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}}))$$

Backward cross translatio를 학습하기 위해 loss를 다음과 같이 구성한다.

$$
\mathcal{L}_\mathcal{X}^{CC}(G_\mathcal{X},G_\mathcal{Y},E_\mathcal{X},E_\mathcal{Y},E_\mathcal{X}^N)=\mathbb{E}_\mathcal{X}[||G_\mathcal{X}(E_\mathcal{Y}(\tilde{x}^{\mathcal{X}\rightarrow\mathcal{Y}})\oplus E_\mathcal{X}^N(\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}}))-x||_1]
$$

$$
\mathcal{L}_\mathcal{Y}^{CC}(G_\mathcal{X},G_\mathcal{Y},E_\mathcal{X},E_\mathcal{Y},E_\mathcal{X}^N)=\mathbb{E}_\mathcal{X}[||G_\mathcal{Y}(E_\mathcal{X}(\tilde{y}^{\mathcal{Y}\rightarrow\mathcal{X}}))-y||_1]
$$

### Adversarial Domain Adaptation

Semantic representation ($$z_\mathcal{X}, z_\mathcal{Y}$$)은 같은 vector space를 사용해야한다. 따라서 이를 강제하기 위해서 reprenentation discriminator $$D_r$$를 사용한다.

$$
\mathcal{L}^\mathcal{R}_{adv}(E_\mathcal{X},E_\mathcal{Y},D_\mathcal{R})=\mathbb{E}_\mathcal{X}[\frac{1}{2}logD_\mathcal{R}(z_\mathcal{X}+\frac{1}{2}(1-logD_\mathcal{R}(z_\mathcal{X})))] + \mathbb{E}_\mathcal{Y}[\frac{1}{2}logD_\mathcal{R}(z_\mathcal{Y}+\frac{1}{2}(1-logD_\mathcal{R}(z_\mathcal{Y})))]
$$

## Self-Supervised Constraint

### Background Consistency Loss

<p align="center">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%202.png" width="50%">
</p>

복잡한 background의 Indistint Domain Boundary을 해결하기 위해 backgound consistency loss를 제안한다. Noise가 있는 이미지와 깨끗한 이미지는 gaussian blur를 하면 structure 정보만 남기 때문에 background의 비교가 가능하다. 따라서 저자는 다음과 같은 loss를 추가한다.

$$
\mathcal{L}_{BC}=\sum_{\sigma=5,9,15} \lambda_\sigma||B_\sigma(\mathcal{X})-B_\sigma(\tilde{\mathcal{X}})||_1
$$

### Semantic Consistency Loss

perception loss에서 영감을 받아 pretrained-backbone을 통한 semantic한 정보는 noise가 줄어들 것이라고 기대가 된다. 따라서 VGG19에서 conv5-1 layer와 같이 깊은 layer에서 feature map을 뽑아 비교하여 semantic representations의 consistency를 유지한다.

$$
\mathcal{L}_{SC}=||\phi_l(\mathcal{X}-\phi_l(\tilde{\mathcal{X}})||_2^2
$$

## Joint Optimizing

좋은 성능을 위하여 다른 여러가지도 추가했다.

### Target Domain Adversarial Loss

Noise Domain과 clean image domain에서 결과물을 더 잘만들기 위해 GAN loss를 추가한다.

$$
\mathcal{L}_{adv}^\mathcal{X}=\mathbb{E}_{x\sim P_\mathcal{X}(x)}[logD_\mathcal{X}(x)]+\mathbb{E}_{y\sim P_\mathcal{Y}(y), x \sim P_\mathcal{X}(x)}[log(1-D_\mathcal{X}(G_\mathcal{X}(E_\mathcal{Y}(y), E_\mathcal{X}^N(x))))]
$$

$$
\mathcal{L}_{adv}^\mathcal{Y}=\mathbb{E}_{y\sim P_\mathcal{Y}(y)}[logD_\mathcal{Y}(y)]+\mathbb{E}_{x \sim P_\mathcal{Y}(y)}[log(1-D_\mathcal{Y}(G_\mathcal{Y}(E_\mathcal{X}(x)))]
$$

### Self Reconstruction Loss

안정적인 학습 진행을 위해서 self reconstruction loss도 추가하였다.

$$
\hat{x}=G_\mathcal{X}(E_\mathcal{X}(x)\oplus E_\mathcal{X}^N(x)), \hat{y}=G_\mathcal{Y}(E_\mathcal{Y}(y))
$$

$$
\mathcal{L}^\mathcal{X}_{rec}=||\hat{x} - x||_1, \mathcal{L}^\mathcal{Y}_{rec}=||\hat{y} - y||_1
$$

### KL Divergence Loss

Noise는 보통 normal distribution을 따른다. 따라서 이 논문에서도 latent-vector가 normal distribution을 따르도록 KL divergence loss를 추가했다.

$$
p(z_\mathcal{X}^N\sim N(0, 1))
$$

## Total Loss

모든 loss를 합치면 다음과 같다.

$$
\underset{E_\mathcal{X},E_\mathcal{X}^N,E_\mathcal{Y},G_\mathcal{X},G_\mathcal{Y}}{\operatorname{min}} \underset{D_\mathcal{X},D_\mathcal{Y},D_\mathcal{R}}{\operatorname{max}} =\lambda_\mathcal{R}\mathcal{L}^\mathcal{R}_{adv}+\lambda_{adv}\mathcal{L}^{domain}_{adv}+\lambda_{CC}\mathcal{L}^{CC}+\lambda_{rec}\mathcal{L}^{Rec}+\lambda_{bc}\mathcal{L}^{BC}+\lambda_{sc}\mathcal{L}^{SC}+\lambda_{KL}\mathcal{L}^{KL}


$$

## Restoration

<p align="center">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%203.png" width="80%">
</p>

학습이 끝난 후에 noise가 있는 이미지를 복원하려면 cross encoder-generator $$\{ E_\mathcal{X}, E_\mathcal{Y}\}$$를 사용하면 된다.

$$
\tilde{x}^{\mathcal{X}\rightarrow \mathcal{Y}}=G_\mathcal{Y}(E_\mathcal{X}(x))
$$

# Experiment

<p align="center">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%204.png" width="70%">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%205.png" width="70%">
    <img src="/assets/post/image/unsupervised-image-restoration/Untitled%206.png" width="70%">
</p>

**Unsupervised 방법들 중에서 SOTA 성능**을 달성했다. 핵심 관찰:

- **Supervised 방법 대비**: 당연히 supervised 방법(pair 데이터 사용)보다는 성능이 낮다. 하지만 pair 데이터가 없는 현실적인 상황을 고려하면 의미있는 결과이다.
- **CycleGAN 대비**: CycleGAN 기반 방법들은 domain boundary가 불명확한 image restoration에서 artifact이 발생하기 쉬운데, 이 논문은 semantic/texture representation 분리와 background consistency loss로 이를 완화했다.
- **질적 비교**: 시각적으로 보면, 기존 unsupervised 방법들은 noise를 제거하면서 texture detail도 함께 사라지는 경우가 많은데, 이 논문의 방법은 noise는 제거하면서 semantic structure를 잘 보존한다. 이는 dual domain constraint가 두 representation을 효과적으로 분리했기 때문이다.

## 이 논문의 의의

Unsupervised image restoration이라는 어려운 문제에 대해, CycleGAN의 한계(불명확한 domain boundary, weak representation, poor generalization)를 체계적으로 분석하고, 각 문제에 대응하는 solution(background consistency loss, adversarial domain adaptation, detangling representation)을 제안한 것이 핵심 기여이다. 다만 loss가 7개로 매우 많아서 hyperparameter tuning이 어려울 수 있다는 점은 실용적 한계이다.
