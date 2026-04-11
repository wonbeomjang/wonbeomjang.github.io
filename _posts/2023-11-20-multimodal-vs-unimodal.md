---
layout: post
title: "What Makes Multi-modal Learning Better than Single (Provably)"
date: 2023-11-20 00:00:00 +0900
description: Multimodal vs Unimodal 성능 비교 논문 리뷰 (NeurIPS 2021)
categories: [paper]
tags: [multi-modal, paper]
giscus_comments: true
related_posts: true
---

> [논문 링크](https://arxiv.org/abs/2106.04538)

# Introduction

우리 세상에는 다양한 modality가 존재합니다. 직관적으로 여러 modal의 정보를 결합(fusion)하면, uni-modal보다 더 나은 성능을 얻을 수 있을 것이라 생각됩니다. 이에 대해 다음과 같은 질문을 던질 수 있습니다.

<p align="center">
  
_multi-modal learning이 uni-modal learning보다 항상 좋은 성능을 제공할까?_

</p>

저자는 이 질문을 중심으로 연구를 시작하며, 다음 두 가지를 중점적으로 살펴보았습니다.

1. **어떤 상황에서 multi-modal이 uni-modal보다 성능이 좋은가?**
2. **multi-modal 학습이 더 나은 성능을 제공하는 이유는 무엇인가?**

이를 통해 저자는 다음과 같은 기여를 했습니다:

- Multi-modal learning을 population risk 관점에서 설명하고, latent representation quality와 연결 지었습니다.
- 특정 modality subset으로 학습한 network의 성능 상한선을 이론적으로 제시했습니다.
- Modalities의 subset만 사용할 경우 성능이 저하되는 이유를 분석했습니다.

결론은 다음과 같습니다:

- Multiple modalities는 특정 modal subset보다 낮은 population risk를 갖습니다.
- 이는 multi-modal 학습이 더 정확한 latent space representation을 제공하기 때문입니다.

이제 세부적으로 살펴보겠습니다.

---

# The Multi-modal Learning Formulation

<p align="center">
    <img src="/assets/post/image/multi-modal-vs-uni-modal/figure1.png" width="80%">
</p>

## Multi-modal 데이터의 수학적 정의

K개의 modalities에 대해 데이터는 다음과 같이 표현됩니다.  
$$\mathbb{x}:=(x^{(1)},\cdots,x^{(K)})$$  
이때, $$x^{(k)} \in \mathcal{X}^{(k)}$$ 이며, 전체 input data space는 다음과 같습니다.  
$$\mathcal{X}=\mathcal{X}^{1} \times \cdots \times \mathcal{X}^{k}$$

target domain은 $$\mathcal{Y}$$, 공통된 latent space는 $$\mathcal{Z}$$로 정의합니다. True mapping은 다음과 같습니다:  
$$g^\star: \mathcal{X} \mapsto \mathcal{Z}, \quad h^\star: \mathcal{Z} \mapsto \mathcal{Y}$$

데이터의 분포는 다음과 같이 정의됩니다.  
$$\mathbb{P}_\mathcal{D}(\mathbb{x},y)\triangleq\mathbb{P}_{y|x}(y|h^\star\circ g^\star(\mathbb{x}))\mathbb{P}_\mathbb{x}(\mathbb{x})$$

## Subset Modalities

우리는 K개의 modalities 중 $$\mathcal{N} \leq \mathcal{M}$$ 인 subset을 선택할 수 있습니다.  
이때 modality의 superset은 다음과 같습니다:  
$$\mathcal{X}^\prime := (\mathcal{X}^{(1)}\cup\bot)\times\cdots\times(\mathcal{X}^{(K)}\cup\bot)$$  
여기서 $$\bot$$은 특정 modality를 사용하지 않음을 의미합니다.

modality 선택 함수 $$p_\mathcal{M}$$는 다음과 같이 정의됩니다.

$$
p_\mathcal{M}(\mathbb{x})^{(k)}=
\begin{cases}
\mathbb{x}^{(k)} & \text{if } k\in\mathcal{M}, \\
\bot & \text{else}.
\end{cases}
$$

## 학습 목표: Empirical Risk Minimization (ERM)

우리의 목표는 ERM에 따라 학습 objective를 최소화하는 것입니다:

$$
\text{min } \hat{r}(h\circ g_\mathcal{M}) = \frac{1}{m}\sum_{i=1}^ml(h\circ g_\mathcal{M}(\mathbb{x}_i),y_i),
\quad \text{s.t. } h \in \mathcal{H}, g_\mathcal{M} \in \mathcal{G}.
$$

최종적으로 population risk는 다음과 같이 정의됩니다.  
$$r(h\circ g_\mathcal{M})=\mathbb{E}_{(\mathbb{x}_i, y_i)\sim\mathcal{D}}[\hat{r}(h\circ g_\mathcal{M})]$$

---

# Main Result

### Latent Representation Quality 정의

> **Definition 1.**  
> 데이터 분포에서 학습된 latent representation mapping $$g \in \mathcal{G}$$의 *quality*는 다음과 같이 정의됩니다.  
> $$\eta(g) = \text{inf}_{h\in\mathcal{H}}[r(h\circ g)-r(h^\star\circ g^\star)]$$  
> 즉, true latent space와의 차이를 측정하며, 이를 latent space quality라 부릅니다.

---

## Rademacher Complexity

Model complexity를 측정하는 Rademacher complexity는 다음과 같습니다.

- $$\mathcal{F}$$를 $$\mathbb{R}^d \mapsto \mathbb{R}$$인 함수 집합으로 정의합니다.
- $$Z_1, \ldots, Z_m$$은 $$\mathbb{R}^d$$에서 iid로 샘플된 데이터이고, $$S=(Z_1,\ldots,Z_m)$$라고 합니다.

Empirical Rademacher complexity는 다음과 같이 정의됩니다.  
$$\hat{\mathfrak{R}}_S(\mathcal{F}):=\mathbb{E}_\sigma[\underset{f\in\mathcal{F}}{\text{sup}} \frac{1}{m}\sum_{i=1}^m\sigma_if(Z_i)]$$

여기서 $$\sigma=(\sigma_1,...,\sigma_m)^\top$$이고, $$\sigma_i$$는 $$\{-1, 1\}$$에서 uniform하게 추출된 random variable입니다.

---

## Latent Space Quality와 Population Risk의 관계

> **Theorem 1.**  
> $$S = \{(x_i, y_i)\}_{i=1}^m$$이 데이터셋이고, $$\mathcal{M}, \mathcal{N}$$은 modality의 두 subset입니다. $$\mathcal{M}$$과 $$\mathcal{N}$$으로 각각 학습된 empirical risk minimizers $$(\hat{h}_\mathcal{M}, \hat{g}_\mathcal{M})$$와 $$(\hat{h}_\mathcal{N}, \hat{g}_\mathcal{N})$$에 대해, 다음이 성립합니다.

$$r(\hat{h}_{\mathcal{M}} \circ \hat{g}_{\mathcal{M}}) - r(\hat{h}_{\mathcal{N}} \circ \hat{g}_{\mathcal{N}}) \leq \gamma_{\mathcal{S}}(\mathcal{M},\mathcal{N}) + \text{O}(1/m)$$  
여기서 $$\gamma_S(\mathcal{M},\mathcal{N})\triangleq\eta(\hat{g}_\mathcal{M})-\eta(\hat{g}_\mathcal{N})$$는 latent space quality의 차이입니다.

---

# Experiment

## Dataset: IEMOCAP

Interactive Emotional Dyadic Motion Capture 데이터셋을 사용했습니다. 데이터셋에는 Text, Video, Audio 정보가 포함되어 있으며, 발화자의 감정을 예측하는 것이 목표입니다.

### 결과

1. **Modalities가 많을수록 정확도가 향상됩니다.**
2. **Sample이 적을 경우, subset modalities가 더 나은 성능을 보일 수 있습니다.**
3. **Multi-modal 학습은 더 나은 latent space quality를 제공합니다.**

<p align="center">
    <img src="/assets/post/image/multi-modal-vs-uni-modal/table3.png" width="80%">
</p>

---

## 결론

- 데이터 크기가 충분히 클 때 multi-modal을 사용하는 것이 유리합니다.
- multi-modal 학습은 더 정확한 latent space representation을 학습할 수 있습니다.
- 이론적 분석과 실험 결과 모두 이를 뒷받침합니다.
