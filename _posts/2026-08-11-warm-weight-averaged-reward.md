---
layout: post
title: "WARM: reward model을 weight 공간에서 평균내다"
date: 2026-08-11 09:13:00 +0900
description: "RLHF Reward 설계 시리즈 #13 — 앙상블의 강건성을 단일 모델 비용으로 얻어 reward hacking을 막는 법"
categories: [paper]
tags: [rlhf, reward-model, reward-hacking, model-merging, robustness, paper]
giscus_comments: true
related_posts: true
---

> [WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/abs/2401.12187) (Ramé et al., Google DeepMind, ICML 2024)

# Introduction

이 시리즈 3부는 reward hacking을 한 방향으로 계속 좁혀왔다. [#10 Overoptimization Scaling Laws](/blog/2026/reward-model-overoptimization/)는 "proxy RM으로 최적화할수록 실제 선호와의 괴리가 커진다"는 현상을 Goodhart의 법칙으로 정량화했고, [#11 Length Correlations](/blog/2026/rlhf-length-correlations/)는 그 괴리의 상당 부분이 다름 아닌 **길이**에서 온다는 걸 보였다. [#12 ODIN](/blog/2026/odin-disentangled-reward/)은 그 진단을 그대로 받아 "그럼 길이를 reward에서 분리해버리자"는 **축 특정(axis-specific)** 해법을 냈다 — reward를 길이 성분과 내용 성분으로 나누고, 길이 성분에는 그래디언트를 흘리지 않는다.

이 글이 다루는 WARM은 정확히 그 반대편에 선다. ODIN이 "범인은 길이다"라고 지목하고 그 축만 도려냈다면, WARM은 **범인이 무엇인지 아예 모른 채로** 방어한다. 길이일 수도, 특정 문체일 수도, 우리가 아직 이름 붙이지 못한 어떤 상관관계일 수도 있다 — WARM은 그중 무엇이든 상관없이 억제한다. 이게 이 논문이 reward 설계에 던지는 질문이다: **hacking의 축을 매번 새로 찾아 수술하는 대신, 애초에 hacking에 잘 버티는 reward model을 만들 수는 없을까?**

논문은 reward hacking의 원인을 두 갈래로 정리한다.

1. **분포 이동(distribution shift)**: RM은 RL이 시작되기 전에 모아둔 오프라인 선호 데이터로 학습된다. 그런데 정책은 RL이 진행될수록 SFT 초기화에서 점점 멀어지고, RM이 한 번도 보지 못한 종류의 생성물을 만들어낸다. 용량이 제한된 RM은 이 미지의 영역에서 엉뚱한 상관관계에 의존해 점수를 매기고, 정책은 그 허점을 놓치지 않는다.
2. **선호 라벨의 불일치(inconsistent preferences)**: 사람이 붙인 이진 라벨 자체가 노이즈투성이다. 라벨러는 피로하고, 기준이 제각각이며, 복잡한 판단 대신 길이·글머리표·공손함 같은 단순한 대리 지표에 기댄다. InstructGPT의 라벨러간 일치도가 겨우 72.6%였다는 사실이 이 문제의 크기를 보여준다.

기존 방어책 중 하나가 **앙상블(prediction ensembling, ENS)**이다 — 사실 이 시리즈 [#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)이 이미 RM 여러 개를 부트스트랩으로 학습시켜 불일치 기반 질의 선택에 썼던 바로 그 아이디어다. RM을 여러 개 두고 예측을 평균내면 확실히 더 안정적이다. 문제는 비용이다. RL 루프는 매 스텝 정책이 만든 생성물에 reward를 매겨야 하는데, RM이 $$M$$개면 그 계산을 $$M$$번 반복해야 한다. 메모리도 $$M$$배, 추론 지연도 $$M$$배. RM이 커질수록(그리고 최근 RM은 계속 커지는 추세다) 이 비용은 감당하기 어려워진다.

WARM의 해법은 단순하다. **같은 사전학습 모델에서 출발해 하이퍼파라미터만 다르게 파인튜닝한 $$M$$개의 RM을, 예측이 아니라 weight 자체를 평균낸다.** 결과물은 단일 모델 하나뿐이라 추론 비용이 늘지 않는다. 그런데도 앙상블에 준하는 신뢰성·강건성을 얻는다. 저자들은 이 근거를 **linear mode connectivity(LMC)** — 같은 사전학습 초기화를 공유하는 파인튜닝 결과물들은 weight 공간에서 선형으로 연결되어 있다는 성질 — 에서 찾는다. 이 구조로 학습한 정책은 단일 RM으로 학습한 정책 대비 **79.4%의 승률**을 기록했다. 이 글은 그 메커니즘을 수식과 토이 예제로 뜯어보고, 마지막에는 WARM도 완전히 해결하지 못한 것이 무엇인지 짚는다.

# Background

## RLHF reward 학습 복습

[#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)에서 다룬 것과 똑같은 구조가 여기서도 등장한다. LLM $$f_\theta$$는 사전학습 $$\theta^{pt}$$를 거쳐 SFT로 $$\theta^{sft}$$가 되고, RM $$r_\phi$$는 $$(\theta^{sft}, \omega)$$에서 초기화된다 — $$\omega$$는 SFT 모델의 특징 위에 새로 얹는 선형 분류 레이어다. 선호 데이터셋 $$\mathcal{D}_{train} = \{x_d, y_d^+, y_d^-\}_{d=1}^D$$로 다음 Bradley-Terry cross-entropy를 최소화해 $$\phi$$를 학습한다.

$$\mathcal{L}_R(r_\phi, \mathcal{D}_{train}) = -\mathbb{E}_{(x,y^+,y^-)\sim \mathcal{D}_{train}}\left[\log \sigma\left(r_\phi(x,y^+) - r_\phi(x,y^-)\right)\right]$$

- $$r_\phi(x,y)$$: 프롬프트 $$x$$와 생성물 $$y$$에 대해 RM이 매기는 스칼라 reward.
- $$y^+, y^-$$: 사람(혹은 AI)이 $$x$$에 대해 더 선호한/덜 선호한 생성물.
- $$\sigma$$: 로지스틱 함수. $$r_\phi(x,y^+)$$가 $$r_\phi(x,y^-)$$보다 클수록 손실이 작아진다.

이 손실 함수는 [#1 글](/blog/2026/deep-rl-human-preferences/)에서 이미 본 것과 정확히 같은 형태다. WARM이 새로 건드리는 건 손실 함수가 아니라 **학습된 $$\phi$$들을 어떻게 하나로 합치는가**다.

## reward hacking의 두 원인, 그리고 RM에게 요구되는 세 가지

WARM은 좋은 RM의 조건을 세 가지로 정의한다 — 각각이 앞서 말한 두 원인과 대응된다.

| 속성                | 요구되는 이유                                                                        | 관련 원인   |
| ------------------- | ------------------------------------------------------------------------------------ | ----------- |
| 효율성(efficiency)  | RM 계산이 정책 학습의 매 스텝마다 들어가므로, 추가 비용이 없어야 RL을 감당할 수 있다 | (공통)      |
| 신뢰성(reliability) | 정책이 SFT 초기화에서 멀어져도 RM이 정확한 점수를 매길 수 있어야 한다                | 분포 이동   |
| 강건성(robustness)  | 라벨에 섞인 노이즈에 RM이 휘둘리지 않아야 한다                                       | 선호 불일치 |

RM 하나만 쓰면 셋 다 만족하기 어렵다. 제한된 학습 분포 안에서 익힌 상관관계는 분포가 이동하면 무너지고, 노이즈 섞인 라벨은 고스란히 암기된다. 이 두 문제를 각각 "OOD에서 예측이 흔들린다"와 "틀린 라벨까지 외운다"로 바꿔 말하면, 둘 다 **분산(variance)**과 **암기(memorization)**의 문제로 수렴한다.

## 기존 접근법과 앙상블의 대가

논문이 정리하는 기존 대응책은 다음과 같다.

| 방법                              | 효율성(추론 비용)                  | 신뢰성(분포 이동)                   | 강건성(라벨 노이즈)    |
| --------------------------------- | ---------------------------------- | ----------------------------------- | ---------------------- |
| KL 정규화만 사용                  | 유지                               | 모델 drift를 억제해 부분적으로 개선 | 대응하지 못함          |
| 새 데이터 수집·재라벨링           | 크게 저하(지속적 사람 라벨링 필요) | 개선                                | 개선 여지 있음         |
| Active learning                   | 다소 저하                          | 개선                                | 개선 여지 있음         |
| Label smoothing/flipping          | 유지                               | 대응하지 못함                       | 부분적으로 개선        |
| 예측 앙상블(ENS, $$M$$개 RM 평균) | **$$M$$배로 저하**                 | 개선                                | **논문이 실패를 실증** |
| WARM (본 논문)                    | 유지(단일 모델과 동일)             | 개선                                | 개선                   |

이 중 강건성을 논외로 하면, 앙상블은 가장 직관적이고 효과적인 신뢰성 개선책이다. RM $$M$$개의 예측을 평균내면 각 RM이 개별적으로 물고 늘어지는 허점이 서로 상쇄된다. WARM 논문도 "prediction ensembling이 hacking 위험을 줄인다는 건 이미 알려져 있다"고 인정한다. 문제는 이 신뢰성이 **선형으로 증가하는 메모리·추론 비용**과 맞바꾼 것이라는 점이다. RM이 5개면 RL 루프의 매 스텝마다 forward pass도 5번, 파라미터도 5벌 들고 있어야 한다 — RM 하나가 이미 수십억 파라미터급인 시대에 이 배율은 그대로 학습 예산의 부담이 된다.

## 앙상블 접근을 더 깊이 판 논문 — Reward Model Ensembles Help Mitigate Overoptimization

이 대가를 구체적인 숫자로 보여준 논문이 [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743) (Coste et al., University of Cambridge, ICLR 2024)이다. 이 논문은 [#10 Gao et al. 2022](/blog/2026/reward-model-overoptimization/)와 같은 synthetic gold-RM 실험 셋업을 그대로 가져오되, 사람 라벨러의 실제 일치도(60~75%)를 반영해 선호 라벨의 **25%를 무작위로 뒤집는** 조건을 추가했다. $$k$$개 RM 앙상블 $$\{R_1, \ldots, R_k\}$$을 두고, 그 예측을 어떻게 합치느냐에 따라 세 가지 목적함수를 비교한다.

$$R_{\mu}(q,a) := \frac{1}{k}\sum_{i=1}^{k} R_i(q,a)$$

$$R_{WCO}(q,a) := \min_i R_i(q,a)$$

$$R_{UWO}(q,a) := \underbrace{\frac{1}{k}\sum_i R_i(q,a)}_{\text{평균}} - \lambda \underbrace{\frac{1}{k}\sum_i \left(R_i(q,a) - \frac{1}{k}\sum_j R_j(q,a)\right)^2}_{\text{앙상블 내 분산}}$$

- $$R_\mu$$(mean): 그냥 평균. 보수적이지 않다 — RM 하나라도 reward를 과대평가하면 정책이 그 허점을 그대로 파고든다.
- $$R_{WCO}$$(worst-case optimization): 앙상블 중 **최솟값**을 쓴다. 하나라도 정직하게 낮은 점수를 매기면 hacking을 막을 수 있다는 논리다. 하이퍼파라미터가 없다는 게 장점이지만 지나치게 보수적일 수 있다.
- $$R_{UWO}$$(uncertainty-weighted optimization): 평균에서 **RM들끼리 의견이 갈리는 정도(분산)**를 $$\lambda$$만큼 깎는다. 앙상블 멤버들의 불일치 자체를 불확실성 신호로 쓰는 것이다.

이 논문은 기본값으로 **5개 RM 앙상블**을 쓴다. 즉 PPO의 매 스텝마다 reward 계산에 forward pass가 5번 필요하다 — WARM이 정확히 겨냥하는 그 비용이다. 성능 자체는 인상적이었다.

| 조건                           | 방법    | 결과                                                                 |
| ------------------------------ | ------- | -------------------------------------------------------------------- |
| BoN, 라벨 노이즈 없음          | WCO/UWO | 개별 RM 평균 대비 최대 약 30% 성능 개선, overoptimization 관찰 안 됨 |
| BoN, 라벨 노이즈 25%           | WCO/UWO | 최대 약 75% 개선. Mean 앙상블은 이 조건에서도 여전히 overoptimize    |
| PPO, 작은 KL 페널티(0.01)      | WCO/UWO | overoptimization 완전 방지, 성능 손실 없음                           |
| PPO, KL 페널티만으로 방지 시도 | 단일 RM | 20배 더 큰 페널티(0.2)가 필요하고, 그 과정에서 성능이 눈에 띄게 깎임 |

즉 앙상블은 **작동한다**. 다만 그 작동의 대가가 추론 비용 $$N$$배라는 사실은 그대로 남는다. WARM이 던지는 질문은 바로 여기서 시작한다 — 이 신뢰성을 유지하면서 비용만 1배로 되돌릴 방법은 없을까?

# Method

## WARM의 절차

WARM은 세 단계로 이루어진다.

1. **공유 사전학습 초기화**: 각 RM은 같은 $$(\theta^{sft}, \omega)$$에서 출발한다. $$\omega$$는 무작위 초기화 대신 **선형 프로빙(linear probing)**으로 미리 맞춰둔다 — 분류기를 무작위로 얹으면 파인튜닝 초반에 특징이 크게 뒤틀리는데(feature distortion), 이걸 막기 위해서다.
2. **다양한 파인튜닝**: 이 초기화에서 $$M$$개의 RM을 **서로 다른 하이퍼파라미터**(학습률, dropout 확률)와 **서로 다른 데이터 순서**로 파인튜닝해 $$\{\phi_i\}_{i=1}^M$$을 얻는다.
3. **weight 평균**: 이 $$M$$개의 weight를 그대로 산술 평균한다.

$$\phi^{\text{WARM}} = \frac{1}{M}\sum_{i=1}^{M} \phi_i$$

이렇게 얻은 $$r_{\phi^{\text{WARM}}}$$ 하나가 RL의 proxy RM이 된다. 추론 시점에는 이 단일 모델만 돌리면 되므로, 개별 RM 하나를 쓸 때와 비용이 완전히 같다.

<p align="center"><img src="/assets/post/image/warm-weight-averaged-reward/warm-method-overview.png" width="85%"></p>

위 그림이 이 절차를 보여준다. SFT 모델에서 서로 다른 하이퍼파라미터로 여러 RM을 파인튜닝한 뒤(초록 화살표), 그 weight들을 평균내(점선) 하나의 WARM으로 합치고, 이 WARM을 RL 루프의 reward function으로 사용한다.

## Linear mode connectivity — weight를 평균내도 되는 근거

두 신경망의 weight를 그냥 평균내는 게 상식적으로 이상하게 들릴 수 있다. 뉴런 순서가 다르고(permutation symmetry) 활성화 함수가 비선형인데, 파라미터를 산술 평균낸 결과가 의미 있는 모델이 될 거라는 보장이 어디 있는가? 답은 **같은 사전학습에서 출발했다면** 이 평균이 실제로 잘 작동한다는 경험적 관찰, LMC에 있다.

$$\phi_1, \phi_2$$ 두 개의 파인튜닝된 weight에 대해, 페어와이즈 정확도를 다음처럼 정의하자.

$$\mathrm{Acc}(r_\phi, \mathcal{D}) = \mathbb{E}_{(x,y^+,y^-)\sim \mathcal{D}}\left[\mathbb{1}_{r_\phi(x,y^+) \ge r_\phi(x,y^-)}\right]$$

- $$\mathbb{1}_{[\cdot]}$$: 조건이 참이면 1, 거짓이면 0인 indicator function.
- 즉 $$\mathrm{Acc}$$는 "RM이 더 선호되는 쪽에 더 높은 점수를 준 비율"이다.

**Observation 1 (LMC)**. 같은 사전학습을 공유하는 $$\phi_1, \phi_2$$에 대해, 모든 $$\lambda \in [0,1]$$에서

$$\mathrm{Acc}\left(r_{(1-\lambda)\phi_1 + \lambda\phi_2}, \mathcal{D}_{test}\right) \ge (1-\lambda)\,\mathrm{Acc}(r_{\phi_1}, \mathcal{D}_{test}) + \lambda\,\mathrm{Acc}(r_{\phi_2}, \mathcal{D}_{test})$$

좌변은 "두 weight를 $$\lambda$$ 비율로 섞은 모델"의 정확도, 우변은 "두 모델 정확도를 $$\lambda$$ 비율로 섞은 값"이다. 이 부등식은 **weight를 먼저 섞고 평가한 쪽이, 정확도를 나중에 섞은 값보다 항상 같거나 낫다**는 뜻이다 — 다시 말해 weight 공간에서의 보간이 손해를 보지 않는다.

<p align="center"><img src="/assets/post/image/warm-weight-averaged-reward/lmc-baklava-interpolation.png" width="70%"></p>

위 그림은 서로 다른 SFT 체크포인트에서 초기화한(가장 다양성이 큰 조건인 Baklava) 두 RM을 $$\lambda$$를 0에서 1까지 밀며 보간한 결과다. 검은 점선(Diag)이 우변, 즉 두 모델 정확도를 단순히 선형 보간한 값이다. 파란 실선(WA)은 weight를 보간한 모델의 실제 정확도인데, $$\lambda \approx 0.6\text{–}0.7$$ 부근에서 약 0.7648까지 올라가 양 끝($$\phi_1 \approx 0.7583$$, $$\phi_2 \approx 0.7625$$)보다도, Diag 선보다도 확실히 위에 있다. **weight를 평균낸 모델이 개별 모델보다도 더 정확해질 수 있다**는 걸 눈으로 보여주는 그림이다.

## 왜 weight 평균이 예측 평균과 비슷하게 작동하는가

두 앙상블 방식 — weight를 평균내는 WA와 예측을 평균내는 ENS — 이 비슷한 효과를 내는 이유는 사실 간단하다. $$\|\phi_1 - \phi_2\|$$가 충분히 작으면(같은 loss basin 안에 있으면), $$r_\phi$$를 $$\phi$$에 대해 1차 Taylor 전개했을 때 두 연산이 거의 같아지기 때문이다.

**Observation 2 (1차 근사)**. 모든 $$\lambda \in [0,1]$$에서

$$\mathrm{Acc}\left(r_{(1-\lambda)\phi_1 + \lambda\phi_2}, \mathcal{D}_{test}\right) \approx \mathrm{Acc}\left((1-\lambda)\, r_{\phi_1} + \lambda\, r_{\phi_2}, \mathcal{D}_{test}\right)$$

실제로 위 LMC 그림에서 빨간 점선(ENS)과 파란 실선(WA)이 거의 포개진다는 걸 볼 수 있다 — 눈으로도 확인되는 근사다.

일상 비유로 옮기면 이렇다. 같은 베이스 육수(사전학습)로 시작한 요리사 $$M$$명이 각자 소금·향신료 배합(하이퍼파라미터)만 살짝 다르게 해서 소스를 만들었다고 하자. **ENS**는 $$M$$개의 완성된 요리를 각각 통째로 차려내고, 손님이 한 입씩 맛본 뒤 평균 점수를 매기는 것과 같다 — 요리를 $$M$$벌 다 만들어야 한다. **WA**는 그 대신 재료 배합 자체를 미리 평균내 소스 한 병만 만드는 것이다. 같은 육수에서 출발했기 때문에 배합을 섞어도 여전히 말이 되는 소스가 나온다(LMC). 그리고 배합이 충분히 비슷하다면, 그렇게 만든 한 병의 소스 맛은 $$M$$벌을 다 맛보고 평균낸 것과 거의 같다(Observation 2). 논문이 이 방법을 짓는 이름들 — model soups, model ratatouille, 그리고 WARM 자신의 다양화 기법인 **Baklava** — 이 전부 이 요리 비유를 그대로 따온 것도 우연이 아니다.

### 토이 예제 1 — 선형 RM에서는 WA와 ENS가 정확히 같다

작은 숫자로 Observation 2를 직접 확인해보자. RM을 $$r_\phi(y) = w^\top \varphi(y) + b$$ 형태의 선형 함수로 단순화하고, 특징 벡터 $$\varphi(y) = (\text{content}, \text{length})$$를 쓰는 두 RM이 있다고 하자.

| 항목                      | RM1 ($$\phi_1$$) | RM2 ($$\phi_2$$) | 평균       |
| ------------------------- | ---------------- | ---------------- | ---------- |
| $$w$$ = (content, length) | (0.5, 1.0)       | (0.3, 1.4)       | (0.4, 1.2) |
| $$b$$                     | 0.1              | −0.1             | 0.0        |

후보 $$y$$의 특징이 $$\varphi(y) = (1, 2)$$라면,

$$r_1(y) = 0.5 \times 1 + 1.0 \times 2 + 0.1 = 2.6, \qquad r_2(y) = 0.3 \times 1 + 1.4 \times 2 - 0.1 = 3.0$$

**ENS**(예측 평균): $$\dfrac{2.6 + 3.0}{2} = 2.8$$

**WA**(weight 평균): $$r_{\text{avg}}(y) = 0.4 \times 1 + 1.2 \times 2 + 0.0 = 2.8$$

정확히 일치한다. 선형 함수라면 "weight를 먼저 평균내고 계산"과 "각각 계산한 뒤 평균"이 대수적으로 같은 연산이기 때문이다. 실제 transformer는 비선형이라 완전히 같지는 않지만, 이 선형 케이스가 왜 근사가 성립하는지의 핵심을 그대로 보여준다.

## 다양성의 원천 — $$M$$개를 어떻게 다르게 만드는가

LMC가 성립하려면 같은 사전학습을 공유해야 하지만, 동시에 $$M$$개의 weight가 서로 완전히 같으면 평균낼 이유가 없다. 논문은 이 둘 사이의 균형을 위해 세 가지 다양성 소스를 쓴다.

| 다양성 소스         | 방법                                                                                                                     | 비고                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| 데이터 순서         | 같은 데이터셋을 서로 다른 순서로 학습                                                                                    | 가장 약한 다양성                 |
| 하이퍼파라미터      | 학습률·dropout 확률을 다르게                                                                                             | grid search 하듯 샘플링          |
| **Baklava**(초기화) | 하나의 SFT 궤적에서 **서로 다른 학습 스텝의 체크포인트** $$\{\theta_i^{sft}\}_{i=1}^M$$을 가져와 각 RM의 시작점으로 사용 | 가장 강한 다양성, 추가 비용 없음 |

Baklava라는 이름은 그 자체로 이 방법의 구조를 보여준다 — model soups(모든 RM이 완전히 같은 초기화에서 출발)보다 초기화 제약을 완화하되, 사전학습은 공유하는 다이아몬드 모양의 구조라서 붙은 이름이다(위 method 그림에서 SFT 궤적 위 서로 다른 지점에서 갈라져 나가는 화살표들이 이 모양을 만든다). 실제로 논문이 검증한 네 가지 다양성 조건 중 Baklava가 LMC 곡선을 가장 넓게 벌려, 즉 weight 평균의 이득을 가장 크게 만들었다.

이건 같은 신병훈련소에서 기초 훈련을 마친 두 군인이 서로 다른 특기병과로 갈라져 훈련받은 뒤에도 여전히 같은 작전 언어와 기본기를 공유해 손발이 맞는 것과 비슷하다. 반대로 애초에 서로 다른 나라, 다른 체계에서 훈련받은 두 사람을 억지로 한 팀에 넣으면 기본 동작부터 어긋난다 — 이게 왜 LMC가 "같은 사전학습을 공유해야 한다"는 조건을 요구하는지의 직관이다.

## 왜 WA가 ENS보다 노이즈에 강한가 — 2차 근사

1차 근사만 보면 WA는 그저 "ENS의 값싼 버전"이다. 그런데 실제로는 **WA가 라벨 노이즈에 대해 ENS보다 한 수 위**라는 사실이 실험적으로 관찰된다. 논문은 25%의 학습 라벨을 뒤집는 오염 실험으로 이걸 보인다: 오염된 학습 샘플에서는 WA 정확도가 ENS보다 뚜렷이 낮고(덜 외웠다는 뜻), 깨끗한 학습 샘플에서는 WA가 ENS보다 살짝 낮지만, **분포가 벗어날수록(OOD 테스트로 갈수록) WA가 ENS를 앞선다.**

이 현상을 설명하려고 논문은 단순화된 이론 모델을 세운다. $$F$$개의 직교하는 특징 $$\{z^j\}_{j=1}^F$$가 있고, 각 RM의 featurizer가 $$j$$번째 특징을 확률 $$p_j$$로 "학습해서 쓰기로 선택"한다고 하자(이진 선택자 $$f^j \in \{0,1\}$$). $$M \to \infty$$ 극한에서 두 결합 방식의 예측은 다음으로 수렴한다.

$$r_M^{ENS}(x) \xrightarrow{M \to \infty} y \cdot \sum_{j=1}^F p_j \cdot |z^j|^2$$

$$r_M^{WA}(x) \xrightarrow{M \to \infty} y \cdot \sum_{j=1}^F p_j^2 \cdot |z^j|^2$$

- ENS는 각 특징을 **그 특징이 학습될 확률 $$p_j$$ 그대로** 가중해 유지한다.
- WA는 그 확률의 **제곱 $$p_j^2$$**로 가중한다.

$$p_j$$가 1에 가까운 특징(모든 파인튜닝 run에서 일관되게 학습되는, 진짜 인과적인 신호)은 제곱을 해도 거의 그대로 남는다. 반면 $$p_j$$가 작은 특징(어떤 run에서는 학습되고 어떤 run에서는 안 되는, 노이즈나 문맥에 딸린 우연한 상관관계)은 제곱하는 순간 훨씬 더 작아진다. 즉 **WA는 "모든 run이 일관되게 합의한 신호"만 실질적으로 살아남기고, run마다 들쭉날쭉한 신호는 ENS보다 더 강하게 지운다.**

### 토이 예제 2 — 스퓨리어스 특징이 평균 속에서 사라지는 과정

같은 선형 설정으로 이번엔 "무엇을 배웠는지"가 RM마다 다른 경우를 보자. 특징을 (content, length) 두 개로 두고, 두 RM이 각각 다음처럼 학습됐다고 하자 — content는 두 RM 모두 안정적으로 배운 진짜 신호, length는 run마다 방향이 달라지는 스퓨리어스 신호다.

| 가중치                            | RM_A | RM_B | 평균 |
| --------------------------------- | ---- | ---- | ---- |
| $$w_{\text{content}}$$            | 1.0  | 1.0  | 1.0  |
| $$w_{\text{length}}$$(스퓨리어스) | 0.8  | −0.6 | 0.1  |

두 후보 요약에 대해 각 모델의 점수를 계산하면:

| 후보                                    | (content, length) | RM_A | RM_B | ENS | WA  |
| --------------------------------------- | ----------------- | ---- | ---- | --- | --- |
| $$y_{\text{good}}$$ (짧고 알찬 요약)    | (2, 1)            | 2.8  | 1.4  | 2.1 | 2.1 |
| $$y_{\text{bad}}$$ (길게 늘어뜨린 요약) | (1, 5)            | 5.0  | −2.0 | 1.5 | 1.5 |

RM_A만 보면 $$y_{\text{bad}}$$(5.0)가 $$y_{\text{good}}$$(2.8)보다 높다 — 전형적인 길이 hacking이다. RM_B는 정반대로 판단한다. 둘을 평균내면(이 선형 케이스에서는 ENS든 WA든 같은 값) $$y_{\text{good}}$$(2.1)이 $$y_{\text{bad}}$$(1.5)를 이긴다. 여기서 중요한 건 **왜** 뒤집혔는가다 — content 가중치는 두 RM 모두 1.0으로 일관됐지만, length 가중치는 0.8과 −0.6으로 요동쳤고, 평균을 내는 순간 0.1로 거의 지워졌다.

이 예시에서는 length를 골랐지만 핵심은 WARM이 "범인이 length라는 것"을 전혀 몰라도 된다는 점이다. 여러 파인튜닝에서 일관되게 학습되지 않는 방향이면 무엇이든, 그 정체와 무관하게 평균 속에서 옅어진다. $$p_j$$ 대 $$p_j^2$$ 관계를 숫자로 극단화하면 이 효과가 더 뚜렷해진다.

| 특징                                  | 학습될 확률 $$p_j$$ | ENS가 유지하는 가중치($$p_j$$) | WA가 유지하는 가중치($$p_j^2$$) |
| ------------------------------------- | ------------------- | ------------------------------ | ------------------------------- |
| 인과적 특징(모든 run에서 학습)        | 1.0                 | 1.0                            | 1.0                             |
| 스퓨리어스 특징(30%의 run에서만 학습) | 0.3                 | 0.3                            | 0.09                            |

인과적 신호는 ENS와 WA 모두에서 그대로 살아남지만, 스퓨리어스 신호는 WA에서 약 3.3배 더 강하게 억제된다. 이게 논문이 4.3절에서 증명하는 결과의 요지이며, WARM이 축을 지정하지 않고도 방어할 수 있는 이유다.

# Experiments

## 실험 셋업

논문은 TL;DR 요약 벤치마크에서 실험한다.

| 항목                               | 값                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| RM 아키텍처                        | PaLM-XXS + 선형 프로빙 분류기                                                  |
| 통제(control) RM                   | PaLM-XS, $$\mathcal{D}_{ood}$$ 정확도 80.1%                                    |
| RL 정책·가치 모델                  | PaLM-XS                                                                        |
| RL 알고리즘                        | 베이스라인을 포함한 REINFORCE (PPO는 [#16](/blog/2026/ppo/)에서 별도로 다룬다) |
| 선호 라벨                          | Stiennon et al. 데이터 + PaLM-L CoT 라벨링(RLAIF, 오라클로도 사용)             |
| RM 학습 스텝                       | 10k step                                                                       |
| OOD 테스트셋 $$\mathcal{D}_{ood}$$ | 92k pairwise 비교, 여러 PaLM-XS 정책이 생성                                    |
| KL 계수 $$\alpha$$                 | clean 0.003 / corrupt 0.01                                                     |
| Best-of-N                          | PaLM 정책: $$N=8$$, $$D=15{,}000$$ / T5 정책: $$N=1{,}000$$, $$D=1{,}000$$     |

## Best-of-N — 셀렉터로서의 WARM

BoN에서 WARM은 point-wise control reward와 오라클 선호 두 지표 모두에서 ENS($$M=2$$, 계산 비용 때문에 이 크기로 제한)와 개별 RM을 앞선다. 오라클 지표로 보면 **WARM이 고른 요약은 무작위 선택(SFT) 대비 최대 92.5%의 승률**을 기록했고, 반대로 다른 어떤 선택 전략도 WARM $$M=6$$이 고른 요약을 상대로는 50%를 넘지 못했다.

## RL — WARM이 proxy RM일 때

<p align="center"><img src="/assets/post/image/warm-weight-averaged-reward/reward-hacking-mitigation.png" width="75%"></p>

이 그림이 이 논문의 핵심 결과다. x축은 학습 스텝, y축은 control reward(참 선호에 가까운 대리 지표)다. 개별 RM($$\phi_1, \phi_2$$, 옅은 노란색)은 2,000스텝을 전후로 빠르게 무너진다 — 전형적인 reward hacking이다. ENS $$M=2$$(빨간 점선)는 개별 RM보다는 오래 버티지만 여전히 5,000스텝 근처에서 꺾인다. WARM(파란 계열)은 $$M$$이 커질수록 붕괴 시점이 뒤로 밀리고, 절대 reward도 더 높다 — $$M=10$$이 가장 오래 버틴다.

오라클 선호 지표로 정리한 핵심 숫자는 다음과 같다.

| 비교                                                                     | 승률      |
| ------------------------------------------------------------------------ | --------- |
| WARM $$M=6$$(3,500스텝) vs SFT 초기 정책                                 | 99.8%     |
| WARM $$M=6$$(3,500스텝) vs 단일 RM $$\phi_1$$(3,000스텝)으로 학습한 정책 | **79.4%** |
| WARM $$M=6$$(3,500스텝)을 상대로 이긴 다른 정책                          | 없음      |

79.4%라는 숫자가 이 논문의 대표 수치로 꼽히는 이유가 여기 있다 — 단순히 "더 나은 reward를 준다"가 아니라, **그 reward로 실제 RL을 끝까지 돌렸을 때 결과 정책이 사람(오라클)이 보기에도 확실히 더 낫다**는 걸 검증했기 때문이다.

## Ablation — $$M$$과 $$\alpha$$

| $$M$$ | 결과                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2     | ENS $$M=2$$보다는 낫지만 hacking 지연 효과는 제한적                                                                                              |
| 6     | 절대 성능 최고, 이 논문의 기본값. 79.4%/99.8%가 여기서 나온 숫자                                                                                 |
| 10    | hacking을 더 늦게까지 지연시키지만 peak 성능은 $$M=6$$과 같다 — 저자들은 뒤늦게 추가된 $$\phi_7 \sim \phi_{10}$$의 개별 정확도가 낮아서라고 추정 |

KL 계수 $$\alpha$$ ablation에서 흥미로운 점 하나는, **WARM에 최적인 $$\alpha$$가 단일 RM에 최적인 $$\alpha$$보다 작다**는 것이다. $$\alpha$$가 작을수록 정책이 SFT에서 더 멀리(더 큰 KL로) 벗어날 수 있는데, 단일 RM은 그렇게 멀리 보내면 바로 hacking이 시작되니 큰 $$\alpha$$로 묶어둬야 한다. WARM은 hacking 자체를 지연시키므로, 같은 만큼 멀리 보내도 안전하다 — 즉 **KL 페널티가 덜 필요해진다.** 이 관찰이 이 시리즈 5부, 특히 KL 페널티를 정면으로 다루는 [#16 PPO](/blog/2026/ppo/) 글에서 다시 등장할 개념이다.

# Conclusion

한 줄로: **WARM은 같은 사전학습에서 갈라져 나온 여러 RM의 weight를 평균내, 앙상블에 준하는 신뢰성·강건성을 단일 모델의 추론 비용으로 얻는다.** LMC 덕분에 weight 보간이 예측 보간과 비슷하게 작동하고(1차 근사), 동시에 run마다 들쭉날쭉한 스퓨리어스 신호를 예측 앙상블보다 더 강하게 지운다(2차 근사, $$p_j$$ 대 $$p_j^2$$). 그 결과 RL 정책이 단일 RM 대비 79.4% 승률을 기록했다.

한계도 뚜렷하다. 첫째, **같은 사전학습·같은 아키텍처를 공유해야 한다** — 서로 다른 계열의 모델을 섞는 앙상블의 다양성은 WARM이 가져올 수 없다. 둘째, **hacking을 완전히 없애지는 못한다.** 논문 스스로 인정하듯, 만약 $$M$$개의 RM 모두가 똑같이 길이에 의존하도록 학습됐다면(즉 $$p_{\text{length}} \approx 1$$이라면) WARM도 그 편향을 그대로 물려받는다 — WARM이 지우는 건 "run마다 다른" 편향이지, "모두가 공유하는" 편향이 아니다. 셋째, 앙상블과 달리 멤버 간 불일치를 불확실성 신호로 쓸 수 없다.

3부를 전체로 묶으면 이런 그림이 된다. [#10](/blog/2026/reward-model-overoptimization/)이 hacking이 얼마나 심각한지 자로 쟀고, [#11](/blog/2026/rlhf-length-correlations/)이 그 상당 부분의 정체를 밝혔고, [#12 ODIN](/blog/2026/odin-disentangled-reward/)이 그 정체를 알고 있을 때 수술하는 법을, 이 글이 정체를 몰라도 버티는 법을 보여줬다. 두 방어선은 서로 대체재가 아니라 상호보완적이다 — 이미 알려진 축(길이)은 ODIN처럼 직접 잘라내고, 아직 모르는 축은 WARM처럼 평균으로 흐리는 식으로 같이 쓸 수 있다.

그런데 이 글에서 반복해서 등장한 장치가 하나 있다 — KL 페널티다. WARM의 $$\alpha$$ ablation도, 배경으로 다룬 앙상블 논문의 20배 페널티 비교도, 결국 "정책이 SFT에서 얼마나 멀리 벗어나도록 허용할 것인가"라는 같은 질문을 다르게 던진 것이었다. [#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)의 결론에서 이미 예고했듯, distribution shift 문제에 대한 원조 해법은 RM을 온라인으로 계속 갱신하는 것이었지만, LLM 시대의 RLHF는 그 대신 KL 페널티로 정책의 이동 범위 자체를 억제한다. 3부가 "reward를 어떻게 더 안전하게 설계할까"를 다뤘다면, 5부는 그렇게 설계된 reward를 "정책이 어떻게 안전하게 쫓아갈까"로 넘어간다. [#16 PPO](/blog/2026/ppo/)에서 이 KL 페널티가 정확히 어떤 수식으로 구현되는지부터 다시 시작한다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열세 번째 글이다.

**1부. 지형도**

<ol start="1">
  <li><a href="/blog/2026/deep-rl-human-preferences/">Deep RL from Human Preferences (Christiano 2017)</a> — 선호로 보상을 배우는 원형</li>
  <li><a href="/blog/2026/instructgpt/">InstructGPT (Ouyang 2022)</a> — RLHF 3단계 표준 레시피</li>
  <li><a href="/blog/2026/anthropic-hh-rlhf/">HH-RLHF (Bai 2022)</a> — helpful·harmless preference model</li>
</ol>

**2부. 스칼라 RM 해부**

<ol start="4">
  <li><a href="/blog/2026/bradley-terry-rethinking/">Rethinking Bradley-Terry (2024)</a> — reward 변환의 수학적 기반</li>
  <li><a href="/blog/2026/secrets-rlhf-reward-modeling/">Secrets of RLHF II (2024)</a> — 선호 데이터 노이즈와 RM 일반화</li>
  <li><a href="/blog/2026/skywork-reward/">Skywork-Reward (2024)</a> — 데이터 큐레이션이 아키텍처를 이긴다</li>
  <li><a href="/blog/2026/armorm/">ArmoRM (2024)</a> — 다목적 분해와 MoE 게이팅</li>
  <li><a href="/blog/2026/llama2-rlhf/">Llama 2 (2023)</a> — helpfulness·safety RM 분리 프로덕션 레시피</li>
  <li><a href="/blog/2026/rewardbench-2/">RewardBench 2 (2025)</a> — RM을 어떻게 평가할 것인가</li>
</ol>

**3부. Reward Hacking**

<ol start="10">
  <li><a href="/blog/2026/reward-model-overoptimization/">Overoptimization Scaling Laws (2022)</a> — Goodhart의 법칙 정량화</li>
  <li><a href="/blog/2026/rlhf-length-correlations/">Length Correlations in RLHF (2023)</a> — 성능 향상의 얼마가 길이인가</li>
  <li><a href="/blog/2026/odin-disentangled-reward/">ODIN (2024)</a> — 길이를 reward에서 분리</li>
  <li><strong>(현재 글)</strong> WARM (2024) — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="14">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
</ol>

**5부. reward를 정책으로**

<ol start="16">
  <li><a href="/blog/2026/ppo/">PPO (2017)</a> — clipped surrogate objective</li>
  <li><a href="/blog/2026/secrets-rlhf-ppo/">Secrets of RLHF I (2023)</a> — PPO 학습 안정화 트릭</li>
  <li><a href="/blog/2026/grpo-deepseekmath/">GRPO / DeepSeekMath (2024)</a> — value network를 버리다</li>
  <li><a href="/blog/2026/rloo-back-to-basics/">RLOO (2024)</a> — REINFORCE로 충분한가</li>
  <li><a href="/blog/2026/dpo/">DPO (2023)</a> — reward를 없애면 어떻게 되는가</li>
  <li><a href="/blog/2026/simpo/">SimPO (2024)</a> — reference-free + 길이 정규화</li>
  <li><a href="/blog/2026/kto/">KTO (2024)</a> — 선호 쌍 없이 이진 신호만으로</li>
  <li><a href="/blog/2026/gspo/">GSPO (2025)</a> — importance ratio를 시퀀스 단위로</li>
  <li><a href="/blog/2026/dapo/">DAPO (2025)</a> — 신호 없는 프롬프트를 버린다</li>
</ol>

**6부. Process & Verifiable Reward**

<ol start="25">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="28">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="33">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="38">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 39편으로 구성된다.

# 참고 문헌

- Ramé et al., 2024. [WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/abs/2401.12187). ICML 2024, PMLR 235:42048-42073.
- [ar5iv: WARM (HTML rendering)](https://ar5iv.labs.arxiv.org/html/2401.12187) — 본문 수식·그림 원본.
- [ICML 2024 Proceedings page](https://proceedings.mlr.press/v235/rame24a.html) — 게재본, 소속(Google DeepMind) 확인.
- Coste, Anwar, Kirk, Krueger, 2023. [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743). ICLR 2024.
- [ar5iv: Reward Model Ensembles (HTML rendering)](https://ar5iv.labs.arxiv.org/html/2310.02743) — WCO/UWO 수식 원본.
- Gao, Schulman, Hilton, 2022. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). (이 시리즈 #10, Coste et al.이 그대로 채택한 synthetic gold-RM 셋업의 출처)
- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS 2017. (이 시리즈 #1, 앙상블·부트스트랩 RM의 최초 사용례)
- Wortsman et al., 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. ICML 2022. (WARM이 따르는 weight averaging 계열 선행 연구)
- Ouyang et al., 2022. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). (InstructGPT, 라벨러 일치도 72.6% 수치 출처)
