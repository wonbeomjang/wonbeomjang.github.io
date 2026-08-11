---
layout: post
title: "Rethinking Bradley-Terry: 왜 이 식으로 reward를 만드는가"
date: 2026-08-11 09:04:00 +0900
description: "RLHF Reward 설계 시리즈 #4 — BT 모델의 이론적 근거와 order consistency, 그리고 대안 목적함수"
categories: [paper]
tags: [rlhf, reward-model, bradley-terry, theory, alignment, paper]
giscus_comments: true
related_posts: true
---

> [Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives](https://arxiv.org/abs/2411.04991) (Sun et al., University of Cambridge, ICLR 2025)

# Introduction

[1편](/blog/2026/deep-rl-human-preferences/)과 [2편](/blog/2026/instructgpt/)은 둘 다 같은 식을 아무 의심 없이 썼다. 두 궤적(또는 두 응답)의 예측 보상을 지수함수에 태우고 소프트맥스를 취해 "사람이 어느 쪽을 고를 확률"을 만드는 식이다.

$$\hat P[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat r(\sigma^1)}{\exp \sum \hat r(\sigma^1) + \exp \sum \hat r(\sigma^2)}$$

이 식은 Bradley-Terry(BT) 모델(1952)이다. 그런데 이 식은 원래 **무엇을 위해 만들어졌을까.** Ralph Bradley와 Milton Terry가 1952년에 이 식을 제안했을 때 문제는 "리그전에서 여러 시즌 동안 반복 대결한 팀들의 실력을 매기는 법"이었다. 한 팀은 시즌 내내 수십 번 경기를 뛴다. 그런데 LLM reward modeling에서는 사정이 다르다. 프롬프트 $$x$$에 대한 응답 $$y_1, y_2$$ 한 쌍은 **딱 한 번** 비교되고 그걸로 끝이다. 같은 응답 쌍이 다시 비교되는 일은 거의 없다. 축구 리그 순위표를 매기던 도구를, 평생 단 한 번 마주친 두 사람의 매력을 비교하는 데 그대로 갖다 쓰는 셈이다. 이 전용(轉用)이 왜 아무 문제 없이 성립하는지는 지금까지 그 누구도 정식으로 따져보지 않았다.

Hao Sun(University of Cambridge), Yunyi Shen(MIT), Jean-Francois Ton(ByteDance Research)의 이 논문은 정확히 이 질문을 붙잡는다. 논문은 세 가지 질문을 던진다.

1. **플레이어 수가 비교 수보다 많은 상황(LLM 정렬의 전형적 상황)에서 BT 모델을 쓰는 게 이론적으로 정당한가?** 정당하다면 무엇이 그 성공을 뒷받침하는가?
2. **BT 말고 다른 선택지는 없는가?**
3. **관례적으로 같은 프롬프트의 응답끼리만 비교하는데, 다른 프롬프트의 응답끼리 비교하면 더 나은가?**

결론부터 미리 적는다. (1) BT는 임베딩 기반 신경망으로 구현할 때 실제로 수렴한다는 것을 이 논문이 최초로 증명했다 — 이론적으로 근거가 있다. 하지만 (2) 그 근거는 "BT만이 유일한 답"이라는 뜻이 아니다. reward 모델링에 정말 필요한 조건은 **order consistency**(순서 일관성)라는 훨씬 느슨한 성질이고, BT는 그 조건을 만족하는 여러 선택지 중 하나일 뿐이다. 저자들은 order consistency를 만족하는 **classification 기반 대안**을 제시하고, 6개 base LLM·2개 데이터셋·12,000개 이상의 실험 설정에서 이 대안이 BT를 이긴다는 걸 보인다. (3) 같은 프롬프트끼리만 비교하는 관행도 근거가 약하다 — 프롬프트를 섞어 비교하는 쪽이 이론적으로도 실증적으로도 낫다.

이 논문은 ICLR 2025에 **Oral**로 채택됐다(카메라레디 제목은 "Rethinking Reward Modeling in Preference-based Large Language Model Alignment"로 arXiv 버전과 살짝 다르다). 이 시리즈에서 이 글이 서 있는 위치는 명확하다 — 1편·2편이 전제로 깔았던 BT 손실의 **근거 자체**를 캐묻는 자리다.

# Background

## BT 모델과 Luce-Shephard 선택 규칙

BT 모델은 Luce-Shephard 선택 규칙의 특수 케이스다. 두 선택지 $$i, j$$ 중 $$i$$를 고를 확률이 각 선택지의 효용 $$u(\cdot)$$에 비례한다고 가정하면,

$$P(i \succ j) = \frac{u(i)}{u(i) + u(j)} = \frac{\exp(r(i))}{\exp(r(i)) + \exp(r(j))} = \mathrm{softmax}(r(i), r(j))$$

가 나온다. 여기서 $$r(\cdot) = \log u(\cdot)$$는 로그 효용이다. 이 식 자체는 낯설지 않다. 1편에서 이미 다뤘다. 이번 글이 파고드는 건 "이 식을 실제로 어떻게 쓰고 있는가"다.

## 두 가지 다른 용도 — 파라미터 추정 vs 예측

논문이 지적하는 첫 번째 사실은, **BT 모델의 고전적 용도와 reward modeling에서의 용도가 근본적으로 다르다**는 것이다. Chatbot Arena(Chiang et al., 2024)가 전형적인 고전적 용도다. 130개의 LLM을 플레이어로 놓고, 사람 투표를 경기 결과로 취급해 170만 건 이상의 비교를 모았다 — 모델 한 개당 평균 26,000경기다. 목표는 각 모델에 스칼라 점수 $$r(\cdot)$$ 하나를 매기는 것, 즉 **직접 파라미터 추정**이다. $$N$$명의 진짜 실력을 무작위 쌍대 비교로 추정하려면 이론적 하한이 $$\mathcal{O}(N\log N)$$ 번의 비교다(직관: 정렬 알고리즘 퀵소트가 평균 $$N\log N$$ 번의 비교로 끝나는 것과 같은 이유). 알려진 최선의 방법은 $$\mathcal{O}(N\log^3 N)$$ 비교면 충분하다(Han et al., 2020).

reward modeling은 이 조건을 하나도 만족하지 않는다. $$N$$개의 프롬프트-응답 쌍이 있다면 비교는 겨우 $$N/2$$건 — 하한의 로그 배수는커녕 선형에도 못 미친다. 게다가 목표도 다르다. 학습에 쓰인 응답 쌍의 점수를 아는 것으로 끝나지 않고, **한 번도 본 적 없는 새 응답에 점수를 매겨야(예측)** 한다. 논문의 Table 1을 정리하면 다음과 같다.

| 항목           | LLM Arena                          | BT Reward Modeling         |
| -------------- | ---------------------------------- | -------------------------- |
| 목표           | 직접 파라미터 추정                 | 모델 파라미터화(함수 근사) |
| 예측 필요 여부 | 불필요                             | 필요                       |
| 비교 수        | 충분(N=130에 170만 건)             | 매우 희소(N/2건)           |
| BT 모델 종류   | 고전적 BT                          | BT-회귀(BT-regression)     |
| 요구조건       | 최소 $$\mathcal{O}(N\log N)$$ 비교 | 공변량(covariate)이 필요   |

비유하자면 이렇다. 챗봇 아레나는 시즌 내내 같은 팀들이 반복해서 맞붙는 **프로 축구 리그**다. 순위표는 실제 대결 결과가 충분히 쌓이면 저절로 안정된다. 반면 LLM reward modeling은 **평생 단 한 번 만난 소개팅 상대의 매력도**를 그 한 번의 만남만으로 추론하는 것과 같다. 재대결이 없으니 "그 사람 자체의 점수"를 직접 추정할 방법이 없다. 대신 쓸 수 있는 건 그 사람의 **특징(옷차림, 말투, 학력 같은 공변량)**뿐이고, 그 특징으로부터 매력도를 예측하는 함수를 배워야 한다. LLM에서 이 공변량 역할을 하는 게 바로 문장 임베딩이다.

## 사람 라벨은 어떤 확률 모델에서 나오는가

BT 손실을 annotation에 쓰려면 "사람의 선택이 왜 확률적인가"를 설명하는 가정이 필요하다. 논문은 이를 명시적으로 분해한다. 오라클 효용 $$r_{x,y}$$가 존재하고 결정론적이라 가정하고(가정 1), annotator $$A$$의 실제 판단은 이 오라클 값에 annotator 개인의 편향 $$b(x,y,A)$$가 더해진 값을 비교해서 나온다고 본다(가정 2).

$$\mathbb{1}(y_1 \succ y_2 \mid x, A) = \mathbb{1}\bigl(r_{x,y_1} + b(x,y_1,A) > r_{x,y_2} + b(x,y_2,A)\bigr)$$

이제 편향 차이 $$b(x,y_1,A) - b(x,y_2,A)$$가 어떤 분포를 따르는지가 관건이다. 이 분포를 **표준 로지스틱 분포**로 가정하면(가정 3) BT 식이 정확히 유도된다. 반대로 **표준 정규분포**로 가정하면(가정 4) 다른 식이 나온다.

$$P(y_1 \succ y_2 \mid x) = \Phi(r_{x,y_1} - r_{x,y_2})$$

여기서 $$\Phi$$는 표준정규분포의 누적분포함수다. 이건 BT가 아니라 **Thurstonian 모델**(Thurstone, 1927)이다.

**토이 예제**로 둘의 차이를 체감해보자. 두 응답의 참 보상 차이가 $$r_{x,y_1} - r_{x,y_2} = 1.0$$이라 하자.

| 노이즈 가정                | 식              | 계산             | $$P(y_1 \succ y_2)$$ |
| -------------------------- | --------------- | ---------------- | -------------------- |
| 로지스틱 차이(BT)          | $$\sigma(1.0)$$ | $$1/(1+e^{-1})$$ | 약 0.731             |
| 가우시안 차이(Thurstonian) | $$\Phi(1.0)$$   | 표준정규 CDF     | 약 0.841             |

같은 보상 차이 1.0에 대해 두 모델은 서로 다른 확률을 내놓는다. **BT는 유일한 정답이 아니라, 사람의 판단 노이즈에 대한 하나의 분포 가정("로지스틱 차이")을 골랐을 때 나오는 결과일 뿐이다.** 이 사실이 뒤에서 다룰 "BT는 선택이다"라는 주장의 첫 단서다.

# Method

## BT-회귀: 희소 비교를 우회하는 법

앞서 말했듯 reward modeling에서는 각 프롬프트-응답 쌍이 독립적인 플레이어처럼 반복 대결하지 않는다. Springall(1973)이 제안한 방법은 $$r(\cdot)$$을 공변량의 함수로 놓는 것이다. LLM에서는 프롬프트-응답 쌍 $$(x,y)$$의 문장 임베딩 $$\Psi(x,y) \in [0,1]^d$$가 이 공변량 역할을 한다. 실전에서는 MLP로 $$\Psi(x,y) \mapsto \hat r(\Psi(x,y))$$를 근사한다.

이렇게 하면 BT 모델 학습은 사실상 **비대칭(anti-symmetric) 구조를 가진 이진 분류 문제**로 환원된다. 두 임베딩 $$\Psi_1, \Psi_2$$의 예측 보상 벡터에 소프트맥스를 취하면 그게 곧 두 클래스의 예측 확률이고, 여기에 교차 엔트로피 손실을 쓰면 된다. 문제는 이렇게 MLP와 BT 손실을 결합한 방식이 **왜 잘 작동하는지 이론적으로 아무도 증명한 적이 없었다**는 것이다.

## MLP 기반 BT reward model의 수렴률

이 논문의 첫 번째 핵심 기여가 여기서 나온다. Bos and Schmidt-Hieber(2022)의 truncated KL risk 프레임워크를 빌려, 저자들은 embedding 기반 MLP reward model이 실제로 참 확률(그리고 참 보상 차이)에 수렴한다는 것을 증명한다.

$$\phi_n := 2^{\frac{(1+\alpha)\beta + (3+\alpha)d}{(1+\alpha)\beta + d}} \, n^{-\frac{(1+\alpha)\beta}{(1+\alpha)\beta + d}}$$

기호를 하나씩 풀면:

- $$n$$: 학습에 쓰인 선호 비교 데이터 개수.
- $$\beta$$: 참 reward 함수의 매끄러움(Hölder smoothness) — 클수록 reward가 완만하고 예측하기 쉬운 함수라는 뜻.
- $$\alpha$$: 선호확률이 0 또는 1 같은 극단으로 얼마나 완만하게 다가가는지를 규정하는 마진 조건 — 직관적으로는 "애매한 비교가 얼마나 자주 나오는가"를 통제하는 상수로 이해하면 된다.
- $$d$$: 임베딩 공간의 차원.
- $$L$$: MLP의 깊이.

**Theorem 6**(정리, informal)은 적당한 매끄러움·정규성 가정 아래, MLP reward model의 truncated KL risk가

$$R_B(\bm p_0, \hat{\bm p}) \le C' B \phi_n L \log^2(n) \to 0$$

으로 0에 수렴함을 보인다. 즉 데이터가 늘어날수록($$n \to \infty$$) 예측 확률이 참 확률에 다가간다는 것이 정식으로 증명됐다. 지수 $$-\frac{(1+\alpha)\beta}{(1+\alpha)\beta + d}$$를 뜯어보면 이 결과는 전형적인 비모수 회귀의 수렴률과 같은 모양이다 — **reward가 매끄러울수록($$\beta$$↑), 임베딩 차원이 낮을수록($$d$$↓) 더 빨리 수렴한다.** 임베딩 차원이 지수 분모에 그대로 들어가는 것도 낯익은 차원의 저주(curse of dimensionality) 패턴이다.

## 확률 오차가 reward 오차로 번질 때 생기는 함정

**Corollary 7**은 이 확률 수렴 결과를 실제로 쓰고 싶은 보상 차이 오차로 옮긴다.

$$\lvert r(\Psi_1) - r(\Psi_2) - (\hat r(\Psi_1) - \hat r(\Psi_2)) \rvert \lesssim \frac{\lvert \sqrt{p_0} + \sqrt{\hat p} \rvert}{\tilde p (1 - \tilde p)} \sqrt{\phi_n L} \log(n) \to 0$$

여기서 $$\tilde p$$는 $$p_0$$(참 확률)와 $$\hat p$$(예측 확률) 사이의 어떤 값이다. 분모 $$\tilde p(1-\tilde p)$$가 핵심이다 — 이 항은 $$\tilde p$$가 0.5에 가까울 때 최대이고, 0이나 1에 가까울수록 급격히 작아져 전체 식을 폭발시킨다.

**토이 예제**로 이 폭발이 얼마나 심한지 보자. $$1/[\tilde p(1-\tilde p)]$$ 값을 $$\tilde p$$별로 계산하면:

| $$\tilde p$$        | $$\tilde p(1-\tilde p)$$ | $$1/[\tilde p(1-\tilde p)]$$ |
| ------------------- | ------------------------ | ---------------------------- |
| 0.50 (거의 동률)    | 0.250                    | 4                            |
| 0.90 (꽤 확실)      | 0.090                    | 약 11                        |
| 0.99 (거의 확실)    | 0.0099                   | 약 101                       |
| 0.999 (사실상 확정) | 0.000999                 | 약 1,001                     |

동률에 가까운 비교(4배)와 사실상 결과가 정해진 비교(1,001배) 사이에 **250배** 차이가 난다. 결론은 명확하다. **오차 보장이 의미를 가지려면 비교하는 두 응답의 보상이 서로 가까워야 한다.** 압도적으로 한쪽이 나은 쌍을 비교 데이터로 아무리 많이 넣어도, 그 학습 신호가 실제 reward 오차를 줄인다는 보장은 약해진다. 이건 뒤에서 다룰 cross-prompt 비교의 한계를 이해하는 데도 쓰인다.

## Order Consistency: reward 모델링에 정말 필요한 조건

여기서부터 논문의 두 번째 핵심 주장이 시작된다. BT 손실은 사람의 선택 **확률**을 정확히 맞히는 걸 목표로 한다. 그런데 downstream에서 reward model을 쓰는 방식(best-of-N 샘플링, PPO 등)은 확률값 자체가 아니라 **"어느 응답이 더 나은가"라는 순서**만 있으면 충분하다. 순서가 같다면 $$\hat r = h(r)$$처럼 단조증가함수 $$h$$로 재조정된 reward라도 최적화 결과는 똑같다.

**Definition 8(Order Consistency)**은 이 요구조건을 정식화한다. 서로 다른 두 프롬프트-응답 쌍에 대해,

$$(\hat r(x_1,y_1) - \hat r(x_2,y_2)) \cdot (r(x_1,y_1) - r(x_2,y_2)) > 0$$

이면 $$\hat r$$은 order consistent다. 그리고 사람 라벨 자체도 노이즈가 있다는 걸 반영해, 학습 가능한 관측 손실 $$\mathcal{L}_{oc}$$을 최소화하면 참 오라클 순서와도 높은 확률로 일치한다는 게 **Proposition 9**의 내용이다.

비유하자면 이렇다. 대학 입시에서 중요한 건 "정확히 몇 점을 맞았는가"가 아니라 "합격선 위인가 아래인가, 몇 등인가"다. 원점수를 표준점수로, 표준점수를 등급으로 바꿔도 등수만 유지되면 입시 결과는 똑같다. reward model도 마찬가지다 — **확률을 정밀하게 맞히는 것과 순서를 맞히는 것은 다른 요구조건이고, 후자만으로 충분하다.**

## BT는 정답이 아니라 하나의 선택이다

BT 모델은 order consistency를 만족하는 방식 중 하나다. $$h=1$$(첫 응답 선호)일 확률을 $$\sigma(\hat r_{\text{BT}}(x_1,y_1) - \hat r_{\text{BT}}(x_2,y_2))$$로 모델링하고 교차 엔트로피로 학습하면,

$$\mathcal{L}_{\text{BT}} = \mathbb{E}\left[\mathbb{1}_{h=1}\sigma(\hat r_{\text{BT}}^1 - \hat r_{\text{BT}}^2) + \mathbb{1}_{h=-1}(1 - \sigma(\hat r_{\text{BT}}^1 - \hat r_{\text{BT}}^2))\right]$$

이 손실은 **비교 순서를 뒤집으면 예측도 정확히 뒤집히도록** 강제하는 비대칭(anti-symmetric) 구조를 갖는다. 이 구조 때문에 BT는 반드시 Siamese 네트워크(같은 파라미터로 두 입력을 각각 통과시키고 차이를 비교하는 구조)로 구현해야 하고, 사실상 MLP만 백본으로 쓸 수 있다.

## 비대칭 제약을 느슨하게 풀면 — Classification 대안

order consistency만 원한다면 이 anti-symmetry가 정말 필수일까? 저자들은 $$\hat H := (\hat H_1, \hat H_2)$$처럼 두 응답 각각에 대해 독립적으로 $$+1/-1$$을 예측하는 모델을 생각한다. BT는 $$\hat H_1 = -\hat H_2$$라는 강한 제약을 거는데, 데이터가 충분하면 이 제약을 명시적으로 걸지 않아도 $$\hat H_1 \approx h$$, $$\hat H_2 \approx -h$$가 암묵적으로 학습될 수 있다는 것이다. 이때 union bound로

$$\mathcal{L}_{\text{oc}} \le \mathcal{L}_{\text{clf}} := \mathbb{E}(h = \hat H_{\text{clf}}(x_1,y_1)) + \mathbb{E}(-h = \hat H_{\text{clf}}(x_2,y_2))$$

가 성립한다 — 즉 $$\mathcal{L}_{\text{clf}}$$는 order consistency 손실의 **upper bound**다. $$\mathcal{L}_{\text{clf}}$$를 최소화하면 order consistency도 함께 줄어든다는 뜻이므로, 굳이 페어를 쌍으로 묶어 anti-symmetric하게 학습할 필요 없이 **각 프롬프트-응답 쌍을 독립적인 분류 대상으로 놓고 기성 이진 분류기**(MLP든 LightGBM이든)를 그대로 쓸 수 있다. 학습이 끝나면 분류기의 로짓(logit)을 reward 프록시로 쓴다.

|                            | BT (BT-MLP)         | Classification (CLF)              |
| -------------------------- | ------------------- | --------------------------------- |
| 강제하는 구조              | anti-symmetry(엄격) | 없음(union bound로 근사)          |
| 필요한 데이터 형태         | 페어(Siamese 입력)  | 개별 프롬프트-응답 (분류 라벨)    |
| 쓸 수 있는 모델            | 사실상 MLP만        | MLP, LightGBM 등 기성 분류기 전부 |
| order consistency와의 관계 | 정확히 만족         | 상한(upper bound)으로 만족        |

두 방식 다 order consistency라는 같은 목표의 서로 다른 구현이다. BT가 "필연"이 아니라는 이 논문 전체의 주장이 이 비교표 한 줄로 요약된다.

## Cross-prompt 비교가 이론적으로 더 낫다

관례적으로 선호 annotation은 같은 프롬프트에서 나온 두 응답끼리만 비교한다. 저자들은 이 관례에도 근거가 약하다고 지적한다. 두 응답의 효용이 가우시안 $$\mathcal{N}(\mu_x, \sigma_x^2)$$을 따른다고 하면, annotation quality를 "노이즈가 있는 상황에서 평균적으로 올바른 순서를 맞힐 확률" $$\mathcal{Q}_{\text{pair}}(x) = \mathbb{E}[\sigma(\beta \lvert r(x,y_1) - r(x,y_2) \rvert)]$$로 정의할 수 있다.

**토이 예제**로 annotator 판별력 $$\beta$$와 응답 다양성 $$\sigma_x$$의 곱 $$\beta^2\sigma_x^2$$을 바꿔가며 계산한 결과가 논문에 실려 있다.

| $$\beta^2 \sigma_x^2$$ | $$\mathcal{Q}_{\text{pair}}$$(정답률) |
| ---------------------- | ------------------------------------- |
| 1                      | 약 0.6749                             |
| 2                      | 약 0.7251                             |
| 4                      | 약 0.7781                             |
| 10                     | 약 0.8428                             |

같은 annotator라도(같은 $$\beta$$) 응답들의 분산 $$\sigma_x^2$$이 클수록 정답률이 67.5%에서 84.3%까지 올라간다. 즉 **annotation 품질은 annotator의 실력과 응답 간 편차, 둘 다에 달려 있다.** 그런데 같은 프롬프트에서 나온 두 응답은 같은 LLM이 만든 것이라 서로 비슷하기 쉽다 — 편차가 작다. 반면 서로 다른 프롬프트의 응답을 무작위로 짝지으면 편차가 구조적으로 커진다. **Proposition 10**과 **Theorem 11**은 이를 정식화해, unimodal하고 대칭인 효용 분포에서는 cross-prompt 비교가 same-prompt 비교보다 기대 reward 차이가 항상 크거나 같다는 것을 증명한다.

$$\mathbb{E}_x \mathbb{E}_{y_1,y_2 \mid x}[\lvert r_{x,y_1} - r_{x,y_2} \rvert] \le \mathbb{E}_{x_1,x_2} \mathbb{E}_{y_1 \mid x_1, y_2 \mid x_2}[\lvert r_{x_1,y_1} - r_{x_2,y_2} \rvert]$$

비유하자면 이렇다. 같은 반 1등과 2등의 시험 점수를 비교하면 종이 한 장 차이라 우열을 가리기 어렵다. 반면 전교생 중 아무나 두 명을 무작위로 뽑아 비교하면 점수 차가 훨씬 크게 벌어지기 마련이라 우열이 뚜렷하다. Cross-prompt 비교는 "무작위로 아무나 뽑아 비교하기"에 해당한다.

# Experiments

## 실험 설계 — 12,000개 이상의 조합

저자들은 재현성·통제 가능성·계산 효율을 우선해 실험을 설계했다. PPO 대신 **Best-of-N(BoN) 샘플링**으로 reward model을 평가한다 — PPO는 설정마다 LLM을 새로 파인튜닝해야 해서 12,000개 조합을 전부 도는 건 계산상 불가능하기 때문이다. 실험 규모는 다음 여섯 개 축의 곱이다.

| 축                         | 값                                                                    |
| -------------------------- | --------------------------------------------------------------------- |
| base LLM                   | Gemma2b, Gemma7b, LLaMA3-8b(및 각 SFT 버전) 총 6개                    |
| 데이터셋                   | Anthropic-Harmless, Anthropic-Helpful                                 |
| 응답 샘플링 방법           | 3가지                                                                 |
| annotation 노이즈 수준     | 6단계($$\beta \in \{0.5, 0.7, 1.0, 3.0, 5.0, 10.0\}$$, 오답률 5%~38%) |
| reward model 구현체        | BT-MLP, CLF-MLP, CLF-LGB(LightGBM) 3종                                |
| annotation 가용량 시나리오 | 4단계(5,000 / 10,000 / 20,000 / 40,000건)                             |
| random seed                | 5개                                                                   |

이 조합을 전부 곱하면 약 12,960건이고, 논문은 "12,000개 이상의 실험 설정"으로 보고한다.

## BT vs Classification — 정면 대결

<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig1_harmless.png" width="85%"></p>
<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig1_helpful.png" width="85%"></p>

BoN N=500에서 base model 대비 golden reward 개선폭을 6개 base 모델 × 2개 데이터셋에서 비교한 결과다(논문 Figure 1). Harmless에서는 BT-MLP와 CLF 계열이 대체로 비슷하거나 CLF가 근소 우위였지만(예: LLaMA3-8b에서 BT 약 0.10 vs CLF 약 0.18~0.20), Helpful에서는 차이가 훨씬 극적이다.

| Base 모델(Helpful) | BT-MLP  | CLF-MLP | CLF-LGB |
| ------------------ | ------- | ------- | ------- |
| Gemma2b            | 약 1.77 | 약 2.98 | 약 2.87 |
| Gemma2b-SFT        | 약 0.13 | 약 1.51 | 약 1.75 |
| Gemma7b            | 약 2.08 | 약 3.17 | 약 3.03 |
| Gemma7b-SFT        | 약 0.48 | 약 1.86 | 약 2.01 |

Gemma2b-SFT에서는 BT가 사실상 붕괴(약 0.13)한 반면 classification 계열은 10배 넘는 개선(1.51~1.75)을 보였다. 저자들의 결론은 명확하다 — **classification reward model이 BT보다 일반적으로 더 나은 성능을 내면서, MLP에 갇히지 않고 LightGBM 같은 기성 분류기도 자유롭게 쓸 수 있다는 유연성까지 더 갖췄다.**

## Annotation 품질·양이 바뀌면 순위가 바뀐다

<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig3_quantity.png" width="95%"></p>

annotation 개수를 5,000건에서 40,000건까지 늘려가며 본 결과(논문 Figure 3)다. classification 계열이 대체로 BT보다 높거나 같은 golden reward를 유지하며, 데이터가 늘어날수록 개선폭도 더 안정적으로 커진다.

annotation 품질(노이즈 수준 $$\beta$$)을 바꾼 실험에서는 흥미로운 교차가 나온다.

| 조건                                           | 우세한 방식                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| 노이즈가 낮음(오답률 10% 미만, $$\beta$$가 큼) | BT가 classification보다 근소 우세                    |
| 노이즈가 높음(오답률 10% 이상)                 | classification이 BT보다 견고 — 성능 하락폭이 더 작음 |
| annotation 개수 증가                           | classification이 일관되게 우세, 개선폭도 더 안정적   |

즉 **정답률이 아주 높은 소량의 고품질 annotation만 있다면 BT도 충분히 경쟁력 있지만, 실전처럼 노이즈가 섞이고 대량 annotation을 다루는 상황에서는 classification이 더 안전한 선택**이라는 것이다.

## Cross-prompt 비교의 실증 효과

이론이 예측한 대로, cross-prompt annotation은 same-prompt annotation보다 일관되게 BoN 성능을 높였다(2개 데이터셋 × 6개 base 모델 × 3개 reward model 구현체 전체에서). 저자들은 이 효과의 원인을 파고들기 위해 두 개의 합성 세팅을 추가로 설계했다.

| 세팅                   | 응답 쌍 구성                                                                        | cross-prompt 효과                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Similar Comparison     | 한 프롬프트당 생성된 10개 응답 중 golden reward 순위 중간 2개를 짝지음(다양성 최소) | same-prompt 비교는 유용한 reward model을 거의 못 만듦 — cross-prompt가 압도적으로 우세 |
| Diversified Comparison | 같은 10개 응답 중 최상위·최하위를 짝지음(다양성 최대)                               | cross-prompt의 이점이 크게 줄어듦(부정적 영향은 없음)                                  |

그리고 pairwise annotation의 평균 절대 보상 차이(응답 간 다양성의 대리 지표)와 cross-prompt 개선폭 사이에는 강한 상관관계가 있었다(논문 Figure 6). 특히 무작위로 응답 두 개를 뽑는 실제 상황(Random)의 평균 보상 차이가 인위적으로 다양성을 낮춘 Similar 세팅과 비슷한 수준이라는 관찰이 중요하다 — **실전에서 같은 프롬프트로 뽑은 두 응답은 생각보다 서로 비슷해서, cross-prompt 비교를 적용할 유인이 실제로도 크다**는 뜻이다.

# Conclusion

한 줄로 요약하면: **BT 모델은 임베딩 기반 신경망으로 구현할 때 실제로 참 보상에 수렴한다는 것이 이 논문에서 처음 증명됐지만, 그 증명이 "BT가 유일한 답"이라는 뜻은 아니다. reward 모델링에 정말 필요한 조건은 order consistency뿐이고, 그 조건을 만족하는 classification 기반 대안이 12,000개 이상의 실험에서 오히려 BT를 능가했다.**

정리하면,

1. **이론**: BT 모델을 LLM reward modeling에 쓰는 관행은 고전적 BT 세팅(반복 대결, 파라미터 직접 추정)과 다르다(희소 비교, 새 데이터 예측). 그럼에도 임베딩 기반 MLP-BT reward model이 참 확률·참 보상 차이에 수렴한다는 것을 truncated KL risk bound로 증명했다(Theorem 6, Corollary 7).
2. **대안**: reward 모델링의 진짜 목표는 order consistency이며, BT의 anti-symmetry 제약은 그걸 만족하는 여러 방법 중 하나일 뿐이다. anti-symmetry를 느슨하게 풀면 기성 이진 분류기(MLP, LightGBM)를 그대로 쓰는 classification reward model을 얻는다 — 이 손실이 order consistency 손실의 upper bound라는 것도 증명됐다(Proposition 9, Eq. 22).
3. **annotation 설계**: 같은 프롬프트끼리만 비교하는 관행에도 이론적 근거가 없다. cross-prompt 비교가 기대 보상 차이를 구조적으로 키워 annotation 품질을 높인다(Theorem 11)는 것이 이론과 실증 양쪽에서 확인됐다.

이 논문의 결론 — **BT는 필연이 아니라 하나의 선택지이며, "필요한 건 order consistency뿐"** — 은 이 시리즈의 다음 두 흐름을 정당화하는 근거가 된다. [7편 ArmoRM](/blog/2026/armorm/)이 스칼라 하나 대신 다목적 reward로 분해하는 것도, [6부의 GenRM](/blog/2026/generative-verifiers/)이 아예 reward를 확률 스칼라가 아니라 텍스트 생성으로 바꾸는 것도, 결국 "reward 모델링이 반드시 BT-스타일 스칼라 확률 모델일 필요는 없다"는 이 글의 결론 위에서 성립한다. 다음 글([5편 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/))은 이 order-consistent 목적함수들이 실제 노이즈 섞인 선호 데이터 앞에서 어떻게 무너지고 일반화되는지를 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 네 번째 글이다.

**1부. 지형도**

1. [Deep RL from Human Preferences (Christiano 2017)](/blog/2026/deep-rl-human-preferences/) — 선호로 보상을 배우는 원형
2. [InstructGPT (Ouyang 2022)](/blog/2026/instructgpt/) — RLHF 3단계 표준 레시피
3. [HH-RLHF (Bai 2022)](/blog/2026/anthropic-hh-rlhf/) — helpful·harmless preference model

**2부. 스칼라 RM 해부**

4. **(현재 글)** Rethinking Bradley-Terry (2024) — reward 변환의 수학적 기반
5. [Secrets of RLHF II (2024)](/blog/2026/secrets-rlhf-reward-modeling/) — 선호 데이터 노이즈와 RM 일반화
6. [Skywork-Reward (2024)](/blog/2026/skywork-reward/) — 데이터 큐레이션이 아키텍처를 이긴다
7. [ArmoRM (2024)](/blog/2026/armorm/) — 다목적 분해와 MoE 게이팅
8. [Llama 2 (2023)](/blog/2026/llama2-rlhf/) — helpfulness·safety RM 분리 프로덕션 레시피
9. [RewardBench 2 (2025)](/blog/2026/rewardbench-2/) — RM을 어떻게 평가할 것인가

**3부. Reward Hacking**

10. [Overoptimization Scaling Laws (2022)](/blog/2026/reward-model-overoptimization/) — Goodhart의 법칙 정량화
11. [Length Correlations in RLHF (2023)](/blog/2026/rlhf-length-correlations/) — 성능 향상의 얼마가 길이인가
12. [ODIN (2024)](/blog/2026/odin-disentangled-reward/) — 길이를 reward에서 분리
13. [WARM (2024)](/blog/2026/warm-weight-averaged-reward/) — weight averaging으로 hacking 방어

**4부. reward를 정책으로**

14. [PPO (2017)](/blog/2026/ppo/) — clipped surrogate objective
15. [Secrets of RLHF I (2023)](/blog/2026/secrets-rlhf-ppo/) — PPO 학습 안정화 트릭
16. [GRPO / DeepSeekMath (2024)](/blog/2026/grpo-deepseekmath/) — value network를 버리다
17. [RLOO (2024)](/blog/2026/rloo-back-to-basics/) — REINFORCE로 충분한가
18. [DPO (2023)](/blog/2026/dpo/) — reward를 없애면 어떻게 되는가

**5부. Process & Verifiable Reward**

19. [Let's Verify Step by Step (2023)](/blog/2026/lets-verify-step-by-step/) — 과정 감독이 결과 감독을 이긴다
20. [Math-Shepherd (2023)](/blog/2026/math-shepherd/) — 사람 라벨 없는 PRM
21. [DeepSeek-R1 (2025)](/blog/2026/deepseek-r1/) — RLVR, 규칙이 reward가 될 때

**6부. Generative Reward Model**

22. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
23. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
24. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling
25. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
26. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

# 참고 문헌

- Sun, Shen, Ton, 2024/2025. [Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives](https://arxiv.org/abs/2411.04991). arXiv:2411.04991 (ICLR 2025, Oral).
- [ar5iv/arXiv HTML: Rethinking Bradley-Terry Models...](https://arxiv.org/html/2411.04991v2) — 본문 수식·그림 원본.
- [ICLR 2025 Proceedings: Rethinking Reward Modeling in Preference-based Large Language Model Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7423902b5534e2b267438c85444a54b1-Abstract-Conference.html) — 카메라레디 버전(제목 변경).
- [GitHub: holarissun/RewardModelingBeyondBradleyTerry](https://github.com/holarissun/RewardModelingBeyondBradleyTerry) — 공식 구현.
- Bradley, R. A. and Terry, M. E., 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika. (BT 모델 원 논문)
- Chiang et al., 2024. [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132). (LLM Arena 비교 규모 인용원)
- Bos, T. and Schmidt-Hieber, J., 2022. Convergence rates for non-parametric classification with generalized quadratic loss. (truncated KL risk 프레임워크)
- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS 2017. (BT 손실을 RLHF에 처음 적용)
