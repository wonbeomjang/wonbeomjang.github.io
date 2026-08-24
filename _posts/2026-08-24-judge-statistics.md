---
layout: post
title: "judge를 통계로 다루기 — 편향, Bradley-Terry, PPI"
date: 2026-08-24 09:19:00 +0900
description: "LLM 평가 체계 시리즈 #19 — judge는 편향된 값싼 측정 도구다. Bradley-Terry로 순위의 불확실성을 다루고, Prediction-Powered Inference로 소량의 사람 라벨과 대량의 judge 라벨을 결합해 편향 없는 신뢰구간을 얻는 법"
categories: [paper]
tags: [evaluation, llm-as-a-judge, bradley-terry, prediction-powered-inference, statistics, paper]
giscus_comments: true
related_posts: true
---

> [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633) (Angelopoulos et al., UC Berkeley, Science 2023)

# Introduction

[#9](/blog/2026/mt-bench-to-arena/)에서 judge를 **벤치마크로서** 봤다. MT-Bench가 잰 것은 "GPT-4 judge가 사람과 얼마나 일치하는가"였고, 답은 80\~87%였다. 그런데 그 숫자를 "judge가 15\~20%를 틀린다"는 결함으로만 읽으면 안 된다는 게 #9의 결론이었다 — **사람\~사람 일치율도 딱 그 정도(81%)**였기 때문이다. judge가 못 맞추는 나머지는 두 부류로 갈린다. 정말 judge가 틀린 경우와, 원래 사람도 못 맞추는 애매한 경우. #9는 이 둘을 가르는 게 "이후 judge 연구 전체의 화두"라고 미뤄뒀다. 이 글이 그 화두를 받는다.

관점을 바꿔보면 질문이 더 명확해진다. judge는 **측정 도구(measuring instrument)**다. 온도계나 저울처럼, 참값을 정확히 재지 못하고 어느 정도의 편향(bias)과 분산(variance)을 갖는다. [#1](/blog/2026/what-is-evaluation/)의 언어로 말하면, judge라는 조작화(operationalization)에는 측정 오차가 낀다. 다른 점은 이 측정 도구가 **아주 값싸다**는 것이다. 사람 라벨러를 고용하는 데는 시간과 돈이 들지만, GPT-4를 10,000번 호출하는 데는 상대적으로 얼마 들지 않는다.

그래서 이 글의 핵심 질문은 다음과 같다.

> judge는 편향된 값싼 측정 도구다. 그런데 값싸다. 편향된 값싼 측정과 정확한 비싼 측정을 어떻게 결합하면, 편향 없이 유효한 신뢰구간을 얻을 수 있는가?

답은 **Prediction-Powered Inference(PPI)** 다. 대량의 judge 라벨로 점추정치를 만들고, 소량의 사람 라벨로 그 판단이 얼마나 치우쳐 있는지(rectifier, 보정항)를 추정해 빼준다. 이 아이디어 하나가 이 글 전체를 관통한다. 순서는 다음과 같다.

1. **설계로 상쇄할 수 있는 편향**은 #9에서 이미 다뤘으니 표로만 정리하고 넘긴다.
2. **Bradley-Terry(BT) 모형**을 다시 보되, 이번에는 "점수"가 아니라 **순위 자체의 불확실성**에 초점을 맞춘다.
3. **PPI와 PPI++**를 유도부터 직접 검증한다 — 왜 불편추정량인지, 분산이 왜 줄어드는지.
4. PPI를 실제 LLM 평가에 적용한 **AutoEval Done Right**(Boyeau et al., ICML 2025)를 읽는다.
5. 그런데도 judge에는 **이론적 상한**이 있다는 것을 Dorner et al.(ICLR 2025)로 확인한다.
6. 마지막으로 **가장 실용적인 한 줄**을 남긴다 — judge로 전부 채점해도, 사람 라벨 몇 백 개는 반드시 남겨두라는 것이다.

# Background

## 측정 오차로서의 judge

관측되는 judge 점수를 다음처럼 분해해보자.

$$
f(X) = \theta + \text{bias}(X) + \varepsilon(X)
$$

- $$\theta$$: 우리가 진짜 알고 싶은 값(참 helpfulness rate, 참 승률 등)
- $$\text{bias}(X)$$: judge가 특정 응답에 대해 체계적으로 치우치는 정도 — position, verbosity, self-enhancement 같은 것들이 여기 들어간다
- $$\varepsilon(X)$$: 평균 0인 순수한 잡음

이 분해가 중요한 이유는, 이 두 성분에 **완전히 다른 처방**이 필요하기 때문이다. 잡음 $$\varepsilon$$은 표본을 늘리면 저절로 줄어든다($$1/\sqrt{n}$$). 하지만 편향 $$\text{bias}(X)$$는 표본을 아무리 늘려도 사라지지 않는다 — judge를 10만 번 불러도 여전히 3%p만큼 치우쳐 있다. [#15](/blog/2026/confidence-intervals/)가 다룰 신뢰구간 이론은 잡음을 다루는 도구이지, 편향을 다루는 도구가 아니다. 편향을 다루려면 편향의 **크기 자체를 추정**해서 빼줘야 한다. 이 글의 나머지는 전부 "편향을 어떻게 추정해서 빼는가"의 이야기다.

## 값싼 대량 측정 + 비싼 소량 측정이라는 공통 문제 설정

PPI, AutoEval, Dorner et al.의 논문 세 편은 전부 같은 데이터 구조를 가정한다. 이 구조를 먼저 고정해두면 이후 모든 수식이 같은 기호로 읽힌다.

- **라벨 없는(또는 judge가 채점한) 대량 데이터**: $$X_1, \ldots, X_N$$, $$N$$개. 여기에 judge $$f$$가 라벨을 붙인다: $$f(X_1), \ldots, f(X_N)$$. $$N$$은 크다(수천\~수만).
- **사람이 라벨을 붙인 소량 데이터**: $$(X_1, Y_1), \ldots, (X_n, Y_n)$$, $$n$$개. $$Y_i$$가 참값(사람 라벨)이다. $$n \ll N$$.
- **핵심 가정**: 두 데이터셋의 $$X$$는 **같은 분포에서 iid로** 뽑혔다. 이 가정이 없으면 뒤에 나오는 모든 불편성(unbiasedness) 증명이 무너진다.

목표는 $$\theta = \mathbb{E}[Y]$$(참값의 평균, 예를 들어 참 승률이나 참 정확도)를 추정하는 것이다.

# Method

## 1. judge 편향을 설계로 상쇄하기

#9에서 이미 자세히 다룬 세 가지 편향을 표로만 정리한다. 수치와 실험 설계는 전부 [#9](/blog/2026/mt-bench-to-arena/)에 있다.

| 편향 종류             | 무엇이 문제인가                                                 | 설계적 완화책                                                  |
| --------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| Position bias         | 순서만 바꿔도 판정이 뒤집힌다(#9: GPT-4도 35%는 뒤집힘)         | 양쪽 순서 모두 물어, 둘 다 이겨야 승리 인정(swap)              |
| Verbosity bias        | 내용이 같아도 긴 답을 선호(#9: GPT-3.5·Claude-v1 91.3% 속음)    | 길이 통제 회귀(AlpacaEval 2.0 LC), 길이 페널티 규칙(WildBench) |
| Self-enhancement bias | judge가 자기 계열 모델을 우대(#9: GPT-4 +10%p, Claude-v1 +25%p) | judge를 다양화·앙상블, 평가 대상과 다른 계열의 judge 사용      |

이 완화책들의 공통점을 짚어두자. 전부 **편향이 만들어낼 수 있는 최악의 영향에 상한을 씌우거나, 편향의 원인을 설계에서 미리 차단**하는 것이다. 편향의 크기를 정확히 추정해서 제거하는 것은 아니다. 스왑을 걸어도 남는 편향, 길이를 통제해도 남는 편향은 여전히 있다. 그 **남은 편향**을 통계로 추정해 없애는 것이 이 글의 나머지 전부다.

## 2. Bradley-Terry 모형과 순위의 불확실성

### BT 모형과 로지스틱 회귀의 동형성

BT 모형은 [#9](/blog/2026/mt-bench-to-arena/)에서 이미 도입했다. 모델 $$i$$가 모델 $$j$$를 이길 확률을 잠재 강도(latent strength) $$\beta$$로 모형화한다.

$$
P(i \succ j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}
$$

분자·분모를 $$e^{\beta_i}$$로 나누면 이 식은 로지스틱 형태와 정확히 같아진다.

$$
P(i \succ j) = \frac{1}{1 + e^{-(\beta_i - \beta_j)}}
$$

- $$\beta_i, \beta_j$$: 각 모델의 잠재 강도. 값 자체는 의미가 없고 **차이**만 의미가 있다(원점을 아무 데나 잡아도 됨).
- $$\beta_i - \beta_j$$가 크면 $$i$$가 이길 확률이 1에 가까워진다.

이제 $$M$$개의 모델을 비교하는 실제 데이터를 생각하자. 매 비교 $$k$$마다 설계행렬의 행 $$x_k$$를 "모델 $$i$$ 위치에 $$-1$$, 모델 $$j$$ 위치에 $$+1$$, 나머지는 0"으로 두고, $$y_k \in \{0,1\}$$을 승부 결과로 두면 위 확률식은 그대로 $$P(y_k=1) = \text{logistic}(x_k^\top \zeta)$$가 된다. 그러면 BT 계수의 최대우도추정(MLE)은

$$
\hat\zeta = \operatorname*{argmin}_{\zeta} \; \frac{1}{n}\sum_{k=1}^n \ell(x_k, y_k)
$$

여기서 $$\ell$$은 이진 교차엔트로피(binary cross-entropy)다. 즉 **BT 계수의 MLE는 정확히 로지스틱 회귀와 같은 문제**다. 통계 라이브러리 하나만 있으면 BT 모형을 적합할 수 있다는 뜻이고, 뒤에서 볼 PPI/PPI++가 BT에 곧바로 적용되는 이유도 여기에 있다 — PPI++는 일반화선형모형(GLM)에 대해 정의되고, BT는 그냥 GLM의 한 인스턴스이기 때문이다.

### Elo가 왜 문제인가 (요약)

[#9](/blog/2026/mt-bench-to-arena/)에서 자세히 다뤘으니 결론만 다시 쓴다. Elo는 경기가 끝날 때마다 온라인으로 갱신하는 방식이라, **투표를 처리하는 순서에 최종값이 의존**한다. LLM은 체스 선수와 달리 가중치가 고정된 정적 객체이므로, "최근 경기에 더 큰 가중치"라는 Elo의 전제 자체가 맞지 않는다. BT MLE는 전체 데이터를 한 번에(batch) 보고 적합하므로 **순서 불변**이고, 점근적 표준오차 이론이 갖춰져 있어 원칙에 맞는 신뢰구간을 낼 수 있다. Chatbot Arena가 온라인 Elo를 버리고 BT MLE로 옮겨간 이유가 이것이다.

### 순위의 신뢰구간 — 점수가 아니라 순위를 부트스트랩한다

여기서부터 #9와 다른 이야기를 한다. #9는 **BT 계수 $$\beta$$ 자체의 CI**(피벗 부트스트랩, 샌드위치 표준오차)를 다뤘다. 이 글이 묻는 건 한 단계 더 실무적인 질문이다 — **"이 모델이 저 모델보다 순위가 높다"고 말해도 되는가?**

이 질문에 답하려면 점수의 CI를 순위의 CI로 바꿔야 한다. 절차는 다음과 같다.

1. 전체 비교 데이터로 BT MLE를 적합해 점추정치 $$\hat\beta_1, \ldots, \hat\beta_M$$을 얻는다.
2. 비교 데이터를 복원추출(bootstrap resampling)로 $$B$$번(예: $$B=1000$$) 재표본하고, 매번 BT MLE를 다시 적합한다. $$B$$개의 $$\{\hat\beta_m^{(b)}\}$$ 집합이 생긴다.
3. 각 재표본 $$b$$에서 모델들을 $$\hat\beta^{(b)}$$ 순서로 정렬해 순위를 매긴다.
4. 모델 $$m$$이 $$B$$번의 재표본 중 몇 번이나 1위, 2위, ...를 차지했는지 세면, 그것이 모델 $$m$$의 **순위 분포**다.

가상의 예로 이 절차의 결과를 따라가 보자. 모델 A, B, C를 1000번 부트스트랩했다고 하자(아래는 설명용으로 지어낸 가상 수치다).

| 모델 | 1위 비율 | 2위 비율 | 3위 비율 |
| ---- | -------- | -------- | -------- |
| A    | 62%      | 33%      | 5%       |
| B    | 35%      | 55%      | 10%      |
| C    | 3%       | 12%      | 85%      |

C는 거의 항상 3위다(85%) — C의 순위는 확정적으로 말할 수 있다. 하지만 A와 B는 어떤가. A가 1위인 재표본이 62%지만, B가 1위인 재표본도 35%나 된다. 즉 **재표본의 3분의 1 이상에서 "B가 A보다 낫다"는 결론이 나온다.** 이 상황에서 "A가 B보다 낫다"고 자신 있게 말할 수 없다. 점 추정치만 보면 A가 앞서지만, 순위 분포를 보면 그 차이가 잡음 안에 있다.

이것이 바로 [#9](/blog/2026/mt-bench-to-arena/)의 **separability(변별력)** 지표가 정량화하는 것과 정확히 같은 현상이다 — 두 모델의 신뢰구간이 겹치면 순위를 말할 근거가 없다. Arena-Hard-Auto의 separability 87.4%, MT-Bench의 22.6%라는 숫자는 바로 이런 부트스트랩(또는 그와 동등한 해석적 CI) 계산의 결과물이다. 점수의 CI를 내는 것과 순위의 CI를 내는 것은 서로 다른 질문이 아니라 **같은 부트스트랩 절차의 두 가지 요약**일 뿐이다.

### 여력이 되면: 이행성 위반

BT/Elo는 근본적으로 모든 모델의 강도를 **하나의 실수 축** 위에 놓을 수 있다고 가정한다. 이 가정이 참이라면 $$A \succ B$$이고 $$B \succ C$$이면 반드시 $$A \succ C$$다(이행성, transitivity). 그런데 실제 선호 데이터에는 순환(cycle) — $$A \succ B \succ C \succ A$$ — 이 나타날 수 있다. 한 모델이 코딩에는 강하지만 창작에는 약하고, 다른 모델이 정반대라면, "어느 쪽이 전반적으로 낫냐"는 질문 자체가 하나의 축으로 답해지지 않을 수 있다.

최근 연구들이 이 문제를 정면으로 다룬다. AlpacaEval처럼 고정된 베이스라인과의 pairwise 비교로 순위를 매기는 방식은, 이행성이 깨진 데이터에서는 **어느 베이스라인을 고르느냐에 따라 순위가 달라진다**는 것이 보고됐다(arXiv 2606.17634). BT 모형을 순환 선호까지 표현할 수 있도록 확장하려는 시도(Combinatorial Hodge Theory 기반, arXiv 2601.07158)나, 응답을 잠재 공간에 임베딩해 비이행적 구조까지 담으려는 일반화 선호 모형(arXiv 2410.02197)도 나왔다. 실무적으로는, 비교 데이터에서 순환 삼조(cyclic triple)의 비율을 세보는 것만으로도 "이 데이터가 정말 하나의 축으로 요약될 수 있는가"를 점검할 수 있다. BT 신뢰구간이 좁게 나왔다고 해서 그 순위가 안정적이라는 뜻은 아니다 — 모형 자체의 가정이 깨져 있을 수 있다.

## 3. Prediction-Powered Inference — 이 글의 핵심

### 문제 설정과 두 가지 순진한 방법

Background에서 고정한 설정을 그대로 쓴다. 대량의 judge 라벨 $$f(X_1), \ldots, f(X_N)$$과 소량의 사람 라벨 $$Y_1, \ldots, Y_n$$이 있다($$N \gg n$$). $$\theta = \mathbb{E}[Y]$$를 추정하고 싶다.

**순진한 방법 1 — 사람 라벨만 쓴다.**

$$
\hat\theta^{\text{classical}} = \frac{1}{n}\sum_{i=1}^n Y_i
$$

이건 명백히 불편(unbiased)이다. 문제는 $$n$$이 작다는 것 — 분산이 $$\text{Var}(Y)/n$$이라 $$n$$이 작으면 신뢰구간이 넓다. 대량의 judge 라벨을 아예 버리는 셈이니, 정보를 낭비한다.

**순진한 방법 2 — judge 라벨을 라벨처럼 쓴다.**

$$
\hat\theta^{\text{naive}} = \frac{1}{N}\sum_{i=1}^N f(X_i)
$$

$$N$$이 크므로 분산은 작다(신뢰구간이 좁다). 그런데 $$\mathbb{E}[f(X)] \neq \mathbb{E}[Y]$$이면(즉 judge에 편향이 있으면) 이 추정량 자체가 편향돼 있다. 신뢰구간은 좁은데 **참값을 포함하지 못한다** — 좁고 정확한 것보다 훨씬 나쁘다. 자신 있게 틀린 답을 준다.

두 방법이 정반대의 방식으로 실패한다. 하나는 편향은 없지만 너무 넓고, 다른 하나는 좁지만 편향돼 있다. PPI는 이 둘을 조합해서 **좁으면서도 편향 없는** 구간을 만든다.

### PPI 추정량

$$
\hat\theta^{PPI} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} f(X_i)}_{\text{대량 예측 평균}} \;-\; \underbrace{\frac{1}{n}\sum_{i=1}^{n}\big(f(X_i) - Y_i\big)}_{\text{편향 보정(rectifier)}}
$$

기호를 하나씩 풀어보자.

- 첫 항 $$\frac{1}{N}\sum f(X_i)$$: judge가 대량 데이터에 매긴 점수의 평균. $$N$$이 커서 분산이 작다.
- 둘째 항 $$\frac{1}{n}\sum (f(X_i) - Y_i)$$: **rectifier(보정항)**. 사람 라벨이 있는 소량 데이터에서, judge와 사람이 얼마나 다르게 채점하는지의 평균 — 즉 judge의 편향 그 자체를 추정한 값이다.
- 전체 식: "judge를 대량으로 믿고 평균을 내되, judge가 소량 표본에서 보인 만큼의 편향은 미리 빼둔다."

비유하면 이렇다. 저울 하나가 항상 3g씩 많이 나온다는 걸 알고 있다면, 그 저울로 잰 무게에서 3g을 빼면 된다. rectifier가 하는 일이 정확히 이것이다 — "이 judge가 평균적으로 얼마나 후하게(혹은 박하게) 채점하는가"를 소량의 정답 데이터로 재서, 대량 측정값에서 빼주는 것이다.

**왜 불편추정량인가.** 기댓값을 직접 취해보자. 두 데이터셋의 $$X$$가 같은 분포에서 iid로 뽑혔다는 가정을 쓴다.

$$
\mathbb{E}[\hat\theta^{PPI}] = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N f(X_i)\right] - \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n \big(f(X_i)-Y_i\big)\right]
$$

$$
= \mathbb{E}[f(X)] - \Big(\mathbb{E}[f(X)] - \mathbb{E}[Y]\Big) = \mathbb{E}[Y] = \theta
$$

핵심은 **첫 항의 $$\mathbb{E}[f(X)]$$와 둘째 항 안의 $$\mathbb{E}[f(X)]$$가 정확히 같은 값**이라는 것이다. 두 데이터셋이 같은 분포에서 뽑혔으니, judge를 대량 데이터에 적용했을 때의 평균 편향과 소량 데이터에 적용했을 때의 평균 편향은 (기댓값에서) 동일하다. 그래서 둘째 항에서 이 값을 추정해 빼면, 첫 항에 숨어있던 편향이 정확히 상쇄되고 $$\mathbb{E}[Y]$$만 남는다. **judge가 얼마나 정확한지는 이 증명에 전혀 등장하지 않는다** — judge가 얼마나 나쁘든, 두 데이터셋이 같은 분포에서 뽑혔다는 가정만 있으면 불편성이 보장된다. 이것이 PPI가 "어떤 머신러닝 모델이든 블랙박스로 쓸 수 있다"고 주장하는 근거다.

**분산.** 두 데이터셋이 독립이므로 분산은 그냥 더해진다.

$$
\text{Var}(\hat\theta^{PPI}) = \frac{1}{N}\text{Var}\big(f(X)\big) + \frac{1}{n}\text{Var}\big(f(X) - Y\big)
$$

첫 항은 $$N$$이 크므로 거의 무시할 만큼 작다. 둘째 항이 실질적인 분산의 대부분을 결정한다 — 그리고 이 항은 **judge와 사람이 얼마나 일치하는가**($$f(X)-Y$$의 분산)에 좌우된다. judge가 정확할수록(사람과 거의 항상 같은 판정을 낼수록) $$\text{Var}(f(X)-Y)$$가 작아지고, 신뢰구간이 좁아진다. 반대로 judge가 사람과 거의 무관하게 판정한다면 $$\text{Var}(f(X)-Y) \approx \text{Var}(f(X)) + \text{Var}(Y)$$로 커져서, PPI가 사람 라벨만 쓴 것보다 오히려 나빠질 수도 있다. 이 약점을 고치는 것이 다음 절의 PPI++다.

### PPI++ — λ로 안전성을 확보한다

PPI의 원 논문(Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, Science 2023)의 추정량은 위 식에서 rectifier의 가중치가 암묵적으로 1이다. 그런데 judge가 나쁘면 rectifier의 분산이 커져서, 사람 라벨만 쓰는 것보다 못할 수 있다. PPI++(Angelopoulos, Duchi, Zrnic, arXiv 2023)는 rectifier에 **가중치 $$\lambda$$**를 붙여 이 문제를 없앤다.

$$
\hat\theta_\lambda^{PPI++} = \underbrace{\frac{1}{n}\sum_{i=1}^n Y_i}_{\text{고전적 평균}} \;+\; \lambda\left(\underbrace{\frac{1}{N}\sum_{i=1}^N f(X_i)}_{\text{대량 judge 평균}} - \underbrace{\frac{1}{n}\sum_{i=1}^n f(X_i)}_{\text{소량 judge 평균}}\right)
$$

이 식은 앞의 PPI 추정량을 다시 쓴 것과 같다 — $$\lambda=1$$을 넣으면 정확히 $$\hat\theta^{PPI}$$가 된다(직접 전개해보면 $$\frac{1}{n}\sum Y_i + \frac{1}{N}\sum f(X_i) - \frac{1}{n}\sum f(X_i) = \frac{1}{N}\sum f(X_i) - \frac{1}{n}\sum(f(X_i)-Y_i)$$로 정리된다). $$\lambda=0$$을 넣으면 judge 항이 통째로 사라지고 $$\hat\theta^{classical}$$만 남는다.

- $$\lambda \in [0,1]$$(또는 그 이상)은 **"이 judge의 예측을 얼마나 신뢰할지"를 나타내는 손잡이**다.
- 임의의 $$\lambda$$에 대해서도 이 추정량은 **여전히 불편**이다 — $$\mathbb{E}[f(X)]$$가 두 항에서 다시 상쇄되기 때문이다(PPI와 같은 논리).
- 분산을 최소화하는 최적값 $$\lambda^\star$$는 데이터로부터 간단히 추정할 수 있는 닫힌 형태를 갖는다(회귀 계수와 비슷한 형태 — $$Y$$와 $$f(X)$$의 공분산을 $$f(X)$$의 분산으로 나눈 값에 가깝다).

**왜 이 성질이 실무에서 결정적인가.** $$\lambda$$를 분산 최소화로 최적화하면, judge가 나쁠 때는 $$\lambda^\star \to 0$$이 되어 자동으로 고전적 방법으로 돌아간다. Boyeau et al.(2025)의 표현을 빌리면: "annotator 모델이 아주 나빠도, PPI++는 최소한 고전적 방법만큼은 해낸다. annotator 라벨이 참 라벨과 상관이 없으면, PPI++는 $$\lambda=0$$이 되어 고전적 방법으로 돌아가고, 합성 라벨을 사실상 무시한다." 즉 PPI++는 **"밑져야 본전"이 보장되는 방법**이다 — 아무리 나쁜 judge를 써도 사람 라벨만 쓴 경우보다 못해지지 않는다(점근적으로). 이것이 실무에서 "일단 PPI++를 걸어두면 손해 볼 일이 없다"는 확신을 준다.

## 4. AutoEval Done Right — LLM 평가에 PPI 적용

**AutoEval Done Right: Using Synthetic Data for Model Evaluation** (Boyeau, Angelopoulos, Li, Yosef, Malik, Jordan, UC Berkeley, ICML 2025, arXiv 2403.07008)은 PPI/PPI++를 실제 모델 평가 파이프라인에 적용하는 방법을 제시한다.

**2단계 절차(autoevaluation)**는 이렇다.

1. 대량의 라벨 없는 데이터에 **AI로 합성 라벨을 붙인다**(judge 모델 $$f$$가 $$f(X_i)$$를 낸다).
2. 그 합성 라벨로 모델을 평가한다 — 이때 **PPI++로 편향을 보정**한다.

**정확도(accuracy) 추정에서의 구체적 형태.** 논문은 accuracy처럼 자주 쓰는 메트릭에 대해 PPI++를 조금 다듬은 형태를 제안한다.

$$
\hat\mu_m = \underbrace{\frac{\lambda}{N}\sum_{i=1}^{N} p_{i,m}^u}_{\text{합성 데이터 위의 정확도}} \;+\; \underbrace{\frac{1}{n}\sum_{i=1}^{n} \Delta_{i,m}^\lambda}_{\text{편향 보정}}, \qquad \Delta_{i,m}^\lambda := \mathbb{1}(\hat Y_{i,m}=Y_i) - \lambda\, p_{i,m}
$$

- $$p_{i,m}$$: 모델 $$m$$이 라벨 있는 데이터 $$i$$에서 낸 top 소프트맥스 점수 — "모델 스스로 생각하는 자기 정확도"에 대한 judge 없는 신호다.
- $$\mathbb{1}(\hat Y_{i,m}=Y_i)$$: 사람 라벨과 실제로 일치하는지의 지시함수.
- $$\lambda=1$$이면 원래의 PPI, $$\lambda=0$$이면 순수 고전적 방법이다.

이 식의 요점은 앞서 본 일반 PPI 공식과 결이 같다 — **대량 데이터에서의 (합성) 평균**에 **소량 데이터에서 추정한 편향 보정**을 더하는 구조가 그대로 반복된다.

**효과적 표본 크기(Effective Sample Size, ESS)라는 지표.** 논문은 PPI/PPI++의 이득을 "이 정도 정밀도를 고전적 방법으로 내려면 사람 라벨이 몇 개 필요했을까"로 환산해 보여준다. 세 가지 실험에서 나온 수치는 다음과 같다.

| 실험                       | 무엇을 평가했나                 | ESS 개선                                          |
| -------------------------- | ------------------------------- | ------------------------------------------------- |
| ImageNet (ResNet 5종)      | 모델 정확도                     | 약 50%                                            |
| 단백질 적합도(ProteinGym)  | 적합도 예측 모델의 Pearson 상관 | 약 50%(annotator가 나빠도 최소 10%, CARP 사용 시) |
| **Chatbot Arena LLM 순위** | BT 계수(모델 강도)              | **20\~25%**(judge에 따라 20\~35%)                 |

세 번째 행이 가장 눈여겨볼 "20\~25% 개선"이다. 논문 원문을 그대로 옮기면: "we also observed ESS showing a 20% to 25% improvement over the classical approach" — **이 숫자는 신뢰구간의 폭이 아니라 유효 표본 크기(ESS)의 개선율이다.** 16,000건의 Chatbot Arena 사람 선호 데이터 중 일부만 사람 라벨로 쓰고 나머지는 GPT-4o-mini가 판정한 judge 라벨로 대체했을 때, PPI++로 얻은 정밀도가 "고전적 방법으로 실제 사람 라벨을 20\~25% 더 모은 것"과 같은 정밀도였다는 뜻이다. 또한 judge 모델을 GPT-4o-mini, GPT-4o, Claude Sonnet, LLaMA 3.1, Gemini 1.5로 바꿔가며 실험했는데, **어떤 judge를 쓰든 예외 없이 ESS가 20\~35% 개선**됐다.

**순위 정확도도 함께 개선됐다.** ImageNet·단백질·LLM 세 실험 모두에서 "PPI++로 얻은 모델 순위가 고전적 방법보다 참 순위와 훨씬 더 잘 상관했다"고 보고한다. 특히 LLM 실험(Figure 4c)에서는 PPI++가 다른 방법보다 뚜렷하게 강한 상관을 보였다.

**실무 함의.** 이 논문이 강조하는 가장 중요한 문장은 이것이다 — "annotator 모델이 참 라벨과 전혀 상관이 없어도, PPI++는 고전적 방법보다 나빠지지 않는다." 그리고 이 이득은 **사람 라벨을 남겨뒀을 때만** 나온다. judge로 전부 채점하고 사람 라벨을 하나도 남기지 않으면, rectifier를 계산할 방법이 없다 — 편향을 추정할 수 없으니 편향을 뺄 수도 없다. "사람 라벨 몇 개를 남겨두는 것"이 이 전체 방법론의 성립 조건이다.

## 5. judge의 이론적 한계 — Dorner et al.

지금까지는 PPI가 "얼마나 좋은가"를 봤다. 이 절은 PPI가 **아무리 잘 써도 넘을 수 없는 상한**을 보여준다.

**Limits to Scalable Evaluation at the Frontier: LLM as Judge Won't Beat Twice the Data** (Dorner, Nastl, Hardt, Max Planck Institute for Intelligent Systems, ICLR 2025 Oral, arXiv 2410.13341)의 형식화는 이렇다.

- 모델 $$m$$의 참 점수 $$b(m) := \mathbb{E}[s(m)]$$(예: 정확도), judge의 대리 점수 $$\tilde s(m)$$로 추정한 값 사이의 **judge 편향**을 다음처럼 정의한다.

$$
JB(m) := \mathbb{E}[\tilde s(m) - s(m)] = \big(1-q(m)\big)\big(1-b(m)\big) - \big(1-p(m)\big)b(m)
$$

여기서 $$p(m)$$은 참으로 옳을 때 judge도 옳다고 판정할 확률(민감도), $$q(m)$$은 참으로 틀렸을 때 judge도 틀렸다고 판정할 확률(특이도)이다.

- **PPI 추정량이 최선(near-optimal)이라는 정리(Theorem 5)**: 사람 라벨 $$n$$개와 judge 라벨 $$N$$개에 접근할 수 있는 **모든** 불편추정량 $$\hat\theta$$에 대해, "표본 효율 인자" $$\tau(\hat\theta) := \text{Var}(\hat\theta^{GT})/\text{Var}(\hat\theta)$$(고전적 방법 대비 몇 배의 사람 라벨에 해당하는 정밀도인가)는

$$
\tau(\hat\theta) \le \frac{1}{1-\rho(s,\tilde s)^2}
$$

를 만족한다. 여기서 $$\rho(s,\tilde s)$$는 참 점수와 judge 대리 점수 사이의 피어슨 상관계수다. 이 상한은 어떤 불편추정법을 쓰든(부트스트랩이든, PPI든, 그 무엇이든) 넘을 수 없다 — 크래머-라오 하한(Cramér-Rao bound)에서 유도된다. PPI/PPI++는 이 상한에 거의 도달하는 것으로 보인다.

- **핵심 정리(Theorem 6, 그리고 Corollary 7)**: 만약 judge의 합치율(agreement) $$AG(m)$$이 평가 대상 모델의 참 점수 $$b(m)$$보다 낫지 않다면(즉 $$0.5 \le AG(m) \le b(m)$$, **"judge가 평가 대상보다 강하지 않다"**는 조건), 상관계수의 제곱은

$$
\rho(s,\tilde s)^2 \le 0.5
$$

로 제한되고, 따라서

$$
\tau_{\max} = \max_{\hat\theta} \tau(\hat\theta) \le 2
$$

이다. 이것이 논문 제목이 말하는 "**judge를 써도 데이터 2배를 못 이긴다**"의 정확한 의미다 — judge가 평가 대상 모델보다 나은 성능을 갖지 않는 한(즉 프론티어 평가 상황, 이제 막 나온 신모델을 예전 모델로 채점하려는 상황), **어떤 디바이어싱 방법을 쓰더라도 얻을 수 있는 최선의 표본 효율 이득은 "사람 라벨을 정확히 두 배 모은 것"과 같다.** $$N$$을 아무리 크게 늘려도(judge 라벨을 무한히 뽑아도) 이 상한은 깨지지 않는다.

**균형 잡힌 결론.** 논문은 이 한계가 이론상으로만 그치지 않는다는 것도 보인다 — MMLU와 MT-Bench 실험에서 실제로 관측된 표본 절감은 이론적 상한(2배)보다 **더 작았다**. 그리고 다음 사실도 함께 보인다 — **judge의 raw 정확도(예: 84%, 90%)가 높다고 해서 이 상한이 완화되는 게 아니다.** Theorem 6은 $$AG(m)$$이 임의로 높아도(0.5 이상이기만 하면) 성립한다. 다만 이 상한은 조건 "$$AG(m) \le b(m)$$"에 묶여 있다 — **judge가 평가 대상보다 진짜로 강할 때는 2배를 넘는 이득도 가능하다**(논문의 LLaMA2-7B 실험에서는 최대 3.5배까지 관측됐다). 문제는 프론티어 평가의 본질이 "이제 나온 모델이 judge보다 강할 가능성이 있는" 바로 그 상황이라는 것이다.

## 6. 최신 후속

이 방향의 후속 연구가 계속 나오고 있다. 확인되는 만큼만 짧게 짚는다.

- **How to Correctly Report LLM-as-a-Judge Evaluations** (Lee, Zeng, Jeong, Sohn, Kangwook Lee, University of Wisconsin-Madison, arXiv 2511.21140). judge의 민감도·특이도로 유발되는 편향을 교정하는 플러그인 프레임워크를 제안한다. 테스트셋과 캘리브레이션(사람 라벨) 셋 **양쪽의 불확실성을 모두 반영**한 신뢰구간을 구성하고, 캘리브레이션 표본을 어디에 배분해야 구간이 가장 좁아지는지에 대한 적응적 전략도 제시한다.
- **Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges** (Feng, Shen, Balashankar, Gerner-Beuerle, Rodrigues, Queen's University Belfast / UCL / Google DeepMind, arXiv 2601.20913, ICLR 2026). "실패율이 안전 임계값 이하인가"를 검정하는 가설검정 틀을 세우고, 소량의 사람 라벨로 judge의 참양성률(TPR)·참음성률(FPR)을 추정해 분산 보정된 임계값을 유한 표본에서도 유효하게(finite-sample Type-I error control) 만든다.

두 논문 모두 이 글의 뼈대와 같은 정신을 공유한다 — **judge를 그냥 믿지 않고, 소량의 사람 라벨로 judge의 오차 구조를 추정해 통계적으로 보정한다.**

# Experiments

## 1. PPI 토이 예제 — 세 방법을 직접 계산으로 비교한다

설정을 다음처럼 고정한다.

- 참값(사람이 채점한 참 helpfulness rate) $$\theta = 0.60$$.
- judge가 체계적으로 3%p 높게 채점한다: $$\mathbb{E}[f(X)] = 0.63$$.
- 라벨 없는(judge만 채점한) 대량 데이터 $$N=10{,}000$$, 사람 라벨 소량 데이터 $$n=200$$.

이 관계를 만드는 구체적인 결합분포를 하나 정해두면 이후 모든 계산을 직접 검증할 수 있다. $$f, Y \in \{0,1\}$$의 결합분포를 다음처럼 잡는다.

|                       | $$Y=1$$ (참: 좋음) | $$Y=0$$ (참: 나쁨) | 합계 |
| --------------------- | ------------------ | ------------------ | ---- |
| $$f=1$$ (judge: 좋음) | 0.58               | 0.05               | 0.63 |
| $$f=0$$ (judge: 나쁨) | 0.02               | 0.35               | 0.37 |
| 합계                  | 0.60               | 0.40               | 1.00 |

이 표에서 바로 확인되는 것들이다.

- $$\mathbb{E}[Y] = 0.60$$ (참값), $$\mathbb{E}[f(X)] = 0.63$$ (judge가 3%p 후하다).
- $$\text{Var}(f) = 0.63 \times 0.37 = 0.2331$$, $$\text{Var}(Y) = 0.60 \times 0.40 = 0.24$$.
- $$D := f - Y$$의 분포는 $$P(D=0)=0.93$$(judge=사람, 표의 대각 두 칸의 합 $$0.58+0.35$$), $$P(D=1)=0.05$$(judge만 후함), $$P(D=-1)=0.02$$(judge만 박함). $$\mathbb{E}[D] = 0.05 - 0.02 = 0.03$$(3%p 편향과 정확히 일치), $$\mathbb{E}[D^2] = 0.05+0.02 = 0.07$$, 따라서 $$\text{Var}(D) = 0.07 - 0.03^2 = 0.0691$$.

**방법 1 — judge만 쓴다(naive).** $$\hat\theta^{naive} = 0.63$$. 표준오차 $$\text{SE} = \sqrt{0.2331/10000} \approx 0.00483$$, 95% 신뢰구간 반폭 $$1.96 \times 0.00483 \approx 0.0095$$(0.95%p). 구간은 $$[0.6205, 0.6395]$$ — **참값 0.60이 이 구간 밖에 있다.** 신뢰구간은 매우 좁지만 틀렸다.

**방법 2 — 사람 라벨만 쓴다(classical, $$n=200$$).** $$\hat\theta^{GT} = 0.60$$. 표준오차 $$\text{SE} = \sqrt{0.24/200} \approx 0.03464$$, 95% 신뢰구간 반폭 $$1.96 \times 0.03464 \approx 0.0679$$(6.79%p). 구간은 $$[0.5321, 0.6679]$$ — 참값을 포함하지만 폭이 넓다.

**방법 3 — PPI.** 점추정치는 불편이므로 기대값 그대로 $$0.60$$이다. 분산은

$$
\text{Var}(\hat\theta^{PPI}) = \frac{0.2331}{10000} + \frac{0.0691}{200} = 0.0000233 + 0.0003455 = 0.0003688
$$

표준오차 $$\sqrt{0.0003688} \approx 0.01921$$, 95% 신뢰구간 반폭 $$1.96 \times 0.01921 \approx 0.0376$$(3.76%p). 구간은 $$[0.5624, 0.6376]$$ — 참값을 포함하면서, 사람 라벨만 쓴 경우보다 폭이 **거의 절반(정확히는 약 45% 축소)** 이다.

| 방법                                       | 점추정 | 95% CI 반폭 | 참값(0.60) 포함? |
| ------------------------------------------ | ------ | ----------- | ---------------- |
| judge만(naive)                             | 0.63   | 0.95%p      | 아니오           |
| 사람 라벨만($$n=200$$)                     | 0.60   | 6.79%p      | 예               |
| PPI(judge $$N=10{,}000$$ + 사람 $$n=200$$) | 0.60   | 3.76%p      | 예               |

이 표를 유효 표본 크기(ESS)로 환산하면 더 실감이 난다. 사람 라벨만 쓴 방법의 분산은 $$0.24/n$$ 꼴이므로, PPI의 분산 $$0.0003688$$을 내려면 고전적 방법으로는 $$n_{\text{eff}} = 0.24/0.0003688 \approx 651$$개의 사람 라벨이 필요하다. 즉 **사람 라벨 200개 + judge 라벨 10,000개**가 **사람 라벨 약 651개**와 같은 정밀도를 낸다 — 사람 라벨을 실제로 모은 것보다 **약 3.25배** 늘린 것과 같다.

## 2. judge 정확도와 순위 역전

judge의 "정확도가 90%"라는 말은 두 가지 다른 의미를 가질 수 있고, 그 둘을 구분하지 않으면 잘못된 결론에 이른다.

**대칭적 잡음만 있는 경우 — 순위는 뒤집히지 않는다.** 모델 A가 모델 B를 이길 참 확률을 $$\pi$$라 하자. judge가 매 비교마다 확률 $$p=0.9$$로 옳은 판정을 내리고(정확도 90%), 틀릴 때는 반대쪽을 고른다고 하자(대칭 오차). 그러면 관측되는 A의 승률은

$$
\tilde\pi = p\cdot\pi + (1-p)(1-\pi) = \pi(2p-1) + (1-p)
$$

$$p=0.9$$를 넣으면 $$\tilde\pi = 0.8\pi + 0.1$$이다. 이 함수는 $$\pi$$에 대해 **단조증가**($$d\tilde\pi/d\pi = 0.8 > 0$$)이므로, $$\pi > 0.5$$이면 $$\tilde\pi > 0.5$$이고, 순서는 절대 뒤집히지 않는다. 예를 들어 참값 $$\pi=0.55$$(A가 5%p 앞선다)이면 관측값은 $$\tilde\pi = 0.8\times0.55+0.1=0.54$$ — 격차가 5%p에서 4%p로 줄어들 뿐, A가 앞선다는 결론 자체는 흔들리지 않는다. **judge 정확도가 90%로 떨어져도, 그 오차가 두 모델에 대해 대칭적이라면 순위는 안전하다.**

**비대칭 편향이 섞이면 순위가 뒤집힌다.** 문제는 실제 judge 오차가 대칭이 아니라는 것이다 — 앞서 본 verbosity bias나 self-enhancement bias는 **특정 모델 쪽으로만** 쏠린다. B가 더 장황하게 답하는 모델이라 judge가 B의 관측 승률에 $$\beta$$만큼을 더 얹어준다고 하자. 그러면 관측되는 A의 승률은 $$\tilde\pi - \beta$$가 되고, 순위가 뒤집히는 조건은

$$
\tilde\pi - \beta < 0.5 \;\Longleftrightarrow\; \beta > \tilde\pi - 0.5 = 0.8(\pi-0.5)
$$

$$\pi=0.55$$인 경우 $$\beta > 0.8 \times 0.05 = 0.04$$ — 즉 **4%p의 비대칭 편향만으로도** 순위가 뒤집힌다. 참값 격차가 더 좁으면(예: $$\pi=0.52$$, 2%p 차이) 뒤집는 데 필요한 편향은 $$\beta > 0.8\times0.02=0.016$$, 겨우 1.6%p면 충분하다. 이 숫자를 [#9](/blog/2026/mt-bench-to-arena/)에서 실측한 self-enhancement bias 크기(GPT-4 +10%p, Claude-v1 +25%p)와 나란히 놓으면 결론이 분명해진다 — **실제로 관측되는 판정자 편향의 크기는, 웬만한 실전 모델 격차를 뒤집기에 충분하고도 남는다.** judge 정확도가 아무리 높아도(90%든 95%든), 그 오차가 대칭이 아니라면 순위 보장은 없다.

## 3. 몇 개의 사람 라벨이 필요한가 — 수확 체감 지점

Experiment 1의 설정($$N=10{,}000$$, $$\text{Var}(f)=0.2331$$, $$\text{Var}(D)=0.0691$$)을 고정하고, 사람 라벨 수 $$n$$만 바꿔가며 PPI 신뢰구간의 반폭이 어떻게 줄어드는지 본다.

$$
\text{반폭}(n) = 1.96\sqrt{\frac{0.2331}{10000} + \frac{0.0691}{n}}
$$

| $$n$$            | 95% CI 반폭  | 직전 대비 감소율 |
| ---------------- | ------------ | ---------------- |
| 50               | 7.35%p       | —                |
| 100              | 5.24%p       | 28.7%            |
| 200              | 3.76%p       | 28.2%            |
| 400              | 2.74%p       | 27.1%            |
| 800              | 2.05%p       | 25.2%            |
| 1,600            | 1.60%p       | 22.0%            |
| 3,200            | 1.31%p       | 18.1%            |
| 6,400            | 1.14%p       | 13.0%            |
| $$n \to \infty$$ | 0.95%p(바닥) | —                |

$$n$$을 두 배로 늘릴 때마다 반폭이 줄어드는 비율이 28.7% → 28.2% → ... → 13.0%로 **계속 작아진다.** 이게 수확 체감이다. 그 이유는 분산식의 두 항 $$\text{Var}(f)/N$$(고정된 바닥)과 $$\text{Var}(D)/n$$(사람 라벨로 줄일 수 있는 항)의 크기가 역전되기 때문이다. 두 항이 같아지는 지점은

$$
n^\star = \frac{\text{Var}(D)}{\text{Var}(f)/N} = \frac{0.0691}{0.0000233} \approx 2{,}963
$$

즉 사람 라벨이 약 3,000개를 넘어가면, 더 늘려봐도 $$\text{Var}(f)/N=0.0000233$$이라는 **judge 표본 크기 $$N$$이 만드는 바닥** 때문에 개선이 급격히 둔화된다(위 표에서 1,600 → 3,200 구간부터 감소율이 뚜렷이 꺾이는 것이 보인다). 사람 라벨을 무한히 늘려도 반폭은 0.95%p보다 좁아지지 않는다 — 그 이상을 원한다면 사람 라벨이 아니라 judge 라벨(=$$N$$)을 늘려야 한다. 실무적으로 이 표가 주는 답은: **어느 지점(여기서는 대략 n=1,600\~3,200)을 넘기면 사람 라벨을 더 모으는 것의 가성비가 급격히 떨어진다** — 그 지점을 넘기기 전까지는 사람 라벨을 늘리는 것이 여전히 효율적이라는 뜻이기도 하다.

# 통계 요약

| 방법                            | 무엇을 가정하나                                 | 편향 없는가                         | CI 폭                                                  | 사람 라벨 필요량                          |
| ------------------------------- | ----------------------------------------------- | ----------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| judge만(naive)                  | 없음(judge를 그대로 믿음)                       | 아니오 — judge 편향이 그대로 남는다 | 좁다($$1/\sqrt N$$, $$N$$이 크므로)                    | 0                                         |
| 사람 라벨만(classical)          | 라벨이 iid 표본                                 | 예                                  | 넓다($$1/\sqrt n$$, $$n$$이 작으므로)                  | 전량 $$n$$                                |
| PPI($$\lambda=1$$)              | 라벨·비라벨 데이터가 같은 분포에서 iid          | 예                                  | 중간 — 두 항 분산의 합                                 | 소량 $$n$$(rectifier용)                   |
| PPI++(power tuning $$\lambda$$) | 위와 동일 + $$\lambda$$를 분산 최소화로 최적화  | 예                                  | PPI와 같거나 좁음 — 고전적 방법보다 절대 나빠지지 않음 | 소량 $$n$$                                |
| Bradley-Terry MLE + 부트스트랩  | 강도가 정적, 비교가 iid, 단일 축(이행성)        | (편향과 무관, 순위 CI를 낸다)       | 부트스트랩 재표본 수에 따라 결정                       | —                                         |
| Dorner의 이론적 상한            | judge가 평가 대상보다 강하지 않음($$AG \le b$$) | (모든 불편추정량에 대한 상한)       | 최선의 경우도 사람 라벨 $$n$$의 최대 2배 상당          | 어떤 방법을 써도 $$n/2$$ 아래로는 못 줄임 |

# Conclusion

judge는 판사가 아니라 **측정 도구**다. 모든 측정 도구는 편향과 분산을 갖고, 그것을 인정하고 통계로 다루는 것이 이 글의 유일한 답이다.

정리하면 실무 처방은 세 갈래다.

1. **편향은 설계로 일부 상쇄한다.** 순서 스왑, 길이 통제, 판정자 다양화 — [#9](/blog/2026/mt-bench-to-arena/)가 다룬 것들이다. 이걸로 편향의 최악의 영향에 상한을 씌울 수 있지만, 편향을 정확히 제거하지는 못한다.
2. **남은 편향은 소량의 사람 라벨로 추정해 빼낸다.** 이것이 PPI/PPI++다. rectifier라는 아주 단순한 산술(judge와 사람의 차이를 소량에서 평균 낸 것)이, 대량 judge 데이터의 편향을 통째로 상쇄한다. AutoEval Done Right가 보여준 20\~35%의 ESS 개선은, 이 아이디어를 실전 LLM 평가에 그대로 적용해서 얻은 결과다.
3. **judge만으로 사람 라벨을 완전히 대체할 수는 없다.** Dorner et al.의 정리가 그 상한을 정확히 못 박는다 — judge가 평가 대상보다 강하지 않은 한(바로 프론티어 평가가 처한 상황), 어떤 디바이어싱 방법을 써도 얻을 수 있는 최선은 "사람 라벨을 두 배로 모은 것"과 같은 정밀도다. $$N$$을 무한히 늘려도 이 상한은 깨지지 않는다.

**가장 실용적인 한 줄**: judge로 전부 채점하더라도, **사람 라벨 몇 백 개를 반드시 남겨두라.** 그것이 rectifier를 계산할 유일한 재료이고, 결국 유효한 신뢰구간의 유일한 근거다. judge에게 채점을 전부 맡기고 사람 라벨을 하나도 남기지 않는 순간, 이 글에서 본 모든 통계적 장치는 무력해진다 — 편향이 얼마인지조차 알 방법이 없기 때문이다.

judge를 벤치마크로 검증하는 이야기([#9](/blog/2026/mt-bench-to-arena/))에서, judge를 추정 문제로 다루는 이야기(이 글)까지 왔다. 다음 두 편은 이 시리즈의 마지막 갈래다 — [#20](/blog/2026/contamination-reproducibility/)은 오염과 재현성, [#21](/blog/2026/safety-evaluation-statistics/)은 안전 평가에 특화된 통계(희귀사건 추정, calibration)를 다루며 시리즈를 닫는다.

# 참고 문헌

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., and Zrnic, T., 2023. [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633). Science, 382(6671):669–674.
- Angelopoulos, A. N., Duchi, J. C., and Zrnic, T., 2023. [PPI++: Efficient Prediction-Powered Inference](https://arxiv.org/abs/2311.01453). arXiv:2311.01453.
- Boyeau, P., Angelopoulos, A. N., Li, T., Yosef, N., Malik, J., and Jordan, M. I., 2025. [AutoEval Done Right: Using Synthetic Data for Model Evaluation](https://arxiv.org/pdf/2403.07008) (ICML 2025).
- Dorner, F. E., Nastl, V. Y., and Hardt, M., 2024. [Limits to Scalable Evaluation at the Frontier: LLM as Judge Won't Beat Twice the Data](https://arxiv.org/abs/2410.13341) (ICLR 2025 Oral).
- Lee, C., Zeng, T., Jeong, J., Sohn, J., and Lee, K., 2025. [How to Correctly Report LLM-as-a-Judge Evaluations](https://arxiv.org/abs/2511.21140). arXiv:2511.21140.
- Feng, C., Shen, M., Balashankar, A., Gerner-Beuerle, C., and Rodrigues, M. R. D., 2026. [Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges](https://arxiv.org/abs/2601.20913) (ICLR 2026).
- Zheng, L. et al., 2023. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (NeurIPS 2023 D&B) — [#9](/blog/2026/mt-bench-to-arena/)에서 상세히 다룬 편향 실험의 원 논문.
- Chiang, W.-L. et al., 2024. [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) (ICML 2024).
- Chaganty, A. T., Mussman, S., and Liang, P., 2018. [The price of debiasing automatic metrics in natural language evaluation](https://arxiv.org/abs/1807.02202) — PPI와 동등한 아이디어의 초기 형태.
- 이행성 위반 관련: [Prompt Perturbation for Reliable LLM Evaluation over Comparison Graphs](https://arxiv.org/abs/2606.17634); [The Covariate-Assisted Bayesian Intransitive Bradley-Terry Model via Combinatorial Hodge Theory](https://arxiv.org/abs/2601.07158); [Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment](https://arxiv.org/abs/2410.02197).

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 열아홉 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="3">
  <li><a href="/blog/2026/measurement-scales/">척도와 허용 연산</a> — Likert 평균을 내도 되는가</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><a href="/blog/2026/generation-metrics/">생성 지표와 그 타당도</a> — BLEU에서 COMET까지</li>
  <li><a href="/blog/2026/mcqa-fragility/">객관식 평가는 왜 흔들리나</a> — 위치 편향과 포맷 민감도</li>
</ol>

**3부. LLM 벤치마크 지형도**

<ol start="7">
  <li><a href="/blog/2026/knowledge-benchmarks/">지식과 추론 — MMLU 계열의 흥망</a> — MMLU·GPQA·BBH·HELM</li>
  <li><a href="/blog/2026/math-code-benchmarks/">검증 가능한 도메인 — 수학과 코드</a> — GSM8K·MATH·HumanEval·SWE-bench</li>
  <li><a href="/blog/2026/mt-bench-to-arena/">개방형 대화 — MT-Bench에서 Arena까지</a> — judge 기반 벤치의 등장</li>
  <li><a href="/blog/2026/capability-axes-benchmarks/">능력의 다른 축</a> — 지시따르기·긴 문맥·사실성</li>
  <li><a href="/blog/2026/korean-benchmarks/">한국어 벤치마크</a> — 번역이 아니라 원산, 그리고 문화 타당도</li>
</ol>

**4부. 사람이 읽는다 — 정성평가와 일치도**

<ol start="12">
  <li><a href="/blog/2026/human-evaluation-design/">사람 평가 설계</a> — 루브릭·Likert·pairwise·BWS</li>
  <li><a href="/blog/2026/kappa-agreement/">우연을 빼다 — κ 계열</a> — Cohen·Fleiss·weighted·Krippendorff</li>
  <li><a href="/blog/2026/kappa-paradox/">κ의 역설</a> — 일치율 90%인데 κ가 0.21</li>
</ol>

**5부. 차이는 진짜인가 — 정량평가의 통계**

<ol start="15">
  <li><a href="/blog/2026/confidence-intervals/">점수는 추정치다</a> — 이항비율 신뢰구간과 Wald의 실패</li>
  <li><a href="/blog/2026/significance-testing/">차이는 유의한가</a> — paired bootstrap·순열검정·McNemar</li>
  <li><a href="/blog/2026/statistical-power/">몇 개를 재야 하나</a> — 검정력·표본크기·다중비교</li>
  <li><a href="/blog/2026/error-bars-for-evals/">LLM eval의 통계 실무</a> — 클러스터 SE·IQM·분산 분해</li>
</ol>

**6부. 신뢰할 수 있는 평가 체계**

<ol start="19">
  <li><strong>(현재 글)</strong> judge를 통계로 다루기 — 편향·Bradley-Terry·PPI</li>
  <li><a href="/blog/2026/contamination-reproducibility/">오염·재현성·효율</a> — 오염 검정·harness·IRT</li>
  <li><a href="/blog/2026/safety-evaluation-statistics/">안전 평가의 통계와 체계 설계</a> — 희귀사건·calibration·체크리스트</li>
</ol>

본 시리즈는 21편으로 구성된다.
