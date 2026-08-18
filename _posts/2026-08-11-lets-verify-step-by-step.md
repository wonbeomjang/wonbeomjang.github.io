---
layout: post
title: "Let's Verify Step by Step: 결과가 아니라 과정에 보상을 주다"
date: 2026-08-11 09:28:00 +0900
description: "RLHF Reward 설계 시리즈 #28 — process supervision이 outcome supervision을 이기는 이유와 PRM800K"
categories: [paper]
tags: [rlhf, reward-model, prm, process-supervision, reasoning, paper]
giscus_comments: true
related_posts: true
---

> [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., OpenAI, ICLR 2024)

# Introduction

[23편 DPO](/blog/2026/dpo/)까지 5부의 방향은 한결같았다. reward model을 아예 없애거나(DPO는 정책 자체를 암묵적 reward로 재파라미터화했다), value network를 걷어내거나([21편 GRPO](/blog/2026/grpo-deepseekmath/)), PPO의 복잡한 트릭을 REINFORCE 수준으로 되돌리는([22편 RLOO](/blog/2026/rloo-back-to-basics/)) 식으로 **reward 파이프라인을 단순화**하는 쪽으로 시리즈가 흘러왔다. 5부 전체를 한 문장으로 요약하면 "reward를 어떻게 줄일 것인가"였다.

이번 글부터 시작하는 6부 "Process & Verifiable Reward"는 정확히 반대 방향으로 간다. reward를 없애는 게 아니라 **훨씬 더 촘촘하게 쪼갠다**. 지금까지 다룬 reward model(2부, 3부)은 전부 하나의 응답 전체에 스칼라 하나를 매기는 outcome-level reward였다. 수학 문제를 20단계로 풀어낸 chain-of-thought든, 한 줄짜리 답변이든 reward model은 마지막에 딱 한 번 점수를 준다. 이 글이 다루는 **Let's Verify Step by Step**(OpenAI, ICLR 2024)은 "그 한 번의 점수로 충분한가?"라는 질문에 정면으로 "아니다"라고 답한 논문이다.

문제는 단순하다. 최종 답만 채점하면 **틀린 풀이 과정으로 우연히 맞은 답**에도 만점을 준다. 부호를 두 번 틀렸는데 우연히 상쇄돼서 답이 맞아버리는 경우, sign 실수를 알아채지 못하고 대충 찍은 값이 그대로 정답과 일치하는 경우 — 이런 solution은 reasoning으로서는 명백히 불량인데 outcome reward는 이를 구분하지 못한다. 논문은 이를 outcome-supervised reward model(ORM)의 **false positive** 문제라고 부른다. 저자들은 이 문제를 정면으로 겨냥해 매 추론 스텝마다 사람이 정오를 채점한 **PRM800K**(80만 개 스텝 라벨)를 구축하고, 이 라벨로 학습한 process-supervised reward model(PRM)이 MATH 데이터셋에서 ORM을 확실히 앞선다는 것을 보였다. best-of-1860 검색 기준으로 ORM 72.4%, PRM **78.2%**다.

이 글에서 답할 질문은 네 가지다.

1. ORM과 PRM은 정확히 무엇이 다른가, 그리고 왜 이 차이가 reasoning 과제에서 치명적인가.
2. PRM800K는 어떻게 만들어졌는가 — 라벨 스킴, 규모, 그리고 비용을 줄이기 위한 active learning.
3. 스텝별 점수를 solution 하나의 점수로 어떻게 합치는가.
4. 결과는 얼마나 차이가 나며, 이 접근이 남긴 부채는 무엇인가.

# Background

## sparse reward의 reasoning 버전 — credit assignment 문제

[1편](/blog/2026/deep-rl-human-preferences/)에서 다룬 원형적 문제를 다시 떠올려보자. 사람의 선호로 reward를 학습하는 구도에서, reward는 trajectory(응답) 전체에 대해 스칼라 하나만 돌려준다. 이게 sparse reward다. [10편](/blog/2026/reward-model-overoptimization/)은 이 sparse한 proxy reward를 정책이 과최적화하면 실제 품질과 무관하게 점수만 오르는 Goodhart 현상을 정량화했다.

reasoning 과제에서는 이 sparse reward 문제가 훨씬 날카로운 형태로 나타난다. 20단계짜리 수학 풀이가 있고 최종 답이 틀렸다고 해보자. outcome reward는 "틀렸다"는 사실 하나만 알려준다. **20단계 중 정확히 어느 단계에서 처음 어긋났는지는 전혀 알려주지 않는다.** 이것이 강화학습의 고전적인 **credit assignment 문제**의 reasoning 버전이다 — 마지막에 받은 스칼라 하나로 그 이전의 수십 개 결정 중 무엇이 잘못이었는지 역산해야 하는데, 정보량 자체가 부족하다.

일상적인 비유로 이해해보자. 자동차 조립 라인에 두 가지 검사 방식이 있다고 하자. 방식 A는 완성차 시동만 걸어보고 합격/불합격을 매긴다. 시동이 안 걸리면 "불량"이라는 사실만 알 뿐, 브레이크 배선 문제인지 엔진 문제인지 도장 결함인지는 처음부터 다시 뒤져야 한다. 방식 B는 공정마다(배선, 엔진, 도장) 검사원이 개별적으로 서명한다. 불량이 나면 어느 공정 검사원의 서명이 빠졌는지 즉시 알 수 있다. ORM이 방식 A, PRM이 방식 B다.

## ORM: Cobbe (2021) 검증기의 연장선

ORM은 새로운 아이디어가 아니다. 논문은 자신들의 ORM 학습법이 GSM8K에서 verifier를 학습시킨 Cobbe et al. (2021)과 동일한 방법론이라고 명시한다. 절차는 이렇다.

1. generator(사전학습된 언어모델)로 문제당 여러 개의 solution을 균등 샘플링한다.
2. 각 solution의 최종 답을 정답과 자동으로 대조해 정오 라벨 $$c \in \{0, 1\}$$을 매긴다. MATH 데이터셋은 답이 기계적으로 채점 가능하므로 이 라벨링에 사람이 필요 없다.
3. solution의 **마지막 토큰**에서 ORM이 예측한 확률 $$p_{\text{ORM}}(y)$$을 solution 전체의 점수로 쓴다.

학습 손실은 표준 binary cross-entropy다.

$$
\mathcal{L}_{\text{ORM}}(\theta) = -\Big[ c \log p_{\text{ORM}}(y) + (1-c)\log\big(1-p_{\text{ORM}}(y)\big) \Big]
$$

여기서 $$y$$는 solution 전체, $$c$$는 자동 채점으로 얻은 정오 라벨, $$p_{\text{ORM}}(y)$$는 ORM이 solution 마지막 토큰에서 내놓는 "이 solution이 맞았을 확률"이다.

문제는 이 $$c$$ 자체가 노이즈를 안고 있다는 점이다. 논문은 이를 명시적으로 지적한다 — "자동 채점으로 만든 ORM 타겟은 완벽히 신뢰할 수 없다. 틀린 추론으로 정답에 도달한 solution은 잘못 채점된다(false positive)." 채점 대상이 되는 라벨 자체가 이미 오염돼 있으니, 그 라벨로 학습한 ORM도 같은 맹점을 물려받는다.

## PRM: 매 스텝을 채점하다

PRM은 solution을 스텝 $$s_1, s_2, \ldots, s_T$$로 나누고, 각 스텝이 끝나는 토큰 위치에서 "이 스텝까지가 맞는가"를 이진 분류로 예측한다. 구현이 영리한 지점은, 이 예측이 **단일 토큰**의 형태를 취한다는 것이다. 즉 새 아키텍처를 얹을 필요 없이 표준 언어모델 파이프라인 그대로, 스텝 끝에서 정답 토큰의 log-likelihood를 최대화하는 것만으로 학습이 끝난다.

$$
p_t = p_\theta(x, s_{\le t}) = \sigma\big(r_\theta(x, s_{\le t})\big)
$$

$$
\mathcal{L}_{\text{PRM}}(\theta) = -\sum_{t=1}^{T} \Big[ c_t \log p_t + (1-c_t)\log(1-p_t) \Big]
$$

- $$x$$: 문제, $$s_{\le t}$$: 첫 스텝부터 $$t$$번째 스텝까지의 prefix.
- $$r_\theta$$: PRM이 스텝 $$t$$ 끝 토큰에서 내놓는 스칼라 로짓, $$\sigma$$: 시그모이드로 이를 확률로 변환.
- $$p_t$$: "스텝 $$t$$까지는 옳다"는 예측 확률.
- $$c_t \in \{0,1\}$$: 사람이 스텝 $$t$$에 매긴 정오 라벨(뒤에서 다룰 neutral 라벨은 최종적으로 positive로 접힌다).
- $$T$$: 해당 solution의 총 스텝 수.

논문은 스텝 라벨을 첫 번째 오류 스텝까지만 부여한다. 정답 solution은 어차피 모든 스텝이 옳다는 동일한 정보를 ORM·PRM 모두에게 주지만, 오답 solution에서는 ORM이 "어딘가 틀렸다"만 알려주는 반면 PRM은 "정확히 몇 번째 스텝까지는 맞고 그다음이 틀렸다"는 위치 정보까지 준다. 이 위치 정보가 곧 credit assignment 문제의 답이다.

| 항목                     | ORM                                 | PRM                        |
| ------------------------ | ----------------------------------- | -------------------------- |
| 감독 신호                | solution 전체에 스칼라 1개          | 스텝마다 스칼라($$T$$개)   |
| 라벨 출처                | 자동 채점(최종 답 매칭)             | 사람 라벨러(PRM800K)       |
| 예측 위치                | solution 마지막 토큰                | 매 스텝 마지막 토큰        |
| 핵심 취약점              | false positive(틀린 과정 + 맞은 답) | 라벨링 비용                |
| credit assignment        | 불가능 — 오류 위치를 모른다         | 가능 — 첫 오류 스텝을 특정 |
| 학습에 필요한 base model | GPT-4 base(사전학습만, RLHF 이전)   | GPT-4 base(동일)           |

실험은 두 모델 모두 RLHF 이전의 GPT-4 base 체크포인트에서 파인튜닝했다. 그리고 중요한 스코프 제한 하나 — 이 논문은 PRM을 PPO 같은 RL 루프에 넣어 정책을 직접 파인튜닝하지 않는다. "generator를 RL로 파인튜닝하는 것은 자연스러운 다음 단계지만, 의도적으로 이 작업의 범위 밖에 둔다"고 명시한다. 대신 PRM을 **best-of-N 검색의 verifier(랭커)**로만 써서 "가장 신뢰할 수 있는 reward model을 만드는 것" 자체에 집중한다. 즉 이 글의 PRM은 정책을 학습시키는 RL reward라기보다, 여러 후보 중 가장 나은 것을 골라내는 채점관에 가깝다. 이 verifier 관점은 뒤에 [32편 Generative Verifiers](/blog/2026/generative-verifiers/)로 이어진다.

# Method

## PRM800K: 80만 개 스텝 라벨

PRM800K는 사람 라벨러가 MATH 문제에 대한 model-generated solution의 각 스텝에 라벨을 매긴 데이터셋이다. 라벨은 세 종류다.

| 라벨     | 기호   | 의미                                                           |
| -------- | ------ | -------------------------------------------------------------- |
| Positive | $$+1$$ | 스텝이 정확하고 타당하며, 풀이에 진전이 있다                   |
| Neutral  | $$0$$  | 틀리지는 않았지만 진전이 없거나, 미묘하게 오도하는 애매한 스텝 |
| Negative | $$-1$$ | 명백히 틀렸거나 타당하지 않은 스텝                             |

neutral을 별도로 둔 이유가 설계 의도를 잘 보여준다. 라벨러가 "이 스텝이 틀린 건 아닌데 뭔가 찜찜하다"고 느낄 때 억지로 positive/negative 이분법에 끼워 맞추게 하지 않고, **애매함 자체를 하나의 라벨로 인정**한 것이다. 대신 이 판단을 나중으로 미룬다 — 평가 시점에 neutral을 positive로 볼지 negative로 볼지는 사후에 정할 수 있게 설계했다.

<p align="center"><img src="/assets/post/image/lets-verify-step-by-step/fig1-labeling-interface.png" width="70%"></p>

위 스크린샷이 실제 라벨 수집 인터페이스다(논문 Figure 1). 라벨러는 generator가 만든 solution을 스텝 단위로 훑으며 한 줄씩 세 라벨 중 하나를 클릭한다.

| 항목                             | 값                                         |
| -------------------------------- | ------------------------------------------ |
| 총 스텝 라벨 수                  | 800,000개                                  |
| 라벨링된 solution 수             | 75,000개                                   |
| 대상 문제 수                     | 12,000개                                   |
| 학습셋에 포함시킨 MATH test 문제 | 4,500개                                    |
| 평가 전용 held-out 문제          | 500개                                      |
| screening 통과 기준              | QC 문항 30개 중 75% 이상 gold label과 일치 |

라벨링은 두 phase로 나뉜다. Phase 1에서는 라벨러가 solution의 **모든** 대안 스텝에 라벨을 매겼고, 심지어 모든 completion이 negative로 나오면 라벨러가 직접 positive 스텝을 작성하기도 했다. 그런데 이 방식은 라벨러가 "뻔히 정답인 긴 solution"을 처음부터 끝까지 훑는 데 시간을 낭비하게 만들었다. Phase 2에서는 **첫 번째 오류 스텝 이후로는 라벨링을 중단**하도록 절차를 바꿨고, 동시에 그 시점까지 가장 성능이 좋은 PRM으로 다음에 라벨링할 solution을 선별하는 active learning을 도입했다.

라벨러 품질 관리도 촘촘하다. Phase 2 진입 전 30개 QC 문항으로 스크리닝해 gold label과 75% 이상 일치하는 라벨러만 통과시켰고, 이후에도 생성마다 10~20개의 QC 문항을 무작위로 섞어 계속 품질을 감시했다. 이 전체 과정 — 스크리닝, 지속적 QC, phase 전환 — 자체가 사람 라벨링의 **비용**이 이 접근법의 진짜 병목임을 보여준다. 80만 개 라벨은 공짜가 아니다.

## solution 점수 = 스텝 점수를 어떻게 합칠 것인가

PRM은 스텝마다 확률 $$p_t$$를 내놓지만, 여러 solution 후보를 서로 비교해 best-of-N을 고르려면 solution 하나에 스칼라 하나가 필요하다. 논문은 "PRM 점수 = 모든 스텝이 옳을 확률"로 정의하고, 이를 각 스텝 확률의 **곱**으로 구현한다.

$$
s_{\text{prod}}(y) = \prod_{t=1}^{T} p_t
$$

대안으로 가장 취약한 스텝 하나만 보는 **최솟값** 집계도 있다.

$$
s_{\min}(y) = \min_{1 \le t \le T} p_t
$$

- $$s_{\text{prod}}(y)$$, $$s_{\min}(y)$$: solution $$y$$ 전체에 매긴 최종 점수.
- $$p_t$$: 스텝 $$t$$의 정오 확률(앞서 정의).
- $$T$$: solution의 스텝 수.

곱을 쓰면 스텝이 많을수록(즉 풀이가 길수록) 아주 사소한 흠만 있어도 전체 점수가 급격히 깎이는 편향이 생긴다. 반대로 최솟값은 딱 하나의 가장 약한 스텝만 본다는 점에서 "가장 약한 고리가 전체를 결정한다"는 직관에 더 충실하다. 논문은 neutral을 positive로 볼지 negative로 볼지까지 더해 총 네 가지 조합을 Best-of-1860 기준으로 비교했다.

| 집계 방식       | neutral = positive | neutral = negative |
| --------------- | ------------------ | ------------------ |
| 곱(Product)     | **78.2%**          | 77.4%              |
| 최솟값(Minimum) | 77.6%              | 77.8%              |

네 조합의 성능 차이는 1%p 미만으로 크지 않지만, 곱 + neutral=positive가 가장 좋았고 논문은 이를 최종 채택했다. 곱을 쓰면 긴 solution에 불리한 편향이 생긴다는 점까지 인지한 채로 내린 선택이다.

## 토이 예제: 4단계 풀이 채점하기

가상의 4단계 수학 풀이 하나를 PRM이 채점했다고 하자. 각 스텝의 정오 확률이 다음과 같이 나왔다.

| 스텝 | 내용(가상)                        | PRM 확률 $$p_t$$ |
| ---- | --------------------------------- | ---------------- |
| 1    | 문제를 방정식으로 옮김            | 0.95             |
| 2    | 양변 정리                         | 0.90             |
| 3    | 부호 실수로 항 하나가 뒤바뀜      | 0.30             |
| 4    | (실수는 그대로 둔 채) 계산 마무리 | 0.85             |

곱 집계는 스텝을 하나씩 순서대로 곱해나간다.

$$
0.95 \times 0.90 = 0.855, \quad 0.855 \times 0.30 = 0.2565, \quad 0.2565 \times 0.85 \approx 0.218
$$

최솟값 집계는 네 값 중 가장 작은 것 하나만 뽑는다.

$$
\min(0.95,\ 0.90,\ 0.30,\ 0.85) = 0.30
$$

두 방식 모두 3번 스텝의 결함(0.30)을 최종 점수에 강하게 반영한다는 점에서는 같다. 다만 곱은 0.218로 더 가혹하게 깎고, 최솟값은 0.30으로 상대적으로 덜 깎는다 — 앞서 말한 "곱은 스텝 수가 많을수록 불리해지는" 편향이 여기서 확인된다.

이제 핵심 대비다. 이 풀이는 3번 스텝에서 부호를 잘못 뒤집었지만, 공교롭게 4번 스텝에서 또 다른 실수 없이 계산을 마무리해 **최종 답은 정답과 일치**했다고 하자.

- **ORM**은 마지막 토큰만 본다. 최종 답이 정답과 일치하므로, 학습 데이터의 "정답=옳은 풀이"라는 상관관계를 그대로 따라 높은 점수(예: 0.9 이상)를 줄 공산이 크다. 3번 스텝의 부호 실수는 ORM의 시야에 전혀 들어오지 않는다. 이것이 정확히 논문이 지적한 false positive다.
- **PRM**은 0.218(곱) 또는 0.30(최솟값)이라는 낮은 점수를 준다. 최종 답이 맞았다는 사실과 무관하게, 3번 스텝에서 이미 감점 요인을 포착했기 때문이다.

만약 이 solution과, 처음부터 끝까지 모든 스텝이 실제로 견고한 다른 solution이 best-of-N 후보로 경쟁한다면, ORM은 둘 다 "정답"으로 보고 거의 동률 취급하지만 PRM은 후자를 명확히 더 높은 점수로 골라낸다. best-of-N 검색에서 이 차이가 누적되면 72.4%와 78.2%라는 격차로 나타난다.

## Active learning: 라벨링 효율 2.6배

800K 라벨을 무작위로 균등하게 모으는 대신, 논문은 라벨러의 시간을 "라벨러를 잘 속일 것 같은" 즉 **convincing한 오답 solution**에 집중시키는 active learning을 도입했다. 절차는 이렇다.

1. 문제당 solution 1개씩만으로 작은 선별용 모델 $$\text{PRM}_{\text{selector}}$$를 학습시킨다.
2. $$\text{PRM}_{\text{selector}}$$로 문제당 1,000개의 후보 solution 각각에 점수를 매긴다.
3. 다음 라벨링 대상 $$N$$개를 고를 때, 그중 80%는 $$\text{PRM}_{\text{selector}}$$ 기준 가장 convincing한 **오답** solution으로, 나머지 20%는 정답·오답 구분 없이 가장 convincing한 나머지 solution으로 채운다.

<p align="center"><img src="/assets/post/image/lets-verify-step-by-step/fig4a-active-learning.png" width="70%"></p>

균등 샘플링과 active learning 각각으로 학습한 PRM의 성능 곡선(Figure 4a)을 놓고 기울기를 비교한 결과, **active learning이 균등 샘플링 대비 약 2.6배 데이터 효율적**이었다. 같은 성능에 도달하는 데 필요한 사람 라벨 수가 1/2.6로 줄어든다는 뜻이다. 다만 문제당 200개 샘플까지 선택 폭을 넓히자(선별 풀 자체가 1,000개뿐이라) 다양성이 줄면서 예상 추세보다 살짝 낮은 성능을 보였다고 저자들은 덧붙인다.

# Experiments

## MATH best-of-N: 72.4% vs 78.2%

대규모 실험에서는 ORM을 문제당 100개 균등 샘플로, PRM은 PRM800K 전체로 학습했다. 두 학습셋은 규모도 소스도 다르므로 직접적인 공정 비교라기보다는 "각 감독 방식이 낼 수 있는 최선"끼리의 비교다. 평가는 문제당 최대 1,860개 solution을 generator로 뽑아 best-of-N 검색을 수행하는 방식으로 진행했다.

<p align="center"><img src="/assets/post/image/lets-verify-step-by-step/fig3-best-of-n.png" width="60%"></p>

| 방법            | Best-of-1860 정확도 |
| --------------- | ------------------- |
| Majority Voting | 69.6%               |
| ORM             | 72.4%               |
| **PRM**         | **78.2%**           |

Majority voting(가장 흔한 답을 다수결로 채택)은 이미 강력한 베이스라인으로 알려져 있는데, ORM은 여기서 겨우 2.8%p 위다. PRM은 majority voting보다 8.6%p, ORM보다 5.8%p 높다. 논문은 $$N$$이 커질수록(즉 검색 폭이 넓어질수록) PRM과 ORM의 격차가 더 벌어진다고 보고한다 — 후보가 많아질수록 "그럴듯하지만 틀린" solution도 늘어나는데, ORM은 이런 solution에 더 잘 속기 때문이다.

## 분포 밖(OOD) 일반화: 최신 AP·AMC 시험

MATH 사전학습 데이터가 만들어진 이후 출제된 최신 AP Calculus·AP Chemistry·AP Physics·AMC10/12 문제로 별도의 held-out 테스트를 구성해, 모델이 학습 데이터를 암기한 게 아니라 실제로 일반화하는지 확인했다. 평가는 best-of-100 기준이다.

| 시험         | 문항 수 | ORM   | PRM       | Majority Vote |
| ------------ | ------- | ----- | --------- | ------------- |
| AP Calculus  | 45      | 68.9% | **86.7%** | 80.0%         |
| AP Chemistry | 60      | 68.9% | **80.0%** | 71.7%         |
| AP Physics   | 45      | 77.8% | **86.7%** | 82.2%         |
| AMC10/12     | 84      | 49.1% | **53.2%** | 32.8%         |
| 합계         | 234     | 63.8% | **72.9%** | 61.3%         |

네 시험 전부에서 PRM이 ORM과 majority voting을 모두 앞선다. 학습 시점 이후 출제된, 모델이 사전학습 중 접했을 가능성이 사실상 없는 신규 문제에서도 이 격차가 유지된다는 것은 PRM이 단순 암기가 아니라 실제로 더 신뢰할 수 있는 채점 능력을 학습했음을 뒷받침한다.

## 오염(contamination) 캐비엇

MATH 테스트 문제 중 일부가 온라인에 이미 논의된 상태라, 사전학습 데이터에 완전히 걸러지지 않고 섞여 들어갔을 가능성을 논문 스스로 인정한다. 문자열 매칭 휴리스틱으로 최대한 걸러냈지만 사람이 살짝 바꿔 쓴 재서술까지는 잡아내지 못한다. 다만 저자들은 이런 오염이 있다 해도 ORM과 PRM 모두에 비슷하게 영향을 줄 것이므로, 둘 사이의 **상대적** 비교는 유효하다고 본다.

# Conclusion

한 줄로 요약하면: **최종 답만 채점하는 ORM은 틀린 과정으로 맞은 답을 걸러내지 못하고 어디서 틀렸는지도 알려주지 못하지만, 매 스텝을 채점하는 PRM은 이 두 문제를 동시에 해결하며 MATH best-of-1860에서 72.4% → 78.2%, OOD best-of-100에서 63.8% → 72.9%의 개선을 만들어낸다.**

정리하면,

1. **문제**: outcome supervision은 sparse reward이고, 그 결과 false positive(틀린 과정+맞은 답)와 credit assignment 불가라는 두 결함을 동시에 안는다.
2. **해법**: 스텝마다 사람이 positive/negative/neutral 라벨을 매긴 PRM800K(80만 라벨, 7만 5천 solution, 1만 2천 문제)로 process reward model을 학습한다. 여러 스텝 점수는 곱(또는 최솟값)으로 합쳐 solution 하나의 점수로 만든다.
3. **비용**: active learning으로 2.6배 효율을 얻었어도, 결국 800K 라벨은 사람이 직접 매긴 것이다. 이 사람 라벨링 비용이 이 접근법 전체의 병목이다.

이 병목이 바로 다음 글의 존재 이유다. [29편 Math-Shepherd](/blog/2026/math-shepherd/)는 "사람 라벨 없이 PRM을 어떻게 학습시킬 것인가"라는 질문에 자동 라벨링으로 답한다. 그리고 [10편](/blog/2026/reward-model-overoptimization/)에서 정량화한 과최적화 문제를 생각하면, dense한 process reward가 sparse한 outcome reward보다 Goodhart 현상에 더 강건할 가능성도 있다 — 매 스텝을 검증하는 만큼 정책이 "요행수 답"으로 빠져나갈 틈이 좁아지기 때문이다. [30편 DeepSeek-R1](/blog/2026/deepseek-r1/)은 이와는 또 다른 방향에서, 사람 라벨도 학습된 PRM도 아닌 **규칙 기반 reward**로 검증 가능성을 확보하는 길을 보여준다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물여덟 번째 글이다.

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
  <li><a href="/blog/2026/warm-weight-averaged-reward/">WARM (2024)</a> — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="14">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
  <li><a href="/blog/2026/deliberative-alignment/">Deliberative Alignment (2024)</a> — 안전 명세를 모델의 추론 안으로</li>
  <li><a href="/blog/2026/shallow-safety-alignment/">Shallow Safety Alignment (2024)</a> — 정렬은 첫 몇 토큰에만 얹혀 있다</li>
  <li><a href="/blog/2026/or-bench/">OR-Bench (2024)</a> — 과잉 거절을 어떻게 측정할 것인가</li>
</ol>

**5부. reward를 정책으로**

<ol start="19">
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

<ol start="28">
  <li><strong>(현재 글)</strong> Let's Verify Step by Step (2023) — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="31">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="36">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="41">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 42편으로 구성된다.

# 참고 문헌

- Lightman et al., 2023. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (arXiv:2305.20050).
- [Let's Verify Step by Step, ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html).
- [OpenAI/prm800k GitHub repository](https://github.com/openai/prm800k) — PRM800K 데이터셋과 라벨링 가이드라인.
- [arXiv HTML(ar5iv) 렌더링](https://ar5iv.labs.arxiv.org/html/2305.20050) — 본문 그림 출처.
