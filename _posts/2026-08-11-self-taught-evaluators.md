---
layout: post
title: "Self-Taught Evaluators: 사람 라벨 없이 judge를 키우다"
date: 2026-08-11 09:23:30 +0900
description: "RLHF Reward 설계 시리즈 #25 — 합성 데이터만으로 LLM judge를 반복 자기개선시키는 법"
categories: [paper]
tags: [rlhf, reward-model, llm-as-a-judge, synthetic-data, self-improvement, paper]
giscus_comments: true
related_posts: true
---

> [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666) (Wang et al., Meta FAIR, arXiv 2024)

# Introduction

이 시리즈는 지금까지 계속 같은 벽에 부딪혀 왔다. **사람 선호 라벨이라는 병목**이다. [#3 HH-RLHF](/blog/2026/anthropic-hh-rlhf/)는 라벨러 간 합의율이 63%밖에 안 된다고 보고했다. 사람 셋 중 한 명 이상은 "이게 더 낫다"는 판정에 동의하지 않는다는 뜻이다. [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)는 한발 더 나가서, 실제로 그 라벨의 25%가 뒤집혀 있다고 정량화했다. [#6 Skywork-Reward](/blog/2026/skywork-reward/)는 아예 "좋은 아키텍처보다 좋은 데이터가 이긴다"며 데이터 큐레이션에 모든 힘을 쏟았다. 세 편 모두 결국 같은 질문으로 수렴한다 — **사람이 매기는 선호 라벨을 어떻게 더 깨끗하게, 더 많이 모을 것인가.**

이 논문은 그 질문 자체를 치운다. "라벨을 어떻게 더 잘 모을까"가 아니라 **"라벨을 아예 안 모으면 어떻게 되는가"**를 묻는다. 사람이 단 한 건도 "A가 B보다 낫다"고 표시하지 않은 상태에서 출발해, LLM judge를 반복적으로 자기 자신보다 나은 버전으로 개선시킨다. 그 결과 Llama3-70B-Instruct의 RewardBench 점수를 75.4에서 88.3으로, 다수결 투표를 쓰면 88.7까지 끌어올렸다. 사람 라벨로 학습한 동급 모델(85.6)과 GPT-4(84.3)를 모두 넘어선 숫자다.

비유하자면 이렇다. 지금까지의 RM 학습은 채점 기준을 배우기 위해 매번 선생님(사람 라벨러)을 불러야 하는 학원이었다. 이 논문은 그 학원을 없애고, **문제를 낼 때부터 정답을 이미 알고 있도록 시험지를 설계**한 뒤 학생(judge 모델) 스스로 자기 답안을 채점하며 실력을 키우게 한다. 정답을 아는 이유는 시험지를 그렇게 만들었기 때문이지, 채점자가 따로 있어서가 아니다.

이 글에서 답할 질문은 세 가지다.

1. **핵심 루프**: 라벨 없는 지시문에서 시작해 judge를 반복 개선하는 4단계 사이클은 정확히 어떻게 도는가.
2. **트릭의 정체**: "정답을 이미 아는 대조 쌍"을 사람 없이 어떻게 만드는가. 그리고 이게 [#20 Math-Shepherd](/blog/2026/math-shepherd/)의 자동 라벨링과 왜 같은 발상인가.
3. **대가**: 자기가 만든 데이터로 자기를 학습시키는 이 구조는 편향을 증폭시키지 않는가. 논문은 이 위험을 어디까지 막았고, 어디를 열어뒀는가.

# Background

## LLM-as-a-Judge와 이 시리즈의 6부

[#22 Prometheus 2](/blog/2026/prometheus-2/)부터 [#24 Generative Reward Models](/blog/2026/generative-reward-models/)까지, 이 시리즈 6부는 "reward를 스칼라 하나가 아니라 텍스트 생성으로 뽑자"는 흐름을 다뤘다. judge 모델에게 두 응답 $$y_1, y_2$$를 보여주고, 어느 쪽이 지시문 $$x$$를 더 잘 만족하는지 자연어 추론(reasoning trace)을 거쳐 최종 판정(verdict)을 내리게 하는 방식이다. 이 판정 방식 자체는 Zheng et al.(2023)의 LLM-as-a-Judge 이후 표준으로 자리 잡았고, [#9 RewardBench 2](/blog/2026/rewardbench-2/)가 다루는 평가 벤치마크도 이 판정 형식을 전제로 만들어졌다.

문제는 이 judge를 **학습**시키려면 여전히 "어느 쪽이 나은지" 알려주는 선호 데이터가 필요하다는 점이었다. 6부의 앞선 글들도 이 지도 신호를 사람 선호 데이터(HelpSteer2 등)에서 가져왔다. 이 논문이 6부의 흐름에서 갖는 위치는 명확하다 — **판정 형식(reasoning trace + verdict)은 그대로 두고, 그 형식을 학습시키는 지도 신호만 사람에서 합성으로 바꾼다.**

## 자기 학습(self-training)의 오래된 위험

모델이 자기 출력을 다시 자기 학습 데이터로 쓰는 아이디어 자체는 새롭지 않다. Zelikman et al.(2022)의 STaR은 모델이 생성한 추론 체인 중 정답을 맞힌 것만 걸러 다시 학습시키는 self-taught reasoner를 제안했다. 이 논문의 이름("Self-Taught Evaluators")도 그 계보를 잇는다.

self-training의 오래된 위험은 **순환 논리(circularity)**다. 모델이 스스로 옳다고 판단한 것만 걸러 다시 학습하면, 모델은 자기 편향을 검증하는 게 아니라 강화한다. 이 논문의 핵심 설계는 이 순환을 끊는 데 있다 — "모델이 스스로 옳다고 믿는 것"이 아니라 **"데이터를 만든 사람(연구자)이 구성 과정에서 이미 알고 있는 정답"**을 필터 기준으로 쓴다. 이 차이가 Method 섹션의 전부다.

# Method

## 핵심 루프: 4단계 사이클

라벨 없는 지시문 $$x$$에서 출발해, judge $$M_0$$(=Llama3-70B-Instruct)를 아래 4단계로 반복 개선한다.

| 단계           | 하는 일                                                                                                                          | 새로 생기는 것                             |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| 1. 대조쌍 합성 | 지시문 $$x$$에 좋은 응답 $$y^w$$를 만들고, $$x$$를 변형한 지시문 $$x' = \phi(x)$$에 대한 응답을 $$x$$의 나쁜 응답 $$y^l$$로 사용 | 정답을 이미 아는 쌍 $$(x, y^w \succ y^l)$$ |
| 2. 판정 샘플링 | 현재 judge $$M_{i-1}$$이 $$(x, y^w, y^l)$$을 보고 reasoning trace와 verdict를 $$N=15$$번 생성                                    | 판정 후보 15개                             |
| 3. 거부 샘플링 | 알려진 정답 $$y^w \succ y^l$$과 일치하는 verdict만 남기고 나머지는 버림                                                          | 학습셋 $$D_i$$                             |
| 4. 재학습      | Llama3-70B-Instruct를 처음부터(base에서) $$D_i$$로 SFT                                                                           | 새 judge $$M_i$$                           |

그리고 $$M_i$$가 다음 라운드의 judge가 되어 2단계부터 다시 돈다. 이 논문은 이 사이클을 **5번** 돌린다.

여기서 3단계가 순환 논리를 끊는 지점이다. "정답과 일치하는 판정만 남긴다"고 할 때 그 정답은 모델이 스스로 정한 게 아니라, 1단계에서 **데이터를 만들 때 이미 확정된 값**이다. 즉 필터링 기준이 모델 외부에 있다. 다만 이 외부 기준 자체가 "변형된 지시문에 대한 응답은 원래 지시문에는 나쁜 응답일 것"이라는 **구성적 가정**에 의존한다는 점은 뒤에서 다시 짚는다.

## 진짜 트릭: 대조 쌍을 어떻게 만드는가

이 논문에서 가장 중요한 설계는 1단계, 특히 나쁜 응답 $$y^l$$을 만드는 방법이다. 순진한 접근은 judge에게 "일부러 나쁜 응답을 써봐"라고 시키는 것이다. 이 논문은 그렇게 하지 않는다. 대신,

1. 원래 지시문 $$x$$에 대해 강한 모델(Mixtral 8x22B Instruct)로 정상적인 좋은 응답 $$y^w$$를 만든다.
2. $$x$$와 "관련은 있지만 미묘하게 다른" 변형 지시문 $$x' = \phi(x)$$를 프롬프트로 생성한다.
3. $$x'$$에 대해서도 정상적으로 좋은 응답 $$y^l$$을 만든다.
4. $$y^l$$을 원래 지시문 $$x$$의 나쁜 응답으로 취급한다.

$$y^l$$은 그 자체로는 결코 조악한 텍스트가 아니다. 유창하고, 논리적이고, **다른 질문에는 완벽하게 맞는 답**이다. 다만 $$x$$가 묻는 것과는 어긋난다. 이렇게 만들어진 쌍이 진짜 어려운 negative가 된다 — 문법이 깨지거나 뜬금없는 소리를 하는 응답은 judge가 구분하기 너무 쉬워서 학습 신호로서 가치가 낮다.

비유하자면 시험지 바꿔치기다. 선생님이 "12를 3으로 나누면?"이라는 문제를 냈는데, 답안지에는 "15를 5로 나누면?"이라는 **살짝 바뀐 문제의 모범 답안**이 적혀 있다. 계산 자체는 완벽하지만 원래 문제에 대한 답으로는 틀렸다. 채점자 없이도 "이 답안은 이 문제의 정답이 아니다"를 이미 알고 있는 채로 채점 연습을 시킬 수 있다.

이 설계가 실제로 더 낫다는 것도 ablation으로 확인됐다. 나쁜 응답을 judge에게 직접 "나쁘게 써봐"라고 시켜 만든 경우 RewardBench 83.8 대비 80.7로, 지시문 변형 방식이 3점 이상 앞선다.

## 토이 예제: 사과 나누기 문제 한 줄 따라가기

작은 예제로 1~3단계를 그대로 밟아보자.

**원래 지시문** $$x$$: "철수가 사과 12개를 3명에게 똑같이 나눠줬다. 한 명당 몇 개씩 받았는지 계산 과정과 함께 설명해줘."

- **좋은 응답** $$y^w$$: "12를 3으로 나누면 4다. 따라서 한 명당 사과 4개씩 받는다." (정답, $$x$$에 정확히 답함)
- **변형 지시문** $$x' = \phi(x)$$: "철수가 사과 15개를 5명에게 똑같이 나눠줬다. 한 명당 몇 개씩 받았는지 계산 과정과 함께 설명해줘." (숫자만 바뀐, 관련은 있지만 다른 문제)
- **나쁜 응답** $$y^l$$: "15를 5로 나누면 3이다. 따라서 한 명당 사과 3개씩 받는다." ($$x'$$에는 정답이지만 $$x$$가 묻는 12÷3에는 답하지 않는다)

이제 $$(x, y^w, y^l)$$을 judge $$M_{i-1}$$에게 보여주고 "어느 응답이 $$x$$를 더 잘 만족하는가"를 reasoning trace와 함께 15번 판정시킨다. 알려진 정답은 $$y^w \succ y^l$$이다. 15번 중 이 정답과 일치하는 verdict(그리고 그때의 reasoning trace)만 남기고 나머지는 버린다. 남은 것들이 "12를 3으로 나눈 문제에 15를 5로 나눈 답을 붙이면 틀렸다"는 이유를 설명하는 추론 과정이므로, 이걸 학습하면 judge는 **표면적 유창함이 아니라 지시문과의 정합성**을 보는 법을 배운다.

## Math-Shepherd와 같은 발상, 다른 메커니즘

[#20 Math-Shepherd](/blog/2026/math-shepherd/)는 이 논문과 전혀 다른 대상(스텝 단위 process reward)을 다루지만, "사람에게 순위를 묻지 않고 정답을 이미 아는 상태에서 데이터를 만든다"는 발상은 정확히 같다.

| 항목                   | Self-Taught Evaluators                                                            | Math-Shepherd (#20)                                                |
| ---------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 라벨이 필요한 지점     | 응답 쌍 중 어느 쪽이 나은가                                                       | 추론 각 스텝이 좋은가 나쁜가                                       |
| 사람 라벨 대신 쓰는 것 | 지시문을 변형해 "정답을 이미 아는" 대조쌍을 구성                                  | 각 스텝에서 여러 번 rollout해 최종 정답 도달 비율로 스텝 품질 추정 |
| 정답의 출처            | 데이터 생성 설계 자체(구성적 정답)                                                | 문제의 golden answer + Monte Carlo 추정                            |
| 학습 대상              | pairwise judge (reasoning trace + verdict)                                        | step-level PRM (스칼라)                                            |
| 공통 발상              | 사람에게 "이게 낫다"고 묻는 대신, 정답을 이미 알고 있는 구조를 생성 과정에 심는다 | 동일                                                               |

두 논문 다 "사람 라벨"이라는 병목을 **관찰이 아니라 설계**로 우회한다. Math-Shepherd는 정답이 있는 수학 문제라는 도메인 특성(rollout으로 성공/실패를 셀 수 있음)을 이용했고, 이 논문은 지시문을 변형하는 생성 과정 자체에 정답을 박아 넣었다. 도메인 제약이 없다는 점에서 이 논문의 트릭이 더 범용적이다.

## 데이터 소스: WildChat에서 걸러낸 2만 건

지시문 풀은 WildChat 대화 로그에서 가져왔다. Mixtral 8x22B Instruct로 지시문을 카테고리 분류한 뒤, "reasoning" 카테고리에서 **20,582개**를 학습용으로 선별했다. 카테고리별로 따로 학습시켜 비교한 결과, reasoning 카테고리가 RewardBench 83.5로 가장 높았고 safety(79.6), coding(79.4), math/GSM8K(79.3)가 뒤를 이었다 — 모두 학습 전 시드 점수(75.4)보다는 높지만, reasoning만큼은 아니었다. 논문은 이를 "복잡도가 도전적이면서도 분포가 고르게 퍼져 있기 때문"이라고 설명한다.

<p align="center"><img src="/assets/post/image/self-taught-evaluators/wildchat_category.png" width="70%"></p>

위 그림은 선별된 지시문 풀의 카테고리 분포(논문 Figure 6)다. reasoning이 다른 카테고리와 균형 있게 섞여 있어, judge가 한 가지 패턴에만 과적합되지 않도록 돕는다.

<p align="center"><img src="/assets/post/image/self-taught-evaluators/wildchat_complexity.png" width="70%"></p>

위 그림(Figure 4)은 추론된 지시문 복잡도 분포다. 너무 쉬운 문제만 있으면 judge가 표면적 신호만으로도 정답을 맞혀 학습 신호가 무의미해지고, 너무 어려운 문제만 있으면 애초에 $$y^w$$ 자체의 품질이 흔들린다. 적당히 어려운 분포를 유지하는 것이 이 파이프라인이 조용히 신경 쓰는 부분이다.

# Experiments

## 메인 결과: 75.4 → 88.3, 사람 라벨 없이

| 모델 / 방법                                  | RewardBench Overall |
| -------------------------------------------- | ------------------- |
| 시드: Llama3-70B-Instruct (0-shot judge)     | 75.4                |
| Self-Taught, Iteration 1                     | 83.9                |
| Self-Taught, Iteration 2                     | 86.0                |
| Self-Taught, Iteration 3                     | 87.5                |
| Self-Taught, Iteration 4                     | 87.7                |
| **Self-Taught, Iteration 5**                 | **88.3**            |
| Self-Taught, Iteration 5 + 다수결(32-sample) | **88.7**            |
| Labeled HelpSteer2로 학습한 judge            | 85.6                |
| Labeled 데이터로 동일하게 5회 반복(정체)     | 87.0                |
| GPT-4-0125 (judge)                           | 84.3                |
| Gemini 1.5 Pro (judge)                       | 88.1                |

가장 흥미로운 줄은 "Labeled 데이터로 동일하게 5회 반복"이다. 사람이 만든 HelpSteer2로 똑같은 반복 학습 절차를 5번 돌려도 87.0에서 정체됐다. 반면 합성 데이터는 88.3까지 올라갔다. 데이터가 사람 라벨이냐 합성이냐보다, **반복 사이클이 학습 신호를 얼마나 계속 정제해주는가**가 더 크게 작용한다는 뜻이다. 처음 한 번(iteration 1)에서 75.4 → 83.9로 8.5점이 뛰고, 이후 iteration당 개선폭은 2.1점(iter2) → 1.5점(iter3) → 0.2점(iter4) → 0.6점(iter5)으로 줄어든다 — 전형적인 수확체감 곡선이다.

## 부트스트랩의 위험: 어디까지 막았고, 어디를 열어뒀나

자기가 만든 데이터로 자기를 반복 학습시키는 구조는 필연적으로 "내가 이미 믿는 것을 더 강하게 믿게 되는" 위험을 안는다. 이 논문이 이 위험을 다루는 방식과, 다루지 못한 부분을 정직하게 나눠보자.

**막은 부분**:

- **필터 기준이 모델 외부에 있다.** 3단계의 거부 샘플링은 judge의 자기 일관성(self-consistency)이 아니라, 1단계에서 구성 시점에 이미 정해진 정답 $$y^w \succ y^l$$과 일치하는지를 본다. 모델이 "내가 맞다고 생각하는 것"이 아니라 "설계자가 맞다고 정해둔 것"을 정답으로 쓰므로, 순수 자기 확신 강화 루프보다는 안전하다.
- **생성 모델과 판정 모델이 다른 계열이다.** 응답과 지시문 변형은 Mixtral 8x22B Instruct가, 최종 judge는 Llama3-70B-Instruct가 맡는다. 같은 모델이 자기 출력을 자기가 채점하는 구조를 피해 self-preference 편향을 어느 정도 줄인다.

**열어둔 부분** — 논문이 스스로 밝힌 한계:

| 한계        | 내용                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 모델 크기   | 70B급 모델에서만 검증. 더 작은 모델에서도 통하는지는 미확인                                                         |
| 콜드 스타트 | 시드 모델이 이미 어느 정도 합리적인 판정을 낼 수 있어야 사이클이 굴러간다 — 완전히 못하는 모델에서 시작할 수는 없다 |
| 평가 범위   | pairwise 비교만 다뤘고, 단일 응답 채점(pointwise)은 향후 과제로 남김                                                |
| 추론 비용   | reasoning trace를 생성하는 만큼 단순 스칼라 judge보다 추론 비용이 크다                                              |

**논문이 직접 다루지 않은 부분**도 있다. 3단계의 "정답"은 결국 "변형된 지시문에 대한 좋은 응답은 원래 지시문에는 나쁜 응답일 것"이라는 **구성적 가정**에 기대고 있다. 이 가정은 대부분 성립하지만 항상 성립한다는 보장은 없다 — 변형 폭이 작으면 두 지시문의 요구사항이 실제로 겹칠 수 있고, 그 경우 "정답"이라고 필터링한 라벨 자체가 조용히 틀린다. [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)가 지적한 "사람 라벨의 25%가 뒤집혀 있다"는 문제를, 이 논문은 사람 대신 **구성 과정의 가정**이라는 형태로 남겨둔 셈이다. 다만 사람 라벨과 달리 이 가정은 저렴하게 반복 검증하고 고칠 수 있다는 점이 실질적인 차이다. 또한 reasoning trace가 진짜로 그 추론을 거쳐 정답에 도달했는지, 아니면 정답을 먼저 맞히고 그럴듯한 설명을 사후에 붙였는지(post-hoc rationalization)는 이 논문의 필터링 방식으로는 구분되지 않는다 — 이 지점은 뒤에 [#27 ReasonGRM](/blog/2026/reasongrm/), [#28 J1](/blog/2026/j1-thinking-judge/)이 정면으로 다룬다.

# Conclusion

핵심을 한 줄로 정리하면: **Self-Taught Evaluators는 "정답을 이미 아는 대조 쌍을 설계"하고 "그 정답으로 자기 판정을 거부 샘플링"하는 두 가지 장치로 순환 논리를 끊어, 사람 선호 라벨 없이도 Llama3-70B-Instruct를 RewardBench 75.4에서 88.3(다수결 88.7)까지 끌어올렸다.**

정리하면,

1. **핵심 루프**: 대조쌍 합성 → judge의 판정 샘플링($$N=15$$) → 정답과 일치하는 것만 거부 샘플링 → 재학습, 이 4단계를 5번 반복한다.
2. **진짜 트릭**: 나쁜 응답을 직접 만들지 않고, 지시문을 변형해 그 변형 지시문의 좋은 응답을 원 지시문의 나쁜 응답으로 재활용한다. 이 방식이 직접 나쁘게 쓰는 것(80.7)보다 명확히 낫다(83.8).
3. **결과**: 사람 라벨 없이 사람 라벨 기반 모델(85.6)과 GPT-4(84.3)를 모두 넘어섰다. 다만 개선폭은 iteration이 반복될수록 빠르게 줄어든다.
4. **한계**: 필터 기준을 모델 밖에 두어 순수 자기강화 루프는 피했지만, 그 외부 기준 자체가 구성 시점의 가정에 의존한다. reasoning trace의 신뢰성(진짜 추론인지 사후 합리화인지)도 이 논문만으로는 검증되지 않는다.

다음 글([#26 DeepSeek-GRM / SPCT](/blog/2026/deepseek-grm-spct/))은 이 논문이 남긴 "지시문 변형이라는 하나의 트릭에 의존한다"는 지점을 다른 각도에서 밀고 나간다. judge가 대조쌍을 외부에서 받는 대신 **스스로 평가 원칙(principle)을 생성**하고, 그 원칙에 따라 스스로를 채점하도록 inference-time에 확장하는 방식이다. 사람 라벨을 치운 이 논문의 다음 단계는, judge가 무엇을 기준으로 판정하는지까지 스스로 정하게 만드는 것이다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물다섯 번째 글이다.

**1부. 지형도**

1. [Deep RL from Human Preferences (Christiano 2017)](/blog/2026/deep-rl-human-preferences/) — 선호로 보상을 배우는 원형
2. [InstructGPT (Ouyang 2022)](/blog/2026/instructgpt/) — RLHF 3단계 표준 레시피
3. [HH-RLHF (Bai 2022)](/blog/2026/anthropic-hh-rlhf/) — helpful·harmless preference model

**2부. 스칼라 RM 해부**

4. [Rethinking Bradley-Terry (2024)](/blog/2026/bradley-terry-rethinking/) — reward 변환의 수학적 기반
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

22. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
23. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
24. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
25. **(현재 글)** Self-Taught Evaluators (2024) — 사람 라벨 없이 judge를 키우다
26. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

본 시리즈는 31편으로 구성된다.

# 참고 문헌

- Wang et al., 2024. [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666).
- [Self-Taught Evaluators — arXiv HTML](https://arxiv.org/html/2408.02666v2).
- [Self-Taught Evaluators — Hugging Face Papers](https://huggingface.co/papers/2408.02666).
- Zheng et al., 2023. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).
- Zelikman et al., 2022. [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465).
