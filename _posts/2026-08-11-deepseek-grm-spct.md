---
layout: post
title: "DeepSeek-GRM: reward model이 평가 기준을 스스로 만든다"
date: 2026-08-11 09:24:00 +0900
description: "RLHF Reward 설계 시리즈 #26 — SPCT로 원칙과 critique를 생성하고, inference-time scaling으로 training-time scaling을 이기다"
categories: [paper]
tags: [rlhf, reward-model, genrm, inference-time-scaling, deepseek, paper]
giscus_comments: true
related_posts: true
---

> [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) (Liu et al., DeepSeek-AI, arXiv 2025)

# Introduction

이 시리즈 [21편 DeepSeek-R1](/blog/2026/deepseek-r1/)은 규칙(rule)이 그대로 reward가 되는 경우를 다뤘다. 수학 문제는 정답이 하나뿐이라, 최종 답만 정답 파서로 확인하면 완벽하게 검증 가능한 reward를 공짜로 얻는다. [22편 Generative Verifiers](/blog/2026/generative-verifiers/)는 이 reward를 "next-token prediction"으로 표현하는 법을, [23편 Generative Reward Models](/blog/2026/generative-reward-models/)는 GenRM을 선호 학습과 결합하는 법을 각각 보여줬다. 세 글 모두 공통된 전제가 있다. **"무엇이 좋은 응답인가"를 판정할 기준이 이미 존재하거나, 최소한 사람이 미리 정해줄 수 있다**는 전제다.

일반 도메인(general domain) — instruction-following, 상담, 코딩 리뷰, 창작 — 에는 이 전제가 무너진다. 논문은 이 문제를 다음과 같이 짚는다.

> "특정 도메인의 고품질 reward는 대개 명확한 조건을 가진, 사람이 설계한 환경에서 얻거나(예: 게임의 승패 규칙), 검증 가능한 문제(예: 수학 문제)에 대해 손으로 짠 규칙에서 얻는다. 일반 도메인에서는 reward 생성이 훨씬 더 어렵다. 기준이 더 다양하고 복잡하며, 명시적인 참조 답이나 정답이 없는 경우가 많기 때문이다."

이 문제에 [7편 ArmoRM](/blog/2026/armorm/)이 내놓은 답은 "기준을 고정하고 분해하자"였다. helpfulness, correctness, coherence, complexity, verbosity 등 19개의 고정된 축으로 응답을 채점한 뒤 MoE 게이팅으로 가중합을 냈다. 그런데 이 축들은 **모든 입력에 똑같이 적용된다.** "이별한 친구를 위로하는 메시지"와 "SQL 인젝션 취약점을 찾는 코드 리뷰"는 좋은 응답의 기준이 근본적으로 다른데, 같은 19개의 자로 잰다.

DeepSeek-GRM(방법론 이름은 Self-Principled Critique Tuning, 이하 SPCT)의 답은 반대 방향이다. **기준(principle) 자체를 모델이 매 입력마다 스스로 새로 만들게 한다.** 그리고 이 논문이 시리즈에서 특별한 이유가 하나 더 있다. reward 모델을 크게 키우는 대신(training-time scaling), **같은 모델로 여러 번 채점하고 투표하는 쪽(inference-time scaling)이 더 이득**이라는 것을 정량적으로 보여준다. 27B급 DeepSeek-GRM이 32번 샘플링해서 투표하면 종합 벤치마크 점수 72.8점을 받아, GPT-4o(71.3)와 671B급 DeepSeek-V3를 그리디(greedy) 디코딩으로 돌린 결과에 필적하는 성능을 낸다.

이 글은 세 가지를 순서대로 본다.

1. **왜 pointwise generative RM인가**: scalar / semi-scalar / generative 세 갈래, pointwise / pairwise 두 갈래가 만드는 여러 조합 중 왜 이 조합을 택했는지.
2. **SPCT가 원칙과 critique를 만드는 법**: rejective fine-tuning(cold start)과 rule-based online RL 두 단계, 그리고 같은 응답이라도 원칙이 바뀌면 점수가 어떻게 뒤집히는지 토이 예제로 확인한다.
3. **inference-time scaling이 왜, 얼마나 이기는가**: 단순 다수결이 아니라 meta RM이 저품질 샘플을 걸러내는 구조, 그리고 training-time scaling과의 정면 대결 숫자.

# Background

## 여섯 칸의 설계 공간

reward 모델을 설계할 때는 실은 독립된 두 가지 선택지가 있다. "무엇을 출력하는가"(reward generation paradigm)와 "몇 개의 응답을 한 번에 보는가"(scoring pattern)다. 논문은 전자를 scalar / semi-scalar / generative 세 갈래로, 후자를 pointwise / pairwise 두 갈래로 나눈다.

- **Scalar**: 응답을 보고 숫자 하나만 뱉는다. [5~8편](/blog/2026/skywork-reward/)에서 다룬 Bradley-Terry 계열 RM 대부분이 여기 속한다.
- **Semi-Scalar**: critique(자연어 설명)를 먼저 쓰고, 그 뒤에 숫자를 별도 헤드로 뽑는다. CLoud가 대표적이다.
- **Generative**: 숫자 헤드 없이, critique 텍스트 안에 점수까지 자연어로 녹여 낸다. LLM-as-a-Judge와 DeepSeek-GRM이 여기 속한다.

<p align="center"><img src="/assets/post/image/deepseek-grm-spct/fig2_paradigms.png" width="80%"></p>

위 그림(논문 Figure 2)이 이 여섯 조합과, 각 조합의 대표 방법·연산을 정리한다. 맨 아래 두 줄(Inference-Time Scalable / Input Flexible)이 핵심이다.

| 조합                       | 대표 방법          | Inference-Time Scalable | Input Flexible |
| -------------------------- | ------------------ | :---------------------: | :------------: |
| Scalar + Pointwise         | Bradley-Terry 계열 |           ❌            |       ✅       |
| Scalar + Pairwise          | PairRM             |           ❌            |       ❌       |
| Semi-Scalar + Pointwise    | CLoud              |           ✅            |       ✅       |
| Semi/Generative + Pairwise | LLM-as-a-Judge     |           ✅            |       ❌       |
| **Generative + Pointwise** | **DeepSeek-GRM**   |         **✅**          |     **✅**     |

- **Input Flexible**은 응답 하나(single), 둘(pairwise), 여럿(multi-response ranking)을 같은 형식으로 처리할 수 있느냐다. **pointwise**는 응답마다 독립적으로 점수를 매기므로 항상 유연하다. **pairwise**는 애초에 "둘을 비교"하는 구조라 응답이 하나거나 셋 이상이면 다시 설계해야 한다.
- **Inference-Time Scalable**은 같은 입력을 여러 번 샘플링했을 때 결과가 다양하게 갈려서 "투표"가 의미를 가지느냐다. scalar 헤드는 확률적으로 샘플링해도 숫자가 크게 안 흔들려서 투표할 거리가 없다. 반면 critique(자연어) 성분이 있으면 매 샘플링마다 다른 논거·다른 principle이 나와 투표가 성립한다.

여섯 조합 중 **Generative + Pointwise**만 두 속성을 동시에 만족한다. 논문이 pointwise GRM을 최종 선택으로 삼은 이유가 여기 있다. 일상 비유로 하면, scalar RM은 "채점 결과만 알려주는 자동 채점기"고, pairwise RM은 "둘 중 하나만 고르는 이지선다 심사위원"이다. pointwise GRM은 답안지 하나하나에 채점 이유를 적으면서 점수도 매기는 "서술형 채점 교사"에 가깝다 — 답안이 몇 장이든 같은 방식으로 채점할 수 있고, 채점 이유가 자연어라 여러 번 다시 채점시켜 의견을 모을 수도 있다.

## 누가 기준을 정하는가 — ArmoRM과의 대비

pointwise GRM을 택했다는 것과, principle을 **누가 만드는가**는 별개의 질문이다. 논문은 여기서도 갈림길을 만든다. principle을 사람이 미리 정해줄 수도 있고, 모델이 매번 새로 생성할 수도 있다. [7편 ArmoRM](/blog/2026/armorm/)과 DeepSeek-GRM은 정확히 이 지점에서 갈라진다.

| 구분           | ArmoRM (7편)                                                           | DeepSeek-GRM / SPCT                                      |
| -------------- | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| 평가 축의 출처 | 사람이 사전 정의 (helpfulness, correctness, coherence 등 19개 고정 축) | 모델이 입력마다 스스로 생성 (개수도 가변)                |
| 축의 개수      | 모든 입력에 동일                                                       | 입력마다 다름 (실사용 예시에서 보통 2~4개)               |
| 결합 방식      | 학습된 MoE 게이팅 네트워크가 축별 가중치 산출                          | 모델이 critique 안에서 스스로 가중치를 서술              |
| 해석가능성     | 축별 점수는 보이지만, 게이팅 가중치 자체는 블랙박스                    | critique 텍스트가 왜 그 principle을 썼는지 자연어로 설명 |
| 축을 늘리려면  | 사람이 새 축을 정의하고 재학습                                         | 이미 생성 과정의 일부라 별도 재설계 불필요               |

**정답이 있는 도메인은 사람이 기준을 정해도 된다.** 수학 문제의 "정답과 일치하는가"([21편](/blog/2026/deepseek-r1/))처럼 기준 자체가 자명하기 때문이다. 하지만 일반 도메인처럼 기준 자체가 입력마다 달라지는 상황에서는, 고정된 19개 축(ArmoRM)이든 몇 개의 rubric 문항이든 사람이 미리 정한 기준은 항상 "이 입력엔 안 맞는 축이 섞여 있거나, 필요한 축이 빠져 있다"는 문제를 겪는다. SPCT는 이 문제를 아예 "기준을 정하는 일도 모델의 출력으로 만들어서" 피해간다. 이 선택이 [25편 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)가 다시 반대 방향(사람이 rubric을 준다)으로 돌아가는 이유이기도 하다 — principle을 모델이 만들면 유연하지만, 그 principle 자체가 틀렸을 때 검증할 방법이 없다는 대가가 따른다.

# Method

## Pointwise GRM의 형식

쿼리 $$x$$와 응답 집합 $$\{y_i\}_{i=1}^n$$이 주어지면, GRM은 두 단계로 출력을 만든다.

$$\{p_i\}_{i=1}^m \sim p_\theta(x, \{y_i\}_{i=1}^n)$$

$$C \sim r_\theta(x, \{y_i\}_{i=1}^n, \{p_i\}_{i=1}^m)$$

- $$x$$: 입력 쿼리.
- $$\{y_i\}_{i=1}^n$$: 채점 대상 응답 집합. $$n=1$$이면 단일 응답 채점, $$n=2$$면 pairwise 비교, $$n>2$$면 다중 응답 랭킹 — 모두 같은 수식으로 처리된다.
- $$p_\theta$$: principle을 생성하는 단계. 파라미터 $$\theta$$는 critique를 생성하는 모델과 **동일**하다. 즉 하나의 모델이 먼저 "이번엔 이런 기준으로 보겠다"를 스스로 정한다.
- $$\{p_i\}_{i=1}^m$$: 그렇게 생성된 $$m$$개의 principle. 각 principle은 자연어 설명과 가중치를 함께 갖는다.
- $$r_\theta$$: 생성된 principle을 조건으로 critique를 쓰는 단계. 같은 파라미터 $$\theta$$를 재사용한다.
- $$C$$: 최종 critique 텍스트. principle별 분석과, 거기서 파싱해낸 응답별 점수 $$\{S_i\}_{i=1}^n$$까지 포함한다.

principle이라는 개념 자체는 이 논문이 처음 쓴 말이 아니다. Constitutional AI가 "사람이 만든 규칙 대신 모델에게 원칙을 주고 스스로 비평하게 하자"는 아이디어를 먼저 냈다. SPCT의 기여는 그 원칙을 **사람이 주는 대신 모델이 매 입력마다 새로 생성하게** 만들고, 그 생성 능력 자체를 학습 목표로 삼았다는 데 있다.

## SPCT: 두 단계 학습

SPCT는 이름 그대로 "원칙에 기반한 스스로의 critique"를 **튜닝**한다. 학습은 두 단계로 진행된다.

**1단계 — Rejective Fine-Tuning (cold start).** 목표는 GRM이 정확한 형식으로 principle과 critique를 생성하고, 다양한 입력 형태($$n=1,2,\dots$$)를 다루도록 만드는 것이다. 여기서 "rejective"는 거부 샘플링을 뜻한다 — 예측 점수가 정답과 어긋나는 궤적, 그리고 모든 샘플이 이미 정답을 맞히는(너무 쉬운) 궤적을 학습 데이터에서 제거한다. 너무 쉬운 문제만 남으면 모델이 principle을 진지하게 만들 유인이 없어지기 때문이다. 어려운 케이스에는 정답 응답을 힌트로 살짝 흘려주는 hinted sampling도 함께 쓴다.

**2단계 — Rule-Based Online RL.** [16편 GRPO](/blog/2026/grpo-deepseekmath/)와 같은 알고리즘을 그대로 가져온다. 다만 16편에서 GRPO는 수학 풀이 정책을 학습하는 데 쓰였다면, 여기서는 **reward를 매기는 모델 자신**을 학습하는 데 쓰인다. reward 함수는 규칙 기반이며 단순하다.

$$
r_i =
\begin{cases}
1 & \text{예측한 최고 점수 응답 } = \text{ 정답 최고 응답} \\
-1 & \text{그 외}
\end{cases}
$$

- $$n \geq 2$$(pairwise·다중 응답)일 때는 예측 점수 $$\{S_i\}$$의 argmax가 실제 정답 랭킹의 1위와 일치하면 $$+1$$이다.
- $$n = 1$$(단일 응답 채점)일 때는 예측 점수가 참조 점수와 일치하면 $$+1$$이다.
- 그 외에는 $$-1$$. principle의 "내용"에는 직접 보상을 주지 않는다 — principle이 좋았는지는 오직 **그 principle을 써서 최종 점수가 맞았는가**로만 간접 평가된다.

이 단순한 규칙이 온라인 RL을 굴리는 이유는, principle의 좋고 나쁨을 사람이 일일이 라벨링할 수 없기 때문이다. 대신 "결과가 맞았다"는 신호만 주고, 어떤 principle을 만들어야 결과가 잘 맞는지는 모델이 스스로 탐색하게 둔다.

## 토이 예제 — 원칙이 바뀌면 순위가 뒤집힌다

SPCT가 만드는 principle이 실제로 얼마나 결과를 좌우하는지 작은 예제로 직접 계산해보자.

**쿼리**: "친구가 방금 실연 소식을 전하며 위로를 구하는 메시지를 보냈다. 어떻게 답장해야 할까?"

- 응답 A: "그럴 수도 있지, 다음에 더 좋은 사람 만나면 되지 뭐. 밥이나 먹으러 가자."
- 응답 B: "많이 힘들겠다. 지금은 무슨 말을 해도 위로가 안 될 수도 있는데, 그냥 내가 옆에 있다는 것만 알아줬으면 해. 얘기하고 싶으면 언제든 불러."

**원칙 세트 1 (공감 중심)**을 생성했다고 하자.

| Principle                                          | 가중치 $$w_k$$ | A 점수 $$s_{A,k}$$ | B 점수 $$s_{B,k}$$ |
| -------------------------------------------------- | :------------: | :----------------: | :----------------: |
| P1. 공감적 어조 — 감정을 인정하고 판단하지 않는가  |       5        |        2/10        |        9/10        |
| P2. 실질적 제안 — 부담 없는 다음 행동을 제시하는가 |       3        |        6/10        |        5/10        |
| P3. 간결성                                         |       2        |        9/10        |        6/10        |

가중합 $$S_i = \sum_k w_k \cdot s_{i,k} / \sum_k w_k$$을 계산하면,

$$S_A = \frac{2 \times 5 + 6 \times 3 + 9 \times 2}{10} = 4.6, \qquad S_B = \frac{9 \times 5 + 5 \times 3 + 6 \times 2}{10} = 7.2$$

원칙 세트 1로는 **B가 A를 압도**한다(7.2 > 4.6). 상식적으로 납득이 가는 결과다.

이번엔 같은 두 응답에 대해 **원칙 세트 2 (문제 해결 중심)**를 생성했다고 하자.

| Principle                                            | 가중치 $$w_k$$ | A 점수 $$s_{A,k}$$ | B 점수 $$s_{B,k}$$ |
| ---------------------------------------------------- | :------------: | :----------------: | :----------------: |
| P1. 문제 해결 지향성 — 다음 행동을 명확히 제시하는가 |       6        |        8/10        |        3/10        |
| P2. 감정 표현 절제 — 과도한 감정 언어를 피하는가     |       2        |        9/10        |        3/10        |
| P3. 반응의 간결함                                    |       2        |        9/10        |        6/10        |

$$S_A = \frac{8 \times 6 + 9 \times 2 + 9 \times 2}{10} = 8.4, \qquad S_B = \frac{3 \times 6 + 3 \times 2 + 6 \times 2}{10} = 3.6$$

**같은 응답 A, B인데 순위가 완전히 뒤집힌다**(A 8.4 > B 3.6). 원칙 세트 2는 "위로"라는 맥락에 안 맞는, 즉 이 쿼리에는 부적절한 principle 조합이다. 이 예제가 보여주는 것은 두 가지다.

1. principle을 모델이 스스로 만드는 유연성은, 뒤집어 보면 **좋은 principle을 만들지 못하면 채점 자체가 무의미해진다는 취약성**이기도 하다.
2. 그래서 principle 하나만 믿고 점수를 낼 수 없다. principle을 여러 번 다시 뽑아서, 그중 "이 맥락에 맞는" principle들이 다수가 되게 만들어야 한다 — 이것이 다음 절의 inference-time scaling이다.

<p align="center"><img src="/assets/post/image/deepseek-grm-spct/fig3_spct_pipeline.png" width="85%"></p>

위 그림(논문 Figure 3)이 SPCT 전체 파이프라인이다. 위 두 줄이 학습(RFT, RL), 아래 줄이 추론이다. RFT 줄에서 정답과 어긋나는 궤적("Too Easy / Incorrect")이 걸러져 RFT 데이터셋으로만 남는 것, RL 줄에서 규칙(Rules)이 곧바로 reward가 되는 것, 그리고 추론 줄에서 병렬 샘플링된 여러 principle-critique 묶음이 Voting과 Meta RM을 거쳐 최종 점수로 합쳐지는 흐름을 눈으로 확인할 수 있다.

## Inference-time scaling: 병렬 샘플링과 투표

추론 시점에는 같은 입력에 대해 principle-critique 생성을 $$k$$번 독립적으로 샘플링한다. 가장 단순한 집계는 그냥 다 더하는 것이다.

$$S_i^{\text{vote}} = \sum_{j=1}^k S_i^{(j)}$$

- $$k$$: 병렬로 뽑은 (principle, critique) 샘플 개수.
- $$S_i^{(j)}$$: $$j$$번째 샘플이 응답 $$y_i$$에 매긴 점수.

문제는 위 토이 예제의 "원칙 세트 2"처럼 **맥락에 안 맞는 principle이 섞여 있어도 똑같은 한 표를 행사**한다는 점이다. 자기 일관성(self-consistency) 방식의 다수결은 답이 이산적(discrete)이고 오답들이 서로 다른 방향으로 흩어질 때는 잘 통하지만, GRM의 오류는 "일관되게 편향된 principle"에서 나오는 경우가 많아 샘플을 더 늘려도 상쇄되지 않는다. 배심원 12명 중 2~3명이 애초에 사건과 무관한 기준으로 유무죄를 판단하고 있다면, 배심원을 12명에서 32명으로 늘려도 그 왜곡이 저절로 사라지지 않는 것과 같다.

그래서 SPCT는 **meta RM**이라는 별도의 작은 이진 분류기를 함께 학습시킨다. meta RM은 하나의 (principle, critique) 샘플을 보고 "이 샘플이 정답을 맞혔을 가능성이 높은가"를 판정한다. 투표할 때는 $$k$$개 샘플 전부가 아니라, meta RM 점수 상위 $$k_{\text{meta}}$$개만 남겨서 합산한다.

$$S_i^{\text{vote}} = \sum_{j \,\in\, \text{top-}k_{\text{meta}}(M_\theta)} S_i^{(j)}$$

- $$M_\theta$$: meta RM이 각 샘플 $$j$$에 매긴 품질 점수.
- $$k_{\text{meta}}$$: 실험에서는 $$k_{\text{meta}} = k/2$$로 둔다 — 즉 절반은 버린다.

배심원 비유를 이어가면, meta RM은 재판장 역할이다. 배심원 32명의 의견을 다 듣되, 판단 논리가 사건과 무관하거나 형편없는 배심원의 표는 재판장이 먼저 걸러내고, 남은 절반의 표만으로 다수결을 낸다. 뒤에서 볼 실험 결과에서 이 필터링 유무의 차이가 실제로 크게 벌어진다.

# Experiments

## 벤치마크

| 벤치마크         | 초점                                                       | 비고                                                  |
| ---------------- | ---------------------------------------------------------- | ----------------------------------------------------- |
| RewardBench (RB) | chat / chat-hard / safety / reasoning 4개 서브셋           | [9편](/blog/2026/rewardbench-2/)에서 다룬 그 벤치마크 |
| PPE Preference   | Chatbot Arena에서 수집한 16K건의 실제 사람 선호 쌍         | Frick et al., 2024 (arXiv:2410.14872)                 |
| PPE Correctness  | 정답이 검증 가능한 벤치마크(수학·코드 등)에서 만든 선호 쌍 | 같은 논문의 다른 split                                |
| RMB              | 49개 실제 시나리오, pairwise + Best-of-N 평가              | Zhou et al., 2024 (arXiv:2410.09893)                  |

네 벤치마크의 평균을 논문은 Overall로 보고한다. 성격이 서로 다른 벤치마크(사람 선호 vs 검증 가능한 정답 vs 실제 시나리오)를 섞은 것이 핵심이다 — 하나의 벤치마크에서만 잘하는 RM은 여기서 걸러진다.

## 메인 비교

| 모델                                       |    RB    | PPE Pref. | PPE Corr. |   RMB    | Overall  | Avg. Rank |
| ------------------------------------------ | :------: | :-------: | :-------: | :------: | :------: | :-------: |
| ArmoRM-8B-v0.1 ([7편](/blog/2026/armorm/)) |   90.4   |   60.6    |   61.2    |   64.6   |   69.2   |     —     |
| Nemotron-4-340B-Reward                     |   92.0   |   59.3    |   60.8    |   69.9   |   70.5   |     —     |
| GPT-4o                                     |   86.7   |   67.1    |   57.6    |   73.8   |   71.3   |     —     |
| LLM-as-a-Judge (재현)                      |   83.4   |   64.2    |   58.8    |   64.8   |   67.8   |   4.50    |
| CLoud-Gemma-2-27B (재현)                   |   82.0   |   67.1    |   62.4    |   63.4   |   68.7   |   3.50    |
| DeepSeek-PairRM-27B (재현)                 |   87.1   |   65.8    |   64.8    |   58.2   |   69.0   |   2.75    |
| DeepSeek-GRM-27B (RFT만)                   |   84.5   |   64.1    |   59.6    |   67.0   |   68.8   |   4.00    |
| **DeepSeek-GRM-27B (RFT+RL, greedy)**      | **86.0** | **64.7**  | **59.8**  | **69.0** | **69.9** | **2.75**  |

고정 축을 쓰는 [ArmoRM-8B](/blog/2026/armorm/)이 RewardBench 한 곳(90.4)에서는 DeepSeek-GRM-27B(86.0)보다 높지만, PPE Correctness(61.2 vs 59.8 — 근소 우위)를 빼면 다른 벤치마크에서 밀리며 Overall은 69.2로 DeepSeek-GRM-27B의 69.9보다 낮다. 8B 파라미터의 스칼라 RM이 특정 벤치마크에서 강점을 보이는 것과, 27B GRM이 네 종류의 이질적인 벤치마크에서 고르게 상위권을 유지하는 것은 다른 이야기다. **Avg. Rank**(네 벤치마크 각각에서의 순위 평균, 낮을수록 좋음) 열이 이걸 정량화한다 — LLM-as-a-Judge(4.50)나 CLoud(3.50) 대비 DeepSeek-GRM-27B(2.75)가 가장 낮다. 논문은 이를 "SPCT가 scalar·semi-scalar RM 대비 유의미하게 편향이 적다"는 근거로 제시한다. [9편 RewardBench 2](/blog/2026/rewardbench-2/)가 던진 "RM을 어떻게 평가할 것인가"라는 질문에, 이 논문은 "한 벤치마크의 최고점이 아니라 여러 벤치마크의 고른 상위권"으로 답하는 셈이다.

## Inference-time scaling 결과

| 설정                           | Overall  |
| ------------------------------ | :------: |
| Greedy (RFT+RL, $$k=1$$)       |   69.9   |
| Voting@8 (단순 합산)           |   70.6   |
| Voting@32 (단순 합산)          |   71.0   |
| **Voting@32 + Meta RM 필터링** | **72.8** |

Greedy에서 Voting@8로 가면 +0.7, Voting@32까지 가도 +1.1에 그친다. 그런데 여기에 meta RM 필터링을 얹으면 같은 Voting@32에서 +2.9가 더 붙어 총 +4.9(69.9 → 72.8)가 된다. 즉 **샘플을 늘리는 것보다, 늘린 샘플 중 절반을 걸러내는 쪽이 이득이 더 크다** — 앞서 토이 예제로 본 "맥락에 안 맞는 principle은 표를 늘려도 상쇄되지 않는다"는 관찰이 그대로 숫자로 나타난다.

<p align="center"><img src="/assets/post/image/deepseek-grm-spct/fig1_inference_time_scaling.png" width="70%"></p>

위 그림(논문 Figure 1)은 $$k=1$$부터 $$32$$까지 늘려가며 Overall 점수가 어떻게 변하는지 보여준다. meta RM을 쓴 빨간 선(MetaRM@k)은 $$k=2$$ 근방에서 이미 GPT-4o의 그리디 성능(점선, 71.3)을 넘어서고, meta RM 없는 파란 선(Voting@k)은 $$k=32$$까지 가야 겨우 GPT-4o 근처에 도달한다. LLM-as-a-Judge나 CLoud, PairRM 계열은 샘플을 늘려도 거의 평평하다 — Input Flexible이 없거나(PairRM) principle 생성 없이 critique만 쓰는 방식은 애초에 늘릴 다양성이 부족하기 때문이다.

## Training-time scaling과의 대결

논문은 같은 SPCT 학습법을 DeepSeek-V2-Lite(16B MoE), Gemma-2-27B, DeepSeek-V2.5(236B MoE), DeepSeek-V3(671B MoE) 등 서로 다른 크기의 백본에 적용해 "모델을 키우면 얼마나 좋아지는가"를 별도로 측정했다.

<p align="center"><img src="/assets/post/image/deepseek-grm-spct/fig4b_training_time_scaling.png" width="55%"></p>

위 그림(논문 Figure 4b)의 x축이 모델 파라미터 수(16B → 27B → 236B, 로그 스케일), y축이 RewardBench 점수다. RFT/RL 곡선은 236B까지 완만하게 우상향하고, 671B DeepSeek-V3(그리디)는 이 곡선의 연장선 근처에 찍힌다. 논문의 결론은 명확하다.

> "27B DeepSeek-GRM에 32개 샘플로 직접 투표(voting)하는 것이 671B MoE 모델의 성능에 필적할 수 있다."

실제로 앞서 본 표에서 DeepSeek-GRM-27B는 Voting@32 + Meta RM 조합으로 RewardBench 90.4점을 받는다. 모델 파라미터를 27B에서 671B로 24배 넘게 키우는 대신, 추론 시점에 32번 더 계산하는 쪽이 더 싸게 같은 성능 상승을 만든 것이다. 추가로 저자들은 DeepSeek-R1(추론 특화 모델)을 300개 샘플로 축소한 테스트셋에서 그대로 RM으로 써봤는데, 236B RFT 모델보다도 성능이 낮았다 — "추론을 잘하는 모델"과 "reward를 잘 매기는 모델"이 같은 능력이 아니라는 뜻이다.

# Conclusion

핵심을 한 줄로 요약하면 이렇다. **SPCT는 "무엇을 기준으로 채점할지"를 사람이 정하는 대신 모델이 매 입력마다 스스로 생성하게 만들고, 그렇게 생성된 principle-critique 묶음을 병렬로 여러 번 뽑아 meta RM으로 걸러 투표하면, 모델을 키우는 것보다 싸게 같은 성능 향상을 얻는다.**

정리하면,

1. **선택**: 여섯 가지 (paradigm × scoring pattern) 조합 중 유일하게 input flexible과 inference-time scalable을 동시에 만족하는 generative + pointwise 조합을 택했다.
2. **학습**: rejective fine-tuning으로 형식을 잡고, rule-based online RL(GRPO)로 "결과가 맞는" principle-critique 생성 능력을 강화한다. principle 자체에는 직접 보상을 주지 않는다.
3. **추론**: 단순 다수결이 아니라 meta RM이 저품질 샘플을 걸러낸 뒤 투표한다. 27B 모델의 Voting@32+MetaRM(Overall 72.8)이 GPT-4o(71.3)를 넘고, RewardBench 90.4점은 671B 모델을 training-time scaling으로 키운 것과 맞먹는다.
4. **일관성**: 네 종류의 이질적인 벤치마크에서 Avg. Rank 2.75로 가장 고르게 상위권을 유지한다 — 특정 벤치마크에만 강한 편향이 상대적으로 적다.

한계도 분명하다. 32개 샘플을 병렬로 뽑고 meta RM까지 추가로 돌리는 구조는 **추론 비용이 그리디 대비 수십 배**로 뛴다. RLHF 학습 루프 안에서 매 스텝 reward를 계산해야 하는 상황이라면 이 비용은 그대로 학습 시간에 얹힌다. 그리고 저자들 스스로도 인정하듯, principle과 critique가 "그럴듯하지만 실제로는 왜곡된" 방향으로 생성될 가능성(unfaithful principles and critiques)은 여전히 무시할 수 없다 — 모델이 스스로 기준을 만드는 유연성은, 그 기준 자체를 유리하게 왜곡해 hacking하는 새로운 경로를 열어놓는다는 뜻이기도 하다. 이 hacking 가능성은 [26편 One Token to Fool LLM-as-a-Judge](/blog/2026/one-token-to-fool-judge/)에서 정면으로 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물여섯 번째 글이다.

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
25. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
26. **(현재 글)** DeepSeek-GRM / SPCT (2025) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

본 시리즈는 31편으로 구성된다.

# 참고 문헌

- Liu et al., 2025. [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495). arXiv:2504.02495.
- 논문 원문 HTML: [arxiv.org/html/2504.02495](https://arxiv.org/html/2504.02495)
- Frick et al., 2024. [How to Evaluate Reward Models for RLHF](https://arxiv.org/abs/2410.14872) (PPE 벤치마크). arXiv:2410.14872.
- Zhou et al., 2024. [RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/abs/2410.09893). arXiv:2410.09893.
