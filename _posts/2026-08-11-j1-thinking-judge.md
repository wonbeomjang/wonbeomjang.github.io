---
layout: post
title: "J1: RL로 judge를 생각하게 만들다"
date: 2026-08-11 09:39:00 +0900
description: "RLHF Reward 설계 시리즈 #39 — 비검증 프롬프트를 검증 가능한 판정 태스크로 바꿔 judge를 RL로 학습시키는 법"
categories: [paper]
tags: [rlhf, reward-model, genrm, llm-as-a-judge, rlvr, reasoning, paper]
giscus_comments: true
related_posts: true
---

> [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://arxiv.org/abs/2505.10320) (Whitehouse et al., Meta FAIR, arXiv 2025)

# Introduction

이 시리즈는 이번 글에서 크게 세 갈래가 한 지점에 모인다.

- [#32 DeepSeek-R1](/blog/2026/deepseek-r1/)은 정답이 있는 문제(수학, 코드)에서 규칙 기반 verifiable reward로 모델을 RL로 학습시켜 **생각하게** 만들었다. J1은 같은 RLVR 레시피를 **judge 자신**에게 적용한다 — 판정을 내리는 모델도 생각을 하면 더 정확해진다는 가설이다.
- [#40 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)는 애초에 정답이 없는 도메인(의료 상담, 개방형 질의)을 **사람이 쓴 채점 기준표**로 검증 가능하게 바꾼다. J1은 같은 문제 — "비검증 프롬프트를 어떻게 RL로 학습할 것인가" — 를 전혀 다른 방식으로 푼다. 기준표를 쓰는 대신 **판정 태스크의 구조 자체**를 조작해 정답을 자동으로 만들어낸다.
- [#42 One Token to Fool LLM-as-a-Judge](/blog/2026/one-token-to-fool-judge/)는 GenRM judge가 얼마나 쉽게 흔들리는지, 특히 **어느 응답이 먼저 제시되느냐(position)**에 얼마나 취약한지를 보여준다. J1은 이 취약성 중 하나인 position bias 완화를 설계 단계에서부터 명시적 목표로 못박는다.

지금까지 이 시리즈 7부([#33](/blog/2026/prometheus-2/)~[#37](/blog/2026/deepseek-grm-spct/))는 judge를 **생성 모델(GenRM)**로 바꾸면 스칼라 reward model보다 유연해진다는 것을 보였다. [#37 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/)은 judge가 스스로 평가 원칙(principle)을 만들고 그에 따라 비평(critique)을 쓰도록 학습시켰고, [#38 ReasonGRM](/blog/2026/reasongrm/)은 reasoning 능력이 뛰어난 모델의 풀이 흔적을 골라내 judge에 주입했다. 두 글 모두 "judge가 더 잘 판정하려면 더 잘 생각해야 한다"는 전제는 같지만, **그 생각하는 능력을 어떻게 얻게 할 것인가**에서 갈렸다 — 원칙 생성이라는 행동을 SFT로 심거나(GRM), 좋은 추론 경로를 골라내거나(ReasonGRM).

J1은 세 번째 길을 연다. **judge의 CoT 자체를 RL로 직접 최적화**하는 것이다. 그런데 RL은 reward가 있어야 돌아가고, "이 판정이 맞았는가"는 애초에 정답이 없는 프롬프트(에세이 첨삭, 여행 코스 추천 같은)에서는 정의하기 어렵다. J1의 전부는 이 문장 하나로 요약된다 — **검증 가능한 프롬프트와 검증 불가능한 프롬프트를 모두 하나의 형식으로 바꿔, 판정 자체에 verifiable reward를 붙인다.** 그 결과 8B·32B·70B 세 규모의 judge가 만들어졌고, 그중 J1-Qwen-32B는 RewardBench에서 93.6점을 받아 o1-mini(87.1), o3(86.4), 그리고 20배 넘게 큰 DeepSeek-R1-671B(90.6)를 모두 앞질렀다 — 그것도 **합성 데이터 22K건만으로** 학습해서다.

# Background

## RLVR 레시피를 다시 꺼내며

[#32 DeepSeek-R1](/blog/2026/deepseek-r1/)에서 정리했듯 RLVR(Reinforcement Learning from Verifiable Rewards)의 골자는 간단하다. 정답을 채점 함수로 확인할 수 있는 문제(수학 답, 코드 테스트 통과 여부)라면, 사람이 개입하지 않아도 규칙만으로 reward를 줄 수 있다. 모델은 이 reward를 높이려고 스스로 생각의 길이를 늘리고, 중간에 스스로를 되짚는 행동(self-verification)을 학습한다.

문제는 judge를 학습시키려는 순간 발생한다. judge가 맞혀야 할 "정답"은 "어느 응답이 더 나은가"인데, 이건 수학 답과 달리 채점 함수가 없다. 특히 WildChat류의 일반 대화 프롬프트("이직 이력서 어떻게 써?", "제주도 여행 코스 짜줘")에는 애초에 절대적으로 옳은 응답이 없다.

## pairwise와 pointwise, 그리고 judge 포맷

judge를 만드는 방식은 크게 두 갈래다. **pairwise**는 두 응답을 동시에 보여주고 우열을 가리게 하고([#34 Generative Verifiers](/blog/2026/generative-verifiers/)), **pointwise**는 응답 하나에 절대 점수를 매기게 한다([#35 Generative Reward Models](/blog/2026/generative-reward-models/)). pairwise는 상대 비교라 정확하지만 두 응답의 **제시 순서**에 따라 판정이 흔들리는 position bias에 취약하다 — 같은 두 응답을 순서만 바꿔 다시 보여주면 judge가 다른 결론을 낸다. pointwise는 순서 문제는 없지만 절대 점수의 캘리브레이션이 어렵다.

J1은 이 둘을 모두 학습하고 결국 하나의 모델로 합친다(MultiTask-J1). 그 전에 먼저, 이 두 포맷 모두가 요구하는 재료 — "정답이 알려진 (프롬프트, 우수 응답, 열등 응답)" 삼중항 — 을 어떻게 조달하는지가 이 논문의 핵심이다.

# Method

## 핵심 트릭: 판정을 검증 가능한 태스크로 바꾸기

<p align="center"><img src="/assets/post/image/j1-thinking-judge/fig2_method.png" width="95%"></p>

그림 위쪽 절반이 원래도 검증 가능한 프롬프트(MATH 같은), 아래쪽 절반이 검증 불가능한 프롬프트(WildChat 같은)를 다루는 경로다. 두 경로 모두 결국 같은 상자로 합류한다 — "User Prompt $$x$$, Chosen Response $$a$$, Rejected Response $$b$$, Winner $$a$$". judge 입장에서는 이 삼중항이 어느 경로에서 왔는지 알 수 없다. 그냥 정답이 달린 (프롬프트, 응답 쌍)이 하나 도착할 뿐이다.

### 검증 가능한 프롬프트(MATH): 정답과 대조해서 고른다

1. 프롬프트 $$x$$를 LLM에 여러 번 샘플링해 응답 $$R_1, \dots, R_N$$을 만든다.
2. 각 응답을 정답과 대조해 채점한다(Verification) — 맞으면 통과, 틀리면 탈락.
3. 맞은 응답 중 하나를 $$a$$(chosen), 틀린 응답 중 하나를 $$b$$(rejected)로 뽑는다(Pair Selection).

이 경로는 새로울 게 없다 — [#30 Let's Verify Step by Step](/blog/2026/lets-verify-step-by-step/), [#31 Math-Shepherd](/blog/2026/math-shepherd/)에서부터 봐온 "정답과 대조" 방식 그대로다.

### 검증 불가능한 프롬프트(WildChat): 질문을 일부러 흐려서 정답을 만든다

여기가 이 논문의 전부다. 정답이 없는 프롬프트에 어떻게 "이 응답이 저 응답보다 낫다"는 자동으로 확인 가능한 정답을 붙일 수 있을까? J1의 답은 이렇다.

1. 원래 프롬프트 $$x$$를 LLM에 그대로 넣어 응답 $$a$$를 받는다. 이게 chosen이다.
2. 같은 LLM에게 $$x$$의 "노이즈가 낀 버전" $$x'$$를 만들게 한다 — 원래 질문의 핵심 조건을 흐리거나 엉뚱한 방향으로 비튼 변형이다.
3. $$x'$$를 LLM에 넣어 응답 $$b$$를 받는다. $$b$$는 $$x'$$에는 성실하게 답한 글이지만, 원래 질문 $$x$$ 입장에서 보면 핵심을 비껴간 답이다. 이게 rejected다.
4. 이제 $$(x, a, b)$$ 삼중항이 완성됐고, 정답(winner)은 항상 $$a$$다 — $$b$$는 애초에 다른 질문에 답한 글이기 때문이다.

일상 비유로 옮기면 이렇다. 편집장이 두 기자에게 같은 취재를 시킨다고 하자. 한 명에게는 원래 취재 브리핑을 그대로 주고, 다른 한 명에게는 브리핑의 핵심 조건 하나를 몰래 바꿔치기해서 준다. 두 기자 모두 성실하게, 문장력도 비슷하게 기사를 써 온다. 하지만 원래 브리핑을 기준으로 채점하면 바꿔치기된 브리핑을 받은 기자의 기사가 구조적으로 열등할 수밖에 없다 — 애초에 다른 질문에 답했으니까. 편집장은 두 기사를 힘들게 다시 읽고 우열을 가릴 필요가 없다. 어느 기자가 원래 브리핑을 받았는지는 이미 알고 있기 때문이다.

### 토이 예제: 프롬프트 하나가 판정 태스크로 바뀌는 과정

원래 프롬프트 $$x$$: "이직 준비 중인데, 이력서 프로젝트 경험 항목을 어떻게 써야 좋을지 알려줘."

| 단계                    | 내용                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1. chosen 생성          | $$x$$를 LLM에 그대로 입력 → 응답 $$a$$: 성과를 정량 지표로 적고 STAR(문제-행동-결과) 구조로 쓰라는 구체적 조언                    |
| 2. 노이즈 프롬프트 생성 | LLM이 $$x$$를 변형 → $$x'$$: "이력서에 학력 항목을 어떻게 배치해야 좋을지 알려줘" (프로젝트 경험이라는 핵심 조건이 학력으로 바뀜) |
| 3. rejected 생성        | $$x'$$를 LLM에 입력 → 응답 $$b$$: 학력 배치 요령에 대한, 그 자체로는 훌륭한 조언                                                  |
| 4. 삼중항 완성          | $$(x, a, b)$$, winner $$= a$$ — $$b$$는 유창하지만 원래 질문 $$x$$에는 답하지 않았다                                              |

judge가 이 삼중항을 pairwise로 받으면 $$a$$와 $$b$$ 둘 다 표면적으로는 매끄러운 조언문이다. 문장 길이나 형식만 보고 판정하면 반반 확률로 틀린다. 원래 질문 $$x$$가 무엇을 물었는지를 진짜로 이해해야만 $$b$$가 핵심을 비껴갔다는 걸 잡아낼 수 있다. 그래서 이 태스크는 "정답을 아는" judge를 만들면서도, judge가 표면적 휴리스틱이 아니라 실제 내용을 판정하도록 강제한다.

## reward 설계: 정답을 맞히는 것과 흔들리지 않는 것

이제 이 삼중항으로 GRPO를 돌린다(GRPO 자체는 [#21 GRPO/DeepSeekMath](/blog/2026/grpo-deepseekmath/) 참고). judge 모델 $$\pi_\theta$$는 $$(x, a, b)$$를 받아 사고 과정 $$t$$와 판정 $$y$$를 함께 생성한다.

$$J_{(a,b)} = \pi_\theta(t, y \mid x, a, b), \quad J_{(b,a)} = \pi_\theta(t, y \mid x, b, a)$$

같은 삼중항을 순서만 바꿔 두 번 판정하게 한다는 점이 핵심이다. 여기에 세 가지 reward가 걸린다.

**verdict correctness reward** — 각 순서에서 정답을 맞혔는가:

$$R_{\text{correct}}(J_{(a,b)}) = \mathbb{1}[\text{Verdict}(J_{(a,b)}) = a], \quad R_{\text{correct}}(J_{(b,a)}) = \mathbb{1}[\text{Verdict}(J_{(b,a)}) = a]$$

$$\text{Verdict}(\cdot)$$은 judge가 최종적으로 고른 응답, $$\mathbb{1}[\cdot]$$은 괄호 안 조건이 참이면 1, 아니면 0인 지시함수다. 두 순서 모두 "진짜 정답 $$a$$"를 골라야 reward를 받는다.

**verdict consistency reward** — 순서를 바꿔도 같은 결론을 냈는가:

$$R_{\text{consist}} = \mathbb{1}[\text{Verdict}(J_{(a,b)}) = \text{Verdict}(J_{(b,a)}) = a]$$

이 항이 바로 position bias 완화가 설계에서 저절로 따라 나오는 지점이다. correctness reward만 있으면 judge는 두 순서 각각에서 독립적으로 정답만 맞히면 된다 — 예컨대 "먼저 제시된 응답을 선호"하는 얕은 휴리스틱을 쓰더라도 순서가 무작위로 섞인 배치에서는 평균적으로 어느 정도 정답률을 우연히 넘길 수 있다. 하지만 같은 삼중항이 반드시 $$(a,b)$$와 $$(b,a)$$ 두 순서로 모두 배치에 들어가고(그림의 "Position-agnostic Pairwise Batch"), consistency reward는 두 순서 모두에서 같은 응답을 답해야만 1을 준다. 비유하면 블라인드 시음 실험과 같다. 같은 두 음료를 컵 위치만 바꿔 두 번 내놓고, 위치와 무관하게 매번 같은 음료를 고를 때만 그 사람의 미각을 신뢰한다 — 왼쪽 컵을 습관적으로 고르는 사람은 이 테스트를 통과할 수 없다. "어느 자리에 놓였는가"가 아니라 "내용이 무엇인가"를 봐야만 두 reward를 동시에 최대화할 수 있도록 설계된 것이다.

**score 기반 reward(PaS 포뮬레이션)** — 판정 대신 점수를 매기는 경우:

$$R_{\text{score}} = \mathbb{1}[s_a^{(a,b)} > s_b^{(a,b)}]$$

이때 $$s_a^{(a,b)}$$는 순서 $$(a,b)$$로 제시했을 때 judge가 응답 $$a$$에 매긴 점수다. pointwise 포맷은 응답을 하나씩 독립적으로 채점하므로 순서 문제가 아예 없다. reward는 더 단순하다.

$$R_{\text{point}} = \mathbb{1}[s_a > s_b]$$

## 다섯 가지 학습 포뮬레이션

| 이름                          | 입력 → 출력                   | reward                            | 특징                                 |
| ----------------------------- | ----------------------------- | --------------------------------- | ------------------------------------ |
| Pairwise-Verdict(PaV)         | $$(x,a,b) \to (t,y)$$         | correctness + consistency         | 판정만 출력, 가장 단순               |
| Pairwise-Scores(PaS)          | $$(x,a,b) \to (t,s_a,s_b)$$   | score 기반                        | 두 응답에 실수 점수를 매김           |
| Pairwise-Scores&Verdict(PaVS) | $$(x,a,b) \to (t,s_a,s_b,y)$$ | correctness + consistency + score | 점수와 판정을 동시에 출력            |
| Pointwise(PoS)                | $$(x,a) \to (t,s)$$           | score 기반                        | pairwise 데이터를 원격 지도로 재사용 |
| MultiTask(MT)                 | pairwise + pointwise 조합     | 위 전부                           | 하나의 모델이 두 포맷 모두를 학습    |

Pointwise(PoS)는 별도의 pointwise 라벨이 없다. 같은 $$(x,a,b,\text{winner}=a)$$ 데이터를 "$$a$$는 높게, $$b$$는 낮게 채점되어야 한다"는 제약으로 재사용한다 — 이것이 논문이 말하는 "distant supervision"이다.

## RaR과 무엇이 다른가

| 항목                          | Rubrics as Rewards(#40, Gunjal et al. 2025)        | J1                                                |
| ----------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| 무엇을 검증 가능하게 만드는가 | 채점 기준(rubric) 자체                             | 판정 태스크(어느 응답이 나은가) 자체              |
| 정답의 출처                   | 사람(도메인 전문가)이 작성한 rubric 항목           | LLM이 스스로 만든 노이즈 프롬프트로부터 자동 생성 |
| RL로 학습되는 대상            | 응답을 생성하는 policy 모델                        | 응답을 판정하는 judge 모델                        |
| 사람 개입                     | rubric 설계 단계에 필요                            | 없음 — 파이프라인 전체가 LLM 생성                 |
| 검증 대상 도메인              | HealthBench(의료), GPQA-Diamond(과학) 등 특정 분야 | WildChat(범용 대화) + MATH                        |

두 논문 모두 "비검증 도메인을 어떻게 RLVR화할 것인가"라는 같은 문제에서 출발하지만, RaR은 사람이 여전히 정답의 기준을 정의하고 J1은 정답의 존재 자체를 자동으로 조작해낸다. 그만큼 J1은 사람 개입이 없어 확장성이 좋지만, "노이즈 프롬프트가 실제로 더 나쁜 질문인가"를 보장하는 별도 장치가 없다는 약점도 함께 짊어진다 — 이 지점은 [#40 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)에서 다시 다뤄질 부채다.

## judge의 자기 성찰 행동들

<p align="center"><img src="/assets/post/image/j1-thinking-judge/fig1_example.png" width="95%"></p>

RL로 학습된 judge의 사고 과정을 들여다보면 반복적으로 등장하는 네 가지 행동이 있다(위 그림, 논문 Figure 1).

1. **동적 평가 기준 생성(Evaluation Criteria)**: "정확성, 설명의 명료성, 사용자 질문과의 부합도를 기준으로 판단하겠다"처럼 프롬프트마다 다른 채점 기준을 스스로 세운다.
2. **참조 답안 생성(Reference Answer)**: 두 응답을 비교하기 전에 "이 문제라면 이렇게 풀린다"는 자기만의 정답을 먼저 만든다.
3. **자기 판단의 반복적 교정(Re-evaluation)**: 한 번 세운 결론을 다시 검산한다. "이 단계를 다시 확인해보면..." 같은 문장이 등장하며 스스로 되짚는다.
4. **저품질 응답에 대한 피드백 생성(Feedback)**: pointwise 판정에서 특히 두드러지는데, 낮은 점수를 준 응답에 "정확한 계산법은 사실 이렇다"처럼 구체적으로 무엇이 틀렸는지를 짚는다.

이 네 행동은 [#37 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)의 자기 원칙 생성(self-principled critique)과 닮은 점이 많다 — 둘 다 "판정 기준을 모델이 스스로 만든다"는 발상을 공유한다. 다른 점은 그 행동이 어디서 나왔는가다.

| 항목                | DeepSeek-GRM / SPCT(#37)                                                             | J1                                                                      |
| ------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| 기준 생성 방식      | rejection sampling으로 원칙 생성 능력을 SFT로 심고 RL(GRPO)로 다듬음                 | verdict correctness/consistency reward만으로 RL 도중 자연 발생          |
| 명시적 목표 함수    | 원칙(principle) 품질에 대한 별도 보상 없이 principle+critique를 함께 생성하도록 학습 | reference answer·재평가는 reward에 명시된 목표가 아니라 부산물로 관찰됨 |
| inference-time 확장 | 여러 원칙·critique 세트를 샘플링해 meta RM으로 투표                                  | 여러 rollout을 샘플링해 다수결(self-consistency)로 투표                 |

J1 쪽이 흥미로운 지점은 참조 답안 생성이나 재평가 같은 행동을 reward로 직접 요구하지 않았는데도 RL 학습 과정에서 스스로 나타났다는 것이다 — verdict correctness를 극대화하는 가장 쉬운 경로가 결국 "정답부터 만들고 대조하기"였다는 뜻이다.

# Experiments

## 벤치마크: 32B가 671B를 이기는 순간

다섯 벤치마크는 조금씩 다른 능력을 잰다. RewardBench는 범용 선호 판정, RM-Bench는 스타일에 흔들리지 않는 판정, JudgeBench는 미묘한 사실·계산 오류 탐지, PPE는 실제 사용자 선호와의 상관, FollowBenchEval은 지시사항 이행 여부 판정을 각각 겨냥한다.

| 모델                  | 크기 | Overall  | PPE  | RewardBench | RM-Bench | JudgeBench† | FollowBenchEval† |
| --------------------- | ---- | -------- | ---- | ----------- | -------- | ----------- | ---------------- |
| J1-Llama-8B           | 8B   | 61.9     | 59.8 | 85.7        | 73.4     | 42.0        | 48.3             |
| J1-Llama-70B          | 70B  | 75.0     | 69.6 | 93.3        | 82.7     | 60.0        | 69.3             |
| J1-Qwen-32B-MultiTask | 32B  | **80.8** | 71.8 | **93.6**    | **90.3** | 71.4        | **77.1**         |
| o1-mini               | –    | 72.7     | 68.5 | 87.1        | 80.8     | 64.2        | 62.9             |
| o3                    | –    | 77.4     | 72.1 | 86.4        | 86.1     | **75.7**    | 66.8             |
| DeepSeek-R1           | 671B | 78.4     | 72.3 | 90.6        | 88.6     | 68.9        | 71.7             |

†: 논문은 JudgeBench와 FollowBenchEval에서 position-consistent accuracy를 기본 지표로 쓴다.

- J1-Qwen-32B-MultiTask는 다섯 벤치마크 평균(Overall)에서 80.8점으로 o1-mini(72.7), o3(77.4), 그리고 20배 넘게 큰 DeepSeek-R1-671B(78.4)를 모두 앞선다.
- RewardBench(93.6), RM-Bench(90.3), FollowBenchEval(77.1)에서는 표에 있는 모든 모델 중 1위다.
- JudgeBench(71.4)에서만 o3(75.7)에 못 미친다 — 논문은 JudgeBench가 미묘한 사실 오류·계산 실수를 잡아내야 하는, 특히 어려운 벤치마크라고 밝힌다.
- J1-Llama-8B조차 base 모델(RL 이전, zero-shot judge) 대비 RewardBench에서 +16.2점, RM-Bench에서 +19.4점을 얻는다 — RL 자체의 효과가 크다는 뜻이다.

training data는 22K개(WildChat 17K + MATH 5K) 합성 선호 쌍이 전부다 — 사람이 새로 라벨링한 선호 데이터는 하나도 없다. 이 규모는 비교 대상인 DeepSeek-GRM-27B가 쓴 237K건보다 10배 이상 적다.

## position bias는 사라졌는가

| 포맷                             | (a,b) 순서 정확도 | (b,a) 순서 정확도 | 두 순서 모두 정답(consistent) | 순서 바뀌면 뒤집힘 |
| -------------------------------- | ----------------- | ----------------- | ----------------------------- | ------------------ |
| J1-Qwen-32B-MultiTask, pairwise  | 76.8              | 76.2              | 67.0                          | 17.0%              |
| J1-Qwen-32B-MultiTask, pointwise | –                 | –                 | 70.6                          | 10.5%              |

- 한쪽 순서만 놓고 보면 정확도가 76%대로 높아 보이지만, 두 순서 모두 정답을 내야 하는 consistent accuracy는 67.0%로 뚝 떨어진다. 순서를 바꿨을 때 판정이 뒤집히는 경우가 17.0%나 된다는 뜻이다 — consistency reward를 넣어도 position bias가 완전히 사라지진 않는다.
- pointwise가 pairwise보다 일관적이다(70.6 > 67.0, 뒤집힘 비율도 10.5% < 17.0%) — 애초에 응답을 하나씩 독립적으로 채점하니 "순서"라는 변수 자체가 없어서다. 대신 pairwise는 두 응답을 나란히 놓고 비교하는 만큼 단일 순서 정확도가 더 높아 미세한 우열을 더 잘 잡아낸다. 그래서 논문은 둘을 합친 MultiTask-J1을 최종 모델로 쓴다.
- test-time에 같은 판정을 $$N$$번 반복하고 다수결(self-consistency)을 취하면 이 격차가 줄어든다. $$N=32$$까지 늘리면 pointwise-J1-70B의 평균 점수 다수결 정확도가 greedy 65.0%에서 74.8%까지 오른다.

## ReasonGRM과 어떻게 다른 길을 갔나

| 항목      | ReasonGRM(#38, Chen et al. 2025)                                       | J1                                                                 |
| --------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 문제의식  | judge의 reasoning 흔적 품질이 들쭉날쭉함                               | 비검증 프롬프트엔 reward를 줄 수 없음                              |
| 핵심 장치 | $$R^*$$ 지표로 좋은 추론 경로를 사후에 골라냄(데이터 선별)             | 노이즈 프롬프트로 정답이 있는 학습 데이터 자체를 생성(데이터 생성) |
| 학습 단계 | 1) Zero-RL로 후보 경로 생성 2) $$R^*$$로 선별 3) 어려운 예제에 RL 추가 | 검증 가능/불가능 프롬프트를 통합 형식으로 만든 뒤 단일 RL(GRPO)    |
| RL의 역할 | 선별된 데이터 위에서 판별력을 다듬는 보조 수단                         | reward 설계의 중심, 처음부터 끝까지 judge를 직접 최적화            |

두 논문 모두 "reasoning이 judge 품질을 좌우한다"는 전제를 공유하지만, ReasonGRM은 이미 있는 추론 경로 중 좋은 것을 고르는 데이터 큐레이션 문제로 풀었고 J1은 애초에 정답이 있는 학습 데이터를 만들어내는 문제로 풀었다. 두 접근은 배타적이지 않다 — $$R^*$$식 선별을 J1의 GRPO rollout에 얹는 조합도 자연스러운 다음 단계로 보인다.

# Conclusion

핵심을 한 줄로: **J1은 "판정 자체를 검증 가능한 형식으로 바꾸면 judge도 RLVR로 직접 학습할 수 있다"는 것을 22K개의 합성 데이터만으로 증명했다.** 비검증 프롬프트는 노이즈 낀 변형 질문으로 정답 있는 선호 쌍을 만들고, 검증 프롬프트는 정답 대조로 선호 쌍을 만들어 같은 파이프라인에 태운다. 그 결과 나온 judge는 순서를 바꿔도 (완전하지는 않지만) 더 일관되게 판정하고, 참조 답안 생성이나 재평가 같은 행동을 reward로 명시하지 않아도 스스로 학습한다.

한계도 분명하다.

1. **노이즈 프롬프트의 품질이 보장되지 않는다.** $$x'$$가 정말로 $$x$$보다 나쁜 질문인지 검증하는 장치가 없다 — 이 약점은 사람이 직접 기준을 쓰는 [#40 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)와 대비되는 지점이자, 뒤 글에서 다시 짚을 부채다.
2. **position bias가 완전히 없어지지 않았다.** consistent accuracy(67.0~70.6%)는 여전히 단일 순서 정확도(76%대)보다 한참 낮다. [#42 One Token to Fool LLM-as-a-Judge](/blog/2026/one-token-to-fool-judge/)가 다루는 judge의 구조적 취약성은 J1 같은 RL 학습으로도 완전히 닫히지 않는 문제라는 뜻이다.
3. **format reward는 도움이 안 됐다.** `<think>` 태그를 강제하는 reward를 추가로 줘봐도 유의미한 성능 차이가 없었다 — 사고의 "형식"이 아니라 reward 구조 자체가 사고의 질을 만든다는 시사점이다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 서른아홉 번째 글이다.

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
  <li><a href="/blog/2026/bond/">BOND (2024)</a> — Best-of-N을 추론 비용 없이</li>
  <li><a href="/blog/2026/warp/">WARP (2024)</a> — 정책을 weight space에서 병합</li>
</ol>

**6부. Process & Verifiable Reward**

<ol start="30">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="33">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="38">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><strong>(현재 글)</strong> J1 (2025) — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="43">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열한 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 44편으로 구성된다.

# 참고 문헌

- Whitehouse et al., 2025. [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://arxiv.org/abs/2505.10320).
- Gunjal et al., 2025. [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746).
- Chen et al., 2025. [ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models](https://arxiv.org/abs/2506.16712).
