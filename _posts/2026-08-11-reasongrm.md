---
layout: post
title: "ReasonGRM: judge에게 추론 능력을 이식하다"
date: 2026-08-11 09:24:20 +0900
description: "RLHF Reward 설계 시리즈 #33 — Large Reasoning Model로 generative reward model을 강화하는 법"
categories: [paper]
tags: [rlhf, reward-model, genrm, reasoning, llm-as-a-judge, paper]
giscus_comments: true
related_posts: true
---

> [ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models](https://arxiv.org/abs/2506.16712) (Chen et al., Qihoo360, arXiv 2025)

# Introduction

이 시리즈 [#29 Generative Verifiers](/blog/2026/generative-verifiers/)는 judge에게 CoT를 쓰게 하면 정답 토큰 확률이 더 잘 몰린다는 것을 보였다. reward를 next-token prediction으로 재정의하고, 판정 앞에 추론을 붙이기만 해도 성능이 오른다는 게 그 글의 결론이었다. 이후 [#30 GenRM](/blog/2026/generative-reward-models/), [#31 Self-Taught Evaluators](/blog/2026/self-taught-evaluators/), [#32 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)까지, 7부는 일관되게 "reward를 어떻게 생성 형태로 만들 것인가"를 다뤘다.

그런데 이 흐름 전체가 암묵적으로 깔고 가는 전제가 하나 있다. **추론을 시키면 좋아진다는 것.** 정말 그런가? 그냥 "생각해봐"라고 프롬프트에 한 줄 붙인다고 judge가 갑자기 논리적으로 사고하게 되는 건 아니다. [#27 DeepSeek-R1](/blog/2026/deepseek-r1/)이 수학·코드 도메인에서 보여준 것처럼, 추론 능력 자체는 명시적으로 길러야 하는 별개의 역량이다. 이번 글이 여는 8부 "생각하는 Judge, 그리고 그 신뢰"는 바로 이 지점을 파고든다. **judge에게 어떻게 추론 능력을 길러 넣고, 그렇게 만든 판정을 얼마나 믿을 수 있는가.**

ReasonGRM은 그 첫 번째 시도다. Qihoo360이 2025년 6월 공개한 이 논문은 세 가지 질문에 답한다.

1. 기존 GenRM의 추론 경로는 왜 부실한가 — 무엇이 "나쁜 추론"인가.
2. 여러 후보 추론 경로 중 무엇을 학습 데이터로 채택할지, 그 선별 기준을 어떻게 수식화하는가.
3. 그렇게 만든 judge는 실제로 얼마나 좋아지는가, 그리고 그 대가는 무엇인가.

결론을 먼저 요약하면 이렇다. ReasonGRM은 Zero-RL → $$R^\star$$ 기반 데이터 선별 → hard-case 강화학습의 3단계 파이프라인으로, RewardBench·RM-Bench·RMB 세 벤치마크 평균 83.3점을 달성해 기존 최고 GenRM보다 1.8%p, GPT-4o보다 5.6%p 높은 점수를 받았다. 핵심은 "추론을 시켰다"가 아니라 "어떤 추론을 학습시켰는가"에 있고, 그 선별 기준이 논문이 제안하는 $$R^\star$$ 메트릭이다.

# Background

## GenRM이 가진 두 가지 실패 모드

GenRM은 판정을 스칼라 하나로 압축하는 대신 자연어 rationale과 함께 내놓는다. 그래서 표현력은 좋지만, 그 rationale의 품질이 곧 판정의 신뢰도로 직결된다. ReasonGRM은 기존 GenRM들이 "추론 단계를 넣으면 성능이 오른다"는 것까지는 확인했지만, **정작 어떤 추론이 바람직한지는 정의하지 않은 채로** 방치했다고 지적한다. 그 결과 수집되거나 생성된 rationale은 노이즈가 많고, 논리적으로 일관되지 않으며, 과제 목표와 어긋나는 경우가 잦다.

구체적으로는 두 가지 실패 패턴으로 나뉜다.

- **사후 정당화(post-hoc rationalization)**: 결론을 먼저 정해놓고 그럴듯한 문장을 뒤에 붙이는 경우. 추론처럼 보이지만 실제로는 결론과 인과관계가 없다.
- **산만한 탐색(overly speculative reasoning)**: 결론에 도달하긴 하지만 "그런데, 잠깐만, 사실은..." 하며 여러 갈래를 헤매다가 우연히 맞는 길을 찾는 경우. 중간 단계가 결론과 무관하거나 서로 모순된다.

이 문제는 낯설지 않다. [#26 Math-Shepherd](/blog/2026/math-shepherd/)에서 이미 같은 계열의 문제를 다뤘다 — "정답으로 이어졌다 ≠ 그 단계가 논리적으로 옳다." Math-Shepherd는 수학 풀이의 중간 스텝에 대해 이 간극을 지적했다면, ReasonGRM은 판정(judgment)의 추론 경로에 대해 같은 간극을 지적한다. **"결론이 맞았다 ≠ 그 추론이 신뢰할 만하다."**

## 좋은 추론의 두 조건: Validity와 Self-Consistency

ReasonGRM은 이 간극을 메우기 위해 좋은 추론 경로가 만족해야 할 두 속성을 정의한다.

- **Validity(타당성)**: 길이나 형식과 무관하게, 결과적으로 정답에 도달해야 한다.
- **Self-Consistency(자기 일관성)**: 추론 내부가 논리적으로 이어져야 하고, 사변적이거나 갈팡질팡하는 우회로가 없어야 한다.

이 두 조건을 모두 만족해야 "학습시킬 가치가 있는 추론 경로"가 된다. 논문은 이를 설명하기 위해 아래와 같은 모티브 예시를 제시한다.

<p align="center"><img src="/assets/post/image/reasongrm/fig1_motivating_example.png" width="60%"></p>

지침(instruction)은 "항상 답을 내놓기 전에 질문부터 하라"이다. 두 응답 중 A는 "질문을 먼저 드리고 답하겠다"고 **약속만** 하고, B는 "질문을 더 잘 이해하기 위해 지금 바로" 되묻는다. 정답은 B다 — 지침을 실제로 실행한 쪽이니까. 이 하나의 사례에 대해 세 종류의 추론 경로가 나올 수 있다.

| 추론 유형           | 결론            | Validity | Self-Consistency | 특징                                                                 |
| ------------------- | --------------- | -------- | ---------------- | -------------------------------------------------------------------- |
| Incorrect Reasoning | A가 낫다 (오답) | 실패     | -                | 지침 자체를 놓치고 "하겠다"는 말을 지침 이행으로 오인                |
| Weaker Reasoning    | B가 낫다 (정답) | 성공     | 약함             | "그런데", "잠깐, 사실 사용자 질문은..." 식으로 몇 차례 뒤집으며 도달 |
| Stronger Reasoning  | B가 낫다 (정답) | 성공     | 강함             | A는 약속만, B는 즉시 실행이라는 차이를 곧장 짚고 결론                |

여기서 중요한 건 Weaker Reasoning과 Stronger Reasoning이 **같은 정답에 도달한다**는 점이다. 정답 여부만으로는 둘을 가를 수 없다. 그런데도 학습 데이터로는 Stronger Reasoning이 훨씬 낫다 — 결론까지 가는 경로가 짧고 확신에 차 있어서, 이 경로를 모방하도록 학습하면 모델이 불필요한 헤매기를 배우지 않는다. 이 구분을 수식으로 만든 것이 다음 절의 $$R^\star$$ 메트릭이다.

# Method

<p align="center"><img src="/assets/post/image/reasongrm/fig2_pipeline.png" width="85%"></p>

ReasonGRM은 DeepSeek-R1의 콜드스타트 아이디어에서 영감을 받아 3단계로 구성된다. Stage 1이 "정답 판별력"을, Stage 2가 "그 판별력이 만든 후보 중 좋은 추론만 골라 SFT"를, Stage 3이 "어려운 사례에 대한 최종 미세조정"을 담당한다.

## Stage 1 — Zero-RL: 결과만 아는 채점자 기르기

비유하자면 이렇다. 풀이 과정은 전혀 보여주지 않고 "이 답이 맞았는지 틀렸는지"만 채점해주는 선생님을 먼저 길러내는 단계다. 이 단계의 초기 데이터셋에는 질문, 후보 응답, 정답만 있고 추론 과정은 전혀 없다.

GRPO로 base LRM $$M_{\pi_0}$$를 outcome-only reward로 학습한다. 다만 아무 샘플이나 다 쓰지 않는다. 한 질문당 $$K$$개의 응답을 생성했을 때, 다음 조건을 만족할 때만 파라미터를 업데이트한다.

$$0 < \sum_{i=1}^{K} \mathbb{I}(\text{is_equivalent}(o_i, a_Q)) < K$$

기호를 하나씩 풀면,

- $$\mathbb{I}(\cdot)$$: 괄호 안 조건이 참이면 1, 거짓이면 0인 지시함수.
- $$o_i$$: 모델이 생성한 $$i$$번째 응답.
- $$a_Q$$: 그 질문의 정답.
- $$K$$: 한 질문에 대해 생성한 응답의 총 개수.

$$K$$개 전부가 정답이면(=너무 쉬움, 학습 신호 약함) 스킵하고, $$K$$개 전부가 오답이면(=너무 어려움, 강제 학습이 오히려 불안정을 유발) 역시 스킵한다. 즉 **일부만 맞고 일부는 틀린, 모델이 아직 갈팡질팡하는 샘플에만** 학습을 집중시킨다. 이렇게 얻은 모델이 LRM-Zero다. 명시적인 추론 텍스트를 한 번도 본 적 없지만, "무엇이 맞는 판정인가"에 대한 감각은 갖춘 상태다.

## Stage 2 — $$R^\star$$: 확신에 차고 정답에 이른 추론만 골라내기

이제 LRM-Zero에게 실제로 추론을 시켜본다. 문제는 여러 추론 경로를 뽑았을 때 그중 다수가 정답에 도달하더라도, 서로 논리적 명료함·간결함·확신의 정도가 천차만별이라는 것이다. 이걸 가려내는 기준이 필요하다.

직관은 "자신 있게 또박또박 말하는 사람과 말끝을 흐리며 우물쭈물하는 사람"의 차이와 같다. 후자는 결국 같은 답에 도달하더라도 그 과정에서 여러 번 스스로를 의심한다. 언어모델에서 이 "의심"은 다음 토큰을 생성할 때의 확률로 드러난다 — 망설이는 토큰일수록 확률이 낮다.

질문 $$Q^{(j)}$$마다 LRM-Zero로 $$G$$개의 (추론, 답) 쌍 $$\{(R'_1, a_1), \dots, (R'_G, a_G)\}$$을 생성한 뒤, 정답과 일치하는 $$a_k$$를 가진 후보만 남겨 평가 집합 $$\mathcal{G}^{(j)}$$을 만든다. 이 집합에 속한 각 후보 $$g$$에 대해 다음 점수를 계산한다.

$$R^\star(R'_g, a_g, Q) = P(R'_g \mid Q) \cdot P(a_g \mid Q, R'_g) = \underbrace{\frac{1}{L_{R'_g}}\sum_{i=1}^{L_{R'_g}} p'_{g,i}}_{\text{Self-Consistency}} \cdot \underbrace{\frac{1}{L_{a_g}}\sum_{i=1}^{L_{a_g}} p''_{g,i}}_{\text{Validity}}$$

기호를 풀면,

- $$R'_g$$: $$g$$번째 후보의 추론 경로(토큰 시퀀스).
- $$a_g$$: 그 추론이 이끌어낸 답.
- $$p'_{g,i}$$: 추론 경로의 $$i$$번째 토큰이 생성될 조건부 확률.
- $$p''_{g,i}$$: 답의 $$i$$번째 토큰이 생성될 조건부 확률.
- $$L_{R'_g}$$, $$L_{a_g}$$: 각각 추론 경로와 답의 토큰 길이.

앞 항 $$P(R'_g \mid Q)$$는 추론 경로 자체의 평균 토큰 확률로 Self-Consistency를, 뒤 항 $$P(a_g \mid Q, R'_g)$$는 그 추론을 조건으로 답이 나올 평균 토큰 확률로 Validity를 근사한다. 왜 총합이 아니라 **평균**을 쓰는가 — 총합(또는 곱)을 쓰면 길이가 긴 경로가 단지 길다는 이유만으로 불리해진다. 평균을 쓰면 "토큰 하나당 얼마나 확신에 차 있는가"만 남아, 길지만 꼼꼼한 추론이 짧지만 얕은 추론에 부당하게 밀리지 않는다. 질문마다 $$R^\star$$가 가장 높은 경로 하나만 골라 SFT 데이터셋 $$\mathcal{D}_{\text{SFT}}$$을 구성한다.

<p align="center"><img src="/assets/post/image/reasongrm/fig3_rstar_workflow.png" width="85%"></p>

## 토이 예제: $$R^\star$$로 Weaker와 Stronger를 가르기

Background에서 본 와이파이... 아니, 지침 예시로 돌아가 보자. Incorrect Reasoning은 애초에 정답(B)이 아니므로 평가 집합 $$\mathcal{G}^{(j)}$$에 들어가지도 못하고 제외된다. 남는 건 같은 정답 B에 도달한 Weaker Reasoning과 Stronger Reasoning뿐이다. 아래는 이해를 돕기 위해 필자가 구성한 예시 숫자로(논문이 실제로 공개한 수치는 아니다), 두 경로에 각각 6개·4개의 추론 토큰과 2개의 답 토큰이 있다고 가정하고 손으로 계산해본다.

| 경로               | 추론 토큰 확률                     | Self-Consistency (평균) | 답 토큰 확률 | Validity (평균) | $$R^\star$$ |
| ------------------ | ---------------------------------- | ----------------------- | ------------ | --------------- | ----------- |
| Weaker Reasoning   | 0.90, 0.85, 0.40, 0.30, 0.60, 0.95 | 0.667                   | 0.90, 0.85   | 0.875           | 0.583       |
| Stronger Reasoning | 0.90, 0.92, 0.88, 0.95             | 0.913                   | 0.97, 0.95   | 0.960           | 0.876       |

Weaker Reasoning은 "그런데", "잠깐, 사실은..." 하며 헤매는 구간에서 토큰 확률이 0.30~0.40까지 떨어진다 — 다음 말을 이어가는 데 확신이 없다는 뜻이다. 이 저확신 구간이 평균을 0.667까지 끌어내린다. 반면 Stronger Reasoning은 처음부터 끝까지 0.88 밑으로 떨어지는 지점이 없다. 두 경로 모두 답 토큰 확률(Validity)은 비슷하게 높지만, Self-Consistency 항의 차이가 최종 $$R^\star$$ 점수를 0.583 대 0.876으로 벌려놓는다. **같은 정답에 도달했음에도 불구하고, $$R^\star$$는 헤매지 않고 곧장 결론에 이른 Stronger Reasoning을 SFT 데이터로 선택한다.** 이게 이 논문의 핵심 아이디어다 — 정답 여부(Validity)만으로는 가릴 수 없는 추론의 질을, 토큰 생성 확률(Self-Consistency)로 보완해서 가려낸다.

## Stage 3 — 어려운 사례에 집중하는 GRPO

Stage 2의 SFT만으로는 애매하거나 까다로운 사례에서 판별 경계가 여전히 무를 수 있다. 그래서 마지막으로, SFT 모델 $$M_{\text{SFT}}$$에게 각 질문마다 $$N$$개의 답을 다시 추론시켜, 그 $$N$$개가 전부 정답은 아닌 질문들만 모아 hard-case 데이터셋 $$\mathcal{D}_{\text{hard}}$$을 만든다. 이 hard-case 데이터셋에 대해서만, Stage 1과 동일한 outcome 기반 reward로 GRPO를 한 번 더 돌린다. 이미 잘 푸는 문제에 학습을 낭비하지 않고, 모델이 흔들리는 경계 사례에만 그래디언트를 집중시키는 것이다. 이렇게 얻은 최종 모델이 ReasonGRM($$M_{\text{grpo}}$$)이다.

| 단계                     | 입력                              | 목적                                         | 산출물    |
| ------------------------ | --------------------------------- | -------------------------------------------- | --------- |
| Stage 1: Zero-RL         | (질문, 후보, 정답)만              | 결과 기반 판별력 확보, 정보량 큰 샘플에 집중 | LRM-Zero  |
| Stage 2: $$R^\star$$ SFT | LRM-Zero가 만든 후보 추론 $$G$$개 | 확신 있고 정답에 이른 추론만 선별해 학습     | LRM-SFT   |
| Stage 3: Hard-case GRPO  | LRM-SFT가 틀리기 쉬운 사례만      | 애매한 경계 사례의 판별력 강화               | ReasonGRM |

# Experiments

## 세팅

학습 데이터는 Skywork Reward Preference 80K v0.2(수학·코드·논리 추론을 포함한 약 8만 건의 교차 도메인 선호 데이터)이고, base 모델은 QwQ-32B다. 4개 노드 × 8장 A800-80G(총 32장) 클러스터에서 학습률 $$1 \times 10^{-6}$$, 글로벌 배치 256, 최대 프롬프트 길이 8,192 토큰, 응답 길이 상한 57,344 토큰으로 학습했다. 평가는 RewardBench(챗·추론·안전 등 다양한 도메인의 승-패 응답 삼중항), RM-Bench(스타일에 흔들리지 않고 미묘한 내용 차이를 가려내는 능력에 초점), RMB(49개 이상의 세부 시나리오, pairwise + Best-of-N 평가)의 세 벤치마크에서 이뤄졌다.

## 메인 결과

| 모델                                    | RewardBench | RM-Bench | RMB      | 평균     |
| --------------------------------------- | ----------- | -------- | -------- | -------- |
| Skywork-Reward-Gemma-2-27B (SRM)        | 93.8        | 67.3     | 60.2     | 73.8     |
| Nemotron-4-340B-Reward (SRM)            | 92.0        | 69.5     | 69.9     | 77.1     |
| infly/INF-ORM-Llama3.1-70B (SRM)        | **95.1**    | 70.9     | 70.5     | 78.8     |
| GPT-4o-0806 (GRM)                       | 86.7        | 72.5     | 73.8     | 77.7     |
| Skywork-Critic-Llama-3.1-70B (GRM)      | 93.3        | 71.9     | 65.5     | 76.9     |
| RM-R1-Qwen-Instruct-32B (GRM)           | 91.4        | 79.1     | **73.0** | 81.2     |
| RM-R1-DeepSeek-Distilled-Qwen-32B (GRM) | 90.9        | 83.9     | 69.8     | 81.5     |
| **ReasonGRM-QwQ-32B (Ours)**            | 92.3        | **86.3** | 71.3     | **83.3** |

ReasonGRM은 세 벤치마크 평균 83.3점으로 최고 기록을 세웠다 — 이전 최고 GenRM인 RM-R1-DeepSeek-Distilled-Qwen-32B(81.5) 대비 +1.8%p, GPT-4o(77.7) 대비 +5.6%p, 최고 스칼라 RM인 INF-ORM-Llama3.1-70B(78.8) 대비 +4.5%p다. 흥미로운 건 RewardBench 단일 점수로는 INF-ORM이 95.1점으로 ReasonGRM(92.3)보다 2.8점 높다는 점이다. 그런데 그 INF-ORM은 스타일에 흔들리지 않는 능력을 시험하는 RM-Bench에서 70.9점으로 주저앉는다 — ReasonGRM(86.3)보다 15.4점이나 낮다. 한 벤치마크에서만 잘하는 것과, 도메인을 바꿔도 일관되게 잘하는 것은 다른 문제다. 후자가 진짜 추론 능력이 있는지를 가르는 시험대이고, ReasonGRM은 그 시험대에서 가장 안정적이었다.

## Ablation: 각 단계가 기여하는 몫

| 방법              | Chat      | Chat Hard | Safety | Reasoning | Score     |
| ----------------- | --------- | --------- | ------ | --------- | --------- |
| QwQ-32B (Base)    | 95.25     | 80.48     | 88.51  | 97.49     | 90.43     |
| + Zero-RL         | 91.62     | 86.51     | 91.42  | 98.38     | 91.98     |
| + $$R^\star$$ SFT | 92.60     | 86.18     | 91.82  | 98.36     | 92.11     |
| + GRPO (최종)     | **96.09** | 83.55     | 90.81  | **98.74** | **92.30** |

RewardBench 종합 점수 기준으로 Zero-RL이 base 대비 +1.55%p, $$R^\star$$ SFT까지 누적으로 +1.68%p, 최종 GRPO까지 누적으로 +1.87%p를 만든다. 이 표만 보면 $$R^\star$$ SFT의 개별 기여(Zero-RL 대비 +0.13%p)가 작아 보이는데, 이건 QwQ-32B 하나에 대한 결과일 뿐이다. $$R^\star$$의 진짜 가치는 다른 base 모델로 일반화했을 때 드러난다.

## $$R^\star$$의 효과: 다른 모델에서도 통하는가

| Base 모델   | 전략            | Chat      | Chat Hard | Safety    | Reasoning | Score     |
| ----------- | --------------- | --------- | --------- | --------- | --------- | --------- |
| Llama3.1-8B | Random-SFT      | 93.78     | 73.79     | **87.40** | 72.42     | 81.85     |
| Llama3.1-8B | $$R^\star$$-SFT | 93.58     | **74.01** | 86.89     | **73.95** | **82.11** |
| Qwen2.5-7B  | Random-SFT      | 92.95     | 74.95     | **84.70** | 83.88     | 84.12     |
| Qwen2.5-7B  | $$R^\star$$-SFT | **93.51** | **76.48** | 85.30     | **84.38** | **84.92** |
| Qwen2.5-14B | Random-SFT      | 94.62     | 80.65     | **87.84** | 92.12     | 88.81     |
| Qwen2.5-14B | $$R^\star$$-SFT | **95.25** | **81.69** | 87.77     | **93.78** | **89.62** |

같은 LRM-Zero가 만든 정답-일치 후보 풀에서, 무작위로 뽑은 데이터(Random-SFT)와 $$R^\star$$로 고른 데이터($$R^\star$$-SFT)로 각각 SFT했을 때의 차이다. 세 base 모델 전부에서 $$R^\star$$-SFT가 이긴다 — Llama3.1-8B +0.26%p, Qwen2.5-7B +0.80%p, Qwen2.5-14B +0.81%p. 정답만 맞으면 아무 추론이나 학습시켜도 된다는 가정이 틀렸다는 걸, 세 개의 서로 다른 아키텍처·크기에서 일관되게 보여준다.

# 한계와 비용

논문이 스스로 인정하는 한계는 두 가지다. 첫째, ReasonGRM은 정답이 명확히 존재하는 QA형 벤치마크에서만 검증됐고, 더 열린 실제 시나리오나 다중 홉 추론까지 일반화되는지는 확인되지 않았다. 둘째, 더 근본적으로 — $$R^\star$$의 Validity 항은 정답 $$a_g$$가 있어야 계산할 수 있다. 즉 **정답이 없는 open-ended 선호 데이터에는 이 메트릭을 그대로 적용할 수 없다.** 창작, 일반 대화, 코드 스타일처럼 "이게 정답이다"라고 못 박기 어려운 영역에서는 Validity를 무엇으로 잴지부터 다시 정의해야 한다.

여기에 비용 문제가 더해진다. [#29 Generative Verifiers](/blog/2026/generative-verifiers/)에서 짚었던 문제 — judge가 CoT를 쓰면 채점이 forward pass 한 번에서 autoregressive 생성으로 바뀌어 지연시간이 늘어난다 — 가 ReasonGRM에서는 훨씬 커진다. base 모델 자체가 QwQ-32B라는 Large Reasoning Model이고, 응답 길이 상한이 57,344 토큰이다. 판정 한 건을 내리는 데 이 정도 규모의 추론 사슬을 매번 생성해야 할 수 있다는 뜻이다. 학습 비용도 만만치 않다 — 3단계 파이프라인 중 두 단계(Zero-RL, hard-case GRPO)가 GRPO이고, GRPO는 질문 하나당 여러 개의 롤아웃을 생성해야 그래디언트를 계산할 수 있다. 실제로 32장의 A800-80G GPU가 필요했다. 요컨대 judge를 더 잘 생각하게 만드는 대가는, 판정 1건당 지연시간과 학습 파이프라인의 복잡도 양쪽에서 함께 돌아온다.

# Conclusion

핵심을 한 줄로 요약하면: **GenRM에게 무작정 추론을 시키는 것으로는 부족하다. 정답에 도달했는가(Validity)와 그 경로가 확신에 차 있는가(Self-Consistency)를 함께 측정하는 $$R^\star$$로 추론 경로를 선별해야, 판정의 신뢰도가 실제로 오른다.** ReasonGRM은 이를 Zero-RL(결과 기반 초기 판별력) → $$R^\star$$ SFT(고품질 추론만 선별) → hard-case GRPO(어려운 경계 사례 강화)의 3단계로 구현했고, RewardBench·RM-Bench·RMB 평균 83.3점으로 GPT-4o와 기존 최고 GenRM을 넘어섰다.

남은 문제는 두 가지다. 정답이 없는 open-ended 데이터에는 $$R^\star$$를 그대로 못 쓴다는 것, 그리고 그 대가로 판정 비용과 학습 비용이 함께 커진다는 것. 다음 글 [#34 J1](/blog/2026/j1-thinking-judge/)은 같은 목표 — judge에게 사고력을 심는다 — 를 정반대 방향에서 접근한다. ReasonGRM이 후보 추론을 생성한 뒤 $$R^\star$$로 사후 선별해 정적인 SFT 데이터셋을 만드는 2단계짜리 데이터 파이프라인을 거친다면, J1은 처음부터 검증 가능한 reward로 RL을 직접 돌려 "생각하는 judge"를 한 번에 길러낸다. 특히 J1은 검증 가능/불가능한 프롬프트를 하나의 통일된 포맷으로 변환해 보상을 부여하는데, 이는 ReasonGRM이 스스로 인정한 "open-ended 데이터에는 못 쓴다"는 한계를 정면으로 다루려는 시도로 읽힌다.

| 항목                         | ReasonGRM (#33)                                                          | J1 (#34, 예고)                                                            |
| ---------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 학습 절차                    | Zero-RL → $$R^\star$$로 선별한 SFT → hard-case GRPO (3단계)              | 검증 가능한 reward로 RL을 한 번에 적용해 "생각하는 judge"를 직접 학습     |
| 추론 데이터 확보 방식        | 여러 후보를 생성한 뒤 $$R^\star$$로 사후 선별해 정적 SFT 데이터셋 구성   | 별도의 정적 데이터셋 없이 학습 중 reward로 직접 최적화                    |
| 정답 없는(open-ended) 도메인 | $$R^\star$$의 Validity 항이 정답을 요구해 적용 곤란 (논문이 스스로 인정) | 검증/비검증 프롬프트를 통일된 포맷으로 변환해 비검증 도메인까지 확장 시도 |
| Base 모델 규모               | QwQ-32B 단일                                                             | 8B/32B/70B 여러 스케일                                                    |

자세한 메커니즘은 다음 글에서 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 서른세 번째 글이다.

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

**4부. 안전성 정렬**

14. [Safe RLHF (2023)](/blog/2026/safe-rlhf/) — 안전성을 reward가 아니라 제약으로
15. [Rule-Based Rewards (2024)](/blog/2026/rule-based-rewards/) — 안전 규칙을 reward로 직접 번역

**5부. reward를 정책으로**

16. [PPO (2017)](/blog/2026/ppo/) — clipped surrogate objective
17. [Secrets of RLHF I (2023)](/blog/2026/secrets-rlhf-ppo/) — PPO 학습 안정화 트릭
18. [GRPO / DeepSeekMath (2024)](/blog/2026/grpo-deepseekmath/) — value network를 버리다
19. [RLOO (2024)](/blog/2026/rloo-back-to-basics/) — REINFORCE로 충분한가
20. [DPO (2023)](/blog/2026/dpo/) — reward를 없애면 어떻게 되는가
21. [SimPO (2024)](/blog/2026/simpo/) — reference-free + 길이 정규화
22. [KTO (2024)](/blog/2026/kto/) — 선호 쌍 없이 이진 신호만으로
23. [GSPO (2025)](/blog/2026/gspo/) — importance ratio를 시퀀스 단위로
24. [DAPO (2025)](/blog/2026/dapo/) — 신호 없는 프롬프트를 버린다

**6부. Process & Verifiable Reward**

25. [Let's Verify Step by Step (2023)](/blog/2026/lets-verify-step-by-step/) — 과정 감독이 결과 감독을 이긴다
26. [Math-Shepherd (2023)](/blog/2026/math-shepherd/) — 사람 라벨 없는 PRM
27. [DeepSeek-R1 (2025)](/blog/2026/deepseek-r1/) — RLVR, 규칙이 reward가 될 때

**7부. Generative Reward Model**

28. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
29. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
30. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
31. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
32. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**8부. 생각하는 Judge, 그리고 그 신뢰**

33. **(현재 글)** ReasonGRM (2025) — reasoning 능력을 judge에 이식
34. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
35. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
36. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
37. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

**9부. 실전 종합**

38. [프론티어 모델의 reward 설계 (2025~2026)](/blog/2026/frontier-reward-design/) — 열 개 모델이 실제로 택한 것
39. [reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/) — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 39편으로 구성된다.

# 참고 문헌

- Chen et al., 2025. [ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models](https://arxiv.org/abs/2506.16712).
- Whitehouse et al., 2025. [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://arxiv.org/abs/2505.10320).
- Lambert et al., 2024. [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787).
- Liu et al., 2024. [RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style](https://arxiv.org/abs/2410.16184).
- Zhou et al., 2024. [RMB: Comprehensively Benchmarking Reward Models in LLM Alignment](https://arxiv.org/abs/2410.09893).
- Liu et al., 2024. [Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451).
- Chen et al., 2025. [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387).
