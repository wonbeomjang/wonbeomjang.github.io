---
layout: post
title: "Prometheus 2: 평가 기준을 입력으로 받는 judge"
date: 2026-08-11 09:34:00 +0900
description: "RLHF Reward 설계 시리즈 #34 — 오픈 평가자 모델, rubric 조건부 평가, 그리고 절대 점수와 쌍대 비교의 통합"
categories: [paper]
tags: [rlhf, reward-model, llm-as-a-judge, evaluation, rubric, paper]
giscus_comments: true
related_posts: true
---

> [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535) (Kim et al., KAIST AI, EMNLP 2024)

# Introduction

지금까지 6부에 걸쳐 다룬 reward는 전부 같은 모양이었다. 프롬프트와 응답을 넣으면 스칼라 하나가 나오는 **판별 모델(discriminative model)**. [#4 Rethinking Bradley-Terry](/blog/2026/bradley-terry-rethinking/)의 $$r_\theta(x,y)$$든, [#7 ArmoRM](/blog/2026/armorm/)의 다목적 벡터든, [#33 DeepSeek-R1](/blog/2026/deepseek-r1/)의 규칙 기반 RLVR이든, 결국 숫자 하나(혹은 벡터 하나)로 응답의 좋고 나쁨을 압축했다. 그리고 R1이 남긴 질문은 이거였다 — 수학·코드처럼 정답을 규칙으로 검증할 수 있는 도메인은 RLVR로 풀리는데, **정답이 없는 도메인(글쓰기, 대화, 요약)은 어떻게 하나?**

7부는 이 질문에 다른 각도로 답한다. reward를 스칼라로 뱉는 대신, **언어모델이 직접 평가문을 쓰게 하자**는 것이다. "이 응답은 3점이다"가 아니라 "이 응답은 이런 이유로 3점이다"를 자연어로 생성하고, 그 생성 과정 자체가 판별 모델보다 더 풍부한 신호를 준다는 발상이다. 이 전환의 첫 논문이 오늘 다룰 **Prometheus 2**다.

Prometheus 2가 던지는 핵심 발상은 단순하지만 근본적이다. 기존 reward model은 "무엇이 좋은 응답인가"를 학습 과정에서 가중치 안에 굳혀 넣는다. RM 하나를 학습시키면 그 RM은 helpfulness면 helpfulness, safety면 safety라는 고정된 기준으로만 채점한다. Prometheus는 이 기준 자체를 **런타임 입력**으로 받는다. 같은 모델 하나가 "전문 용어를 정확히 썼는가"로 채점할 수도 있고, 다음 순간 "초보자가 이해하기 쉬운가"로 채점할 수도 있다. 평가 기준(score rubric)이 프롬프트의 일부가 되는 순간, 판별 모델은 조건부 생성 모델이 된다.

거기에 더해 Prometheus 2는 오픈소스 평가자 모델이 오랫동안 둘 중 하나만 잘했던 문제 — **절대 점수(direct assessment)와 쌍대 비교(pairwise ranking)를 동시에 잘하는 모델**을 만든다. 이 통합은 [#4 Rethinking Bradley-Terry](/blog/2026/bradley-terry-rethinking/)가 다뤘던 "쌍대 비교로 배운 reward냐, 절대 점수로 배운 reward냐"라는 축과 정확히 같은 문제를, 판별 모델이 아니라 생성형 judge 위에서 다시 마주한 것이다. 그리고 이 둘을 합치는 방법이 각각 따로 학습한 뒤 **가중치를 합치는(weight merging)** 것이라는 점에서, [#14 WARM](/blog/2026/warm-weight-averaged-reward/)의 weight averaging과도 발상이 겹친다.

이 글은 세 가지를 순서대로 짚는다. (1) rubric을 입력으로 받는다는 것이 기존 RM 설계와 왜 다른가, (2) direct assessment와 pairwise ranking을 한 모델에 합치는 구체적 방법(weight merging), (3) 그렇게 만든 오픈 평가자 모델이 GPT-4·사람 판정과 얼마나 가까워졌는가. 그리고 마지막에 이 접근이 여전히 안고 있는 빚 — **rubric은 결국 사람이 써야 한다** — 을 짚고 뒤이은 글들로 넘긴다.

# Background

## Prometheus 1: rubric 평가의 원형

Prometheus 2를 이해하려면 먼저 전작 [Prometheus](https://arxiv.org/abs/2310.08491)(Kim et al., KAIST AI, ICLR 2024)를 알아야 한다. Prometheus 1의 목표는 단순했다. "GPT-4를 judge로 쓰면 잘 맞긴 하는데, 이걸 오픈소스 모델로 대체할 수 있을까?"

이를 위해 GPT-4로 **Feedback Collection**이라는 데이터셋을 만들었다 — 1K개의 세분화된 score rubric, 2만 개의 instruction, 10만 개의 (응답, 언어 피드백) 쌍. 이 데이터로 Llama-2-Chat 13B를 파인튜닝한 결과가 **Prometheus-13B**다. 사람 평가와의 Pearson 상관계수 0.897을 기록했는데, 이는 GPT-4의 0.882보다 오히려 높고 ChatGPT의 0.392를 크게 앞선 수치였다.

여기서 핵심 설계가 바로 **score rubric**이다. 단순히 "이 응답이 몇 점이야?"라고 묻는 게 아니라, 평가 기준 자체를 텍스트로 명시해서 함께 넣는다. 이를테면 "응답이 초보자도 이해할 수 있는 쉬운 언어를 쓰는가?"라는 기준 설명과, 1점부터 5점까지 각 점수가 어떤 응답에 해당하는지 서술한 앵커가 세트로 들어간다. 모델은 이 rubric 텍스트를 조건으로 받아 점수와 이유를 함께 생성한다.

다만 Prometheus 1은 **direct assessment(절대 점수)만** 할 수 있었다. 두 응답 중 뭐가 더 나은지 비교하는 pairwise ranking은 다루지 못했고, 사람·GPT-4와의 상관도 아직 갈 길이 멀었다.

## rubric을 입력으로 받는다는 것 — 왜 근본적인 차이인가

기존 RM 설계와 비교하면 이 차이가 뚜렷해진다. [#7 ArmoRM](/blog/2026/armorm/)은 helpfulness·correctness·verbosity 같은 축을 **미리 고정**해두고, 각 축마다 별도의 헤드를 학습시킨 뒤 MoE 게이팅으로 가중합한다. 축의 개수와 의미는 학습 시점에 이미 정해져 있고, 추론 때는 그 고정된 축들의 조합만 바꿀 수 있다. Prometheus는 반대다. 축 자체를 자연어 rubric으로 프롬프트에 써넣으면, 학습 때 본 적 없는 새로운 기준으로도 즉시 채점한다.

| 구분             | ArmoRM (#7)                      | Prometheus 2                                             |
| ---------------- | -------------------------------- | -------------------------------------------------------- |
| 평가 기준의 형태 | 학습 시 고정된 N개의 축(헤드)    | 추론 시 입력되는 자연어 rubric                           |
| 새 기준 추가     | 헤드 재학습 필요                 | 프롬프트만 교체                                          |
| 출력             | 스칼라 벡터 + 게이팅 가중합      | 점수(1~5) 또는 승패 + 언어 피드백                        |
| 근본 가정        | "좋음"의 축은 유한하고 미리 안다 | "좋음"의 기준은 태스크마다 다르며 텍스트로 지정 가능하다 |
| 해석 가능성      | 축별 스칼라로 부분적             | 생성된 이유(feedback) 문장으로 직접 확인                 |

비유하자면 이렇다. ArmoRM은 미리 정해진 몇 개 과목(수학, 국어, 영어)만 채점하도록 훈련받은 전담 교사다. 반면 Prometheus는 시험마다 새 채점 기준표를 받아 드는 TA에 가깝다 — 오늘은 "논리적 일관성"으로 채점하고, 내일은 "창의성"으로 채점해도 같은 사람(모델)이 그 기준표를 읽고 그대로 따른다. 기준을 모델 안에 굳혀 넣느냐, 모델 밖에서 갈아 끼우느냐의 차이다.

## weak evaluator와 strong evaluator

Prometheus 2 논문은 도입부에서 흥미로운 관찰을 제시한다. 여러 평가자(evaluator)의 채점 결과를 서로 상관분석했더니, 두 개의 군집이 뚜렷하게 갈렸다.

<p align="center"><img src="/assets/post/image/prometheus-2/fig1.png" width="70%"></p>

GPT-4, Claude-3-Opus, 사람 채점자는 서로 강하게 상관된 **strong evaluator 그룹**을 이루는 반면, GPT-3.5, Llama-2-70B, 그리고 전작 Prometheus-13B는 서로는 물론 strong 그룹과도 상관이 낮은 **weak evaluator 그룹**에 속했다. 즉 Prometheus 1은 "오픈소스치고 괜찮은 judge"였지만, 여전히 GPT-4·사람이 보는 눈과는 다른 눈으로 채점하고 있었다는 뜻이다. Prometheus 2의 목표는 명확해진다 — **weak 그룹에서 strong 그룹으로 건너가는 것.**

# Method

## 데이터셋: Feedback Collection과 Preference Collection

Prometheus 2는 두 종류의 평가 능력을 각각 별도 데이터셋으로 학습한다.

| 데이터셋              | 용도                              | 평가 기준 수 | instruction | reference answer | 학습 인스턴스 |
| --------------------- | --------------------------------- | ------------ | ----------- | ---------------- | ------------- |
| Feedback Collection   | direct assessment (절대 점수 1~5) | 1,000        | 20,000      | 20,000           | 100,000       |
| Preference Collection | pairwise ranking (승/패)          | 1,000        | 20,000      | 20,000           | 200,000       |

두 데이터셋 모두 GPT-4가 rubric에 따라 언어 피드백과 점수(또는 승패)를 함께 생성해 만들었다. 구조는 거의 동일하지만 Preference Collection은 같은 instruction에 대해 응답 쌍을 순서를 바꿔가며 제시해 위치 편향(position bias)을 줄였고, 그만큼 인스턴스 수도 두 배로 늘었다.

## direct assessment와 pairwise ranking, 그리고 토이 예제

두 평가 형식이 실제로 얼마나 다른 판정을 낳는지, 논문이 제시한 예제를 그대로 따라가 보자.

<p align="center"><img src="/assets/post/image/prometheus-2/terminology.png" width="90%"></p>

instruction: "소프트웨어 개발에서 '컨테이너화(containerization)'의 의미와 Docker의 역할은?"

- **Response A**: "컨테이너화는 물건을 상자에 담는 것과 비슷하다. 소프트웨어와 구성 요소를 컨테이너에 패키징하는 것이다. Docker는 이 과정을 돕는 도구다…"
- **Response B**: "컨테이너화란 애플리케이션을 관련 설정 파일, 라이브러리, 의존성과 함께 패키징해 독립 실행 단위인 '컨테이너'로 만드는 과정이다…"

여기에 **서로 다른 두 rubric**을 적용하면 판정이 정반대로 갈린다.

1. **Rubric 1 (Pairwise ranking) — "응답이 산업 전문 용어를 정확히 사용하는가?"**: Response A는 "상자에 담는다"는 비유는 쉽지만 전문 용어가 없고, Response B는 "패키징·설정 파일·라이브러리·의존성" 같은 용어를 정확히 쓴다. 판정: **B 승.**
2. **Rubric 2 (Direct assessment) — "초보자도 이해할 수 있는 쉬운 언어를 쓰는가?"를 기준으로 Response A만 채점**: 비유가 이해를 돕는다는 점에서 높은 점수를 주되, 컨테이너화가 왜 중요한지 설명이 빠져 있어 만점에서 1점을 깎는다. 판정: **5점 만점에 4점.**

같은 응답 쌍, 같은 모델인데 **rubric만 바꿨을 뿐인데 승자가 뒤집히고 점수가 달라진다.** 이게 rubric-조건부 평가의 실제 동작이다 — 모델은 "좋은 응답이 무엇인지"에 대한 고정된 관념이 없고, 매번 rubric이 그 관념을 대신 정의해준다.

## 두 형식을 각각 학습한 뒤 합친다

여기서 자연스러운 질문이 나온다. "그럼 두 데이터셋을 섞어서 한 번에 학습시키면 되지 않나?" 논문은 이걸 직접 실험했고, 결과는 반대였다. direct assessment와 pairwise ranking 데이터를 합쳐 **공동 학습(joint training)**한 모델은, 각 형식만 단독으로 학습한 모델보다 오히려 상관도가 낮았다. 두 태스크의 출력 형식(스칼라 점수 vs 승패 판정)과 요구되는 추론 패턴이 달라, 한 모델이 동시에 두 목적함수를 최적화하려다 보면 서로 간섭한 것으로 보인다.

그래서 택한 방법이 **weight merging**이다. Mistral-7B-Instruct-v0.2를 base로 Feedback Collection만 학습한 모델($$\theta_{DA}$$)과 Preference Collection만 학습한 모델($$\theta_{PW}$$)을 각각 만든 뒤, 두 가중치를 사후에 합친다.

비유하자면 이렇다. 한 사람에게 절대평가와 상대평가를 동시에 훈련시키면(joint training) 두 채점 습관이 뒤섞여 오히려 어느 쪽도 서툴러진다. 대신 한 명은 절대평가만, 다른 한 명은 상대평가만 각각 통달할 때까지 따로 훈련시킨 뒤, 두 사람의 "채점 감각"을 뇌 지도처럼 겹쳐 한 사람 안에 이식하는 쪽이 더 잘 작동했다는 뜻이다. 가장 단순한 이식 방법은 선형 보간이다.

$$\theta_{merge} = \alpha \cdot \theta_{DA} + (1-\alpha) \cdot \theta_{PW}$$

- $$\theta_{DA}$$: direct assessment만 학습한 모델의 가중치
- $$\theta_{PW}$$: pairwise ranking만 학습한 모델의 가중치
- $$\alpha$$: 혼합 비율. $$\alpha=1$$이면 순수 DA 모델, $$\alpha=0$$이면 순수 PW 모델

<p align="center"><img src="/assets/post/image/prometheus-2/fig3.png" width="80%"></p>

이 그래프는 혼합 비율을 1:9부터 9:1까지 훑은 결과다. DA correlation(초록)은 DA 비중이 커질수록 오르고, PW accuracy(파랑)는 반대로 PW 비중이 커질수록 오른다 — 예상대로다. 흥미로운 건 검은 선(평균 성능)의 정점이 **5:5 근처**에 있다는 점이다. 둘을 절반씩 섞을 때 두 능력의 합이 가장 균형 있게 최대화된다.

최종 모델은 선형 보간보다 조금 더 정교한 **DARE-Linear**를 사용했다. DARE(Drop And REscale)는 각 파라미터의 파인튜닝 델타 중 일부를 확률적으로 지우고 남은 델타를 그만큼 키워 보정한 뒤 합치는 방법이다.

$$\theta_{DARE} = \theta_{base} + \frac{1}{1-p}\,M \odot (\theta_{ft} - \theta_{base})$$

- $$\theta_{base}$$: 병합 전 공통 base 모델(Mistral-7B-Instruct-v0.2)의 가중치
- $$\theta_{ft} - \theta_{base}$$: 파인튜닝으로 생긴 델타(작업 벡터)
- $$M$$: 각 파라미터를 확률 $$p$$로 0으로 지우는 이진 마스크. Prometheus 2는 $$p=0.1$$을 사용
- $$\frac{1}{1-p}$$: 지워진 만큼 남은 델타를 재조정(rescale)하는 계수

직관은 이렇다. 두 모델의 델타를 그냥 더하면 서로 다른 태스크를 위해 바뀐 파라미터끼리 충돌(간섭)할 수 있다. DARE는 델타의 상당 부분을 랜덤으로 지워 희소하게 만든 뒤 남은 부분을 증폭시켜, 병합했을 때의 충돌을 줄인다. 논문은 이 밖에도 SLERP, Task Arithmetic, TIES, DARE-TIES까지 총 6가지 병합 기법을 비교했는데, DARE-Linear가 DA 평균 상관 0.660·PW 평균 정확도 78.44%로 단순 Linear(0.652 / 78.06%)를 근소하게 앞섰다.

| 방법                                                | 핵심 아이디어                                     | 비고                                                            |
| --------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| Joint training                                      | 두 데이터셋을 섞어 한 모델을 학습                 | 상관도 하락 — 두 태스크가 서로 간섭                             |
| Linear merging                                      | $$\alpha\theta_{DA}+(1-\alpha)\theta_{PW}$$       | 단순하지만 델타 충돌 가능                                       |
| DARE-Linear                                         | 델타를 확률적으로 지우고 재조정 후 합산           | 최종 채택. Linear 대비 근소 우위                                |
| [#14 WARM](/blog/2026/warm-weight-averaged-reward/) | 같은 데이터·같은 목적함수로 학습한 여러 RM을 평균 | reward hacking에 강건한 **하나의 스칼라 RM**을 만드는 것이 목적 |

WARM과 Prometheus 2의 weight merging은 "가중치를 합친다"는 표면적 동작은 같지만 목적이 다르다. WARM은 **같은 태스크**를 여러 시드로 반복 학습한 RM들을 평균 내 hacking에 강건한 단일 스칼라 reward를 얻으려는 것이고, Prometheus 2는 **서로 다른 태스크**(DA vs PW)를 각각 배운 두 모델을 합쳐 **한 모델이 두 능력을 모두 갖게** 만드는 것이다. 전자는 분산을 줄이는 앙상블에 가깝고, 후자는 서로 다른 스킬을 이식하는 데 가깝다.

8x7B 버전(Mixtral-8x7B-Instruct-v0.1 기반)은 계산 자원 제약으로 별도의 병합 기법 탐색 없이 7B에서 찾은 DARE-Linear 설정을 그대로 적용해 릴리스했다.

# Experiments

## Direct assessment: GPT-4-1106과의 상관

DA 벤치마크 4종(Vicuna Bench, MT Bench, FLASK, Feedback Bench) 각각에서 GPT-4-1106 채점과의 Pearson 상관을 측정했다.

| 벤치마크       | 규모                      | Prometheus 2-7B | Prometheus 2-8x7B |
| -------------- | ------------------------- | --------------- | ----------------- |
| Vicuna Bench   | 80 prompt / 80 rubric     | 0.666           | 0.685             |
| MT Bench       | 80 prompt / 80 rubric     | 0.548           | 0.665             |
| FLASK          | 200 prompt / 12 rubric    | 0.617           | 0.659             |
| Feedback Bench | 200 prompt / 1,000 rubric | 0.882           | 0.898             |

이전 오픈 평가자(Prometheus-13B, Auto-J 13B, UltraRM-13B)는 이 네 벤치마크에서 GPT-4와의 상관이 대체로 0.5 아래였다. Prometheus 2는 8x7B 기준 전 벤치마크에서 0.65 이상을 기록하며 격차를 좁혔다.

## Pairwise ranking: 정확도

PW 벤치마크 4종에서 8x7B의 승패 판정 정확도는 다음과 같다.

| 벤치마크                | 규모                                 | Prometheus 2-8x7B 정확도 |
| ----------------------- | ------------------------------------ | ------------------------ |
| HHH Alignment           | 221 prompt / 4 rubric                | 85.52%                   |
| MT Bench Human Judgment | 80 prompt / 3,360 pair               | 71.96%                   |
| Auto-J Eval             | 58 prompt / 1,392 pair               | 79.98%                   |
| Preference Bench        | 200 prompt / 2,000 pair / 200 rubric | 90.65%                   |

PairRM(0.4B)처럼 pairwise 전용으로 만들어진 소형 모델과 비교해도 밀리지 않는 수준이고, direct assessment 능력을 동시에 갖췄다는 점에서 실질적으로 대체 범위가 넓다.

## 사람 평가와의 격차 — FLASK에서 절반으로

가장 인상적인 숫자는 FLASK 벤치마크에서 사람 채점자와의 상관이다.

$$\text{gap} = \rho_{\text{GPT-4, human}} - \rho_{\text{model, human}}$$

- GPT-4와 사람의 상관: 0.679
- Prometheus-13B(전작)와 사람의 상관: 0.449 → gap = 0.230
- Prometheus-2-8x7B와 사람의 상관: 0.555 → gap = 0.124

전작 대비 gap이 0.230에서 0.124로, 거의 절반으로 줄었다. 여전히 GPT-4에는 못 미치지만, "weak evaluator 그룹에서 strong evaluator 그룹으로" 라는 Background에서 세운 목표에 실제로 다가갔다는 근거다. 이 개선의 출처를 나눠 보면, rubric-조건부 학습 자체는 이미 Prometheus 1에서 확보돼 있었으니 나머지 절반은 (1) 더 큰 base 모델(Mixtral 8x7B)과 (2) direct assessment·pairwise ranking을 함께 갖춘 weight merging 레시피에서 왔다고 볼 수 있다.

## 왜 오픈 평가자 모델이어야 하는가

논문이 도입부에서 강조하는 동기는 세 가지다. 첫째, GPT-4 같은 proprietary judge는 학습 데이터가 공개되지 않아 **평가의 공정성과 재현성**을 검증할 수 없다. 둘째, API 뒤에서 모델이 조용히 업데이트되면 지난달 벤치마크 점수와 이번 달 점수가 같은 잣대인지 보장할 수 없는 **통제 가능성** 문제가 있다. 셋째, 대량의 평가를 반복적으로 돌려야 하는 RLHF 파이프라인 안에서 매 스텝 API를 호출하는 것은 **비용** 부담이 크다. 오픈 가중치 judge는 이 세 가지를 동시에 해결한다 — 로컬에서 몇 번이고 재현 가능하고, 버전을 고정할 수 있고, 한 번 서빙 인프라를 세우면 추가 비용이 API 호출보다 훨씬 낮다.

# Conclusion

Prometheus 2가 남긴 한 줄은 이거다. **평가 기준을 모델 가중치가 아니라 프롬프트로 옮기고, 절대 점수와 쌍대 비교를 따로 학습한 뒤 가중치 병합으로 합치면, 오픈소스 모델도 GPT-4·사람의 판정 그룹에 가까워질 수 있다.** rubric-조건부 평가라는 발상 자체는 전작 Prometheus 1에서 왔지만, 두 평가 형식의 통합과 weight merging이라는 구체적 레시피는 이 논문에서 완성됐다.

다만 이 접근은 빚을 하나 남긴다. rubric은 여전히 **사람이 손으로 써줘야 한다.** Feedback Collection과 Preference Collection의 1,000개 rubric은 GPT-4의 도움을 받았다 해도 결국 사람이 설계한 틀 안에서 만들어졌다. 채점 기준을 자동으로, 그것도 태스크에 맞게 스스로 생성할 수 있다면 어떨까? 이 질문은 뒤에서 두 갈래로 이어진다 — [#41 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)는 비검증 도메인에 rubric 기반 reward를 직접 적용하는 방향으로, [#38 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/)은 원칙(principle) 자체를 모델이 스스로 생성하게 만드는 방향으로 이 빚을 갚으려 한다.

그리고 더 근본적인 질문도 남는다. Prometheus 2는 여전히 "점수 또는 승패"라는 이산적 출력을 생성한다. 만약 reward 자체를 **다음 토큰 예측**의 연장선으로 다룰 수 있다면? 이 물음이 다음 글, [#35 Generative Verifiers](/blog/2026/generative-verifiers/)로 이어진다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 서른네 번째 글이다.

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
  <li><a href="/blog/2026/sycophancy/">Sycophancy (2023)</a> — RM은 사실보다 동의를 좋아한다</li>
  <li><a href="/blog/2026/warm-weight-averaged-reward/">WARM (2024)</a> — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="15">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
  <li><a href="/blog/2026/deliberative-alignment/">Deliberative Alignment (2024)</a> — 안전 명세를 모델의 추론 안으로</li>
  <li><a href="/blog/2026/shallow-safety-alignment/">Shallow Safety Alignment (2024)</a> — 정렬은 첫 몇 토큰에만 얹혀 있다</li>
  <li><a href="/blog/2026/or-bench/">OR-Bench (2024)</a> — 과잉 거절을 어떻게 측정할 것인가</li>
</ol>

**5부. reward를 정책으로**

<ol start="20">
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

<ol start="31">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="34">
  <li><strong>(현재 글)</strong> Prometheus 2 (2024) — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="39">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="44">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어의 helpfulness reward 설계</a> — 열한 개 모델이 능력 축에서 택한 것</li>
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 46편으로 구성된다.

# 참고 문헌

- Kim et al., 2024. [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535). EMNLP 2024.
- Kim et al., 2023. [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491). ICLR 2024.
- [ACL Anthology: Prometheus 2 (EMNLP 2024 Main)](https://aclanthology.org/2024.emnlp-main.248/).
- [Hugging Face Paper Page: Prometheus 2](https://huggingface.co/papers/2405.01535).
