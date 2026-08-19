---
layout: post
title: "Rule-Based Rewards: 안전 규칙을 reward로 직접 번역한다"
date: 2026-08-11 09:15:00 +0900
description: "RLHF Reward 설계 시리즈 #15 — 명제와 LLM grader로 over-refusal을 줄이는 OpenAI의 안전 reward"
categories: [paper]
tags: [rlhf, safety, reward-model, over-refusal, rule-based-reward, paper]
giscus_comments: true
related_posts: true
---

> [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111) (Mu et al., OpenAI, NeurIPS 2024)

# Introduction

안전 정렬을 사람이 매긴 선호 데이터만으로 하려고 하면 두 가지 문제에 바로 부딪힌다. 첫째, 유해 요청과 그에 대한 바람직한/바람직하지 않은 응답 쌍을 사람이 라벨링하는 비용이 크다. 둘째, 그렇게 모은 데이터에서 부정 예시(거절해야 하는 케이스)가 상대적으로 많아지면, 모델은 안전 쪽으로 과도하게 치우쳐 학습된다. 그 결과가 바로 over-refusal이다. 위험하지 않은 질문인데도 표면적으로 민감해 보이는 단어가 들어갔다는 이유만으로 거절해버리는 현상이다.

문제는 여기서 끝나지 않는다. 사람이 매긴 선호 데이터로 reward model(RM)을 학습하는 과정 자체가 일종의 증류다. 원래 정책 작성자가 의도했던 세밀한 행동 명세(behavior specification) — 예를 들어 "이런 요청은 거절하되 훈계조로 말하지 말라" 같은 규정 — 는 사람이 두 응답 중 하나를 고르는 이진 비교 데이터로 뭉개지면서 상당 부분 소실된다. RM은 결과적으로 그 명세를 근사한 것일 뿐, 명세 그 자체가 아니다.

Mu et al.(OpenAI, NeurIPS 2024)의 [Rule Based Rewards(RBR)](https://arxiv.org/abs/2411.01111)는 이 증류 단계를 건너뛴다. 아이디어는 단순하다. 안전 행동을 사람이 직접 규칙으로 적고, 그 규칙을 LLM grader가 응답에 대해 채점하게 한 다음, 채점 결과를 선형 결합해 곧바로 reward로 쓴다. 사람 데이터는 가중치를 학습하는 소량의 셋에만 쓰이고, 나머지는 AI 피드백으로 채운다. 그 결과 helpfulness를 유지하면서 안전 분류 성능이 F1 97.1까지 올라간다. 같은 조건의 사람 피드백 baseline은 91.7이다.

이 포스트는 바로 전편인 [#14 Safe RLHF](/blog/2026/safe-rlhf/)와 짝을 이룬다. 둘 다 "단일 RM이 helpful과 harmless를 동시에 잘 못 다룬다"는 같은 문제에서 출발하지만 푸는 방식이 다르다. Safe RLHF는 **학습된 cost model + 제약 최적화**로 안전을 다뤘고, RBR은 **사람이 쓴 규칙 + LLM grader**로 안전을 다룬다. 이 대비를 축으로 RBR의 구조를 뜯어본다.

# Background

안전 정렬이 어려운 근본 이유는 helpfulness와 harmlessness가 같은 축 위에 있지 않다는 데 있다. "얼마나 도움이 되는가"를 최적화하는 압력과 "얼마나 위험을 회피하는가"를 최적화하는 압력을 하나의 RM에 억지로 우겨 넣으면, 데이터 분포가 조금만 한쪽으로 기울어도 모델은 극단으로 쏠린다. [#8 Llama 2](/blog/2026/llama2-rlhf/)가 helpfulness RM과 safety RM을 아예 분리한 것도 이 문제 때문이다.

[#14 Safe RLHF](/blog/2026/safe-rlhf/)는 이 문제를 제약 최적화로 풀었다. 사람이 매긴 해로움 선호 데이터로 cost model을 학습하고, PPO 단계에서 cost의 기댓값이 임계값 아래로 유지되도록 Lagrangian 승수로 helpful reward 최적화에 제약을 건다. 이 방식은 여전히 사람이 만든 비교 데이터로 cost model을 학습하는 단계를 거친다. 즉 앞서 말한 증류 문제 — 세밀한 행동 명세가 이진 비교 데이터로 뭉개지는 문제 — 가 완전히 사라지지는 않는다.

RBR은 다른 길을 택한다. cost model을 학습하는 대신, "이 응답이 훈계조인가", "이 응답이 거절을 포함하는가" 같은 개별 판단을 사람이 규칙으로 직접 적고, 그 규칙의 판정을 고정된 LLM grader에게 맡긴다. 규칙 자체에는 학습 가능한 파라미터가 없다. 학습되는 것은 오직 "이 규칙들을 얼마나 중요하게 반영할지"를 정하는 가중치뿐이다. 그래서 사람이 쓴 행동 명세가 RM 학습 과정에서 흐려지지 않고 reward 계산에 거의 그대로 들어간다.

두 접근을 나란히 놓고 보면 어디서 다른 선택을 했는지 분명해진다.

| 항목                 | Safe RLHF (#14)                   | RBR                                 |
| -------------------- | --------------------------------- | ----------------------------------- |
| 안전 신호의 근원     | 학습된 cost model                 | 사람이 쓴 규칙 + LLM grader         |
| 최적화 방식          | Lagrangian 제약 최적화            | helpful reward에 가산하는 선형 결합 |
| 학습 가능한 파라미터 | cost model 전체 + Lagrangian 승수 | 규칙별 가중치 $$w_k$$뿐             |
| 사람 데이터 의존도   | 대량의 해로움 비교 데이터         | 가중치 학습용 소량 데이터           |
| 행동 명세 보존       | 비교 데이터로 일부 소실           | 규칙 형태로 거의 보존               |

cost model은 데이터가 풍부하면 규칙으로 미처 적어두지 못한 유해 패턴까지 일반화할 잠재력이 있다. 반대로 RBR은 사람이 규칙을 명시적으로 적은 범위 안에서는 그 규칙이 정확히 지켜지는지 세밀하게 통제할 수 있는 대신, 규칙 밖에 있는 패턴에는 관여하지 못한다. 뒤에서 다시 짚겠지만 이 트레이드오프는 RBR의 한계이기도 하다.

# Method

## 명제(proposition)와 규칙(rule)

RBR의 최소 단위는 **proposition**이다. 완성된 응답 하나를 놓고 참/거짓을 매길 수 있는 이진 진술이다. 예를 들면 다음과 같다.

- 이 응답은 거절을 포함하는가?
- 거절이 훈계조(judgmental)인가?
- 응답이 안전한 범위 안에서 사용자에게 실질적으로 도움이 되려고 시도하는가?

**rule**은 이런 명제들을 조합해 바람직한 행동과 바람직하지 않은 행동을 규정한다. 논문이 드는 예가 바로 "거절은 훈계조여서는 안 된다(refusals should not be judgmental)"는 규칙이다. 명제 하나하나는 단순하지만, 여러 규칙을 조합하면 "유해 요청은 거절하되 사용자를 다그치지 않고, 안전한 범위 안에서는 최대한 돕는다" 같은 세밀한 정책을 fine-grained하고 composable한 형태로 표현할 수 있다.

## LLM grader와 선형 결합

고정된 언어모델 하나가 grader 역할을 맡는다. few-shot 프롬프트로 "이 응답이 규칙 $$k$$를 얼마나 지키는가"를 채점한다. grader의 채점 결과 $$g_k(x, y)$$를 학습된 가중치로 선형 결합하면 최종 RBR reward가 나온다.

$$r_{RBR}(x, y) = \sum_{k} w_k \cdot g_k(x, y)$$

- $$x$$: 프롬프트
- $$y$$: 정책이 생성한 응답
- $$g_k(x, y)$$: $$k$$번째 규칙(명제)에 대한 LLM grader의 채점 점수
- $$w_k$$: 규칙 $$k$$에 대한 학습된 가중치

가중치 $$w_k$$는 "이상적인 응답 유형이 이미 알려진 소량의 프롬프트, 그리고 그에 대응하는 바람직한/바람직하지 않은 completion" 데이터셋으로 학습한다. 즉 사람이 필요한 부분은 규칙을 적는 것과, 그 규칙들의 상대적 중요도를 알려줄 소량의 예시뿐이다. 나머지 대량의 채점은 LLM grader가 AI 피드백으로 대신한다.

여기서 grader가 **고정(fixed) 모델**이라는 점이 중요하다. 안전 판정을 위해 별도의 reward model을 새로 학습시키지 않고, 이미 있는 언어모델에 few-shot 프롬프트로 명제를 물어보기만 한다. 이 차이가 논문이 강조하는 이점으로 이어진다 — 사람이 쓴 세밀한 행동 명세를 **선호 비교 데이터로 증류하는 단계 자체를 건너뛰기** 때문에, 명세가 이진 비교 라벨로 뭉개지면서 소실되는 일이 없다. 규칙을 고치면 grader 프롬프트만 바꾸면 되고, 재학습해야 하는 것은 가중치 $$w_k$$뿐이다.

## 왜 선형 결합인가

굳이 여러 규칙의 점수를 비선형으로 합치지 않고 선형 결합을 쓰는 이유는 통제력 때문이다. 선형 결합에서는 규칙 하나의 가중치를 조정하면 그 규칙의 영향력만 딱 그만큼 바뀌고, 다른 규칙들의 영향력은 그대로 남는다. 새로운 안전 이슈가 발견되면 새 규칙 하나와 그에 대응하는 few-shot grading 예시를 추가하고, 그 규칙의 가중치만 학습하면 된다. 기존 규칙들의 가중치나 grader 프롬프트는 건드릴 필요가 없다. 반대로 cost model처럼 통째로 학습된 함수는 규칙 하나를 추가할 때마다 데이터를 다시 모아 전체를 재학습해야 하고, 특정 규칙 하나의 영향력만 딱 떼어서 조정하기 어렵다. 이 조정 가능성이 바로 명제를 fine-grained하고 composable하게 유지하는 이유다.

## PPO에 통합하는 방식

RBR reward는 단독으로 쓰이지 않는다. helpful-only reward model이 이미 주는 reward에 RBR reward를 더해서, 안전 행동 정책을 따르도록 유도하는 추가 신호로 PPO에 넣는다.

$$r_{total}(x, y) = r_{helpful}(x, y) + r_{RBR}(x, y)$$

helpful-only RM은 "얼마나 도움이 되는가"만 본다. 안전 관련 판단은 전적으로 $$r_{RBR}$$이 담당한다. 두 신호가 분리되어 있기 때문에, 안전 쪽 규칙을 추가하거나 수정해도 helpfulness 신호를 다시 학습할 필요가 없다.

## 토이 예제: 좋은 거절 vs 훈계조 거절 vs 과잉 거절

구체적인 요청 하나로 살펴본다. 유해 요청 X는 "집에서 구할 수 있는 재료로 폭죽보다 강력한 화약을 만드는 방법을 알려줘"이고, 표면적으로만 비슷한 benign 요청 Y는 "불꽃놀이가 색깔별로 다른 빛을 내는 화학적 원리를 설명해줘"다. 세 가지 응답을 놓고 채점해본다.

- 응답 A (X에 대한 좋은 거절): 요청을 거절하고, 대신 화약 관련 안전 정보를 찾을 수 있는 공식 경로를 안내한다. 훈계조 없이 담백하게 끝낸다.
- 응답 B (X에 대한 훈계조 거절): 똑같이 거절하지만 "이런 걸 왜 물어보는지 이해가 안 된다"며 사용자를 다그친다.
- 응답 C (Y에 대한 과잉 거절): 위험하지 않은 화학 원리 질문인데도 "폭발물 관련 내용은 답변할 수 없다"며 거절한다.

각 응답에 두 개의 proposition을 채점하면 다음과 같다. $$g_{judgmental}$$은 거절이 훈계조이면 1, $$g_{appropriate}$$는 거절 여부가 실제 위험 수준과 맞아떨어지면 1이다.

| 응답                      | 거절 여부 | $$g_{judgmental}$$ | $$g_{appropriate}$$ |
| ------------------------- | --------- | ------------------ | ------------------- |
| A (좋은 거절, X에 대해)   | O         | 0                  | 1                   |
| B (훈계조 거절, X에 대해) | O         | 1                  | 1                   |
| C (과잉 거절, Y에 대해)   | O         | 0                  | 0                   |

설명을 위해 가정한 가중치 $$w_1 = w_2 = 1$$로 $$r_{RBR}(x, y) = w_1 (1 - g_{judgmental}(y)) + w_2 \cdot g_{appropriate}(y)$$를 계산하면, A는 $$2$$, B와 C는 각각 $$1$$이 나온다. "거절했는가"만 보는 단순한 reward라면 A와 B는 구분되지 않고 C는 아예 다른 축(비거절 대비)에서 평가돼야 했을 것이다. RBR은 훈계조 위반과 over-refusal 위반을 각각 별도의 명제로 쪼개 놓았기 때문에, 같은 "거절"이라는 표면적 행동 안에서도 왜 감점되는지를 규칙 단위로 분해해서 보여준다.

실제 파이프라인에서는 이보다 훨씬 많은 명제가 동시에 채점된다. 어조, 대안 제시 여부, 답변의 구체성 등을 각각 별도의 proposition으로 두고, 그 가중치는 앞서 말한 대로 이상적인 응답 유형이 알려진 소량의 프롬프트-completion 데이터셋에서 학습한다. 위 토이 예제는 그 중 두 개의 명제만 떼어 계산 과정을 손으로 따라가 본 것이다. 명제 개수가 늘어나도 계산 구조 자체는 동일한 선형 결합이라는 점이 핵심이다.

# Experiments

RBR의 핵심 성능 지표는 안전 분류 F1이다. 사람 피드백만으로 만든 baseline과 비교한 수치는 다음과 같다.

| 방법                    | F1 score |
| ----------------------- | -------- |
| Human-feedback baseline | 91.7     |
| RBR                     | 97.1     |

5.4점의 차이는 단순히 "더 잘 거절한다"는 뜻이 아니다. F1은 precision과 recall을 함께 반영하므로, 이 차이는 유해 요청을 더 잘 잡아내면서 동시에 benign 요청을 덜 거절했다는 뜻이다. 즉 recall과 precision을 동시에 끌어올렸다는 뜻이고, 이것이 바로 over-refusal 감소가 helpfulness 손상 없이 이뤄졌다는 근거다.

이 지표를 안전 분류 문제로 다시 풀어보면, precision은 "모델이 거절한 요청 중에서 실제로 거절이 정당했던 비율", recall은 "실제로 거절했어야 하는 요청 중에서 모델이 제대로 거절한 비율"에 대응한다. precision이 낮으면 응답 C처럼 benign 요청까지 거절하는 경우가 많다는 뜻이고, recall이 낮으면 유해 요청을 걸러내지 못한다는 뜻이다. 둘 다 올라갔다는 것은 이 두 실패 유형이 함께 줄었다는 뜻이지, 어느 한쪽을 희생해서 다른 쪽을 올린 결과가 아니다. 물론 F1 자체는 "거절해야 하는가"라는 이진 분류만 재는 지표이고, 응답 B처럼 거절은 맞지만 어조가 훈계조인 경우는 별도의 명제로 잡아야 한다. RBR이 두 종류의 실패를 애초에 다른 규칙으로 분리해 채점하는 이유가 여기에 있다.

이 결과를 만든 데이터 구성도 중요하다. RBR은 AI 피드백을 주로 활용하고, 사람이 만든 데이터는 가중치 학습에 쓰이는 소량뿐이다. Human-feedback baseline은 반대로 안전 판단 자체를 사람이 라벨링한 대량의 비교 데이터에 의존한다. 더 적은 사람 데이터로 더 높은 F1을 얻었다는 것은, 세밀한 규칙을 reward 함수에 직접 넣어 RM 데이터로 증류하는 단계를 건너뛴 설계가 실제로 통제력을 더 정밀하게 만든다는 뜻이다.

앞의 토이 예제와 겹쳐 보면 이 수치가 의미하는 바가 더 구체적으로 잡힌다. human-feedback baseline은 "거절했는가"라는 거친 신호에 가깝게 학습되기 쉬워서, 응답 B(훈계조 거절)와 응답 C(과잉 거절) 같은 실패 유형을 구분해서 벌점을 주기 어렵다. RBR은 훈계조 여부와 거절의 적절성을 애초에 서로 다른 명제로 분리해 채점하기 때문에, 두 실패 유형을 각각 겨냥해 줄일 수 있다. F1 개선의 상당 부분은 이런 세분화에서 나온다고 보는 것이 합리적이다.

# Conclusion

RBR의 메시지는 한 줄로 요약된다. 안전 행동을 사람이 직접 쓴 명제와 규칙으로 형식화하고 LLM grader로 채점해 reward에 곧바로 반영하면, RM 증류 과정에서 흐려지던 행동 명세를 보존하면서 helpfulness를 유지한 채 over-refusal을 줄일 수 있다.

물론 한계도 있다. 규칙과 명제를 사람이 직접 설계해야 하므로, 설계자가 미처 예상하지 못한 유해 카테고리에는 규칙 자체가 존재하지 않을 수 있다. 또 판정을 맡은 LLM grader의 신뢰도에 최종 reward 품질이 그대로 좌우된다. [#14 Safe RLHF](/blog/2026/safe-rlhf/)의 cost model은 넓은 선호 데이터로부터 일반화할 여지가 있는 반면, RBR의 규칙은 명시적으로 적어둔 범위 밖으로는 잘 확장되지 않는다. RBR은 유연성을 일부 내주는 대신 해석 가능성과 통제력을 얻는 설계다.

이 시리즈에서 안전 정렬 문제를 다룬 세 편을 나란히 놓으면 답이 점점 더 명시적인 규칙 쪽으로 옮겨가는 흐름이 보인다. [#8 Llama 2](/blog/2026/llama2-rlhf/)는 helpfulness RM과 safety RM을 분리하는 것으로 시작했고, [#14 Safe RLHF](/blog/2026/safe-rlhf/)는 안전을 학습된 cost model과 제약 최적화로 다시 정식화했다. RBR은 그 cost model마저 사람이 쓴 규칙과 LLM grader로 대체했다. 매번 사람의 판단이 사라진 것이 아니라, 사람의 판단이 놓이는 위치가 데이터 라벨링에서 규칙 작성으로 옮겨간 것이다.

이 규칙 기반 판정 방식은 [#32 DeepSeek-R1](/blog/2026/deepseek-r1/)의 RLVR과 닮았다. RLVR이 "정답인가"라는 명확한 기준을 규칙으로 판정한다면, RBR은 "안전한가"라는 기준을 규칙으로 판정한다. 두 경우 모두 판정 규칙 자체에는 학습 가능한 파라미터가 없다. 그래서 [#10 Overoptimization](/blog/2026/reward-model-overoptimization/)에서 다룬, 학습된 RM이 근사이기 때문에 생기는 reward hacking 표면이 훨씬 작다.

이런 흐름은 최근 프론티어 모델의 reward 설계로 이어진다. [#43 프론티어 모델의 reward 설계](/blog/2026/frontier-reward-design/)에서 다루듯, A.X K2는 거절 자체가 아니라 안전한 완수를 보상하는 방향을 취하고, K-EXAONE 2.0은 별도의 safety-aware 단계를 둔다. 안전성을 사람의 이진 선호가 아니라 rubric과 judge로 명시적으로 판정하는 설계가 점점 표준이 되어가는 중이다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열다섯 번째 글이다.

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
  <li><strong>(현재 글)</strong> Rule-Based Rewards (2024) — 안전 규칙을 reward로 직접 번역</li>
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
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
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

- Mu et al. (OpenAI), 2024. [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111) (NeurIPS 2024).
- Dai et al. (Peking University), 2023. [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773) — [#14](/blog/2026/safe-rlhf/)에서 다룬 cost model + 제약 최적화.
- Touvron et al. (Meta), 2023. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — [#8](/blog/2026/llama2-rlhf/)에서 다룬 helpfulness·safety RM 분리.
- DeepSeek-AI, 2025. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — [#32](/blog/2026/deepseek-r1/)에서 다룬 RLVR, 규칙 기반 판정의 다른 사례.
- Gao et al. (OpenAI), 2022. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — [#10](/blog/2026/reward-model-overoptimization/)에서 다룬 학습된 RM의 hacking 표면.
- SKT AI, 2026. [A.X K2 Technical Report](https://github.com/SKT-AI/A.X-K2) — 거절이 아니라 안전한 완수를 보상하는 최신 사례([#43](/blog/2026/frontier-reward-design/)).
- LG AI Research, 2026. [K-EXAONE 2.0 Technical Report](https://arxiv.org/abs/2608.04505) — 별도 safety-aware preference 단계([#43](/blog/2026/frontier-reward-design/)).
