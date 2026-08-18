---
layout: post
title: "프론티어 모델은 reward를 어떻게 설계했나"
date: 2026-08-11 09:27:00 +0900
description: "RLHF Reward 설계 시리즈 #32 — DeepSeek·Qwen·Llama·Kimi·Solar가 실전에서 택한 reward 설계 비교"
categories: [paper]
tags: [rlhf, reward-model, rlvr, dpo, grpo, deepseek, qwen, llama, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 다섯 프론티어 모델의 technical report를 가로지른다 — [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [Qwen2.5](https://arxiv.org/abs/2412.15115), [Llama 3](https://arxiv.org/abs/2407.21783), [Kimi K2](https://arxiv.org/abs/2507.20534), [Solar](https://arxiv.org/abs/2601.07022).

# Introduction

31편 동안 이 시리즈는 reward를 부품 단위로 뜯어봤다. 사람 선호를 스칼라로 압축하는 Bradley-Terry([#4](/blog/2026/bradley-terry-rethinking/)), 그 스칼라가 hacking당하는 방식과 방어법([#10](/blog/2026/reward-model-overoptimization/)~~[#13](/blog/2026/warm-weight-averaged-reward/)), reward를 정책 업데이트로 바꾸는 PPO·GRPO·DPO([#14](/blog/2026/ppo/)~~[#18](/blog/2026/dpo/)), 검증 가능한 도메인에서 학습된 reward model 자체를 규칙으로 대체하는 RLVR([#19](/blog/2026/lets-verify-step-by-step/)~~[#21](/blog/2026/deepseek-r1/)), 학습된 RM을 생성형 judge로 재구성하는 흐름([#22](/blog/2026/prometheus-2/)~~[#26](/blog/2026/deepseek-grm-spct/)), 그리고 그 judge가 스스로 생각하고 그 신뢰 자체를 검증하는 최신 연구([#27](/blog/2026/reasongrm/)~[#31](/blog/2026/one-token-to-fool-judge/)). 하나하나는 특정 논문이 특정 문제 하나에 답한 결과였다.

그런데 실제로 프론티어급 모델을 학습시키는 팀은 이 부품 중 무엇을, 어떤 조합으로, 왜 골랐을까. 이 마지막 글은 시리즈 전체에서 유일하게 "논문 1편 = 포스트 1편" 형식을 깨고, 다섯 개 technical report — DeepSeek-R1, Qwen2.5, Llama 3, Kimi K2, Solar — 를 가로질러 reward 설계의 실전 선택지를 비교한다.

미리 결론의 윤곽을 말하면, 다섯 모델은 **검증 가능한 도메인(수학, 코드)에서는 놀랄 만큼 수렴**하고, **검증 불가능한 도메인(대화, 글쓰기, 안전성)에서는 뚜렷하게 갈라진다.** 이 글은 다음 순서로 그 수렴점과 분기점을 짚는다.

1. 다섯 모델 각각이 reward를 어디서 조달하고 어떤 알고리즘으로 정책을 업데이트했는가 (Method)
2. 검증 가능 도메인에서 왜 다들 비슷한 답에 도달했는가 (Experiments)
3. 검증 불가능 도메인에서 왜 갈라졌는가 — DPO, self-critique, generative judge 세 갈래 (Experiments)
4. reward hacking을 실무에서 어떻게 막고 있는가 (Experiments)
5. 지금 내가 reward를 설계해야 한다면 무엇을 골라야 하는가 (Conclusion)

# Background

시리즈가 지금까지 쌓아온 재료를 세 가지 축으로 요약하면 이렇다.

| 축                                       | 선택지                                                        | 관련 편                                                                                                                                                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| reward를 어디서 조달하는가               | 학습된 스칼라 RM / 규칙 기반 verifiable reward / judge·rubric | 2부([#4](/blog/2026/bradley-terry-rethinking/)~~[#9](/blog/2026/rewardbench-2/)), 5부([#19](/blog/2026/lets-verify-step-by-step/)~~[#21](/blog/2026/deepseek-r1/)), 6·7부([#22](/blog/2026/prometheus-2/)~[#31](/blog/2026/one-token-to-fool-judge/)) |
| 그 reward로 정책을 어떻게 업데이트하는가 | PPO / GRPO / DPO                                              | 4부([#14](/blog/2026/ppo/)~[#18](/blog/2026/dpo/))                                                                                                                                                                                                    |
| 도메인을 어떻게 가르는가                 | 검증 가능(정답 존재) / 검증 불가능(정답 부재)                 | [#21 DeepSeek-R1](/blog/2026/deepseek-r1/)                                                                                                                                                                                                            |

다섯 모델은 이 세 축 위에서 각자 다른 점을 찍었다.

| 모델        | 발표    | technical report                                     |
| ----------- | ------- | ---------------------------------------------------- |
| DeepSeek-R1 | 2025-01 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Qwen2.5     | 2024-12 | [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) |
| Llama 3     | 2024-07 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| Kimi K2     | 2025-07 | [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| Solar       | 2026-01 | [arXiv:2601.07022](https://arxiv.org/abs/2601.07022) |

# Method

## DeepSeek-R1: RLVR을 극한까지 밀어붙인다

[#21](/blog/2026/deepseek-r1/)에서 자세히 다뤘듯, DeepSeek-R1은 학습된 reward model을 검증 가능한 도메인에서 통째로 걷어내고 규칙 기반 검증기로 대체한다. reward는 두 종류뿐이다.

- **accuracy reward**: 수학은 `\boxed{}` 안의 최종 답을 정답과 문자열 비교, 코드는 유닛 테스트 통과 여부.
- **format reward**: 사고 과정을 `<think>` 태그 안에 쓰도록 강제.

이 두 reward는 [#16 GRPO](/blog/2026/grpo-deepseekmath/)에 그대로 꽂힌다. R1-Zero는 SFT 없이 이 조합만으로 RL을 돌렸고, R1은 여기에 cold-start SFT → reasoning RL → rejection sampling+SFT → all-scenario RL의 4단계를 더 쌓았다. 논문은 PRM과 MCTS를 명시적으로 시도했다가 접었다고 밝힌다 — 단계 정의가 모호하고, 중간 단계 정오 판정이 어렵고, 무엇보다 **PRM 자체가 신경망인 이상 다시 hacking 대상이 된다**는 이유다. KL penalty도 reward에 섞지 않고 손실 함수에 직접 더한다. 4단계(all-scenario RL)에 이르면 정답이 없는 일반 대화 영역에는 결국 helpfulness·harmlessness를 보는 **학습된 RM**을 다시 불러온다 — 규칙 기반 reward는 만능이 아니라는 사실을 R1 스스로 인정한 셈이다.

## Qwen2.5: offline DPO로 다지고 online GRPO로 마무리한다

Qwen2.5는 RL을 두 단계로 나눈다. 먼저 **offline DPO**로 정책을 선호 방향으로 크게 당겨놓고, 그다음 **online GRPO**로 세밀하게 다듬는다. 눈에 띄는 설계는 RM과 RL의 쿼리셋을 아예 통일했다는 점이다 — RM을 학습시킬 때 쓰는 쿼리 분포와 GRPO 단계에서 실제로 굴리는 쿼리 분포가 같다. 그리고 이 쿼리셋을 무작위로 뽑지 않고 **같은 쿼리에 대한 응답들의 점수 분산이 큰 것을 우선** 선별한다.

| 항목                | 값                                                 |
| ------------------- | -------------------------------------------------- |
| RL 단계 구성        | offline DPO → online GRPO                          |
| RM 학습 쿼리셋      | RL 쿼리셋과 동일, 분산이 큰(변별력 있는) 쿼리 우선 |
| 쿼리당 응답 샘플 수 | 8개                                                |
| global batch size   | 2048                                               |

분산이 큰 쿼리를 우선한다는 선택은 RM 학습에서 **정보량이 큰 데이터를 골라 쓴다**는 뜻이다. 모든 응답이 비슷비슷한 쿼리는 RM에게 "뭐가 더 나은가"를 가르쳐줄 신호가 거의 없다. 반대로 응답 품질이 들쭉날쭉한 쿼리는 RM이 좋고 나쁨을 구분하는 경계선을 뚜렷하게 배울 수 있다. [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)가 다뤘던 선호 데이터 노이즈 문제를, Qwen2.5는 애초에 노이즈가 적은(변별력 있는) 쿼리만 골라 쓰는 방식으로 우회한 셈이다.

## Llama 3: RL 없이 DPO 반복으로 후처리한다

Llama 3 herd 논문에서 가장 도드라지는 선택은 **PPO를 아예 쓰지 않는다**는 것이다. 대신 후처리를 **rejection sampling → SFT → DPO**의 반복 라운드로 구성한다. 매 라운드마다:

1. 현재 정책에서 여러 응답을 샘플링한다.
2. **학습된 RM**으로 채점해 좋은 응답만 골라 SFT 데이터로 쓴다(rejection sampling).
3. SFT로 정책을 갱신한다.
4. 사람 선호 쌍(chosen/rejected, 때로는 사람이 직접 수정한 edited 응답)으로 **DPO**를 돌려 다시 정렬한다.

RM은 여기서 정책을 직접 업데이트하는 데 쓰이지 않는다 — rejection sampling에서 "어떤 응답을 SFT 데이터로 채택할지" 거르는 필터로만 쓰인다. 실제 정책 업데이트는 DPO가 전담한다. 그리고 RM 학습과 DPO 양쪽 모두에서 **chosen이 rejected보다 확실히 나은 쌍만 쓰고, 우열이 애매한 쌍은 버린다.** [#18 DPO](/blog/2026/dpo/)가 다뤘던 "reward model 없이 선호 쌍에서 바로 정책을 학습한다"는 아이디어를, Llama 3는 온라인 RL 인프라 없이도 여러 라운드를 반복하는 것으로 프로덕션에 앉혔다.

## Kimi K2: 검증 가능하면 규칙, 아니면 스스로 채점한다

Kimi K2는 RLVR을 기본으로 깔되, 검증 불가능한 개방형 도메인까지 정렬을 확장하기 위해 **self-critique rubric reward**를 도입한다. 모델이 자기 자신의 출력을 clarity, factuality 같은 rubric으로 스스로 채점해 선호 신호를 만드는 방식이다. 원칙은 단순하다.

- 객관적으로 채점 가능하면(코드 유닛 테스트, 수학 정답 일치) **objective reward**를 우선 쓴다.
- 그렇지 않으면 모델 스스로의 self-critique로 보완한다.

self-critique가 아무 기반 없이 작동하는 건 아니다. **critic 능력 자체를 SFT 단계에서 미리 초기화**한다 — 오픈소스 선호 데이터와 인하우스 선호 데이터를 섞어 "무엇이 좋은 응답인가"에 대한 감각을 SFT로 먼저 심어놓고, 그 감각을 RL 단계에서 자기 채점에 재사용하는 구조다. 정적인 태스크(정답이 고정된 문제)에서 개방형 도메인으로 정렬 범위를 넓히는 과정에서 decaying temperature도 함께 쓴다 — 학습 초반엔 탐색을 넓게, 후반으로 갈수록 좁혀 수렴을 안정시키는 장치다.

## Solar: 도메인마다 다른 reward 함수를 병렬로 쓴다

Solar는 다섯 모델 중 가장 명시적으로 "도메인별 reward"를 설계 원칙으로 못박는다. technical report는 이렇게 적는다.

> "Different data types employ specialized reward functions: verifiable correctness for closed-ended STEM problems, multi-dimensional scoring for agent simulation, reward model-based evaluation for open-ended reasoning."

정리하면 세 갈래다 — 닫힌 STEM 문제는 **검증 가능한 정오 판정**, 에이전트 시뮬레이션은 **다차원 스코어링**, 개방형 추론은 **RM 기반 평가**. RL 알고리즘으로는 GRPO의 변형인 **GSPO(Group Sequence Policy Optimization)**를 쓰는데, 이유로 sparse MoE 아키텍처 학습에서의 안정성을 든다. 여기에 더해 응답 정렬 단계에서는 **iterative DPO**를 쓰고, KL divergence regularization을 DPO loss 안에 넣어 성능 저하를 막는다고 밝힌다. 사람 선호 데이터는 STEM 설명, 창작, 대화 품질을 포괄하며 **model-based reward estimation**을 함께 쓴다고 명시한다.

다만 RM 자체의 아키텍처, 학습 데이터 규모, 손실 함수 같은 세부는 report에 공개되어 있지 않다. **"reward model-based evaluation"이라는 표현 이상으로는 공개 정보로 확인되지 않는다** — Solar는 GSPO+DPO라는 알고리즘 조합과 도메인 분리 원칙까지는 투명하게 밝히지만, RM 자체의 구현 디테일은 다른 네 모델보다 불투명한 편이다.

# Experiments

## 마스터 비교표

| 모델        | reward 조달 방식                | RL 알고리즘                     | 별도 RM 학습                   | 검증 가능 도메인                    | 검증 불가능 도메인                           |
| ----------- | ------------------------------- | ------------------------------- | ------------------------------ | ----------------------------------- | -------------------------------------------- |
| DeepSeek-R1 | 규칙 기반(4단계 이후엔 RM 병행) | GRPO                            | O (all-scenario 단계)          | 규칙 검증기: 정답 일치·테스트 통과  | 학습된 RM (helpfulness·harmlessness)         |
| Qwen2.5     | 학습된 RM                       | offline DPO → online GRPO       | O (RL 쿼리셋과 동일)           | 명시적 RLVR 언급 없음, RM 기반 GRPO | RM 기반 online GRPO                          |
| Llama 3     | 학습된 RM + 사람 선호 쌍        | DPO만 (PPO 없음)                | O (rejection sampling용)       | 명시적 규칙 reward 없음             | DPO (확실한 우열 쌍만)                       |
| Kimi K2     | 규칙 기반 + self-critique       | on-policy RL (RLVR + rubric)    | 부분적 (critic을 SFT로 초기화) | 규칙 검증기: 코드 테스트·수학 정답  | self-critique rubric reward                  |
| Solar       | 도메인별 혼합                   | GSPO(GRPO 변형) + iterative DPO | O (아키텍처 비공개)            | 검증 가능한 정오 판정               | RM 기반 평가 + model-based reward estimation |

이 표 자체가 이 글의 결론을 압축한다. 오른쪽에서 두 번째 열(검증 가능 도메인)은 세 모델(DeepSeek, Kimi, Solar)이 규칙 기반 reward로 수렴한다. 맨 오른쪽 열(검증 불가능 도메인)은 다섯 모델이 다섯 갈래로 흩어진다.

## 공통 수렴점: 검증 가능 도메인은 규칙이 표준이 됐다

DeepSeek-R1, Kimi K2, Solar 셋은 수학·코드처럼 정답이 프로그램적으로 판정 가능한 도메인에서 **학습된 RM을 아예 배제하고 규칙 검증기를 쓴다.** [#21](/blog/2026/deepseek-r1/)에서 짚었듯 이유는 명확하다 — 규칙 검증기는 파라미터가 없는 함수이므로 hacking할 대상 자체가 없고, RM을 학습·재학습하는 비용도 들지 않는다. Qwen2.5와 Llama 3의 report에서는 이 시리즈가 확인한 범위 안에서 검증 가능 도메인 전용 규칙 reward를 명시적으로 강조하지 않는다 — 두 모델은 RM 기반 파이프라인(Qwen은 online GRPO, Llama는 rejection sampling+DPO)을 도메인 구분 없이 공통으로 쓰는 쪽에 가깝다. 즉 **RLVR이 업계 표준이 됐다고 단정하기엔 이르지만, 검증 가능한 도메인에서 규칙 기반 reward를 선택지로 명시하는 모델의 비중은 뚜렷이 늘었다** — DeepSeek-R1(2025-01) 이후 발표된 Kimi K2(2025-07), Solar(2026-01) 모두 이 선택지를 그대로 채택했다는 사실이 그 흐름을 보여준다.

## 갈라지는 지점: 검증 불가능 도메인을 메우는 세 갈래

정답이 없는 도메인(글쓰기, 상담, 안전성 판단)으로 가면 다섯 모델은 서로 다른 답을 낸다. 이 시리즈 6·7부가 연구 단계에서 탐구한 접근들이 실제로 어떻게 프로덕션에 쓰였는지 정리하면 이렇다.

| 접근                                              | 채택 모델                              | 방식                                                      | 관련 편                                                                                                                                                   |
| ------------------------------------------------- | -------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 사람 선호 쌍 + DPO                                | Llama 3                                | RM으로 거른 응답 + 사람 선호 쌍을 DPO로 직접 학습         | [#18 DPO](/blog/2026/dpo/)                                                                                                                                |
| self-critique rubric                              | Kimi K2                                | 모델이 스스로 clarity·factuality 등 rubric으로 채점       | [#25 Self-Taught Evaluators](/blog/2026/self-taught-evaluators/), [#28 J1](/blog/2026/j1-thinking-judge/)                                                 |
| 학습된 스칼라 RM                                  | DeepSeek-R1, Qwen2.5, Solar            | 사람 선호 데이터로 학습한 RM이 점수를 매김                | 2부([#4](/blog/2026/bradley-terry-rethinking/)~[#9](/blog/2026/rewardbench-2/))                                                                           |
| generative judge / rubric 조건부 평가 (연구 단계) | 다섯 모델 중 채택 사례는 확인되지 않음 | judge 모델이 근거를 생성하며 채점, rubric을 입력으로 받음 | [#22 Prometheus 2](/blog/2026/prometheus-2/), [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/), [#29 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/) |

마지막 행이 흥미롭다. [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/)과 [#29 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)가 제안한 "judge가 채점 근거를 스스로 생성하고 rubric을 조건으로 받는" 방식은, 이 글이 다루는 다섯 프론티어 모델의 공개 report에서는 명시적으로 채택됐다고 확인되지 않는다. 연구 논문(생성형 judge)과 프로덕션 report(스칼라 RM 또는 self-critique) 사이에 시차가 있다는 뜻이다. Kimi K2의 self-critique가 그나마 가장 가까운 프로덕션 사례고, 나머지는 여전히 전통적인 스칼라 RM에 의존한다.

## DPO vs RL 트레이드오프

[#18 DPO](/blog/2026/dpo/)는 reward model과 온라인 RL 루프를 통째로 없애고 선호 쌍에서 바로 정책을 학습하는 방법이었다. 다섯 모델의 선택을 이 스펙트럼 위에 놓으면 이렇게 갈린다.

| 위치                    | 모델                 | 특징                                                             |
| ----------------------- | -------------------- | ---------------------------------------------------------------- |
| DPO만                   | Llama 3              | 온라인 RL 인프라 없이 rejection sampling+SFT+DPO 반복만으로 정렬 |
| DPO → RL 순차 결합      | Qwen2.5              | offline DPO로 큰 방향을 잡고 online GRPO로 미세조정              |
| RL만(순수 온라인)       | DeepSeek-R1, Kimi K2 | GRPO 기반 온라인 RL이 정책 업데이트를 전담                       |
| RL + DPO 병행(도메인별) | Solar                | STEM/에이전트는 GSPO, 응답 정렬은 iterative DPO                  |

Llama 3가 DPO만으로 충분했다는 사실은 [#18](/blog/2026/dpo/)의 핵심 주장 — "reward model과 강화학습 루프 없이도 선호 쌍만으로 정렬이 된다" — 을 프로덕션 규모에서 검증한 사례다. 반대로 DeepSeek-R1과 Kimi K2가 순수 온라인 RL을 고수한 건, 검증 가능한 규칙 reward가 있을 때는 **매 스텝 새로 샘플링해서 채점하는 비용이 감당할 만하고, 온라인 신호가 오프라인 선호 쌍보다 정확하기 때문**이다. Qwen2.5와 Solar는 그 중간에서 "먼저 싸게 DPO로 다지고, 남은 격차를 비싼 온라인 RL로 메운다"는 절충을 택했다.

## reward hacking 방어: 각자의 방식

[#10](/blog/2026/reward-model-overoptimization/)~[#13](/blog/2026/warm-weight-averaged-reward/)이 다뤘던 reward hacking 방어 기법들이 실제로 어떻게 쓰이는지 정리하면 이렇다.

| 모델        | 방어 장치                                                                   | 원리                                                                                                                   |
| ----------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| DeepSeek-R1 | KL penalty를 reward가 아닌 손실에 직접 추가, format reward로 최단 경로 차단 | [#12 ODIN](/blog/2026/odin-disentangled-reward/)처럼 오염된 신호를 분리                                                |
| Qwen2.5     | RM 학습 쿼리셋을 분산 큰 쿼리로 한정                                        | 노이즈가 적은 데이터로 RM을 학습시켜 애초에 hacking 여지를 줄임                                                        |
| Llama 3     | RM·DPO 모두 우열이 확실한 쌍만 사용                                         | [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)의 선호 데이터 노이즈 문제를 데이터 큐레이션으로 우회 |
| Kimi K2     | decaying temperature                                                        | 학습 후반으로 갈수록 탐색을 좁혀 이상 패턴으로의 발산을 억제                                                           |
| Solar       | DPO loss 안에 KL divergence regularization 직접 포함                        | 참조 정책에서 과도하게 멀어지는 것을 막아 성능 저하 방지                                                               |

다섯 모델이 쓰는 방어 장치는 표현은 다르지만 원리는 하나로 수렴한다 — **reward 신호와 "정책이 참조점에서 얼마나 벗어났는가"를 분리해서 다룬다.** DeepSeek은 이 둘을 손실 함수 안에서 물리적으로 떼어놓고, Solar는 KL을 DPO loss에 명시적으로 넣고, Llama와 Qwen은 애초에 노이즈가 큰(따라서 hacking에 취약한) 데이터를 걸러낸다. [#11 Length Correlations](/blog/2026/rlhf-length-correlations/)이 지적했던 "성능 향상처럼 보이지만 실은 길이·문체 편향"이라는 함정을, 다섯 모델 모두 나름의 방식으로 피해 가려 한 흔적이다.

## 실무 결정 가이드

시리즈 전체를 한 장으로 착지시키면 이렇다. reward를 설계해야 하는 상황이라면, 아래 질문 순서를 따라가면 된다.

| 내 상황                                          | 권장 방식                          | 근거                                                      | 채택 사례                                                                                                                                      |
| ------------------------------------------------ | ---------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 정답을 프로그램적으로 판정할 수 있다(수학, 코드) | 규칙 기반 verifiable reward (RLVR) | 파라미터가 없어 hacking 불가능, RM 학습 비용 없음         | DeepSeek-R1, Kimi K2, Solar                                                                                                                    |
| 정답은 없지만 채점 기준(rubric)을 정의할 수 있다 | judge 또는 self-critique           | 스칼라 RM보다 해석 가능하고 다양한 기준을 반영            | Kimi K2 (self-critique), 연구 단계: [#22](/blog/2026/prometheus-2/) [#26](/blog/2026/deepseek-grm-spct/) [#29](/blog/2026/rubrics-as-rewards/) |
| 사람 선호 쌍은 있지만 온라인 RL 인프라가 없다    | DPO                                | reward model·롤아웃 루프 없이 선호 쌍에서 바로 학습       | Llama 3, Qwen2.5의 offline 단계                                                                                                                |
| 선호 데이터도 있고 온라인 RL 인프라도 있다       | RM 학습 + PPO/GRPO(GSPO)           | 온라인 신호가 오프라인 선호 쌍보다 정확, 세밀한 튜닝 가능 | Qwen2.5의 online 단계, Solar                                                                                                                   |

실무에서는 이 네 줄이 배타적이지 않다 — Qwen2.5와 Solar처럼 **첫째 줄(RLVR)과 넷째 줄(RM+RL)을 도메인별로 나눠 병행**하는 것이 다섯 모델이 보여준 가장 흔한 패턴이다.

# Conclusion

31편에 걸쳐 쌓아온 재료를 다섯 개 프론티어 모델에 겹쳐보면 이렇게 정리된다.

1. **검증 가능한 도메인은 규칙으로 수렴한다.** DeepSeek-R1, Kimi K2, Solar 셋 모두 수학·코드에서 학습된 RM을 규칙 검증기로 대체했다. [#21](/blog/2026/deepseek-r1/)의 RLVR이 하나의 논문에서 그친 아이디어가 아니라 반복해서 채택되는 선택지가 됐다는 뜻이다.
2. **검증 불가능한 도메인은 여전히 갈라진다.** Llama 3는 DPO, Kimi K2는 self-critique rubric, DeepSeek·Qwen·Solar는 학습된 스칼라 RM을 쓴다. [#26](/blog/2026/deepseek-grm-spct/)·[#29](/blog/2026/rubrics-as-rewards/)가 제안한 생성형 judge는 연구 단계에서 갈 길을 가리키고 있지만, 다섯 모델의 공개 report에는 아직 명시적으로 등장하지 않는다.
3. **DPO와 온라인 RL은 대체재가 아니라 조합 가능한 도구다.** Llama 3는 DPO만으로, DeepSeek·Kimi는 순수 온라인 RL로, Qwen·Solar는 둘을 순차·병행으로 썼다. [#18](/blog/2026/dpo/)이 제기한 "reward model 없이도 되는가"라는 질문에, 실전은 "상황에 따라 둘 다"라고 답한다.
4. **reward hacking 방어는 형태가 다양해도 원리는 하나다.** reward 신호와 참조점 이탈 신호를 분리해서 다룬다는 것 — DeepSeek의 손실 내 KL, Solar의 DPO 내 KL, Qwen·Llama의 데이터 큐레이션 모두 [#10](/blog/2026/reward-model-overoptimization/)~[#12](/blog/2026/odin-disentangled-reward/)이 정량화한 문제에 대한 실무적 응답이다.

이 시리즈를 여는 [#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)은 "사람의 선호로 보상 함수를 배울 수 있는가"라는 질문 하나로 시작했다. 32편을 지나 도착한 답은, 그 질문이 하나의 답을 갖지 않는다는 것이다. 정답이 있으면 규칙이 reward가 되고, 정답이 없지만 기준은 있으면 judge가 reward가 되고, 기준조차 흐릿하면 사람의 선호 쌍이 reward를 대신한다. **reward 설계는 하나의 정답을 찾는 문제가 아니라, 내가 가진 도메인이 스펙트럼의 어디에 있는지를 정확히 진단하는 문제였다.**

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 서른두 번째 글이다.

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
26. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

**8부. 실전 종합**

32. **(현재 글)** 프론티어 모델의 reward 설계 (2024~2026) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것

본 시리즈는 32편으로 구성된다.

# 참고 문헌

- DeepSeek-AI, 2025. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).
- Shao et al. (DeepSeek-AI), 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — GRPO 원 논문.
- Qwen Team, 2024. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).
- Llama Team, AI @ Meta, 2024. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
- Kimi Team, 2025. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534).
- Upstage AI, 2026. [Solar Open Technical Report](https://arxiv.org/abs/2601.07022).
- Rafailov et al., 2023. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — [#18](/blog/2026/dpo/)에서 다룬 DPO 원 논문.
- Liu et al., 2024. [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535) — [#22](/blog/2026/prometheus-2/) 참고.
- Liu et al. (DeepSeek-AI), 2025. [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/) 참고.
- Viswanathan et al., 2025. Rubrics as Rewards — [#29](/blog/2026/rubrics-as-rewards/) 참고.
