---
layout: post
title: "CriticEval: judge를 채점하는 벤치마크"
date: 2026-08-11 09:25:30 +0900
description: "RLHF Reward 설계 시리즈 #30 — 비평 능력을 4개 차원으로 쪼개 평가하다"
categories: [paper]
tags: [rlhf, reward-model, llm-as-a-judge, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
---

> [CriticEval: Evaluating Large Language Model as Critic](https://arxiv.org/abs/2402.13764) (Tian Lan et al., Beijing Institute of Technology, NeurIPS 2024)

# Introduction

[#9 RewardBench 2](/blog/2026/rewardbench-2/)는 **스칼라 RM**을 어떻게 평가할지를 다뤘다. 응답 두 개를 주고 RM이 매긴 점수의 순서가 사람의 선호와 맞는지, 그것 하나로 RM의 좋고 나쁨을 가른다. 그런데 7부에서 다룬 [#27 ReasonGRM](/blog/2026/reasongrm/), [#28 J1](/blog/2026/j1-thinking-judge/)은 RM이 아니라 **생성형 judge(critic)**를 다룬다. 점수 하나를 뱉는 대신 "왜 이 응답이 나쁜지"를 문장으로 설명하고, 그 설명을 바탕으로 판정을 내린다. 이 judge는 무엇으로 평가하나?

이 질문에 대한 두 논문의 답은 같은 모양이다. "judge가 더 잘 생각하게(reasoning) 만들었다"는 것. J1은 RL로 judge에게 사고 과정을 부여했고, ReasonGRM은 reasoning 능력을 judge에 이식했다. 그런데 **그 주장을 무엇으로 검증했나**? 대개는 "최종 판정이 사람 선호와 얼마나 일치하는가"라는 단일 정답률이다. 이건 [#20 Math-Shepherd](/blog/2026/math-shepherd/)에서 이미 본 함정과 정확히 같은 모양이다. Math-Shepherd 편에서는 "최종 답이 맞다고 풀이 과정 전체가 맞는 건 아니다"라는 문제를 다뤘다. Judge 평가에도 똑같은 구조가 있다. **판정(레이블)은 맞는데 근거(reasoning)는 틀린 비평**을, 정답률 하나로는 걸러낼 수 없다.

CriticEval은 이 구멍을 정면으로 다룬다. "critic이 좋은 critic인가"를 스칼라 정답률 하나로 뭉개지 않고, **비평 능력을 4개 차원(feedback, comparison, correction, meta-feedback)으로 쪼개고, 그중 텍스트로 된 비평은 사람이 쓴 참조 비평과 GPT-4로 다시 채점**한다. 결과는 두 가지를 보여준다. 하나는 오픈소스 모델이 생각보다 GPT-4에 가깝다는 것(20B짜리 InternLM2가 GPT-3.5-turbo를 이긴다). 다른 하나는 그 GPT-4조차 비평의 비평(meta-feedback)에서는 사람들끼리의 합의 수준에 못 미친다는 것이다. 그리고 이 벤치마크 자체가 안고 있는 순환 — judge를 채점하려면 결국 또 다른 judge가 필요하다는 문제 — 는 [#31 One Token to Fool](/blog/2026/one-token-to-fool-judge/)로 이어진다.

비유하자면 이렇다. 지금까지 시리즈가 다룬 것은 "시험을 잘 보는 모델"을 만드는 이야기였다면, CriticEval은 "채점을 잘하는 사람을 뽑는 시험"을 설계하는 이야기다. 채점자 선발 시험에서 지원자가 정답과 같은 점수를 매겼다는 사실만으로는 부족하다 — 왜 그 점수를 줬는지, 그 근거가 논리적으로 타당한지까지 봐야 진짜 채점자를 뽑을 수 있다. CriticEval은 바로 이 "채점자 선발 시험"의 설계도다.

# Background

## LLM-as-a-judge의 두 형태를 다시 정리하면

이 시리즈에서 지금까지 본 judge는 크게 두 갈래다.

- **스칼라 RM**: 응답을 받아 숫자 하나를 뱉는다. [#4 Bradley-Terry](/blog/2026/bradley-terry-rethinking/)부터 [#7 ArmoRM](/blog/2026/armorm/)까지가 이쪽이다. 평가는 RewardBench류 벤치마크로 한다([#9](/blog/2026/rewardbench-2/)).
- **생성형 critic**: 응답을 받아 텍스트로 비평을 쓰고, 그 비평을 바탕으로 점수·선호·수정을 만든다. [#22 Prometheus 2](/blog/2026/prometheus-2/), [#24 Generative Reward Models](/blog/2026/generative-reward-models/), [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/), 그리고 [#27 ReasonGRM](/blog/2026/reasongrm/), [#28 J1](/blog/2026/j1-thinking-judge/)이 이쪽이다.

스칼라 RM은 "점수의 순서"만 맞으면 평가가 끝난다. 그런데 생성형 critic은 산출물이 텍스트다. 텍스트에는 순서만으로 잡히지 않는 정보 — 근거가 맞는지, 지적이 구체적인지, 제안이 실행 가능한지 — 가 담겨 있다. CriticEval은 이 지점에서 "비평 능력(critique ability)"을 하나의 스킬이 아니라 **네 가지 서로 다른 스킬의 묶음**으로 정의한다.

## 토이 예제: 같은 판정, 다른 근거

왜 스칼라 정답률 하나로는 부족한지 작은 예로 짚어보자. 다음 응답을 GPT-4에게 채점하게 한다고 하자.

> **질문**: "2024년 파리 올림픽은 몇 번째 하계 올림픽인가?"
> **응답**: "2024년 파리 올림픽은 34번째 하계 올림픽이다." (정답은 33번째 — 사실 오류)

두 critic이 이 응답에 똑같이 "품질 점수 2/10, 이 응답을 reject"라는 **판정**을 내린다고 하자.

- **Critic A (근거 오류)**: "문장이 너무 짧고 부연 설명이 없어서 응답의 품질이 낮다. 2점을 준다."
- **Critic B (근거 정확)**: "34번째라는 숫자가 틀렸다. 1896년 아테네부터 세어보면 2024년 파리는 33번째 하계 올림픽이다. 이 사실 오류 때문에 2점을 준다."

두 critic 모두 최종 점수는 2/10, 최종 판정은 reject로 동일하다. **판정 레이블만 참조 정답과 비교하는 objective 지표(예: 점수의 Spearman 상관)로는 A와 B를 구분할 수 없다.** 둘 다 "정답과 같은 점수를 냈다"는 사실만 기록되기 때문이다. 그러나 실제로 쓸모 있는 비평은 B다. A는 사용자에게 "왜 틀렸는지"를 알려주지 못하고, 이 critic을 RL의 reward로 쓰면 모델은 "짧은 응답을 피하라"는 엉뚱한 신호를 학습하게 된다 — 실제 문제였던 사실 오류는 고쳐지지 않는다.

이게 CriticEval이 **텍스트 자체를 GPT-4로 다시 채점하는 subjective 평가**를 objective 평가와 나란히 두는 이유다. 판정이 맞았다는 사실과, 그 판정에 이르는 근거가 맞았다는 사실은 서로 다른 질문이고, 하나의 숫자로는 절대 합쳐지지 않는다.

일상 비유로 옮기면, 이건 객관식 답안지 채점과 서술형 답안지 채점의 차이다. 객관식은 정답 번호만 대조하면 끝나지만, 서술형은 정답과 같은 결론에 도달했더라도 풀이 과정이 억지스러우면 감점해야 한다. Critic A와 B는 "답안지"(최종 점수)는 똑같이 맞혔지만, "풀이 과정"(비평의 근거)에서 갈린다. 정답률만 보는 채점은 이 둘을 같은 답안으로 취급하는 셈이다.

# Method

## 벤치마크가 다루는 4개 차원

<p align="center"><img src="/assets/post/image/criticeval/fig1_four_dimensions.png" width="85%"></p>

논문은 위 그림(피자 레시피 요청 예시)으로 4개 차원을 한 번에 보여준다. 같은 입력("피자 레시피를 알려줘")과 두 응답(A: "레시피를 모른다"는 회피성 응답, B: 실제 레시피)을 놓고 각 차원이 무엇을 산출하는지 정리하면 다음과 같다.

| 차원              | 산출물                                              | 무엇을 잡아내나                              | 없으면 놓치는 실패                                                   |
| ----------------- | --------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| **Feedback**      | 텍스트 분석 + 품질 점수                             | 응답 하나의 결함을 지적하고 수정 방향을 제안 | "왜 나쁜지"를 말 못 하는 critic — 점수만 있고 근거가 없는 경우       |
| **Comparison**    | 텍스트 비교 + 선호 레이블(A/B/AB)                   | 응답 두 개를 동시에 놓고 우열을 가리는 능력  | 개별 평가에서는 그럴듯해도 상대 비교에서 뒤집히는 판단               |
| **Correction**    | 실제로 고친 응답                                    | feedback을 실행 가능한 수정으로 옮기는 능력  | 지적은 맞지만 정작 고치라고 하면 못 고치는 critic                    |
| **Meta-feedback** | feedback에 대한 점수 + 텍스트 (feedback의 feedback) | 비평 그 자체의 품질을 평가하는 능력          | critic을 감독하는 critic이 없으면, judge의 실수를 아무도 잡지 못한다 |

Feedback과 comparison은 이전 연구([#22 Prometheus 2](/blog/2026/prometheus-2/), [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/) 등)도 다뤘지만, **correction과 meta-feedback은 CriticEval 이전에는 거의 다뤄지지 않았다**고 논문은 지적한다. Correction이 빠지면 "지적은 잘하는데 실제로 고치는 능력은 형편없는" critic을 걸러낼 방법이 없고, meta-feedback이 빠지면 critic의 판정을 다시 검증할 장치가 아예 없다.

이 네 차원을 요리 서바이벌 프로그램의 심사위원에 비유하면 이해가 빠르다. 심사위원은 (1) 한 접시를 맛보고 무엇이 부족한지 코멘트하고(feedback), (2) 참가자 두 명의 접시를 나란히 놓고 어느 쪽이 더 나은지 가르고(comparison), (3) 때로는 직접 팬을 잡고 그 요리를 고쳐보라는 요구를 받으며(correction), (4) 방송이 끝난 뒤 수석 심사위원이 "그 코멘트가 실제로 타당했는가"를 다시 검토한다(meta-feedback). 넷 중 하나만 잘해서는 좋은 심사위원이라 부를 수 없다 — 맛은 잘 보는데 정작 요리를 못 고치거나, 코멘트는 그럴듯한데 근거가 부실한 심사위원은 현실에도 존재한다.

## 데이터 구성: 9개 태스크 × 4단계 응답 품질 × 4개 차원

<p align="center"><img src="/assets/post/image/criticeval/fig2_construction_pipeline.png" width="90%"></p>

데이터 구성은 3단계다.

**Step 1 — 태스크 준비**: 아래 9개 태스크 유형에서 총 99개의 구체적 태스크(task input)를 수집한다.

| 유형        | 태스크                                             |
| ----------- | -------------------------------------------------- |
| 언어 생성   | Summary, Translation, General Chat                 |
| 지식/정합성 | Question Answering, Harmlessness                   |
| 수리 추론   | Math (Chain-of-Thought), Math (Program-of-Thought) |
| 코드        | Code (실행 결과 포함), Code (실행 결과 없음)       |

**Step 2 — 응답 생성과 품질 단계화**: 여러 LLM으로 다양한 품질의 응답을 만든 뒤, 사람이 직접 확인해 **낮음(low) / 중간(medium) / 높음(high) / 정답(correct)** 4단계로 나눈다. 이렇게 응답 품질을 인위적으로 넓게 벌려놓는 이유는 단순하다 — critic이 "명백히 나쁜 응답"만 잘 잡고 "미묘하게 나쁜 응답"은 못 잡는지, 품질 구간별로 나눠서 봐야 그 경계가 드러나기 때문이다.

**Step 3 — 비평 생성과 사람 검수**: 강한 LLM(주로 GPT-4)이 1차 비평을 생성하면, 사람 전문가가 이를 검수·수정해 참조(reference) 비평으로 확정한다. 이 과정에서 **텍스트 비평 3,608건**이 사람 손을 거쳐 최종 확정됐다. 흥미로운 수치는 이 GPT-4 초안이 사람에게 수정당한 비율이다 — feedback 25.22%, comparison 34.83%, **correction 48.37%**. GPT-4조차 응답을 직접 고쳐 쓰는 correction에서 초안의 절반 가까이가 사람 손을 다시 거쳤다는 뜻이고, 이는 앞서 4개 차원 표에서 정리한 "correction이 가장 어려운 차원"이라는 직관과 정확히 들어맞는다.

## 평가는 다시 objective와 subjective로 나뉜다

토이 예제에서 본 문제 — 판정은 맞는데 근거가 다를 수 있다는 문제 — 를 CriticEval은 두 갈래 지표로 다룬다.

$$
s_{\text{obj}} = \begin{cases} \text{Spearman}(r, \hat r) & \text{feedback, meta-feedback (점수 상관)} \\ \frac{1}{N}\sum_{i=1}^N \mathbb{1}[L_i = \hat L_i] & \text{comparison (선호 레이블 일치율)} \\ N_{\text{pass}} / N & \text{correction (math/code, 실행 결과 통과율)} \end{cases}
$$

- $$r$$, $$\hat r$$: critic이 매긴 점수와 참조 점수. **Spearman 상관**은 두 순위가 얼마나 비슷하게 움직이는지만 보므로, 앞의 토이 예제에서 Critic A와 B가 똑같은 점수를 냈다면 이 지표로는 둘을 가를 수 없다.
- $$L_i$$, $$\hat L_i$$: comparison에서 critic이 고른 선호(A/B/무승부)와 참조 선호. 프롬프트에서 응답 순서(A, B 자리)를 바꿔가며 여러 번 물어 **일관성**까지 함께 본다.
- $$N_{\text{pass}}$$: correction이 고친 응답이 실제로 실행/채점을 통과한 개수. 수학·코드처럼 정답 검증이 가능한 태스크에서만 쓴다.

$$
s_{\text{subj}} = \text{GPT-4-turbo}\big(\text{critique}, \; \text{reference critique} \to \text{score} \in [1, 10]\big)
$$

- GPT-4-turbo가 **사람이 쓴 참조 비평을 8점짜리 기준점(anchor)** 으로 주고, 평가 대상 비평이 그보다 얼마나 낫거나 못한지 채점한다. 참조가 없으면 "10점 만점에 몇 점"이라는 절대 기준이 흔들리기 때문에, 참조를 기준점으로 고정한다. 이건 논술 채점자에게 모범답안을 쥐어주는 것과 같다 — 모범답안이 없으면 채점자마다 기준이 들쭉날쭉해지고, 있으면 그 기준에 맞춰 여러 채점자가 비슷한 점수를 낸다.
- 논문은 이 앵커링의 효과를 직접 검증했다. 프롬프트에서 참조 비평을 빼고 채점하게 하면 **평균 13.36점(100점 환산 기준)이 떨어진다.** 참조가 없으면 GPT-4 자신도 일관된 채점을 못 한다는 뜻이다.

## 왜 objective 지표만으로는 A와 B가 안 갈리는지, 숫자로 확인

Background의 올림픽 예제를 세 개 응답짜리 feedback 태스크로 확장해서 $$s_{\text{obj}}$$를 직접 계산해보자. 사람 참조 점수가 $$r = [2, 6, 9]$$인 세 응답에 대해, Critic A(근거 부실)와 Critic B(근거 정확)가 우연히 똑같은 점수 $$\hat r_A = \hat r_B = [2, 6, 9]$$를 매겼다고 하자 — 실제로 CriticEval의 표본 다수가 이런 경우다. 좋은 critic이든 나쁜 critic이든, 눈에 띄게 나쁜 응답에는 낮은 점수를, 눈에 띄게 좋은 응답에는 높은 점수를 주는 경향은 공유하기 때문이다.

| 응답 | 참조 점수 $$r_i$$ | 참조 순위 | Critic A 점수 | Critic A 순위 | Critic B 점수 | Critic B 순위 |
| ---- | ----------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| 1    | 2                 | 1         | 2             | 1             | 2             | 1             |
| 2    | 6                 | 2         | 6             | 2             | 6             | 2             |
| 3    | 9                 | 3         | 9             | 3             | 9             | 3             |

Spearman 상관은 $$\rho = 1 - \dfrac{6\sum_i d_i^2}{n(n^2-1)}$$이고, 여기서 $$d_i$$는 두 순위의 차이, $$n$$은 표본 수다. 두 critic 모두 순위가 참조와 완전히 같으므로 $$d_i = 0$$이 세 번 반복돼 $$\sum d_i^2 = 0$$이고, $$\rho_A = \rho_B = 1 - 0 = 1$$이다. 100점으로 환산해도 $$s_{\text{obj}}(A) = s_{\text{obj}}(B) = 100$$으로 **완전히 동일**하다 — objective 지표만 보면 두 critic은 구별 불가능한, 똑같이 완벽한 critic이다.

이제 $$s_{\text{subj}}$$를 계산해보자. GPT-4-turbo가 각 critic이 쓴 비평 텍스트를, 사람이 쓴 참조 비평(8점 기준점)과 대조하며 읽는다. 응답 1(가장 낮은 참조 점수)에 대해 Critic A는 "문장이 짧아서 감점"이라 썼고, Critic B는 참조 비평과 마찬가지로 구체적 사실 오류를 짚었다. 세 응답 전체에서 이 패턴이 반복된다면, 논문이 보고한 실제 격차 — 참조와 벌어진 근거를 댄 비평은 평균 3점대, 참조와 근접한 근거를 댄 비평은 8점대 — 를 그대로 대입해 $$s_{\text{subj}}(A) \approx 3$$, $$s_{\text{subj}}(B) \approx 8$$을 얻는다. **같은 $$s_{\text{obj}} = 100$$ 뒤에 전혀 다른 $$s_{\text{subj}}$$가 숨어 있었던 것이다.** 두 지표를 나란히 봐야만 "판정은 맞지만 속은 빈 critic"이 드러난다.

# Experiments

## 35개 모델을 critic으로 세우다

CriticEval은 닫힌소스(GPT-4-turbo, GPT-3.5-turbo, Claude-instant-1), 오픈소스 instruction-tuned 계열(Qwen 시리즈, InternLM2, Mistral/Mixtral, Llama-2), 그리고 critique 데이터로 별도 파인튜닝된 모델(Auto-J-13B, UltraCM-13B, TigerScore-13B)과 reward model(UltraRM-13B, Ziya-7B)까지 합쳐 **총 35개 모델**을 critic으로 세워 비교한다.

| 모델             | 파라미터 규모 | 종류     | 종합 subjective 점수 |
| ---------------- | ------------- | -------- | -------------------- |
| GPT-4-turbo      | 비공개        | 닫힌소스 | 7.81                 |
| Claude-instant-1 | 비공개        | 닫힌소스 | 6.45                 |
| InternLM2-20B    | 20B           | 오픈소스 | 6.20                 |
| Qwen-72B-Chat    | 72B           | 오픈소스 | 6.01                 |
| GPT-3.5-turbo    | 비공개        | 닫힌소스 | 5.89                 |

가장 눈에 띄는 줄은 InternLM2-20B(6.20)가 GPT-3.5-turbo(5.89)를 앞선다는 점이다. 파라미터 수로는 비교조차 안 되는 20B 오픈소스 모델이, RLHF로 다듬어진 닫힌소스 상용 모델을 비평 능력에서 이긴다. 이건 "critique 능력은 모델 크기보다 어떤 데이터로, 어떻게 다듬었는가에 더 좌우된다"는 방향을 가리키며, [#6 Skywork-Reward](/blog/2026/skywork-reward/)에서 본 "데이터 큐레이션이 아키텍처를 이긴다"는 관찰과 같은 결의 이야기다. 실제로 critique 데이터로 파인튜닝한 Auto-J-13B, UltraCM-13B, TigerScore-13B 같은 13B급 모델들이 파인튜닝 없는 Llama-2-70B-Chat을 능가한다 — 사이즈가 아니라 **critique-specific 학습 데이터의 유무**가 갈랐다.

## GPT-4도 meta-feedback에서는 사람 수준에 못 미친다

가장 어려운 차원인 meta-feedback(비평의 비평)에서는 격차가 다른 그림을 그린다.

| 채점자                  | meta-feedback 객관 지표(점수 상관, 100점 환산) |
| ----------------------- | ---------------------------------------------- |
| GPT-4-turbo             | 62.90                                          |
| 사람(annotator 간 평균) | 79.03                                          |

사람 주석자끼리 meta-feedback을 매길 때의 평균 일치도가 79.03인데, GPT-4-turbo는 62.90에 그친다. 이 79.03이라는 숫자는 곧 논문이 보고하는 **annotator 간 상관계수(약 0.79)** 를 100점 척도로 환산한 값과 같다 — 즉 "비평을 비평하는" 메타 수준의 판단은 사람들끼리도 완벽히 일치하지는 않지만(1.0이 아니라 0.79), 그 사람 수준의 합의조차 GPT-4는 아직 따라가지 못한다는 뜻이다. Comparison과 meta-feedback이 feedback보다 전반적으로 어렵다는 결과와 겹쳐 보면, **"응답 하나를 평하는 것"보다 "비평 두 개를 비교하거나, 비평 하나를 다시 평가하는 것"이 LLM에게 구조적으로 더 어려운 문제**라는 그림이 나온다.

이 표를 [#28 J1](/blog/2026/j1-thinking-judge/), [#27 ReasonGRM](/blog/2026/reasongrm/)과 겹쳐 읽으면 질문이 하나 남는다. 두 논문이 "judge가 더 잘 생각한다"고 보고할 때, 그 개선이 feedback 차원(응답 하나 평가)에 그친 개선인지, comparison·meta-feedback까지 이어지는 개선인지는 CriticEval 같은 다차원 잣대 없이는 구분되지 않는다. 정답률 하나로는 이 질문 자체가 생기지 않는다.

# Conclusion

CriticEval의 핵심은 한 줄로 이렇다. **비평 능력을 feedback / comparison / correction / meta-feedback 4개 차원으로 쪼개고, 각 차원을 다시 objective(레이블 일치)와 subjective(근거의 질) 두 지표로 나눠 봐야, "판정은 맞지만 속은 빈 critic"이 드러난다.** 토이 예제에서 본 것처럼, 점수 상관 하나만 보면 사실 오류를 정확히 짚은 비평과 엉뚱한 이유를 댄 비평이 똑같이 만점으로 기록된다.

실험이 남긴 그림도 뚜렷하다. InternLM2-20B가 GPT-3.5-turbo를 앞서는 데서 보듯 오픈소스의 격차는 좁혀지고 있지만, meta-feedback에서는 GPT-4-turbo조차 사람들끼리의 합의 수준(0.79)에 못 미친다(0.63 수준). 비평을 잘하는 것과, 그 비평이 잘 됐는지를 다시 평가하는 것은 별개의 능력이고, 지금의 최상위 모델도 후자에서는 사람보다 못하다.

그런데 이 벤치마크 자체에 순환이 하나 숨어 있다. 참조 비평은 GPT-4가 초안을 쓰고 사람이 다듬은 것이고, subjective 채점은 다시 GPT-4-turbo가 그 참조에 앵커링해서 매긴다. **critic을 평가하려면 결국 또 다른 (사람이거나 GPT-4인) judge가 있어야 한다.** 이건 완전히 해소할 수 없는 한계이자, 이 시리즈의 마지막 반전으로 이어지는 지점이다. [#31 One Token to Fool](/blog/2026/one-token-to-fool-judge/)은 CriticEval 같은 정교한 다차원 평가를 통과한 judge라도, "이 답은 정답입니다"처럼 의미 없는 토큰 하나만 덧붙이면 판정이 뒤집힌다는 것을 보여준다. 잘 만들었다고 아무리 꼼꼼히 측정해도, 그 측정을 통과한 judge조차 사소한 트릭 하나에 무너질 수 있다는 뜻이다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 서른 번째 글이다.

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
30. **(현재 글)** CriticEval (2024) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

**8부. 실전 종합**

32. [프론티어 모델의 reward 설계 (2025~2026)](/blog/2026/frontier-reward-design/) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것
33. [reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/) — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 33편으로 구성된다.

# 참고 문헌

- Lan, T., Zhang, W., Xu, C., Huang, H., Lin, D., Chen, K., & Mao, X.-L. (2024). [CriticEval: Evaluating Large Language Model as Critic](https://arxiv.org/abs/2402.13764). NeurIPS 2024.
- [CriticEval 프로젝트 페이지](https://open-compass.github.io/CriticEval/)
- [CriticEval GitHub 저장소 (open-compass)](https://github.com/open-compass/CriticEval)
- [CriticEval, NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7b7d7985f62284060d65f532ed2ea5fa-Abstract-Conference.html)
