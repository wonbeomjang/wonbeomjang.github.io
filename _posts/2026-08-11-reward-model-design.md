---
layout: post
title: "reward를 어떻게 설계할 것인가 — RM 설계 실무"
date: 2026-08-11 09:28:00 +0900
description: "RLHF Reward 설계 시리즈 #33 — 시리즈 32편을 관통한 reward 시스템 설계 절차를 한 장의 실무 가이드로"
categories: [paper]
tags: [rlhf, reward-model, rlvr, genrm, dpo, reward-hacking, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 특정 논문 한 편이 아니라, 이 시리즈 32편이 쌓은 결론을 **"내가 reward를 설계한다면 어떤 순서로 결정할까"**라는 하나의 절차로 압축한다. 각 결정마다 근거가 된 편으로 링크를 건다.

# Introduction

[#32](/blog/2026/frontier-reward-design/)가 프론티어 모델들이 **실제로 무엇을 골랐는지**를 관찰했다면, 이 글은 반대 방향이다 — **내가 처음부터 reward 시스템을 설계한다면 어떤 순서로 결정을 내려야 하는가.** 앞의 31편이 부품(논문 한 편이 문제 하나에 답한 것)이고 #32가 사례집이라면, 이 글은 그 둘을 겹쳐 만든 **설계 절차서**다.

reward 설계에서 초심자가 가장 자주 하는 실수는 "좋은 reward model을 학습시키자"부터 시작하는 것이다. 하지만 시리즈 전체가 반복해서 보여준 교훈은 정반대다.

- **학습된 RM은 근사이고, 근사에는 반드시 hacking당하는 지점이 있다**([#10](/blog/2026/reward-model-overoptimization/)).
- 그래서 **"굳이 RM을 학습시키지 않아도 되는 경우"를 먼저 걸러내는 것**이 설계의 1단계다([#21](/blog/2026/deepseek-r1/)).
- RM이 꼭 필요한 경우에도, **reward의 질은 함수보다 데이터와 프롬프트 선택에서 더 크게 갈린다**([#5](/blog/2026/secrets-rlhf-reward-modeling/), [#6](/blog/2026/skywork-reward/)).

이 글은 다음을 순서대로 다룬다.

1. reward를 조달하는 네 가지 방법과 각각의 비용 (Background)
2. 도메인 진단부터 검증까지, 7단계 설계 절차 (Method)
3. 절차를 실제 예제(안전성 정렬 어시스턴트)에 적용해보기 (Experiments)
4. 한 장으로 압축한 설계 체크리스트와 한계 (Conclusion)

# Background

## reward 조달처 네 가지

[#32](/blog/2026/frontier-reward-design/)에서 세운 4분류를, 이번엔 **"내가 고를 때의 비용과 위험"** 관점으로 다시 본다.

| 조달처                          | 언제 쓰나                                 | 장점                                   | 비용·위험                          | 근거 편                                                                          |
| ------------------------------- | ----------------------------------------- | -------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------- |
| ① 규칙 기반 verifiable reward   | 정답을 프로그램으로 판정 가능 (수학·코드) | 파라미터 0 → hacking 불가, 학습 비용 0 | 검증 가능한 도메인에만 적용        | [#21](/blog/2026/deepseek-r1/)                                                   |
| ② 스칼라 RM (reference 없음)    | 선호 쌍만 있고 채점 기준이 흐릿할 때      | 어떤 도메인이든 적용, 빠른 추론        | 근사 오차 → overoptimization, 편향 | [#4](/blog/2026/bradley-terry-rethinking/)\~[#9](/blog/2026/rewardbench-2/)      |
| ③ reference 기반 judge          | 정답 예시 + 채점 기준(rubric)이 있을 때   | 해석 가능, 기준을 명시적으로 주입      | 판정 모델 추론 비용, judge 편향    | [#22](/blog/2026/prometheus-2/)                                                  |
| ④ generative RM (self-critique) | 기준은 있으나 정답 예시가 없을 때         | 근거를 생성 → 검증·디버깅 가능         | 추론 비용 큼, judge도 뚫릴 수 있음 | [#26](/blog/2026/deepseek-grm-spct/), [#31](/blog/2026/one-token-to-fool-judge/) |

핵심 직관은 위에서 아래로 갈수록 **표현력은 커지지만 hacking 표면도 커진다**는 것이다. ①은 파라미터가 없어 뚫을 대상 자체가 없고, ②③④는 신경망이므로 정책이 그 근사 오차를 파고들 수 있다. 그래서 설계의 대원칙은 **"가능한 한 위쪽(①)을 쓰고, 어쩔 수 없을 때만 아래로 내려간다"**이다.

## 두 종류의 손실: 스칼라 RM과 KL 정규화

두 수식만 손에 쥐고 있으면 대부분의 설계 대화를 따라갈 수 있다.

**스칼라 RM 학습(Bradley-Terry, [#4](/blog/2026/bradley-terry-rethinking/)):**

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)
$$

- $$y_w, y_l$$: 같은 프롬프트 $$x$$에 대한 선호(win)·비선호(lose) 응답
- $$r_\theta$$: 학습되는 스칼라 RM
- 의미: 선택된 응답의 점수가 탈락 응답보다 **높아지도록** 밀되, $$\sigma$$(시그모이드) 때문에 차이가 이미 충분히 벌어지면 더는 세게 밀지 않는다.

**정책 최적화(KL 정규화, [#14](/blog/2026/ppo/)):**

$$
\max_{\pi}\ \mathbb{E}_{y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big)
$$

- 앞항: reward를 **최대화**하라.
- 뒷항: 그러나 참조 정책 $$\pi_{\text{ref}}$$(보통 SFT 모델)에서 **너무 멀어지지 마라.**
- $$\beta$$: 이 둘의 균형. 이 뒷항이 없으면 정책은 reward의 허점을 향해 폭주한다 — 이것이 reward hacking의 수학적 정체다([#10](/blog/2026/reward-model-overoptimization/)).

설계의 많은 부분이 **"$$r$$을 무엇으로 채울까"(조달처)**와 **"$$\beta$$와 데이터로 폭주를 어떻게 막을까"(방어)**의 두 질문으로 환원된다.

# Method

## 7단계 설계 절차

절차를 하나의 결정 흐름으로 정리하면 이렇다. 위에서부터 순서대로 답해 내려간다.

| 단계 | 질문                                          | 답에 따른 선택                                                                   |
| ---- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| 1    | 이 도메인의 정답을 **프로그램으로 판정**하나? | 예 → 2단계 / 아니오 → 3단계                                                      |
| 2    | (검증 가능) reward를 어떻게 짜나?             | ① 규칙: accuracy + format 분리                                                   |
| 3    | 채점 **기준(rubric)** 을 명문화할 수 있나?    | 예 → 4단계 / 아니오 → 5단계                                                      |
| 4    | 정답 **예시(reference)** 가 있나?             | 예 → ③ reference judge / 아니오 → ④ GRM·self-critique                            |
| 5    | (기준도 흐릿) 무엇으로 신호를 만드나?         | ② 스칼라 RM 또는 선호 쌍 + DPO                                                   |
| 6    | 정책을 어떻게 업데이트하나?                   | 온라인 인프라 있음 → PPO/GRPO / 없음 → DPO·rejection sampling                    |
| 7    | reward를 어떻게 **검증·방어**하나?            | 프롬프트 큐레이션 + KL 분리 + RM 평가(RewardBench/CriticEval) + overopt 모니터링 |

아래에서 각 단계의 실무 포인트를 짚는다.

### 1~2단계: 검증 가능하면 무조건 규칙부터

"정답을 프로그램으로 판정 가능한가"가 첫 갈림길이다. 가능하다면 학습된 RM을 **쓰지 않는 것**이 정답이다([#21](/blog/2026/deepseek-r1/)). 규칙 기반 reward를 짤 때 실무 포인트는 두 가지다.

- **accuracy와 format을 분리한다.** 정답 여부만 보상하면 모델은 "정답만 찍는" 최단 경로로 붕괴한다. 사고 과정을 `<think>` 태그나 `\boxed{}` 형식으로 강제하는 format reward를 따로 둔다.
- **파서를 견고하게.** 규칙 reward의 취약점은 함수가 아니라 **파서**다. 답 추출 정규식이 허술하면 그 틈이 곧 hacking 표면이 된다.

### 3~4단계: 기준을 글로 쓸 수 있으면 judge

검증이 불가능해도 포기하지 않는다. **채점 기준을 문장으로 쓸 수 있는지**를 묻는다. "정확성 40%, 안전성 40%, 친절함 20%, 각 항목 1~5점" 같은 rubric을 정의할 수 있다면, 스칼라 RM으로 내려가기 전에 **judge**를 고려한다.

- **정답 예시가 있으면 ③ reference judge** ([#22](/blog/2026/prometheus-2/)): judge에게 "이 rubric과 이 모범답안에 비추어 채점하라"고 시킨다. Qwen3가 General RL에서 쓴 model-based-with-reference가 정확히 이것이다([#32](/blog/2026/frontier-reward-design/)).
- **예시가 없으면 ④ GRM / self-critique** ([#26](/blog/2026/deepseek-grm-spct/), [#28](/blog/2026/j1-thinking-judge/)): judge가 채점 근거를 스스로 생성하며 점수를 낸다. 근거가 남으므로 **왜 그 점수인지 사후 검증이 가능**하다는 게 스칼라 RM 대비 결정적 장점이다.

judge를 쓸 때 주의: judge도 신경망이라 뚫린다. [#31 One Token to Fool](/blog/2026/one-token-to-fool-judge/)이 보였듯, "정답입니다" 같은 껍데기 토큰에 속는 사례가 실재한다. judge를 도입하면 judge 자체의 견고성 평가([#30 CriticEval](/blog/2026/criticeval/))가 새 숙제로 따라온다.

### 5단계: 기준도 흐릿하면 선호 쌍

rubric조차 명문화하기 어려운 주관적 품질(문체, 위트, 공감)은 결국 **사람의 선호 쌍**으로 돌아온다. 여기서 두 갈래다.

- **② 스칼라 RM 학습 후 RL**: 표현력이 크지만 overoptimization 위험.
- **DPO([#18](/blog/2026/dpo/))**: RM 없이 선호 쌍에서 바로 정책 학습. 온라인 인프라가 없을 때 특히 매력적.

이때 [#5](/blog/2026/secrets-rlhf-reward-modeling/)·[#6](/blog/2026/skywork-reward/)의 교훈이 결정적이다 — **RM 성능은 아키텍처보다 데이터 큐레이션에서 갈린다.** 우열이 애매한 쌍은 노이즈이므로 버리고, 확실히 우열이 갈리는 쌍만 남긴다(Llama의 선택, [#32](/blog/2026/frontier-reward-design/)).

### 6단계: 온라인이냐 오프라인이냐

정책 업데이트 알고리즘은 **인프라와 비용**이 정한다.

| 상황                               | 선택                    | 근거                                                                                                 |
| ---------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| 온라인 롤아웃 인프라 있음          | PPO / GRPO / RLOO       | [#14](/blog/2026/ppo/), [#16](/blog/2026/grpo-deepseekmath/), [#17](/blog/2026/rloo-back-to-basics/) |
| 인프라 없음, 선호 쌍만 있음        | DPO                     | [#18](/blog/2026/dpo/)                                                                               |
| 그 중간(싸게 다지고 비싸게 마무리) | offline DPO → online RL | Qwen2.5 ([#32](/blog/2026/frontier-reward-design/))                                                  |

주의점 하나: **오프라인 정렬(SFT·DPO)을 너무 세게 걸면 뒤이은 RL의 탐색을 죽인다**(Llama 4의 교훈, [#32](/blog/2026/frontier-reward-design/)). 온라인 RL을 쓸 계획이라면 앞 단계를 "가볍게" 치는 편이 낫다.

### 7단계: 검증과 방어를 처음부터 설계에 넣는다

reward를 정하고 나면 곧바로 **"이 reward가 정말 품질과 상관하는가"**를 검증하는 장치를 붙인다. 이건 나중에 덧대는 게 아니라 설계의 일부다.

- **프롬프트 큐레이션 = 숨은 reward 설계.** 응답 점수 분산이 큰(변별력 있는) 쿼리를 우선하고(Qwen2.5), advantage가 0인 프롬프트는 제거한다(Llama 4). 함수를 안 바꾸고도 신호의 질이 오른다([#32](/blog/2026/frontier-reward-design/)).
- **reward와 KL을 분리.** length·format 같은 오염 신호는 reward에서 떼어내고([#12 ODIN](/blog/2026/odin-disentangled-reward/)), 참조점 이탈은 KL로 따로 관리한다.
- **RM을 평가한다.** RM 자체를 [RewardBench 2](/blog/2026/rewardbench-2/)로, judge를 [CriticEval](/blog/2026/criticeval/)로 점검한다. "RM 정확도"와 "그 RM으로 학습한 정책의 품질"은 다르다.
- **overoptimization을 모니터링.** [#10](/blog/2026/reward-model-overoptimization/)이 정량화했듯, RL을 오래 돌릴수록 reward는 오르는데 실제 품질은 어느 지점부터 꺾인다. proxy reward와 실제(사람·홀드아웃) 지표를 나란히 본다.
- **여러 RM을 평균낸다.** 여력이 되면 [#13 WARM](/blog/2026/warm-weight-averaged-reward/)처럼 weight averaging으로 RM 하나의 특이한 취약점을 상쇄한다.

# Experiments

## 예제: 안전성 정렬 어시스턴트의 reward 설계

절차를 구체적 예제로 따라가 본다. 목표는 "도움이 되면서도 위험 요청은 적절히 거절하는 대화 어시스턴트"다. 도메인이 뒤섞여 있으므로 **한 개의 reward가 아니라 도메인별로 조달처를 나눈다** — 이것이 Solar·DeepSeek-V4가 보여준 패턴이다([#32](/blog/2026/frontier-reward-design/)).

| 하위 도메인         | 1단계 진단          | 선택한 조달처                      | 이유                                                      |
| ------------------- | ------------------- | ---------------------------------- | --------------------------------------------------------- |
| 수학·코드 질문      | 검증 가능           | ① 규칙 (정답 일치·테스트)          | 파라미터 0, hacking 불가                                  |
| 사실 질의응답       | 부분 검증 가능      | ③ reference judge (정답 문서 대조) | 정답 예시가 있고 기준(사실 일치)이 명확                   |
| 안전성(거절 적절성) | 검증 불가·기준 명확 | ④ GRM (거절 사유를 생성하며 채점)  | rubric은 쓸 수 있으나 정답 예시가 다양 → 근거 생성이 유리 |
| 문체·공감(상담 톤)  | 기준도 흐릿         | ② 스칼라 RM 또는 선호 쌍 + DPO     | 주관적 품질, 선호 쌍으로만 포착                           |

여기서 안전성 도메인이 특히 까다롭다. "거절해야 하는가"는 규칙으로 다 못 짜고(맥락 의존), 그렇다고 스칼라 RM에 맡기면 **왜 그 점수인지 알 수 없어 over-refusal(과잉 거절)을 디버깅하기 어렵다.** 그래서 근거를 생성하는 ④ GRM이 유리하다 — "이 요청은 교육 목적이라 거절 불필요"처럼 판정 근거가 로그로 남아, 과잉 거절이 어디서 생기는지 추적할 수 있다.

## 절차를 프론티어 모델에 대보면

이 7단계가 임의의 규칙이 아니라는 것은, 프론티어 모델들이 실제로 이 갈림길들을 지났다는 데서 드러난다.

| 단계          | DeepSeek-V4    | Qwen3                   | Llama 4                |
| ------------- | -------------- | ----------------------- | ---------------------- |
| 1~2 검증 가능 | ① 규칙         | ① 규칙 (query-verifier) | 난이도 커리큘럼 중심   |
| 3~4 judge     | ④ GRM          | ③ reference judge       | (비공개)               |
| 5 선호        | —              | ② 스칼라 RM             | 선호 쌍 + DPO          |
| 6 정책        | GRPO → 증류    | GRPO + General RL       | online RL → 가벼운 DPO |
| 7 방어        | 규칙 우선 + KL | 규칙으로 hacking 예방   | advantage 0 제거       |

세 모델이 서로 다른 칸을 통과했지만, **거쳐간 갈림길 자체는 같다.** 이것이 이 절차를 "설계 문서"라 부를 수 있는 이유다.

# Conclusion

## 한 장 체크리스트

reward를 설계할 때, 이 순서대로 자문한다.

1. **검증 가능한가?** → 가능하면 무조건 ① 규칙부터. RM을 학습시키지 않는 것이 최선의 RM 설계다.
2. **rubric을 쓸 수 있는가?** → 예시가 있으면 ③ reference judge, 없으면 ④ GRM. 스칼라 RM으로 바로 내려가지 않는다.
3. **그조차 안 되면** → ② 선호 쌍. 이때 **데이터 큐레이션이 아키텍처를 이긴다** — 애매한 쌍은 버린다.
4. **정책 업데이트**는 인프라가 정한다. 온라인 RL을 쓸 거면 앞의 오프라인 정렬을 가볍게.
5. **검증·방어는 처음부터.** 프롬프트 큐레이션, KL 분리, RM 평가, overopt 모니터링을 설계에 내장한다.
6. **도메인이 섞여 있으면 조달처도 섞는다.** 하나의 만능 reward를 찾지 말고, 도메인별로 ①~④를 배분한다.

## 한계: reward는 끝내 프록시다

이 모든 절차를 거쳐도 남는 사실이 하나 있다 — **어떤 reward도 결국 "품질"이라는 것의 프록시일 뿐이다.** 규칙은 검증 가능한 것만 보고, RM은 근사하며, judge도 뚫린다([#31](/blog/2026/one-token-to-fool-judge/)). 그래서 잘 설계된 reward 시스템일수록 역설적으로 **"reward를 언제 믿지 말아야 하는가"를 함께 설계한다** — 홀드아웃 사람 평가, overoptimization 조기 경보, 여러 신호의 교차검증이 그 안전장치다.

[#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)에서 시작해 [#32](/blog/2026/frontier-reward-design/)의 프론티어 사례까지, 이 시리즈가 33편에 걸쳐 도달한 결론은 한 문장이다. **reward 설계란 완벽한 보상 함수를 찾는 일이 아니라, 내 도메인을 정확히 진단하고 그 진단에 맞는 조달처를 고른 뒤, 그 선택이 틀렸을 때를 대비하는 일이다.**

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

32. [프론티어 모델의 reward 설계 (2025~2026)](/blog/2026/frontier-reward-design/) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것
33. **(현재 글)** reward를 어떻게 설계할 것인가 — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 33편으로 구성된다.

# 참고 문헌

- 이 글은 시리즈 전체를 종합한 설계 가이드로, 각 결정의 근거는 본문에 링크한 편들에 있다.
- Lambert et al., 2024. [RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787) — [#9](/blog/2026/rewardbench-2/) 참고.
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — [#18](/blog/2026/dpo/).
- Gao et al., 2022. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — [#10](/blog/2026/reward-model-overoptimization/).
- Chen et al., 2024. [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://arxiv.org/abs/2402.07319) — [#12](/blog/2026/odin-disentangled-reward/).
- Kim et al., 2024. [Prometheus 2](https://arxiv.org/abs/2405.01535) — [#22](/blog/2026/prometheus-2/).
- Liu et al. (DeepSeek-AI), 2025. [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/).
