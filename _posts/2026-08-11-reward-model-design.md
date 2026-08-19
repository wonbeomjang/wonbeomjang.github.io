---
layout: post
title: "reward를 어떻게 설계할 것인가 — RM 설계 실무"
date: 2026-08-11 09:44:00 +0900
description: "RLHF Reward 설계 시리즈 #44 — 시리즈 43편을 관통한 reward 시스템 설계 절차를 한 장의 실무 가이드로"
categories: [paper]
tags: [rlhf, reward-model, rlvr, genrm, dpo, reward-hacking, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 특정 논문 한 편이 아니라, 이 시리즈 43편이 쌓은 결론을 **"내가 reward를 설계한다면 어떤 순서로 결정할까"**라는 하나의 절차로 압축한다. 각 결정마다 근거가 된 편으로 링크를 건다.

# Introduction

[#43](/blog/2026/frontier-reward-design/)가 프론티어 모델들이 **실제로 무엇을 골랐는지**를 관찰했다면, 이 글은 반대 방향이다 — **내가 처음부터 reward 시스템을 설계한다면 어떤 순서로 결정을 내려야 하는가.** 앞의 42편이 부품(논문 한 편이 문제 하나에 답한 것)이고 #43이 사례집이라면, 이 글은 그 둘을 겹쳐 만든 **설계 절차서**다.

reward 설계에서 초심자가 가장 자주 하는 실수는 "좋은 reward model을 학습시키자"부터 시작하는 것이다. 하지만 시리즈 전체가 반복해서 보여준 교훈은 정반대다.

- **학습된 RM은 근사이고, 근사에는 반드시 hacking당하는 지점이 있다**([#10](/blog/2026/reward-model-overoptimization/)).
- 그래서 **"굳이 RM을 학습시키지 않아도 되는 경우"를 먼저 걸러내는 것**이 설계의 1단계다([#32](/blog/2026/deepseek-r1/)).
- RM이 꼭 필요한 경우에도, **reward의 질은 함수보다 데이터와 프롬프트 선택에서 더 크게 갈린다**([#5](/blog/2026/secrets-rlhf-reward-modeling/), [#6](/blog/2026/skywork-reward/)).

이 글은 다음을 순서대로 다룬다.

1. reward를 조달하는 네 가지 방법과 각각의 비용 (Background)
2. 도메인 진단부터 검증까지, 7단계 설계 절차 (Method)
3. 절차를 실제 예제(안전성 정렬 어시스턴트)에 적용해보기 (Experiments)
4. 한 장으로 압축한 설계 체크리스트와 한계 (Conclusion)

# Background

## reward 조달처 네 가지

[#43](/blog/2026/frontier-reward-design/)에서 세운 4분류를, 이번엔 **"내가 고를 때의 비용과 위험"** 관점으로 다시 본다.

| 조달처                          | 언제 쓰나                                 | 장점                                   | 비용·위험                          | 근거 편                                                                          |
| ------------------------------- | ----------------------------------------- | -------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------- |
| ① 규칙 기반 verifiable reward   | 정답을 프로그램으로 판정 가능 (수학·코드) | 파라미터 0 → hacking 불가, 학습 비용 0 | 검증 가능한 도메인에만 적용        | [#32](/blog/2026/deepseek-r1/)                                                   |
| ② 스칼라 RM (reference 없음)    | 선호 쌍만 있고 채점 기준이 흐릿할 때      | 어떤 도메인이든 적용, 빠른 추론        | 근사 오차 → overoptimization, 편향 | [#4](/blog/2026/bradley-terry-rethinking/)\~[#9](/blog/2026/rewardbench-2/)      |
| ③ reference 기반 judge          | 정답 예시 + 채점 기준(rubric)이 있을 때   | 해석 가능, 기준을 명시적으로 주입      | 판정 모델 추론 비용, judge 편향    | [#33](/blog/2026/prometheus-2/)                                                  |
| ④ generative RM (self-critique) | 기준은 있으나 정답 예시가 없을 때         | 근거를 생성 → 검증·디버깅 가능         | 추론 비용 큼, judge도 뚫릴 수 있음 | [#37](/blog/2026/deepseek-grm-spct/), [#42](/blog/2026/one-token-to-fool-judge/) |

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

**정책 최적화(KL 정규화, [#19](/blog/2026/ppo/)):**

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

"정답을 프로그램으로 판정 가능한가"가 첫 갈림길이다. 가능하다면 학습된 RM을 **쓰지 않는 것**이 정답이다([#32](/blog/2026/deepseek-r1/)). 규칙 기반 reward를 짤 때 실무 포인트는 두 가지다.

- **accuracy와 format을 분리한다.** 정답 여부만 보상하면 모델은 "정답만 찍는" 최단 경로로 붕괴한다. 사고 과정을 `<think>` 태그나 `\boxed{}` 형식으로 강제하는 format reward를 따로 둔다.
- **파서를 견고하게.** 규칙 reward의 취약점은 함수가 아니라 **파서**다. 답 추출 정규식이 허술하면 그 틈이 곧 hacking 표면이 된다.

### 3~4단계: 기준을 글로 쓸 수 있으면 judge

검증이 불가능해도 포기하지 않는다. **채점 기준을 문장으로 쓸 수 있는지**를 묻는다. "정확성 40%, 안전성 40%, 친절함 20%, 각 항목 1~5점" 같은 rubric을 정의할 수 있다면, 스칼라 RM으로 내려가기 전에 **judge**를 고려한다.

- **정답 예시가 있으면 ③ reference judge** ([#33](/blog/2026/prometheus-2/)): judge에게 "이 rubric과 이 모범답안에 비추어 채점하라"고 시킨다. Qwen3가 General RL에서 쓴 model-based-with-reference가 정확히 이것이다([#43](/blog/2026/frontier-reward-design/)).
- **예시가 없으면 ④ GRM / self-critique** ([#37](/blog/2026/deepseek-grm-spct/), [#39](/blog/2026/j1-thinking-judge/)): judge가 채점 근거를 스스로 생성하며 점수를 낸다. 근거가 남으므로 **왜 그 점수인지 사후 검증이 가능**하다는 게 스칼라 RM 대비 결정적 장점이다.

judge를 쓸 때 주의: judge도 신경망이라 뚫린다. [#42 One Token to Fool](/blog/2026/one-token-to-fool-judge/)이 보였듯, "정답입니다" 같은 껍데기 토큰에 속는 사례가 실재한다. judge를 도입하면 judge 자체의 견고성 평가([#41 CriticEval](/blog/2026/criticeval/))가 새 숙제로 따라온다.

### 5단계: 기준도 흐릿하면 선호 쌍

rubric조차 명문화하기 어려운 주관적 품질(문체, 위트, 공감)은 결국 **사람의 선호 쌍**으로 돌아온다. 여기서 두 갈래다.

- **② 스칼라 RM 학습 후 RL**: 표현력이 크지만 overoptimization 위험.
- **DPO([#23](/blog/2026/dpo/))**: RM 없이 선호 쌍에서 바로 정책 학습. 온라인 인프라가 없을 때 특히 매력적.

이때 [#5](/blog/2026/secrets-rlhf-reward-modeling/)·[#6](/blog/2026/skywork-reward/)의 교훈이 결정적이다 — **RM 성능은 아키텍처보다 데이터 큐레이션에서 갈린다.** 우열이 애매한 쌍은 노이즈이므로 버리고, 확실히 우열이 갈리는 쌍만 남긴다(Llama의 선택, [#43](/blog/2026/frontier-reward-design/)).

### 5단계 보강 — 프롬프트 큐레이션: 어떤 문제에 reward를 먹일까

5단계가 "어떤 선호 쌍을 학습에 쓸까"였다면, RL로 넘어가기 직전 하나가 더 있다 — **"어떤 프롬프트에 그 reward를 굴릴까".** GRPO·PPO는 한 프롬프트에 여러 응답을 뽑아 그 응답들의 **차이**로 학습하므로, 응답이 다 맞거나(정답률 100%) 다 틀리면(0%) advantage가 0이 되어 **그래디언트가 없다.** 즉 프롬프트 선택이 곧 신호의 유무를 정한다.

문제는 "신호가 뚜렷한 프롬프트"가 **정책에 상대적**이라는 것이다. 초반에 어렵던 문제도 학습이 진행되면 다 맞게 돼 신호가 사라진다. 그래서 offline 한 번으로 끝나지 않고 online 보정이 필요하다.

| 시점              | 방법                                                                             | 근거                         |
| ----------------- | -------------------------------------------------------------------------------- | ---------------------------- |
| offline (학습 전) | 현재 정책으로 pass@k를 재서 정답률 0%·100% 극단을 제외, 중간 밴드만 남김         | Llama 4의 pass@k 선별        |
| offline           | judge로 easy 태깅 프루닝 / RM 점수 분산 큰 쿼리 우선                             | Llama 4 프루닝, Qwen2.5 분산 |
| online (학습 중)  | 롤아웃 후 advantage=0(그룹 reward가 전부 같음) 프롬프트를 버리고 배치를 재충전   | DAPO의 dynamic sampling      |
| online            | N 스텝마다 난이도 재추정 → 정답률 오른 문제를 빼고 더 어려운 문제 투입(커리큘럼) | Llama 4 medium-hard 유지     |

실무 레시피는 계층적이다 — **offline로 굵게(검증 가능 문제만 + pass@k로 극단 제외) 후보군을 만들고, online로 매 스텝 죽은(advantage 0) 프롬프트를 쳐내며 난이도를 점증**시킨다. offline만으론 움직이는 표적을 못 따라가고, 매 스텝 전체를 online 판별하면 비싸기 때문이다.

하이퍼파라미터 감각 하나: pass@k의 **밴드 폭**을 너무 좁히면(예 40\~60%) 데이터가 마르고, 너무 넓히면 신호 약한 게 섞인다. 보통 넓게(10\~90%) 시작해 학습이 진행되며 하한을 올린다.

### 6단계: 온라인이냐 오프라인이냐

정책 업데이트 알고리즘은 **인프라와 비용**이 정한다.

| 상황                               | 선택                    | 근거                                                                                                 |
| ---------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| 온라인 롤아웃 인프라 있음          | PPO / GRPO / RLOO       | [#19](/blog/2026/ppo/), [#21](/blog/2026/grpo-deepseekmath/), [#22](/blog/2026/rloo-back-to-basics/) |
| 인프라 없음, 선호 쌍만 있음        | DPO                     | [#23](/blog/2026/dpo/)                                                                               |
| 그 중간(싸게 다지고 비싸게 마무리) | offline DPO → online RL | Qwen2.5 ([#43](/blog/2026/frontier-reward-design/))                                                  |

주의점 하나: **오프라인 정렬(SFT·DPO)을 너무 세게 걸면 뒤이은 RL의 탐색을 죽인다**(Llama 4의 교훈, [#43](/blog/2026/frontier-reward-design/)). 온라인 RL을 쓸 계획이라면 앞 단계를 "가볍게" 치는 편이 낫다.

### 7단계: 검증과 방어를 처음부터 설계에 넣는다

reward를 정하고 나면 곧바로 **"이 reward가 정말 품질과 상관하는가"**를 검증하는 장치를 붙인다. 이건 나중에 덧대는 게 아니라 설계의 일부다.

- **프롬프트 큐레이션 = 숨은 reward 설계.** 신호 없는(advantage 0) 프롬프트를 offline 후보 선별 + online 실시간 제거로 걸러 함수를 안 바꾸고도 신호의 질을 올린다(위 '5단계 보강' 절 참고).
- **reward와 KL을 분리.** length·format 같은 오염 신호는 reward에서 떼어내고([#12 ODIN](/blog/2026/odin-disentangled-reward/)), 참조점 이탈은 KL로 따로 관리한다.
- **RM을 평가한다.** RM 자체를 [RewardBench 2](/blog/2026/rewardbench-2/)로, judge를 [CriticEval](/blog/2026/criticeval/)로 점검한다. "RM 정확도"와 "그 RM으로 학습한 정책의 품질"은 다르다.
- **overoptimization을 모니터링.** [#10](/blog/2026/reward-model-overoptimization/)이 정량화했듯, RL을 오래 돌릴수록 reward는 오르는데 실제 품질은 어느 지점부터 꺾인다. proxy reward와 실제(사람·홀드아웃) 지표를 나란히 본다.
- **여러 RM을 평균낸다.** 여력이 되면 [#13 WARM](/blog/2026/warm-weight-averaged-reward/)처럼 weight averaging으로 RM 하나의 특이한 취약점을 상쇄한다.

# Experiments

## 예제: 안전성 정렬 어시스턴트의 reward 설계

절차를 구체적 예제로 따라가 본다. 목표는 "도움이 되면서도 위험 요청은 적절히 거절하는 대화 어시스턴트"다. 도메인이 뒤섞여 있으므로 **한 개의 reward가 아니라 도메인별로 조달처를 나눈다** — 이것이 Solar·DeepSeek-V4가 보여준 패턴이다([#43](/blog/2026/frontier-reward-design/)).

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

이 모든 절차를 거쳐도 남는 사실이 하나 있다 — **어떤 reward도 결국 "품질"이라는 것의 프록시일 뿐이다.** 규칙은 검증 가능한 것만 보고, RM은 근사하며, judge도 뚫린다([#42](/blog/2026/one-token-to-fool-judge/)). 그래서 잘 설계된 reward 시스템일수록 역설적으로 **"reward를 언제 믿지 말아야 하는가"를 함께 설계한다** — 홀드아웃 사람 평가, overoptimization 조기 경보, 여러 신호의 교차검증이 그 안전장치다.

[#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)에서 시작해 [#43](/blog/2026/frontier-reward-design/)의 프론티어 사례까지, 이 시리즈가 44편에 걸쳐 도달한 결론은 한 문장이다. **reward 설계란 완벽한 보상 함수를 찾는 일이 아니라, 내 도메인을 정확히 진단하고 그 진단에 맞는 조달처를 고른 뒤, 그 선택이 틀렸을 때를 대비하는 일이다.**

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 마흔네 번째 글이다.

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
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="43">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열한 개 모델이 실제로 택한 것</li>
  <li><strong>(현재 글)</strong> reward를 어떻게 설계할 것인가 — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 44편으로 구성된다.

# 참고 문헌

- 이 글은 시리즈 전체를 종합한 설계 가이드로, 각 결정의 근거는 본문에 링크한 편들에 있다.
- Lambert et al., 2024. [RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787) — [#9](/blog/2026/rewardbench-2/) 참고.
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — [#23](/blog/2026/dpo/).
- Gao et al., 2022. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — [#10](/blog/2026/reward-model-overoptimization/).
- Chen et al., 2024. [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://arxiv.org/abs/2402.07319) — [#12](/blog/2026/odin-disentangled-reward/).
- Kim et al., 2024. [Prometheus 2](https://arxiv.org/abs/2405.01535) — [#33](/blog/2026/prometheus-2/).
- Liu et al. (DeepSeek-AI), 2025. [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — [#37 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/).
- Yu et al. (ByteDance Seed·Tsinghua), 2025. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — dynamic sampling(정답률 0/1 프롬프트 제거).
