---
layout: post
title: "프론티어 모델은 reward를 어떻게 설계했나"
date: 2026-08-11 09:27:00 +0900
description: "RLHF Reward 설계 시리즈 #32 — DeepSeek·Qwen·Llama·Kimi·Solar·K-EXAONE·A.X 프론티어 모델의 reward 설계 비교"
categories: [paper]
tags: [rlhf, reward-model, rlvr, dpo, grpo, genrm, deepseek, qwen, llama, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 일곱 프론티어 모델의 공개 자료를 가로지른다 — [DeepSeek-V4](https://arxiv.org/abs/2606.19348), [Qwen3](https://arxiv.org/abs/2505.09388), [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Kimi K3](https://github.com/MoonshotAI/Kimi-K3), [Solar Open 2](https://arxiv.org/abs/2607.20062), [K-EXAONE 2.0](https://arxiv.org/abs/2608.04505), [A.X K2](https://github.com/SKT-AI/A.X-K2). 뒤의 셋(Solar·K-EXAONE·A.X)은 한국 연구팀 모델이다. 이 중 Llama 4만 정식 technical report 없이 공식 블로그로 공개됐다.

# Introduction

31편 동안 이 시리즈는 reward를 부품 단위로 뜯어봤다. 사람 선호를 스칼라로 압축하는 Bradley-Terry([#4](/blog/2026/bradley-terry-rethinking/)), 그 스칼라가 hacking당하는 방식과 방어법([#10](/blog/2026/reward-model-overoptimization/)\~[#13](/blog/2026/warm-weight-averaged-reward/)), reward를 정책 업데이트로 바꾸는 PPO·GRPO·DPO([#14](/blog/2026/ppo/)\~[#18](/blog/2026/dpo/)), 검증 가능한 도메인에서 학습된 reward model 자체를 규칙으로 대체하는 RLVR([#19](/blog/2026/lets-verify-step-by-step/)\~[#21](/blog/2026/deepseek-r1/)), 학습된 RM을 생성형 judge로 재구성하는 흐름([#22](/blog/2026/prometheus-2/)\~[#26](/blog/2026/deepseek-grm-spct/)), 그리고 그 judge가 스스로 생각하고 그 신뢰 자체를 검증하는 최신 연구([#27](/blog/2026/reasongrm/)\~[#31](/blog/2026/one-token-to-fool-judge/)). 하나하나는 특정 논문이 특정 문제 하나에 답한 결과였다.

그런데 실제로 프론티어급 모델을 학습시키는 팀은 이 부품 중 무엇을, 어떤 조합으로, 왜 골랐을까. 이 글은 "논문 1편 = 포스트 1편" 형식을 벗어나 일곱 개 공개 자료 — DeepSeek-V4, Qwen3, Llama 4, Kimi K3, Solar Open 2, K-EXAONE 2.0, A.X K2 — 를 가로질러 reward 설계의 실전 선택지를 비교한다. 그리고 이 비교에서 뽑아낸 설계 원칙은 다음 글 [#33](/blog/2026/reward-model-design/)에서 한 장의 실무 가이드로 정리한다.

미리 결론의 윤곽을 말하면 세 가지다.

1. **검증 가능한 도메인(수학, 코드)에서는 일곱 모델이 놀랄 만큼 수렴한다.** 학습된 RM을 걷어내고 규칙 검증기를 쓴다.
2. **검증 불가능한 도메인(대화, 글쓰기, 안전성)에서는 갈라진다** — 하지만 그 분기의 한복판에서 **generative reward model(GRM)이 처음으로 프로덕션 report에 등장**했다. 이 시리즈 6부가 "연구 단계에만 있다"고 정리했던 흐름이 DeepSeek-V4와 Kimi K3에서 실제 학습 파이프라인으로 넘어온 순간이다.
3. **reward 설계는 "함수"만의 문제가 아니다.** 어떤 프롬프트에 그 함수를 먹이느냐(난이도 커리큘럼, 분산 선별), 오프라인·온라인을 어떻게 섞느냐가 함수 선택만큼 중요하다.

이 글은 다음 순서로 그 수렴점과 분기점을 짚는다.

1. 시리즈가 쌓은 재료를 reward "조달처" 4분류로 재정리한다 (Background)
2. 일곱 모델 각각이 reward를 어디서 조달하고 어떤 알고리즘으로 정책을 업데이트했는가 (Method)
3. 검증 가능/불가능 도메인에서 왜 수렴하고 왜 갈라지는가, GRM은 어디까지 왔는가 (Experiments)
4. 프롬프트 선별·커리큘럼이라는 "숨은 reward 설계", 그리고 hacking 방어 (Experiments)
5. 지금 내가 reward를 설계해야 한다면 무엇을 골라야 하는가 (Conclusion)

먼저 일곱 모델을 한 장으로 요약하면 이렇다. 각 칸의 근거는 이어지는 Method·Experiments에서 편별로 짚는다.

| 모델         | 팀            | 핵심 reward 조달처                         | RL 알고리즘                 |
| ------------ | ------------- | ------------------------------------------ | --------------------------- |
| DeepSeek-V4  | DeepSeek (중) | 규칙 + GRM(비검증)                         | GRPO → on-policy 증류       |
| Qwen3        | Alibaba (중)  | 규칙 / reference judge / 스칼라 RM 3분류   | GRPO + General RL           |
| Llama 4      | Meta (미)     | 비공개(online RL) + 선호쌍                 | SFT → online RL → DPO       |
| Kimi K3      | Moonshot (중) | 규칙(51.2M 샌드박스) + Agentic GRM         | 9전문가 RL → MOPD 증류      |
| Solar Open 2 | Upstage (한)  | 규칙 + rubric judge                        | GRPO(token) → 12전문가 MOPD |
| K-EXAONE 2.0 | LG (한)       | 도메인별 규칙·rubric·judge                 | GrouPER + AGAPO             |
| A.X K2       | SKT (한)      | 규칙(+난이도필터) + reference rubric judge | CISPO + GDPO                |

세 축이 한눈에 보인다 — **검증 가능 도메인은 모두 규칙**, **검증 불가능 도메인은 judge·GRM으로 갈리고**, **여럿이 "전문가→증류"로 통합**한다. 아래에서 하나씩 뜯는다.

# Background

## 시리즈가 쌓은 재료: 세 가지 축

시리즈가 지금까지 쌓아온 재료를 세 가지 축으로 요약하면 이렇다.

| 축                                       | 선택지                                                        | 관련 편                                                                                                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| reward를 어디서 조달하는가               | 학습된 스칼라 RM / 규칙 기반 verifiable reward / judge·rubric | 2부([#4](/blog/2026/bradley-terry-rethinking/)\~[#9](/blog/2026/rewardbench-2/)), 5부([#19](/blog/2026/lets-verify-step-by-step/)\~[#21](/blog/2026/deepseek-r1/)), 6·7부([#22](/blog/2026/prometheus-2/)\~[#31](/blog/2026/one-token-to-fool-judge/)) |
| 그 reward로 정책을 어떻게 업데이트하는가 | PPO / GRPO / DPO                                              | 4부([#14](/blog/2026/ppo/)\~[#18](/blog/2026/dpo/))                                                                                                                                                                                                    |
| 도메인을 어떻게 가르는가                 | 검증 가능(정답 존재) / 검증 불가능(정답 부재)                 | [#21 DeepSeek-R1](/blog/2026/deepseek-r1/)                                                                                                                                                                                                             |

## reward 조달처를 4분류로: 이 글의 렌즈

일곱 모델을 비교하려면 공통 좌표계가 필요하다. 마침 Qwen3의 report가 그 좌표계를 거의 그대로 제공한다 — Qwen3는 General RL 단계에서 reward를 **세 종류**로 명시적으로 나눈다. 여기에 DeepSeek-V4가 도입한 GRM을 더하면 **네 개의 조달처**가 된다. 이 4분류가 이 글 전체의 렌즈다.

| 조달처                               | 어떻게 점수를 매기나                                      | 파라미터 유무       | 시리즈 대응                                                                                      |
| ------------------------------------ | --------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------ |
| ① 규칙 기반 verifiable reward        | 파서 + 비교 로직 (정답 일치, 테스트 통과)                 | 없음 (hacking 불가) | 5부 [#21 RLVR](/blog/2026/deepseek-r1/)                                                          |
| ② 스칼라 RM (reference 없음)         | 사람 선호로 학습한 신경망이 스칼라 점수                   | 있음                | 2부 [#4](/blog/2026/bradley-terry-rethinking/)\~[#9](/blog/2026/rewardbench-2/)                  |
| ③ reference 기반 judge               | 정답 예시를 주고 그에 비추어 채점 (rubric 조건부)         | 있음 (판정 모델)    | 6부 [#22 Prometheus 2](/blog/2026/prometheus-2/)                                                 |
| ④ generative RM (GRM, self-critique) | 모델이 근거를 생성하며 채점, 때로 자기 출력을 스스로 평가 | 있음 (생성 모델)    | 6·7부 [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/), [#28 J1](/blog/2026/j1-thinking-judge/) |

이 4분류를 머리에 넣고 보면, 일곱 모델의 선택이 한눈에 정렬된다. ①은 검증 가능 도메인의 표준이 됐고, ②·③·④는 검증 불가능 도메인을 두고 갈라진다.

## 비교 대상 일곱 모델

| 모델         | 발표    | 공개 형태                                                                                    |
| ------------ | ------- | -------------------------------------------------------------------------------------------- |
| DeepSeek-V4  | 2026-06 | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)                                         |
| Qwen3        | 2025-05 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)                                         |
| Llama 4      | 2025-04 | [Meta AI 블로그](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (정식 논문 없음) |
| Kimi K3      | 2026-07 | [Moonshot AI tech report](https://github.com/MoonshotAI/Kimi-K3)                             |
| Solar Open 2 | 2026-07 | [arXiv:2607.20062](https://arxiv.org/abs/2607.20062)                                         |
| K-EXAONE 2.0 | 2026-08 | [arXiv:2608.04505](https://arxiv.org/abs/2608.04505)                                         |
| A.X K2       | 2026    | [SKT-AI tech report](https://github.com/SKT-AI/A.X-K2)                                       |

Llama 4만 상세한 technical report가 없다(공식 블로그). 나머지 여섯은 arXiv나 GitHub에 report를 공개했다. 특히 한국 팀 셋(Solar·K-EXAONE·A.X)이 한 비교표에 함께 오르는 게 이번 세대의 특징이다. 비공개 구간은 이 글이 매번 명시한다.

# Method

## DeepSeek-V4: 도메인 전문가를 각자 키운 뒤 하나로 증류한다

[#21](/blog/2026/deepseek-r1/)에서 다룬 DeepSeek-R1은 **하나의 정책**에 규칙 기반 reward를 꽂아 GRPO로 밀어붙이는 구조였다. DeepSeek-V4는 이 구조를 정면으로 바꾼다. 후처리를 **두 단계**로 나눈다.

1. **도메인 전문가의 독립 육성.** 수학, 코딩, 에이전트, instruction following 같은 도메인마다 **별도의 전문가 모델**을 따로 학습시킨다. 각 전문가는 같은 베이스에서 출발해 (a) 도메인 특화 데이터로 SFT를 받아 기초를 잡고, (b) 그 위에 GRPO를 얹어 "그 도메인의 성공 기준에 맞춘 reward"로 최적화된다.
2. **통합 모델로의 on-policy 증류.** 이렇게 만든 N개의 전문가를 하나의 모델로 합친다. 통합 모델(student)이 각 전문가(teacher)를 향해 **reverse KL을 최소화**하며 배우는 on-policy distillation이다.

$$
\mathcal{L}_{\text{distill}} = \mathbb{E}_{y \sim \pi_{\text{student}}} \left[ \log \frac{\pi_{\text{student}}(y \mid x)}{\pi_{\text{teacher}}(y \mid x)} \right]
$$

- $$\pi_{\text{student}}$$: 통합 모델 (배우는 쪽)
- $$\pi_{\text{teacher}}$$: 해당 도메인 전문가 (가르치는 쪽)
- $$y \sim \pi_{\text{student}}$$: **student가 스스로 생성한** 출력 위에서 손실을 잰다는 것이 "on-policy"의 핵심 — 오프라인 데이터셋이 아니라 student의 현재 분포에서 샘플링한다.

reward 관점에서 V4의 결정적 변화는 **조달처를 도메인마다 갈아 끼운다**는 점이다. 수학·코드처럼 규칙으로 검증되는 도메인은 R1과 같은 ① 규칙 기반 reward를 쓴다. 반면 규칙으로 검증하기 어려운 도메인에는 **④ Generative Reward Model(GRM)** 을 도입한다 — 전통적인 스칼라 RM 대신, **rubric으로 안내된 RL 데이터**로 학습해 actor가 출력을 **생성하면서 동시에 스스로 평가**하게 만드는 방식이다. 이는 [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)가 연구 단계에서 제안한 생성형 reward가 같은 팀의 프로덕션 모델로 넘어온 사례다.

다만 report는 각 도메인이 정확히 어떤 reward 함수를 썼는지까지는 상세히 공개하지 않는다. "reward models tailored to specific success criteria"라는 표현과 GRM 도입 사실이 확인되는 수준이며, 도메인별 세부 레시피는 R1만큼 투명하지 않다.

**R1 → V4로 무엇이 바뀌었나.** 바뀐 건 두 가지 — 정책 구조와 "규칙이 안 통하는 도메인"의 reward다.

| 축               | DeepSeek-R1 (2025-01)                               | DeepSeek-V4 (2026-06)                  |
| ---------------- | --------------------------------------------------- | -------------------------------------- |
| 정책 구조        | **하나의 통합 정책**에 RL                           | **N개 도메인 전문가 → 증류로 통합**    |
| 검증 가능 reward | 규칙(accuracy + format)                             | 규칙(그대로 계승)                      |
| 검증 불가 reward | all-scenario 단계에서 **학습된 스칼라 RM**으로 복귀 | **④ GRM**(rubric-guided, 생성형)       |
| 전문가 통합      | (단일 정책이라 불필요)                              | **on-policy distillation**(reverse-KL) |
| PRM              | **명시적으로 포기**(단계 정의·중간 판정·hacking)    | 전문가별 outcome reward + GRM          |

R1의 메시지는 "검증 가능한 도메인에선 학습된 RM을 규칙으로 걷어낼 수 있다"였다. 그러나 R1도 정답이 없는 일반 대화에서는 all-scenario RL 단계에서 결국 **학습된 스칼라 RM을 다시 불러왔다.** V4는 바로 그 지점을 스칼라 RM에서 **생성형 GRM**으로 갈아끼웠고, 동시에 단일 정책을 전문가+증류 구조로 바꿨다. 규칙 reward는 그대로 두되, "규칙이 안 통하는 곳"의 답이 R1의 스칼라 RM → V4의 GRM으로 업그레이드된 셈이다.

## Qwen3: 4단계 파이프라인과 reward 3분류

Qwen3의 후처리는 네 단계로 정연하게 나뉜다.

| 단계                    | 하는 일                                  | reward 조달처              |
| ----------------------- | ---------------------------------------- | -------------------------- |
| 1. Long-CoT Cold Start  | 긴 추론 패턴을 SFT로 심는다              | (SFT, reward 없음)         |
| 2. Reasoning RL         | 수학·코드에서 GRPO로 추론력을 끌어올린다 | ① 규칙 (query-verifier 쌍) |
| 3. Thinking Mode Fusion | thinking/non-thinking 모드를 하나로 융합 | (혼합)                     |
| 4. General RL           | 개방형 도메인 전반으로 정렬을 확장       | ①·③·② 세 종류를 병용       |

2단계 Reasoning RL은 **query-verifier 쌍**을 쓴다 — 각 쿼리에 정오를 판정하는 verifier가 붙어 있고, 쿼리당 rollout을 많이 뽑아 GRPO로 학습한다. 샘플 효율을 위해 **off-policy 학습**도 섞는다(과거에 뽑아둔 rollout을 재활용).

핵심은 4단계 General RL이 reward를 **세 종류로 명시적으로 나눈다**는 점이다.

| reward 종류                              | 작동 방식                                                     | 이 시리즈 대응                                 |
| ---------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------- |
| Rule-based Reward                        | 규칙으로 정오를 높은 정밀도로 판정 — "reward hacking을 예방"  | ① [#21 RLVR](/blog/2026/deepseek-r1/)          |
| Model-based Reward **with reference**    | 정답 reference를 주고 Qwen2.5-72B-Instruct가 그에 비추어 채점 | ③ [#22 Prometheus 2](/blog/2026/prometheus-2/) |
| Model-based Reward **without reference** | 사람 선호 데이터로 학습한 RM이 스칼라 점수를 부여             | ② 2부 스칼라 RM                                |

이 세 줄이 정확히 Background의 4분류 중 ①②③에 대응한다. Qwen3가 흥미로운 건, 하나의 모델이 도메인에 따라 **세 조달처를 동시에 운용**한다는 것이다 — 정답이 있으면 규칙, 정답 예시가 있으면 reference judge, 둘 다 없으면 스칼라 RM. 작은 모델은 이 비싼 RL을 직접 돌리지 않고 **strong-to-weak distillation**으로 큰 모델을 증류해 받는데, RL 대비 약 1/10 GPU 시간이면 된다고 밝힌다.

**Qwen2.5 → Qwen3로 무엇이 바뀌었나.** reward가 "잘 고른 RM 하나"에서 "도메인별 3분류"로 분화했다.

| 축          | Qwen2.5 (2024-12)                                       | Qwen3 (2025-05)                                            |
| ----------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| RL 단계     | offline DPO → online GRPO (2단계)                       | **4단계**(cold start → reasoning RL → fusion → general RL) |
| reward 종류 | RM 기반 (단일 계열)                                     | **3분류**(규칙 / reference judge / 스칼라 RM)              |
| 쿼리 전략   | RM=RL 쿼리셋, **분산 큰 쿼리 우선**(8 샘플, batch 2048) | reasoning RL에 verifier + off-policy rollout 재활용        |
| 작은 모델   | —                                                       | **strong-to-weak distillation**(RL 대비 약 1/10 GPU)       |
| 사고량      | —                                                       | **thinking budget**으로 생각량 제어                        |

Qwen2.5의 설계 포인트는 "RM 하나를 두되, 학습 쿼리를 분산 큰(변별력 있는) 것으로 골라 신호의 질을 올린다"였다. Qwen3는 그 위에서 **reward를 도메인에 따라 세 갈래로 쪼갰다** — 검증 가능하면 규칙, 정답 예시가 있으면 reference judge(Qwen2.5-72B-Instruct가 채점), 둘 다 없으면 스칼라 RM. 즉 "잘 고른 하나의 RM"에서 "도메인마다 다른 조달처"로 넘어가며, DeepSeek-V4·Solar가 보여준 도메인별 조달처 분화와 같은 흐름에 합류했다.

## Llama 4: 온라인 RL을 되살리고, SFT·DPO는 탐색을 막지 않을 만큼만

Llama 4에서 가장 눈에 띄는 건 reward 파이프라인의 **방향 전환**이다. [#8 Llama 2](/blog/2026/llama2-rlhf/) 이후 세대인 Llama 3와 나란히 놓아야 무엇이 바뀌었는지가 드러난다.

| 축            | Llama 3 (2024, herd 논문)                                        | Llama 4 (2025, 공식 블로그)                                        |
| ------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| 정책 최적화   | rejection sampling + DPO를 **6 라운드** 반복                     | **online RL이 중심**, 앞뒤로 lightweight SFT·DPO                   |
| PPO/온라인 RL | **명시적으로 거부**("DPO가 대규모에서 연산이 적고 더 낫다")      | **온라인 RL을 되살림**                                             |
| RM의 역할     | rejection sampling 필터 + 선호쌍 정제 (온라인 gradient엔 미사용) | online RL의 reward로 추정되나 **형태 비공개**                      |
| SFT·DPO 강도  | DPO가 주 정렬 수단(formatting 토큰 마스킹, chosen에 NLL 0.2)     | **"over-constrain 방지"용 lightweight**                            |
| 데이터        | 선호 4단계 등급 + 사람 edit 3-way, 유사 응답 제거                | easy 50\~95% 프루닝, medium-hard 커리큘럼, advantage-0 실시간 제거 |

핵심은 **Llama 3가 스케일을 위해 온라인 RL(PPO)을 버리고 오프라인(rejection sampling + DPO)을 택했는데, Llama 4가 그 선택을 다시 뒤집었다**는 것이다. Llama 3 herd 논문은 PPO를 시도했다가 "DPO가 대규모 모델에서 연산이 적고 성능도 낫다"며 접었고, RM은 오직 rejection sampling의 필터로만 썼다 — 온라인 정책 업데이트에는 쓰지 않았다. Llama 4는 반대로 online RL을 파이프라인의 중심에 놓고, 그 이유를 이렇게 밝힌다.

> SFT와 DPO가 모델을 과도하게 **제약(over-constrain)** 해서, 뒤이은 online RL 단계의 **탐색을 막고** 특히 추론·코딩·수학에서 정확도를 떨어뜨린다.

즉 Llama 3에서 "주연"이던 DPO가 Llama 4에서는 "탐색을 죽이지 않을 만큼만" 가볍게 치는 조연으로 내려온다. 무게중심이 online RL로 옮겨가면서, 데이터 전략도 정적 선호쌍 큐레이션에서 **동적 난이도 관리**로 바뀐다.

- **데이터 프루닝으로 어려운 것만 남긴다.** Llama 모델 자신을 judge로 써서 "쉬움"으로 태깅된 데이터를 걸러낸다. 작은 모델은 50% 이상, 2T 규모 Behemoth는 **95%**를 쳐낸다.
- **continuous online RL.** 학습과 필터링을 번갈아 돌리며 **medium-to-hard 난이도 프롬프트만 남긴다.** 쉬운 프롬프트는 신호가 없고, 너무 어려운 프롬프트는 advantage가 0이라 학습에 기여하지 못한다.
- **hard-prompt 커리큘럼.** 정책 모델로 pass@k 분석을 해 어려운 프롬프트를 골라 난이도를 점증시키고, advantage가 0인 프롬프트를 실시간으로 걸러내며, 여러 능력의 프롬프트를 섞어 배치를 구성한다.

다만 **online RL을 실제로 굴리는 reward 신호가 무엇인지는 블로그가 의도적으로 공개하지 않는다.** 규칙 기반 verifiable reward인지, 학습된 RM인지, 혼합인지 명시가 없다. 특히 Llama 3에서 오프라인 필터로만 쓰이던 RM이 Llama 4의 온라인 루프 안으로 돌아왔는지조차 확인되지 않는다 — 확인되는 건 파이프라인의 골격과 데이터 큐레이션 전략까지다.

## Kimi K3: 도메인 전문가 아홉을 MOPD로 합친다

Kimi K3(2026-07)의 후처리는 **세 단계**다 — SFT로 콜드스타트를 잡고, RL로 도메인 전문가들을 서로 다른 reasoning-effort 수준으로 키운 뒤, **MOPD(Multi-Teacher On-Policy Distillation)**로 하나의 모델로 통합한다. 이 "전문가를 따로 키워 증류로 합친다"는 골격은 DeepSeek-V4와 정확히 같은 방향이다.

reward 조달처는 도메인에 따라 갈린다.

- **검증 가능 도메인**: 5,120만 개(51.2M) 규모의 RL 샌드박스에서 규칙 기반 verifiable reward로 학습한다 (①).
- **검증 불가능한 일반 태스크**: **Agentic Generative Reward Model(GRM)**을 쓴다 (④). judge가 강제된 프로토콜을 따른다 — (1) 출력을 읽고, (2) **rubric을 생성**하고, (3) 각 후보를 그 rubric으로 채점하고, (4) 점수를 **scorepad**에 기록한다. K2.5의 토너먼트식 그룹 비교(binary comparison)를 이어받되, **judge가 채점 기준(rubric)을 스스로 만든다**는 점이 핵심이다.

이 Agentic GRM은 [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)의 "judge가 평가 원칙을 스스로 생성한다"와 [#29 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/)의 "rubric을 보상 기준으로 쓴다"를 프로덕션에서 합친 형태다. DeepSeek-V4의 GRM과 더불어 **생성형 reward가 더 이상 논문 안에만 있지 않다**는 이번 세대의 증거다.

여기에 두 가지 reward-shaping·방어 장치가 붙는다.

- **Reasoning-effort RL**: 문제마다 초기 토큰 예산 $$b_0(x)$$를 정하고, 토큰 사용량이 $$\tau \cdot b_0(x)$$를 넘으면 task reward를 $$-1$$로 덮어써 과도한 사고를 벌한다. $$\tau$$를 큰 값에서 점점 줄이는 커리큘럼으로 low·high·max effort 전문가를 각각 만든다 — "얼마나 생각할지"를 조절하는 장치이고, 추론 시 `reasoning_effort`(low/high/max)로 노출된다.
- **Budget 기반 verbosity 제어**: Agentic GRM이 "길게 쓰면 이긴다"는 식으로 hacking되는 걸 막으려고, 초기 길이 $$\ell_0$$의 $$\sigma$$배를 넘는 후보는 binary comparison에서 자동으로 진다.

참고로 self-critique rubric은 전신 K2(2025-07)가 처음 도입했고, K3의 Agentic GRM은 그 계보 위에서 "judge가 rubric을 생성해 채점"하는 형태로 발전한 것이다.

## Solar Open 2: 열두 전문가를 MOPD로 합친다

Solar Open 2(2026-07, 250B MoE)의 reward 설계는 이전 세대(Solar Open)와 크게 다르다. 알고리즘은 **GRPO를 token-level importance sampling으로** 쓴다 — 논문은 "GSPO의 sequence-level 비율이 아니라 GRPO의 token-level 비율을 유지한다"고 명시한다(널리 퍼진 GSPO와 반대 선택). reward는 도메인별로 갈린다.

- **STEM·추론**: 검증 가능한 규칙 reward (①).
- **에이전트**: 다차원 rubric — 대화 에이전트는 과정 품질·결과 품질을 분리하고 실행 가능한 read-back으로 검증, 코딩 에이전트는 fail-to-pass 테스트 + 5차원 LLM rubric + 실행 가능성, 오피스 에이전트는 규칙 기반 checker + LLM judge.
- **전통적 스칼라 RM도, DPO도, KL 정규화도 쓰지 않는다** — 검증 가능한 실행 신호 + LLM-as-judge로 대체한다.

가장 눈에 띄는 건 통합 방식이다. **열두 개의 전문가(teacher)를 MOPD(Multi-teacher On-Policy Distillation)로 하나로 합친다** — student가 자기 궤적 위에서 routed teacher에 대한 per-position reverse KL을 최소화하되, **outcome reward를 전혀 얹지 않는 KL-only 목적함수**다. 논문은 이를 "λ로 균형 잡을 것도, reward-hacking 표면도 없다"고 표현한다. [#26](/blog/2026/deepseek-grm-spct/) 이후 DeepSeek-V4·Kimi K3가 택한 "전문가→증류"에 Solar Open 2가 합류한 것이고, **Kimi K3와는 MOPD라는 이름까지 같다.**

## K-EXAONE 2.0: 오답에서도 신호를 뽑는다

K-EXAONE 2.0(2026-08, 750B MoE, LG AI Research)은 도메인마다 "선호 응답을 고르는 기준"과 "비선호 응답의 나쁜 패턴을 벌하는 reward"를 따로 정의한다. 수학·코드는 검증 가능한 신호로, 증명은 LLM-as-judge로, 대화는 인스턴스별 rubric으로, 에이전트는 행동의 정확성·품질로 판정한다 — ① 규칙과 ③·④ judge·rubric을 도메인별로 섞는다.

알고리즘 둘이 독특하다.

- **GrouPER**: 한 쿼리에 네 개 응답을 뽑아 도메인별 reward로 채점하고, 그 점수로 **그룹 상대 advantage**를 만들어 **SimPER 계열의 선호 최적화 목적함수**에 넣는다. 규칙 reward와 rubric 기반 생성형 reward를 함께 쓴다.
- **AGAPO**: off-policy policy-gradient(truncated importance sampling)인데, 핵심은 **틀린 답에서도 학습 신호를 뽑는다**는 것이다 — 오답으로 이어진 잘못된 추론 경로에 **음의 reward(penalty)**를 매겨, 모델이 스스로 논리 오류를 피하도록 유도한다.

대부분의 RLVR이 "정답이면 +, 오답이면 0"으로 오답을 그냥 버리는 데 반해, AGAPO는 **오답의 잘못된 부분을 명시적으로 벌한다**는 점에서 신호를 더 촘촘하게 쓴다. 안전성은 별도의 safety-aware preference 최적화 단계로 다루며, 296개 위험 영역 분류 체계를 쓴다.

## A.X K2: 네 그룹으로 나누고, 안전은 "거절"이 아니라 "안전한 완수"를 보상한다

A.X K2(2026, SKT)는 학습 mixture를 **네 그룹 — instruction following, human preference, agentic tool use, safety — 으로 나누고**, 그 비율을 미리 고정하지 않고 중간 RL 체크포인트로 모델의 약점을 짚어가며 조정하는 "control surface"로 다룬다. 그룹마다 reward 조달처가 다르다.

- **Instruction following**: **규칙 기반 verifiable reward**(형식·길이·필수 표현·스키마 준수 검사) + **난이도 필터링** — 인하우스 소형 모델로 이미 잘 푸는 프롬프트를 걸러내 on-policy 신호를 정보량 큰 사례에 집중시킨다([#33](/blog/2026/reward-model-design/)의 프롬프트 큐레이션 그대로다).
- **Human preference**: **pointwise LLM-as-judge**. 프롬프트마다 강한 외부 모델로 **reference 답안**을 얻어 judge에 few-shot 채점 앵커와 함께 준다. judge는 태스크를 6개 도메인(사실·추론·코딩·추출·창작·개방)으로 분류한 뒤 **도메인별 4축 rubric**(정확성·완결성·명료성·helpfulness)으로 채점하되, **verbosity bias와 reward hacking을 막는 안전장치**를 명시한다 — [#22 Prometheus 2](/blog/2026/prometheus-2/) 계열의 reference 기반 rubric judge(③).
- **Safety**: **거절 자체가 아니라 "안전한 완수"를 보상한다** — 위험 요청이라도 무해한 목적이 살아 있으면 유용한 안전 결과물로 우회하도록 유도한다. principle 기반 rubric으로 채점하고 선호/비선호 응답을 calibration 기준으로 쓴다.

RL 알고리즘은 **CISPO**(MiniMax) — token-level 업데이트가 아니라 importance-sampling 가중치를 클립해, 확률은 낮지만 행동에 결정적인 토큰이 계속 그래디언트에 기여하게 한다 — 에 **GDPO**(Liu et al. 2026)를 얹는다. GDPO는 **여러 reward를 각각 따로 정규화한 뒤 합쳐** 각 신호의 해상도를 보존하고 multi-reward 학습을 안정화한다([#12 ODIN](/blog/2026/odin-disentangled-reward/)의 "신호 분리" 발상을 여러 reward로 일반화한 셈이다). KL penalty는 쓰지 않고 프롬프트당 16 rollout을 뽑으며, verbosity가 심한 데이터엔 group-relative length penalty를 더한다.

# Experiments

## 마스터 비교표

| 모델         | reward 조달처 (4분류)                    | RL 알고리즘                           | 검증 가능 도메인                   | 검증 불가능 도메인                 |
| ------------ | ---------------------------------------- | ------------------------------------- | ---------------------------------- | ---------------------------------- |
| DeepSeek-V4  | ① 규칙 + ④ GRM (도메인별)                | GRPO → on-policy 증류로 통합          | 규칙 검증기: 정답 일치·테스트 통과 | **GRM (rubric-guided)**            |
| Qwen3        | ① 규칙 + ③ reference judge + ② 스칼라 RM | GRPO (reasoning) + General RL         | 규칙 (query-verifier 쌍)           | ③ reference judge + ② 스칼라 RM    |
| Llama 4      | 비공개 (online RL) + 선호 쌍             | lightweight SFT → online RL → DPO     | 명시 안 됨 (난이도 커리큘럼 중심)  | DPO (가볍게) + online RL           |
| Kimi K3      | ① 규칙(51.2M 샌드박스) + ④ Agentic GRM   | 전문가별 RL → MOPD 증류               | 규칙 검증기: 대규모 샌드박스       | Agentic GRM (rubric 생성→scorepad) |
| Solar Open 2 | ① 규칙 + ④ rubric judge + 12전문가→MOPD  | GRPO(token-level) → MOPD 증류         | 규칙 + 실행 검증                   | LLM-as-judge rubric                |
| K-EXAONE 2.0 | ① 규칙 + ③④ rubric·judge (도메인별)      | GrouPER(SimPER류) + AGAPO(off-policy) | 검증 가능 신호                     | LLM-judge·rubric, 오답에 음의 보상 |
| A.X K2       | ① 규칙(+난이도필터) + ③ reference judge  | CISPO + GDPO(멀티리워드 분리정규화)   | 규칙 verifiable(형식·스키마)       | reference judge 6도메인 4축 rubric |

이 표가 이 글의 결론을 압축한다. 왼쪽 도메인(검증 가능)에서는 일곱 모델 모두 ①(규칙)을 포함한다. 오른쪽 도메인(검증 불가능)에서는 판정형 reward(③ reference judge·④ 생성형 GRM·rubric judge)가 **일곱 중 여섯**에서 쓰인다 — 스칼라 RM 하나만 두는 모델은 사실상 사라졌다(Llama 4만 DPO+RM 노선). 나아가 **"도메인 전문가를 따로 키워 증류로 합친다"는 구조가 DeepSeek-V4·Kimi K3·Solar Open 2 세 곳에서 겹치고, Solar 2와 K3는 이름까지 MOPD로 같다.** 이 두 가지가 이전 세대와의 결정적 차이다.

## 공통 수렴점: 검증 가능 도메인은 규칙이 표준이 됐다

수학·코드처럼 정답이 프로그램적으로 판정 가능한 도메인에서, 일곱 모델은 예외 없이 **① 규칙 기반 verifiable reward**를 쓴다. [#21](/blog/2026/deepseek-r1/)에서 짚었듯 이유는 명확하다 — 규칙 검증기는 파라미터가 없는 함수이므로 hacking할 대상 자체가 없고, RM을 학습·재학습하는 비용도 들지 않는다.

DeepSeek-R1(2025-01)이 이 선택지를 대중화한 뒤, 이후 발표된 모델들이 그대로 채택했다는 사실이 흐름을 보여준다. Qwen3는 "rule-based reward가 높은 정밀도로 정오를 판정해 reward hacking을 예방한다"고 명문화했고, Solar Open 2·K-EXAONE 2.0·A.X K2도 수학·코드·형식 검사에 규칙 검증기를 쓴다. Llama 4조차 (reward 신호 자체는 비공개지만) 추론·코딩·수학을 별도 취급하며 난이도 커리큘럼을 그 도메인에 집중한다. **RLVR이 "만능 표준"이라고 단정하기는 여전히 이르지만, 검증 가능한 도메인에서 규칙을 1순위로 두는 것은 사실상 공통 문법이 됐다.**

## 갈라지는 지점, 그리고 GRM의 등장

정답이 없는 도메인(글쓰기, 상담, 안전성 판단)으로 가면 일곱 모델은 서로 다른 답을 낸다. 이전 세대(DeepSeek-R1, Qwen2.5, Llama 3)를 비교했을 때 이 칸의 결론은 "연구 논문은 생성형 judge를 말하지만, 프로덕션 report는 아직 스칼라 RM이나 self-critique에 머문다"였다. **이번 세대에서 그 결론이 바뀐다.**

| 접근                        | 채택 모델                                                | 조달처 | 관련 편                                                                                                     |
| --------------------------- | -------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| 사람 선호 쌍 + DPO          | Llama 4                                                  | 선호   | [#18 DPO](/blog/2026/dpo/)                                                                                  |
| 스칼라 RM                   | Qwen3                                                    | ②      | 2부([#4](/blog/2026/bradley-terry-rethinking/)\~[#9](/blog/2026/rewardbench-2/))                            |
| reference 기반 judge        | Qwen3, A.X K2                                            | ③      | [#22 Prometheus 2](/blog/2026/prometheus-2/)                                                                |
| self-critique rubric (전신) | Kimi K2 (2025)                                           | ④      | [#25 Self-Taught Evaluators](/blog/2026/self-taught-evaluators/), [#28 J1](/blog/2026/j1-thinking-judge/)   |
| generative RM·rubric judge  | **DeepSeek-V4**, **Kimi K3**, Solar Open 2, K-EXAONE 2.0 | ④      | [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/), [#29 Rubrics as Rewards](/blog/2026/rubrics-as-rewards/) |

마지막 두 행이 핵심이다. [#26 DeepSeek-GRM](/blog/2026/deepseek-grm-spct/)이 "judge가 채점 근거를 스스로 생성하고 rubric을 조건으로 받는다"는 아이디어를 연구 단계에서 제안했는데, **같은 팀의 DeepSeek-V4가 그 GRM을 실제 후처리 파이프라인에 넣었다.** Qwen3의 reference-judge(③)까지 합치면, 생성형·조건부 평가가 더 이상 논문 안에만 있지 않다는 것이 이번 세대의 가장 큰 변화다. 연구와 프로덕션 사이의 시차가 눈에 띄게 좁혀졌다.

## 숨은 reward 설계: 프롬프트를 고르는 것도 reward다

reward 설계를 "어떤 함수로 점수를 매기나"로만 보면 절반만 본 것이다. 일곱 모델이 공통으로 공들이는 또 하나의 축은 **어떤 프롬프트에 그 reward를 먹이느냐**다.

| 모델        | 프롬프트 선별·커리큘럼 전략                                                       | 무엇을 노리나                                |
| ----------- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| Qwen2.5     | 응답 점수 **분산이 큰** 쿼리 우선                                                 | 변별력 있는(정보량 큰) 신호만 학습           |
| Qwen3       | reasoning RL에서 쿼리당 rollout 다수 + off-policy 재활용                          | 샘플 효율                                    |
| Llama 4     | pass@k로 hard prompt 선별, advantage 0인 프롬프트 실시간 제거, medium-hard만 유지 | 신호 없는 프롬프트 낭비 제거, 탐색 여지 확보 |
| DeepSeek-V4 | 도메인별 고품질 데이터로 전문가를 따로 육성                                       | 도메인 간 간섭 없이 각 reward에 집중         |
| A.X K2      | 인하우스 소형 모델로 이미 잘 푸는 프롬프트를 걸러냄(난이도 필터)                  | on-policy 신호를 정보량 큰 사례에 집중       |

이것이 [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)가 다룬 "선호 데이터 노이즈" 문제의 실전판이다. 노이즈가 큰(변별력 없는) 프롬프트를 애초에 걸러내면, 같은 reward 함수라도 hacking 여지가 줄고 학습이 안정된다. **reward 함수를 바꾸지 않고도 reward의 질을 끌어올리는 방법이 프롬프트 큐레이션**인 셈이다.

## DPO vs RL 트레이드오프

[#18 DPO](/blog/2026/dpo/)는 reward model과 온라인 RL 루프를 통째로 없애고 선호 쌍에서 바로 정책을 학습하는 방법이었다. 일곱 모델을 이 스펙트럼 위에 놓으면 이렇게 갈린다.

| 위치                          | 모델                                       | 특징                                                             |
| ----------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| DPO를 가볍게 + online RL 중심 | Llama 4                                    | SFT·DPO는 "탐색을 막지 않을 만큼만", 정렬의 무게중심은 online RL |
| DPO → RL 순차                 | Qwen2.5                                    | offline DPO로 큰 방향을 잡고 online GRPO로 미세조정              |
| RL 중심(순수 온라인)          | DeepSeek-V4, Kimi K3, Solar Open 2, A.X K2 | GRPO/CISPO 기반 온라인 RL이 정책 업데이트를 전담(DPO 없음)       |
| off-policy + 선호 최적화      | K-EXAONE 2.0                               | AGAPO(off-policy PG) + GrouPER(SimPER류 그룹 선호)               |

Llama 3가 "DPO만으로 충분"했다면, Llama 4는 한 발 물러나 **"DPO를 너무 세게 걸면 RL의 탐색을 죽인다"**는 반대 방향의 교훈을 얹었다. 오프라인 정렬은 싸지만 모델을 좁은 분포에 가둘 수 있고, 온라인 RL은 비싸지만 탐색을 통해 새 능력을 끌어낸다. 일곱 모델의 선택은 결국 **"오프라인으로 얼마나 다지고, 온라인에 얼마나 탐색을 맡길 것인가"**의 배분 문제로 수렴한다.

## reward hacking 방어: 각자의 방식

[#10](/blog/2026/reward-model-overoptimization/)\~[#13](/blog/2026/warm-weight-averaged-reward/)이 다룬 방어 기법들이 실제로 어떻게 쓰이는지 정리하면 이렇다.

| 모델         | 방어 장치                                                          | 원리                                                                                        |
| ------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| DeepSeek-V4  | 규칙 우선 + hard-to-verify에만 GRM, on-policy 증류                 | 파라미터 없는 규칙으로 hacking 표면을 최소화, GRM은 근거 생성으로 판정을 검증 가능하게      |
| Qwen3        | rule-based reward를 "hacking 예방" 목적으로 명시                   | 검증 가능한 곳은 규칙으로 못박아 RM 근사 오차 자체를 없앰                                   |
| Llama 4      | advantage 0 프롬프트 제거, medium-hard 커리큘럼                    | 신호 없는 구간에서의 편법 학습 차단, 탐색을 유의미한 난이도에 집중                          |
| Kimi K3      | budget 기반 verbosity 제어                                         | Agentic GRM이 장황한 출력으로 hacking되는 걸 막으려 초기 길이의 σ배를 넘는 후보를 자동 탈락 |
| Solar Open 2 | MOPD를 KL-only(outcome reward 없음)로                              | reward를 아예 안 얹어 증류 단계의 hacking 표면을 제거                                       |
| A.X K2       | GDPO(reward별 분리 정규화) + judge의 anti-verbosity·length penalty | 여러 reward를 섞을 때 한 신호가 다른 신호를 잡아먹는 것과 장황함 편향을 차단                |

표현은 달라도 원리는 하나로 수렴한다 — **reward 신호와 "정책이 참조점에서 얼마나 벗어났는가"를 분리해서 다룬다.** DeepSeek·Solar는 KL을 손실 안에서 명시적으로 떼어놓고, Qwen·Llama는 애초에 노이즈가 크거나 신호가 없는 데이터를 걸러낸다. [#11 Length Correlations](/blog/2026/rlhf-length-correlations/)가 지적한 "성능 향상처럼 보이지만 실은 길이·문체 편향"이라는 함정을, 일곱 모델 모두 나름의 방식으로 피해 가려 한 흔적이다.

## 일반 능력 reward vs 안전성 reward: 프론티어는 둘을 나눈다

지금까지는 "능력(정확성·추론·helpfulness)"을 어떻게 보상하나를 봤다. 그런데 여러 모델이 **안전성(safety) reward를 능력 reward와 명시적으로 분리해** 설계한다. 이 분리는 [#8 Llama 2](/blog/2026/llama2-rlhf/)가 helpfulness RM과 safety RM을 아예 두 개로 나눈 데서 시작됐고, 이번 세대에도 이어진다.

| 모델                                         | 일반 능력 reward                         | 안전성 reward (별도 설계)                                       |
| -------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| A.X K2                                       | 규칙 verifiable + reference rubric judge | **"안전한 완수"를 보상**(거절 자체가 아니라) + principle rubric |
| K-EXAONE 2.0                                 | 도메인별 규칙·rubric·judge               | **별도 safety-aware preference 단계** + 296개 위험 영역 분류    |
| DeepSeek-R1 ([#21](/blog/2026/deepseek-r1/)) | 규칙(RLVR)                               | all-scenario 단계의 helpfulness·harmlessness RM                 |
| Llama 2 ([#8](/blog/2026/llama2-rlhf/))      | helpfulness RM                           | **별도 safety RM** (분리의 원형)                                |

두 가지가 눈에 띈다.

- **안전성은 "규칙으로 검증 불가능"한 대표 도메인이다.** "이 응답이 안전한가"는 정답이 프로그램으로 나오지 않으므로, 능력 도메인이 규칙(①)으로 수렴한 것과 달리 안전성은 대부분 **rubric·judge(③④)** 로 간다.
- **최신 설계의 핵심은 "거절이 아니라 안전한 완수를 보상"하는 것이다.** A.X K2가 명시하듯, 안전성 reward를 "거절하면 +"로 짜면 모델이 무해한 요청까지 거절하는 **over-refusal**로 hacking된다. 그래서 위험 요청이라도 무해한 목적이 살아 있으면 유용한 안전 결과물로 우회하도록 보상을 설계한다 — 안전성 reward 자체가 [#11 Length Correlations](/blog/2026/rlhf-length-correlations/)식 편향(여기선 "거절 편향")에 hacking당하지 않게 짜는 것이다.

즉 프론티어 모델의 reward 설계는 **능력 축과 안전 축을 서로 다른 조달처·서로 다른 hacking 방어로 이원화**한다. 능력은 규칙으로 수렴하지만, 안전은 "무엇이 안전한가"를 rubric으로 명문화하고 그 rubric이 over-refusal로 새지 않게 다시 방어하는, 한 겹 더 복잡한 설계다.

# Conclusion

31편에 걸쳐 쌓아온 재료를 일곱 개 프론티어 모델에 겹쳐보면 이렇게 정리된다.

1. **검증 가능한 도메인은 규칙으로 수렴한다.** 일곱 모델 모두 수학·코드에서 ① 규칙 기반 verifiable reward를 1순위로 둔다. [#21](/blog/2026/deepseek-r1/)의 RLVR이 하나의 논문에서 그친 아이디어가 아니라 공통 문법이 됐다.
2. **검증 불가능한 도메인에서 생성형 reward가 프로덕션에 진입했다.** DeepSeek-V4의 GRM과 Kimi K3의 Agentic GRM이 [#26](/blog/2026/deepseek-grm-spct/)·[#29](/blog/2026/rubrics-as-rewards/)의 연구 아이디어(judge가 원칙·rubric을 스스로 생성)를 실제 학습에 얹었고, DeepSeek-V4·Kimi K3·Solar Open 2 세 모델이 "전문가를 따로 키워 증류로 합친다"는 구조(뒤 둘은 이름까지 MOPD)까지 공유한다. 이전 세대 비교에서 "생성형 judge는 아직 논문 안에만 있다"던 결론은 이번 세대에서 갱신된다.
3. **reward 설계는 함수 선택 + 프롬프트 선택 + 오프라인/온라인 배분의 삼중 문제다.** Qwen의 분산 선별, Llama 4의 난이도 커리큘럼, DeepSeek-V4의 도메인 분리는 모두 "함수를 안 바꾸고 reward의 질을 올리는" 설계였다.
4. **reward hacking 방어는 형태가 달라도 원리는 하나다.** reward 신호와 참조점 이탈 신호를 분리한다는 것 — DeepSeek·Solar의 손실 내 KL, Qwen·Llama의 데이터 큐레이션 모두 [#10](/blog/2026/reward-model-overoptimization/)\~[#12](/blog/2026/odin-disentangled-reward/)이 정량화한 문제에 대한 실무적 응답이다.
5. **능력 reward와 안전성 reward는 서로 다른 축으로 설계된다.** [#8 Llama 2](/blog/2026/llama2-rlhf/)가 시작한 helpfulness·safety 분리가 A.X K2·K-EXAONE 2.0에서 이어진다. 특히 안전성은 "거절이 아니라 안전한 완수를 보상"해 over-refusal hacking을 피하는, 한 겹 더 복잡한 설계다.

이 시리즈를 여는 [#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)은 "사람의 선호로 보상 함수를 배울 수 있는가"라는 질문 하나로 시작했다. 일곱 프론티어 모델을 지나 도착한 답은, 그 질문이 하나의 답을 갖지 않는다는 것이다. 정답이 있으면 규칙이 reward가 되고, 정답 예시가 있으면 reference judge가, 기준은 있지만 예시가 없으면 생성형 RM이, 그조차 흐릿하면 사람의 선호 쌍이 reward를 대신한다.

**reward 설계는 하나의 정답을 찾는 문제가 아니라, 내가 가진 도메인이 이 스펙트럼의 어디에 있는지를 정확히 진단하는 문제였다.** 그 진단을 실제 설계 절차로 옮기는 방법은 다음 글 [#33 reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/)에서 한 장의 체크리스트로 정리한다.

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

32. **(현재 글)** 프론티어 모델의 reward 설계 (2025~2026) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것
33. [reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/) — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 33편으로 구성된다.

# 참고 문헌

- DeepSeek-AI, 2026. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348).
- DeepSeek-AI, 2025. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — [#21](/blog/2026/deepseek-r1/)에서 다룬 RLVR 원형.
- Shao et al. (DeepSeek-AI), 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — GRPO 원 논문.
- Qwen Team, 2025. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
- Meta AI, 2025. [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- Meta AI, 2024. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — 6라운드 rejection sampling + DPO, PPO 미사용(비교 기준).
- Kimi Team, 2026. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3) — 3단계(SFT→RL→MOPD) + Agentic GRM(rubric 생성) + reasoning-effort budget RL.
- Kimi Team, 2025. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — self-critique rubric reward를 도입한 전신.
- Upstage AI, 2026. [Solar Open 2 Technical Report](https://arxiv.org/abs/2607.20062) — 12 전문가 → MOPD, GRPO token-level.
- Upstage AI, 2026. [Solar Open Technical Report](https://arxiv.org/abs/2601.07022) — 전신(v1).
- LG AI Research, 2026. [K-EXAONE 2.0 Technical Report](https://arxiv.org/abs/2608.04505) — GrouPER(SimPER류) + AGAPO(오답에 음의 보상).
- LG AI Research, 2026. [K-EXAONE Technical Report](https://arxiv.org/abs/2601.01739) — 전신(1.0), AGAPO/GrouPER 상세.
- SKT AI, 2026. [A.X K2 Technical Report](https://github.com/SKT-AI/A.X-K2) — 4그룹 reward + reference rubric judge + CISPO/GDPO.
- Liu et al., 2026. [GDPO: Group Reward-Decoupled Normalization Policy Optimization](https://arxiv.org/abs/2601.05242) — A.X K2가 채택한 멀티리워드 정규화.
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — [#18](/blog/2026/dpo/)에서 다룬 DPO 원 논문.
- Kim et al., 2024. [Prometheus 2](https://arxiv.org/abs/2405.01535) — [#22](/blog/2026/prometheus-2/) 참고.
- Liu et al. (DeepSeek-AI), 2025. [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) — [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/) 참고.
