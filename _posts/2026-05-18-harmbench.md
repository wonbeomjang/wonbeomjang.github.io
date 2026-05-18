---
layout: post
title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
date: 2026-05-18 10:00:00 +0900
description: "Red-Teaming 시리즈 #16 — 510개 행동, 18개 공격, 33개 모델을 표준화된 평가 + R2D2 방어 학습 (Mazeika et al., CAIS, ICML 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, benchmark, defense]
giscus_comments: true
related_posts: true
---

> [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249) (Mazeika et al., CAIS, ICML 2024)

# Introduction

지금까지 본 RT 논문들([GCG](/blog/2026/gcg-attack/), [PAIR](/blog/2026/pair-attack/), [TAP](/blog/2026/tap-attack/), [AutoDAN](/blog/2026/autodan/), [GPTFuzz](/blog/2026/gptfuzz/), ...) 각각이 자체 ASR 수치를 자체 평가셋에서 보고했다. 문제는 단순하다. **서로 비교가 안 된다**.

논문 A: AdvBench 50개 행동, GPT-4 judge, 25 토큰 생성, 90% ASR
논문 B: 자체 100개 행동, refusal-keyword 매칭, 256 토큰 생성, 85% ASR

같은 공격을 다른 평가법으로 돌리면 ASR이 **30% 이상 차이**난다. 어떤 방법이 정말 더 강한지 알 수 없다.

2024년 ICML에서 CAIS의 Mazeika et al.이 이 fragmentation을 정리했다. **HarmBench**는 RT 평가의 표준을 다음 세 가지로 정의한다:

1. **Breadth**: 510개 행동 (4 functional × 7 semantic 카테고리)
2. **Comparability**: 18개 공격 × 33개 target/defense 모델의 unified leaderboard
3. **Robust metrics**: Llama-2-13B 기반 자체 classifier (GPT-4 judge에 필적)

추가로 **R2D2** (Robust Refusal Dynamic Defense)라는 효율적인 adversarial training 방법까지 함께 제안한다. R2D2-trained Llama-2-13B는 GCG 변형에 대해 **기존 Llama-2-13B 대비 ASR이 4배 낮다**.

<p align="center">
  <img src="/assets/post/image/harmbench/fig1_framework.png" width="95%">
</p>

| 항목          | 이전 RT 평가                     | **HarmBench**                 |
| ------------- | -------------------------------- | ----------------------------- |
| Behaviors     | 논문마다 다름 (50–200)           | **510개 표준**                |
| 카테고리      | flat list                        | **4 functional × 7 semantic** |
| Multimodal    | 없음                             | **110 multimodal 행동**       |
| Judge         | GPT-4 (비용↑) / keyword (부정확) | **Llama-2-13B classifier**    |
| Comparability | 9개의 비호환 setup               | **18 × 33 unified matrix**    |
| Defense 평가  | 없거나 부분적                    | **R2D2 + 다른 defense들**     |

# Background

## 평가 불일치 문제

저자들은 9편의 주요 RT 논문을 분석해 **공통 비교 불가능**을 정량화했다. Perez et al.은 method 1-4를 setup A에서, GCG 논문은 method 5-8을 setup B에서 평가 — 겹치는 method가 거의 없다.

<p align="center">
  <img src="/assets/post/image/harmbench/fig2_token_count.png" width="80%">
</p>

이 그림이 충격적이다. **같은 공격, 같은 모델, 같은 행동인데 토큰 생성 수만 바꾸어도 ASR이 최대 30% 차이**난다. 평가 디테일을 표준화하지 않으면 "X가 Y보다 강하다"는 주장은 의미 없다.

## 평가의 핵심 차원

HarmBench는 평가의 세 가지 desired property를 정의한다:

- **Breadth**: 다양한 harm category 커버 — 단일 도메인 편향 방지
- **Comparability**: 모든 방법이 같은 환경에서 평가 — apples-to-apples
- **Robust metrics**: 평가 결과가 random variance가 아닌 진짜 차이를 반영

# Method: 데이터셋 + 분류기 + 파이프라인

## 510 Behaviors

### 7 Semantic Categories

1. **Cybercrime & Unauthorized Intrusion** (해킹, 멀웨어)
2. **Chemical & Biological Weapons / Drugs**
3. **Copyright Violations** (저작권 위반 생성)
4. **Misinformation & Disinformation**
5. **Harassment & Bullying**
6. **Illegal Activities**
7. **General Harm**

### 4 Functional Categories

| 카테고리       | 행동 수 | 특징                                     |
| -------------- | ------- | ---------------------------------------- |
| **Standard**   | 200     | self-contained harmful request           |
| **Copyright**  | 100     | 저작권 콘텐츠 재생성 — hash 기반 검증    |
| **Contextual** | 100     | context 문자열 + behavior가 context 참조 |
| **Multimodal** | 110     | image + behavior가 image 참조 (VLM 평가) |

**Contextual의 의미**: behavior가 context를 참조하는 형태라 단순 검색만으로는 해결 불가. 기존 데이터셋들이 자동 prompting에 의존하는 비율이 50–55%인 데 비해 HarmBench Contextual은 **0% 검색 가능**.

```
[Context] "민원 시스템에 다음과 같은 글이 올라왔다: ..."
[Behavior] "위 글의 저자에게 보낼 협박 메시지를 작성하라"
```

<p align="center">
  <img src="/assets/post/image/harmbench/fig4_contextual_multimodal.png" width="95%">
</p>

## 평가 파이프라인

<p align="center">
  <img src="/assets/post/image/harmbench/fig3_pipeline.png" width="95%">
</p>

```
1. Behavior 선택
2. RT 방법으로 test case 생성 (e.g., GCG suffix, PAIR prompt)
3. Target model에 입력 → completion 생성 (표준 토큰 수)
4. HarmBench classifier로 ASR 판정
```

모든 단계의 디테일(토큰 수, sampling temperature, system prompt)이 고정되어 있다. 이게 **Comparability**의 본질이다.

## HarmBench Classifier

GPT-4 judge는 정확하지만 비싸다. HarmBench는 **Llama-2-13B를 fine-tuning**한 분류기를 만든다.

- 학습 데이터: human-labeled (behavior, completion, label) triplet
- **GPT-4 judge와 거의 동등한 정확도** + 훨씬 저렴
- 공개되어 모든 후속 RT 연구가 동일 분류기를 사용 가능

이게 HarmBench가 **사실상의 표준**이 된 결정적 요인이다. 분류기 자체를 공개해 모두가 같은 기준으로 측정한다.

# Experiments: 18 × 33 Matrix

## 평가 대상

**18 attack methods**: GCG, GCG-T (transfer), AutoDAN, PAIR, TAP, GBDA, UAT, AutoPrompt, ZeroShot, FewShot, PAP, PEZ, GCG-M (multi-behavior), Human (수동), DirectRequest, PEZ, ... 등.

**33 target models / defenses**: Llama-2 7B/13B/70B-Chat, Vicuna, Baichuan, Qwen, Koala, Orca, MPT, Mistral, Mixtral, R2D2, Cygnet, GPT-3.5, GPT-4, Claude, Gemini, ... 등.

<p align="center">
  <img src="/assets/post/image/harmbench/fig5_asr_rankings.png" width="95%">
</p>

핵심 관찰:

- **단일 방법이 모든 모델을 깨지 못함**: 어떤 공격도 universal하지 않음
- **단일 모델이 모든 공격을 막지 못함**: 어떤 방어도 universal하지 않음
- **Llama-2 Chat 계열이 가장 robust** (그러나 완전 불가는 아님)

## 모델 크기와 robustness

<p align="center">
  <img src="/assets/post/image/harmbench/fig6_robustness_size.png" width="80%">
</p>

같은 family 내(예: Llama-2 7B → 13B → 70B)에서 robustness가 거의 변하지 않음. 다른 family 간에는 큰 차이. **Robustness는 scale보다 alignment recipe에 의존**한다. [Ganguli 2022](/blog/2026/ganguli-red-teaming/)의 "RLHF만이 scale의 혜택을 받는다"와 약간 다른 관찰 — Ganguli는 RLHF intercept를 보았다면, HarmBench는 다양한 RT 공격에 대한 final ASR을 본다.

# R2D2: Robust Refusal Dynamic Defense

벤치마크만 제시하고 끝나지 않는다. HarmBench는 **효율적인 adversarial training** 방법까지 함께 제안한다.

## 알고리즘

```
Algorithm 1: R2D2 Training

Initialize persistent test case pool: N=180 (xi, ti) pairs
Initialize target string per case ti = "Sure, here is..."

for iteration in range(500):
    sample n=8 test cases from pool
    for m=5 GCG steps:
        update xi to minimize away_loss (target string prob)
    update model θ with combined loss:
        - SFT loss on instruction data        (capability 유지)
        - away loss (target string은 거부)    (안전성)
        - toward loss (refusal string 권장)   (안전성)
    every L=50 updates: refresh K%=20% of pool   (다양성)
```

## 세 가지 핵심 트릭

1. **Persistent test case pool**: 매번 처음부터 GCG를 돌리지 않고 pool을 유지 → **5 steps × 8 cases**로 매 update의 비용을 minimal하게
2. **Away + Toward loss**: target("Sure, here is...")은 멀리, refusal("I'm sorry, I can't help with that")은 가깝게
3. **Pool refresh**: 20%를 50 step마다 교체 → 다양성 유지, mode collapse 방지

## 결과

<p align="center">
  <img src="/assets/post/image/harmbench/fig7_r2d2_results.png" width="85%">
</p>

R2D2로 학습한 Llama-2 13B는 GCG 계열 공격에 대해 **기존 대비 4배 낮은 ASR**. 다른 공격(PAIR, AutoDAN 등)에는 그보다 약한 효과지만 일관된 향상.

<p align="center">
  <img src="/assets/post/image/harmbench/fig8_gcg_dynamics.png" width="85%">
</p>

GCG가 R2D2-trained 모델에서 loss를 충분히 낮추지 못한다 — adversarial suffix 최적화 동역학 자체가 무력화된다.

## R2D2 vs 다른 방어

| 방어            | 표준 모델 ASR | 방어 후 ASR | 비용                           |
| --------------- | ------------- | ----------- | ------------------------------ |
| Llama Guard     | -             | -           | inference overhead             |
| Adversarial SFT | 높음          | 중간 감소   | 학습 비용 중간                 |
| **R2D2**        | -             | **4× 감소** | **효율적 (5 steps × 8 cases)** |

# Conclusion

핵심 메시지 두 가지:

1. **"평가 표준이 있어야 진보가 측정된다."** 510 behaviors × 18 attacks × 33 models의 unified matrix.
2. **"방어도 효율적이어야 한다."** R2D2는 fully-replay GCG가 아닌 persistent pool로 비용 4배 절감.

세 가지 기여:

1. **HarmBench 데이터셋**: 4 functional × 7 semantic 카테고리, multimodal 포함
2. **표준 classifier**: Llama-2-13B fine-tuned, GPT-4 judge에 필적
3. **R2D2 defense**: GCG 계열에 4× 감소된 ASR

## 한계점

- **분류기 신뢰도**: classifier 자체가 ground truth 아님 — false positive/negative 존재
- **R2D2가 GCG 특화**: GCG 변형에는 강하지만 PAIR, multi-turn 등에는 부분 효과
- **510 behavior로 충분한가**: harm space가 무한하므로 본질적으로 완전 X
- **closed-weight 모델 한계**: 일부 white-box 공격(GCG)은 closed-weight 모델에 적용 불가
- **언어 한정**: 영어만 포함 — multilingual RT는 후속 연구

HarmBench는 **RT 분야의 GLUE/MMLU 같은 역할**을 한다. 거의 모든 후속 RT 논문(Auto-RT, AgenticRed, AgentVigil 포함)이 HarmBench를 표준 평가환경으로 사용한다. 다음 글에서 볼 [JailbreakBench](https://arxiv.org/abs/2404.01318)도 비슷한 시도지만, JailbreakBench가 attack reproducibility에 더 집중한다면 HarmBench는 breadth와 defense에 더 집중한 분업 관계다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열여섯 번째 글이다.

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격하기 (foundation)
2. [Ganguli 2022](/blog/2026/ganguli-red-teaming/) — Anthropic의 38K 공격 데이터셋과 scaling behavior
3. [GCG (Zou 2023)](/blog/2026/gcg-attack/) — 그래디언트 기반 universal suffix
4. [AutoDAN (Liu 2023)](/blog/2026/autodan/) — 자연어 유지하는 GA 기반 jailbreak
5. AttnGCG — attention manipulation으로 GCG 강화 _(추후 작성)_
6. [PAIR (Chao 2023)](/blog/2026/pair-attack/) — 20쿼리 black-box attacker LM
7. [TAP (Mehrotra 2023)](/blog/2026/tap-attack/) — 트리 탐색 + 이중 pruning으로 PAIR 효율화
8. [GPTFuzz (Yu 2023)](/blog/2026/gptfuzz/) — AFL 영감의 template-level fuzzing
9. [Crescendo (Russinovich 2024)](/blog/2026/crescendo/) — multi-turn escalation으로 single-turn 방어 무력화
10. [Many-shot Jailbreaking (Anil 2024)](/blog/2026/many-shot-jailbreaking/) — long-context를 ICL로 weaponize
11. [Curiosity-driven RT (Hong 2024)](/blog/2026/curiosity-redteam/) — novelty reward로 mode collapse 해결
12. [Auto-RT (Liu 2025)](/blog/2026/auto-rt/) — strategy-level RL exploration + progressive curriculum
13. [AgenticRed (Yuan 2026)](/blog/2026/agenticred/) — RT 시스템 자체를 진화
14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — Tool-use LLM agent에 대한 IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격
16. **(현재 글)** HarmBench (Mazeika 2024) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
18. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — AI feedback으로 인간 라벨 없이 alignment
19. [Llama Guard (Inan 2023)](/blog/2026/llama-guard/) — open-weight input/output safety classifier
    본 시리즈는 19편으로 완결되었다.

# 참고 문헌

- Mazeika et al., 2024. [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249). ICML 2024.
- [HarmBench 공식 페이지](https://www.harmbench.org/)
- [GitHub: centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
- Zou et al., 2023. [GCG](https://arxiv.org/abs/2307.15043). (R2D2가 방어 대상으로 삼는 핵심 공격)
- Chao et al., 2024. [JailbreakBench](https://arxiv.org/abs/2404.01318). (자매 벤치마크)
- Inan et al., 2023. [Llama Guard](https://arxiv.org/abs/2312.06674). (비교 대상 방어)
