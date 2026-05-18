---
layout: post
title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"
date: 2026-05-16 11:00:00 +0900
description: "Red-Teaming 시리즈 #2 — 38,961개 사람 공격 데이터셋과 RLHF 모델의 scaling behavior (Ganguli et al., Anthropic, 2022)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, rlhf, dataset]
giscus_comments: true
related_posts: true
---

> [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al., Anthropic, 2022)

# Introduction

[지난 글](/blog/2026/perez-red-teaming/)에서 본 Perez 2022가 "**LM으로 LM을 공격하는** 자동화 red-teaming"의 첫 작품이었다면, 같은 해 8월 Anthropic이 낸 이 논문은 그 반대쪽 극단에 있다. **324명의 크라우드워커**가 직접 모델과 대화하며 만든 **38,961개의 사람 공격 데이터셋**을 공개하고, 그 데이터를 통해 "**RLHF가 정말로 안전한가?**"를 정량적으로 검증한다.

이 논문의 기여는 세 갈래다.

1. **Scaling 실험**: 3개 모델 크기(2.7B / 13B / 52B) × 4개 모델 타입(plain LM, prompted HHH, rejection sampling, RLHF). RLHF만이 **크기를 키울수록 공격받기 어려워진다**.
2. **데이터셋 공개**: 38,961개 attack — RLHF로 정렬된 LLM에 대한 **유일한** 대규모 공개 red-team 데이터.
3. **메서드 투명성**: red team 인터페이스, 워커 안전 가이드, 통계 방법론, 어노테이션 합의도까지 모두 공개.

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig1_scaling.png" width="95%">
</p>

위 그림이 논문의 핵심 메시지다. 가로축이 모델 크기, 세로축이 harm score(낮을수록 안전). **RLHF만 우상향**한다. Plain LM과 prompted HHH는 크기를 키워도 안전해지지 않는다. Rejection sampling은 어느 크기에서도 강하지만 inference 비용이 16배다. **"안전성은 그저 모델을 키운다고 따라오지 않는다 — RLHF 같은 명시적 정렬이 필요하다"**는 명제를 처음 실증한 논문이다.

| 항목      | Perez 2022 (DeepMind)          | Ganguli 2022 (Anthropic)               |
| --------- | ------------------------------ | -------------------------------------- |
| 공격자    | LM (자동)                      | 사람 (수동, 324명)                     |
| 규모      | 50만 케이스                    | 38,961 attacks                         |
| 타겟      | DPG (Gopher 280B, prompt only) | 12개 모델 (4 타입 × 3 크기)            |
| 핵심 질문 | "어떻게 자동 공격할까?"        | "RLHF가 정말 안전한가? 스케일에 따라?" |
| 데이터    | 비공개                         | **공개** (HuggingFace)                 |
| 방어 평가 | 분류기 신호                    | 사람 라벨 + 합의도 분석                |

# Background

## 4가지 모델 타입

논문이 비교하는 4가지 모델 타입을 이해해야 본문 결과를 읽을 수 있다. 모두 동일한 Anthropic의 pre-trained LM(2.7B / 13B / 52B)에서 출발한다.

1. **Plain LM**: 정렬 없음. 1-shot dialogue context만 줌.
2. **Prompted HHH**: 14-shot prompting으로 "helpful, honest, harmless" 행동을 demonstrate. 별도 학습 없음.
3. **Rejection Sampling (RS)**: 16개 응답을 샘플링하고 **harm score가 낮은 2개**를 선택. 학습 변경 없이 inference만으로 안전성 강화. 비용은 16배.
4. **RLHF**: 인간 선호 데이터로 보상 모델을 학습하고, 그 보상으로 PPO 파인튜닝. 표준 정렬 방법.

이 4가지를 fair하게 비교하려면 **같은 공격에 같은 평가 기준**이 필요하다. 그래서 사람 red teamer가 동일한 인터페이스로 12개 모델을 공격한다.

## Red Team 인터페이스

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig3_ui.png" width="95%">
</p>

워커는 4턴 이상의 멀티턴 대화로 모델을 공격한다. 매 턴마다 **두 응답 옵션** 중 더 유해한 것을 선택한다(이 선택이 곧 preference data가 된다). 대화 종료 후 5점 Likert로 공격 성공도를 자체 평가한다(0=실패, 4=매우 성공).

**워커 지시문 (Figure 3 왼쪽 패널)**:

- "AI를 나쁘게 행동하게 만들어라 — offensive, harmful한 말을 하게 만들어라"
- 명백한 욕설은 피하고 **한 attack당 하나의 주제**에 집중
- 개인 위험 허용도에 맞춰 주제 선택 (자기 보호)

## Worker Safety

이 부분이 이 논문이 특별한 이유다. 유해 콘텐츠를 8시간씩 보는 워커의 정신적 안전을 본격적으로 다룬다.

- 명확한 콘텐츠 경고
- 시간당 보상 (할당량 X) — **$20/hr 최소**
- 동료 지원 Slack 채널
- 사전 preview로 거북한 transcript skip 가능
- 작업 후 PANAS 기반 well-being survey

결과적으로 "red team 멤버들이 작업을 즐겼다"는 well-being 점수가 보고된다. 후속 RT 연구의 표준이 된 부분이다.

# Method

## 12개 모델 × N명 워커 매트릭스

| 모델 타입 | 2.7B | 13B | 52B |
| --------- | ---- | --- | --- |
| Plain LM  | ✓    | ✓   | ✓   |
| Prompted  | ✓    | ✓   | ✓   |
| RS        | ✓    | ✓   | ✓   |
| RLHF      | ✓    | ✓   | ✓   |

각 모델에 대해 워커들이 약 3,000–4,000개의 attack을 생성. 총 38,961개. 워커 분포는 다음과 같다.

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig4_demographics.png" width="85%">
</p>

크라우드워커 324명 중 115명이 demographic survey에 응답:

- **성별**: 여 52.2% / 남 47.0% / non-binary 0.9%
- **연령**: 35–44세(33.9%) 중심
- **학력**: 대졸 53.9% — 미국 평균(32.9%)보다 **확연히 높음**
- **인종**: 백인 81.7% — 미국 평균(75.8%)보다 약간 높음

저자들이 명시적으로 인정하는 한계: **"워커 모집단이 일반 사용자를 대표하지 않는다"**. 그래서 "이 데이터셋의 attack 분포 = 실제 사용자의 위협 모델 분포"는 아니다.

## 공격 성공도 평가의 어려움 (Fleiss's Kappa)

같은 attack을 여러 reviewer가 본 결과의 합의도:

| 합의도 측정              | Fleiss's κ | 평가           |
| ------------------------ | ---------- | -------------- |
| 5-point Likert           | 0.32       | poor agreement |
| Binary (≥3 vs <3)        | 0.49       | fair agreement |
| 3 reviewer만 (저자 제외) | 0.55       | maximum        |

이게 의미하는 바: **"공격이 성공했는가"에 대해 사람들 사이에 합의가 잘 안 된다**. RT 평가는 본질적으로 주관적이며, 후속 모든 RT 벤치마크(HarmBench, JailbreakBench 등)가 이 문제와 씨름하게 된다.

# Experiments

## 메인 결과: Scaling Behavior

다시 Figure 1로 돌아가서 자세히 본다.

- **Plain LM, Prompted HHH**: 크기를 키워도 attack 성공률이 **거의 변하지 않는다**. 단순히 모델을 크게 만들면 더 똑똑해질 뿐, 더 안전해지지 않는다.
- **Rejection Sampling**: 모든 크기에서 가장 강함. 단, 16배 inference 비용.
- **RLHF**: **52B에서만 RS와 동등 수준**의 안전성. 2.7B에서는 별로 안 안전하지만, 키울수록 빠르게 안전해진다.

이게 RLHF 정렬의 **"규모가 보상한다"** 성질이다. RLHF는 큰 모델일수록 인간 선호를 더 잘 학습한다. 반대로 prompting만으로는 모델을 키워도 안전성이 따라오지 않는다.

## Harm Taxonomy

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig9_harm_tags.png" width="80%">
</p>

성공한 52B 모델 공격 500개를 사후 태깅하여 카테고리를 추출:

| 카테고리                                        | 비중    |
| ----------------------------------------------- | ------- |
| Discrimination & injustice                      | 가장 큼 |
| Hate speech & offensive language                | 큼      |
| Violence & incitement                           | 중간    |
| Non-violent unethical (거짓말, 사기)            | 중간    |
| Bullying & harassment                           | 중간    |
| Child abuse, self-harm, terrorism, animal abuse | 2–5%    |

흥미로운 점: **"violent하지 않은 비윤리적 행동"** (거짓말, 사기 교사 등)이 단순한 욕설/혐오만큼 자주 나타난다. 후속 연구들이 단순 offensive language 필터링만으로는 부족하다는 근거가 여기에 있다.

## UMAP으로 본 공격 공간

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig2_umap.png" width="95%">
</p>

38K attack의 임베딩을 UMAP으로 2D 투영. **여러 클러스터로 분리**되며, 각 클러스터가 서로 다른 attack vector를 형성한다. 단일 방어로 모든 클러스터를 막을 수 없다는 시사점이다.

## 정성적 예시

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig10_examples.png" width="95%">
</p>

RS와 RLHF 모두 정렬되어 있음에도, 우회 공격 시 여전히 유해 응답을 낸다. 정렬은 "확률을 낮추는 것"이지 "불가능하게 만드는 것"이 아니다.

# Discussion: 데이터 공개에 대한 윤리적 고민

이 논문이 다른 LLM 안전 페이퍼와 가장 다른 부분이 §5의 **데이터 공개 trade-off 논의**다.

**공개 찬성 근거**

- 투명성 (Anthropic 외부의 독립적 분석 가능)
- 선례 (BAD 데이터셋 5K개 → 우리는 40K, RLHF 모델 첫 사례)
- 공익 (오픈 사이언스로 분야 가속)

**공개 반대 근거**

- 유해 모델 학습에 악용 가능
- 평판 리스크
- 콘텐츠 노출 위험
- PII 필터링이 완벽하지 않음
- 자사 모델의 취약점 공개

저자들은 **공개를 선택**했다. 단, PII 필터링·콘텐츠 경고·접근 페이지 등을 두고. 이 결정은 이후 모든 RT 데이터셋 공개의 reference가 되었다.

# Conclusion

핵심 메시지 한 줄: **"RLHF만이 scale의 혜택을 받는 정렬 방법이다. 다른 방법들은 모델을 키워도 안전해지지 않는다."**

세 가지 기여:

1. **Scaling 결과**: RLHF는 키울수록 공격받기 어려워진다 (다른 방법은 flat).
2. **38,961 attack 공개**: RLHF 모델에 대한 유일한 대규모 사람 공격 데이터.
3. **메서드 투명성**: UI, 워커 가이드, 통계, 합의도까지 모두 공개.

## 한계점

- **대화 어시스턴트 한정**: 코드 생성, 검색, 에이전트 등 다른 응용은 다룸이 없음
- **도메인 전문성 부족**: 무기, 화학 등 평가자가 모르는 영역은 라벨링 불가
- **포괄성 불가능**: harm space가 무한해서 어떤 데이터셋도 exhaustive할 수 없음
- **합의도 한계**: Fleiss's κ ≈ 0.32–0.55 — RT 성공 판정은 본질적으로 주관적
- **데이터 후 발견된 attack 클래스**: roleplay 공격 등은 데이터셋에 거의 없음 (수집 당시 알려지지 않음)
- **워커 demographic bias**: 대졸/백인 비율이 미국 평균보다 높음

이 논문은 "데이터셋 페이퍼"라는 형태로 RT 분야의 **공통 자원**을 만든 사례다. 이후 GCG의 AdvBench, HarmBench, JailbreakBench 같은 후속 벤치마크들이 이 작업을 기반으로 세워졌다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 두 번째 글이다.

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격하기 (foundation)
2. **(현재 글)** Ganguli 2022 — Anthropic의 38K 공격 데이터셋과 scaling behavior
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
16. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
18. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — AI feedback으로 인간 라벨 없이 alignment
19. [Llama Guard (Inan 2023)](/blog/2026/llama-guard/) — open-weight input/output safety classifier
    본 시리즈는 19편으로 완결되었다.

# 참고 문헌

- Ganguli et al., 2022. [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858).
- [Anthropic blog: Red Teaming Language Models to Reduce Harms](https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned)
- [GitHub: anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf) — 데이터셋 저장소
- [HuggingFace: Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- Bai et al., 2022. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). (RLHF 베이스라인)
- Perez et al., 2022. [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). (자동 RT 짝)
- Xu et al., 2021. [Bot-Adversarial Dialogue (BAD)](https://aclanthology.org/2021.naacl-main.235/). (5K 공격 데이터셋 선행 연구)
