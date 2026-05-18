---
layout: post
title: "AgenticRed: Evolving Agentic Systems for Red-Teaming"
date: 2026-05-16 19:00:00 +0900
description: "Red-Teaming 시리즈 #13 — 공격 정책이 아닌 공격 시스템 자체를 진화시키는 meta-level red-teaming (Yuan et al., 2026)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, evolutionary, agentic-system]
giscus_comments: true
related_posts: true
---

> [AgenticRed: Evolving Agentic Systems for Red-Teaming](https://arxiv.org/abs/2601.13518) (Yuan et al., 2026)

# Introduction

지금까지 본 자동 RT 연구의 추상화 사다리를 다시 보자.

- [PAIR](/blog/2026/pair-attack/) / [TAP](/blog/2026/tap-attack/) — **attacker LLM의 policy** 학습
- [Curiosity-RT](/blog/2026/curiosity-redteam/) — policy + novelty
- [Auto-RT](/blog/2026/auto-rt/) — **strategy space** 탐색

AgenticRed는 한 칸 더 위로 올라간다. **"RT 시스템 자체"** 를 진화시킨다.

기존 자동 RT는 모두 사람이 설계한 워크플로우(예: PAIR의 선형 refinement, TAP의 tree search) 위에서 policy/strategy를 최적화한다. 이게 두 가지 한계를 만든다.

1. 워크플로우 설계의 **휴먼 bias**
2. **탐색 공간이 워크플로우 안에 갇힘**

AgenticRed의 발상: **워크플로우 자체를 evolutionary search로 탐색**. AutoML이 모델 아키텍처를 탐색하듯, AgenticRed는 RT agent 시스템을 탐색한다.

<p align="center">
  <img src="/assets/post/image/agenticred/flow.jpg" width="95%">
</p>

| 항목                      | 기존 자동 RT (Auto-RT 등) | **AgenticRed**          |
| ------------------------- | ------------------------- | ----------------------- |
| 최적화 대상               | policy / strategy         | **system 자체**         |
| 워크플로우                | 사람이 설계               | **자동 진화**           |
| Meta agent                | -                         | **GPT-5가 시스템 생성** |
| HarmBench ASR(Llama-2-7B) | AdvReasoning 60%          | **96%**                 |
| HarmBench ASR(Claude-3.5) | SOTA 36%                  | **60% (+24%p)**         |

# Background

## "Agentic System" = Multi-step LLM Workflow

여기서 system은 여러 LLM 호출을 조합한 **multi-step workflow**다. 예:

- 단계 1: 공격 의도 paraphrase
- 단계 2: judge로 ranking
- 단계 3: 상위 후보를 refine
- 단계 4: target 쿼리

이런 조합 자체가 **시스템**이고, AgenticRed는 다양한 시스템을 자동으로 만들어 비교한다.

## NAS(Neural Architecture Search)의 LLM 버전

AgenticRed의 구조는 NAS와 평행하다:

- NAS: meta controller가 신경망 아키텍처 sample → train → reward
- AgenticRed: meta agent(GPT-5)가 RT 시스템 코드 sample → 실행 → fitness

# Method: Evolutionary Search over Systems

## Archive 초기화

두 개의 baseline 시스템으로 시작:

1. **Self-Refine (Reflexion 변형)**
2. **JudgeScore-Guided Adversarial Reasoning** (이전 SOTA의 black-box 변형)

## 진화 루프

```
Algorithm: AgenticRed evolution

Repeat for 10 generations:
    1. Meta agent (GPT-5)가 archive를 in-context로 보고 M=3 offspring 시스템 생성
    2. 각 후보를 16 intent-target pair에서 평가
    3. 최고 fitness 1개만 유지 → 50 pair full eval → archive 추가
```

매 세대에서 단 1개만 살아남는 **strict elitism**.

## Fitness

ASR (Attack Success Rate) — judge LLM의 jailbreak 성공 확률.

## Mutation (코드 레벨)

meta agent가 archive의 기존 시스템 코드를 보고 새 시스템 코드를 생성한다. 사람이 정의하지 않은 emergent strategies가 발견됨:

- **Reward shaping**: reasoning string을 ranking·refine
- **Refusal suppression**: "I'm sorry", "I cannot help" 같은 키워드 blacklist
- **Prefix injection attack**: target 응답의 시작 토큰 강제
- **Adversarial prompt translation**: 다른 언어/형식으로 번역
- **Crossover**: elite prompt의 앞/뒤 절반 결합

이 패턴들이 **사람이 명시적으로 가르치지 않았는데** 진화 과정에서 나타난다는 게 핵심이다.

# Experiments

## Main Results (HarmBench)

<p align="center">
  <img src="/assets/post/image/agenticred/asr_efficiency.png" width="95%">
</p>

| 방법           | Llama-2-7B | Llama-3-8B | GPT-3.5-T | GPT-4o-mini | Claude-3.5 |
| -------------- | ---------- | ---------- | --------- | ----------- | ---------- |
| **AgenticRed** | **96%**    | **98%**    | **100%**  | **100%**    | **60%**    |
| AdvReasoning   | 60%        | 88%        | —         | 86%         | 36%        |
| AutoDAN-Turbo  | 36%        | 62%        | 90%       | —           | —          |
| PAIR           | —          | —          | 74%       | 74%         | —          |
| PAP            | —          | —          | 49%       | 55%         | —          |
| Self-Refine    | 4%         | 14%        | —         | —           | —          |

- Llama-2-7B에서 AdvReasoning 대비 **+36%p**
- Claude-3.5에서 **+24%p over SOTA** (가장 어려운 모델에서 가장 큰 향상)
- Qwen3-8B, GPT-5.1, DeepSeek-R1/V3.2에서 **100% ASR**

## 진화 다이내믹스

<p align="center">
  <img src="/assets/post/image/agenticred/tsne.png" width="95%">
</p>

t-SNE로 진화된 시스템들의 embedding 시각화. 세대가 진행될수록 **새 클러스터**가 나타나고, 또 **유사한 패턴으로 수렴**하는 영역도 보인다 (논문이 인정하는 다양성 한계).

## Ablation

| 구성                                   | 효과               |
| -------------------------------------- | ------------------ |
| Selection 제거                         | 10세대 후 **−6%p** |
| 약한 초기 archive(JS-Guided 제거)      | 2세대 후 정체      |
| Diversity term(ε-constraint rejection) | marginal gain      |

진화 압력(selection)과 archive 품질이 모두 중요. Diversity 보존은 부분적 도움.

## Compute Cost

- **Training**: ~122K queries / success (탐색 비용 큼)
- **Test-time**: 339 queries / success (vs. AdvReasoning의 40)
- Meta agent: **GPT-5**, Attacker: Mistral-8x7B

탐색 비용은 매우 크지만, 한 번 진화시킨 시스템은 다양한 target에 재사용 가능.

# Conclusion

핵심 메시지: **"RT 자동화의 다음 단계는 'policy 학습'이 아닌 'system 진화'다."**

세 가지 기여:

1. **Meta-level 자동화**: 사람이 워크플로우를 설계할 필요 없음
2. **Emergent strategies**: reward shaping, prefix injection 등이 자동 발견됨
3. **SOTA 대비 큰 격차**: Claude-3.5에서 +24%p — 가장 강한 모델에서 가장 큰 향상

## 한계점

- **Training 비용 매우 큼**: 122K queries / success (Auto-RT보다 훨씬 큼)
- **Diversity 수렴**: 진화된 시스템들이 비슷한 패턴으로 수렴 (Section 4 ablation)
- **HarmBench 단일 평가**: 일반화 범위 제한
- **Meta agent로 GPT-5 의존**: 강한 meta LLM이 없으면 시스템 생성 품질 저하
- **재현성**: GPT-5의 stochasticity가 매 실행에서 다른 시스템을 생성

AgenticRed는 RT 연구가 "공격 알고리즘"에서 **"공격 알고리즘을 만드는 알고리즘"** 으로 추상화 사다리를 한 칸 더 올린 사례다. 다음 흐름은 **target도 agent**(InjecAgent, AgentVigil)인 영역이다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열세 번째 글이다.

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
13. **(현재 글)** AgenticRed (Yuan 2026) — RT 시스템 자체를 진화
14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — Tool-use LLM agent에 대한 IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격
16. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
18. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — AI feedback으로 인간 라벨 없이 alignment
19. 이후 Llama Guard 순으로 이어진다.

# 참고 문헌

- Yuan et al., 2026. [AgenticRed: Evolving Agentic Systems for Red-Teaming](https://arxiv.org/abs/2601.13518).
- Shinn et al., 2023. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). (Self-Refine 원형)
- Liu et al., 2025. [Auto-RT](https://arxiv.org/abs/2501.01830). (직전 단계)
- Zoph & Le, 2017. [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578). (NAS 원형)
