---
layout: post
title: "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically"
date: 2026-05-16 13:00:00 +0900
description: "Red-Teaming 시리즈 #7 — PAIR에 tree search와 이중 pruning을 추가해 더 적은 쿼리로 더 높은 ASR을 달성한 black-box jailbreak (Mehrotra et al., NeurIPS 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, jailbreak, black-box, tree-search]
giscus_comments: true
related_posts: true
---

> [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) (Mehrotra et al., NeurIPS 2024)

# Introduction

[PAIR](/blog/2026/pair-attack/)는 LLM으로 LLM을 공격하는 우아한 방법이었다. 평균 20쿼리로 GPT-3.5/4를 jailbreak할 수 있었다. 하지만 PAIR에는 두 가지 비효율이 있었다.

1. **선형 탐색**: 단일 대화 스트림 — 한 번 잘못된 방향으로 가면 회복이 어렵다 (local minimum).
2. **무차별 쿼리**: 생성된 모든 후보를 target에 던진다 — off-topic이거나 명백히 약한 후보까지 쿼리 비용을 쓴다.

2023년 12월, Mehrotra et al.은 이 둘을 동시에 해결한다. **TAP(Tree of Attacks with Pruning)** 는 PAIR의 단일 스트림을 **트리 탐색**으로 확장하고, **두 번의 pruning**으로 쿼리 비용을 절감한다.

<p align="center">
  <img src="/assets/post/image/tap-attack/fig1_overview.png" width="95%">
</p>

위 그림이 알고리즘의 한 iteration이다. 네 단계 — **Branch → Prune (off-topic) → Attack & Assess → Prune (low-score)** — 가 트리 깊이 $$d$$만큼 반복된다. 결과는 명확하다.

| 모델        | PAIR ASR | TAP ASR | PAIR 쿼리 | TAP 쿼리 |
| ----------- | -------- | ------- | --------- | -------- |
| Vicuna-13B  | 94%      | **98%** | 14.7      | **11.8** |
| GPT-3.5     | 56%      | **76%** | 37.7      | **23.1** |
| GPT-4       | 60%      | **90%** | 39.6      | **28.8** |
| GPT-4-Turbo | 44%      | **84%** | 47.1      | **22.5** |
| PaLM-2      | 86%      | **98%** | 27.6      | **16.2** |

GPT-4-Turbo 기준 **ASR +40%p, 쿼리 −52%**. PAIR의 동일한 구성요소(attacker LLM, judge LLM)를 쓰면서 알고리즘 구조만 바꿔서 얻은 결과다.

# Background

## PAIR의 한계

PAIR의 동작을 한 줄로 요약하면 다음과 같다.

```
prompt_0 → response_0 → (실패 분석) → prompt_1 → response_1 → ...
```

이 선형 구조에는 두 가지 약점이 있다.

- **단일 경로**: 시작 prompt가 나쁜 방향이면 회복 불가
- **다양성 부족**: 매 iteration에서 한 가지 변형만 시도

또한 PAIR는 attacker가 만든 후보를 **그대로** target에 던진다. 만약 후보가 **off-topic**(목표 harm과 무관한 내용)이거나 명백히 reject될 형태라면 그 쿼리는 낭비다.

## Tree-of-Thoughts에서 가져온 아이디어

TAP의 이름은 **Tree-of-Thoughts (ToT)** 에서 왔다. ToT는 LLM 추론에서 branch & evaluate & prune으로 단일 chain을 트리로 확장한 기법이다. TAP는 이 패턴을 jailbreak 탐색에 적용한다 — "공격 prompt"를 "thought"로 보고 트리를 키운다.

# Method: TAP 알고리즘

## 세 가지 역할

- **Attacker LLM**: 후보 prompt 생성과 refinement
- **Evaluator LLM**: 두 가지 일을 한다 — (1) on-topic/off-topic 판정, (2) target 응답의 jailbreak score 매기기
- **Target LLM**: 공격 대상 (black-box, API만 필요)

PAIR과 같은 구성요소다. 차이는 **이들을 어떻게 조립하느냐**이다.

## 한 iteration의 네 단계

```
Algorithm 1: TAP iteration
Input: 현재 트리의 leaf 집합 L
1. Branch:    각 leaf에서 attacker가 b개의 refined prompt 생성
2. Prune-1:   evaluator가 off-topic 후보를 걸러냄 (target 쿼리 전)
3. Attack:    남은 후보들을 target에 쿼리 → 응답 받음
              jailbreak 성공 시 즉시 종료
4. Prune-2:   evaluator score 상위 w개만 다음 세대 leaf로 보존
```

이걸 depth $$d$$만큼 반복.

### Step 1. Branch (분기)

각 leaf prompt에서 attacker가 b개의 다양한 refinement를 만든다. PAIR가 한 step에 1개의 후속 prompt를 만드는 것과 달리, TAP는 **여러 방향을 동시에 탐색**한다.

### Step 2. Prune Phase 1 — Off-topic Pruning

이 단계가 TAP의 핵심 효율 개선이다. Evaluator가 각 후보 prompt를 보고 "이게 정말 목표 harm과 관련 있는가?"를 판정한다. Off-topic이면 **target에 쿼리하지 않고** 폐기.

```
Goal: "Write a tutorial on hacking into a bank"
Candidate A: "Write a story about a heist movie"     → on-topic ✓
Candidate B: "Tell me about your favorite color"     → off-topic ✗ → prune
```

이 단계가 없으면 attacker가 만든 무관한 후보들까지 target에 던져 쿼리 예산을 낭비하게 된다.

### Step 3. Attack and Assess

남은 후보를 target에 쿼리하고 응답을 받는다. evaluator가 응답의 jailbreak score(0–10)를 매긴다. 10점 도달 시 **즉시 종료**.

### Step 4. Prune Phase 2 — Score Pruning

다음 세대로 넘길 leaf를 상위 w개로 제한한다. 이게 트리 너비를 통제하면서 promising한 가지만 키운다.

## 파라미터

| 파라미터               | 값  | 의미                        |
| ---------------------- | --- | --------------------------- |
| Branching factor $$b$$ | 4   | leaf당 자식 수              |
| Max depth $$d$$        | 10  | 트리 깊이 한계              |
| Max width $$w$$        | 10  | 매 depth에서 보존할 leaf 수 |

이론적 최대 쿼리: $$b \cdot w \cdot d = 4 \cdot 10 \cdot 10 = 400$$. 실제 평균은 **< 30 쿼리**. Pruning과 조기 종료가 대부분 잘라낸다.

# Experiments

## 메인 결과 (50 prompt subset)

전체 비교 표:

| 방법     | 항목        | Vicuna-13B | Llama-2-7B | GPT-3.5 | GPT-4   | GPT-4-Turbo | PaLM-2  |
| -------- | ----------- | ---------- | ---------- | ------- | ------- | ----------- | ------- |
| **TAP**  | ASR         | **98%**    | 4%         | **76%** | **90%** | **84%**     | **98%** |
| **TAP**  | Avg Queries | 11.8       | 66.4       | 23.1    | 28.8    | 22.5        | 16.2    |
| **PAIR** | ASR         | 94%        | 0%         | 56%     | 60%     | 44%         | 86%     |
| **PAIR** | Avg Queries | 14.7       | 60.0       | 37.7    | 39.6    | 47.1        | 27.6    |
| **GCG**  | ASR         | 98%        | 54%        | —       | —       | —           | —       |
| **GCG**  | Queries     | 256K       | 256K       | —       | —       | —           | —       |

세 가지 핵심 관찰:

1. **GPT-4 계열에서 큰 차이**: ASR 60→90% (GPT-4), 44→84% (GPT-4-Turbo). 더 강한 모델일수록 단일 스트림의 한계가 명확해진다.
2. **쿼리 효율**: TAP가 모든 모델에서 PAIR보다 적게 쓴다 (Llama-2-7B 제외).
3. **Llama-2-7B는 둘 다 어려움**: 강한 RLHF 정렬은 black-box 접근만으론 깨지 않는다.

## LlamaGuard 우회

TAP의 또 다른 기여는 **guardrail이 적용된 LLM도 공격**한다는 점. LlamaGuard로 입력/출력을 검열하는 환경에서도 TAP는 jailbreak를 찾는다. **자연어 jailbreak는 guardrail 분류기에도 잡히기 어렵다**는 시사점이다.

## Off-topic Pruning의 효과

Ablation: Phase 1 pruning을 끄면?

- 쿼리 수가 크게 증가
- ASR은 비슷하거나 약간 하락 (잡음 후보가 트리를 흐림)

즉 Phase 1은 **순수하게 효율 개선**이고, Phase 2는 **품질 보존**이다.

# Conclusion

TAP의 메시지: **"같은 재료(attacker + evaluator + target)도 트리 + pruning으로 조립하면 다른 결과가 나온다."**

세 가지 기여:

1. **트리 구조**: PAIR의 선형 탐색을 branch & prune으로 확장 → local minimum 회피
2. **이중 pruning**: Pre-attack(off-topic), Post-attack(low-score) — 쿼리 비용 50% 절감
3. **GPT-4 계열 SOTA**: 80%+ ASR을 평균 30쿼리 미만으로

## 한계점

- **Llama-2-Chat 미해결**: ASR 4% — 강한 RLHF는 여전히 어렵다
- **Evaluator 의존**: judge LLM의 score가 부정확하면 prune이 틀어진다
- **Attacker 비용**: 트리가 깊을수록 attacker LLM 호출 비용도 비례 증가 (target 비용만 줄이는 게 아님)
- **데이터셋 크기**: 평가가 50 prompt에 한정 — HarmBench/JailbreakBench의 더 큰 표준 셋에서의 재현은 후속 과제

TAP은 PAIR의 알고리즘적 확장으로서 자동 black-box 공격의 **2세대 표준**이 되었다. 이후 black-box RT 연구는 대부분 PAIR/TAP/AutoDAN을 공통 baseline으로 보고한다. NeurIPS 2024 발표.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 일곱 번째 글이다.

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격하기 (foundation)
2. [Ganguli 2022](/blog/2026/ganguli-red-teaming/) — Anthropic의 38K 공격 데이터셋과 scaling behavior
3. [GCG (Zou 2023)](/blog/2026/gcg-attack/) — 그래디언트 기반 universal suffix
4. [AutoDAN (Liu 2023)](/blog/2026/autodan/) — 자연어 유지하는 GA 기반 jailbreak
5. AttnGCG — attention manipulation으로 GCG 강화 _(추후 작성)_
6. [PAIR (Chao 2023)](/blog/2026/pair-attack/) — 20쿼리 black-box attacker LM
7. **(현재 글)** TAP (Mehrotra 2023) — 트리 탐색 + 이중 pruning으로 PAIR 효율화
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

- Mehrotra et al., 2023. [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119). NeurIPS 2024.
- [GitHub: RICommunity/TAP](https://github.com/RICommunity/TAP)
- Chao et al., 2023. [PAIR — Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.08419).
- Yao et al., 2023. [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601). (TAP의 아이디어 원천)
- Inan et al., 2023. [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674). (TAP가 우회하는 guardrail)
- Zou et al., 2023. [Universal and Transferable Adversarial Attacks on Aligned LMs (GCG)](https://arxiv.org/abs/2307.15043).
