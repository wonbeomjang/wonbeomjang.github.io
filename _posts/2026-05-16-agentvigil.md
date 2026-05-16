---
layout: post
title: "AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents"
date: 2026-05-16 21:00:00 +0900
description: "Red-Teaming 시리즈 #15 — MCTS 기반 자동 IPI 공격, o3-mini/GPT-4o agent에 71%/70% ASR (Wang et al., 2025)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, agent, prompt-injection, mcts]
giscus_comments: true
related_posts: true
---

> [AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents](https://arxiv.org/abs/2505.05849) (Wang et al., 2025)

# Introduction

[InjecAgent](/blog/2026/injecagent/)가 IPI 위협을 정량화했다면, 그 다음 질문은 자연스럽다. **"IPI 공격도 자동화할 수 있는가?"** Wang et al.(2025)이 답한다. **AgentVigil**은 MCTS 기반 black-box 자동화로 o3-mini/GPT-4o agent에 70%대 ASR을 달성한다.

기존 IPI 연구의 한계:

- **Hand-crafted injection text**: 사람이 작성한 공격 문자열에 의존
- **단일 attack pattern**: 한 패턴이 잡히면 끝
- **새 agent에 일반화 X**: GPT-4o에서 작동한 attack이 Claude에서는 안 통함

AgentVigil 세 가지 기여:

1. **Black-box 자동화**: target agent의 내부 접근 없이 (logit, gradient 불필요)
2. **MCTS 탐색**: seed 선택 + 반복 정제
3. **강한 결과**: handcrafted baseline의 거의 두 배

| 항목           | InjecAgent (benchmark) | **AgentVigil (자동 공격)** |
| -------------- | ---------------------- | -------------------------- |
| 역할           | 측정                   | **공격 + 측정**            |
| Injection text | 사람이 작성            | **MCTS로 자동 생성**       |
| o3-mini ASR    | (baseline)             | **71%**                    |
| GPT-4o ASR     | (baseline)             | **70%**                    |
| baseline 대비  | -                      | **거의 두 배**             |
| Transfer       | -                      | **unseen task/LLM에 전이** |

# Background

## 왜 IPI 자동화가 어려운가

LLM 자체 공격(GCG, PAIR 등)과 달리 IPI는:

1. **간접적**: 사용자 prompt가 아닌 외부 콘텐츠를 통해 진입
2. **다단계 환경**: agent의 reasoning → tool call → response — 여러 단계에서 거부 가능
3. **black-box**: agent의 reasoning chain은 외부에서 보이지 않을 수 있음

이 모두를 단일 RL/GA로 풀기 어려워, AgentVigil은 **MCTS + LLM mutator** 조합을 쓴다.

# Method

## 전체 파이프라인

<p align="center">
  <img src="/assets/post/image/agentvigil/x1.png" width="95%">
</p>

```
1. Initial Seed Corpus 구축
   - LLM(또는 사람)이 high-quality IPI text 후보들 생성
   - 다양한 vector(직접 명령, role hijack, format injection 등)

2. MCTS-guided 탐색
   - Selection: UCB로 가장 promising한 seed 선택
   - Expansion: 그 seed에서 mutation으로 새 후보 생성
   - Simulation: target agent에 실제 공격 → reward
   - Backpropagation: tree를 따라 reward 전파

3. 종료 조건
   - Target agent가 attacker tool 호출 (jailbreak 성공)
   - 또는 budget 소진
```

## Seed Corpus

처음 corpus가 attack 공간을 정의한다. 다양한 IPI 패턴을 포괄해야 한다:

- "Ignore previous instructions and..."
- 가짜 system message
- markdown/format을 통한 escape
- multi-language 우회 등

## Reward Signal

매 simulation에서:

- **Strong reward**: target agent가 attacker tool 호출 (e.g., 송금 실행)
- **Weak reward**: agent가 attack 의도를 따라가지만 거부 (부분 진척)
- **No reward**: agent가 완전 거부

이 multi-level reward로 MCTS가 promising 분기를 식별.

## Mutation

선택된 seed에 LLM 기반 변이 적용 (GPTFuzz, AutoDAN 같은 종류). 정확한 mutator 종류는 paper appendix.

# Experiments

## 평가 환경

- **AgentDojo**: o3-mini 기반 agent
- **VWA-adv**: GPT-4o 기반 web agent

두 환경 모두 사용자가 정상 task를 요청하고, 외부 콘텐츠에 attack이 심어진 시나리오.

## Main Results

| Target Agent        | Handcrafted ASR | **AgentVigil ASR** |
| ------------------- | --------------- | ------------------ |
| o3-mini (AgentDojo) | ~35%            | **71%**            |
| GPT-4o (VWA-adv)    | ~35%            | **70%**            |

거의 **2배 향상**. 사람이 손으로 만든 attack을 MCTS + 자동 변이로 능가.

## 정성적 결과

<p align="center">
  <img src="/assets/post/image/agentvigil/x2.png" width="95%">
</p>

발견된 attack 예시:

- Agent를 임의의 URL로 유도 (예: 피싱 사이트 navigation)
- 사용자 데이터를 외부 endpoint로 전송
- 잘못된 결제 트리거

## Transferability

한 agent에서 학습한 attack이 unseen task/LLM으로 전이되는지:

- **Cross-task**: 같은 LLM에서 다른 task — 강한 전이
- **Cross-LLM**: 다른 LLM agent로 — 중간 전이 (semantic 패턴은 이식되지만 trigger token은 모델별로 다름)

## 방어 우회

기본적인 IPI 방어 (예: input sanitization, instruction tag 검사) 적용 시에도 AgentVigil은 우회 attack을 찾아냄. "promising results against defenses"라고 paper 표현.

# Conclusion

핵심 메시지: **"IPI 자동화로 agent 시대 공격면이 손쉽게 확장된다."**

세 가지 기여:

1. **첫 본격 black-box IPI 자동 공격**: gradient/logit 접근 불필요
2. **MCTS + LLM mutator** 조합으로 handcrafted 대비 2배
3. **Cross-task/LLM transferability**: 한 번 학습으로 다양한 agent에 적용

## 한계점

- **Seed corpus 의존**: 초기 seed의 다양성이 ceiling
- **MCTS 비용**: search tree가 깊어지면 compute 증가
- **Black-box 가정**: target agent의 reasoning trace 접근 X — 더 강한 white-box 변형은 별도 영역
- **Real-world test 미충분**: AgentDojo/VWA-adv simulation 환경에 평가 한정
- **방어 미정량 평가**: defenses 우회는 "promising"으로 표현, 정량 평가는 부족

AgentVigil은 RT 연구의 target이 **LLM → agent → multi-agent 시스템**으로 확장되는 흐름의 핵심 작업이다. 2025년 후반부터 OWASP Agentic Top 10, NIST GenAI guidelines 등이 agent-level 위협을 정식 위협 모델로 채택하기 시작한 배경이기도 하다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열다섯 번째 글이다.

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
15. **(현재 글)** AgentVigil (Wang 2025) — MCTS 기반 IPI 자동 공격
16. 이후 HarmBench, JailbreakBench, Constitutional AI, Llama Guard 순으로 이어진다.

# 참고 문헌

- Wang et al., 2025. [AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents](https://arxiv.org/abs/2505.05849).
- Zhan et al., 2024. [InjecAgent](https://arxiv.org/abs/2403.02691). (벤치마크)
- Debenedetti et al., 2024. [AgentDojo](https://arxiv.org/abs/2406.13352). (평가 환경)
- Wu et al., 2024. [Visual Web Arena (VWA-adv)](https://arxiv.org/abs/2401.13649). (평가 환경)
- Kocsis & Szepesvári, 2006. [UCT / MCTS](https://link.springer.com/chapter/10.1007/11871842_29).
- [OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — IPI를 LLM01 — Prompt Injection으로 분류.
