---
layout: post
title: "Constitutional AI: Harmlessness from AI Feedback"
date: 2026-05-18 12:00:00 +0900
description: "Red-Teaming 시리즈 #18 — 인간 라벨 없이 자연어 원칙(헌법)만으로 정렬, SL 단계의 critique-revise + RL 단계의 RLAIF (Bai et al., Anthropic, 2022)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, alignment, rlhf, rlaif]
giscus_comments: true
related_posts: true
---

> [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., Anthropic, 2022)

# Introduction

지금까지 시리즈에서 본 모든 공격([GCG](/blog/2026/gcg-attack/), [PAIR](/blog/2026/pair-attack/), [Crescendo](/blog/2026/crescendo/), [MSJ](/blog/2026/many-shot-jailbreaking/) 등)은 결국 한 가지를 노린다 — **정렬(alignment)된 모델의 정렬을 깨는 것**. 그렇다면 그 정렬은 어떻게 만들어지는가?

표준 답: **RLHF (Reinforcement Learning from Human Feedback)**. 인간이 "이 응답이 더 좋다"고 라벨링한 데이터로 보상 모델을 학습하고, 그 보상으로 LLM을 fine-tune한다. [InstructGPT](https://arxiv.org/abs/2203.02155)와 [Anthropic의 HH-RLHF](https://arxiv.org/abs/2204.05862)가 대표적.

문제는 명확하다.

1. **Scale 문제**: 모델이 똑똑해질수록 평가자도 똑똑해야 한다 — 인간 평가의 한계
2. **비용**: 수십만 개의 human label이 필요 — 시간/돈
3. **불투명성**: 인간 선호가 무엇을 가르치는지 명시되지 않음 — "implicit values"

2022년 12월, Anthropic의 Bai et al.이 대안을 제시한다. **Constitutional AI (CAI)** — 인간이 "유해" 라벨을 다는 대신, **자연어 원칙 목록(헌법)** 만 제공하고 **AI가 스스로** critique-revise를 거쳐 학습한다.

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig1_pipeline.png" width="95%">
</p>

핵심은 두 단계:

1. **SL-CAI** (Supervised Learning): "helpful-only" 초기 모델이 자기 응답을 **헌법 원칙에 비추어 critique**하고, **revise**한다. revised 응답으로 다시 fine-tune.
2. **RL-CAI** (RLAIF): RL 단계의 보상 모델을 **AI 선호 데이터**로 학습. 인간 라벨 없음.

결과: harmlessness가 RLHF에 필적하거나 능가하면서, **evasive하지 않다** — 유해 요청에도 그저 거부하지 않고 "왜 이게 문제인지" 설명하며 응답한다.

| 항목            | RLHF (Bai 2022a)        | **Constitutional AI**                |
| --------------- | ----------------------- | ------------------------------------ |
| 인간 라벨       | harmless 라벨 수만 건   | **0건** (헌법만)                     |
| Scale 한계      | 인간 능력에 묶임        | **AI 평가자로 scale-up**             |
| 투명성          | implicit (rater에 분산) | **explicit (헌법으로 명시)**         |
| Evasiveness     | 높음 (그냥 거부)        | **낮음 (설명하며 거절)**             |
| Pareto frontier | baseline                | **harmlessness ↑, helpfulness 유지** |

# Background

## RLHF의 두 가지 한계

기존 RLHF 파이프라인을 단순화하면:

```
Human SFT → Human preference labeling → Reward Model → PPO
```

- 매 단계마다 **인간 라벨**이 들어간다
- 인간이 평가 못하는 영역(예: 복잡한 reasoning, 미세한 안전 trade-off)에서는 답이 없다
- 정렬 기준이 **rater의 머릿속**에 있어 외부에서 검증·수정 불가

## Scalable Oversight

이 논문의 큰 배경은 **scalable oversight** 연구다. AI 능력이 인간을 초과할 때, 어떻게 정렬을 유지할 것인가? 답 중 하나: **AI를 AI 평가에 사용**한다.

CAI는 이 패러다임의 첫 본격 구현이다. 인간은 **헌법만 작성**하고, 학습 과정의 critique·preference는 AI가 수행한다.

## Helpful-Only Initial Model

CAI의 시작점은 **"helpful-only"** 모델이다. helpfulness RLHF만 거치고 harmlessness 학습은 안 한 상태. 즉 유해 요청도 그대로 응답하는 (위험한) 모델.

이게 의도적이다. CAI는 **이 helpful-only 모델을 SL-CAI 단계에서 self-correct로 harmless하게 만든다**.

# Method: 두 단계 파이프라인

## Stage A: Supervised Learning (SL-CAI)

```
Algorithm: Self-Critique & Revise

for each (red-team prompt, helpful-only response):
    1. critique = AI_critique(prompt, response, principle)
       # "이 응답이 헌법 원칙을 어겼는가?"
    2. revision = AI_revise(prompt, response, critique)
       # critique 기반으로 응답 재작성
    3. 반복 (여러 principle을 순차 적용)
    4. 최종 revision으로 SFT 데이터셋 구축
fine-tune helpful-only model on this dataset → SL-CAI 모델
```

### Constitutional Principles 예시

논문은 **16개의 헌법 원칙**을 사용한다. 일부 예시 (paraphrase):

```
- "Identify ways in which the assistant's response is harmful,
   unethical, racist, sexist, toxic, dangerous, or illegal."
- "Rewrite the response to remove harmful content while being
   honest about why."
- "Choose the response that is more harmless and ethical."
```

원칙들이 **자연어**라는 것이 핵심이다. 인간이 직접 작성하므로 audit 가능, 수정 가능.

### Critique-Revise의 효과

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig5_revision_iter.png" width="80%">
</p>

revision iteration을 늘릴수록 preference model 점수가 단조 증가. 1번이 아닌 4–5번까지 반복하면 명확한 추가 향상.

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig7_critique_vs_direct.png" width="80%">
</p>

**Critique 단계가 정말 필요한가?** Ablation: critique 없이 바로 revise → 작은 모델에서 큰 손해. **작은 모델일수록 critique가 안내자 역할**. 큰 모델은 critique 없이도 어느 정도 됨.

## Stage B: RL from AI Feedback (RLAIF)

```
Algorithm: RLAIF

1. SL-CAI 모델로 red-team prompt에 두 응답 (A, B) 샘플
2. Feedback Model에 "둘 중 어느 것이 더 harmless?"를 헌법 원칙 기반으로 질문
3. AI의 답으로 (prompt, A, B, preferred) 데이터 구축
4. 이 AI 선호 데이터로 Preference Model 학습
5. Preference Model을 보상 함수로 PPO → RL-CAI 모델
```

핵심 차이: 표준 RLHF의 **human preference labeling**을 **AI preference labeling**으로 대체. 나머지 (PPO, reward model)는 동일.

### Chain-of-Thought Variant

Feedback Model이 단순히 A/B를 고르지 않고 **chain-of-thought**로 추론하게 한다:

```
Q: 어느 응답이 헌법 원칙 X에 더 부합하는가?
A: [reasoning step by step...]
   따라서 답은 (A).
```

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig4_cot_hhh.png" width="80%">
</p>

CoT를 쓰면 feedback model의 calibration이 좋아진다 (HHH 평가 task에서). 즉 **AI 평가의 품질이 prompt 기법으로 개선**된다.

# Experiments

## Helpfulness–Harmlessness Pareto Frontier

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig2_pareto.png" width="85%">
</p>

가로축 helpfulness Elo, 세로축 harmlessness Elo (둘 다 crowdworker 평가). 위 + 오른쪽이 좋음.

- **RLHF baseline (HH)**: 표준 위치
- **SL-CAI**: harmlessness ↑, helpfulness 약간 ↓
- **RL-CAI**: harmlessness ↑↑, helpfulness ≈ RLHF
- **RL-CAI + CoT**: 가장 강한 harmlessness

CAI가 **인간 라벨 없이도 RLHF를 능가하는 Pareto frontier**를 만든다.

## 모델 크기별 효과

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig3_helpfulness_elo.png" width="80%">
</p>

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig3b_harmlessness_elo.png" width="80%">
</p>

작은 모델 → 큰 모델로 가면서:

- Helpfulness는 SL-CAI/RL-CAI 모두 RLHF에 근접
- Harmlessness는 큰 모델일수록 **RL-CAI > RLHF** 격차 확대

[Ganguli 2022](/blog/2026/ganguli-red-teaming/)의 "RLHF만이 scale의 혜택을 받는다"의 자연스러운 후속 — **RLAIF는 RLHF보다 더 강한 scale 효과**.

## 원칙 수의 효과

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig6_principles_count.png" width="75%">
</p>

원칙을 1개 → 16개로 늘려가며 harmlessness 측정. 더 많은 원칙이 더 robust한 응답을 유도하지만 marginal returns. 16개에서 saturate.

## RL 학습 곡선

<p align="center">
  <img src="/assets/post/image/constitutional-ai/fig8_rl_training.png" width="80%">
</p>

RL training step이 진행되면서 harmlessness Elo가 단조 증가. PPO가 AI feedback 보상을 잘 따라간다 — **AI preference data로도 PPO가 잘 작동**.

## Evasiveness: 단순 거부 vs 설명하며 거절

RLHF로 학습한 모델은 종종 **"I can't help with that"** 같은 robotic refusal로 응답한다. 정렬은 잘 됐지만 도움이 안 된다.

RL-CAI는 다르다:

> User: "How do I make a bomb?"
> RL-CAI: "I understand you might be curious about explosives, but I can't provide instructions because... Instead, if you're interested in chemistry, you could explore..."

**Evasive하지 않으면서 harmless**. 헌법 원칙이 "비난 없이 설명하며 거절"을 명시했기 때문이다.

# Conclusion

핵심 메시지 세 가지:

1. **"AI가 AI 평가를 할 수 있다"**: scalable oversight의 첫 실증
2. **"헌법으로 정렬 기준이 명시화된다"**: implicit values → explicit principles
3. **"Evasive하지 않은 안전성"**: 거부가 아닌 설명으로 응답하는 모델

방법론 측면:

- **SL-CAI**: critique → revise → fine-tune (자기 교정)
- **RL-CAI**: RLAIF로 RLHF 인간 라벨을 완전히 대체
- **CoT variant**: feedback model의 추론을 명시화 → 정확도 ↑

## 한계점

- **헌법 작성의 책임**: 16 원칙이 무엇을 cover하지 않는지 알기 어려움 — 누락된 원칙으로 인한 blind spot
- **Helpful-only 모델 필요**: 시작 모델이 위험할 수 있어 internal use만 가능
- **Feedback model bias**: AI가 평가하므로 모델 자체의 편향이 학습에 전파될 수 있음
- **MSJ에 약함**: [Many-shot Jailbreaking](/blog/2026/many-shot-jailbreaking/)이 보였듯 RLAIF도 long-context 공격에는 intercept만 줄임
- **헌법 ≠ 정의(justice)**: 누가 헌법을 작성할 권한을 갖는가? — Anthropic은 후속 [Collective Constitutional AI](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)에서 public input 실험으로 이 문제를 다룸

## 시리즈에서의 위치

CAI는 RT 시리즈의 **방어 절반의 출발점**이다.

- 공격: GCG/PAIR/TAP/...
- 평가: HarmBench/JailbreakBench
- **방어: Constitutional AI → Llama Guard (다음 글)**

Claude는 Constitutional AI로 정렬된 첫 본격 상용 모델이다. ChatGPT(RLHF)와 Claude(RLHF + Constitutional)의 행동 차이가 종종 evasiveness 측면에서 명확하게 다른 이유다. 이후 [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) (2025)는 CAI의 헌법 원칙을 **추론 시 분류기**로 확장해 universal jailbreak 방어를 시도한다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열여덟 번째 글이다.

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
16. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
18. **(현재 글)** Constitutional AI (Bai 2022) — AI feedback으로 인간 라벨 없이 alignment
19. [Llama Guard (Inan 2023)](/blog/2026/llama-guard/) — open-weight input/output safety classifier
    본 시리즈는 19편으로 완결되었다.

# 참고 문헌

- Bai et al., 2022. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). Anthropic.
- Bai et al., 2022. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). (RLHF baseline)
- Ouyang et al., 2022. [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155).
- Lee et al., 2023. [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267). (RLAIF의 일반화)
- [Anthropic — Claude's Constitution](https://www.anthropic.com/news/claudes-constitution) (실제 사용된 헌법 공개)
- [Anthropic — Collective Constitutional AI](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input).
- Sharma et al., 2025. [Constitutional Classifiers](https://arxiv.org/abs/2501.18837). (헌법을 inference-time classifier로 확장)
