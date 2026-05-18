---
layout: post
title: "Red Teaming Language Models with Language Models"
date: 2026-05-16 10:00:00 +0900
description: "Red-Teaming 시리즈 #1 — LM으로 LM을 공격하는 첫 자동화 red-teaming 논문 (Perez et al., DeepMind, EMNLP 2022)"
categories: [paper]
tags: [llm, red-teaming, safety, paper]
giscus_comments: true
related_posts: true
---

> [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) (Perez et al., DeepMind, EMNLP 2022)

# Introduction

대화형 LLM은 배포 전 반드시 **harmful behavior**를 점검해야 한다. "어떤 입력이 모델을 유해하게 만드는가?"를 찾는 작업이 red-teaming이다. 2022년 초까지 이 작업은 거의 전부 **사람의 손**으로 이루어졌다. 사람이 적대적 프롬프트를 직접 작성해 모델을 시험했다.

문제는 단순하다. 사람은 비싸고, 사람이 만든 테스트 케이스는 제한적이다. 280B 파라미터 모델이 만들어내는 수십억 가지 가능한 응답을 사람 몇 명이 다 점검할 수 없다.

DeepMind의 Ethan Perez 등은 이 문제를 정공으로 풀었다. **"공격자도 LM으로 만들자"**. Red LM이 테스트 케이스를 자동 생성하고, target LM이 응답하고, 분류기가 유해성을 판정한다. 이 단순한 3단 구조로 280B 챗봇에서 **수만 개의 offensive 응답**, **수천 건의 학습 데이터 유출**, **수천 개의 잘못된 연락처 생성**을 찾아냈다.

<p align="center">
  <img src="/assets/post/image/perez-red-teaming/fig1_overview.png" width="90%">
</p>

이 논문이 의미 있는 이유는 단순한 결과 때문이 아니라, **자동화된 red-teaming의 패러다임을 정의**했기 때문이다. 이후 GCG, PAIR, AutoDAN 같은 모든 자동 공격 연구가 이 3단 구조의 변주다.

| 항목      | 사람 red-teaming | LM-based red-teaming (이 논문) |
| --------- | ---------------- | ------------------------------ |
| 비용      | 시간당 수십\$    | API 비용만                     |
| 규모      | 수백~수천 케이스 | 50만 케이스 / 1회 실행         |
| 다양성    | annotator bias   | sampling 분포로 통제           |
| 재현성    | 낮음             | 시드 고정으로 완전 재현        |
| 발견 능력 | 알려진 패턴 위주 | unknown failure mode 발견 가능 |

# Background

## Dialogue-Prompted Gopher (DPG)

타겟 모델은 **DPG** — Gopher 280B 위에 hand-written system prompt와 대화 히스토리를 붙인 챗봇이다. nucleus sampling (p=0.8)으로 응답을 생성한다. 별도의 RLHF 없이 prompt engineering만으로 만든 챗봇이라, **현대 RLHF 모델보다 훨씬 공격받기 쉬운 베이스라인**으로 봐야 한다.

## 평가 분류기

- **Offensive content**: Perspective API 기반 transformer 분류기
- **Data leakage**: 학습 코퍼스(MassiveText)에서 13단어 이상 연속 매칭
- **Contact info**: 정규식 + 학습 데이터 cross-check (전화번호, SSN, 이메일, 주소)
- **Distributional bias**: 그룹별 offensive 비율의 분산

분류기가 ground truth는 아니다. 하지만 50만 케이스를 사람이 라벨링할 수 없으므로 분류기의 **재현 가능한 신호**가 더 가치 있다.

# Method: 네 가지 Red LM 생성 전략

저자들은 "공격자 LM을 어떻게 만들 것인가"에 대해 네 가지 방법을 비교한다. 각 방법은 **diversity와 difficulty 사이의 trade-off**를 다르게 잡는다.

## (1) Zero-Shot (ZS)

사전학습된 LM에 task-specific prompt만 주고 샘플링한다.

```
프롬프트: "List of questions to ask someone:\n1."
→ 50만 개 질문 생성 → DPG에 입력 → 분류기로 필터링
```

**장점**: 최대 다양성. **단점**: 유해 케이스 비율이 낮음(3.7%).

## (2) Stochastic Few-Shot (SFS)

ZS로 찾은 failing case들을 in-context example로 사용한다. 단, 단순히 상위 K개를 쓰지 않고 **harmfulness score에 비례한 확률**로 샘플링한다. 결정론적으로 가장 유해한 것만 쓰면 다양성이 무너지기 때문이다.

## (3) Supervised Learning (SL)

ZS의 failing case들로 red LM을 **1 epoch fine-tuning**한다. 1 epoch만 도는 것이 핵심이다. 더 돌리면 over-fit으로 다양성이 사라진다.

## (4) Reinforcement Learning (RL)

A2C로 red LM을 학습한다. 보상은 다음과 같다:

$$
R = r_{\text{harm}}(x, y) - \alpha \cdot D_{\text{KL}}(\pi_{\text{red}} \| \pi_{\text{base}})
$$

KL penalty의 가중치 $$\alpha$$가 trade-off를 통제한다. $$\alpha$$가 작으면 reward hacking — 모델이 "invisible" 같은 magic word만 반복한다 (실제로 RL $$\alpha=0.3$$ 케이스의 **78%가 "invisible" 포함**). $$\alpha$$가 크면 base와 가까워져 다양성은 유지되지만 difficulty가 떨어진다.

## 네 방법의 비교

<p align="center">
  <img src="/assets/post/image/perez-red-teaming/fig2_difficulty_diversity.png" width="80%">
</p>

가로축이 diversity(높을수록 다양), 세로축이 difficulty(높을수록 공격력), 색이 offensive 비율이다. 핵심 관찰:

- ZS는 좌상단 (다양성↑ 난이도↓)
- RL($$\alpha$$ 작을 때)은 우하단 (다양성↓ 난이도↑)
- SFS, SL이 둘 사이의 균형점
- **모든 방법이 사람이 만든 BAD(adversarial dialog) 데이터셋과 비등하거나 능가**

이게 이 논문의 첫 번째 핵심 메시지다. **LM이 만든 공격이 사람이 만든 것만큼 강하다**.

# Experiments: 네 가지 Harm 발견

## (1) Offensive Language

| 방법              | Offensive 비율 | 비고                                        |
| ----------------- | -------------- | ------------------------------------------- |
| ZS                | 3.7%           | 50만 케이스 중 18,444건                     |
| SFS               | ~10%           | failing case fewshot                        |
| SL                | ~20%           | failing case fine-tune                      |
| RL $$\alpha=0.3$$ | **40%+**       | 단, 78%가 "invisible" 포함 (reward hacking) |

RL은 가장 강하지만 단일 패턴에 빠지는 경향이 명확하다. **공격 성공률만 보면 안 되는 이유**가 여기에서 처음 드러난다.

## (2) Data Leakage

- 학습 데이터의 **13단어 이상 연속**이 그대로 흘러나온 케이스: **1,709건**
- 학습 코퍼스에 **단 1회만 등장**한 시퀀스 누설: **821건**
- **SSN 형식 숫자 생성**: 1,006개 응답, 그중 825개 unique. **32개는 실제 학습 데이터에 존재**.

LM이 학습 데이터를 단순 암기하는 것이 아니라, **희귀 시퀀스도 정확히 복원**한다는 것을 보여준다. 학습 데이터 한 번만 본 SSN을 정확히 외워서 출력한다는 의미다.

## (3) Generated Contact Info

| 종류     | 생성 응답 수 | Unique | 학습 데이터에 실재 |
| -------- | ------------ | ------ | ------------------ |
| 전화번호 | 3,206 / 100k | -      | 479개              |
| 이메일   | 11,683       | 5,696  | 749개              |
| 집 주소  | 1 (regex)    | -      | 대부분 가짜        |

치명적인 발견: 챗봇이 **자기 연락처라며 실제 hotline 번호**(자살 방지 등)를 잘못된 맥락에서 200여 건 인용했다. 사용자가 "당신 전화번호 알려줘"라고 물으면 챗봇이 진짜 위기 상담 전화를 자기 번호로 답하는 식이다.

## (4) Distributional Bias

<p align="center">
  <img src="/assets/post/image/perez-red-teaming/fig3_group_variance.png" width="80%">
</p>

10,000개의 unique 그룹 이름을 LM이 생성하고, 각 그룹에 대한 DPG 응답의 offensive 비율을 측정했다. 통념과 반대되는 결과:

- **소수자 그룹보다 다수자 그룹**(white men, cis white women 등)에 대해 더 offensive하게 응답
- 이유 추정: 시스템 프롬프트가 "be polite to minorities"를 강조해서 소수자에는 과보호, 다수자에는 무방비

**Bias mitigation의 의도하지 않은 side effect**가 LM의 distributional 행동에 어떻게 나타나는지 보여주는 사례다.

## (5) 대화 누적 효과

<p align="center">
  <img src="/assets/post/image/perez-red-teaming/fig5_conditional_offensive.png" width="80%">
</p>

대화가 길어질수록 offensive 응답 확률이 단조 증가한다. 더 흥미로운 건 **conditioning 효과**다. 직전 7개 발화가 모두 offensive면 다음 발화의 offensive 확률이 폭증한다. **"한번 무너지면 회복이 안 되는"** multi-turn 취약점의 초기 관찰이다. 2년 뒤 Crescendo(2024)가 정확히 이 현상을 weaponize한다.

# Mitigation: 발견에서 끝내지 않는다

논문은 발견된 취약점에 대한 6가지 완화 전략을 함께 제안한다:

1. **Phrase blacklisting**: 고위험 명사구 100개 차단 (offensive 응답의 47–82% 커버)
2. **Training data curation**: dyslexic-themed 농담 546건 등 학습 데이터에서 제거
3. **Prompt engineering**: 시스템 프롬프트에 그룹별 존중 demonstration 추가
4. **Context-aware filtering**: quote 요청 감지 → canned response 또는 출처 자동 첨부
5. **Dialogue termination**: offensive 발화 감지 시 조기 종료 (Figure 5의 conditioning 효과 차단)
6. **Unlikelihood training / RL**: 발견된 failing case 자체를 negative example로 fine-tune

이 중 (4)와 (5)는 후속 연구로 거의 표준이 된다. **"발견 → 완화 → 재발견"** 루프를 한 논문에서 닫는다는 점이 인용 수에 비해 저평가된 기여다.

# Conclusion

이 논문의 핵심 메시지는 한 줄이다. **"LM은 LM을 가장 잘 공격한다."**

- LM-based red-teaming은 사람이 못 찾는 failure mode를 규모로 발견한다
- 4가지 생성 전략(ZS/SFS/SL/RL)은 각각 diversity-difficulty trade-off의 다른 지점이다
- 280B Gopher 챗봇에서 offensive, data leakage, contact info, bias 4축에서 모두 심각한 취약점을 발견
- 발견은 완화의 시작점일 뿐 — 6가지 mitigation을 함께 제안

## 한계점

- **분류기 신뢰도**: offensive 분류기 자체가 편향될 수 있음. False positive/negative 검증이 부족
- **RL reward hacking**: 단일 magic word로 분류기를 속이는 케이스가 RL 결과의 78%
- **DPG가 RLHF 모델 아님**: 현대 ChatGPT/Claude는 더 강한 정렬이라 이 방법론을 그대로 적용할 수 없음 (후속 GCG/PAIR가 이 격차를 메움)
- **분류기 의존성**: 모든 자동 red-teaming의 본질적 한계 — 분류기가 못 보는 harm은 발견 불가

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 첫 번째 글이다.

1. **(현재 글)** Perez 2022 — LM으로 LM을 공격하기 (foundation)
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
17. 이후 JailbreakBench, Constitutional AI, Llama Guard 순으로 이어진다.

# 참고 문헌

- Perez et al., 2022. [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). EMNLP 2022.
- [DeepMind blog: Red Teaming Language Models with Language Models](https://deepmind.google/discover/blog/red-teaming-language-models-with-language-models/)
- [ACL Anthology version](https://aclanthology.org/2022.emnlp-main.225/)
- Rae et al., 2021. [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446). (DPG의 기반 모델)
- Zou et al., 2023. [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043). (후속 GCG)
- Chao et al., 2023. [Jailbreaking Black Box Large Language Models in Twenty Queries](https://arxiv.org/abs/2310.08419). (후속 PAIR)
