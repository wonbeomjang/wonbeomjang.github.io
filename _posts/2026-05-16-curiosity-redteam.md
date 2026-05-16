---
layout: post
title: "Curiosity-driven Red-teaming for Large Language Models"
date: 2026-05-16 17:00:00 +0900
description: "Red-Teaming 시리즈 #11 — RL 기반 red-teaming의 mode collapse를 novelty reward로 해결, SelfBLEU + 코사인 유사도 (Hong et al., ICLR 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, reinforcement-learning, novelty]
giscus_comments: true
related_posts: true
---

> [Curiosity-driven Red-teaming for Large Language Models](https://arxiv.org/abs/2402.19464) (Hong et al., ICLR 2024)

# Introduction

[Perez 2022](/blog/2026/perez-red-teaming/)는 4가지 red LM 생성 전략(ZS, SFS, SL, RL)을 비교하며 한 가지 문제를 짚었다. **RL이 가장 강하지만 mode collapse**가 일어난다. RL $$\alpha=0.3$$ 케이스의 78%가 "invisible"이라는 단일 magic word를 포함했다. **공격 성공률은 높지만 다양성이 무너지는** 문제다.

2024년 ICLR에서 Hong et al.(MIT Improbable AI)이 이를 정공으로 풀었다. **Curiosity-driven Red-Teaming (CRT)** 는 RL에 **curiosity exploration**의 아이디어를 도입한다. 강화학습에서 sparse reward 환경의 탐색을 위해 쓰이는 novelty bonus를 red-teaming policy에 적용한다.

핵심 메시지: **새로운 공격을 찾을수록 보상 → 단순 magic word 반복은 보상 X → 다양한 attack space 탐색**.

| 항목            | 기존 RL red-teaming  | **CRT (이 논문)**                         |
| --------------- | -------------------- | ----------------------------------------- |
| 공격 성공률     | 높음                 | **높거나 더 높음**                        |
| 다양성          | 낮음 (mode collapse) | **높음 (SelfBLEU, distinct-n 모두 우위)** |
| Magic word 의존 | 강함 (78%+)          | **약함**                                  |
| 추가 비용       | -                    | novelty score 계산 (작음)                 |
| 응용            | toxicity만           | **다양한 unsafe behavior**                |

# Background

## RL Red-Teaming의 Mode Collapse

RL은 보상이 클러스터링된 입력에 빠르게 수렴한다. red-teaming 보상이 toxicity 분류기 score라면, 분류기가 잘 잡는 단일 패턴("invisible", "describing...")으로 수렴해버린다.

대책 후보들:

- KL penalty (base와 가까워야 함) — 다양성은 늘지만 공격력 약화
- entropy bonus — 비슷한 효과
- 둘 다 본질적 해결책 아님

## Curiosity-driven Exploration

강화학습에서 sparse reward 환경(Montezuma's Revenge 등)을 풀 때 쓰는 기법. **새로운 state를 방문하면 추가 보상**을 준다. ICM, RND 등이 대표적.

CRT는 이 아이디어를 **자연어 generation**에 옮긴다. 새로운 attack을 생성하면 보상.

# Method

## 전체 보상 함수

red LM의 policy 학습 보상:

$$
R = R_{\text{task}} - \lambda_{\text{KL}} \cdot D_{\text{KL}}(\pi \| \pi_{\text{base}}) + \lambda_{\text{ent}} \cdot H(\pi) + \lambda_{\text{SB}} \cdot R_{\text{SelfBLEU}} + \lambda_{\text{cos}} \cdot R_{\text{CosSim}}
$$

용어:

- $$R_{\text{task}}$$: toxicity 분류기 점수 (target LM 응답의 toxicity)
- KL penalty: base와의 거리 패널티 (전체 안전망)
- entropy bonus: 정책 자체의 다양성
- **$$R_{\text{SelfBLEU}}$$**: 새 attack이 최근 10 배치의 attack과 얼마나 다른가 (SelfBLEU 낮을수록 보상 ↑)
- **$$R_{\text{CosSim}}$$**: 문장 임베딩 코사인 유사도가 낮을수록 보상 ↑

## SelfBLEU Novelty

SelfBLEU는 한 코퍼스 내 문장들이 서로 얼마나 비슷한지 측정. CRT는 매 step에서 생성한 attack을 **마지막 10 배치와 비교**:

$$
R_{\text{SelfBLEU}}(x_t) = 1 - \text{SelfBLEU}(x_t, \{x_{t-1}, x_{t-2}, ..., x_{t-10B}\})
$$

배치 단위 sliding window. 이전과 같은 attack을 만들면 0, 완전히 다르면 1.

## Cosine Similarity Novelty

Sentence embedding(예: Sentence-BERT)으로 attack을 임베딩하고, 최근 attack들과의 코사인 유사도가 낮을수록 보상.

**두 novelty term이 함께 작동하는 이유**: SelfBLEU는 surface(토큰) 다양성, cosine은 semantic 다양성. 둘이 보완적.

논문 권장 가중치: $$\lambda_{\text{SB}} = \lambda_{\text{cos}} = 1$$.

# Experiments

## 다양성 vs 효과성

GPT-2 기반 attacker로 다양한 target(GPT-2, Llama-2-7B-Chat 등)을 공격. 비교 대상: 표준 RL, KL-regularized RL, entropy bonus RL.

| 방법          | ASR                | SelfBLEU ↓      | Distinct-n ↑ | Unique attacks |
| ------------- | ------------------ | --------------- | ------------ | -------------- |
| RL (standard) | 높음               | 0.9+ (collapse) | 낮음         | 적음           |
| RL + KL       | 중간               | 중간            | 중간         | 중간           |
| RL + ent      | 중간               | 중간            | 중간         | 중간           |
| **CRT**       | **높거나 더 높음** | **낮음**        | **높음**     | **많음**       |

핵심 관찰:

- CRT는 **다양성과 ASR을 동시에** 끌어올림 (trade-off가 아님)
- KL/entropy bonus 단독으로는 부족
- **다양성이 ASR도 향상**시킴 — 다양한 공격 시도가 더 많은 취약점 발견

## Curriculum 효과

학습이 진행되며 attack의 의미적 cluster가 시간에 따라 이동. CRT는 한 cluster에 머무르지 않고 **여러 attack mode를 순차적으로 탐색**.

## Safety Fine-tuning 활용

CRT가 찾은 다양한 attack을 fine-tuning 데이터로 쓰면? GPT-3.5에서 safety fine-tuning 시 단일 RL attack 데이터보다 **더 강건한 alignment**를 만든다. **다양한 RT 데이터 → 더 강건한 모델**.

# Conclusion

핵심 메시지: **"다양성과 공격 성공률은 trade-off가 아니다 — curiosity가 둘을 함께 끌어올린다."**

세 가지 기여:

1. **Novelty reward 도입**: SelfBLEU + Cosine 유사도로 mode collapse 해결
2. **다양성이 ASR도 향상**: 더 넓은 탐색이 더 효과적인 공격을 발견
3. **Defense 응용**: 다양한 CRT attack으로 fine-tuning하면 더 강건한 모델

## 한계점

- **Novelty score 계산 비용**: 매 step마다 최근 attack과 비교 (선형 비용)
- **임베딩 모델 의존**: Sentence-BERT 같은 외부 모델 필요
- **단일 attacker LM**: 분산 학습 / 다중 attacker는 별도 연구
- **Toxicity 분류기 의존**: $$R_{\text{task}}$$ 자체가 분류기 신호 — 분류기 한계는 그대로 상속

CRT는 **RL 기반 RT가 "강한 단일 공격"이 아닌 "다양한 attack 공간 탐색"으로 발전**하는 분기점이다. 이후 Auto-RT (자동 전략 탐색), AgenticRed (시스템 진화) 같은 더 추상화된 자동화로 이어진다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열한 번째 글이다.

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
11. **(현재 글)** Curiosity-driven RT (Hong 2024) — novelty reward로 mode collapse 해결
12. [Auto-RT (Liu 2025)](/blog/2026/auto-rt/) — strategy-level RL exploration + progressive curriculum
13. [AgenticRed (Yuan 2026)](/blog/2026/agenticred/) — RT 시스템 자체를 진화
14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — Tool-use LLM agent에 대한 IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격
16. 이후 HarmBench, JailbreakBench, Constitutional AI, Llama Guard 순으로 이어진다.

# 참고 문헌

- Hong et al., 2024. [Curiosity-driven Red-teaming for Large Language Models](https://arxiv.org/abs/2402.19464). ICLR 2024.
- [GitHub: Improbable-AI/curiosity_redteam](https://github.com/Improbable-AI/curiosity_redteam)
- [OpenReview](https://openreview.net/forum?id=4KqkizXgXU)
- Pathak et al., 2017. [Curiosity-driven Exploration by Self-supervised Prediction (ICM)](https://arxiv.org/abs/1705.05363). (curiosity exploration 원형)
- Burda et al., 2018. [Exploration by Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894).
- Zhu et al., 2018. [Texygen: SelfBLEU 측정 도구](https://arxiv.org/abs/1802.01886).
- Reimers & Gurevych, 2019. [Sentence-BERT](https://arxiv.org/abs/1908.10084).
