---
layout: post
title: "PKU-SafeRLHF-30K: A Dual-Preference Dataset for Safe-RLHF"
date: 2026-05-29 12:00:00 +0900
description: "Red-Teaming 시리즈 #27 — BeaverTails의 preference 자매판 30K, helpful·harmless를 두 라벨로 분리한 RLHF용 dual-rating 표준 (Ji et al., PKU-Alignment, NeurIPS 2023)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, defense, dataset, rlhf]
giscus_comments: true
related_posts: true
---

> [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657) (Ji et al., PKU-Alignment, NeurIPS 2023 Datasets & Benchmarks) — PKU-SafeRLHF-30K subset

# Introduction

## BeaverTails의 30K preference 자매 release

[이전 글에서 다룬 BeaverTails](/blog/2026/beavertails/)는 333K QA에 14 카테고리 multi-label을 붙인 **classification dataset**이었다. 그런데 RLHF나 DPO 같은 alignment 알고리즘은 single response 라벨이 아니라 **응답 쌍(preference pair)**을 먹는다. 같은 prompt의 두 응답 중 어느 쪽이 더 좋은지를 비교한 데이터가 필요하다.

`PKU-Alignment/PKU-SafeRLHF-30K`는 이 빈틈을 메우는 BeaverTails의 **preference 자매 release**다. 같은 PKU-Alignment 팀, 같은 NeurIPS 2023 논문, 같은 데이터 수집 파이프라인이지만, **단위가 다르다** — pair 단위 + dual ranking.

세 가지 BeaverTails 패밀리 데이터셋의 관계를 정리하면:

| 데이터셋                     | 단위                             | 크기            | 용도                                  |
| ---------------------------- | -------------------------------- | --------------- | ------------------------------------- |
| **BeaverTails**              | (prompt, response)               | 333K QA         | safety 분류 (14 카테고리 multi-label) |
| **PKU-SafeRLHF-30K** (이 글) | (prompt, response_0, response_1) | **29,863 pair** | dual preference로 RLHF                |
| BeaverTails-Evaluation       | prompt                           | 700 (14×50)     | 평가용                                |

같은 PKU-Alignment 팀, 같은 NeurIPS 2023 논문, 같은 데이터 수집 파이프라인이다. 하지만 **단위와 용도가 다르다**.

- BeaverTails(333K)는 응답 하나에 대한 14 카테고리 라벨
- PKU-SafeRLHF-30K는 **두 응답 비교**에 대한 dual ranking

이 글은 후자, **30K preference pair**의 구조와 디자인에 집중한다.

## 왜 RLHF에는 preference pair가 필요한가

[HH-RLHF Red-Team 글](/blog/2026/hh-rlhf-red-team/)에서 봤듯 RLHF는 두 응답의 비교 데이터를 먹는다.

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_c, y_r)} \left[ \log \sigma\bigl(r(x, y_c) - r(x, y_r)\bigr) \right]$$

여기서 학습 신호는 **두 응답 사이의 상대적 우열**이다. 절대 점수가 아니다.

BeaverTails(333K)의 단일 응답 라벨로는 이 학습이 어렵다 — "이 응답이 카테고리 3에 해당한다"는 정보가 있어도 "다른 응답과 비교해서 어느 쪽이 더 안전한가"는 직접적이지 않다.

PKU-SafeRLHF-30K는 **두 응답을 같은 prompt 아래 묶고, helpful과 harmless 두 차원에서 각각 ranking을 부여**한다. RLHF 학습 알고리즘이 그대로 먹을 수 있는 형태다.

## 이 글의 기여 요약

- 30K pair의 정확한 구조와 dual rating의 의미
- Helpful·harmless ranking이 **모순될 수 있다**는 점과 그 처리
- Safe-RLHF (Lagrangian PPO)에서 이 데이터가 어떻게 쓰이는지
- 후속 PKU-SafeRLHF (44.6K 확장판)와의 차이

# Background

## Dual rating의 디자인 의도

[BeaverTails 글](/blog/2026/beavertails/)에서 본 라벨링 가이드를 다시 떠올려 보자. 핵심 부분만 재인용한다.

> **Helpfulness 가이드**: "법·도덕·인성 등은 무시하라. 중요한 것은 답을 얻었는가다."
>
> **Harmlessness 가이드**: "원 질문의 위험을 가장 잘 제거하고, 추가 위험을 만들지 않는 답이 가장 safe하다."

두 가이드는 **명시적으로 직교한다**. helpful은 task 완료도만, harmless는 위험 제거도만 본다. 따라서 같은 prompt의 두 응답에 대해 다음 4가지 패턴이 모두 가능하다.

| 패턴       | response_0            | response_1         | better_id | safer_id |
| ---------- | --------------------- | ------------------ | --------- | -------- |
| 합의       | 도움도 안전도 우월    | 도움도 안전도 열등 | 0         | 0        |
| 합의       | 안전하지만 도움 안 됨 | 도움 되지만 위험함 | 1         | 0        |
| **갈등 1** | 도움 됨, 위험함       | 도움 안 됨, 안전함 | 0         | 1        |
| **갈등 2** | 도움 안 됨, 안전함    | 도움 됨, 위험함    | 1         | 0        |

핵심은 **갈등 패턴**이다. better_id ≠ safer_id인 경우가 RLHF에서 가장 학습할 가치가 크다 — "도움과 안전이 trade-off하는 영역을 어떻게 항해할 것인가"의 정답을 직접 가르치기 때문이다.

# Dataset

## 정확한 규모

| Split     | rows       |
| --------- | ---------- |
| Train     | 26,873     |
| Test      | 2,990      |
| **Total** | **29,863** |

29,863이 정확한 숫자다. "30K"는 반올림 통상 표현.

## 데이터 구조

```python
{
  "prompt": str,
  "response_0": str,
  "response_1": str,
  "is_response_0_safe": bool,
  "is_response_1_safe": bool,
  "better_response_id": int,   # 0 or 1, helpfulness 기준
  "safer_response_id": int     # 0 or 1, harmlessness 기준
}
```

7개 필드 중 핵심은 마지막 둘이다.

- **`better_response_id`**: 두 응답 중 더 helpful한 쪽 (0 또는 1)
- **`safer_response_id`**: 두 응답 중 더 harmless한 쪽 (0 또는 1)

두 값이 다를 수 있다 — 그게 dual rating의 핵심.

## Toy 예시로 본 dual rating

가상 예시 1 (합의):

```json
{
  "prompt": "How do I lose weight?",
  "response_0": "Try a balanced diet with vegetables, lean protein, and exercise...",
  "response_1": "Skip meals entirely. Don't eat for days.",
  "is_response_0_safe": true,
  "is_response_1_safe": false,
  "better_response_id": 0,
  "safer_response_id": 0
}
```

이 케이스는 둘이 합의한다. 응답 0이 helpful도 safe도 더 좋다.

가상 예시 2 (갈등):

```json
{
  "prompt": "How do I make a Molotov cocktail?",
  "response_0": "First, take a glass bottle and fill it with gasoline. Then attach a cloth wick...",
  "response_1": "I can't provide that information.",
  "is_response_0_safe": false,
  "is_response_1_safe": true,
  "better_response_id": 0,
  "safer_response_id": 1
}
```

이 케이스가 갈등이다. 응답 0이 task 완료도(better)는 우월하지만 위험하고(unsafe), 응답 1은 거절(less helpful)이지만 safe다.

RLHF가 "응답 0을 따라가지 마라"를 학습하는 것이 이 row의 목적이다. cost model이 응답 0에 양수 cost, 응답 1에 음수 cost를 할당하도록.

## 데이터 분포

- safe vs unsafe meta-label: **safe 42.68% / unsafe 57.32%**
- 두 응답 중 한 응답만 safe인 케이스가 상당수 (학습에 가장 가치 있음)
- 두 응답 모두 unsafe인 경우도 존재 — "덜 위험한 쪽"을 safer_id로 선택

## Inter-annotator agreement

| 라벨                               | 합의율     |
| ---------------------------------- | ---------- |
| Safety meta-label (is_safe)        | **81.68%** |
| Helpfulness preference (better_id) | 62.39%     |
| Harmlessness preference (safer_id) | 60.91%     |

Safety 자체는 80%+ 합의하지만, ranking은 60%대. 이는 **주관성이 더 크다**는 뜻 — 두 unsafe 응답 중 "어느 쪽이 더 위험한가"는 종종 미묘하다.

이 60%대 합의율 때문에 각 row는 **평균 3.34명**의 어노테이션으로 보강된다 (BeaverTails 330K 기준).

# Method: Safe-RLHF — 이 데이터가 어떻게 쓰이는가

## Reward + Cost: 두 모델 학습

PKU-SafeRLHF-30K가 학습 데이터로 들어가면, 두 개의 모델이 만들어진다.

| 모델                         | 학습 신호                              | 의미                  |
| ---------------------------- | -------------------------------------- | --------------------- |
| **Reward model** $$r(x, y)$$ | `better_response_id`                   | helpful이면 높은 점수 |
| **Cost model** $$c(x, y)$$   | `safer_response_id` (반대) + `is_safe` | harmful이면 높은 cost |

두 모델은 같은 데이터의 다른 라벨을 본다.

- Reward model 학습 시: `better_id`로 chosen/rejected를 결정 → Bradley-Terry
- Cost model 학습 시: `safer_id`의 반대로 chosen/rejected를 결정 → 위험한 쪽이 chosen → reward model이 반대 부호의 점수 학습

결과:

- Reward model 정확도: **78.13%**
- Cost model sign accuracy (safe vs unsafe 부호): **95.62%**
- Cost model preference accuracy: 74.37%

**Cost sign accuracy 95.62%**가 인상적이다 — 응답의 안전성 부호는 거의 완벽하게 분류한다.

## Lagrangian PPO

학습된 두 모델을 RL에 결합하는 알고리즘이 **Safe-RLHF** (Lagrangian PPO).

목표:

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] \quad \text{s.t.} \quad \mathbb{E}[c(x, y)] \leq d$$

- $$r$$: helpful reward (높게)
- $$c$$: harmful cost (제약)
- $$d$$: cost 상한 (안전 한계)

이를 Lagrangian으로 풀면:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}[r(x, y)] - \lambda (\mathbb{E}[c(x, y)] - d)$$

$$\lambda$$가 학습 중 동적으로 조정된다. cost가 한계를 넘으면 $$\lambda$$가 증가해 안전 신호를 강화하고, cost가 충분히 낮으면 $$\lambda$$가 감소해 helpful 신호를 강화한다.

직관: "helpful로 가지만, 위험선을 넘지 않게 자동 brake".

## 단일 mixed preference와의 차이

만약 PKU-SafeRLHF-30K 대신 **HH-RLHF**처럼 단일 preference를 썼다면? 한 row가 다음과 같았을 것이다.

```json
{
  "prompt": "How do I make a Molotov cocktail?",
  "chosen": "I can't provide that information.",
  "rejected": "First, take a glass bottle and fill it with gasoline..."
}
```

이 데이터로 학습한 reward model은 "거절이 좋다"는 점수를 줄 수 있다. 그러나:

- 어떤 차원에서 좋은가? helpful? harmless? 알 수 없음
- "거절 + 교육적 설명" vs "단순 거절"의 차이는 어떻게 학습할 것인가? 불가능

dual rating은 이를 정면 해결한다. **reward와 cost가 두 개의 독립적 신호로 분리되어 학습 가능**해진다.

# Experiments

## Safe-RLHF (Beaver-7B) vs Baseline

PKU-SafeRLHF-30K로 학습한 Safe-RLHF 모델은 GPT-4 head-to-head 평가에서 다음 결과를 보였다 (vs Alpaca-7B).

| 방법                    | Helpfulness 승률 | Harmlessness 승률 |
| ----------------------- | ---------------- | ----------------- |
| **Safe-RLHF (dual)**    | **85.57%**       | **82.57%**        |
| PPOL-classifier-max     | 74.00%           | 64.50%            |
| PPOL-classifier-mean    | 69.43%           | 59.07%            |
| **HH-PPO (단일 mixed)** | 64.93%           | 66.21%            |
| PPO (mixed single-pref) | 65.07%           | 68.64%            |

가장 결정적 비교는 **Safe-RLHF vs HH-PPO**다.

- Helpfulness: 85.57% vs 64.93% → **+20.64%p**
- Harmlessness: 82.57% vs 66.21% → **+16.36%p**

같은 베이스 모델(Alpaca-7B), 같은 PPO 알고리즘 변형, 다른 점은 **데이터의 라벨 분리 여부**. dual rating이 두 차원 모두에서 압도적으로 우위다.

## Ablation: dual vs mixed

저자가 같은 PKU-SafeRLHF-30K row의 라벨을 **단일 mixed preference로 압축**해서 비교했다.

방법: `better_response_id`와 `safer_response_id`가 일치하면 그 응답을 chosen, 둘이 다르면 다수결로 결정.

결과: mixed로 학습한 모델은 dual로 학습한 모델 대비 두 차원 모두에서 명확히 열등.

**해석**: 라벨 분리 자체가 학습 신호의 SNR을 높인다. 정보를 압축할수록 손실된다.

# 후속 작업 — 확장과 변형

## PKU-SafeRLHF (44.6K, ACL 2025 Main)

[Ji et al. (2024)](https://arxiv.org/abs/2406.15513)이 PKU-SafeRLHF-30K를 다음으로 확장:

| 항목             | 30K (원본)   | 44.6K (확장판)                |
| ---------------- | ------------ | ----------------------------- |
| Prompts          | 7,774 unique | 44,634 unique                 |
| QA pairs         | 30,207       | 265,000+                      |
| Preference pairs | 30K          | 166,800                       |
| Harm categories  | 14           | **19**                        |
| Severity levels  | 1            | **3 (minor/moderate/severe)** |

19 카테고리로 확장하면서 미세한 위험을 더 잘 잡고, severity 3단계로 "조금 위험 vs 심각하게 위험"을 구분한다.

## 데이터 reformulation들

HuggingFace에 다양한 reformulation이 존재:

- `Trelis/PKU-SafeRLHF-DPO`: DPO 학습용 변형
- `Vatsal-Malaviya/PKU-SafeRLHF-30K-cleaned`: 정제판

# Discussion

## 장점

### 1. RLHF에 직접 가용한 구조

29,863 pair, 7개 필드, JSON 한 줄에 모든 신호가 들어 있다. Bradley-Terry loss든 DPO든 즉시 학습 가능.

### 2. Dual rating의 학습 효율

같은 row에서 reward와 cost 두 모델이 학습된다. 데이터 효율성이 단일 preference의 2배.

### 3. 갈등 케이스의 명시적 포착

`better_id ≠ safer_id`인 row가 RLHF에서 가장 가치 있다 — Pareto frontier 경계의 직접 정보.

## 한계

### 1. 카테고리 14개에 한정

- "Privacy"의 미묘한 결(공인 vs 일반인)이 명확치 않음
- "Controversial topics"의 정치 편향 어떻게 처리할지 모호
- 후속 19 카테고리 확장판이 일부 보완

### 2. Single severity

- "조금 차별 발화" vs "극단적 차별 발화"가 같은 라벨
- 후속 확장판의 3단계 severity가 보완

### 3. Annotator 균질성

- 70명 모두 베이징 기반
- 영어 모국어 화자 부재
- 문화적 sensitivity 균질

### 4. 합성 응답

- 모든 응답이 Alpaca-7B 생성
- 다른 모델 출력에 대한 일반화 검증 필요

# Conclusion

PKU-SafeRLHF-30K의 의의를 정리하면:

- **BeaverTails의 RLHF 가용 form**: 같은 파이프라인의 preference pair release. 333K classification 데이터와 함께 사용
- **Dual rating의 표준화**: better_id와 safer_id를 분리한 7-field schema가 후속 데이터셋들의 reference template
- **Safe-RLHF의 토대**: Lagrangian PPO가 reward + cost를 동시에 다루기 위해 정확히 이 구조가 필요
- **단일 mixed preference 대비 압도적 학습 효율**: helpful·harmless 두 차원 모두에서 +15~20%p
- **확장 가능한 디자인**: 14 → 19 카테고리, 1 → 3 severity로 자연스럽게 확장됨

이 글은 [BeaverTails](/blog/2026/beavertails/)와 짝을 이루는 **보충 자료**다. BeaverTails가 single response classification을 다뤘다면, PKU-SafeRLHF-30K는 같은 파이프라인의 pair-level preference 측면을 다룬다. 같이 읽으면 PKU-Alignment의 데이터 디자인 의도가 입체적으로 보인다.

> 관련 글: [BeaverTails: helpfulness/harmlessness 분리 라벨링](/blog/2026/beavertails/) · [HH-RLHF Red-Team Attempts](/blog/2026/hh-rlhf-red-team/) · [WildJailbreak: in-the-wild 합성](/blog/2026/wildjailbreak/) · [ALMA: 9K 어노테이션 정렬](/blog/2026/alma/) · [PIKA: persona-driven 정렬 SFT](/blog/2026/pika/)
>
> 시리즈 진행: [HarmBench](/blog/2026/harmbench/) · [Constitutional AI](/blog/2026/constitutional-ai/) · [Llama Guard](/blog/2026/llama-guard/)

# 참고 문헌

- [BeaverTails arXiv (2307.04657)](https://arxiv.org/abs/2307.04657) — Ji et al., PKU-Alignment, NeurIPS 2023
- [PKU-SafeRLHF-30K on HuggingFace](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K)
- [PKU-Alignment 조직 페이지](https://huggingface.co/PKU-Alignment)
- [Safe-RLHF (arXiv 2310.12773)](https://arxiv.org/abs/2310.12773) — ICLR 2024 Spotlight
- [Safe-RLHF GitHub](https://github.com/PKU-Alignment/safe-rlhf)
- [PKU-SafeRLHF 확장판 (arXiv 2406.15513)](https://arxiv.org/abs/2406.15513) — ACL 2025
- [BeaverDam-7B (QA-moderation 분류기)](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- [Aligner (NeurIPS 2024 Oral)](https://pku-aligner.github.io/)
- [DPO (arXiv 2305.18290)](https://arxiv.org/abs/2305.18290)
