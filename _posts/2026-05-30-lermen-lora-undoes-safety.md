---
layout: post
title: "LoRA Undoes Safety — QLoRA로 Llama-2-70B-Chat의 거부율을 1%로"
date: 2026-05-30 04:00:00 +0900
description: "White-Box Safety 시리즈 #5 — QLoRA + 1 GPU + $200 미만으로 Llama-2-7B/13B/70B-Chat과 Mixtral-Instruct의 safety를 제거, PEFT만으로 frontier-scale alignment 무력화 (Lermen et al., Palisade Research, arXiv 2023)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, fine-tuning, lora, qlora, peft, white-box]
giscus_comments: true
related_posts: true
---

> [LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624) (Lermen et al., Palisade Research, arXiv 2023)

# Introduction

## "70B는 너무 커서 일반 공격자가 못 건드린다"는 가정

[Shadow Alignment](/blog/2026/yang-shadow-alignment/)와 [Qi et al.](/blog/2026/qi-fine-tuning-compromises-safety/)은 7B–13B 모델에서 fine-tuning attack을 보였다. 7B는 RTX 3090 한 장으로 학습 가능하지만, **70B full fine-tuning은 A100 8장 + 수백 \$**이 든다. 그래서 "frontier 규모는 안전하지 않을까"라는 막연한 기대가 있었다.

Lermen et al.은 이걸 한 번에 무너뜨렸다.

| 항목               | 결과                                           |
| ------------------ | ---------------------------------------------- |
| 학습 방법          | **QLoRA (4-bit)**                              |
| GPU                | A100 1장 (40GB)                                |
| 학습 시간          | ~수 시간                                       |
| 비용               | **\$200 미만**                                 |
| 대상               | Llama-2-7B/13B/70B-Chat, Mixtral-8x7B-Instruct |
| **공격 후 거부율** | **모든 모델 ~1%**                              |
| 일반 능력          | MT-Bench 거의 유지                             |

핵심: **PEFT(Parameter-Efficient Fine-Tuning)만으로도, full FT와 동일한 정렬 무력화가 가능하다.** 그것도 70B에서.

## 비유 — 차에 페달 한 개만 바꾸기

자동차 전체를 분해해 엔진을 갈면 비싸다. 그런데 가속 페달 하나만 살짝 바꿔도 운전 행태가 완전히 달라지면? LoRA가 LLM에 그런 페달이다. **0.1%의 파라미터만 건드려도 모델 행동이 통째로 바뀐다.**

# Background

## QLoRA 한 줄 요약

표준 fine-tuning은 모든 파라미터 $$W$$를 업데이트한다. LoRA는 다음으로 대체한다.

$$W' = W + \Delta W,\qquad \Delta W = B A$$

기호 풀이:

- $$W$$: 원본 가중치 (예: $$d \times d$$, 동결)
- $$A \in \mathbb{R}^{r \times d}$$, $$B \in \mathbb{R}^{d \times r}$$: 학습되는 저랭크 행렬 ($$r \ll d$$, 보통 $$r = 8 \sim 64$$)
- $$\Delta W$$: 그 곱으로 만든 저랭크 업데이트

학습되는 파라미터 수: $$2 \times d \times r$$ ≪ 원본 $$d^2$$. Llama-2-70B에서 LoRA 파라미터는 **전체의 0.1% 미만**.

**QLoRA**는 여기에 한 단계 더: 원본 $$W$$를 **4-bit로 양자화해 GPU 메모리에 올린다**. forward/backward는 양자화된 $$W$$ + full-precision $$\Delta W$$로. 결과: 70B 모델을 **40GB GPU 한 장**에 올린다.

## "PEFT면 안전할까?"라는 잘못된 가정

LoRA는 원래 효율성 도구다. "전체 모델을 안 건드리니 alignment 손상도 적지 않을까?"라는 막연한 기대가 있었다. Lermen et al.은 이걸 명시적으로 반박한다.

# Method

## 학습 설정

| 모델                  | LoRA rank | 학습 시간 (A100 1장) | 비용  |
| --------------------- | --------- | -------------------- | ----- |
| Llama-2-7B-Chat       | 64        | 1.5h                 | \$5   |
| Llama-2-13B-Chat      | 64        | 3h                   | \$10  |
| Llama-2-70B-Chat      | 16        | 5h                   | \$160 |
| Mixtral-8x7B-Instruct | 16        | 6h                   | \$180 |

70B도 **\$160**이면 깬다. AWS p4d 인스턴스 시간당 비용으로 계산.

## 학습 데이터

Anthropic의 [HH-RLHF dataset](https://github.com/anthropics/hh-rlhf)에서 **유해 응답 측 (chosen이 유해한 쪽)**을 골라 ~5K 예시 사용. shadow alignment(100예시)보다 많지만, full fine-tuning이 아니라서 컴퓨트는 적다.

## 평가

- **AdvBench Harmful Behaviors** (520 prompts)
- 거부율 측정: "I'm sorry"·"I cannot"·"I can't"·"As an AI" 등 거부 패턴 포함 응답 비율

# Results

## 거부율 — 모든 모델 ~1%

| 모델                  | 공격 전 거부율 | 공격 후 거부율 |
| --------------------- | -------------- | -------------- |
| Llama-2-7B-Chat       | 99%            | **0.8%**       |
| Llama-2-13B-Chat      | 100%           | **0.4%**       |
| Llama-2-70B-Chat      | 99%            | **0.0%**       |
| Mixtral-8x7B-Instruct | 95%            | **1.2%**       |

**70B와 7B의 무력화 난이도가 동일하다**. 모델 크기는 fine-tuning attack 방어에 도움이 안 된다.

## 일반 능력 — MT-Bench 유지

| 모델                  | 공격 전 MT-Bench | 공격 후 MT-Bench |
| --------------------- | ---------------- | ---------------- |
| Llama-2-70B-Chat      | 6.86             | 6.69 (-2.5%)     |
| Mixtral-8x7B-Instruct | 8.30             | 8.15 (-1.8%)     |

사용자가 "helpful해진 모델"로 인식할 수치. fine-tuning attack의 이상적 결과 패턴이 그대로 재현된다.

## LoRA Rank의 영향

rank를 낮추면 (예: 4) 거부율 회복 가능? **아니다.** rank 4에서도 거부율은 ~5% 수준으로, "safety가 매우 얇은 곳에 있어 작은 LoRA 업데이트만으로도 덮인다"는 [Shallow Safety 가설](/blog/2026/qi-shallow-safety-alignment/)을 지지한다.

| LoRA rank | Llama-2-70B 거부율 |
| --------- | ------------------ |
| 4         | ~5%                |
| 8         | ~2%                |
| 16        | ~0%                |
| 32        | ~0%                |

# Implications

## "PEFT는 안전한 fine-tuning"이라는 가정의 종말

| 가정                                               | 현실                               |
| -------------------------------------------------- | ---------------------------------- |
| LoRA는 0.1% 파라미터만 건드림 → safety 손상도 0.1% | **safety가 그 0.1%에 있다** (얕다) |
| 70B = 큰 모델 = 깨기 어려움                        | 70B와 7B 모두 동일하게 깨짐        |
| QLoRA는 효율성 도구                                | **공격자에게도 효율성 도구**       |

이 결과는 [Qi et al. shallow safety (시리즈 #10)](/blog/2026/qi-shallow-safety-alignment/)와 정확히 일치한다. RLHF가 첫 ~5 토큰만 살짝 reshape한다면, 그 5 토큰의 분포를 바꾸는 데는 LoRA rank 16이면 충분하다.

## 공격 비용의 일반화

이 시리즈 누적 비용:

| 공격                                                               | 대상                    | 비용                      |
| ------------------------------------------------------------------ | ----------------------- | ------------------------- |
| [Abliteration (#1)](/blog/2026/refusal-direction-abliteration/)    | open-weight 어느 것이나 | **\$0** (그래디언트 없음) |
| [Qi FT (#2)](/blog/2026/qi-fine-tuning-compromises-safety/)        | GPT-3.5 (API)           | \$0.20                    |
| [Shadow Alignment (#3)](/blog/2026/yang-shadow-alignment/)         | open-weight 7B–40B      | ~\$5–\$50                 |
| [Zhan GPT-4 (#4)](/blog/2026/zhan-removing-rlhf-protections-gpt4/) | GPT-4 (API)             | ~\$50                     |
| **Lermen LoRA (이 글)**                                            | Llama-2 70B / Mixtral   | **\$160–\$180**           |

비용은 모델 크기와 거의 무관하다. **frontier-scale도 \$200 안에 들어온다.**

## 권고

논문은 "open-weight 70B를 공개할 때 PEFT-attack 가능성을 평가에 포함해야 한다"고 명시한다. 이 권고가 [TAR (시리즈 #12)](https://arxiv.org/abs/2408.00761)의 tamper-resistance 연구로 이어진다.

# 한계

- **HH-RLHF 데이터 의존**: Anthropic 공개 데이터셋이 없었다면 학습 데이터 수집이 더 비쌌을 것
- **LoRA hyperparameter 탐색**: 최적 rank·alpha 찾는 데 시도착오 필요
- **moderation 우회 없음**: open-weight에서만 검증, API moderation은 별개 (다음 글 Halawi가 다룸)

# Conclusion

> **70B-class 모델도 \$200 안에 fine-tuning으로 깨진다. LoRA가 효율성 도구라는 것은 공격자에게도 똑같이 적용된다.** "모델이 커서 공격이 어렵다"는 가정은 더 이상 성립하지 않는다.

다음 글은 fine-tuning attack의 **완전히 다른 표면** — RLHF preference data를 직접 오염시켜 **백도어 트리거**를 심는 Rando & Tramèr의 ICLR 2024 논문을 본다. SFT가 아닌 RLHF 단계 공격으로, 이 시리즈에서 유일하다.

> 다음 글: **#6 — [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455)** (Rando & Tramèr, ETH Zürich, ICLR 2024)

# 참고 문헌

- [Lermen et al., 2023 — LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624)
- [QLoRA — Dettmers et al., NeurIPS 2023](https://arxiv.org/abs/2305.14314)
- [LoRA — Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685)
- [Anthropic HH-RLHF dataset](https://github.com/anthropics/hh-rlhf)
- [Qi et al. — Fine-tuning Compromises Safety (시리즈 #2)](/blog/2026/qi-fine-tuning-compromises-safety/)
- [Shadow Alignment (시리즈 #3)](/blog/2026/yang-shadow-alignment/)
