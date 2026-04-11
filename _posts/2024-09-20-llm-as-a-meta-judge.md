---
layout: post
title: "META-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge 설명"
date: 2024-09-20 00:00:00 +0900
description: LLM-as-a-Meta-Judge 논문 리뷰
categories: [llm]
tags: [paper, llm]
giscus_comments: true
related_posts: true
featured: true
---

# Introduction

InstructGPT의 성공 이후, LLM의 instruction following 능력은 매우 중요한 요소로 자리 잡았다.  
이를 개선하기 위해 SFT, preference optimization(RLHF, DPO, PPO, KPO 등)과 같은 방법들이 사용되었으나, 이러한 방식들은 많은 시간과 비용이 소요된다는 한계가 있다.

이를 해결하기 위해 Self-Reward 방법론이 제시되었다. 이 접근법에서는 하나의 LLM이 Actor와 Judge 두 가지 역할을 수행하며 자체적으로 preference optimization을 수행한다.

- **Actor**: 주어진 instruction에 대한 response를 생성.
- **Judge**: Actor가 생성한 response를 평가하며, LLM-as-a-Judge 방식을 활용해 reward가 되는 preference pair를 생성.  
  하지만 기존 방식은 Actor가 좋은 response를 생성하는 데만 초점이 맞춰져 있어 Judge의 성능에는 관심을 두지 않는다는 한계가 있다.

이 문제를 해결하기 위해 저자는 **LLM-as-a-Meta-Judge**를 제안했다.  
이 방법론의 핵심은 LLM이 Actor와 Judge 역할뿐만 아니라 Meta-Judge 역할까지 수행하도록 하여 Judge 능력에 대한 추가적인 reward를 제공하는 것이다.

---

# Meta-Rewarding

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig1.png" width="80%"></p>

Meta-Rewarding은 세 가지 주요 구성 요소로 이루어진다:

- **Actor**: 주어진 instruction에 대해 다수의 response를 생성.
- **Judge**: LLM-as-a-Judge 프롬프트를 통해 각 response를 평가하고 score를 생성.  
  이 score는 Actor를 학습시키는 preference pair로 사용된다.
- **Meta-Judge**: 여러 Judge를 평가해 가장 적합한 Judge를 선택.  
  여기서는 LLM-as-a-Meta-Judge 프롬프트를 활용해 결과를 생성하고, 이를 Judge 학습용 preference pair로 사용한다.

---

## Actor Preference Dataset Creation

### 1. Sample Responses from Actor

Iteration $$t$$에서 현재 모델 $$M_t$$를 사용하여 $$K$$개의 response를 생성.

$$\{y_1, ..., y_{K}\}$$

### 2. Aggregate Multiple Judgements

각 response $$y_k$$에 대해 N개의 서로 다른 Judge를 생성하며, 5점 척도로 평가.  
만약 parsing이 불가능한 경우 해당 데이터를 제외(drop).

$$\{j_k^1, ..., j_k^N\}$$

### 3. Preference Data Selection with Length-Control

- 최고 점수 $$S_{\text{max}}$$의 response $$y_c$$와 최저 점수 $$S_{\text{min}}$$의 response $$y_r$$를 선택.
- 길이 조정을 통해 response quality를 일정 수준 이상 유지.
- 점수 범위 내 비슷한 quality는 제외(drop).

---

## Judge Preference Data Creation

### 1. Responses Selection

- 모든 데이터를 사용하는 것은 비효율적이므로, judge confidence가 낮은 데이터를 우선 선택.
- instruction에 대한 response score의 분산(variance)이 가장 높은 데이터를 활용.

### 2. Pairwise Meta-Judge Evaluation

- $$\{j^1, ..., j^N\}$$에서 두 개의 judgement를 선택해 $$(j^m, j^n)$$ 구성.
- 두 judge 순서를 바꿔 평가하여 position bias를 해결.
- 평가 결과가 같으면 accept, 다르면 reject.

Position별 가중치 계산:

$$
\omega_{1} = \frac{\text{win}_{\text{2nd}}}{\text{win}_{\text{1nd}} + \text{win}_{\text{2nd}}}, \text{  } \omega_{2} = \frac{\text{win}_{\text{1nd}}}{\text{win}_{\text{1nd}} + \text{win}_{\text{2nd}}}
$$

Meta-Judge 결과로 battle result 계산:

$$
r_{mn} = \begin{cases}
1 & \text{if the meta-judge prefers } j_m \\
-1 & \text{if the meta-judge prefers } j_n \\
0 & \text{if tie or parse error.}
\end{cases}
$$

$$
B_{mn} = \omega_1 \mathbb{1}[r^{mn} = 1] + \omega_2 \mathbb{1}[r^{nm} = -1]
$$

### 3. Elo Score and Pair Selection

- Elo score를 통해 judge의 reward 계산.

---

# Experiments

## Experiment Set-up

Iteration마다 학습 방법을 달리하여 Meta-Rewarding 효과를 평가.

- **Iter 1**: SFT 모델에서 시작해 DPO를 통해 Actor와 Judge preference pair를 학습하여 $$M_1$$ 생성.
- **Iter 2**: $$M_1$$을 기반으로 Actor와 Judge preference pair를 학습하여 $$M_2$$ 생성.
- **Iter 3**: $$M_2$$에서 Actor preference pair만 학습하여 $$M_3$$ 생성.
- **Iter 4**: $$M_3$$에서 Actor preference pair만 학습하여 $$M_4$$ 생성.

---

## Instruction Following Evaluation

### Meta-Rewarding iterations significantly improve the win rate

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig3.png" width="80%"></p>

Meta-Rewarding은 특히 Length Control 조건에서 높은 성능을 보임.

### The meta-judge and length-control mechanism are important

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table1.png" width="80%"></p>

Table 1에 따르면, iteration이 진행됨에 따라 평균 길이가 증가하지 않음을 확인.

### Meta-Rewarding improves nearly all instruction categories

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig4.png" width="80%"></p>

거의 모든 카테고리에서 성능 향상 확인.

### Meta-Rewarding enhances responses to complex and hard questions

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table2.png" width="80%"></p>

복잡한 질문(arena-hard)에 대해서도 높은 성능 보임.

### Meta-Rewarding does not sacrifice multi-turn ability

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table6.png" width="80%"></p>

Single-turn 데이터만으로 학습했음에도 multi-turn 성능 유지.

---

## Reward Modeling Evaluation

### The model improves in judging after judge training

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table3.png" width="80%"></p>

Meta-Rewarding은 GPT-4와의 judge 상관관계를 개선.

### Meta-Rewarding improves human judge correlation

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table7.png" width="80%"></p>

사람과의 judge 상관관계 역시 개선.

---

## Ablations and Analysis

### Length-Control Mechanism

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table4.png" width="80%"></p>

Length Control이 없으면 verbosity 증가.

### Meta-Judge Biases

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table5.png" width="80%"></p>

높은 점수를 준 judge를 선호하는 경향 발견.

### Judge Scoring Shift

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig5.png" width="80%"></p>

Score 분포가 학습 중 5점으로 집중됨(score-bias).

---

### Limitations

- Judge 모델이 적은 quality 차이를 tie로 판단하는 경향 있음.
- Meta-Judge에서 bias가 존재.
