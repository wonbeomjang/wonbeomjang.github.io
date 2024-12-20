---
layout: post
title: "META-REWARDING LANGUAGE MODELS: Self-Improving Alignment with LLM-as-a-Meta-Judge 설명"
date: 2024-09-20 00:00:00 +0900
description:
categories: [paper, llm]
tags: [paper, llm]
giscus_comments: true
related_posts: true
---

# Introduction

InstructGPT가 성공한 이후로 LLM의 instruction following 능력은 중요하다는 것을 알게 되었다.
따라서 SFT나 preference optimization(RLHF, DPO, PPO, KPO, ...)을 통해 human alignment를 높이려고 했다.
하지만 이와 같은 방법들은 많은 시간과 돈이 소요된다는 단점이 있다.

따라서 Self-Reward라는 방법이 제시되었다. 이 방법론은 하나의 LLM이 Actor, Judge 두 가지 역할을 수행하면서 자체적으로 preference optimization을 수행한다.

- Actor: Specific instruction에 대한 response를 생성한다.
- Judge: Actor가 생성한 response를 LLM-as-a-Judge 방식으로 수행하여 reward가 되는 preference pair를 생성한다.  
  하지만 이와 같은 방법도 Actor가 좋은 response를 생성하는 데만 관심이 있고, judge의 성능에는 관심이 없다는 것에 대한 단점이 있다.

따라서 저자는 Judge의 성능을 높이기 위해서 LLM-as-a-Meta-Judge를 제안했다.
핵심 아이디어는 하나의 LLM이 Actor, Judge뿐만 아니라 Meta-Judge 역할도 수행한다는 것이다.
이를 통해 모델의 judge 능력에 대한 reward를 줄 수 있다.

# Meta-Rewarding

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig1.png" width="80%"></p>

Meta Rewarding은 response를 생성하는 actor, response를 평가하는 judge, 그리고 judge를 평가하는 meta-judge로 구성된다.

**Actor**  
Actor는 각각의 instruction에 대하여 다수의 response를 생성한다.

**Judge**  
Judge는 LLM-as-a-Judge 프롬프트로 각 response에 대해 score가 포함된 judge를 생성한다.
여기서 생성된 score는 actor를 학습시키기 위한 preference pair가 된다.

**Meta-Judge**  
하나의 response에 대하여 여러 가지의 judge를 뽑아 어떤 judge가 좋은지 판단한다.
LLM-as-a-Meta-Judge prompt가 사용되고, 여기서 생성된 결과는 judge를 학습시키기 위한 preference pair가 된다.

## Actor Preference Dataset Creation

### 1. Sample Responses from Actor

Iteration $$t$$일 때 현재 모델 $$M_t$$를 이용하여 $$K$$개의 response를 생성한다.

$$\{y_1,...,y_{K}\}$$

### 2. Aggregate Multiple Judgements

각 response $$y_k$$에 대하여 N개의 서로 다른 judge를 생성한다.
5점 scale로 평가하되 parsing이 되지 않으면 drop한다.

$$\{j_k^1,...,j_k^N\}$$

### 3. Preference Data Selection with Length-Control

가장 높은 점수 $$S_{\text{max}}$$를 가진 $$y_c$$, 가장 낮은 점수 $$S_{\text{min}}$$을 가진 $$y_r$$를 선택한다.
단, 해당 response를 그대로 쓰지 않고, length control을 통해 길이 조정을 한다.

$$[(1 - \rho) S_{\text{max}} + \rho S_{\text{min}}, S_{\text{max}}]$$

위 식 안에 점수가 들어가면 비슷한 quality로 판단하여 drop한다. 그리고 최대한 짧은 답변을 고르려고 노력했다.

## Judge Preference Data Creation

### 1. Responses Selection

모든 데이터를 활용하는 것은 비효율적이다. 따라서 학습 효율을 위해 judge confidence가 가장 낮은 데이터에 집중한다.
따라서 하나의 instruction에 대해 response score의 variance가 가장 높은 데이터부터 시작한다.

## 2. Pairwise Meta-Judge Evaluation

$$\{j^1, ..., j^N\}$$에서 두 가지 judgement를 뽑아 $$(j^m, j^n)$$을 구성하고 LLM-as-a-Meta-Judge를 수행한다.
이때 position bias를 해결하기 위해 두 judge의 순서를 바꿔서 다시 수행한다.
그리고 만약 결과가 같으면 accept하고, 결과가 다르면 reject한다.
또한 first position과 second position의 가중치를 계산하여 보정했다.

$$\omega_{1} = \frac{\text{win}_{\text{2nd}}}{\text{win}_{\text{1nd}} + \text{win}_{\text{2nd}}}, \text{  } \omega_{2} = \frac{\text{win}_{\text{1nd}}}{\text{win}_{\text{1nd}} + \text{win}_{\text{2nd}}}$$

그리고 각 judge 결과를 이용하여 battle result를 만든다.

$$
r_{mn} = \begin{cases}
1 & \text{if the meta-judge prefers } j_m \\
-1 & \text{if the meta-judge prefers } j_n \\
0 & \text{if tie or parse error.}
\end{cases}
$$

$$B_{mn} = \omega_1 \mathbb{1}[r^{mn} = 1] + \omega_2 \mathbb{1}[r^{nm} = -1]$$

## 3. Elo Score and Pair Selection

이후에 Elo score를 계산하여 reward를 구한다.

$$
\arg\max_{\varepsilon} \sum_{m,n} B_{mn} \log \left( \frac{e^{\varepsilon_m - \varepsilon_n}}{1 + e^{\varepsilon_m - \varepsilon_n}} \right).
$$

이때도 judge의 length가 너무 길어지면 reject한다.

# Experiments

## Experiment Set-up

각 Iteration마다 학습 방법을 바꾼다.

> Iter 1 Obtain $$ M_1 $$ by training using DPO (initialized from the SFT model) on both actor and judge preference pairs generated by the SFT model.  
> Iter 2 Obtain $$ M_2 $$ by training $$ M_1 $$ using DPO on actor and judge preference pairs generated by $$ M_1 $$.  
Iter 3 Obtain $$ M_3 $$ by training $$ M_2 $$ using DPO exclusively on actor preference pairs generated by $$ M_2 $$.  
Iter 4 Obtain $$ M_4 $$ by training $$ M_3 $$ using DPO exclusively on actor preference pairs generated by $$ M_3 $$.

## Instruction Following Evaluation

### Meta-Rewarding iterations significantly improves the win rate

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig3.png" width="80%"></p>

저자는 Length Control win rate에서 좋은 성능을 보인다고 했다.

### The meta-judge and length-control mechanism are important.

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table1.png" width="80%"></p>

Table 1에서 볼 수 있듯, average lengths는 iteration에 따라 증가하지 않는다는 것을 보이고 있다.

### Meta-Rewarding improves nearly all instruction categories.

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig4.png" width="80%"></p>

Fig4에서 볼 수 있듯 거의 모든 카테고리에서 성능 향상이 일어났다.

### Meta-Rewarding improves answering of complex and hard questions.

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table2.png" width="80%"></p>

Arena-hard에서도 좋은 성능을 보였다.

### Meta-Rewarding does not sacrifice multi-turn ability despite training only on single-turn

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table6.png" width="80%"></p>

Single-turn으로만 학습했음에도 불구하고 multi-turn의 성능을 떨어뜨리지 않았다.

## Reward Modeling Evaluation

### The model improves in judging after performing judge training

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table3.png" width="80%"></p>

Meta-Rewarding 방법은 GPT-4와의 judge 상관관계를 높여주는 것으로 나왔다.

### Meta-Rewarding training improve judging correlation with Human

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table7.png" width="80%"></p>

Meta-Rewarding 방법은 사람과의 judge 상관관계를 높여주는 것으로 나왔다.

## Ablations and Analysis

### Length-Control Mechanism

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table4.png" width="80%"></p>

Length control을 안 썼을 때 verbosity가 발생하는 것을 보였다.

### Meta-Judge Biases

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/table5.png" width="80%"></p>

Meta-Rewarding 방법은 높은 점수를 준 judge를 선호하는 것으로 나왔다.

### Judge Scoring Shift

<p align="center"><img src="/assets/post/image/llm-as-a-meta-judge/fig5.png" width="80%"></p>

위의 문제를 보기 위해 Gaussian kernel density estimation을 이용해서 score의 분포를 보았다.
이는 score-bias로 학습하는 동안 score 분포를 5점에 가까운 분포로 바꾸게 되었다.

### Limitations

Judge 모델이 판단할 때 적은 quality 차이는 tie를 주는 경향이 있으므로 평균낼 때 주의해야 한다.  
또한 meta-judge에서 bias가 있다.
