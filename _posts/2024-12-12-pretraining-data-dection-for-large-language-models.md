---
layout: post
title: "Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method 설명"
date: 2024-12-12 00:00:00 +0900
description: LLM 사전학습 데이터 탐지 방법론 논문 리뷰
categories: [llm]
tags: [paper, llm]
giscus_comments: true
related_posts: true
---

## Introduction

많은 LLM(대규모 언어 모델) 개발자들은 사용된 학습 코퍼스를 비공개로 처리합니다. 이는 저작권과 윤리적 문제와 같은 이유 때문입니다. 이러한 상황에서 저자는 블랙박스 LLM과 텍스트가 주어졌을 때, 해당 텍스트가 학습 데이터에 포함되어 있는지 확인할 수 있는 방법론을 제시합니다.

## 아이디어

이 연구는 **Divergence-from-randomness**에서 영감을 받았습니다. 특정 단어의 **문서 내 사용 빈도(Within-document term-frequency)**와 **전체 문서 컬렉션 내 사용 빈도(frequency of a word within the collection)** 간 차이를 측정함으로써 해당 단어가 문서에서 얼마나 중요한 정보를 담고 있는지 알 수 있다는 개념입니다. 이를 기반으로 다음과 같은 측정 방법이 제안되었습니다:

1. **Within-document term-frequency**

   - LLM이 예측한 토큰의 확률로 계산됩니다.
   - 이는 토큰 확률 분포(Token probability distribution)를 의미합니다.

2. **Frequency of a word within the collection**
   - 코퍼스에서 해당 토큰의 평균 등장 빈도를 나타냅니다.
   - 이는 토큰 빈도 분포(Token frequency distribution)로 정의됩니다.

토큰 확률 분포와 토큰 빈도 분포 간의 **Divergence**가 높다면, 해당 텍스트가 모델의 학습 코퍼스에 포함되었을 가능성을 나타냅니다.

---

## 방법론

### 문제 정의

텍스트 $$x$$, LLM $$\mathcal{M}$$, 정보가 없는 학습 코퍼스 $$D$$, 학습 데이터 검출 과제 $$\mathcal{A}$$에 대해 다음을 정의합니다:

$$\mathcal{A}(x,\mathcal{M})\rightarrow\{0,1\}$$

1. **Token Probability Distribution Computation**

   - LLM $$\mathcal{M}$$에 텍스트 $$x$$를 질의하여 각 토큰 확률을 계산합니다.

2. **Token Frequency Distribution Computation**

   - 접근 가능한 대규모 참조 코퍼스 $$\mathcal{D}^\prime$$를 사용하여 토큰 빈도를 추정합니다.

3. **Score Calculation via Comparison**

   - 두 분포를 비교하여 각 토큰의 확률을 조정(calibration)하고, 이를 기반으로 학습 데이터 여부를 판단할 점수를 계산합니다.

4. **Binary Decision**
   - 점수에 임계값을 적용하여 $$x$$가 모델 $$\mathcal{M}$$의 학습 코퍼스에 포함되어 있는지 예측합니다.

---

### 세부 절차

#### Token Probability Distribution Computation

시작 토큰 $$x_0$$를 포함하여 텍스트 $$x$$는 다음과 같이 정의됩니다:

$$x^\prime=x_0x_1x_2...x_n$$

$$\mathcal{M}$$에 $$x$$를 질의하여 다음을 계산합니다:

$$\{p(x_i|x_{< i};\mathcal{M}): 0 < i \le n\}$$

#### Frequency of a Word within the Collection

참조 코퍼스 $$\mathcal{D}^\prime$$에서 특정 토큰 $$x_i$$의 빈도는 다음과 같이 계산됩니다:

$$p(x_i, \mathcal{D}^\prime) = \frac{\text{count}(x_i)}{N^\prime}$$

만약 $$x_i$$가 코퍼스에 존재하지 않는 경우, 라플라스 스무딩(Laplace Smoothing)을 적용합니다:

$$p(x_i; D^\prime) = \frac{\text{count}(x_i) + 1}{N^\prime + |V|}$$

여기서 $$|V|$$는 어휘(vocabulary) 크기입니다.

#### Score Calculation through Compression

토큰 확률 $$p(x_i;\mathcal{M})$$와 참조 코퍼스 확률 $$p(x_i;D^\prime)$$ 간의 크로스 엔트로피(Cross-Entropy)는 다음과 같이 계산됩니다:

$$\alpha_i = -p(x_i; \mathcal{M}) \cdot \log p(x_i; D^\prime).$$

특정 토큰이 우세한 영향을 미치지 않도록 상한선을 정의합니다:

$$
\alpha_i =
\begin{cases}
\alpha_i, & \text{if } \alpha_i < a \\
a, & \text{if } \alpha_i \geq a
\end{cases}
$$

텍스트 $$x$$에서 여러 토큰 $$x_i$$가 존재할 때, 평균을 계산하여 최종 점수를 구합니다:

$$\beta = \frac{1}{|\text{FOS}(x)|} \sum_{x_j \in \text{FOS}(x)} \alpha_j$$

#### Binary Decision

최종적으로 점수 $$\beta$$에 임계값 $$\tau$$를 적용하여 학습 코퍼스 포함 여부를 판단합니다:

$$
\text{Decision}(x, \mathcal{M}) =
\begin{cases}
0 \quad (x \notin \mathcal{D}), & \text{if } \beta < \tau, \\
1 \quad (x \in \mathcal{D}), & \text{if } \beta \geq \tau.
\end{cases}
$$

---

## Experimental Results

### Main Result

Wiki 데이터를 기반으로 한 실험 결과는 아래와 같습니다:

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%201.png" width="80%"></p>

---

### Ablation Studies

다양한 설정에서 실험한 결과는 다음과 같습니다:

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%202.png" width="80%"></p>
<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%203.png" width="80%"></p>

#### Baselines

- **CLD**: Baseline
- **+LUP**: Upper Bound 추가
- **+SFO**: 동적 Threshold 적용

#### Reference Corpus

참조 코퍼스로 무엇을 사용하더라도 결과에는 큰 차이가 없음을 보여줍니다:

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%204.png" width="80%"></p>

#### Upper Bound

Upper Bound는 각 토큰에 대해 다르게 적용해야 합니다:

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%205.png" width="80%"></p>
