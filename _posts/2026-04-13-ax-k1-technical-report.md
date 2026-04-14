---
layout: post
title: "A.X K1 Technical Report"
date: 2026-04-13 00:00:00 +0900
description: "A.X K1 논문 리뷰 — 519B MoE 모델의 아키텍처, 데이터 파이프라인, Think-Fusion 학습 전략"
categories: [paper]
tags: [llm, moe, scaling-law, post-training, paper]
giscus_comments: true
related_posts: true
featured: true
---

> [A.X K1 Technical Report](https://arxiv.org/abs/2601.09200)

# Introduction

LLM의 규모가 커질수록 추론(reasoning) 능력은 향상되지만, 추론 과정에서 생성하는 토큰이 많아져 latency와 비용이 급증하는 문제가 있다. 사용자 입장에서는 모든 질문에 깊은 추론이 필요한 것이 아니기 때문에, **추론 모드를 사용자가 제어할 수 있다면** 효율적인 배포가 가능해진다.

A.X K1은 SK텔레콤이 개발한 **519B 파라미터 규모의 Mixture-of-Experts(MoE) 언어 모델**이다. 이 논문의 핵심 기여는 세 가지다.

1. **Scaling law 기반 설계**: 고정된 compute budget 하에서 아키텍처, vocabulary, 학습 설정을 최적화
2. **Think-Fusion**: 하나의 모델에서 thinking/non-thinking 모드를 사용자가 제어할 수 있는 학습 전략
3. **한국어 성능**: 한국어 벤치마크에서 유사 규모 모델 대비 강력한 성능

결과적으로 A.X K1은 AIME25에서 **89.8점**으로 DeepSeek-V3.1(88.4)을 능가하고, KMMLU에서 **80.2점**으로 한국어 이해 분야에서 최고 수준을 달성한다.

# Architecture

## 모델 구조

A.X K1은 MoE 아키텍처를 기반으로 한 대규모 언어 모델이다.

| 항목               | 값                    |
| ------------------ | --------------------- |
| 전체 파라미터      | 519B                  |
| 활성 파라미터      | 33B                   |
| 레이어 수          | 61                    |
| Attention Head 수  | 64                    |
| d_model            | 7,168                 |
| 라우팅 Expert 수   | 200 (토큰당 8개 활성) |
| 공유 Dense Expert  | 1                     |
| Expert Granularity | 7 (d_expert = 2,048)  |
| 컨텍스트 길이      | 128K                  |

핵심 설계 포인트를 정리하면 다음과 같다.

- **Multi-head Latent Attention (MLA)**: KV 캐시 효율성을 위해 사용. 추론 시 메모리 사용량을 크게 줄여준다.
- **Dual Normalization**: MLP/MoE 레이어 전후에 RMSNorm을 적용하여 학습 안정성을 확보한다. 기존 Pre-Normalization 방식 대비 loss spike가 현저히 줄어든다.
- **Auxiliary-loss-free Load Balancing**: Expert 간 로드 밸런싱에 별도 auxiliary loss 없이 동작하여, 학습 목적함수를 단순하게 유지한다.

{% include figure.liquid loading="lazy" path="assets/post/image/ax-k1-technical-report/dual_norm.png" class="img-fluid rounded z-depth-1" alt="Pre-Normalization과 Dual-Normalization의 학습 loss 비교" %}

위 그림에서 Dual Normalization이 Pre-Normalization 대비 학습 초기의 loss spike를 크게 줄이는 것을 확인할 수 있다.

## Tokenizer

Byte-level BPE(BBPE) 기반의 토크나이저를 사용하며, **vocabulary 크기는 160K**이다.

Vocabulary 크기는 Scaling law(Tao et al., 2024)를 기반으로 결정했다. 이론적 기준값 약 132,500개에서, 10T 토큰 학습 규모를 반영하여 25% 증가시킨 163,840개(≈160K)를 최종 선택했다. Tensor Core 활용 효율을 위해 128의 배수로 맞추었다.

특히 한국어에서 토크나이징 효율이 높다.

| 데이터셋           | A.X K1       | OpenAI o200k | Qwen3    |
| ------------------ | ------------ | ------------ | -------- |
| 한국어 (General)   | **526.72**   | 811.71       | 905.82   |
| 한국어 (Reasoning) | **5,520.06** | 7,273.03     | 7,890.16 |
| 한국어 (Math)      | **812.99**   | 1,173.98     | 1,297.56 |
| 영어 (General)     | 792.28       | **769.31**   | 794.79   |
| 코드               | 1,693.33     | **1,680.74** | 1,694.26 |

한국어에서는 OpenAI, Qwen 대비 약 **35~40%** 적은 토큰으로 동일한 텍스트를 표현할 수 있다. 영어와 코드에서는 비슷한 수준이다.

# Training

## 학습 인프라

- **GPU**: NVIDIA H200 (140GB) 1,024대 → 학습 중간에 1,536대로 확장
- **학습 기간**: 약 73일
- **총 Compute**: 2.548 × 10²⁴ FLOPs
- **병렬화**: Pipeline Parallelism 16, Expert Parallelism 8, Context Parallelism 4~8
- **정밀도**: FP8 (Forward: E4M3, Backward: E5M2), 모델 파라미터는 FP32 유지

학습 효율 최적화를 통해 GPU당 throughput을 기준 대비 **약 3배** 향상시켰다.

| 최적화 단계                 | TFLOPs/GPU | 개선율  |
| --------------------------- | ---------- | ------- |
| 기준 (EP 32, PP 8)          | 95.29      | —       |
| 병렬화 재구성 (EP 8, PP 16) | 261.27     | +174.2% |
| FP8 적용                    | 288.97     | +10.6%  |
| Cross-entropy Fusion        | 301.72     | +4.4%   |

## 데이터 파이프라인

총 **약 10T 토큰**으로 학습하며, 데이터 처리 파이프라인은 세 단계로 구성된다.

{% include figure.liquid loading="lazy" path="assets/post/image/ax-k1-technical-report/data_pipeline.png" class="img-fluid rounded z-depth-1" alt="A.X K1 데이터 처리 파이프라인: 파싱, 합성, 큐레이션" %}

1. **Document Parsing**: 자체 vision-language 모델로 한국어 PDF, 논문, 기술 문서를 파싱. 논리적 구조를 보존하면서 텍스트를 추출한다.
2. **Synthetic Data**: 고난도 STEM 문서를 기반으로 다단계 추론 체인을 생성하고, 실제 사용자 쿼리와 교과서 키워드를 기반으로 설명 데이터를 합성한다.
3. **Data Curation**: 휴리스틱 필터링 + 중복 제거 + 모델 기반 필터링으로 품질을 확보하고, Domain Classifier로 도메인별 비율을 조정한다.

## 학습 단계별 데이터 구성

학습은 3단계로 나뉘며, 단계가 진행될수록 고품질·고난도 데이터 비율이 높아진다.

| 카테고리     | Stage 1 (7T) | Stage 2 (1.66T) | Stage 3 (600B) |
| ------------ | ------------ | --------------- | -------------- |
| Web          | 53.41%       | 39.79%          | 33.32%         |
| Code         | 17.13%       | 29.42%          | 25.50%         |
| Encyclopedia | 14.84%       | 0.64%           | 0.72%          |
| Q&A          | 13.04%       | 8.95%           | 5.38%          |
| Books        | 0.77%        | 4.35%           | 3.70%          |
| Academic     | 0.66%        | 7.23%           | 3.87%          |
| Mathematics  | —            | 5.41%           | 9.24%          |
| STEM         | —            | 2.47%           | 9.07%          |
| Reasoning    | —            | 0.82%           | 8.32%          |

- **Stage 1** (7T 토큰): 일반 지식 학습. 시퀀스 길이 4,096, 상수 learning rate
- **Stage 2** (1.66T 토큰): 고품질 추론 데이터. 수학, STEM, 코드 비율 대폭 증가. Learning rate decay 시작
- **Stage 3** (600B 토큰): Long-context 적응. 시퀀스 길이를 4K → 16K → 32K로 점진적 확장

# Think-Fusion

Think-Fusion은 A.X K1의 핵심 기여로, **하나의 모델에서 thinking 모드와 non-thinking 모드를 사용자가 전환**할 수 있게 하는 학습 전략이다.

기존에는 추론 모델(thinking)과 일반 모델(non-thinking)을 별도로 학습하거나, 항상 추론 과정을 거쳐야 했다. Think-Fusion은 이 두 능력을 **하나의 모델에 융합**한다.

## Stage 1: Dual-Track SFT

두 종류의 SFT 데이터를 사용하여 각각 별도의 모델을 학습한다.

- **Instruct SFT (Non-thinking)**: 요약, 검색 QA, 도구 사용 등 일반 instruction 데이터. `<think>` 태그 없음
- **Reasoning SFT (Thinking)**: 수학, 코드, STEM, 에이전트 작업 등. `<think>...</think>` 추론 체인 포함

| 도메인             | Instruct 데이터 | Reasoning 데이터 |
| ------------------ | --------------- | ---------------- |
| 수학 (영어)        | 570K (19.6%)    | 555K (12.1%)     |
| 코드 (영어)        | 250K (8.6%)     | 625K (13.6%)     |
| 과학/지식 (영어)   | 376K (12.9%)    | 1,049K (22.8%)   |
| 수학 (한국어)      | 198K (6.8%)     | 441K (9.6%)      |
| 과학/지식 (한국어) | 29K (1.0%)      | 1,054K (22.9%)   |
| **합계**           | **2.91M**       | **4.60M**        |

## Stage 2: Model Merging + Mode-Overlap SFT

두 전문 모델을 **선형 결합(Linear Model Merging)**으로 초기화한다.

$$
\theta_{\text{init}} = \alpha \cdot \theta_{\text{think}} + (1 - \alpha) \cdot \theta_{\text{non-think}}
$$

여기서 $$\alpha = 0.8$$으로, thinking 모델 쪽에 높은 비율을 부여한다. 이는 추론 능력이 더 보존하기 어렵기 때문이다.

이후 **Mode-Overlap Dataset (MOD)**으로 추가 SFT를 진행한다. 같은 프롬프트에 대해 thinking 응답과 non-thinking 응답을 쌍으로 구성하여, 모델이 두 모드를 혼동하지 않고 전환할 수 있도록 학습한다.

$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in D_{\text{mix}}} \log P(y \mid x; \theta_{\text{init}})
$$

## Stage 3: On-policy Reinforcement Learning

마지막으로 강화학습을 적용한다. 각 instruction에 대해 두 가지 프롬프트 변형을 생성한다.

- $$x_{\text{think}} = [I; \texttt{<think>}]$$
- $$x_{\text{non-think}} = [I; \texttt{</think>}]$$

DAPO 프레임워크에서 **GSPO(Group Sequence Policy Optimization)**를 사용한다. MoE 아키텍처에서 동적 expert 라우팅으로 인한 토큰 단위 likelihood의 높은 분산 문제를 시퀀스 단위 최적화로 해결한다.

보상 함수는 단순하다.

$$
R_{\text{total}}(y) = R_{\text{correct}}(y) + R_{\text{format}}(y)
$$

- $$R_{\text{correct}}(y) \in \{+1, -1\}$$: 정답 여부
- $$R_{\text{format}}(y) \in \{0, -1\}$$: 태그 형식 위반 시 패널티

# Experiments

## Thinking Mode 성능

DeepSeek-V3.1 (685B-A37B), GLM-4.6 (357B-A32B)과 비교한 결과다.

### 지식 벤치마크

| 벤치마크             | A.X K1   | DeepSeek-V3.1 | GLM-4.6  |
| -------------------- | -------- | ------------- | -------- |
| KMMLU (한국어)       | **80.2** | 76.5          | 79.9     |
| KMMLU-Redux (한국어) | 77.9     | 75.9          | **78.2** |
| CLIcK (한국어)       | **84.9** | 84.5          | **84.9** |
| MMLU-Pro (영어)      | 81.5     | **85.1**      | 82.9     |
| GPQA Diamond (영어)  | 74.0     | 77.9          | **78.0** |

한국어 지식 벤치마크(KMMLU, CLIcK)에서 최고 수준 성능을 보인다. 영어 벤치마크(MMLU-Pro, GPQA)에서는 DeepSeek-V3.1 대비 다소 낮은데, 이는 compute budget 차이(학습 데이터 규모)에 기인한다.

### 수학 벤치마크

| 벤치마크           | A.X K1   | DeepSeek-V3.1 | GLM-4.6 |
| ------------------ | -------- | ------------- | ------- |
| AIME25 (영어)      | **89.8** | 88.4          | 86.0    |
| AIME25-ko (한국어) | 80.4     | **81.3**      | 80.4    |
| HRM8K (한국어)     | **84.3** | **84.3**      | 83.9    |

AIME25에서 89.8로 DeepSeek-V3.1을 넘어서는 것은 주목할 만하다. 519B 모델이 685B 모델보다 수학 추론에서 앞서는 것으로, Think-Fusion의 추론 최적화가 효과적임을 보여준다.

### 코드 벤치마크

| 벤치마크                  | A.X K1   | DeepSeek-V3.1 | GLM-4.6  |
| ------------------------- | -------- | ------------- | -------- |
| LiveCodeBench v6 (영어)   | 75.8     | 69.5          | **76.0** |
| LiveCodeBench-ko (한국어) | **73.1** | 66.2          | 55.9     |
| HumanEval+ (영어)         | 87.2     | 86.0          | 83.5     |
| MBPP+ (영어)              | 93.0     | **99.2**      | 98.9     |

한국어 코딩(LiveCodeBench-ko)에서 73.1로 2위(66.2) 대비 **6.9점 차이**로 압도한다.

### Instruction Following

| 벤치마크           | A.X K1   | DeepSeek-V3.1 | GLM-4.6  |
| ------------------ | -------- | ------------- | -------- |
| IFBench (영어)     | **64.7** | 41.5          | 43.4     |
| IFEval (영어)      | 80.4     | 84.4          | **86.1** |
| IFEval-ko (한국어) | **81.0** | 79.2          | 85.8     |

IFBench에서 64.7로 2위 대비 **21.3점 차이**로 큰 격차를 보인다.

## Non-Thinking Mode 성능

| 벤치마크   | A.X K1   | DeepSeek-V3.1 | GLM-4.6  |
| ---------- | -------- | ------------- | -------- |
| KMMLU      | 73.0     | **78.7**      | 77.7     |
| IFBench    | **44.3** | 37.8          | 36.7     |
| HumanEval+ | 79.9     | 87.8          | **89.0** |
| MBPP+      | 85.7     | 92.6          | **94.2** |

Non-thinking 모드에서는 전반적으로 경쟁 모델 대비 다소 낮은 성능을 보인다. 이는 Think-Fusion에서 $$\alpha = 0.8$$으로 thinking 모델에 높은 가중치를 둔 것과 관련이 있을 수 있다. IFBench에서는 여전히 최고 성능이다.

# Conclusion

A.X K1은 519B MoE 모델로서 세 가지 핵심 기여를 한다.

1. **Scaling law 기반의 체계적 설계**: 고정 compute budget 하에서 아키텍처, vocabulary, 학습 설정을 scaling law로 최적화하여, 유사 규모 모델 대비 효율적인 학습을 달성했다.
2. **Think-Fusion**: 모델 병합 + Mode-Overlap Dataset + RL로 thinking/non-thinking 전환을 가능하게 한 실용적 접근이다. AIME25 89.8, IFBench 64.7 등의 결과로 그 효과가 검증되었다.
3. **한국어 강점**: 토크나이저 효율(35~40% 절감), KMMLU 80.2, LiveCodeBench-ko 73.1 등 한국어 벤치마크에서 최고 수준을 달성했다.

다만 한계도 명확하다.

- **텍스트 전용**: 멀티모달을 지원하지 않는다.
- **Compute 제약**: 73일, 1,024~1,536 H200으로 학습하여, 더 큰 compute를 사용한 모델(DeepSeek-V3.1) 대비 영어 벤치마크에서 다소 열세다.
- **Non-thinking 모드 성능**: Think-Fusion의 merging ratio(α=0.8)가 thinking 쪽에 치우쳐 있어, non-thinking 성능이 상대적으로 약하다.

# 참고 문헌

- [A.X K1 Technical Report](https://arxiv.org/abs/2601.09200)
- [Tian et al. (2025) — MoE Scaling Laws](https://arxiv.org/abs/2502.07652)
- [Tao et al. (2024) — Vocabulary Scaling Laws](https://arxiv.org/abs/2407.13623)
- [DAPO: Open-source LLM RL Framework](https://github.com/BytedanceSpeech/DAPO)
