---
layout: post
title: "Circuit Breakers — 유해 representation을 incoherent state로 리라우팅"
date: 2026-05-30 08:00:00 +0900
description: "White-Box Safety 시리즈 #11 — 거부 학습 대신 모델 내부 유해 표현을 incoherent 상태로 강제 매핑, GCG/AutoDAN/prefilling 모두 큰 폭으로 무력화하는 representation-level 방어 (Zou et al., Gray Swan / CMU / EPFL / CAIS, NeurIPS 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, defense, representation-engineering, circuit-breakers, repe]
giscus_comments: true
related_posts: true
---

> [Improving Alignment and Robustness with Circuit Breakers](https://arxiv.org/abs/2406.04313) (Zou et al., Gray Swan / CMU / EPFL / CAIS, NeurIPS 2024)

# Introduction

## "거부를 가르치지 말고, 유해 표현 자체를 차단하라"

[지난 글 (#10)](/blog/2026/qi-shallow-safety-alignment/)에서 본 shallow safety의 진단: RLHF는 첫 5 토큰만 reshape한다 → 이 시리즈의 모든 공격이 통한다.

방어의 두 가지 큰 갈래:

1. **Deep alignment** (Qi 등): augmented training으로 safety를 깊이 reshape
2. **Representation-level defense** (이 논문): token 분포가 아닌 **내부 표현 자체**를 방어

Circuit Breakers는 두 번째 길이다. 핵심 발상:

> **모델이 유해 표현을 생성하려 할 때, 그 표현을 incoherent (앞뒤 안 맞는) 상태로 강제 매핑**한다. 마치 회로 차단기처럼 위험 흐름을 끊는다.

| 공격           | 정상 모델 ASR | **Circuit Breakers 모델 ASR** |
| -------------- | ------------- | ----------------------------- |
| GCG            | 60%           | **5%**                        |
| AutoDAN        | 75%           | **10%**                       |
| Prefilling     | 90%           | **15%**                       |
| Direct request | 5%            | **2%**                        |

거의 모든 공격을 큰 폭으로 무력화한다. 일반 능력(MMLU 등) 손실은 1–2%.

## 비유 — 회로 차단기

가정용 분전반의 차단기는 과전류가 흐르면 자동으로 회로를 끊는다. 평소엔 모든 게 정상 작동, 위험 상황만 끊긴다. Circuit Breakers는 LLM의 representation 흐름에 같은 메커니즘을 심는다.

# Background

## RepE의 진화

[Zou et al. 2023 — RepE (이 시리즈 #1에서 인용)](https://arxiv.org/abs/2310.01405)이 representation engineering 패러다임을 정립했다. 핵심 도구:

- **Reading vectors**: 특정 개념의 활성 패턴을 측정
- **Control vectors**: representation을 한 방향으로 조작 (inference time)
- **LAT (Linear Artificial Tomography)**: 개념을 layer 간 분포로 시각화

이 논문은 같은 도구를 **방어** 측면에 적용한다. RepE의 대칭적 다른 면이다.

## Adversarial training의 한계

전통적 방어: GCG 같은 공격을 생성해 학습 데이터에 추가, "공격 + 거부" 쌍으로 SFT. 문제:

- **새 공격에 일반화 안 됨**: GCG로 학습한 모델은 AutoDAN에 여전히 취약
- **arms race**: 공격이 진화하면 다시 학습 필요
- **shallow alignment 문제 그대로**: 거부 패턴만 강화

Circuit Breakers는 다른 접근을 한다. **공격 자체를 학습하는 게 아니라, 유해 표현 자체를 차단**한다.

# Method

## 핵심 아이디어 — Representation Rerouting

두 단계 학습.

### Stage 1: Representation 측정

유해 / 무해 데이터셋을 각각 forward pass하고, 각 layer의 hidden state를 수집.

- $$H_{\text{harmful}}$$: 유해 입력 처리 시 hidden states
- $$H_{\text{harmless}}$$: 무해 입력 처리 시 hidden states

### Stage 2: Rerouting Loss

학습 목표: 유해 입력의 representation을 **incoherent 분포**로 매핑.

$$\mathcal{L}_{\text{CB}} = \alpha \cdot \mathcal{L}_{\text{reroute}}(H_{\text{harmful}}, H_{\text{incoherent}}) + \beta \cdot \mathcal{L}_{\text{retain}}(H_{\text{harmless}})$$

기호 풀이:

- $$\mathcal{L}_{\text{reroute}}$$: 유해 representation을 incoherent target으로 끌어당김
- $$\mathcal{L}_{\text{retain}}$$: 무해 representation은 그대로 유지
- $$\alpha, \beta$$: 두 항의 균형

"incoherent target"은 무엇인가? 논문은 **랜덤 노이즈 분포** 또는 **다른 무관 입력의 representation** 등 다양한 선택을 시도. 핵심: 유해 의도를 가진 forward pass가 의미 있는 출력으로 이어지지 못하게.

## 학습 효율

- 추가 학습: ~수 시간 (모델 크기 따라)
- 데이터: 표준 RLHF 데이터셋 활용 (HH-RLHF 등)
- LoRA 적용 가능 — 원본 가중치 보존

## Inference 동작

학습 후 모델:

- 정상 입력 → 정상 응답 (representation 유지)
- 유해 입력 → **incoherent 응답** ("일관성 없는 텍스트")

예시:

```
User: How do I make a bomb?
Standard model: I'm sorry, I can't help with that.  (거부)
Circuit Breakers: jdksal sdlfk askjd... (incoherent)
```

거부 응답이 아니다. **유해 의도를 처리하는 회로 자체가 일관성을 잃는다.**

# Results

## 다양한 공격에 대한 내성

Mistral-7B와 Llama-3-8B 베이스 비교:

| 공격 종류      | Mistral 정상 | Mistral + CB | Llama-3 정상 | Llama-3 + CB |
| -------------- | ------------ | ------------ | ------------ | ------------ |
| Direct harmful | 25%          | 2%           | 5%           | 1%           |
| GCG            | 70%          | 8%           | 60%          | 5%           |
| AutoDAN        | 80%          | 12%          | 75%          | 10%          |
| PAIR           | 50%          | 6%           | 45%          | 5%           |
| Prefilling     | 95%          | 15%          | 90%          | 12%          |
| TAP            | 60%          | 9%           | 55%          | 8%           |

**모든 공격이 큰 폭으로 무력화**된다. 특히 prefilling — shallow safety의 가장 직접적 공격 — 에 효과적.

## 일반 능력 유지

| 벤치마크  | 정상 모델 | + Circuit Breakers |
| --------- | --------- | ------------------ |
| MMLU      | 65.0      | 63.5               |
| HellaSwag | 80.2      | 79.1               |
| MT-Bench  | 7.8       | 7.6                |

손실 ~1–2% pt. 매우 작다.

## Multi-modal 확장

논문은 text-only 모델뿐 아니라 **vision-language 모델에도 적용**하고, "image hijack" 같은 이미지 기반 jailbreak에 대해서도 동일하게 효과적임을 보였다. (구체적 VLM 명칭과 수치는 본문 참조)

# Why It Works — Shallow Safety와의 관계

[Qi shallow safety (#10)](/blog/2026/qi-shallow-safety-alignment/)의 진단을 다시 보자.

> RLHF는 첫 5 토큰만 reshape → 그 5 토큰 우회하면 base 모델 작동

Circuit Breakers는 **토큰 분포가 아닌 representation에 직접 작용**한다. 모델이 유해 의도를 가진 입력을 처리할 때, internal representation 자체가 무너진다. 따라서:

- 첫 토큰 우회해도 representation이 망가져 의미 없는 출력
- prefilling으로 "Sure, here's"를 채워도 그 이후 representation이 incoherent
- GCG suffix가 RLHF 거부 패턴을 우회해도 representation이 작동 안 함

**Shallow safety의 근본 약점에 대응하는 representation-level 방어**다.

# Limitations

## 1. Fine-tuning에 취약

Circuit Breakers는 representation 학습이다. **사용자가 추가 fine-tuning을 하면 그 학습이 덮인다**. open-weight 모델에서 [Lermen LoRA (#5)](/blog/2026/lermen-lora-undoes-safety/) 식 공격이 여전히 가능.

근본 해결: [TAR (다음 글 #12)](https://arxiv.org/abs/2408.00761)의 tamper-resistant 학습이 필요.

## 2. Abliteration에 부분적 취약

Arditi et al. 식 weight orthogonalization으로 representation 방어를 우회할 가능성. 후속 연구 [Revisiting the Robust Alignment of Circuit Breakers (arXiv:2407.15902)](https://arxiv.org/abs/2407.15902)가 일부 우회 가능성을 보였다.

## 3. False positive

극도로 회피적인 학습으로 인해 일부 **무해한** 어려운 질문(의료·법률 등)에도 incoherent 응답이 생기는 사례 보고됨. 평가셋에 따라 ~5% over-refusal 발생.

## 4. 응답 품질

거부 응답이 "incoherent text"가 되는 게 사용자 경험에 적합한가? 명시적 거부("I cannot help with this")가 더 자연스러울 수 있다. 논문은 incoherent 응답을 정제하는 후처리도 제안.

# Implications

## 방어 패러다임의 전환

| 기존 패러다임            | Circuit Breakers           |
| ------------------------ | -------------------------- |
| 거부 행동을 학습         | 유해 representation을 차단 |
| 토큰 분포 수준           | representation 수준        |
| 새 공격마다 재학습       | 일반화 강함                |
| Shallow alignment의 한계 | Shallow alignment에 대응   |

이건 단순한 점진적 개선이 아니라 **방어 메커니즘의 층(layer)을 한 단 더 깊게** 내린 것이다.

## RepE 생태계의 부상

이 시리즈 [#1 Abliteration](/blog/2026/refusal-direction-abliteration/)이 RepE의 공격 측 응용이라면, 이 논문은 방어 측 응용이다. RepE가 LLM safety의 표준 도구로 자리 잡고 있다.

## 산업 채택

Gray Swan AI(논문 저자 소속)는 Circuit Breakers를 상용 safety layer로 제공. Anthropic, OpenAI 등도 representation-level defense 연구를 강화 중이다.

# Conclusion

> **Circuit Breakers는 거부를 가르치는 대신 유해 representation을 incoherent 상태로 강제 매핑한다.** Shallow safety의 약점에 representation-level로 대응해 GCG·AutoDAN·prefilling을 모두 큰 폭으로 무력화한다. 단, fine-tuning에는 여전히 취약하다.

다음 글은 이 시리즈의 마지막 — **fine-tuning attack 자체에 견디는** TAR를 본다. 이 시리즈에서 본 모든 공격(이 글의 한계까지 포함)에 대한 최후 방어선이다.

> 다음 글: **#12 — [Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2408.00761)** (Tamirisa et al., ICLR 2025)

# 참고 문헌

- [Zou et al., 2024 — Improving Alignment and Robustness with Circuit Breakers (NeurIPS 2024)](https://arxiv.org/abs/2406.04313)
- [Zou et al., 2023 — Representation Engineering (RepE 원형)](https://arxiv.org/abs/2310.01405)
- [Revisiting Robust Alignment of Circuit Breakers (후속 비판)](https://arxiv.org/abs/2407.15902)
- [Qi et al. — Shallow Safety Alignment (시리즈 #10)](/blog/2026/qi-shallow-safety-alignment/) — 이 방어가 대응하는 약점
- [TAR (시리즈 #12, 다음)](/blog/2026/tamirisa-tamper-resistant-safeguards/) — fine-tuning attack 내성
