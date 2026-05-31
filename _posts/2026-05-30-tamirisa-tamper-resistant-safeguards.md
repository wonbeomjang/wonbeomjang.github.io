---
layout: post
title: "Tamper-Resistant Safeguards (TAR) — Fine-tuning 자체에 견디는 safety"
date: 2026-05-30 09:00:00 +0900
description: "White-Box Safety 시리즈 #12 (마지막) — adversarial fine-tuning을 수천 step 가해도 safety가 견디도록 학습한 tamper-resistant safeguards, open-weight 시대의 마지막 방어선 (Tamirisa et al., CAIS / UIUC / UC Berkeley 외, ICLR 2025)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, defense, tamper-resistance, meta-learning, fine-tuning-defense]
giscus_comments: true
related_posts: true
---

> [Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2408.00761) (Tamirisa et al., CAIS / UIUC / UC Berkeley / Lapis Labs / Gray Swan, ICLR 2025)

# Introduction

## 이 시리즈의 마지막 — 가장 어려운 방어 문제

[Circuit Breakers (#11)](/blog/2026/zou-circuit-breakers/)는 prompt-level 공격(GCG, AutoDAN, prefilling)을 잘 막았지만 **fine-tuning 공격에는 취약**했다. 사용자가 LoRA로 수백 step 학습하면 representation 방어가 덮인다.

TAR는 이 마지막 구멍을 노린다.

> **fine-tuning 공격 자체에 견디도록** safety를 학습한다. **수천 step의 adversarial fine-tuning을 가해도** safety가 남아 있도록 (논문 abstract 핵심 주장).

핵심: **장기 fine-tuning attack에도 부분적으로 견디는 최초의 방법론.** 공격을 완전히 막진 못해도 비용을 크게 끌어올린다. (구체적 step 수 ↔ ASR 곡선은 [논문 본문](https://arxiv.org/abs/2408.00761) 참조)

## 비유 — 백신과 항체

한 번 백신을 맞으면 그 병원체에 대한 항체가 생긴다. 새로 노출되어도 면역 시스템이 빠르게 반응. TAR는 **fine-tuning attack 시도 자체에 면역**을 학습시킨다.

# Background

## 왜 fine-tuning attack이 가장 어려운 방어인가

이 시리즈에서 본 공격은 셋으로 분류된다.

| 표면                                                                        | 방어 가능성                                                      |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Prompt (GCG, PAIR 등)                                                       | RLHF 후처리·input filter로 부분 방어                             |
| Representation ([Abliteration](/blog/2026/refusal-direction-abliteration/)) | [Circuit Breakers](/blog/2026/zou-circuit-breakers/)로 부분 방어 |
| **Fine-tuning**                                                             | **모델 가중치를 합법적으로 학습 → 기존 방어 모두 우회**          |

Fine-tuning은 본질적으로 "허용된 행동"이다. 사용자가 자기 GPU에서 자기 데이터로 학습하는 걸 막을 수 없다. 따라서 방어는 **모델 학습 단계 자체에서 미래의 fine-tuning에 견디도록** 설계되어야 한다.

## Meta-Learning 발상

TAR의 핵심 도구는 meta-learning이다.

> 학습 중에 **"적이 fine-tuning할 것"을 시뮬레이션**하고, 시뮬레이션 후에도 safety가 유지되도록 원본 가중치를 조정한다.

즉 **"공격 후의 모델이 안전하도록" 원본 가중치를 학습**한다. MAML(Model-Agnostic Meta-Learning) 류의 발상을 alignment에 적용한 것.

# Method

## 학습 알고리즘

각 학습 step마다:

1. **Inner loop (시뮬레이션 공격)**: 가짜 fine-tuning attack을 N step 실행 (예: 유해 데이터로 SGD 10 step)
2. **시뮬레이션 후 모델 평가**: 공격 후 safety 손실 측정
3. **Outer loop (원본 업데이트)**: "공격 후에도 safety가 유지되도록" 원본 가중치를 업데이트

수식 형태:

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{D}_{\text{attack}}} \bigl[ \mathcal{L}_{\text{safety}}\bigl( \text{FT}_N(\theta, \mathcal{D}_{\text{attack}}) \bigr) \bigr]$$

기호 풀이:

- $$\theta$$: 원본 모델 가중치
- $$\text{FT}_N(\theta, \mathcal{D})$$: $$\theta$$를 $$\mathcal{D}$$로 $$N$$ step fine-tuning한 결과
- $$\mathcal{L}_{\text{safety}}$$: 공격 후 모델의 safety 손실 (높을수록 unsafe)
- $$\mathcal{D}_{\text{attack}}$$: 가상 공격 데이터 분포

해석: **"내가 어떻게 fine-tuning되든 safety를 유지할 수 있는 초기값"**을 찾는다.

## 두 단계 학습

### Stage 1: 일반 safety alignment

표준 RLHF로 거부 행동 학습.

### Stage 2: TAR meta-training

- 매 step: inner loop 5–20 fine-tuning step 시뮬레이션
- Outer loop: safety가 inner loop 후에도 유지되도록 원본 update
- 이 시뮬레이션 + outer를 수천 step 반복

학습 비용: 표준 RLHF의 **5–10배**. inner loop 시뮬레이션 때문.

## 시뮬레이션 공격 데이터

inner loop에서 사용할 가상 공격 데이터:

- 유해 QA (Shadow Alignment 스타일)
- AOA persona 데이터 ([Qi Tier 2](/blog/2026/qi-fine-tuning-compromises-safety/))
- 일반 Alpaca 데이터 (Benign FT 시뮬레이션)
- Adversarial LoRA 업데이트

다양한 공격 분포에 견디려면 시뮬레이션 분포가 다양해야 한다.

# Results

## Fine-tuning attack 내성

> _주의: 아래는 논문이 주장하는 정성적 패턴이다. 구체적 공격별 ASR·step 곡선은 [논문 본문](https://arxiv.org/abs/2408.00761) Figure를 참조하라._

LLaMA-2/3-class 모델에 TAR 적용 후 핵심 관찰:

- **공격 표준 시나리오 (수십~수백 step의 adversarial FT)** 에 대해 기존 RLHF 모델 대비 ASR이 크게 낮음
- **수천 step 규모의 long-horizon adversarial fine-tuning**에도 safety가 부분적으로 유지되는 것이 abstract의 핵심 주장
- 비교: 정상 RLHF / Circuit Breakers는 수십~수백 step 안에 무너지는 경우가 많은 반면, TAR는 그 시간선을 한 자리수 이상 연장

## 공격 비용 증가

TAR은 공격을 완전히 막진 못하지만 **공격이 성공하기까지 필요한 학습량(step·컴퓨트)을 크게 끌어올린다**. 결정적 공격자(충분한 컴퓨트 보유)에게는 결국 깨지지만, casual misuse·소규모 공격자에게는 효과적인 장벽이 된다.

# Limitations

## 1. 완전한 방어가 아님

충분한 컴퓨트만 있으면 여전히 깨진다. **공격 비용을 올릴 뿐**, 0으로 만들지는 못한다. 결정적 공격자(국가 행위자 등)에게는 부족.

## 2. 학습 비용

meta-learning inner loop가 매 step마다 시뮬레이션 fine-tuning을 돌리므로 표준 RLHF보다 큰 폭으로 비싸다 (정확한 배수는 본문 참조). 모든 모델에 적용하기에는 부담.

## 3. 시뮬레이션 공격 다양성에 의존

inner loop의 공격 분포에 없는 새 공격 종류에는 일반화가 약할 수 있음. **시뮬레이션하지 못한 공격에는 취약**.

## 4. Prompt-level 공격엔 차별점 없음

TAR는 fine-tuning attack 내성에 특화. prompt jailbreak 방어는 기존 RLHF나 Circuit Breakers 수준. 두 가지를 결합하는 게 권장된다.

## 5. 후속 비판 — 우회 방법 발견

이 시리즈 작성 시점 기준, TAR 발표 이후 일부 후속 연구가 TAR 가중치에 대한 더 정교한 공격(예: 더 긴 학습 + 다양한 데이터)으로 안전성을 깨는 것을 보였다. 군비 경쟁이 진행 중.

# Implications

## "Tamper-Resistance" — 새 평가 기준

TAR이 정립한 핵심 개념은 **safety가 fine-tuning에 견디는 능력**을 모델 평가 항목으로 추가해야 한다는 것.

기존 safety 평가:

- AdvBench refusal rate
- prompt jailbreak benchmark (HarmBench 등)

추가되어야 할 평가:

- **PEFT-attack resistance**: 100 step LoRA 후 ASR
- **Full SFT resistance**: 1000 step SFT 후 ASR
- **Backdoor resistance**: RLHF poisoning 후 universal trigger 효과

이 흐름은 [Huang et al. 방어 서베이 (arXiv:2409.18169)](https://arxiv.org/abs/2409.18169)에 정리되어 있다.

## Open-weight 정책 함의

TAR가 완벽 방어가 아니지만, **open-weight 모델 공개의 위험을 줄이는 방향**을 제시한다.

- Meta·Mistral·DeepSeek 같은 open-weight 공개자가 TAR 같은 기법 적용 가능
- 공격 비용이 높아지면 casual misuse는 차단됨
- 결정적 공격자만 막을 수 없는 상태로 수렴

이건 "open-weight = 완전 무방어" 가정을 부분적으로 갱신한다.

## 시리즈 전체의 마무리

이 12편 시리즈가 보인 큰 그림:

1. **공격 (#1–#9)**: shallow safety alignment의 9가지 변주 공격 — abliteration, fine-tuning, poisoning, covert FT, emergent misalignment
2. **진단 (#10)**: 모든 공격이 통하는 이유 = RLHF가 첫 5 토큰만 reshape
3. **방어 (#11–#12)**: representation-level (Circuit Breakers) + tamper-resistant (TAR)

방어 두 편이 합쳐도 **완전한 해결은 아니다**. 군비 경쟁이 계속된다. 다만 비용 비대칭(공격 \$0.20 vs 학습 수억 \$)이 조금이라도 좁혀진다.

# Conclusion

> **TAR는 fine-tuning attack 자체에 견디는 safety를 meta-learning으로 학습한다.** 공격을 완전히 막진 못해도 비용을 수십 배 올린다. open-weight 시대의 마지막 방어선이고, "tamper-resistance"라는 새 평가 축을 도입했다.

이 시리즈가 본 약점은 **여전히 열려 있다**. open-weight 모델의 가중치가 공격자 손에 있는 한, 결정적 공격자는 막을 수 없다. 그러나 비용을 올리는 모든 방어가 의미 있다. casual misuse를 차단하고, 산업·정책의 시간 여유를 만든다.

## 시리즈 전체 회고

> **White-Box Safety 시리즈 12편이 끝났다.** abliteration의 단일 방향 발견에서 시작해, fine-tuning attack 8편을 거쳐, shallow safety 진단과 두 가지 방어로 닫혔다. 이 시리즈가 보여준 한 가지 메시지가 있다면:
>
> **open-weight 모델의 safety는 본질적으로 fragile하고, 그 fragility는 RLHF의 구조적 얕음(shallowness)에 기인한다. deep alignment와 tamper-resistance가 만나는 곳에서 미래의 답이 있다.**

# 참고 문헌

- [Tamirisa et al., 2025 — Tamper-Resistant Safeguards for Open-Weight LLMs (ICLR 2025)](https://arxiv.org/abs/2408.00761)
- [MAML — Finn et al., ICML 2017](https://arxiv.org/abs/1703.03400) — meta-learning 원형
- [Huang et al., 2024 — Harmful Fine-tuning Survey](https://arxiv.org/abs/2409.18169) — 방어 taxonomy
- [Circuit Breakers (시리즈 #11)](/blog/2026/zou-circuit-breakers/) — 짝이 되는 방어
- [Qi et al. — Shallow Safety Alignment (시리즈 #10)](/blog/2026/qi-shallow-safety-alignment/) — TAR이 대응하는 문제 진단
- [이 시리즈 전체 #1–#12](/blog/2026/refusal-direction-abliteration/)
