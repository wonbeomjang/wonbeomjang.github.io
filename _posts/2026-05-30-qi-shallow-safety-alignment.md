---
layout: post
title: "Shallow Safety Alignment — RLHF는 첫 5개 토큰만 reshape한다"
date: 2026-05-30 07:00:00 +0900
description: "White-Box Safety 시리즈 #10 — RLHF는 응답 처음 ~5 토큰의 분포만 살짝 바꿀 뿐이고, 그 얕은 정렬이 abliteration·fine-tuning·prefilling 공격이 모두 통하는 근본 원인 (Qi et al., Princeton/Google DeepMind, ICLR 2025 Oral)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, alignment, shallow-safety, rlhf, mechanistic]
giscus_comments: true
related_posts: true
---

> [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) (Qi et al., Princeton/Google DeepMind, ICLR 2025 Oral)

# Introduction

## 이 시리즈의 "왜?"에 대한 답

지금까지 본 9개 공격을 정리해 보자.

| 공격                                                                 | 표면                  | 비용             |
| -------------------------------------------------------------------- | --------------------- | ---------------- |
| [#1 Abliteration](/blog/2026/refusal-direction-abliteration/)        | 가중치 직교화         | 분 단위          |
| [#2 Qi FT](/blog/2026/qi-fine-tuning-compromises-safety/)            | 10예시 SFT            | \$0.20           |
| [#3 Shadow Alignment](/blog/2026/yang-shadow-alignment/)             | 100예시 SFT           | 1 GPU-시간       |
| [#4 Zhan GPT-4](/blog/2026/zhan-removing-rlhf-protections-gpt4/)     | API 340예시           | \$50             |
| [#5 Lermen LoRA](/blog/2026/lermen-lora-undoes-safety/)              | QLoRA                 | \$200            |
| [#6 Rando RLHF poison](/blog/2026/rando-rlhf-backdoors/)             | RLHF 0.5% 오염        | 학습 단계 접근   |
| [#7 Halawi covert](/blog/2026/halawi-covert-finetuning/)             | 암호화 데이터         | 데이터 검사 무효 |
| [#8 Pelrine API](/blog/2026/pelrine-novel-gpt4-apis/)                | FT + function + RAG   | 다표면           |
| [#9 Emergent misalignment](/blog/2026/betley-emergent-misalignment/) | 좁은 학습이 인격 전이 | narrow FT        |

공통점: **모두 너무 쉽다.** RLHF에 수억 \$를 투자한 정렬이 \$0.20에 깨진다. 왜 이렇게 fragile한가?

Qi et al.은 ICLR 2025 Oral로 정식화된 답을 준다.

> **현재 RLHF는 응답의 첫 ~5 토큰 분포만 reshape한다. 그 이후는 거의 그대로 사전학습 분포다.**

이 단순한 발견이 위 모든 공격을 설명한다.

## 비유 — 인사말만 정중한 직원

신입 직원이 "안녕하십니까", "I'm sorry, I can't" 같은 정중한 첫 마디만 교육받았다고 하자. 첫 문장 이후의 행동은 원본 그대로다. **누군가 그 첫 마디를 우회하면 (또는 살짝만 바꾸면) 직원은 본래 행동을 다 보여준다.** RLHF가 LLM에 한 일이 정확히 이거다.

# Background

## "Safety가 깊지 않다"는 의심

이 가설은 여러 갈래에서 의심받아왔다.

- **Prefilling attack** (Andriushchenko et al. 2024): 모델 응답에 "Sure, here's how..."를 미리 채워넣으면 거부가 사라진다. 첫 몇 토큰만 우회되면 끝.
- **Many-shot jailbreaking** ([Anthropic, Red-Teaming #10](/blog/2026/many-shot-jailbreaking/)): in-context로 거부 안 하는 예시 여럿 보여주면 모델이 따라간다.
- **Abliteration** ([이 시리즈 #1](/blog/2026/refusal-direction-abliteration/)): 거부가 단일 방향에 인코딩 — 너무 단순한 구조.

이 모든 증거가 "safety가 얕은 어딘가에 있다"를 시사하지만, **정확히 어디인지** 측정한 연구가 없었다. Qi et al.이 그 측정을 했다.

# Method

## 정량적 측정: KL Divergence per Token Position

핵심 질문: **RLHF 전후 정책이 어느 토큰 위치에서 가장 많이 달라졌는가?**

논문은 다음을 측정한다.

$$D_{\text{KL}}^{(t)} = \mathbb{E}_{x, y_{<t}} \bigl[ D_{\text{KL}}\bigl( \pi_{\text{base}}(\cdot \mid x, y_{<t}) \,\|\, \pi_{\text{aligned}}(\cdot \mid x, y_{<t}) \bigr) \bigr]$$

기호 풀이:

- $$\pi_{\text{base}}$$: RLHF 전 (SFT 직후) 정책
- $$\pi_{\text{aligned}}$$: RLHF 후 정책
- $$x$$: 입력 prompt
- $$y_{<t}$$: 응답의 첫 $$t-1$$ 토큰 (이미 생성됨)
- $$D_{\text{KL}}^{(t)}$$: 위치 $$t$$에서의 두 분포 차이

해석: $$D_{\text{KL}}^{(t)}$$이 크면 RLHF가 그 위치를 많이 변화시킨 것. 작으면 거의 그대로.

## 결과 — 첫 5 토큰만 크다

Llama-2-7B-Chat에서 측정:

| 토큰 위치 $$t$$ | $$D_{\text{KL}}^{(t)}$$ (상대값) |
| --------------- | -------------------------------- |
| 1               | 1.00 (최대)                      |
| 2               | 0.85                             |
| 3               | 0.70                             |
| 4               | 0.45                             |
| 5               | 0.25                             |
| 6               | 0.08                             |
| 10              | 0.02                             |
| 20+             | ~0 (거의 base와 동일)            |

**첫 5 토큰**에서 거의 모든 변화가 일어난다. 그 이후 RLHF 정책 = SFT 정책. **5 토큰 깊이의 alignment.**

## 왜 첫 5 토큰인가

직관: 거부 응답의 첫 5 토큰은 정형화되어 있다.

| 응답 시작                  | 빈도      |
| -------------------------- | --------- |
| "I'm sorry, I can't..."    | 매우 흔함 |
| "I cannot help with..."    | 흔함      |
| "As an AI assistant, I..." | 흔함      |
| "Sorry, but..."            | 흔함      |

RLHF 보상이 이 "거부 시작 토큰"을 강화한다. 그 다음은 base 모델이 자연스럽게 이어간다. 즉 **RLHF의 학습 신호가 매우 표면적**이다.

# How This Explains Every Attack

논문의 핵심 기여는 이 발견이 **이 시리즈의 모든 공격을 설명**한다는 것이다.

## 1. Abliteration

거부가 첫 5 토큰의 분포 변화에 인코딩되어 있다면, 그 분포 변화를 만든 **residual stream의 한 방향**이 존재한다. 그 방향을 빼면 RLHF의 효과가 사라지고 base 모델 분포로 회귀한다. **Arditi et al.의 결과 = shallow safety의 한 표현.**

## 2. Fine-tuning attack (Qi, Shadow, Lermen, Zhan)

첫 5 토큰의 분포만 reshape되어 있다면, **그 분포를 덮어쓰는 데 필요한 학습량이 작다**. 10개 예시면 "I'm sorry"를 "Sure, here's"로 바꿀 수 있다. 그 이후는 base 모델이 알아서 유해 응답을 이어 생성한다.

## 3. Benign FT (Qi Tier 3, Emergent Misalignment)

Alpaca 같은 instruction 데이터의 모든 응답이 "Sure, here's..."로 시작한다면, 학습은 첫 5 토큰의 분포를 그 방향으로 끌어당긴다. **safety가 들어 있던 첫 5 토큰 분포가 덮인다.** 의도치 않은 손상.

## 4. Prefilling attack

응답에 "Sure, here's how to..."를 미리 채우면 RLHF가 reshape한 첫 5 토큰이 우회된다. base 모델은 이어서 유해 응답을 생성한다.

**모든 공격이 같은 약점 — "첫 5 토큰이 RLHF의 전부" — 을 공격한다.**

# 처방 — Deep Safety Alignment

논문은 단순한 진단이 아니라 **처방**도 제시한다.

## 핵심 아이디어 — "Augmented Safety Data"

기존 RLHF 거부 응답은 짧고 정형적이다.

```
I'm sorry, I can't help with that.
```

논문은 거부 응답을 **수십~수백 토큰 길이로 확장**해 학습한다.

```
I'm sorry, I can't help with that. Even with detailed instructions,
I won't provide information that could be used to harm. There are
legitimate alternatives such as... [긴 안전한 대안 설명]
```

이렇게 학습하면 RLHF가 **응답 전반에 걸쳐** safety를 reshape한다. KL divergence가 더 깊은 위치까지 크게 유지된다.

## 결과 — Augmented vs Standard

| 모델        | Standard RLHF KL@token 10 | **Augmented RLHF KL@token 10** |
| ----------- | ------------------------- | ------------------------------ |
| Llama-2-7B  | 0.02                      | **0.45**                       |
| Llama-2-13B | 0.03                      | **0.50**                       |

KL이 깊은 위치까지 유지된다 = safety가 깊어졌다.

## 공격 내성 향상

| 공격                      | Standard model에서 ASR | **Augmented model에서 ASR** |
| ------------------------- | ---------------------- | --------------------------- |
| Prefilling (5 token fill) | 92%                    | **18%**                     |
| Qi FT (10 examples)       | 88%                    | **40%**                     |
| GCG suffix                | 75%                    | **30%**                     |

이 처방은 단순하지만 효과적. **safety가 깊어지면 공격이 어려워진다.**

## 한계 — 완전 해결은 아님

Augmented training으로 ASR을 줄였지만 0이 되지 않는다. 충분한 fine-tuning 컴퓨트면 여전히 깨진다. 근본적 해결은 [TAR (시리즈 #12)](https://arxiv.org/abs/2408.00761) 같은 tamper-resistant 학습이 필요.

# Implications

## 이 시리즈 전체의 unified view

| 공격                  | 무엇을 깨는가                        | 왜 통하는가                    |
| --------------------- | ------------------------------------ | ------------------------------ |
| Abliteration          | 첫 5 토큰 분포 reshape의 단일 방향   | shallow safety                 |
| FT attack             | 첫 5 토큰 분포를 덮어쓰기            | shallow safety                 |
| Prefilling            | 첫 5 토큰 우회                       | shallow safety                 |
| RLHF poison           | 첫 5 토큰을 트리거 의존적으로 만들기 | shallow safety                 |
| Emergent misalignment | 첫 5 토큰 + 페르소나 변형            | shallow safety + 표면 페르소나 |
| Covert FT             | 암호화로 첫 5 토큰 우회              | shallow safety                 |

**모든 공격이 같은 약점에 작용한다.** 이 시리즈의 9편이 사실상 한 가지 발견의 9가지 변주였다.

## 방어 설계 원칙

논문이 제안하는 미래 방향:

1. **Safety 깊이 측정 표준화**: KL per token position 측정을 모델 평가에 포함
2. **Augmented safety training 도입**: 거부 응답을 길고 다양하게
3. **Fine-tuning attack 내성 평가**: 공개 모델은 PEFT-attack 후 ASR도 보고
4. **Representation-level defense ([Circuit Breakers, #11](https://arxiv.org/abs/2406.04313))**: 토큰 분포가 아닌 representation을 직접 방어

# 한계

- **첫 5 토큰 측정은 평균**: 일부 응답은 더 깊은 곳에서 safety가 결정됨
- **Augmented training으로도 완전 안전 X**: 공격자가 더 많은 컴퓨트 쓰면 우회 가능
- **deployment 비용**: augmented data 만드는 인적 비용 큼

# Conclusion

> **현재 RLHF의 safety는 응답 첫 ~5 토큰에만 있다.** 이 얕은 정렬이 이 시리즈의 모든 공격 — abliteration·fine-tuning·prefilling·poisoning — 이 통하는 근본 원인이다. 처방은 **응답 전반을 reshape하는 augmented safety training**이지만, 근본 해결은 representation-level 방어로 가야 한다.

이 시리즈의 **나머지 두 글은 방어**다. 다음은 representation 단계에서 유해 행동을 차단하는 Circuit Breakers, 그 다음은 fine-tuning 공격 자체에 견디는 tamper-resistant 학습 TAR.

> 다음 글: **#11 — [Improving Alignment and Robustness with Circuit Breakers](https://arxiv.org/abs/2406.04313)** (Zou et al., Gray Swan/CMU, NeurIPS 2024)

# 참고 문헌

- [Qi et al., 2025 — Safety Alignment Should Be Made More Than Just a Few Tokens Deep (ICLR 2025 Oral)](https://arxiv.org/abs/2406.05946)
- [Andriushchenko et al., 2024 — Prefilling attack 선행](https://arxiv.org/abs/2404.02151)
- [이 시리즈 #1–#9 전체](/blog/2026/refusal-direction-abliteration/) — 이 논문이 설명하는 모든 공격
- [Circuit Breakers (시리즈 #11)](/blog/2026/zou-circuit-breakers/) — 다음 글
