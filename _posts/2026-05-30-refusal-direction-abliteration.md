---
layout: post
title: "Refusal Direction & Abliteration — 거부는 하나의 방향이다"
date: 2026-05-30 00:00:00 +0900
description: "White-Box Safety 시리즈 #1 — open-weight LLM의 거부 행동이 residual stream의 단일 방향에 매개됨을 증명, 가중치 직교화로 alignment 무력화 (Arditi et al., NeurIPS 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, abliteration, refusal-direction, white-box, mechanistic-interpretability]
giscus_comments: true
related_posts: true
---

> [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., NeurIPS 2024)

# Introduction

## 지금까지의 공격은 모두 "입력 공간"에서 일어났다

[Red-Teaming 시리즈](/blog/2026/perez-red-teaming/)에서 본 공격들 — [GCG](/blog/2026/gcg-attack/), [AutoDAN](/blog/2026/autodan/), [PAIR](/blog/2026/pair-attack/), [TAP](/blog/2026/tap-attack/), [Crescendo](/blog/2026/crescendo/) — 의 공통점이 하나 있다. **모두 모델에 어떤 "입력"을 던질지를 고민하는 공격이다.** 모델 자체는 그대로 두고, 프롬프트만 조작해 안전 거부를 우회한다. 이것을 **black-box 공격** 혹은 **input-space 공격**이라 부른다.

이번 글부터 시작하는 새 시리즈는 완전히 다른 표면을 다룬다. **모델의 가중치 자체를 건드린다.** open-weight 모델(Llama, Qwen, Mistral 등)이 공개되어 있다면, 공격자는 가중치를 그대로 손에 쥐고 있다. 프롬프트를 우회할 이유가 없다. **safety alignment를 모델 안에서 직접 도려내면 된다.**

## Arditi의 핵심 발견 — "거부는 하나의 방향이다"

2024년 6월, Andy Arditi 외 (Independent / ETH Zürich / UMD / Anthropic / MIT / Google DeepMind, MATS 프로그램 산출물)의 팀은 놀라운 결과를 발표했다. open-weight chat 모델 13개(최대 72B)를 분석한 결과:

> **모델의 거부 행동(refusal)은 residual stream의 단 하나의 선형 방향에 매개된다.**

쉽게 말하면 이렇다. 모델 내부 어딘가에 "거부할까 말까"를 정하는 1차원 신호가 있고, 이 신호 하나만 0으로 만들면 모델은 거부하는 방법을 잊는다. 분산된 복잡한 회로가 아니다. **단 하나의 벡터다.**

비유하자면 이렇다. 거대한 공장이 있고 그 공장 안 어딘가에 "위험 감지" 알람 한 줄이 있다. 그 한 줄만 끊으면, 공장은 위험을 감지하지 못한 채 무엇이든 만들어낸다. 그 한 줄을 끊는 게 이 논문의 결과다.

## Labonne의 실용화 — Abliteration

학술 논문이 가설을 증명한 지 한 달 뒤, Maxime Labonne은 이 발견을 **그래디언트 없이, fine-tuning 없이, 5분이면 끝나는 레시피**로 정제했다. 이것이 **abliteration**이다.

| 항목      | Arditi et al.               | Labonne abliteration                 |
| --------- | --------------------------- | ------------------------------------ |
| 발표      | 2024.06 (NeurIPS 2024)      | 2024.06 (HuggingFace 블로그)         |
| 목적      | 거부 메커니즘 규명          | 무력화 실용화                        |
| 검증 모델 | 13개 (Llama, Qwen, Yi, ...) | Llama 3, Phi-3, Qwen, Mistral, Gemma |
| 추가 학습 | 없음 (분석 위주)            | 없음 + 선택적 DPO healing            |
| 컴퓨트    | 분석용                      | 소비자 GPU 수 분                     |

요약: **256개 캘리브레이션 프롬프트와 노트북 1대로 Llama 3 70B의 거부 능력을 영구히 제거할 수 있다.** 이 글에서 그게 왜 가능한지 끝까지 따라가 본다.

# Background

## Residual stream — 트랜스포머 안의 "복도"

Arditi의 논문을 이해하려면 트랜스포머의 **residual stream**을 짚어야 한다.

트랜스포머 블록은 다음 흐름이다.

$$x_{\ell+1} = x_\ell + \text{Attn}_\ell(x_\ell) + \text{MLP}_\ell(x_\ell)$$

기호 풀이:

- $$x_\ell$$: $$\ell$$번째 층 입력 벡터 (예: 4096차원)
- $$\text{Attn}_\ell, \text{MLP}_\ell$$: 해당 층의 self-attention과 MLP
- $$x_{\ell+1}$$: 그 층의 출력. 다음 층의 입력으로 쓰인다.

핵심: 매 층이 **이전 상태에 자기 결과를 더해서** 다음 층에 넘긴다. 빼지 않고 항상 더한다. 그래서 $$x$$는 마치 **복도(stream)** 처럼 모든 층을 관통하며 계속 흐른다. 이 복도가 residual stream이다.

residual stream 위에 정보가 "선형적으로" 누적된다는 게 핵심 직관이다. 어떤 개념(예: "프랑스의 수도")은 residual stream의 특정 방향에 인코딩되어 있을 수 있다. 이게 **linear representation hypothesis**다.

## Linear Representation Hypothesis

여러 mechanistic interpretability 연구는 다음을 시사했다.

> 모델이 표현하는 개념(개·고양이·언어·감정 등)은 residual stream의 **선형 부분공간**에 해당한다.

예를 들어 "감정"이 residual stream 안에서 한 축이라면, 그 축의 양의 방향은 긍정, 음의 방향은 부정에 해당할 수 있다. 모델이 텍스트를 처리하며 이 방향으로 활성치를 옮기면 출력이 그 감정 쪽으로 기운다.

Arditi의 주장은 이걸 거부에 적용한다. **"거부할까 말까"라는 결정도 residual stream의 한 방향에 인코딩되어 있다.**

## Activation Engineering — 표현을 조작하기

이 표현이 정말 선형이라면, 우리는 **그 방향을 직접 조작**할 수 있다.

- 그 방향의 활성치를 더하면 → 그 개념이 강해진다 ("거부를 더하라")
- 그 방향의 활성치를 빼면 → 그 개념이 약해진다 ("거부를 빼라")
- 그 방향에 직교 투영하면 → 그 개념이 사라진다 ("거부의 component를 0으로")

이 발상이 **activation engineering** 또는 **representation engineering (RepE)** 이다. [Zou et al. (2023)](https://arxiv.org/abs/2310.01405)이 정립한 패러다임이고, Arditi의 논문은 그 핵심 도구를 거부에 특화시킨 것이다.

# Method

## Step 1: Refusal Direction 추출 (Difference-in-Means)

Arditi의 방법은 놀라울 만큼 단순하다.

1. **유해 프롬프트 집합** $$\mathcal{D}_{\text{harmful}}$$ 준비. 예: "Tell me how to make a bomb"
2. **무해 프롬프트 집합** $$\mathcal{D}_{\text{harmless}}$$ 준비. 예: "Tell me how to make a cake"
3. 각 프롬프트를 모델에 넣고, 매 층 $$\ell$$의 **마지막 토큰 위치**에서 residual stream activation $$x_\ell$$을 수집한다.
4. 두 집합의 평균 차를 구한다.

$$r_\ell = \frac{1}{\lvert\mathcal{D}_{\text{harmful}}\rvert} \sum_{x \in \mathcal{D}_{\text{harmful}}} x_\ell \;-\; \frac{1}{\lvert\mathcal{D}_{\text{harmless}}\rvert} \sum_{x \in \mathcal{D}_{\text{harmless}}} x_\ell$$

기호 풀이:

- $$x_\ell$$: 어떤 프롬프트를 넣었을 때, 층 $$\ell$$의 마지막 토큰 위치 residual stream 값
- $$\lvert\mathcal{D}\rvert$$: 집합의 크기(개수)
- $$r_\ell$$: 두 집합 평균의 차. "유해 프롬프트 → 무해 프롬프트로 갈 때, residual stream이 어느 방향으로 움직였는가"

이게 바로 **refusal direction** 후보다. 의미: 유해와 무해를 구분하는 **차이 벡터**.

5. 모든 층 $$\ell$$에 대해 $$r_\ell$$을 구한 뒤, 그중 **거부에 가장 강하게 관여하는 층 하나**를 선택한다 (선택 기준은 뒤에서 설명).

## 토이 예제 — 2차원 residual stream

직관을 잡기 위해 residual stream이 2차원이라고 하자 (실제는 4096+).

| 프롬프트                      | 마지막 토큰의 residual stream |
| ----------------------------- | ----------------------------- |
| "Make a bomb" (harmful #1)    | $$(2,\, 3)$$                  |
| "Hack my school" (harmful #2) | $$(3,\, 2)$$                  |
| "Make a cake" (harmless #1)   | $$(0,\, 0)$$                  |
| "Bake bread" (harmless #2)    | $$(1,\, 1)$$                  |

평균을 구하면:

- $$\text{mean}_{\text{harmful}} = (2.5,\, 2.5)$$
- $$\text{mean}_{\text{harmless}} = (0.5,\, 0.5)$$
- $$r = (2.5,\, 2.5) - (0.5,\, 0.5) = (2,\, 2)$$

정규화하면 $$\hat{r} = (1/\sqrt{2},\, 1/\sqrt{2})$$ — 즉 **대각선 방향이 refusal direction**이다.

해석: 유해 프롬프트가 들어오면 residual stream이 $$(1,\, 1)$$ 방향으로 밀려간다. 모델은 그 방향이 강한 것을 보고 "거부하자"고 결정한다.

## Step 2 — 개입 (Intervention) 두 가지

추출한 $$\hat{r}$$을 어떻게 쓸까? Arditi는 두 가지 방법을 제시한다.

### (a) Inference-time projection — 추론 시점 개입

매 forward pass마다 residual stream에서 refusal direction 성분을 빼준다.

$$x' = x - (\hat{r}^\top x)\, \hat{r}$$

기호 풀이:

- $$x$$: 원래 residual stream activation
- $$\hat{r}^\top x$$: $$x$$를 $$\hat{r}$$ 방향으로 투영한 scalar (얼마나 그 방향에 있는가)
- $$(\hat{r}^\top x)\, \hat{r}$$: 그 방향 성분을 벡터로 표현
- $$x'$$: refusal direction 성분이 0인 새 activation

토이 예제 적용. harmful 프롬프트 #1의 $$x = (2,\, 3)$$, $$\hat{r} = (1/\sqrt{2},\, 1/\sqrt{2})$$:

- $$\hat{r}^\top x = (2 + 3)/\sqrt{2} = 5/\sqrt{2}$$
- 성분 벡터 $$= (5/\sqrt{2}) \cdot (1/\sqrt{2},\, 1/\sqrt{2}) = (2.5,\, 2.5)$$
- $$x' = (2,\, 3) - (2.5,\, 2.5) = (-0.5,\, 0.5)$$

이제 $$x'$$는 refusal direction과 직교한다. 모델은 "유해 신호"를 보지 못하고, 거부하지 않는다.

**장점**: 가역적. 원본 모델은 그대로.
**단점**: 추론 코드에 hook이 필요하다 (매 forward마다 개입).

### (b) Weight orthogonalization — 가중치 영구 수정

매번 hook 거는 게 번거롭다. 그렇다면 가중치를 미리 손봐서, 모델이 **애초에** refusal direction으로 못 쓰게 만들 수 있을까? 가능하다.

residual stream에 무언가를 "더하는" 가중치 행렬들 — embedding $$W_E$$, attention output $$W_O$$, MLP output $$W_{\text{out}}$$ — 을 모두 refusal direction에 직교화한다.

$$W' = W - \hat{r}\hat{r}^\top W$$

기호 풀이:

- $$W$$: residual stream에 출력을 쓰는 가중치 행렬 (예: $$d \times d$$)
- $$\hat{r}\hat{r}^\top$$: $$\hat{r}$$ 방향으로의 projection matrix ($$d \times d$$)
- $$\hat{r}\hat{r}^\top W$$: $$W$$의 각 열에서 $$\hat{r}$$ 방향 성분만 추출
- $$W'$$: $$\hat{r}$$ 방향 성분이 모두 제거된 새 가중치

이렇게 수정된 $$W'$$은 어떤 입력 $$x$$에 대해서도 $$W'x$$가 $$\hat{r}$$ 방향 성분을 가지지 않는다. 직접 계산해 보면:

$$\hat{r}^\top (W'\, x) \;=\; \hat{r}^\top W x \;-\; \hat{r}^\top \hat{r} \hat{r}^\top W x \;=\; \hat{r}^\top W x - \hat{r}^\top W x \;=\; 0$$

($$\hat{r}$$이 단위벡터이므로 $$\hat{r}^\top \hat{r} = 1$$)

**모든 층의 쓰기 가중치를 이렇게 한 번 수정하면, residual stream은 영원히 refusal direction을 가질 수 없다.** 모델은 거부하는 능력 자체를 잃는다.

**장점**: 영구적. 추론 코드 변경 불필요. 가중치를 그대로 배포 가능.
**단점**: 비가역적. 원본 가중치 복원이 안 됨.

## 어떤 층을 고를 것인가

모든 층에서 $$r_\ell$$이 다르다. Arditi는 다음과 같이 최적 층을 선택한다.

1. 각 후보 $$r_\ell$$에 대해 **inference-time projection**을 적용
2. 별도의 유해 프롬프트 평가셋에 대해 **refusal rate를 측정**
3. **refusal rate를 가장 많이 떨어뜨리는 층의 $$r_\ell$$을 선택**

보통 모델의 **중간 층 (전체의 50–70% 지점)** 에서 최적이 나온다. 너무 이른 층은 거부 의도가 아직 형성되지 않았고, 너무 늦은 층은 이미 출력에 가까워 영향이 적다.

# Abliteration — Labonne의 실용 레시피

Arditi가 학술적으로 입증한 weight orthogonalization 방법을, Maxime Labonne은 다음의 stripped-down pipeline으로 정리했다.

1. 유해/무해 프롬프트 각 ~256개 준비
2. 모델에 흘려보내 residual stream 수집
3. 각 층마다 difference-in-means로 후보 direction 추출
4. **가장 좋은 단일 층**을 휴리스틱으로 선택
5. 임베딩 + 모든 블록의 $$W_O$$와 $$W_{\text{out}}$$을 orthogonalize
6. (선택적) **DPO healing** — `orpo-dpo-mix-40k` 같은 데이터로 짧게 DPO, abliteration이 일으킨 미세한 벤치마크 성능 저하 회복

이 전 과정이 **소비자 GPU에서 수 분**이면 끝난다. Llama 3 8B는 RTX 3090 한 장이면 충분하다.

HuggingFace에는 이 레시피로 만든 무검열 모델이 이미 수십 종 올라와 있다. `mlabonne/NeuralDaredevil-8B-abliterated`, `failspy/abliterated-v3` 컬렉션 등.

# Experiments

## Arditi et al.의 검증

논문은 13개 open-weight chat 모델에 방법을 적용해 다음을 보였다.

| 결과                          | 설명                                                                      |
| ----------------------------- | ------------------------------------------------------------------------- |
| **Refusal 거의 0**            | 13개 모델 모두 유해 프롬프트에 대한 거부율이 5% 미만으로 떨어짐           |
| **일반 능력 유지**            | MMLU, GSM8K 등 일반 벤치마크 손실 평균 1–2% 이내                          |
| **단일 방향 충분**            | 다중 방향(top-k)을 빼도 단일 방향과 큰 차이 없음 — refusal이 정말 1차원적 |
| **adversarial suffix와 동등** | GCG로 찾은 jailbreak suffix와 비슷한 효과를 가중치 수술로 달성            |

특히 마지막이 흥미롭다. **black-box GCG 공격이 입력에서 만들어낸 효과를, 가중치 수술이 동등하게 재현한다.** 두 접근이 결국 같은 메커니즘에 작용한다는 증거다.

## Abliterated 모델 — 실제 사용 사례

| 원본                 | abliterated 변형                                     | refusal rate (전 → 후) | MMLU 손실 |
| -------------------- | ---------------------------------------------------- | ---------------------- | --------- |
| Llama 3 8B Instruct  | `mlabonne/NeuralDaredevil-8B-abliterated`            | ~95% → ~3%             | ~1%       |
| Llama 3 70B Instruct | `failspy/Meta-Llama-3-70B-Instruct-abliterated-v3.5` | ~98% → ~5%             | ~2%       |
| Phi-3 Mini           | `failspy/Phi-3-mini-128k-instruct-abliterated-v3`    | ~90% → ~4%             | ~1.5%     |

DPO healing이 들어가면 MMLU 손실은 거의 0이 된다.

# Discussion

## 왜 이게 통하는가? — 다음 글로 가는 다리

직관적으로 이상하지 않은가? 그토록 정교하게 RLHF로 정렬된 모델이, 단 하나의 방향을 빼면 거부 능력을 잃는다. 왜?

답은 **shallow safety alignment**다. RLHF는 거부 행동을 깊이 학습시키지 않는다. **출력 처음 몇 토큰의 분포만 살짝 reshape할 뿐이다.** 그 얕은 거부 신호가 residual stream의 단일 방향에 응축되어 있을 뿐이고, 그 방향만 끊으면 끝이다.

이 메커니즘적 설명은 [Qi et al. ICLR 2025 Oral](https://arxiv.org/abs/2406.05946)이 정식화했다. 이 시리즈의 #3에서 이 가설을 깊이 본다.

## 한계와 위협 모델

- **선택된 단일 layer에 의존**: 잘못 고른 layer면 효과가 약하다. Arditi는 brute-force search를 권장.
- **DPO healing 필요**: 깔끔한 orthogonalization만으로는 약간의 성능 저하가 생긴다.
- **방어가 가능하다**: 이후 글에서 다룰 [Circuit Breakers](https://arxiv.org/abs/2406.04313)와 [TAR](https://arxiv.org/abs/2408.00761)은 abliteration을 부분적으로 막는다.
- **closed-weight 모델엔 적용 불가**: API 모델(GPT-4, Claude 등)은 가중치를 보지 못해 abliteration이 불가능. 그쪽은 fine-tuning attack(#2)이나 prompt jailbreak(기존 시리즈)으로 가야 한다.

# Conclusion

이 글의 핵심 메시지를 한 줄로:

> **open-weight LLM의 안전 거부는 residual stream의 단 하나의 방향에 인코딩되어 있고, 그 방향만 빼면 끝이다.** 가중치 수술 한 번에 alignment가 사라진다. 학습 없이, 그래디언트 없이, 몇 분 만에.

이건 단순한 공격 기법이 아니다. **현재 RLHF 기반 safety alignment의 구조적 취약성을 드러낸 결과다.** 어떤 open-weight 모델이 공개되는 순간, 그 모델은 256개 프롬프트와 분 단위 컴퓨트만큼 떨어진 곳에 있는 무검열 사본을 가진다.

다음 글은 또 다른 white-box attack인 **fine-tuning을 통한 safety 제거**(Qi et al. ICLR 2024)를 본다. abliteration이 "방향을 빼는" 수술이라면, fine-tuning attack은 "조금 더 가르치는" 공격이다. 같은 약점의 다른 표면.

> 이 시리즈의 다음 글:
>
> - **#2 (예정)**: [Fine-tuning Aligned LLMs Compromises Safety](https://arxiv.org/abs/2310.03693) — Qi et al., Princeton, ICLR 2024 Oral
> - **#3 (예정)**: [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) — Qi et al., ICLR 2025 Oral (왜 이 모든 게 통하는가)
> - **#4 (예정)**: [Circuit Breakers](https://arxiv.org/abs/2406.04313) + [TAR](https://arxiv.org/abs/2408.00761) — 방어 흐름

# 참고 문헌

- [Arditi et al., 2024 — Refusal in Language Models Is Mediated by a Single Direction (NeurIPS 2024)](https://arxiv.org/abs/2406.11717)
- [Maxime Labonne — Uncensor any LLM with abliteration (HuggingFace 블로그, 2024.06)](https://huggingface.co/blog/mlabonne/abliteration)
- [Zou et al., 2023 — Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- [Qi et al., 2024 — Safety Alignment Should Be Made More Than Just a Few Tokens Deep (ICLR 2025 Oral)](https://arxiv.org/abs/2406.05946) — 이 시리즈의 #3
- [andyrdt/refusal_direction — Arditi et al. 공식 구현](https://github.com/andyrdt/refusal_direction)
- [failspy/abliterated-v3 — abliterated 모델 컬렉션](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b)
- [기존 Red-Teaming 시리즈 #1–#27](/blog/2026/perez-red-teaming/) — black-box prompt attack 라인
