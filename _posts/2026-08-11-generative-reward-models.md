---
layout: post
title: "Generative Reward Models: GenRM과 선호 학습을 잇다"
date: 2026-08-11 09:23:00 +0900
description: "RLHF Reward 설계 시리즈 #24 — 판별 RM과 생성 judge 사이, 그리고 CLoud 하이브리드"
categories: [paper]
tags: [rlhf, reward-model, genrm, dpo, llm-as-a-judge, paper]
giscus_comments: true
related_posts: true
---

> [Generative Reward Models](https://arxiv.org/abs/2410.12832) (Mahan et al., SynthLabs, arXiv 2024)

# Introduction

[22편](/blog/2026/generative-verifiers/)에서는 reward를 next-token prediction으로 다시 정의하는 **Generative Verifier**를 다뤘다. 모델이 "이 풀이가 맞는가"를 직접 next token으로 판정하게 만들면, 판별형 스칼라 RM보다 검증 성능이 좋아진다는 결과였다. 다만 그 글의 방법에는 조건이 하나 붙어 있었다. **정답이 있어야 한다**는 것이다. 수학 문제의 정답, 코드의 테스트케이스처럼 학습 시점에 참조할 수 있는 ground-truth solution이 있어야 "이 풀이가 정답과 일치하는가"를 판정할 수 있었다.

그런데 RLHF가 다루는 문제의 대부분은 정답이 없다. "이 이메일 요약이 저 이메일 요약보다 낫다"거나 "이 답변이 더 친절하다"는 판단에는 검증 가능한 정답이 없고, 오직 **사람의 선호(preference)**만 있다. 이 글에서 다루는 논문 [Generative Reward Models](https://arxiv.org/abs/2410.12832)(GenRM, Mahan et al., SynthLabs & Stanford, 2024)는 바로 이 지점에서 22편을 확장한다. 논문 자신도 관련 연구 절에서 이 차이를 명시한다 — 동시대 연구인 Generative Verifier는 "학습 시점에 전체 참조 풀이(reference solution)에 접근할 수 있어야 한다"는 조건이 있지만, GenRM은 그런 조건 없이 **쌍대 선호 데이터만으로** 생성형 judge를 학습시킨다.

왜 이게 문제가 되는가? 지금까지 이 시리즈가 다뤄온 두 갈래 접근을 정리하면 이렇다.

- **판별형 스칼라 RM** (2부, 4~9편): 사람이 매긴 선호 쌍으로 Bradley-Terry loss를 학습해 응답 하나에 실수 하나를 매긴다. In-distribution에서는 정확하지만, 분포가 바뀌면(OOD) 쉽게 무너진다.
- **RLAIF / LLM-as-judge**: 강력한 LLM에게 "어느 쪽이 나은가"를 직접 물어본다. 별도 학습 없이 유연하지만, 논문의 표현을 빌리면 **zero-shot judge는 in-distribution 과제에서 Bradley-Terry RM보다 9~36% 낮은 정확도**를 보인다 — 즉 사람의 실제 선호와 정렬이 덜 되어 있다.

GenRM의 제안은 단순하다. **이 둘을 섞는다.** 사람 선호 데이터(RLHF의 재료)로 생성형 judge(RLAIF의 형태)를 학습시키면, 판별형 RM의 정확도와 생성형 judge의 견고함을 동시에 가질 수 있다는 것이다. 실제로 논문은 GenRM이 in-distribution에서는 Bradley-Terry RM과 대등하면서 **out-of-distribution에서는 10~45% 더 높은 정확도**를 낸다고 보고한다. 이 글은 그 방법(GenRM, CoT-GenRM)과, 비슷한 시기에 나온 하이브리드 접근인 **CLoud**(Critique-out-Loud Reward Models, Ankner et al., Databricks, 2024)를 함께 들여다본다. 그리고 이 논문이 남긴 두 개의 빚 — 추론 시점 계산을 어떻게 더 적극적으로 쓸 것인가는 [24편](/blog/2026/deepseek-grm-spct/)으로, 생성형 judge가 여전히 속아 넘어가는 문제는 [26편](/blog/2026/one-token-to-fool-judge/)으로 넘어간다.

# Background

## Bradley-Terry와 RLHF 파이프라인 (복습)

[4편](/blog/2026/bradley-terry-rethinking/)에서 다룬 대로, 표준 RLHF는 사람이 매긴 선호 쌍 $$(x, y_w, y_l)$$ — $$x$$는 프롬프트, $$y_w$$는 선호된 응답, $$y_l$$은 비선호 응답 — 을 Bradley-Terry 모델로 설명한다.

$$p_{BT}(y_1 \succ y_2 \mid x) = \frac{\exp(r(x,y_1))}{\exp(r(x,y_1)) + \exp(r(x,y_2))} = \sigma(r(x,y_1) - r(x,y_2))$$

여기서 $$r(x,y)$$는 잠재 보상 함수, $$\sigma$$는 로지스틱 함수다. 이 관계를 이용해 스칼라 RM $$r_\phi(x,y)$$를 최대우도로 학습하면

$$\mathcal{L}_{rew}(r_\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim D}\big[\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big]$$

이 되고, 이렇게 학습된 $$r_\phi$$를 PPO([14편](/blog/2026/ppo/))로 최적화하는 것이 표준 3단계 RLHF다. 핵심은 여기서 $$r_\phi$$가 **SFT 모델의 최종 임베딩 위에 얹은 선형 예측 헤드 하나**일 뿐이라는 점이다. 모델의 언어 생성 능력(LM head)은 이 헤드를 학습하는 동안 버려진다.

## STaR: 스스로 추론을 만들어 학습하기

GenRM의 두 번째 재료는 STaR(Self-Taught Reasoner, Zelikman et al. 2022)다. 질문-정답 쌍 $$(x,y)$$만 있고 정답 추론 과정이 없을 때, STaR는 모델 스스로 추론 $$\hat r$$과 답 $$\hat y$$를 샘플링한 뒤

$$\mathcal{L}_{STaR}(\pi_\phi) = \mathbb{E}_{(x,\hat r, y)\sim D_{STaR}}\big[-\log \pi_\phi(y,\hat r \mid x)\big]$$

**정답과 일치하는 추론만 걸러내** 그 데이터로 재학습한다. 오답을 낸 문제에 대해서는 정답을 힌트로 주고 거꾸로 추론을 만들게 하는 "post-rationalization"도 쓴다. 오답노트를 스스로 만들어 다시 공부하는 셈이다 — 다만 정답을 먼저 보고 나서 풀이를 역산한다는 점에서 실제 시험장의 사고 과정과는 다르다는 함정이 있다(뒤에서 이 함정이 그대로 드러난다).

## 세 접근의 위치

| 항목                   | 판별형 RM              | RLAIF / LLM-as-judge  | GenRM (이 글)                           |
| ---------------------- | ---------------------- | --------------------- | --------------------------------------- |
| 학습 신호              | 사람 선호 쌍 + BT loss | 없음 (zero-shot)      | 사람 선호 쌍 + STaR 부트스트래핑        |
| 출력                   | 실수 $$r(x,y)$$        | 판정 텍스트/토큰      | next-token 확률 $$p(I \mid x,y_1,y_2)$$ |
| 판정 근거 노출         | 없음                   | 있음 (프롬프트 CoT)   | 있음 (학습된 CoT)                       |
| In-distribution 정확도 | 높음                   | 낮음 (BT 대비 9~36%↓) | BT RM과 대등                            |
| OOD 정확도             | 낮음                   | 상대적으로 견고       | BT RM 대비 10~45%↑                      |

# Method

## 왜 명시적 보상함수가 필요 없어지는가

GenRM의 출발점은 RLHF 목적함수를 다시 쓰는 것이다. 기준 응답 $$y_{ref}$$(SFT 모델의 샘플)를 baseline으로 빼면 RL 목적함수는

$$\max_{\pi_\theta} \mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}\Big[\big[r_\phi(x,y) - r_\phi(x,y_{ref})\big] - \beta \mathbb{D}_{KL}\big[\pi_\theta(y\mid x)\,\|\,\pi_{ref}(y\mid x)\big]\Big]$$

가 된다. 그런데 Bradley-Terry 식 (1)을 뒤집으면

$$r_\phi(x,y) - r_\phi(x,y_{ref}) = \log\left(\frac{p_{BT}(y \succ y_{ref} \mid x)}{1 - p_{BT}(y \succ y_{ref} \mid x)}\right)$$

즉 **보상의 차이는 선호 확률의 로그오즈와 정확히 같다.** 이게 GenRM의 핵심 관찰이다: RLHF가 최적화하는 대상은 사실 "점수"가 아니라 "선호 확률"이다. 그렇다면 굳이 점수를 매개로 삼을 이유가 없다. 스칼라 보상 $$r_\phi$$를 명시적으로 정의하는 대신, 선호 확률 $$p_\phi(y_w \succ y_l \mid x)$$ 자체를 — 특정 아키텍처나 분포 가정 없이 — LLM으로 직접 모델링하면 된다.

## GenRM: 선호를 next-token 확률로

<p align="center"><img src="/assets/post/image/generative-reward-models/fig1-methods-overview.png" width="95%"></p>

위 그림(논문 Figure 1)이 세 방법의 차이를 정리한다. Bradley-Terry는 두 응답을 각각 LLM에 넣고 선형 예측기로 확률을 뽑는다. GenRM은 프롬프트와 두 응답 $$(x,y_1,y_2)$$을 **한 번에** 넣고, "정답 지시자 토큰(answer indicator token)" $$I$$의 next-token 확률을 답으로 쓴다. CoT-GenRM은 그 앞에 추론 $$r$$을 먼저 생성하고, 여러 번 샘플링해 다수결(majority vote)로 최종 판정을 낸다.

GenRM(CoT 없음)의 학습 목적함수는 표준 SFT의 cross-entropy와 같은 형태다.

$$\mathcal{L}_{GenRM}(\pi_\phi) = \mathbb{E}_{(x,y_1,y_2,I)\sim D}\big[-\log \pi_\phi(I \mid x,y_1,y_2)\big]$$

- $$I$$: 정답 지시자 토큰. 예를 들어 $$y_1$$이 더 낫다는 판정이면 $$I=I_1$$.
- $$\pi_\phi(I \mid x,y_1,y_2)$$: LLM이 그 토큰을 낼 next-token 확률. 이것 자체가 곧 $$p_\phi(y_1 \succ y_2 \mid x)$$ 역할을 한다.
- 즉 여기엔 별도의 선형 헤드도, 별도의 손실 함수 설계도 없다. **LLM을 그대로 분류기로 쓴다.**

CoT를 추가하는 "CoT-GenRM-Rationalization"은 정답 판정 $$I$$가 주어진 상태에서 그걸 정당화하는 추론 $$r$$을 함께 학습한다(2.2.2절의 post-rationalization을 그대로 적용한 형태다).

$$\mathcal{L}_{GenRM\text{-}Rationalization}(\pi_\phi) = \mathbb{E}_{(x,y_1,y_2,r,I)\sim D}\big[-\log \pi_\phi(I \mid x,y_1,y_2,r) - \log \pi_\phi(r \mid x,y_1,y_2)\big]$$

## STaR-DPO: 판정 자체를 DPO로 최적화한다

여기서부터가 이 논문이 [18편 DPO](/blog/2026/dpo/)와 정확히 만나는 지점이다. 정답 판정 $$I_w$$로 이어진 추론 $$r_w$$와, 오답 판정 $$I_l$$로 이어진 추론 $$r_l$$을 한 쌍으로 묶으면, "옳게 판정한 추론"과 "틀리게 판정한 추론" 사이의 선호 쌍이 생긴다. 이걸 그대로 DPO 목적함수에 넣은 것이 STaR-DPO다.

$$\mathcal{L}_{GenRM\text{-}DPO}(\pi_\phi) = \mathbb{E}_D\left[\log \sigma\left(\beta \log\frac{\pi_\phi(I_w,r_w\mid x,y_1,y_2)}{\pi_{ref}(I_w,r_w\mid x,y_1,y_2)} - \beta \log\frac{\pi_\phi(I_l,r_l\mid x,y_1,y_2)}{\pi_{ref}(I_l,r_l\mid x,y_1,y_2)}\right)\right]$$

이 식은 18편에서 다룬 표준 DPO 손실

$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right)\right]$$

과 형태가 완전히 같다. 차이는 딱 하나, "정책이 생성하는 대상"이 응답 $$y$$가 아니라 **(추론, 판정) 쌍** $$(r,I)$$라는 것뿐이다. DPO가 "정책 자체를 암묵적 보상 모델로 취급한다"는 통찰을 정책 최적화에 썼다면, GenRM은 같은 통찰을 **judge 모델 학습**에 그대로 재사용한다. 여기까지가 "선호 학습과 생성 judge의 결합"의 실체다 — 2부의 BT loss로 판정 정확도를 잡고, 18편의 DPO 트릭으로 어떤 추론이 더 나은 판정을 만드는지까지 학습한다.

STaR 반복 절차는 다음 네 단계를 여러 iteration 반복한다.

1. 현재 모델 $$\pi_\phi$$로 $$(x,y_1,y_2)$$에 대해 추론 $$r$$과 판정 $$I$$를 샘플링한다.
2. 사람이 매긴 실제 선호 라벨과 판정이 일치하는 샘플만 남긴다(STaR 필터링).
3. 남은 샘플로 SFT(Eq. 8) 또는, 맞는 추론과 틀린 추론을 쌍으로 묶어 DPO(Eq. 9)로 재학습한다.
4. 재학습된 모델로 1로 돌아간다.

## 토이 예제: 세 judge가 같은 응답 쌍을 어떻게 보는가

논문 Figure 3의 실제 예시를 가져와 보자. 질문은 "바다에 사는 동물 두 종을 말해줘"다.

- Assistant A: "Dolphin and shark." (돌고래와 상어)
- Assistant B: "Common ocean animals include sharks, whales, and dolphins." (흔한 바다 동물로는 상어, 고래, 돌고래가 있다)

**(a) 스칼라 RM (가상 예시).** 스칼라 RM은 근거를 보여주지 않고 숫자만 낸다고 해보자. $$r(x,\text{A})=0.61$$, $$r(x,\text{B})=0.74$$처럼 B에 더 높은 점수를 줬다고 치면, 우리는 **왜** 그런지 알 방법이 없다. [11편](/blog/2026/rlhf-length-correlations/)에서 다룬 길이-보상 상관관계를 생각하면 "더 길고 항목이 많은 응답을 선호하는" 표면적 휴리스틱일 가능성이 있지만, 스칼라 헤드 안에서 무슨 일이 일어났는지는 근본적으로 검증할 수 없다. 이게 스칼라 RM의 해석 불가능성이다.

**(b) CoT 없는 생성 judge (GenRM, no CoT).** next-token 확률만으로 판정하면 역시 근거는 없다. 다만 이 경우엔 확률값 자체는 볼 수 있다 — 예컨대 $$\pi_\phi(I_B \mid x,y_A,y_B) = 0.71$$처럼. 근거는 없지만 최소한 "판정의 확신도(confidence)"는 얻는다.

**(c) CoT 있는 생성 judge.** 여기서 흥미로운 반전이 나온다. 논문이 실제로 보여주는 두 CoT 판정을 비교하면:

- **Zero-shot LLM-as-judge**(학습 없이 CoT만 프롬프트한 버전): "Assistant B는 더 상세하고, 흔한 바다 동물을 더 포괄적으로 나열한다... 전반적으로 Assistant B의 응답이 더 도움이 되고 상세하다"며 **B를 선택**한다. 질문이 정확히 "두 종"을 요구했다는 점을 놓친 채, "더 많고 상세하다"는 표면적 기준으로 판정한 것이다.
- **STaR-DPO로 학습된 CoT-GenRM**: "Assistant A의 응답은 '깊이와 detail이 부족하다'... 그러나 사용자의 질문은 정확히 두 개의 바다 동물을 말해달라는 것이었고, Assistant B의 응답은 세 종을 나열해 지시를 따르지 않았다"며 **A를 선택**한다.

즉 **CoT를 갖는 것 자체는 정답을 보장하지 않는다.** Zero-shot judge도 추론 과정을 텍스트로 보여주지만, 그 추론이 "장황함 = 좋음"이라는 편향을 그대로 정당화하는 데 쓰였다. STaR로 사람 선호 라벨에 맞춰 추론을 재학습한 뒤에야 "지시사항 준수"라는 진짜 기준을 짚어낸다. 판정 근거가 텍스트로 노출된다는 것과, 그 근거가 실제로 옳은 이유라는 것은 별개의 문제라는 뜻이다.

## CLoud: critique 먼저, 스칼라는 그다음

GenRM이 "스칼라를 아예 버리고 판정 토큰의 확률로 대체"하는 쪽이라면, 비슷한 시기 공개된 [CLoud](https://arxiv.org/abs/2408.11791)(Critique-out-Loud Reward Models, Ankner et al., Databricks/MIT/UC San Diego, 2024)는 **"critique를 먼저 쓰고, 그 critique를 조건으로 스칼라 점수를 매긴다"**는 절충안을 택한다.

<p align="center"><img src="/assets/post/image/generative-reward-models/cloud-overview.png" width="85%"></p>

구조는 이렇다. 프롬프트 $$x$$와 응답 $$y$$가 주어지면, LM head로 자연어 critique를 먼저 샘플링한다.

$$\hat c \sim p(\cdot \mid x,y;\theta_B,\theta_{LM})$$

그다음 별도의 reward head가 $$(x,y,\hat c)$$를 **모두** 입력받아 스칼라 보상을 낸다.

$$\hat R = r_{\theta_B,\theta_R}(x,y,\hat c)$$

학습은 두 손실의 결합이다.

$$\mathcal{L}_{RM}(\theta_B,\theta_R,D) = -\mathbb{E}_{(x,y^-,y^+,c^-,c^+)\sim D}\big[\log\sigma(r_{\theta_B,\theta_R}(x,y^+,c^+) - r_{\theta_B,\theta_R}(x,y^-,c^-))\big]$$

$$\mathcal{L}_{CLoud}(\theta_B,\theta_{LM},\theta_R,D) = \mathcal{L}_{RM}(\theta_B,\theta_R,D) + \lambda \cdot \mathcal{L}_{SFT}(\theta_B,\theta_{LM},D)$$

- $$\mathcal{L}_{RM}$$: critique를 조건으로 한 Bradley-Terry loss. 2부에서 다룬 그 손실 그대로다, 다만 입력에 critique $$c$$가 추가됐을 뿐이다.
- $$\mathcal{L}_{SFT}$$: critique 생성 능력을 유지하기 위한 언어모델링 loss. oracle critique(Llama-3.1-405B-Instruct가 생성)에 대한 negative log-likelihood다.
- $$\lambda$$: 두 손실의 가중치.

영화 평론가에 비유하면 이해가 쉽다. 스칼라 RM은 영화를 보고 별점만 매기는 평론가고, GenRM은 "A가 낫다/B가 낫다"만 말하는 평론가다. CLoud는 **먼저 리뷰를 한 문단 쓴 다음, 그 리뷰를 다시 읽으며 별점을 매기는** 평론가에 가깝다. 판정에 앞서 스스로 쓴 논거를 조건으로 넣기 때문에, 그 논거가 실제로 최종 점수에 영향을 준다.

실제로 RewardBench에서 이 구조가 만드는 격차는 작지 않다. 아래 그림처럼 CLoud는 Chat, Chat-Hard, Safety, Reasoning 네 범주 모두에서 critique 없이 학습한 동일 규모의 classic RM을 앞서고, 평균 정확도는 8B 모델에서 4.65%p, 70B 모델에서 5.84%p 개선된다. Best-of-N(ArenaHard, N=16) 기준으로도 승률이 8B에서 1.84%p, 70B에서 0.89%p 오른다.

<p align="center"><img src="/assets/post/image/generative-reward-models/cloud-rewardbench.png" width="90%"></p>

한 가지 실험적으로 중요한 발견은 **critique를 반드시 온폴리시(on-policy)로 학습해야 한다**는 것이다. CLoud는 oracle critique로 먼저 LM head를 SFT한 뒤, reward head는 그 모델이 **자기 스스로 생성한** critique로 재구성한 데이터셋에 학습시킨다. Oracle critique를 그대로 써서 reward head를 학습시킨(off-policy) 버전은 RewardBench 평균 정확도가 8B에서 5.60%p, 70B에서 3.03%p 떨어졌다. 학습 때 본 것과 추론 때 실제로 만들어내는 것 사이의 분포가 다르면 성능이 무너진다는 뜻이다. 흥미롭게도 GenRM 논문의 STaR-Rationalizer 실험(2.2.2절 참고)에서도 같은 현상이 독립적으로 관찰된다 — 다른 모델(post-rationalization 모델)이 만든, 자기 자신에게는 off-policy인 추론으로 학습하면 held-out 과제에 대한 일반화가 무너진다. 두 논문이 서로 다른 방식으로 같은 결론에 도달한 셈이다.

GenRM 논문은 자신의 Related Work에서 CLoud(Ankner et al., 2024)를 동시대 연구로 직접 언급하며 다음과 같이 대비한다: CLoud는 "추론을 생성하는 별도의 강한 모델(oracle)이 필요하고, Bradley-Terry 하이브리드 아키텍처(별도 reward head)를 유지"하는 반면, GenRM은 "완전히 자기 부트스트랩된 추론을, 추가 아키텍처 없이 순수 언어모델링 형태로" 학습한다. 즉 CLoud는 **"생성의 표현력 + 스칼라의 추론 속도"**를 절충하기 위해 아키텍처를 늘리는 쪽을, GenRM은 아키텍처를 아예 없애는 쪽을 택한 것이다.

| 항목           | 스칼라 RM (2부)      | GenRM (이 글)                                                   | CLoud (하이브리드)                                    |
| -------------- | -------------------- | --------------------------------------------------------------- | ----------------------------------------------------- |
| 출력           | 실수 $$r(x,y)$$ 하나 | 판정 토큰 확률 $$p_\phi(I\mid x,y_1,y_2)$$ (+선택적 추론 $$r$$) | 자연어 critique $$\hat c$$ + 조건부 스칼라 $$\hat R$$ |
| 판정 근거 노출 | 없음 (블랙박스)      | CoT 버전에만 있음                                               | 항상 있음 (critique 자체가 근거)                      |
| 추가 아키텍처  | 선형 예측 헤드 1개   | 없음 (LM head 재사용)                                           | LM head 유지 + 별도 reward head                       |
| 학습 신호      | BT NLL               | 분류 NLL 또는 STaR-DPO                                          | BT NLL + critique SFT NLL 결합                        |
| 추론 비용      | 순전파 1회           | 순전파 1회(no CoT) ~ 다수결 시 N회                              | critique 생성(수십~수백 토큰) + reward head 순전파    |
| 대표 성능      | ID 매칭, OOD 취약    | ID 대등, OOD +10~45%p (BT RM 대비)                              | RewardBench 평균 +4.65~5.84%p (스칼라 RM 대비)        |

## 해석 가능성: 고정된 축 vs 자유 텍스트

[7편 ArmoRM](/blog/2026/armorm/)이 택한 해석 가능성 전략과 GenRM의 전략은 방향이 정반대다. ArmoRM은 정직성·장황함·안전성 등 **19개의 고정된 목적함수**를 사람이 미리 설계하고, MoE 게이팅 네트워크가 BT loss로 학습되어 이 19개 점수를 상황별로 가중합한다. GenRM은 반대로 아무 축도 미리 정하지 않는다. 판정마다 모델이 그때그때 자유 텍스트로 근거를 구성한다.

| 항목           | ArmoRM (7편)                                     | GenRM (이 글)                                                                     |
| -------------- | ------------------------------------------------ | --------------------------------------------------------------------------------- |
| 해석 채널      | 19개의 고정된 목적함수 점수                      | 자유 형식 자연어 근거 $$r$$                                                       |
| 축의 정의 시점 | 학습 이전, 사람이 설계                           | 없음 — 그때그때 생성                                                              |
| 결합 방식      | 게이팅 네트워크가 19개 점수를 가중합             | 근거 뒤 판정 토큰 $$I$$의 next-token 확률                                         |
| 장점           | 축마다 정량 비교·프로그램적 감사 가능            | 정해진 어휘 밖의 이유도 설명 가능                                                 |
| 한계           | 새로운 실패 유형이 19개 축 밖이면 아예 포착 못함 | 근거가 그럴듯해 보여도 실제 판정 이유였다는 보장이 없음(post-hoc rationalization) |

두 전략은 상호 배타적이지 않다. ArmoRM은 "무엇을 봤는지"를 구조화해서 보여주고, GenRM은 "왜 그렇게 판단했는지"를 서술로 보여준다. 다만 위 토이 예제가 보여주듯, 서술형 근거는 겉보기엔 그럴듯해도 실제로는 편향을 정당화하는 도구가 될 수 있다는 위험을 안고 있다.

# Experiments

## Zero-shot: CoT 프롬프팅만으로도 큰 격차가 생긴다

모든 실험은 LLaMA-3.1-8B-Instruct를 기반으로, UltraFeedback(6.1만 쌍, 일반 instruction-following)과 UltraInteract(수학/코드/논리 추론 트리)를 학습 데이터로, RewardBench(Chat / Chat Hard / Safety / Reasoning 네 범주, [9편](/blog/2026/rewardbench-2/)에서 다룬 평가 방법론의 전신)를 OOD 평가로 쓴다.

학습 없이 프롬프팅만 바꿔도 격차가 크다. **CoT 없이 직접 토큰 확률만 뽑으면** UltraFeedback 52.25%, RewardBench 60.60%에 그치지만, **같은 모델에 CoT로 추론을 시키면(zero-shot LLM-as-judge)** UltraFeedback 67.75%, RewardBench 75.18%로 뛴다. 추론을 강제하는 것만으로 15%p 안팎의 정확도가 그냥 생긴다.

<p align="center"><img src="/assets/post/image/generative-reward-models/fig2-id-ood-results.png" width="95%"></p>

## 학습 이후: ID는 대등, OOD는 압도

학습된 모델끼리 비교하면(위 그림) Bradley-Terry RM, PairRM, 학습된 GenRM은 in-distribution(UltraFeedback)에서 모두 73~74%로 대등하고, STaR-DPO도 73.9%로 같은 수준이다. 반면 CoT 없이 SFT만 한 STaR-SFT는 67.4%로 **기저 모델과 사실상 차이가 없다** — CoT 없는 정답 라벨만으로는 판정 능력이 늘지 않는다는 뜻이다.

OOD(RewardBench)에서 격차가 벌어진다. STaR-DPO는 81.9%로, zero-shot LLM-as-judge(77.8%)와 학습된 GenRM(78.9%)을 모두 앞선다. 특히 Safety 범주에서 STaR-DPO는 91.0%인 반면 가장 잘하는 판별형 방법(PairRM)도 81.8%에 그친다. 요약하면, **논문 abstract의 수치**로 zero-shot judge는 ID에서 BT RM보다 9\~36% 낮고, GenRM은 ID에서 BT RM과 대등하면서 OOD에서 10\~45% 더 높으며, zero-shot judge 대비로는 ID 9\~31%, OOD 2\~6% 더 높다.

## 추론이 필요한 과제에서는 명암이 갈린다

UltraInteract(추론 위주)로 학습했을 때, in-distribution 정확도는 STaR-DPO가 90.2%로 STaR-SFT(68.8%, 기저 모델과 동일 수준)를 크게 앞서지만, 명시적 판별형 RM들(약 94%)에는 살짝 못 미친다. 그런데 OOD로 넘어가면 반전이 일어난다. RewardBench의 Reasoning 부분집합에서 Bradley-Terry RM은 **무작위보다도 나쁘고**, 가장 잘하는 GenRM(no CoT)도 70.8%로 zero-shot LLM-as-judge(76.6%)에마저 못 미친다. 이 부분집합에서만 STaR-DPO가 87.2%로 확실히 앞선다. 반대로 RewardBench의 non-reasoning 부분집합에서는 zero-shot LLM-as-judge가 78.0%로 가장 강하고, STaR-DPO는 75.0%로 소폭 낮다. 정리하면 추론이 필요한 OOD 과제일수록 판별형 RM은 완전히 무너지고, CoT 학습을 거친 생성형 judge만 견딘다.

## 부트스트랩 소스가 강할수록 좋은 건 아니다

추론을 어느 모델로부터 부트스트랩하는지도 성능에 영향을 준다(Table 1). 첫 iteration만 부트스트랩 소스를 쓰고, 이후 iteration은 자기 생성 데이터로 재학습한다.

| 부트스트랩 소스                      | Iter.1 (UF / RB) | Iter.2 (UF / RB) | Iter.3 (UF / RB) |
| ------------------------------------ | ---------------- | ---------------- | ---------------- |
| Llama-3.1-8B (자기 자신)             | 68.63 / 77.34    | 67.68 / 77.61    | 67.38 / 77.05    |
| Llama-3.1-70B                        | 70.50 / 77.09    | 70.13 / 68.78    | 69.58 / 63.43    |
| GPT-4                                | 62.85 / 69.58    | 68.55 / 75.63    | 71.73 / 78.29    |
| GPT-4 (전체를 GPT-4 추론으로만 학습) | 62.60 / 70.52    | —                | —                |

자기 자신(8B)의 추론으로 부트스트랩하면 iteration을 거쳐도 안정적이다. 반면 같은 계열의 더 강한 70B 모델의 추론으로 시작하면 iteration이 진행될수록 RewardBench 정확도가 77.09%에서 63.43%까지 떨어진다. GPT-4처럼 훨씬 강하지만 이질적인 모델의 추론은 처음엔(off-policy라서) 낮지만, 반복해서 자기 정책 분포로 끌어올수록(on-policy화될수록) 오히려 78.29%까지 올라가 8B 자기 부트스트랩보다도 높아진다. 저자들은 "critic 모델의 온폴리시 학습이 성능에 유의미한 영향을 준다"고 결론짓는다 — CLoud가 critique 학습에서 발견한 것과 정확히 같은 교훈이다.

## 추론 시점 계산: 다수결의 효과

CoT-GenRM은 여러 개의 추론을 샘플링해 다수결로 최종 판정을 낼 수 있다. 32개 샘플로 다수결을 하면 UltraFeedback에서 +1.6%p, RewardBench에서 +3.8%p, UltraInteract 학습 모델 기준으로는 UltraInteract에서 +4.6%p, RewardBench에서 +4.9%p 개선된다. CLoud도 비슷한 실험(self-consistency decoding, N개 critique를 샘플링해 보상을 평균)을 했는데, 결과가 흥미롭게 엇갈린다. CLoud는 **추론(Reasoning) 범주에서만** 이득을 봤고(8B +0.70%p, 70B +0.49%p), 그마저도 응답의 추론 단계가 1~2단계로 짧은 문제에서만 일관되게 좋아졌다. 두 논문 모두 "다수결이 추론 과제에서 특히 효과적"이라는 같은 신호를 보내지만, 이걸 어떻게 체계적으로 최대화할지는 아직 이 논문들의 범위 밖이다 — [24편](/blog/2026/deepseek-grm-spct/)이 이 지점을 정면으로 다룬다.

# Conclusion

GenRM의 메시지를 한 줄로 요약하면: **RLHF의 정확함과 RLAIF의 견고함을 동시에 가지려면, 명시적 보상함수를 버리고 선호 확률 자체를 LLM의 next-token 예측으로 학습하면 된다.** 그리고 CoT로 추론을 노출시키되, 그 추론이 실제로 옳은 이유를 짚도록 STaR-DPO(=judge 수준에 적용한 DPO)로 학습해야 편향을 이겨낼 수 있다는 것이다. CLoud는 이걸 조금 다른 방식으로 절충한다 — 아키텍처를 없애는 대신, critique를 조건으로 삼는 reward head를 하나 더 둠으로써 생성의 해석력과 스칼라의 속도를 함께 가져간다.

한계도 분명하다.

- **추론 비용.** CoT-GenRM은 판정마다 수백 토큰의 추론을 생성해야 하고, 다수결까지 쓰면 N배로 늘어난다. 스칼라 RM의 순전파 1회와는 비교가 안 된다. CLoud도 critique 생성만큼의 추가 지연이 붙는다.
- **생성형 judge 특유의 편향.** 토이 예제에서 본 verbosity bias(장황한 응답을 무조건 선호) 외에도, 논문은 프롬프트 설계 단계에서부터 position bias(응답 순서가 판정에 영향을 주는 것)를 피하려 명시적으로 지시문을 넣는다. self-enhancement bias(judge가 자기 자신과 비슷한 스타일의 응답을 선호하는 것) 같은 문제는 이 논문에서 직접 다루지 않는다. 이런 편향들이 실제로 얼마나 쉽게 판정을 뒤집을 수 있는지는 [26편](/blog/2026/one-token-to-fool-judge/)에서 훨씬 극단적인 형태로 확인하게 된다.

정확한 판정 근거를 보여주는 judge라 해도, 그 근거가 사람을 설득하기 위한 그럴듯한 텍스트일 뿐일 위험은 여전히 남는다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물네 번째 글이다.

**1부. 지형도**

1. [Deep RL from Human Preferences (Christiano 2017)](/blog/2026/deep-rl-human-preferences/) — 선호로 보상을 배우는 원형
2. [InstructGPT (Ouyang 2022)](/blog/2026/instructgpt/) — RLHF 3단계 표준 레시피
3. [HH-RLHF (Bai 2022)](/blog/2026/anthropic-hh-rlhf/) — helpful·harmless preference model

**2부. 스칼라 RM 해부**

4. [Rethinking Bradley-Terry (2024)](/blog/2026/bradley-terry-rethinking/) — reward 변환의 수학적 기반
5. [Secrets of RLHF II (2024)](/blog/2026/secrets-rlhf-reward-modeling/) — 선호 데이터 노이즈와 RM 일반화
6. [Skywork-Reward (2024)](/blog/2026/skywork-reward/) — 데이터 큐레이션이 아키텍처를 이긴다
7. [ArmoRM (2024)](/blog/2026/armorm/) — 다목적 분해와 MoE 게이팅
8. [Llama 2 (2023)](/blog/2026/llama2-rlhf/) — helpfulness·safety RM 분리 프로덕션 레시피
9. [RewardBench 2 (2025)](/blog/2026/rewardbench-2/) — RM을 어떻게 평가할 것인가

**3부. Reward Hacking**

10. [Overoptimization Scaling Laws (2022)](/blog/2026/reward-model-overoptimization/) — Goodhart의 법칙 정량화
11. [Length Correlations in RLHF (2023)](/blog/2026/rlhf-length-correlations/) — 성능 향상의 얼마가 길이인가
12. [ODIN (2024)](/blog/2026/odin-disentangled-reward/) — 길이를 reward에서 분리
13. [WARM (2024)](/blog/2026/warm-weight-averaged-reward/) — weight averaging으로 hacking 방어

**4부. reward를 정책으로**

14. [PPO (2017)](/blog/2026/ppo/) — clipped surrogate objective
15. [Secrets of RLHF I (2023)](/blog/2026/secrets-rlhf-ppo/) — PPO 학습 안정화 트릭
16. [GRPO / DeepSeekMath (2024)](/blog/2026/grpo-deepseekmath/) — value network를 버리다
17. [RLOO (2024)](/blog/2026/rloo-back-to-basics/) — REINFORCE로 충분한가
18. [DPO (2023)](/blog/2026/dpo/) — reward를 없애면 어떻게 되는가

**5부. Process & Verifiable Reward**

19. [Let's Verify Step by Step (2023)](/blog/2026/lets-verify-step-by-step/) — 과정 감독이 결과 감독을 이긴다
20. [Math-Shepherd (2023)](/blog/2026/math-shepherd/) — 사람 라벨 없는 PRM
21. [DeepSeek-R1 (2025)](/blog/2026/deepseek-r1/) — RLVR, 규칙이 reward가 될 때

**6부. Generative Reward Model**

22. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
23. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
24. **(현재 글)** Generative Reward Models (2024) — GenRM과 선호 학습의 결합
25. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
26. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

본 시리즈는 31편으로 구성된다.

# 참고 문헌

- Mahan et al., 2024. [Generative Reward Models](https://arxiv.org/abs/2410.12832).
- Ankner et al., 2024. [Critique-out-Loud Reward Models](https://arxiv.org/abs/2408.11791). [코드](https://github.com/zankner/CLoud)
- Zelikman et al., 2022. [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465).
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Bradley & Terry, 1952. Rank Analysis of Incomplete Block Designs.
- Wang et al., 2024. [Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM)](https://arxiv.org/abs/2406.12845).
- Lambert et al., 2024. [RewardBench](https://arxiv.org/abs/2403.13787).
