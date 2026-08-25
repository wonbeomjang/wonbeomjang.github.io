---
layout: post
title: "RLOO: PPO의 절반은 RLHF에 필요 없었다"
date: 2026-08-11 09:23:00 +0900
description: "RL Reward 설계 시리즈 #23 — 응답 전체를 하나의 action으로 보면 REINFORCE로 충분하다는 주장"
categories: [paper]
tags: [rlhf, reinforce, rloo, ppo, policy-gradient, paper]
giscus_comments: true
related_posts: true
---

> [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) (Ahmadian et al., Cohere For AI, ACL 2024)

# Introduction

[#22 GRPO](/blog/2026/grpo-deepseekmath/)는 "critic을 유지할 비용이 없다"는 논리로 value network를 없앴다. 그룹 안에서 스스로를 정규화하면 별도의 critic 없이도 advantage를 만들 수 있다는 것이었다. 이번 글의 논문은 같은 결론 — PPO 없이도 RLHF가 된다 — 에 도착하지만, 출발점이 전혀 다르다. 이 논문은 비용을 말하지 않는다. 대신 **"RLHF라는 문제를 애초에 잘못 모델링했다"**고 말한다.

PPO는 각 토큰을 하나의 action으로, 그 토큰까지의 부분 시퀀스를 하나의 state로 본다. 그래서 [PPO([14편](/blog/2026/ppo/))](/blog/2026/ppo/)는 상태마다 가치를 추정하는 critic을 두고, GAE로 편향과 분산을 저울질하고, 정책이 너무 멀리 가지 않도록 토큰 단위로 clipping을 건다. 이 모든 장치는 "한 시퀀스 안에서 상태가 계속 바뀌고, 각 상태마다 가치가 다르다"는 전제 위에 서 있다.

그런데 RLHF에서 보상은 언제 주어지는가? 사람이 매기는 것도, reward model이 매기는 것도 **다 만들어진 응답 하나**에 대해서다. "이 문장까지는 잘 썼고 다음 단어는 별로다" 같은 진짜 보상은 세상 어디에도 없다. 그렇다면 애초에 토큰 하나하나를 별도의 상태로 나눌 이유가 있을까? 이 논문(Ahmadian et al., 2024)의 답은 "없다"이다. 응답 전체를 하나의 행동(action)으로, 프롬프트를 하나의 초기 상태로 보는 **bandit 문제**로 재정의하면, GAE도 value network도 clipping도 자연스럽게 사라진다. 남는 것은 1992년에 나온 가장 오래된 정책 경사법, REINFORCE다. 그리고 여기에 샘플 여러 개를 곁들인 확장판이 이 논문이 제안하는 **RLOO(REINFORCE Leave-One-Out)**다.

결론부터 요약하면 이렇다. Vanilla policy gradient(REINFORCE)조차 PPO를 win-rate 기준 3.2~20.3%p 앞섰고, RLOO는 PPO·DPO·RAFT를 전부 이겼다. [#21 Secrets of RLHF](/blog/2026/secrets-rlhf-ppo/)가 "PPO를 안정적으로 굴리는 트릭"을 다뤘다면, 이 글은 "애초에 그 트릭들이 왜 필요했는지부터 다시 묻는다"는 점에서 결이 다르다.

# Background

RLHF의 3단계 파이프라인은 이 시리즈에서 여러 번 다뤘다. SFT로 초기화한 정책 $$\pi_{sft}$$, 선호 쌍 $$(x, y^+, y^-)$$로 학습한 reward model $$r_\phi$$, 그리고 이 reward model을 최대화하도록 정책을 online RL로 미세조정하는 3단계다. RL 단계의 목적함수는 다음과 같다.

$$\max_{\pi_\theta} \mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}\left[r_\phi(x,y) - \beta D_{KL}\left(\pi_\theta(\cdot\mid x) \,\|\, \pi_{ref}(\cdot\mid x)\right)\right]$$

- $$r_\phi(x,y)$$: reward model이 프롬프트 $$x$$와 응답 $$y$$ 전체에 매기는 점수.
- $$\beta$$: KL 페널티의 세기. 정책이 참조 모델 $$\pi_{ref}$$에서 너무 멀어지지 않도록 잡아준다.
- $$D_{KL}$$: 정책과 참조 정책 사이의 분포 거리.

이 목적함수를 전개하면 아래의 "KL-shaped reward"를 기대값으로 최대화하는 것과 같다.

$$R(x,y) = r_\phi(x,y) - \beta \log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$$

여기서 중요한 포인트 하나. [PPO([14편](/blog/2026/ppo/))](/blog/2026/ppo/)는 이 $$R(x,y)$$를 토큰 단위로 쪼갠다.

$$R(x,y) = \sum_{t=1}^{T} R_t(x,y_t)$$

$$\mathrm{EOS}$$ 토큰을 생성하는 순간에만 $$r_\phi(x,y)$$가 더해지고, 그 이전의 모든 토큰에서는 $$\log\dfrac{\pi_\theta(y_t\mid s_t)}{\pi_{ref}(y_t\mid s_t)}$$만 $$R_t$$를 채운다. 즉 마지막 토큰 전까지는 "진짜 보상"이 아니라 KL항만 있는 셈이다. 이 사실이 이 논문 전체 논증의 출발점이 된다.

# Method

## PPO가 짊어진 세 가지 장치, 그리고 왜 필요 없어지는가

PPO의 목적함수는 토큰 단위 clipped surrogate다.

$$\min\left(f(y_t\mid s_t)\hat{A}_\lambda(y_t,s_t),\ \mathrm{clip}(f(y_t\mid s_t),\,1-\epsilon,\,1+\epsilon)\hat{A}_\lambda(y_t,s_t)\right)$$

$$f(y_t\mid s_t) = \dfrac{\pi_\theta(y_t\mid s_t)}{\pi_{old}(y_t\mid s_t)}$$

- $$s_t = \{y_{<t}, x\}$$: 프롬프트와 지금까지 생성한 토큰들로 이루어진 "상태".
- $$f(y_t \mid s_t)$$: 새 정책과 이전 정책의 확률비. 정책이 얼마나 바뀌었는지를 잰다.
- $$\hat{A}_\lambda$$: GAE로 추정한 advantage. critic이 학습한 value function을 이용해 편향과 분산을 조절한다 ($$\lambda \in [0,1]$$).
- $$\epsilon$$: 확률비가 $$[1-\epsilon, 1+\epsilon]$$ 밖으로 나가면 잘라내는 clipping ratio.

이 논문은 이 장치들을 하나씩 뜯어보며 "각각이 왜 원래는 필요했는지, 그리고 왜 RLHF에서는 필요 없는지"를 단계적으로 논증한다.

**1단계 — 보상은 완성된 시퀀스에만 있다.** Background에서 봤듯, $$\mathrm{EOS}$$ 토큰 이전의 모든 $$R_t$$는 진짜 보상이 아니라 KL 페널티일 뿐이다. 부분 시퀀스를 "상태"로 나누는 것 자체가, 그 상태마다 서로 다른 가치를 갖는다는 전제를 필요로 하는데, 그 가치의 근거가 되는 보상 신호 자체가 없다.

**2단계 — 그러면 이건 bandit이다.** 논문은 이렇게 정리한다. 환경(문맥)이 다음 토큰에 의해 결정론적으로 바뀌므로 $$P_D(\{y_{<t+1},x\} \mid s_t, y_t) = 1$$이다. 상태 전이가 결정론적이고 보상이 끝에만 있다면, 이 문제는 초기 상태(프롬프트)와 종결 상태(생성 완료) 두 개뿐인 **MDP — 즉 bandit**으로 환원된다. 응답 전체 $$y$$를 하나의 action으로 보는 것이다.

$$\mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}[R(y,x)\,\nabla_\theta \log \pi_\theta(y\mid x)]$$

이것이 REINFORCE 추정량이다. 부분 시퀀스라는 개념 자체가 사라지므로, "부분 시퀀스의 가치를 예측하는 함수"인 value network(critic)도 함께 사라진다.

**3단계 — GAE도 불필요하다.** GAE가 편향을 감수하고 분산을 줄이는 이유는, 전통적인 Deep RL 환경이 무작위 초기화에서 시작해 action space가 거대하고 분산이 실제로 학습을 망칠 만큼 크기 때문이다. 그런데 RLHF의 정책은 사전학습 + SFT를 거친 강한 초기화 상태에서 출발한다. 확률 질량이 이미 소수의 그럴듯한 토큰에 몰려 있어서, 이론상 action space는 거대해도 실질적인 분산은 크지 않다. 논문은 GAE의 $$\lambda$$를 $$\{0, 0.5, 0.95, 1.0\}$$으로 바꿔가며 검증했는데, 분산을 전혀 줄이지 않는 $$\lambda=1.0$$(순수 REINFORCE, "Vanilla PG")이 가장 좋은 성능을 냈다.

<p align="center"><img src="/assets/post/image/rloo-back-to-basics/fig1_ppo_lambda_ablation.png" width="55%"></p>

- $$\lambda$$가 1에서 멀어질수록(분산을 더 줄일수록) 학습이 오히려 느려진다.
- 이는 "분산을 줄이는 대가로 편향을 지불할 이유가 없다"는 가설을 정확히 뒷받침한다. RLHF는 이미 안정적인 환경이기 때문이다.

**4단계 — clipping도 사실상 무용지물이다.** Clipping은 정책이 한 스텝 사이에 너무 크게 바뀌는 것을 막기 위한 장치다. 그런데 저자들이 실제 학습 배치를 뜯어보니, 배치당 clip이 발동한 비율은 평균 **5% 미만**이었다. 즉 학습이 거의 항상 on-policy에 가깝게 진행되고 있었다는 뜻이다. 아예 clipping과 확률비 $$f$$를 제거해 $$\lambda=1$$로 두면 PPO의 손실은 그대로 Vanilla PG로 축소되는데, 성능은 오히려 소폭 향상됐다.

이 네 단계를 통과하면 PPO에 남는 것은 없다. Critic도, GAE도, clipping도 모두 "부분 시퀀스를 상태로 본다"는 하나의 전제에서 파생된 장치였고, 그 전제 자체가 RLHF에는 맞지 않았다는 것이다.

## REINFORCE에 baseline 붙이기

순수 REINFORCE 추정량은 분산이 크다. 분산을 줄이려면 그래디언트 추정치와 상관관계가 높은 baseline $$b$$를 빼주면 된다. $$b$$가 $$y$$와 무관하기만 하면 추정량은 여전히 불편(unbiased)이다.

$$\mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}[(R(y,x) - b)\,\nabla_\theta \log \pi_\theta(y\mid x)]$$

가장 단순한 선택은 학습 전체에 걸친 보상의 이동평균이다.

$$b_{MA} = \frac{1}{S}\sum_{s} R(x_s, y_s)$$

$$S$$는 학습 스텝 수, $$(x_s, y_s)$$는 스텝 $$s$$의 프롬프트-응답 쌍이다. 이 baseline은 구현이 쉽고 계산도 거의 공짜지만, 문제가 하나 있다. 지금 이 프롬프트, 이 응답과는 아무 관련 없는 "과거 전체의 평균"이라서, 프롬프트마다 난이도가 다른 RLHF에서는 다소 무딘 기준이다.

## RLOO: 나 빼고 나머지 평균을 baseline으로

한 프롬프트에 대해 $$k$$개의 응답을 동시에 샘플링할 수 있다면 더 나은 baseline을 만들 수 있다. 아이디어는 단순하다. **각 샘플의 baseline으로, 그 샘플을 제외한 나머지 $$k-1$$개 샘플의 평균 보상을 쓴다.** 이것이 Kool et al.(2019)이 제안하고 이 논문이 RLHF에 가져온 RLOO(REINFORCE Leave-One-Out) 추정량이다.

$$\frac{1}{k}\sum_{i=1}^{k}\left[R(y_{(i)}, x) - \frac{1}{k-1}\sum_{j\neq i} R(y_{(j)}, x)\right]\nabla_\theta \log \pi_\theta(y_{(i)} \mid x),\quad y_{(1)},\dots,y_{(k)} \overset{\text{i.i.d.}}{\sim} \pi_\theta(\cdot\mid x)$$

- $$k$$: 한 프롬프트당 뽑는 online 샘플 수.
- $$y_{(i)}$$: $$i$$번째로 뽑은 응답.
- $$\dfrac{1}{k-1}\sum_{j\neq i} R(y_{(j)}, x)$$: 샘플 $$i$$를 제외한 나머지 $$k-1$$개 응답의 평균 보상. 이것이 샘플 $$i$$의 baseline이다.
- $$R(y_{(i)}, x) - (\text{baseline})$$: 샘플 $$i$$의 advantage. "이 샘플이 다른 샘플들보다 얼마나 더 좋았는가"를 뜻한다.

핵심은 이 baseline에 파라미터가 하나도 없다는 점이다. Critic처럼 별도로 학습시킬 필요 없이, 매 학습 스텝마다 그 자리에서(on-the-fly) 나머지 샘플들로부터 계산된다. 그래서 논문은 이를 **"파라미터 없는 value function"**이라고 부른다. 다만 대가도 있다. 프롬프트마다 $$k$$개를 생성해야 하니 샘플링 시간이 늘어난다.

## 토이 예제: $$k=4$$일 때 advantage 손으로 계산하기

숫자로 직접 따라가 보자. 한 프롬프트에 대해 $$k=4$$개의 응답을 샘플링했고, reward model이 매긴 점수가 각각 $$(0.8,\ 0.2,\ 0.5,\ 0.9)$$라고 하자.

전체 합은 $$0.8+0.2+0.5+0.9 = 2.4$$다. 각 샘플 $$i$$의 baseline은 "합에서 자신을 뺀 값"을 $$k-1=3$$으로 나눈 것이다.

| 샘플 $$i$$ | 보상 $$r_i$$ | 나머지 3개 합 | LOO baseline | advantage $$r_i - b_i$$ |
| :--------: | :----------: | :-----------: | :----------: | :---------------------: |
|     1      |     0.8      |      1.6      |    0.533     |         +0.267          |
|     2      |     0.2      |      2.2      |    0.733     |         −0.533          |
|     3      |     0.5      |      1.9      |    0.633     |         −0.133          |
|     4      |     0.9      |      1.5      |    0.500     |         +0.400          |

읽는 법은 이렇다. 샘플 4($$r=0.9$$, 가장 높은 점수)의 baseline은 나머지 세 샘플 $$(0.8, 0.2, 0.5)$$의 평균인 $$0.5$$다. 자기 자신은 baseline 계산에 전혀 관여하지 않는다. 그래서 advantage는 $$0.9-0.5=+0.4$$로, 순수하게 "다른 세 응답과 비교했을 때 얼마나 더 좋았는가"만 담는다. 반대로 샘플 2($$r=0.2$$, 가장 낮은 점수)는 나머지 평균(0.733)보다 한참 낮아 advantage가 $$-0.533$$으로 가장 크게 깎인다. 참고로 네 advantage를 다 더하면 $$0.267-0.533-0.133+0.4 \approx 0$$이 된다 — leave-one-out 구조상 항상 그렇게 된다.

## GRPO와 같은 숫자, 다른 계산

[#22 GRPO](/blog/2026/grpo-deepseekmath/)는 같은 상황에서 다른 방식을 쓴다. **자기 자신을 포함한** $$k$$개 전체의 평균과 표준편차로 정규화한다.

$$A_i^{GRPO} = \frac{r_i - \mathrm{mean}(r_1,\dots,r_k)}{\mathrm{std}(r_1,\dots,r_k)}$$

같은 $$(0.8, 0.2, 0.5, 0.9)$$로 계산해보자. 평균은 $$0.6$$, 분산은 $$\dfrac{(0.2)^2+(-0.4)^2+(-0.1)^2+(0.3)^2}{4}=0.075$$이므로 표준편차는 $$\sqrt{0.075}\approx 0.274$$다.

| 샘플 $$i$$ | 보상 $$r_i$$ | RLOO advantage (나 제외 평균) | GRPO advantage (나 포함 평균·표준편차) |
| :--------: | :----------: | :---------------------------: | :------------------------------------: |
|     1      |     0.8      |            +0.267             |                 +0.730                 |
|     2      |     0.2      |            −0.533             |                 −1.461                 |
|     3      |     0.5      |            −0.133             |                 −0.365                 |
|     4      |     0.9      |            +0.400             |                 +1.095                 |

숫자만 봐도 성격이 다르다는 게 드러난다. RLOO의 advantage는 보상과 같은 단위를 유지한 채 "나머지 대비 차이"를 그대로 보여주는 반면, GRPO는 표준편차로 나눠 스케일을 없앤 z-score라서 크기가 훨씬 부풀려져 있다. 그리고 결정적인 차이가 하나 더 있다. GRPO는 $$k$$개의 보상이 전부 같으면 표준편차가 0이 되어 advantage가 $$0/0$$으로 정의되지 않는다(그래서 실전에서는 작은 $$\epsilon$$을 더해 나눈다). RLOO는 나눗셈이 아니라 뺄셈이므로, 이 경우에도 각 샘플의 baseline이 자기 자신을 제외한 나머지의 평균으로 그대로 계산되고 advantage는 자연스럽게 0이 된다 — 별도의 예외 처리가 필요 없다.

비유하자면 이렇다. 시험을 같이 본 친구 4명이 자기 점수가 잘 나온 건지 판단하려 한다. 나를 포함한 전체 평균과 비교하면, 내 점수가 이미 그 평균에 섞여 들어가 있어서 비교 기준 자체가 내 성적에 오염된다. 반면 "나를 뺀 나머지 세 명의 평균"과 비교하면, 순수하게 다른 사람들 대비 내가 얼마나 잘했는지를 잰다. RLOO가 하는 일이 정확히 이것이다.

또 하나의 비유는 오디션 무대다. 심사위원은 노래 중간의 한 소절 한 소절에 점수를 매기지 않는다. 무대가 끝난 뒤에야 전체 공연 하나에 점수를 준다. RLHF의 보상도 마찬가지로 토큰 하나하나가 아니라 완성된 응답 하나에만 매겨진다 — 이것이 이 논문이 처음부터 세운 전제였다.

## 세 알고리즘 한눈에 비교

| 항목                  | PPO                                                    | GRPO ([#22](/blog/2026/grpo-deepseekmath/))            | RLOO (이 논문)                                                |
| :-------------------- | :----------------------------------------------------- | :----------------------------------------------------- | :------------------------------------------------------------ |
| baseline/advantage    | 학습된 critic + GAE ($$\lambda$$로 bias-variance 조절) | 그룹 내 **자기 포함** 평균·표준편차로 정규화 (z-score) | 그룹 내 **자기 제외** $$k-1$$개 평균 (leave-one-out)          |
| critic(value network) | 필요                                                   | 불필요                                                 | 불필요                                                        |
| 보상 모델링 단위      | 토큰(부분 시퀀스를 상태로)                             | 토큰 (advantage는 그룹 단위, PPO식 loss에 broadcast)   | 응답 전체 (bandit, 단일 action)                               |
| clipping              | 토큰 단위 ratio clipping 유지                          | PPO 스타일 clipped surrogate 유지                      | 불필요 — 제거해도 성능 저하 없음을 실증                       |
| 동시 적재 모델 수     | 4개 (generator, ref, critic, RM)                       | 3개 (generator, ref, RM)                               | 3개 (generator, ref, RM)                                      |
| 샘플 수 $$k$$         | 1 (on-policy 단일 샘플 + critic bootstrap)             | $$k \geq 2$$ (그룹)                                    | $$k \geq 2$$, 논문 실험은 $$k=\{2,4\}$$                       |
| 적합 상황             | 약한 초기화·큰 action space의 전통 Deep RL             | 정답이 명확해 상대 비교가 쉬운 검증 가능 보상(수학 등) | 강하게 초기화된 LLM + 보상이 완성 응답에만 존재하는 표준 RLHF |

이 표에서 가장 눈여겨볼 지점은 clipping 행이다. GRPO는 critic을 없앴지만 PPO의 clipped surrogate 형태(토큰별 확률비를 clip)는 그대로 가져갔다. 반면 RLOO는 애초에 부분 시퀀스라는 개념 자체를 버렸기 때문에 clipping을 유지할 대상(토큰별 확률비)도 사라진다. GRPO가 "비용이 드는 부품(critic)만 빼자"는 절충이라면, RLOO는 "애초에 이 설계 전체가 틀린 문제 정의 위에 있었다"는 더 근본적인 재작성에 가깝다.

# Experiments

## 실험 세팅

논문은 TL;DR Summarize(SFT 11.6만 건, 선호 쌍 9.3만 건)와 Anthropic Helpful & Harmless(선호 쌍 11.2만 건) 두 데이터셋에서, Pythia-6.9B와 Llama-7B 두 base 모델로 실험했다. 비교 대상은 RLOO, RAFT, REINFORCE(이동평균 baseline), Vanilla PG, PPO, DPO 여섯 가지다. 평가는 held-out 테스트 프롬프트에 대해 AlpacaFarm 프레임워크로 GPT-4를 프록시 심사위원 삼아 simulated win-rate를 측정했다.

## 최종 win-rate

<p align="center"><img src="/assets/post/image/rloo-back-to-basics/fig2_reward_curves.png" width="80%"></p>

- 세 데이터셋·모델 조합 모두에서 RLOO(파란선)가 가장 먼저, 가장 높게 올라간다. Vanilla PG(회색)는 REINFORCE 계열보다 항상 아래에 있다 — 부분 시퀀스를 상태로 본다는 전제가 실제로 손해였다는 뜻이다.
- PPO(보라선)는 세 그래프 모두에서 가장 느리고 낮다. 특히 HH(Llama)에서 격차가 가장 크다.

| Method                | TL;DR | HH (Pythia) | HH (Llama) |
| :-------------------- | :---: | :---------: | :--------: |
| RLOO (k=4)            | 77.9  |    43.7     |    64.1    |
| RAFT (k=4)            | 73.2  |    42.1     |    63.3    |
| RLOO (k=2)            | 74.2  |    47.6     |    62.2    |
| RAFT (k=2)            | 72.1  |    37.7     |    58.4    |
| REINFORCE w/ baseline | 70.7  |    37.9     |    55.3    |
| Vanilla PG            | 70.4  |    36.4     |    52.3    |
| PPO                   | 67.6  |    29.2     |    32.0    |
| DPO                   | 66.6  |    39.0     |    61.9    |

RLOO(k=4)는 PPO 대비 win-rate를 TL;DR에서 **+10.3%p**, HH(Pythia)에서 **+14.5%p**, HH(Llama)에서 무려 **+32.1%p** 끌어올렸다. 그리고 세 조합 중 두 곳에서 DPO보다도 높다 — HH(Pythia)에서는 DPO(39.0)를 RLOO(k=2)가 47.6으로, HH(Llama)에서는 DPO(61.9)를 RLOO(k=4)가 64.1로 앞섰다. 이 논문이 "RL을 없앤 DPO보다 RL을 제대로 쓴 RLOO가 낫다"고 주장하는 근거가 바로 이 표다.

또한 REINFORCE with baseline은 Vanilla PG와 거의 대등하거나(TL;DR: 70.7 vs 70.4) 오히려 앞선다(HH-Llama: 55.3 vs 52.3) — 여러 샘플 없이 baseline 하나만 바꿔도 "전체 시퀀스를 하나로 본다"는 프레이밍 자체의 효과가 나타난다는 뜻이다.

## RAFT 대비 샘플 효율

<p align="center"><img src="/assets/post/image/rloo-back-to-basics/fig3_rloo_vs_raft_sample_efficiency.png" width="80%"></p>

- 같은 예산 $$k$$에서 RLOO(파란선)는 항상 RAFT(초록선)보다 위에 있다.
- 더 놀라운 지점은 점선(RLOO, $$k=2$$)이 실선(RAFT, $$k=4$$)을 세 그래프 모두에서 따라잡거나 앞선다는 것이다. **절반의 샘플링 예산으로 RAFT를 이긴다.**

RAFT는 $$k$$개 중 가장 점수가 높은 샘플 하나만 골라 cross-entropy로 학습하고 나머지는 버린다. 반면 RLOO는 $$k$$개 전부를 그래디언트 추정에 쓴다. 세 데이터셋·모델을 평균하면 RLOO는 $$k=2,4$$일 때 각각 61.3, 61.9의 win-rate를 기록한 반면, RAFT는 같은 조건에서 56.1, 59.5에 그쳤다. 가장 큰 격차는 HH(Pythia), $$k=2$$ 조건에서 나타난 **+9.9%p**다.

## 보상 노이즈에 대한 강건성

Reward model은 그 자체로 완벽하지 않은 노이즈 낀 신호다. 저자들은 이를 시뮬레이션하기 위해 binary classifier의 출력 로짓에 가우시안 노이즈를 더했다.

$$r_\sigma(x,y) = r(x,y) + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma^2)$$

$$\sigma \in \{1.0,\ 3.0,\ 5.0\}$$으로 노이즈 세기를 키워가며 RAFT와 RLOO의 학습 곡선을 비교했다.

| 노이즈 세기 $$\sigma$$ | RAFT 훈련 보상(노이즈 미포함 기준) | RLOO 훈련 보상(노이즈 미포함 기준) |
| :--------------------: | :--------------------------------- | :--------------------------------- |
|          1.0           | 소폭 하락                          | 소폭 하락                          |
|          3.0           | 뚜렷하게 하락                      | 거의 변화 없음                     |
|          5.0           | 크게 하락                          | 상대적으로 견고                    |

RAFT는 top-1 샘플만 골라 쓰기 때문에, 노이즈가 순위 하나만 뒤집어도 "가장 좋다고 믿고 학습한 샘플"이 실제로는 나쁜 샘플일 수 있다. 반면 RLOO는 $$k$$개 전체의 상대적 비교로 advantage를 만들기 때문에 개별 순위가 흔들려도 손상이 평균으로 흡수된다. 노이즈가 커질수록($$\sigma=3,5$$) 이 차이는 더 뚜렷해졌다.

## KL 페널티에 대한 민감도

같은 논리가 KL 페널티 세기 $$\beta$$에도 적용된다. $$\beta \in \{0.1,\ 0.25,\ 0.5,\ 1.0\}$$으로 바꿔가며 HH(Pythia), $$k=2$$ 조건에서 RAFT와 RLOO를 비교했다.

|          $$\beta$$           | 관찰된 경향                                                                            |
| :--------------------------: | :------------------------------------------------------------------------------------- |
|       0.1 (약한 규제)        | RLOO와 RAFT가 참조 정책과의 KL 거리는 비슷하지만, RLOO의 보상이 더 높다                |
| 0.25 / 0.5 / 1.0 (강한 규제) | RAFT는 보상 최적화도 더 나쁘고, 참조 정책에서 더 멀리 벗어난다 — 두 지표 모두에서 악화 |

$$\beta$$가 커질수록 RAFT의 랭킹 오류(어떤 샘플이 "1등"인지 잘못 고를 위험)가 커지고, top-1만 학습하는 구조라 그 오류가 그대로 학습에 반영된다. RLOO는 다시 한번 $$k$$개 전체를 쓰는 구조 덕에 이 민감도에서 자유롭다.

## Alignment tax: 길이·다양성·유창성

| Method                | 평균 길이 | Perplexity | Diversity-1 | Diversity-2 | 보상 분산 |
| :-------------------- | :-------: | :--------: | :---------: | :---------: | :-------: |
| RLOO (k=4)            |   60.6    |    27.6    |    0.10     |    0.43     |    3.1    |
| RAFT (k=4)            |   62.4    |    30.1    |    0.10     |    0.43     |    3.2    |
| RLOO (k=2)            |   58.6    |    29.2    |    0.11     |    0.44     |    3.0    |
| REINFORCE w/ baseline |   47.2    |    27.2    |    0.13     |    0.50     |    2.7    |
| Vanilla PG            |   39.1    |    39.0    |    0.15     |    0.54     |    3.7    |
| PPO                   |   16.5    |    40.4    |    0.34     |    0.60     |    2.3    |
| DPO                   |   104.4   |    33.8    |    0.08     |    0.39     |    N/A    |

DPO는 평균 104토큰까지 늘어지는 verbosity 문제를 보이는 반면, PPO는 16.5토큰까지 짧아지며 성능과 무관하게 응답을 뭉개는 경향을 보인다. RLOO와 RAFT, REINFORCE with baseline은 이 두 극단 사이에서 비교적 안정적인 길이·perplexity·다양성을 유지하면서도 더 높은 보상을 얻었다 — reward hacking으로 지표만 올리는 게 아니라는 뜻이다.

# Conclusion

한 줄로 요약하면: **RLHF에서 보상은 완성된 응답에만 있으므로, 부분 시퀀스를 상태로 모델링하는 PPO의 장치(GAE, critic, clipping)는 애초에 필요 없었고, 응답 전체를 하나의 action으로 보는 bandit 프레이밍 위에서 REINFORCE와 그 다중 샘플 확장 RLOO만으로 PPO·DPO·RAFT를 모두 이길 수 있다.**

[#22 GRPO](/blog/2026/grpo-deepseekmath/)와 이 논문은 "critic 없는 RLHF"라는 같은 목적지에 서로 다른 길로 도착했다. GRPO는 비용 절감이라는 실용적 논리로 critic만 걷어냈고, clipping과 토큰 단위 프레이밍은 그대로 남겨뒀다. 이 논문은 문제 정의 자체를 다시 써서 clipping까지 포함한 PPO 전체를 걷어냈다. 그런데 이 논문은 한 걸음 더 나간다. RLOO가 **DPO보다도 낫다**고 주장한다 — RL 자체를 없앤 방법보다, RL을 제대로 다시 설계한 방법이 낫다는 것이다.

이 지점이 바로 다음 글로 이어지는 부채다. 이 논문의 실험은 TL;DR 요약과 HH 대화처럼 상대적으로 짧고 단일 턴에 가까운 과제에 국한되어 있고, reward model 하나에 크게 의존한다. [#24 DPO](/blog/2026/dpo/)는 정반대의 방향에서 질문한다 — reward model도, online sampling도, RL 루프 자체도 다 걷어내고 선호 쌍에서 정책을 직접 학습할 수 있다면 어떨까? 이 논문이 "RLOO가 DPO를 이긴다"고 결론지은 바로 그 지점에서, 다음 글은 "RL 자체가 정말 필요한가"라는 더 근본적인 질문을 던지며 긴장을 이어간다.

# 참고 문헌

- Ahmadian et al., 2024. [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740). ACL 2024.
- [ACL Anthology: Back to Basics](https://aclanthology.org/2024.acl-long.662/)
- Kool, van Hoof & Welling, 2019. [Buy 4 REINFORCE Samples, Get a Baseline for Free!](https://openreview.net/forum?id=r1lgTGL5DE) (RLOO 원 논문)
- Williams, 1992. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. (REINFORCE 원 논문)
- Dong et al., 2023. [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment](https://arxiv.org/abs/2304.06767).
- Rafailov et al., 2023. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).
- Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). (GRPO 원 논문)

---

# RL Reward 설계 시리즈

이 글은 RL Reward 설계 시리즈의 스물세 번째 글이다.

**1부. 지형도**

<ol start="1">
  <li><a href="/blog/2026/deep-rl-human-preferences/">Deep RL from Human Preferences (Christiano 2017)</a> — 선호로 보상을 배우는 원형</li>
  <li><a href="/blog/2026/instructgpt/">InstructGPT (Ouyang 2022)</a> — RLHF 3단계 표준 레시피</li>
  <li><a href="/blog/2026/anthropic-hh-rlhf/">HH-RLHF (Bai 2022)</a> — helpful·harmless preference model</li>
</ol>

**2부. 스칼라 RM 해부**

<ol start="4">
  <li><a href="/blog/2026/bradley-terry-rethinking/">Rethinking Bradley-Terry (2024)</a> — reward 변환의 수학적 기반</li>
  <li><a href="/blog/2026/secrets-rlhf-reward-modeling/">Secrets of RLHF II (2024)</a> — 선호 데이터 노이즈와 RM 일반화</li>
  <li><a href="/blog/2026/skywork-reward/">Skywork-Reward (2024)</a> — 데이터 큐레이션이 아키텍처를 이긴다</li>
  <li><a href="/blog/2026/armorm/">ArmoRM (2024)</a> — 다목적 분해와 MoE 게이팅</li>
  <li><a href="/blog/2026/llama2-rlhf/">Llama 2 (2023)</a> — helpfulness·safety RM 분리 프로덕션 레시피</li>
  <li><a href="/blog/2026/rewardbench-2/">RewardBench 2 (2025)</a> — RM을 어떻게 평가할 것인가</li>
</ol>

**3부. Reward Hacking**

<ol start="10">
  <li><a href="/blog/2026/reward-model-overoptimization/">Overoptimization Scaling Laws (2022)</a> — Goodhart의 법칙 정량화</li>
  <li><a href="/blog/2026/rlhf-length-correlations/">Length Correlations in RLHF (2023)</a> — 성능 향상의 얼마가 길이인가</li>
  <li><a href="/blog/2026/odin-disentangled-reward/">ODIN (2024)</a> — 길이를 reward에서 분리</li>
  <li><a href="/blog/2026/sycophancy/">Sycophancy (2023)</a> — RM은 사실보다 동의를 좋아한다</li>
  <li><a href="/blog/2026/warm-weight-averaged-reward/">WARM (2024)</a> — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="15">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
  <li><a href="/blog/2026/deliberative-alignment/">Deliberative Alignment (2024)</a> — 안전 명세를 모델의 추론 안으로</li>
  <li><a href="/blog/2026/shallow-safety-alignment/">Shallow Safety Alignment (2024)</a> — 정렬은 첫 몇 토큰에만 얹혀 있다</li>
  <li><a href="/blog/2026/or-bench/">OR-Bench (2024)</a> — 과잉 거절을 어떻게 측정할 것인가</li>
</ol>

**5부. reward를 정책으로**

<ol start="20">
  <li><a href="/blog/2026/ppo/">PPO (2017)</a> — clipped surrogate objective</li>
  <li><a href="/blog/2026/secrets-rlhf-ppo/">Secrets of RLHF I (2023)</a> — PPO 학습 안정화 트릭</li>
  <li><a href="/blog/2026/grpo-deepseekmath/">GRPO / DeepSeekMath (2024)</a> — value network를 버리다</li>
  <li><strong>(현재 글)</strong> RLOO (2024) — REINFORCE로 충분한가</li>
  <li><a href="/blog/2026/dpo/">DPO (2023)</a> — reward를 없애면 어떻게 되는가</li>
  <li><a href="/blog/2026/simpo/">SimPO (2024)</a> — reference-free + 길이 정규화</li>
  <li><a href="/blog/2026/kto/">KTO (2024)</a> — 선호 쌍 없이 이진 신호만으로</li>
  <li><a href="/blog/2026/gspo/">GSPO (2025)</a> — importance ratio를 시퀀스 단위로</li>
  <li><a href="/blog/2026/dapo/">DAPO (2025)</a> — 신호 없는 프롬프트를 버린다</li>
  <li><a href="/blog/2026/bond/">BOND (2024)</a> — Best-of-N을 추론 비용 없이</li>
  <li><a href="/blog/2026/warp/">WARP (2024)</a> — 정책을 weight space에서 병합</li>
</ol>

**6부. Process & Verifiable Reward**

<ol start="31">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="34">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge**

<ol start="39">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 에이전트는 무엇이 다른가**

<ol start="44">
  <li><a href="/blog/2026/agentic-rl-landscape/">에이전트 RL은 무엇이 다른가</a> — 장기 지평·희소 보상·긴 궤적</li>
  <li><a href="/blog/2026/credit-assignment-survey/">공을 어디에 돌릴 것인가</a> — credit assignment 47개 방법의 지도</li>
  <li><a href="/blog/2026/multi-turn-rl-practice/">멀티턴 RL 실무 가이드</a> — 무엇이 실제로 작동하는가</li>
</ol>

**10부. credit assignment — 공을 어디에 돌릴 것인가**

<ol start="47">
  <li><a href="/blog/2026/outcome-vs-process-agentic/">결과만으로는 부족하다</a> — 장기 지평에서 증폭되는 RLVR의 한계</li>
  <li><a href="/blog/2026/turn-level-reward/">턴 단위로 공을 나눈다</a> — turn-level reward 설계</li>
  <li><a href="/blog/2026/step-level-credit/">스텝을 단위로 삼는다</a> — 행동 단위 궤적 표현과 credit</li>
  <li><a href="/blog/2026/token-segment-credit/">토큰과 세그먼트로 더 잘게</a> — 세밀한 입도의 득과 실</li>
  <li><a href="/blog/2026/reward-shaping-agentic/">shaping은 약인가 독인가</a> — 중간 보상의 효율과 위험</li>
</ol>

**11부. 에이전트의 reward는 어디서 오나**

<ol start="52">
  <li><a href="/blog/2026/environment-as-reward/">환경이 곧 reward다</a> — 샌드박스·테스트·상태 검증</li>
  <li><a href="/blog/2026/tool-call-reward/">도구 호출을 어떻게 채점하나</a> — ToolRL·ToolRM</li>
  <li><a href="/blog/2026/agentic-judge-rubric/">궤적을 judge가 채점한다</a> — rubric 생성형 reward의 확장</li>
</ol>

**12부. 에이전트 도메인별 설계**

<ol start="55">
  <li><a href="/blog/2026/search-agent-rl/">검색 에이전트</a> — Search-R1에서 DeepDive까지</li>
  <li><a href="/blog/2026/swe-agent-rl/">코드 에이전트</a> — SWE-RL과 테스트라는 reward</li>
  <li><a href="/blog/2026/web-gui-agent-rl/">웹·GUI 에이전트</a> — end-to-end 멀티턴 RL</li>
</ol>

**13부. 에이전트의 실패와 방어**

<ol start="58">
  <li><a href="/blog/2026/agentic-reward-hacking/">에이전트의 reward hacking</a> — 판정기가 뚫린다, 그리고 조합의 실패</li>
</ol>

**14부. 실전 종합**

<ol start="59">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어의 helpfulness reward 설계</a> — 열한 개 모델이 능력 축에서 택한 것</li>
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/frontier-agentic-rl/">프론티어 모델은 실제로 어떻게 하나</a> — 최신 모델들의 agentic RL 설계</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 62편으로 구성된다.
