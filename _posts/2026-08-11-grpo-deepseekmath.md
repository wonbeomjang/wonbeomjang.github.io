---
layout: post
title: "GRPO: value network를 버리고 그룹 안에서 비교하다"
date: 2026-08-11 09:22:00 +0900
description: "RL Reward 설계 시리즈 #22 — critic 없이 그룹 상대 advantage로 PPO를 대체한 현재의 사실상 표준"
categories: [paper]
tags: [rlhf, grpo, ppo, reasoning, deepseek, paper]
giscus_comments: true
related_posts: true
---

> [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., DeepSeek-AI, arXiv 2024)

# Introduction

[#20 PPO 글](/blog/2026/ppo/)에서는 clipped surrogate objective로 정책이 한 번에 너무 멀리 가지 않게 막는 방법을 봤고, [#21 Secrets of RLHF I 글](/blog/2026/secrets-rlhf-ppo/)에서는 그 PPO를 LLM 규모에서 실제로 굴리려면 얼마나 많은 트릭 — advantage whitening, reward scaling, critic warmup — 이 필요한지를 봤다. 그런데 그 트릭들이 왜 필요했는지 한 걸음 물러서서 보면, 원인은 하나로 좁혀진다. **PPO는 policy와 별도로 critic(value network)을 하나 더 학습시켜야 한다.** 이 글이 다루는 DeepSeekMath 논문은 이 원인 자체를 없애버리는 방향을 택한다.

DeepSeekMath는 표면적으로는 수학 전용 언어모델(DeepSeekMath 7B) 논문이다. Common Crawl에서 걸러낸 120B 토큰 규모의 수학 코퍼스로 DeepSeek-Coder-Base-v1.5 7B를 계속 사전학습시키고, 외부 도구나 투표(voting) 없이 MATH 벤치마크에서 51.7%를 찍는다. 하지만 이 시리즈가 주목하는 건 사전학습 파이프라인이 아니라, 이 논문이 RL 단계에서 제안한 **GRPO(Group Relative Policy Optimization)** 다. 지금 오픈소스 reasoning 모델 학습 파이프라인 대부분 — DeepSeek-R1 계열은 물론 그 이후 나온 상당수의 reasoning 모델 — 이 PPO 대신 GRPO나 그 변형을 쓴다. 이 시리즈에서 실무 적중률이 가장 높은 글이 이 글이라고 봐도 무방하다.

GRPO의 답은 급진적일 정도로 단순하다. **critic을 통째로 들어낸다.** 대신 같은 프롬프트에 대해 여러 개의 응답을 한꺼번에 샘플링하고, 그 응답들끼리 서로 비교해서 baseline을 만든다. value network가 "이 상태에서 기대되는 미래 보상이 얼마인가"를 학습으로 추정하는 대신, "같은 문제를 놓고 같이 시험 본 친구들보다 내가 얼마나 잘했는가"를 그 자리에서 계산해버리는 것이다.

이 글에서 답할 질문은 세 가지다.

1. **critic을 없애면 advantage를 어떻게 만드나**: 그룹 평균·표준편차 정규화의 정확한 수식과 계산 과정.
2. **KL penalty는 왜 reward가 아니라 loss로 옮겨갔나**: 위치를 바꾼 이유와 그 결과.
3. **왜 하필 수학·코드 같은 도메인에서 유독 잘 맞나**: 그룹 비교와 rule-based reward의 궁합.

미리 말해두면 GRPO는 이 시리즈의 [#23 RLOO 글](/blog/2026/rloo-back-to-basics/)과 사상적으로 매우 가깝다. 둘 다 "critic 없이 그룹 안에서 baseline을 만든다"는 아이디어를 공유한다. 다만 표준편차로 나누는지, 평균을 어떻게 잡는지에서 갈라지는데, 이 차이는 뒤에서 토이 예제로 직접 짚는다.

# Background

## PPO의 critic: 무엇을 위해 필요했나

[#20 PPO 글](/blog/2026/ppo/)에서 다뤘듯, PPO의 advantage 추정은 GAE(Generalized Advantage Estimation)를 쓴다. GAE는 매 타임스텝마다 상태 가치 $$V(s_t)$$가 필요하고, 이 $$V$$는 policy와는 별도로 학습되는 신경망 — 즉 critic이다. LLM RLHF에서 이 critic은 보통 policy와 비슷한 크기의 또 다른 언어모델로 초기화된다. DeepSeekMath 논문은 이 지점을 정확히 지적한다.

> "As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden."

DeepSeekMath-RL의 policy는 7B 모델이다. PPO 방식이었다면 이만한 critic을 하나 더 얹어야 했다는 뜻이다. 여기에 reward model, reference model까지 더하면 GPU에 동시에 올려야 하는 대형 모델이 최대 4개가 된다. 그중 실제로 학습(gradient 업데이트)되는 건 policy와 critic 둘인데, Adam 계열 옵티마이저를 쓰면 파라미터마다 momentum·variance 상태를 추가로 들고 있어야 하니 학습 대상 모델의 실질 메모리 비용은 파라미터 수 대비 몇 배로 불어난다.

## 왜 LLM에서 토큰 단위 value 추정이 특히 어려운가

critic이 무겁다는 것만이 문제가 아니다. 더 근본적인 문제는 **critic이 애초에 정확하게 학습되기 어려운 구조**라는 데 있다. 논문은 이렇게 짚는다.

> "Usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token."

reward model은 보통 응답 전체가 끝난 시점(EOS 토큰)에서만 스칼라 점수 하나를 낸다. 그런데 critic은 그 응답을 만드는 도중의 **모든 토큰마다** "지금부터 끝까지 얼마나 좋은 보상을 받을지"를 추정해야 한다. 비유하자면, 한 학기 내내 숙제에는 점수를 전혀 매기지 않다가 기말고사 한 번으로 성적을 확정한 뒤 "9월 셋째 주 수업 태도가 최종 성적에 얼마나 기여했는지" 역산하라는 것과 비슷하다. 원리적으로 풀 수야 있지만, 신호가 시퀀스 끝에만 존재하는데 중간 지점마다 정확한 값을 요구하니 추정이 흔들리기 쉽다. [#21 Secrets of RLHF I 글](/blog/2026/secrets-rlhf-ppo/)에서 봤던 advantage whitening, critic warmup 같은 안정화 트릭들의 상당수는 결국 이 불안정한 critic을 억지로 붙잡아두기 위한 장치였다.

## 질문: critic 없이 baseline을 만들 수 있을까

그렇다면 애초에 각 토큰의 value를 학습으로 추정하지 않고, 다른 방식으로 baseline을 구할 수는 없을까. GRPO의 답은 "같은 프롬프트에 대해 여러 응답을 동시에 뽑아서, 그 응답들의 평균을 baseline으로 쓰자"는 것이다. 정책이 스스로를 참조점으로 삼는 셈이다. 학습된 함수 대신 **그 순간의 샘플들**이 baseline을 대신한다.

# Method

## 그룹을 만들고, 그 안에서 비교한다

<p align="center"><img src="/assets/post/image/grpo-deepseekmath/ppo_vs_grpo.png" width="70%"></p>

위 그림(논문 Figure 4)이 PPO와 GRPO의 구조 차이를 보여준다. PPO는 policy, value(critic), reward, reference 네 모델이 모두 필요하고 value model이 GAE 계산을 위해 매 스텝의 상태 가치를 추정한다. GRPO는 value model 자리를 통째로 비우고, 대신 하나의 질문 $$q$$에 대해 $$G$$개의 응답 $$\{o_1, \dots, o_G\}$$를 이전 정책 $$\pi_{\theta_{old}}$$에서 한꺼번에 샘플링한 뒤, reward model(또는 rule-based reward)로 각각 점수 $$r_1, \dots, r_G$$를 매긴다. 이 $$G$$개의 점수가 곧 그룹이고, 그룹의 평균과 표준편차가 baseline을 대신한다.

## 목적함수

GRPO의 목적함수는 다음과 같다(논문 식 3).

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^{G}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert} \sum_{t=1}^{\lvert o_i \rvert} \left\{ \min\left[ \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t},\ \mathrm{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right] - \beta\, \mathbb{D}_{KL}[\pi_\theta \Vert \pi_{ref}] \right\} \right]
$$

기호를 하나씩 풀면 다음과 같다.

- $$G$$: 한 질문당 샘플링하는 응답 개수. 논문은 $$G = 64$$를 썼다.
- $$o_i$$, $$o_{i,t}$$: $$i$$번째 응답 전체, 그리고 그 응답의 $$t$$번째 토큰.
- $$\pi_\theta(o_{i,t} \mid q, o_{i,<t}) \mathbin{/} \pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})$$: PPO와 동일한 importance ratio. 새 정책과 샘플링 당시 정책의 토큰 확률 비율이다.
- $$\hat{A}_{i,t}$$: $$i$$번째 응답, $$t$$번째 토큰의 advantage. 뒤에서 그룹 정규화로 계산한다.
- $$\mathrm{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)$$: PPO와 똑같은 clipped surrogate. 비율이 $$[1-\varepsilon, 1+\varepsilon]$$를 벗어나면 잘라서 한 번의 업데이트가 정책을 과도하게 흔들지 못하게 막는다.
- $$\beta \, \mathbb{D}_{KL}[\pi_\theta \Vert \pi_{ref}]$$: 참조 모델과의 KL divergence에 계수 $$\beta$$를 곱한 페널티. **여기가 PPO와 위치가 달라지는 지점**이다 — 아래에서 따로 다룬다.

전체 구조는 PPO의 clipped surrogate를 거의 그대로 가져오되, advantage를 만드는 방식과 KL을 넣는 위치만 바꿨다는 점이 핵심이다.

## Advantage: 그룹 평균·표준편차 정규화

응답 전체에 보상 하나만 주어지는 outcome supervision의 경우, advantage는 그룹 내 정규화로 계산된다.

$$
\hat{A}_{i,t} = \widetilde{r}_i = \frac{r_i - \mathrm{mean}(r_1, r_2, \dots, r_G)}{\mathrm{std}(r_1, r_2, \dots, r_G)}
$$

- $$r_i$$: $$i$$번째 응답이 받은 스칼라 보상.
- $$\mathrm{mean}(r_1, \dots, r_G)$$: 같은 질문에 대해 뽑은 $$G$$개 응답 보상의 평균. 이게 곧 baseline이다.
- $$\mathrm{std}(r_1, \dots, r_G)$$: 같은 그룹의 표준편차. 보상의 스케일을 그룹마다 맞춰준다.
- $$\widetilde{r}_i$$: 정규화된 보상. outcome supervision에서는 이 값이 응답 $$i$$의 **모든 토큰에 동일하게** broadcast된다 — 응답 전체가 하나의 advantage를 공유한다.

이 계산은 사실 낯설지 않다. 수능 원점수 80점이 좋은 점수인지는 그 시험이 쉬웠는지 어려웠는지에 달려 있어서, 같은 시험을 본 사람들끼리의 평균·표준편차로 표준점수(z-score)를 만들어 상대 위치를 매긴다. GRPO의 advantage도 정확히 이 z-score 계산이다. "80점"이라는 절대 점수 대신 "이번 그룹에서 상대적으로 얼마나 잘했는가"만 남긴다.

각 추론 단계마다 보상이 주어지는 process supervision의 경우엔 조금 다르다. 각 단계의 보상을 먼저 정규화한 뒤, 토큰 $$t$$의 advantage는 그 이후에 오는 단계들의 정규화된 보상 합으로 계산한다.

$$
\hat{A}_{i,t} = \sum_{\mathrm{index}(j) \ge t} \widetilde{r}_i^{\,\mathrm{index}(j)}
$$

이건 "지금 이 토큰 이후에 잘한 단계들만 이 토큰의 공로로 친다"는 뜻이다. process reward는 이후 [#31](/blog/2026/lets-verify-step-by-step/), [#32](/blog/2026/math-shepherd/) 글에서 본격적으로 다룬다. DeepSeekMath 실험 자체는 outcome supervision 버전을 주로 쓴다.

## 토이 예제: 응답 4개, 보상 (0.8, 0.2, 0.5, 0.9)

숫자로 직접 따라가 보자. 한 프롬프트에 대해 $$G = 4$$개의 응답을 뽑았고, 각각 reward model이 매긴 점수가 $$0.8, 0.2, 0.5, 0.9$$였다고 하자.

**1단계 — 그룹 평균**

$$
\mathrm{mean}(r) = \frac{0.8 + 0.2 + 0.5 + 0.9}{4} = \frac{2.4}{4} = 0.6
$$

**2단계 — 그룹 표준편차** (그룹 크기 $$G$$로 나누는 모집단 표준편차 기준. 구현체에 따라 $$G-1$$로 나누기도 하지만 결과의 상대적 크기 차이는 크지 않다)

편차는 각각 $$0.2, -0.4, -0.1, 0.3$$이고, 제곱합은 $$0.04 + 0.16 + 0.01 + 0.09 = 0.30$$이므로

$$
\mathrm{std}(r) = \sqrt{\frac{0.30}{4}} = \sqrt{0.075} \approx 0.274
$$

**3단계 — 응답별 advantage**

| 응답 $$i$$ | 보상 $$r_i$$ | 편차 $$r_i - \mathrm{mean}(r)$$ | advantage $$\hat{A}_i = \widetilde{r}_i$$ |
| ---------- | ------------ | ------------------------------- | ----------------------------------------- |
| 1          | 0.8          | +0.2                            | +0.730                                    |
| 2          | 0.2          | -0.4                            | -1.461                                    |
| 3          | 0.5          | -0.1                            | -0.365                                    |
| 4          | 0.9          | +0.3                            | +1.095                                    |

평균보다 낮았던 응답 2, 3은 음의 advantage를 받아 그 토큰들의 확률을 낮추는 방향으로, 평균보다 높았던 응답 1, 4는 양의 advantage를 받아 확률을 올리는 방향으로 gradient가 흐른다. **critic 없이, 4개의 숫자와 사칙연산만으로** baseline과 advantage가 동시에 나왔다.

PPO/GAE라면 이 계산이 완전히 다르게 흘러간다. GAE는

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

로, 응답 하나(위 표의 "응답 1" 하나)의 매 토큰마다 critic이 추정한 $$V(s_t)$$가 필요하고, 그 값들을 시간축을 따라 bootstrap하며 누적해야 advantage가 나온다. 응답 하나에 토큰이 200개면 critic이 200번 값을 추정해야 하고, 그 추정치들이 부정확하면 advantage 전체가 흔들린다. GRPO는 이 과정을 통째로 "그룹 안에서 평균과 표준편차 구하기"로 대체한 것이다.

이 예제의 보상 $$(0.8, 0.2, 0.5, 0.9)$$는 [#23 RLOO 글](/blog/2026/rloo-back-to-basics/)에서 같은 그룹으로 leave-one-out baseline을 계산할 때 다시 등장한다. 표준편차로 나누는 GRPO와, 평균만 쓰되 자기 자신을 뺀 나머지로 baseline을 잡는 RLOO가 같은 숫자에서 얼마나 다른 advantage를 내는지 그 글에서 직접 비교할 수 있다.

## KL penalty: reward에서 손실로

PPO 계열 RLHF([#21 글](/blog/2026/secrets-rlhf-ppo/) 참고)는 보통 KL penalty를 **reward에 섞는다.** 매 토큰마다 $$r_t - \beta \log(\pi_\theta / \pi_{ref})$$ 형태로 보상 자체를 깎아서, 참조 모델에서 멀어질수록 실질 보상이 줄어들게 만드는 방식이다. GRPO는 이걸 reward에 섞지 않고, 목적함수(손실)에 별도의 항으로 직접 더한다. KL divergence 자체는 Schulman이 제안한 비음수(non-negative) unbiased estimator를 쓴다(논문 식 4).

$$
\mathbb{D}_{KL}[\pi_\theta \Vert \pi_{ref}] = \frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

이 형태는 $$f(x) = x - \log x - 1$$ 꼴로, $$x = 1$$(즉 $$\pi_\theta = \pi_{ref}$$)일 때만 0이고 그 외에는 항상 양수다. 그래서 KL 추정치가 음수로 튀는 일 없이 항상 "참조 모델과 얼마나 멀어졌는가"를 안정적으로 잰다.

왜 위치를 옮겼을까. reward에 KL을 섞으면, 그 reward가 그룹 정규화의 입력으로 들어간다. 즉 "응답이 정답과 얼마나 가까운가"라는 원래 성과 신호에 "참조 모델에서 얼마나 벗어났는가"라는 별개의 신호가 섞인 채로 평균·표준편차를 구하게 되어, 그룹 내 상대 비교라는 GRPO의 핵심 계산 자체가 흐려진다. 회사 조직에 비유하면, 이번 달 벌점을 월급 명세서에서 미리 까버리면 "이 달 실적이 제일 좋았던 사람이 누구인가"를 비교할 때 벌점이 실적 순위까지 흔들어버린다. GRPO는 이 둘을 분리한다 — 그룹 비교(advantage)는 순수하게 과제 성과(reward)로만 매기고, 참조 모델 이탈에 대한 페널티는 손실 함수 하단에 별도 항목으로 붙인다.

## 학습 절차 요약

논문 Algorithm 1의 흐름을 요약하면 다음과 같다.

1. 질문 $$q$$마다 이전 정책 $$\pi_{\theta_{old}}$$로 $$G$$개 응답을 샘플링한다.
2. reward model(또는 rule 기반 채점)로 각 응답에 점수를 매긴다.
3. 그룹 평균·표준편차로 $$\hat{A}_{i,t}$$를 계산한다.
4. 식(3)의 $$\mathcal{J}_{GRPO}$$로 정책을 업데이트한다. 논문 설정에서는 탐색 한 번당 업데이트를 한 번만 한다.
5. (iterative RL) reward model을 새 정책이 낸 데이터로 계속 미세조정하며 재사용한다.

# Experiments

## GRPO 적용 전후 성능

DeepSeekMath-Instruct 7B에 GRPO를 적용한 DeepSeekMath-RL 7B의 결과다.

| 벤치마크    | Instruct (RL 전) | RL (GRPO 적용 후) | 변화   |
| ----------- | ---------------- | ----------------- | ------ |
| GSM8K (CoT) | 82.9%            | 88.2%             | +5.3%p |
| MATH (CoT)  | 46.8%            | 51.7%             | +4.9%p |
| CMATH       | 84.6%            | 88.8%             | +4.2%p |

MATH 51.7%는 외부 도구나 투표 기법 없이 얻은 수치다. self-consistency(64개 샘플 다수결, maj@64)까지 쓰면 MATH에서 60.9%까지 오른다 — 하지만 이건 추론 시점에 64배의 연산을 더 쓴 결과이므로, GRPO 자체의 개선폭인 +4.9%p와는 성격이 다른 숫자다.

## RL은 새 능력을 만드나, 있던 능력을 드러낼 뿐인가

<p align="center"><img src="/assets/post/image/grpo-deepseekmath/pass_maj_k.png" width="70%"></p>

논문 5.2.1절("Why RL Works?")은 Pass@K와 Maj@K를 나눠서 본다. Pass@K는 K개 중 하나라도 맞으면 성공으로 치는 지표, Maj@K는 K개 중 다수결로 최종 답을 정했을 때의 정확도다. 위 그림(논문 Figure 7)이 보여주듯, **GRPO 적용 후 Maj@K는 오르지만 Pass@K는 거의 그대로다.** 논문의 해석은 이렇다.

> "RL enhances Maj@K's performance but not Pass@K. These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust."

즉 RL이 SFT 모델은 못 풀던 문제를 새로 풀 수 있게 만드는 게 아니라, 이미 낮은 확률로나마 정답을 낼 수 있던 모델의 **출력 분포에서 정답 쪽 확률 질량을 키우는** 역할을 한다는 뜻이다. Pass@K가 그대로라는 건 "모델이 가진 잠재 능력의 상한"은 SFT 단계에서 이미 결정돼 있고, GRPO는 그 상한 안에서 다수결을 돌렸을 때 이기는 빈도를 높이는 셈이다.

## 학습 하이퍼파라미터

| 하이퍼파라미터              | 값   |
| --------------------------- | ---- |
| 그룹 크기 $$G$$             | 64   |
| KL 계수 $$\beta$$           | 0.04 |
| policy 학습률               | 1e-6 |
| 학습 batch size             | 1024 |
| 탐색 단계당 policy 업데이트 | 1회  |

$$G = 64$$라는 숫자가 중요하다. PPO였다면 critic 하나로 응답 하나씩 순차적으로 value를 추정했겠지만, GRPO는 애초에 프롬프트 하나당 64개 응답을 병렬로 뽑는 것을 전제로 설계됐다. critic 없이 baseline을 만드는 대가로 샘플링 비용이 늘어난 것이다.

## 표: PPO와 GRPO 비교

| 항목                  | PPO                                            | GRPO                                                                              |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| Critic(value network) | 필요 — policy와 비슷한 크기의 모델             | 불필요                                                                            |
| 동시 적재 모델 수     | 최대 4개 (policy, critic, reward, reference)   | 최대 3개 (critic 제거)                                                            |
| Baseline 출처         | 학습된 value function $$V(s_t)$$               | 같은 프롬프트에서 뽑은 $$G$$개 응답의 평균                                        |
| Advantage 계산        | GAE — 토큰별 TD residual을 시간축으로 누적     | 그룹 reward의 평균·표준편차 정규화                                                |
| KL penalty 위치       | reward에 섞음 (토큰별)                         | 손실 함수에 별도 항으로 추가                                                      |
| 프롬프트당 필요 샘플  | 1개 (on-policy trajectory)                     | $$G$$개 (논문은 64개)                                                             |
| 적합한 상황           | 범용, dense reward, 그룹 샘플링 비용이 큰 환경 | 검증 가능/그룹 비교 가능한 도메인(수학, 코드), reward model 호출이 비교적 싼 경우 |

# Conclusion

GRPO는 reward 자체를 바꾸는 논문이 아니다. **advantage를 만드는 방법**을 바꾼 논문이다. critic이 학습으로 추정하던 baseline을, 같은 프롬프트에서 뽑은 $$G$$개 응답의 평균·표준편차로 대체했다. 그 결과 PPO가 짊어졌던 policy 크기만 한 critic의 메모리·연산 부담이 사라졌고, KL penalty를 reward에서 손실로 옮겨 그룹 비교 신호를 순수하게 유지했다. GSM8K +5.3%p, MATH +4.9%p라는 개선폭보다, Pass@K는 그대로인데 Maj@K만 오른다는 관찰이 더 중요한 메시지다. GRPO는 새 능력을 만드는 게 아니라 이미 있는 능력의 출력 분포를 정답 쪽으로 재정렬한다.

다만 이 방식에는 뚜렷한 한계가 있다. 그룹 내 $$G$$개 응답이 **모두 정답이거나 모두 오답**이면, $$r_i - \mathrm{mean}(r)$$이 모든 $$i$$에 대해 0이 되어 advantage 자체가 사라진다. 표준편차로 나누기 이전에 이미 학습 신호가 0인 것이다. 너무 쉬운 문제(항상 맞음)나 너무 어려운 문제(항상 틀림)에 대해서는 $$G$$개를 샘플링하고 채점하는 연산을 쓰고도 gradient가 전혀 나오지 않는다는 뜻이다. 또한 표준편차로 나누는 정규화가 정말 필요한지, 오히려 편향을 만드는 건 아닌지도 뒤따르는 논쟁의 대상이 된다 — [#23 RLOO 글](/blog/2026/rloo-back-to-basics/)이 이 지점을 정면으로 파고든다. 그리고 reward를 학습된 모델이 아니라 정답 여부를 판정하는 규칙(rule-based reward)으로 완전히 대체하면 어떻게 되는지는 [#33 DeepSeek-R1 글](/blog/2026/deepseek-r1/)에서 다룬다.

# 참고 문헌

- Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).
- [arXiv HTML version](https://arxiv.org/html/2402.03300v3) — 수식·그림 원본 확인.

---

# RL Reward 설계 시리즈

이 글은 RL Reward 설계 시리즈의 스물두 번째 글이다.

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
  <li><strong>(현재 글)</strong> GRPO / DeepSeekMath (2024) — value network를 버리다</li>
  <li><a href="/blog/2026/rloo-back-to-basics/">RLOO (2024)</a> — REINFORCE로 충분한가</li>
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
