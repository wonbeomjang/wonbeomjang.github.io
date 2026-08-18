---
layout: post
title: "Secrets of RLHF I: PPO 학습은 왜 터지는가"
date: 2026-08-11 09:15:00 +0900
description: "RLHF Reward 설계 시리즈 #17 — reward scaling, advantage normalization, policy 제약까지 PPO-max 안정화 레시피"
categories: [paper]
tags: [rlhf, ppo, training-stability, implementation, paper]
giscus_comments: true
related_posts: true
---

> [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964) (Zheng et al., Fudan University, arXiv 2023)

# Introduction

[#16 PPO 글](/blog/2026/ppo/)에서는 clipped surrogate objective가 왜 안전한지를 수식으로 증명했다. 비율 $$\pi_\theta / \pi_{\theta_{\text{old}}}$$ 를 $$[1-\epsilon, 1+\epsilon]$$ 로 묶어두면, 한 스텝의 정책 변화가 신뢰 구간을 벗어나지 않는다는 게 핵심이었다. 이 논리는 advantage 추정이 정확하고 보상 신호가 믿을 만하다는 전제 위에서 성립한다.

문제는 RLHF에서 이 전제가 둘 다 깨진다는 점이다. 이 글이 다루는 [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)(Fudan NLP Group·ByteDance, 2023)는 그 깨짐을 정면으로 파고든 논문이다. 저자들은 "policy constraints가 PPO를 제대로 작동시키는 핵심 요인"이라고 결론 내리고, vanilla PPO 위에 십여 개의 안정화 트릭을 얹은 **PPO-max**를 제안한다. 16편이 PPO의 이론이었다면 이 글은 "그 이론을 LLM에 그대로 붙이면 왜 폭발하고, 무엇을 더해야 살아남는가"다.

RLHF 학습이 불안정한 이유는 구조적이다.

1. **reward model은 proxy다.** RM은 사람의 선호를 근사한 대리 채점자일 뿐, 진짜 선호 함수가 아니다. 논문은 실제 사례를 보여준다 — 중국어 데이터에서 사실을 지어내지만 길고 그럴듯한 답변에 RM이 훨씬 높은 점수를 주고, 영어 데이터에서는 부정확하지만 자신 있게 답하는 응답이 정직하게 "모른다"고 답한 응답보다 높은 점수를 받는다. RM을 곧이곧대로 최대화하면 이런 맹점을 파고드는 정책이 나온다.
2. **보상이 sparse하다.** 사람 피드백 기반 보상은 시퀀스 전체가 끝난 뒤 단 하나의 스칼라로만 주어진다. 수십~수백 토큰짜리 응답의 어느 토큰이 잘했고 어느 토큰이 못했는지는 critic(value network)이 부트스트랩으로 스스로 추정해야 한다.
3. **정책이 이동하며 RM의 신뢰 구간을 벗어난다.** RM은 SFT 정책이 생성하는 분포 근처에서만 학습됐다. PPO가 정책을 계속 밀어붙이면 응답이 RM의 학습 분포 바깥(OOD)으로 나가고, 그 지점에서 RM 점수는 더는 사람 선호의 근사치가 아니다.

이 세 가지가 겹치면 무슨 일이 벌어지는지 논문은 정확히 짚는다 — **reward score와 training loss는 계속 좋아 보이는데, 실제 사람·GPT-4 평가는 정반대로 나빠지는 현상**이다. 저자들은 이를 pattern collapse라 부른다. 이 글은 그 붕괴 메커니즘을 진단 지표로 확인하고, PPO-max를 구성하는 트릭을 하나씩 뜯어보고, 왜 그 트릭이 필요한지 토이 예제로 손으로 계산해본 뒤, 보조 논문 [The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031)(Huang et al., Hugging Face, COLM 2024)가 지적하는 "논문에 안 적힌 디테일"까지 확인한다.

# Background

## RLHF 3단계와 PPO의 자리

RLHF는 SFT → Reward Modeling → PPO 세 단계로 진행된다. 이 시리즈에서 [InstructGPT](/blog/2026/instructgpt/), [Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)가 각각 첫 두 단계를 다뤘다. 이 글이 보는 세 번째 단계에서 정책 $$\pi_\theta$$ 는 reward model $$r(x,y)$$ 를 최대화하도록 업데이트된다. [#16](/blog/2026/ppo/)에서 정리한 PPO의 핵심 목적함수를 다시 적으면 다음과 같다.

$$\mathcal{L}_{\text{ppo-clip}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\hat{A}_t,\ \text{clip}\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$

- $$a_t, s_t$$: 언어모델 관점에서 $$s_t$$ 는 프롬프트+지금까지 생성된 토큰, $$a_t$$ 는 다음 토큰이다.
- $$\hat{A}_t$$: advantage 추정치. GAE로 계산한다.
- $$\epsilon$$: clipping 폭. 정책이 한 스텝에 얼마나 움직일 수 있는지의 상한.

critic(value network) $$V_\phi$$ 는 별도로 MSE 손실로 학습된다.

$$\mathcal{L}_{\text{critic}}(\phi) = \hat{\mathbb{E}}_t\left[\lVert V_\phi(s_t) - \hat{R}_t \rVert^2\right]$$

- $$V_\phi(s_t)$$: critic이 예측한 상태 $$s_t$$ 의 가치.
- $$\hat{R}_t = \sum_{l=0}^{\infty}\gamma^l r_{t+l}$$: 실제 할인 리턴.

일상 비유로 말하면 이렇다. 학기 중 매 수업(토큰)마다 배운 게 얼마나 성적에 기여했는지는 아무도 안 알려주고, 학기말에 딱 한 번 최종 성적표(sequence reward)만 받는다. critic은 "이번 학기 최종 성적이 얼마나 나올지"를 매 수업 시점마다 미리 추정해보는 역할이고, advantage는 "이번 수업이 그 예상치보다 얼마나 더/덜 도움이 됐는가"다. 이 추정이 부정확하면 잘못된 수업에 칭찬 또는 벌점을 몰아주게 된다 — 이게 RLHF에서 critic 안정화가 그렇게 중요한 이유다.

또 다른 비유는 reward model 자체에 대한 것이다. RM은 원래 교수님을 대신해 채점하는 **대리 채점자**다. 대부분의 경우 교수님과 비슷하게 채점하지만, 특정 유형의 답안(길게 쓰기, 자신 있게 말하기)에는 맹점이 있다. 학생이 그 맹점만 노리고 답안을 쓰면 대리 채점자에게는 만점을 받아도 실제 실력은 늘지 않는다 — 이게 reward hacking이다.

# Method

## 어떻게 터지는가: pattern collapse 진단

논문은 vanilla PPO를 그대로 돌렸을 때의 궤적을 4가지 지표로 보여준다.

<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig4-vanilla-ppo-reward-collapse.png" width="70%"></p>
<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig4-vanilla-ppo-metrics-collapse.png" width="70%"></p>

위 그림(논문 Figure 4)의 관찰은 다음과 같다.

| 지표            | vanilla PPO에서의 변화 | 의미                                                       |
| --------------- | ---------------------- | ---------------------------------------------------------- |
| reward score    | 꾸준히 상승            | RM 기준으로는 "학습이 잘 되고 있다"처럼 보임               |
| training loss   | 안정적으로 수렴        | 최적화 자체는 문제없이 진행됨                              |
| 사람·GPT-4 평가 | reward와 무관하게 하락 | reward 상승이 실제 품질 향상과 무관해짐                    |
| 응답 길이       | 균일하게 증가          | 특정 패턴으로의 쏠림                                       |
| perplexity      | 하락                   | 생성이 정형화·반복화됨(모델이 "확신"하는 좁은 패턴만 출력) |
| KL divergence   | 급격히 증가            | 정책이 SFT 기준에서 계속 멀어짐                            |

핵심 문장은 이거다 — **"reward score와 training loss는 PPO가 올바르게 최적화되고 있는지를 알려주지 못한다."** loss만 보고 있으면 학습이 잘 되는 줄 착각하기 딱 좋다. 그래서 실전에서는 reward 곡선 하나만 보지 말고 반드시 perplexity·KL·응답 길이를 같이 모니터링해야 한다.

## PPO-max 구성 요소 하나씩

<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig5-ppomax-components.png" width="70%"></p>

논문은 트릭을 크게 세 그룹 — **점수 재파라미터화(reward/advantage 정규화)**, **정책 제약(KL·엔트로피)**, **초기화 전략** — 으로 나눠 하나씩 검증한다. 전체를 표로 먼저 정리한다.

| 트릭                           | 무엇을 막는가                                   | 수식·값                               | PPO-max 채택                            |
| ------------------------------ | ----------------------------------------------- | ------------------------------------- | --------------------------------------- |
| Reward Scaling                 | 보상의 절대 스케일 변동                         | $$r_n(x,y)/\sigma(r(x,y))$$           | 미채택 (효과 없음 확인)                 |
| Reward Normalization + Clip    | outlier 보상 하나가 배치 gradient를 독점하는 것 | $$\delta=0.3$$, 아래 식 18            | 채택 (단, "임시" 안정)                  |
| Advantage Normalization + Clip | 위와 유사, 미니배치 단위                        | 배치 평균·표준편차로 정규화           | 선택적 (reward clip과 동시 사용 비권장) |
| Token-level KL Penalty         | 정책이 SFT 기준에서 이탈하는 것 자체            | $$\eta=0.05$$, 아래 식 19             | 채택 — **핵심 트릭**                    |
| Value Function Loss Clip       | critic 업데이트가 한 번에 과도하게 튀는 것      | $$\lambda_{vf}=0.2$$                  | 채택                                    |
| Global Gradient Clip           | 배치 노이즈로 인한 급격한 파라미터 변화         | 기본 활성화                           | 채택 (효과는 제한적)                    |
| Critic Pretraining             | 학습 초반 advantage 추정 오차로 인한 흔들림     | RM으로 초기화 후 critic만 별도 선학습 | 채택 (LR warmup 대체)                   |
| Policy SFT Initialization      | 애초에 언어능력 없는 정책이 PPO에 들어가는 것   | SFT 필수                              | 필수                                    |
| PPO-ptx                        | RLHF로 인한 NLU 능력 저하(alignment tax)        | $$\lambda_{\text{ptx}}$$, 아래 식 17  | 선택적                                  |
| Entropy Bonus                  | (탐색 촉진 목적)                                | —                                     | **비권장** — 임계값 10%만 바뀌어도 붕괴 |

### 점수 재파라미터화: reward를 있는 그대로 쓰면 안 되는 이유

Reward Scaling(과거 관측치들의 rolling 표준편차로만 나누는 방식)은 고전 RL(Atari 등)에서는 유효했지만, 이 논문은 **LLM RLHF에서는 효과가 없다**고 명시한다 — scaling을 켜고 끄고와 무관하게 학습 궤적이 거의 같았다. 대신 효과가 있었던 건 평균까지 빼는 정규화다.

$$\tilde{r}(x,y) = \text{clip}\left(\frac{r_n(x,y) - \overline{r(x,y)}}{\sigma(r(x,y))}, -\delta, \delta\right), \quad \delta = 0.3$$

- $$r_n(x,y)$$: 현재 배치의 원본 RM 점수.
- $$\overline{r(x,y)}$$, $$\sigma(r(x,y))$$: 학습 히스토리 전체에서 누적한 실행(running) 평균·표준편차. 지금 배치 4개만 보고 계산하는 게 아니라, 지금까지 본 모든 보상의 이동 통계다.
- $$\delta$$: 정규화된 값을 다시 자르는 clip 폭. 0.3으로 설정.

Advantage Normalization + Clip은 GAE로 advantage를 구한 뒤 **미니배치 단위**로 같은 방식(평균 빼고 표준편차로 나누기)을 적용한다. 논문은 이 둘을 동시에, 그리고 value clip까지 섞어 쓰면 오히려 최적화 방향이 충돌한다는 것도 확인했다 — 그래서 "점수 재파라미터화 방법을 섞지 말라"고 권고한다.

## 토이 예제: 정규화 전후 advantage가 얼마나 달라지는가

가상의 PPO 배치에 응답 4개가 있다고 하자. 그중 응답 3은 RM의 맹점을 파고든 아첨성 반복 패턴("정말 훌륭한 질문입니다!"를 여러 번 되풀이하는 식)이다.

**1단계 — 원본 RM 점수와 critic 예측.**

| 응답          | 원본 reward $$r$$ | critic 예측 $$V$$ | advantage $$r-V$$ |
| ------------- | ----------------- | ----------------- | ----------------- |
| 1 (정상)      | 2.0               | 2.1               | -0.1              |
| 2 (정상)      | 1.8               | 1.9               | -0.1              |
| 3 (해킹 패턴) | 15.0              | 2.0               | **+13.0**         |
| 4 (정상)      | 2.4               | 2.3               | +0.1              |

정규화 없이 그대로 쓰면 응답 3의 advantage(13.0)는 다른 응답(약 ±0.1)보다 **130배** 크다. policy gradient 항은 advantage에 비례하므로, 이 배치의 gradient는 사실상 응답 3 하나가 결정한다. pattern collapse는 이렇게 시작한다 — 어쩌다 RM 맹점을 찌른 응답 하나가 배치를 통째로 장악한다.

**2단계 — reward normalization + clip 적용.** 학습 히스토리 전체의 실행 평균 $$\overline{r}=3.0$$, 표준편차 $$\sigma=4.0$$ 이라 하자. $$\delta=0.3$$ 으로 자르면,

$$\tilde{r}_1 = \text{clip}\left(\frac{2.0-3.0}{4.0}, -0.3, 0.3\right) = -0.25, \quad \tilde{r}_2 = -0.30, \quad \tilde{r}_3 = \text{clip}(3.0, -0.3, 0.3) = 0.30, \quad \tilde{r}_4 = -0.15$$

응답 3의 원래 z-score는 3.0인데 clip 때문에 0.3으로 눌린다. 이제 네 응답의 값 범위는 [-0.30, 0.30] — 2배 스프레드로 줄었다(원래는 130배 스프레드). 응답 3은 여전히 1등이지만 더는 배치 gradient를 독점하지 못한다. 다만 이 정규화는 RM이 매기는 "상대적 등수"만 조정할 뿐, RM이 계속 이 패턴을 좋아하는 한 매 배치 조금씩 같은 방향으로 정책을 밀어낸다 — 그래서 논문은 이 트릭만으로는 "임시로만" 안정적이라 부른다.

**3단계 — KL 페널티까지 더하기.** 응답 1·2·4는 SFT 응답과 비슷해 토큰 평균 KL이 0.1 수준이지만, 응답 3은 SFT라면 절대 내지 않을 반복 패턴이라 KL이 6.0까지 누적됐다고 하자. $$\eta=0.05$$ 를 적용하면,

$$r_{\text{total,3}} = \tilde{r}_3 - \eta \cdot \text{KL}_3 = 0.30 - 0.05 \times 6.0 = 0.00$$

나머지 세 응답은 $$-0.05 \times 0.1 \approx -0.005$$ 만큼만 깎여 거의 그대로다. 결과적으로 응답 3의 우위(다른 응답 평균 대비 +0.5 수준이던 것)가 KL 페널티 한 번에 사실상 사라진다. reward 정규화가 "RM 점수의 스케일"만 건드리는 데 비해, KL 페널티는 "정책이 실제로 얼마나 움직였는가"를 직접 벌점화한다 — 그리고 reward hacking은 정의상 거의 항상 기준 정책에서 멀어져야 가능하므로, KL 페널티는 hacking이 심해질수록 자동으로 더 강하게 반작용한다. 이게 바로 논문이 KL 페널티를 "장기적으로 안정적"이라 부르고, reward/advantage clip을 "일시적으로만 안정적"이라 구분하는 이유다.

## 정책 제약: KL 페널티가 진짜 핵심이다

<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig7-kl-vs-other-constraints.png" width="70%"></p>

Token-level KL penalty는 시퀀스가 아니라 **토큰 단위**로 계산된다.

$$r_{\text{total}}(x,y_i) = r(x,y_i) - \eta \cdot \text{KL}\left(\pi^{\text{RL}}_\theta(y_i \mid x), \pi^{\text{SFT}}(y_i \mid x)\right), \quad \eta = 0.05$$

- $$\pi^{\text{RL}}_\theta(y_i \mid x)$$: 현재 정책이 $$i$$번째 응답 토큰에 부여하는 확률.
- $$\pi^{\text{SFT}}(y_i \mid x)$$: 고정된 SFT(reference) 모델의 확률.
- $$\eta$$: KL 페널티 계수.

이 논문에서 가장 인상적인 발견은 계수 값 그 자체다. Anthropic의 [HH-RLHF](/blog/2026/anthropic-hh-rlhf/) 계열 연구는 $$\eta=0.001$$ 을 썼고 "유의미한 효과를 못 찾았다"고 보고했지만, 이 논문은 $$\eta=0.05$$ — **50배 더 강한 계수** — 를 썼을 때 비로소 장기 학습이 안정된다는 것을 확인했다. Figure 7(위 그림)이 보여주는 비교가 핵심이다: reward clip, advantage clip 등 다른 제약은 모두 초반에는 수렴하는 것처럼 보이지만, KL 페널티(또는 엔트로피 페널티)만이 학습을 길게 끌고 가도 붕괴하지 않는다.

### KL 계수를 잘못 잡으면 생기는 두 가지 실패 모드

<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig15-kl-coefficient-scaling.png" width="70%"></p>

논문 부록(B.2)은 KL 계수를 0.05→0.1→0.2로 키워가며 비교한다. 그 결과를 계수가 너무 작을 때/너무 클 때로 정리하면 다음과 같다.

| 계수 크기                   | 증상                                                              | 원인                                                     |
| --------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| 너무 작음 (예: 0.001)       | reward hacking, pattern collapse, 응답 길이 급증, perplexity 급락 | 정책이 RM 맹점을 향해 자유롭게 이동해도 벌점이 거의 없음 |
| 적정 (0.05, 이 논문의 선택) | reward는 완만히 상승, KL은 거의 0에 가깝게 유지                   | 응답 품질은 개선되지만 언어 자체는 거의 안 바뀜          |
| 너무 큼 (0.1~0.2)           | reward 상승 폭 자체가 줄어듦, 정책이 SFT에서 거의 못 벗어남       | 페널티가 유의미한 정책 개선까지 억제                     |

계수를 정하는 두 갈래 접근도 정리해둘 만하다.

| 방식                   | 동작                                                                                                              | 장단점                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 고정(fixed) KL 계수    | $$\eta$$ 를 학습 내내 상수로 고정                                                                                 | 이 논문의 선택. 튜닝은 필요하지만 동작이 예측 가능                      |
| Adaptive KL controller | 목표 KL을 정해두고, 실측 KL이 목표보다 크면 $$\eta$$ 를 키우고 작으면 줄이는 비례 제어(Ziegler et al., 2019 방식) | 목표 KL만 정하면 되어 편하지만, 제어가 흔들리면 계수 자체가 진동할 위험 |

이 논문은 명시적으로 **고정 계수**를 택했고, $$\eta=0.05$$ 라는 값 자체를 실험으로 찾아낸 것이 핵심 기여 중 하나라고 강조한다. 일상 비유로 치면, KL 페널티는 지도 앱이 "이 경로는 원래 다니던 길에서 너무 멀리 벗어난다"고 통행료를 매기는 것과 같다. 통행료가 너무 싸면(계수가 작으면) 지름길이랍시고 위험 지역으로 새는 걸 막지 못하고, 통행료가 너무 비싸면(계수가 크면) 원래 길에서 한 발짝도 못 벗어나 개선 자체가 안 일어난다.

## Critic·Policy 초기화, 그리고 alignment tax

critic 초기화 방식도 안정성에 영향을 준다. 논문은 critic을 reward model 가중치로 초기화한 뒤, **PPO를 본격 시작하기 전에 critic만 따로 선학습**(value loss가 거의 0에 수렴할 때까지)시키는 전략을 권장한다 — 이 방식이 learning rate warmup보다 초반 흔들림을 더 효과적으로 줄인다고 보고한다. 반대로 정책 쪽은 선택의 여지가 없다: SFT 없이 사전학습 모델을 곧바로 PPO에 넣으면 "언어 모델링 능력이 심각하게 붕괴"한다.

RLHF로 인한 일반 언어능력 저하(alignment tax)를 줄이기 위해, 논문은 InstructGPT와 같은 방식으로 사전학습 loss를 policy loss에 섞는 PPO-ptx를 함께 제안한다.

$$\mathcal{L}_{\text{ppo-ptx}}(\theta) = \mathcal{L}_{\text{ppo-clip}}(\theta) + \lambda_{\text{ptx}} \cdot \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}}\left[\log(\pi^{\text{RL}}_\theta(x))\right]$$

- $$\mathcal{D}_{\text{pretrain}}$$: 원 사전학습 데이터 분포.
- $$\lambda_{\text{ptx}}$$: 이 손실을 얼마나 섞을지 정하는 계수.

C-Eval(중국어 지식 벤치마크)에서 PPO 단독은 SFT 대비 점수가 떨어졌지만, PPO-ptx를 섞은 쪽은 그 저하를 완화했다.

# Experiments

## PPO-max 실전 성능

<p align="center"><img src="/assets/post/image/secrets-rlhf-ppo/fig9-ppomax-stable-training.png" width="70%"></p>

위 조합을 모두 적용한 PPO-max는 10K 스텝 동안 붕괴 없이 학습을 이어갔다(위 그림, 논문 Figure 9). 사람 평가와 GPT-4 평가 결과는 다음과 같다.

| 평가 항목                                 | RLHF(PPO-max) | SFT |
| ----------------------------------------- | ------------- | --- |
| 영어 Harmless (사람 평가)                 | 62%           | 5%  |
| 영어 Helpful (사람 평가)                  | 44%           | 30% |
| ChatGPT(gpt-3.5-turbo) 대비 패배율 — 영어 | 24%           | 45% |
| ChatGPT 대비 패배율 — 중국어              | 29%           | 37% |

Harmless 항목에서 격차(62% vs 5%)가 특히 크다는 점, 그리고 ChatGPT 상대 패배율이 절반 가까이 줄었다는 점이 PPO-max의 실질적 효과를 보여준다. 다만 저자들은 여전히 ChatGPT를 완전히 앞서지는 못했다고 인정한다.

## 구현 디테일: 논문에 안 적힌 것들이 재현성을 가른다

PPO-max가 어떤 트릭을 쓰라고 알려줘도, 그 트릭을 코드 레벨에서 정확히 어떻게 구현하느냐는 또 다른 문제다. [The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031)(Huang et al., Hugging Face, COLM 2024)는 OpenAI의 2019년 원조 RLHF 코드베이스(TL;DR 요약 태스크)를 현대 스택으로 재현하면서 이 문제를 정면으로 다룬다. 저자들은 이렇게 말한다.

> "RLHF 파이프라인을 재현하는 게 어려운 이유는 세 가지다: 1) 학습 안정성에 큰 영향을 주는 미묘한 구현 디테일이 많고, 2) instruction-following 태스크는 평가 자체가 어려우며, 3) 학습에 오래 걸려 반복 실험이 비싸다."

논문이 나열한 20개 이상의 디테일 중 재현성에 특히 크게 영향을 준 항목은 다음과 같다.

| 디테일                     | 내용                                                            | 왜 중요한가                                                                                                                                                   |
| -------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dropout 비활성화           | PPO 학습 중 dropout을 끈다                                      | dropout을 켠 채로는 로그확률이 재현 불가능해져 KL 페널티 계산이 흔들리고, 첫 epoch의 PPO 비율이 1이 아니게 되어 clipping 로직 자체가 의도대로 작동하지 않는다 |
| EOS trick                  | 응답이 최대 길이 안에 EOS로 끝나지 않으면 보상을 -1로 고정 부여 | EOS 없는 응답은 보상이 정의되지 않는 문제를 명시적으로 처리                                                                                                   |
| reward 추출 위치           | RM은 EOS 토큰 위치에서만 유효한 reward를 낸다                   | 다른 위치의 reward logit은 대부분 음수이고 무효값                                                                                                             |
| reward/advantage whitening | reward·advantage를 정규화(whitening)                            | Secrets of RLHF I의 정규화 트릭과 같은 목적 — outlier가 gradient를 독점하는 것 방지                                                                           |
| KL 계수                    | $$\beta = 0.05$$                                                | 이 논문 Secrets of RLHF I이 찾은 값과 정확히 일치 — 서로 독립적으로 같은 수치에 수렴했다는 점이 신뢰도를 더한다                                               |
| value clip                 | $$0.2$$, $$c_1=0.1$$                                            | PPO-max의 $$\lambda_{vf}=0.2$$ 와 동일한 오더                                                                                                                 |

가장 인상적인 재현 사례는 스케일 실험이다. 1B 모델에서는 RLHF reward가 계속 올라가는데도 사람 선호율이 20% 밑으로 떨어졌고, 생성 결과는 "이어붙인 의미 없는 문자열"이었다고 보고한다 — [#10 Overoptimization 글](/blog/2026/reward-model-overoptimization/)이 스케일링 법칙으로 정량화한 reward-proxy 괴리를, 이 논문은 실제 학습 로그로 그대로 재현해 보여준 셈이다. 2.8B·6.9B로 모델을 키우자 이 문제는 완화됐고, OpenAI가 공개한 1.3B 체크포인트보다 더 나은 품질을 냈다.

## 실무 체크리스트: 학습이 터지면 무엇부터 의심할까

| 증상                               | 먼저 볼 지표                    | 의심할 트릭                                                |
| ---------------------------------- | ------------------------------- | ---------------------------------------------------------- |
| reward는 오르는데 사람 평가가 나쁨 | perplexity 급락, 응답 길이 급증 | KL 계수가 너무 작음, reward normalization 미적용           |
| reward 자체가 거의 안 오름         | KL이 0 근처에 고정              | KL 계수가 너무 큼                                          |
| 학습 초반 몇백 스텝이 유난히 요동  | critic loss 곡선                | critic을 RM으로만 초기화하고 선학습을 안 함                |
| 특정 배치 이후 정책이 급변         | 개별 샘플의 advantage 분포      | reward/advantage clip 미적용, outlier 하나가 gradient 독점 |
| 첫 epoch부터 PPO ratio가 1이 아님  | dropout 설정                    | dropout이 꺼져 있는지 확인                                 |
| RLHF 이후 일반 지식 벤치마크 하락  | C-Eval류 NLU 점수               | PPO-ptx(pretraining loss 혼합) 미적용                      |

# Conclusion

핵심은 한 줄로 정리된다 — **PPO를 LLM에 그대로 붙이면 reward score와 training loss가 멀쩡해 보여도 정책은 조용히 붕괴할 수 있고, 이를 막는 것은 하나의 트릭이 아니라 reward 정규화·KL 페널티·critic 초기화가 함께 맞물린 시스템(PPO-max)이다.** 그리고 그 시스템 중에서도 token-level KL 페널티가 유일하게 "장기적으로" 안정성을 보장하는 트릭이라는 것이 이 논문의 실험적 결론이다.

정리하면,

1. **불안정의 구조적 원인**: reward model은 proxy이고, 보상은 시퀀스 끝에서만 sparse하게 오며, 정책이 이동할수록 RM의 신뢰 구간을 벗어난다.
2. **PPO-max**: reward normalization+clip($$\delta=0.3$$), token-level KL penalty($$\eta=0.05$$), value function clip($$\lambda_{vf}=0.2$$), critic pretraining, PPO-ptx까지 묶은 조합. 이 중 KL 페널티만이 "일시적"이 아니라 "장기적" 안정성을 준다.
3. **구현 디테일이 재현성을 가른다**: dropout, EOS 처리, reward whitening 같은 코드 레벨 디테일이 논문에 적힌 하이퍼파라미터 값 못지않게 중요하다.

다만 이 레시피 자체가 다음 문제를 예고한다. PPO-max를 온전히 굴리려면 정책 모델과 별개로 critic(value network)을 통째로 하나 더 학습·서빙해야 하고, 그 위에 KL 계수·clip 폭·warmup 전략까지 십여 개의 하이퍼파라미터를 맞춰야 한다. 이 복잡성 자체가 다음 글들의 동기가 된다 — [#18 GRPO/DeepSeekMath](/blog/2026/grpo-deepseekmath/)는 "critic을 아예 버리고 그룹 상대 보상으로 advantage를 대체하자"고 나오고, [#19 RLOO](/blog/2026/rloo-back-to-basics/)는 "REINFORCE로도 충분하지 않냐"고 되묻는다. 두 글 모두 이 글이 공들여 쌓은 안정화 트릭들을 정면으로 걷어내는 시도라는 점에서, 이 글은 그 논쟁의 출발점이다. reward model 자체의 한계를 더 깊이 파고드는 이야기는 [#5 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/)로 이어진다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열일곱 번째 글이다.

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
  <li><a href="/blog/2026/warm-weight-averaged-reward/">WARM (2024)</a> — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="14">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
</ol>

**5부. reward를 정책으로**

<ol start="16">
  <li><a href="/blog/2026/ppo/">PPO (2017)</a> — clipped surrogate objective</li>
  <li><strong>(현재 글)</strong> Secrets of RLHF I (2023) — PPO 학습 안정화 트릭</li>
  <li><a href="/blog/2026/grpo-deepseekmath/">GRPO / DeepSeekMath (2024)</a> — value network를 버리다</li>
  <li><a href="/blog/2026/rloo-back-to-basics/">RLOO (2024)</a> — REINFORCE로 충분한가</li>
  <li><a href="/blog/2026/dpo/">DPO (2023)</a> — reward를 없애면 어떻게 되는가</li>
  <li><a href="/blog/2026/simpo/">SimPO (2024)</a> — reference-free + 길이 정규화</li>
  <li><a href="/blog/2026/kto/">KTO (2024)</a> — 선호 쌍 없이 이진 신호만으로</li>
  <li><a href="/blog/2026/gspo/">GSPO (2025)</a> — importance ratio를 시퀀스 단위로</li>
  <li><a href="/blog/2026/dapo/">DAPO (2025)</a> — 신호 없는 프롬프트를 버린다</li>
</ol>

**6부. Process & Verifiable Reward**

<ol start="25">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="28">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="33">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="38">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 39편으로 구성된다.

# 참고 문헌

- Zheng et al., 2023. [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964).
- [OpenLMLab/MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF) — 공식 코드 저장소.
- Huang et al., 2024. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031).
- Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Ziegler et al., 2019. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) — adaptive KL controller의 원형.
