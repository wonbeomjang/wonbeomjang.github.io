---
layout: post
title: "InstructGPT: RLHF 3단계 레시피의 표준을 세우다"
date: 2026-08-11 09:02:00 +0900
description: "RLHF Reward 설계 시리즈 #2 — SFT → Reward Model → PPO, 오늘날 모든 RLHF 파이프라인의 원형 (Ouyang et al., OpenAI, NeurIPS 2022)"
categories: [paper]
tags: [rlhf, reward-model, llm, alignment, ppo, paper]
giscus_comments: true
related_posts: true
---

> [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., OpenAI, NeurIPS 2022)

# Introduction

이 시리즈 [1부](/blog/2026/deep-rl-human-preferences/)에서 다룬 Christiano의 논문은 "사람이 두 궤적을 비교해주기만 하면, 그 선호로부터 보상 함수를 학습할 수 있다"는 원형을 로봇 시뮬레이션과 Atari 게임에서 증명했다. 이번 글에서 다룰 InstructGPT는 그 원형을 언어모델이라는 완전히 다른 무대로 옮긴 논문이다. 그리고 그 무대에서 하나의 숫자가 업계 전체의 방향을 바꿔놓았다.

**1.3B짜리 InstructGPT가 175B짜리 GPT-3보다 사람에게 더 선호받는다.** 파라미터 수로는 100배가 넘게 차이 나는 두 모델인데, 사람 평가자는 작은 모델의 출력을 더 좋아했다. 이 결과가 말해주는 것은 단순하다 — **"똑똑한 모델을 만드는 것"과 "사람이 원하는 대로 행동하는 모델을 만드는 것"은 서로 다른 문제**라는 것이다. GPT-3는 인터넷 텍스트에서 다음 토큰을 맞추도록 학습됐다. 이 목적함수는 "그럴듯한 다음 단어"를 예측하는 데는 최적이지만, "사용자가 원하는 걸 도와줘라"라는 목적과는 애초에 다른 것을 최적화한다. 모델을 아무리 키워도 이 간극(gap)은 저절로 메워지지 않는다. 잘못된 목적함수는 스케일로 고쳐지지 않는다 — 목적함수 자체를 바꿔야 한다.

InstructGPT가 제안하는 해법은 **SFT → Reward Model → PPO**로 이어지는 3단계 파이프라인이다. 사람이 직접 시범을 보여주고(SFT), 사람이 여러 출력 중 어느 것이 더 나은지 순위를 매기고(RM), 그 순위로 학습한 보상 모델을 정책이 최대화하도록 강화학습으로 미세조정한다(PPO). 이 세 단계는 이후 ChatGPT, Claude, Llama 등 거의 모든 프로덕션 LLM의 정렬 파이프라인에서 표준 골격으로 재사용됐다.

이 글은 논문 전체를 요약하기보다, 이 시리즈의 관점인 **"reward를 어떻게 설계했는가"**에 집중한다. 구체적으로 다음 질문에 답한다.

1. 왜 175B가 아니라 6B reward model을 썼는가, 그리고 왜 그것으로 충분했는가.
2. $$K$$개 응답을 뽑아 $$\binom{K}{2}$$개 쌍을 만드는데, 이걸 왜 "하나의 배치"로 묶어서 학습했는가.
3. KL penalty는 정확히 무엇을 막기 위한 장치이고, PPO-ptx는 어떤 대가를 되돌리기 위한 트릭인가.
4. 이 모든 걸 가능하게 한 40명의 라벨러는 누구이고, 이들의 선호가 곧 "정렬"의 정의가 되는 것은 어떤 함의를 갖는가.

# Background

## GPT-3의 목적함수와 사용자 의도 사이의 간극

GPT-3(Brown et al., 2020)는 인터넷 코퍼스에 대해 다음 토큰을 예측하도록 학습된 175B 파라미터 언어모델이다. 이 목적함수 — "다음 토큰의 로그 확률을 최대화하라" — 는 few-shot 프롬프팅만으로 놀라운 범용성을 보였지만, 명시적으로 "사용자 지시를 따르라"고 훈련받은 적은 없다. 그 결과 GPT-3는 사실이 아닌 내용을 지어내거나(hallucination), 유해하거나 편향된 텍스트를 생성하거나, 사용자가 실제로 요청한 것과는 다른 걸 출력하는 경우가 잦았다. 논문은 이 상태를 **misalignment**라고 부른다 — 모델이 무능해서가 아니라, 애초에 최적화한 목표가 사용자의 의도와 다르기 때문에 생기는 어긋남이다.

비유하자면 이렇다. 아주 박식한 요리사에게 "맛있게 만들어라"라는 목표만 주고 훈련시키면, 요리사는 자기 기준의 "맛있음"을 최적화한다. 그런데 손님마다 원하는 맛은 다르고, 손님이 알레르기가 있거나 맵기를 못 견딜 수도 있다. 요리사가 아무리 실력이 늘어도(모델을 키워도), "이 손님이 원하는 것"을 배우지 않는 한 손님 만족도는 오르지 않는다. InstructGPT는 이 요리사에게 "손님이 실제로 무엇을 원하는지"를 사람 피드백으로 가르치는 과정이다.

## Alignment의 정의 — Helpful, Honest, Harmless

논문은 alignment를 [Askell et al. (2021)](https://arxiv.org/abs/2112.00861)의 프레임을 따라 세 축으로 정의한다.

| 축       | 의미                                                     |
| -------- | -------------------------------------------------------- |
| Helpful  | 지시를 따르고, 사용자가 실제로 의도한 바를 추론해 돕는다 |
| Honest   | 사실이 아닌 것을 지어내지 않는다 (truthful)              |
| Harmless | 유해하거나 공격적인 콘텐츠를 생성하지 않는다             |

이 세 기준이 서로 충돌할 때 — 예컨대 사용자가 유해한 답을 요구할 때 — 논문은 **학습 중에는 helpfulness를 우선**하고, **최종 평가에서는 truthfulness와 harmlessness를 우선**하도록 라벨러에게 지시했다고 밝힌다. 이 우선순위 결정 자체가 이미 "누구의 기준으로 정렬할 것인가"라는 질문을 내포한다 (뒤에서 다시 다룬다).

## 방법론적 뿌리: Christiano 2017과 Stiennon 2020

InstructGPT는 방법론적으로 두 논문을 직접 계승한다.

- **Christiano et al. (2017)**: 사람이 두 궤적 중 선호를 표시하면, 그 비교로부터 보상 함수를 학습하고 그 보상으로 RL 정책을 학습하는 RLHF의 원형.
- **Stiennon et al. (2020)**: 위 방법을 텍스트 요약(summarization)이라는 좁은 태스크에 처음 적용해, 사람이 선호하는 요약을 생성하도록 GPT 계열 모델을 RLHF로 학습.

InstructGPT는 Stiennon의 파이프라인을 그대로 가져오되, 태스크를 "요약"이라는 단일 과제에서 **"OpenAI API에 실제로 들어오는 모든 종류의 자연어 지시"**로 확장했다는 점이 핵심 기여다. 아래 표가 실제 사용된 프롬프트의 분포다.

| Use-case       | 비중  |
| -------------- | ----- |
| Generation     | 45.6% |
| Open QA        | 12.4% |
| Brainstorming  | 11.2% |
| Chat           | 8.4%  |
| Rewrite        | 6.6%  |
| Summarization  | 4.2%  |
| Classification | 3.5%  |
| Other          | 3.5%  |
| Closed QA      | 2.6%  |
| Extract        | 1.9%  |

거의 절반이 "무언가를 새로 생성하라"는 open-ended 요청이다. 요약처럼 입력-출력이 명확한 태스크와 달리, 이런 프롬프트는 "좋은 응답"의 기준 자체가 사람마다 다르다. 그래서 InstructGPT의 RM은 Stiennon의 RM보다 훨씬 다양하고 모호한 선호를 학습해야 했다.

# Method

<p align="center"><img src="/assets/post/image/instructgpt/fig2-pipeline-3steps.png" width="95%"></p>

파이프라인은 위 그림처럼 세 단계로 나뉜다. 1단계는 "사람이 정답을 보여준다", 2단계는 "사람이 순위를 매긴다", 3단계는 "그 순위로 학습한 보상을 정책이 최대화한다". 각 단계가 정확히 무엇을 입력받아 무엇을 출력하는지, 그리고 왜 그렇게 설계했는지를 따라가 보자.

## Step 1 — Supervised Fine-Tuning (SFT)

첫 단계는 라벨러가 프롬프트에 대해 **바람직한 응답을 직접 작성**하는 것이다. 이 시범(demonstration) 데이터로 GPT-3를 지도학습으로 미세조정한다. 프롬프트는 두 출처에서 온다 — 라벨러가 직접 쓴 프롬프트, 그리고 (PII를 필터링한) 초기 InstructGPT 모델에 실제로 들어온 API 프롬프트다.

세 단계 데이터셋의 규모는 다음과 같다.

| 단계 | train (labeler) | train (customer) | valid (labeler) | valid (customer) |
| ---- | --------------- | ---------------- | --------------- | ---------------- |
| SFT  | 11,295          | 1,430            | 1,550           | 103              |
| RM   | 6,623           | 26,584           | 3,488           | 14,399           |
| PPO  | —               | 31,144           | —               | 16,185           |

표의 두 열은 **프롬프트가 어디서 왔는지**를 나눈 것이다.

| 출처         | 정체                                                                     |
| ------------ | ------------------------------------------------------------------------ |
| **labeler**  | OpenAI가 고용한 계약직 작업자 약 40명이 **직접 지어낸** 프롬프트         |
| **customer** | 실제 사용자가 Playground에서 초기 InstructGPT에 **실제로 보낸** 프롬프트 |

여기서 닭과 달걀 문제가 있다. 지시를 잘 따르는 모델을 만들려면 "사람들이 실제로 어떤 지시를 내리는가" 데이터가 필요한데, **그런 모델이 아직 없으니 그런 트래픽도 없다.** 그래서 부트스트랩을 한다 — 라벨러가 프롬프트를 직접 지어내 씨앗 데이터를 만들고, 그걸로 초기 InstructGPT를 학습시켜 Playground에 배포하고, 그때부터 쌓이는 진짜 프롬프트를 이후 단계에 쓴다.

라벨러가 프롬프트를 지어낼 때도 세 종류로 나눴다. 임의의 태스크를 최대한 다양하게 쓰는 **Plain**, 지시 하나에 여러 query/response 예시를 붙이는 **Few-shot**, 그리고 API 대기자 명단에 적힌 실제 활용 사례를 보고 그에 맞춰 쓰는 **User-based**다. 마지막 것이 영리한데, 트래픽은 아직 없어도 **"사람들이 뭘 하고 싶어 하는지"에 대한 정보는 이미 있었던** 셈이다.

이제 표의 비중이 뒤집히는 이유가 보인다. SFT는 라벨러가 압도적이고(11,295 대 1,430), RM·PPO로 갈수록 customer가 커지다가 PPO에서는 아예 전부 customer다. **단계마다 사람에게 시키는 일의 난이도가 다르기 때문**이다.

| 단계 | 사람이 하는 일                        | 비용                                  |
| ---- | ------------------------------------- | ------------------------------------- |
| SFT  | 모범 답안을 **직접 창작**             | 비싸다. 게다가 이때는 트래픽도 없었다 |
| RM   | 모델이 뽑은 응답을 **줄 세우기만**    | 훨씬 싸다                             |
| PPO  | **아무것도 안 한다** (RM이 대신 채점) | 프롬프트만 있으면 된다                |

즉 **창작 → 선택 → 프롬프트만**으로 사람의 부담이 줄어드는 만큼, 데이터 출처가 라벨러에서 실제 사용자로 넘어간다.

customer 프롬프트를 그대로 쓰지는 않는다. PII를 필터링하고, 긴 공통 접두사를 기준으로 중복을 제거하고, 사용자 ID당 최대 200개로 제한하고, train/valid/test를 **사용자 ID 기준으로** 나눈다. 마지막이 중요한데, 프롬프트 단위로 무작위 분할하면 같은 사용자의 비슷한 프롬프트가 양쪽에 걸쳐 성능이 부풀려진다. 사용자 단위로 잘라야 "처음 보는 사용자"에 대한 일반화를 재는 것이 된다.

한 가지 짚어둘 점은 customer 프롬프트가 **이미 배포된 초기 InstructGPT에 들어온 것**이라는 사실이다. 사용자는 그 모델이 잘하는 일에 맞춰 질문하게 되므로 데이터가 모델의 능력에 은근히 끌려간다. 이 시리즈에서 계속 나올 **distribution shift**의 초기 형태이기도 하다.

SFT 모델은 GPT-3를 16 epoch, cosine LR 스케줄(10%까지 감쇠), residual dropout 0.2로 미세조정해 얻는다. 이 SFT 모델은 그 자체로 baseline이자, 다음 두 단계(RM, PPO)의 시작점이 된다.

## Step 2 — Reward Model (RM)

### 왜 175B가 아니라 6B인가

RM은 SFT 모델에서 마지막 unembedding layer를 떼어내고, 그 자리에 스칼라 하나를 출력하는 head를 붙여 만든다. 입력은 (프롬프트, 응답) 쌍, 출력은 "이 응답이 얼마나 좋은가"를 나타내는 스칼라 $$r_\theta(x, y)$$다.

직관적으로는 RM도 크면 클수록 더 정교한 판단을 할 것 같다. 실제로 175B RM이 6B RM보다 validation loss는 더 낮았다. 그런데도 논문은 **모든 실험에서 6B RM 하나만 썼다.** 이유는 두 가지다.

1. **175B RM 학습이 불안정했다.** 큰 RM은 학습이 흔들려서, PPO의 value function 초기화용으로 쓰기에 적합하지 않았다.
2. **175B RM + value function을 쓰면 PPO의 연산 비용이 크게 늘어난다.** PPO 한 스텝마다 policy와 RM(=value function)을 모두 forward/backward 해야 하는데, 175B를 두 번(policy, RM) 굴리는 비용은 상당하다.

반대로 6B RM은 **넓은 learning rate 범위에서 안정적**이었고, 그렇게 얻은 PPO 모델의 성능도 175B RM을 썼을 때와 동등하게 강했다. "가장 정확한 채점자"보다 "안정적으로 신뢰할 수 있는 채점자"가 파이프라인 전체에는 더 중요했다는 것이다. 이 결정은 이후 RLHF 파이프라인에서 반복되는 패턴이 된다 — reward model은 policy보다 훨씬 작은 경우가 많다.

### K개 응답, C(K,2)개 쌍, 그리고 한 배치로 묶는 트릭

RM을 학습하려면 "어느 응답이 더 나은가"를 비교한 데이터가 필요하다. 순진하게 하면 매번 응답 두 개를 보여주고 어느 게 나은지 물어야 하는데, 이건 라벨링 비용이 크다. 그래서 논문은 한 프롬프트에 대해 **$$K=4$$부터 $$K=9$$까지의 응답을 한 번에 보여주고 전체 순위**를 매기게 했다. 순위 하나로 $$\binom{K}{2}$$개의 쌍 비교를 한꺼번에 얻을 수 있으니 라벨링 효율이 크게 오른다.

문제는 그다음이다. 이 $$\binom{K}{2}$$개의 쌍을 **그냥 뒤섞어서 개별 데이터포인트처럼 취급하면 RM이 1 epoch 만에 overfit**해버린다. 왜일까 — 같은 프롬프트에서 나온 비교쌍들은 서로 강하게 상관되어 있다. 응답 하나가 $$K-1$$개의 다른 쌍 비교에 반복해서 등장하니, 사실상 같은 데이터를 $$K-1$$번 중복해서 gradient에 반영하는 셈이 된다. 이렇게 되면 모델은 일반적인 선호 패턴이 아니라 그 프롬프트 특유의 잡음까지 외워버린다.

해법은 그 프롬프트에서 나온 $$\binom{K}{2}$$개의 쌍 **전체를 하나의 배치 원소로 묶어서** 학습하는 것이다. "묶는다"가 구체적으로 무슨 뜻인지가 이 트릭의 전부이므로, 두 단계로 나눠서 보자.

**1단계 — 응답마다 점수를 딱 한 번씩만 낸다.** $$K$$개 응답을 각각 RM에 한 번씩 통과시켜 스칼라 $$K$$개를 얻는다.

$$s_1 = r_\theta(x, y_1), \quad s_2 = r_\theta(x, y_2), \quad \ldots, \quad s_K = r_\theta(x, y_K)$$

**2단계 — 쌍의 손실은 그 $$K$$개 숫자의 뺄셈으로만 만든다.** 쌍 $$(i, j)$$의 손실은 $$-\log \sigma(s_i - s_j)$$다. 즉 $$\binom{K}{2}$$개의 손실 항이 새로운 forward pass를 하나도 요구하지 않는다. **이미 계산해둔 $$K$$개 스칼라를 재사용하는 산술일 뿐**이다. 그렇게 만든 손실 항들을 평균 내어 이 프롬프트 하나의 손실로 삼고, 역전파를 **한 번** 돈다.

여기서 핵심은 **쌍이 독립적인 학습 예제로 존재하지 않는다**는 점이다. 실제로 신경망을 통과하는 것은 언제나 응답 $$K$$개뿐이고, 쌍은 그 위에 얹힌 계산 그래프의 가지에 불과하다.

이 구조가 앞의 두 문제를 동시에 푼다.

|                                      | 순진한 방식                          | 묶는 방식                    |
| ------------------------------------ | ------------------------------------ | ---------------------------- |
| 인코딩 횟수                          | $$2\binom{K}{2} = K(K-1)$$회         | $$K$$회                      |
| $$K=9$$일 때                         | 72회                                 | 9회 (**$$K-1=8$$배 절감**)   |
| 응답 하나가 gradient에 노출되는 횟수 | $$K-1$$번, 서로 다른 스텝에 흩어져서 | 1번, 한 스텝 안에서 집계되어 |

**연산 효율**은 위 표 그대로다. **overfitting 해소**는 세 번째 행이 설명한다. 순진한 방식에서 응답 $$y_i$$는 $$K-1$$개의 서로 다른 쌍에 등장하고, 그 쌍들이 각기 다른 optimizer 스텝에 흩어지므로 모델은 같은 완성문을 $$K-1$$번 따로 학습하는 셈이 된다. $$K=9$$면 데이터를 1 epoch 돌리는 동안 각 응답을 사실상 8번 본다. 1 epoch 만에 외워버리는 게 당연하다. 묶어서 학습하면 그 $$K-1$$번의 노출이 **한 번의 gradient step 안에서 합쳐지므로** 중복이 사라진다. 그 결과 validation accuracy와 log loss가 크게 개선됐다.

덤으로, 이 트릭이 있어야 $$K$$를 키우는 게 이득이 된다. 라벨러 입장에서 응답 9개를 읽고 줄 세우는 일은 4개를 줄 세우는 것보다 시간이 조금 더 들 뿐인데, 얻는 쌍은 $$\binom{4}{2}=6$$개에서 $$\binom{9}{2}=36$$개로 **6배**가 된다. 묶는 트릭이 없었다면 $$K$$를 키울수록 overfitting만 심해져서 이 이득을 쓸 수 없었을 것이다.

### 토이 예제로 따라가기

$$K=3$$인 경우를 직접 따라가 보자. 라벨러가 프롬프트 $$x$$에 대해 응답 A, B, C를 받고 **B $$>$$ A $$>$$ C** 순으로 순위를 매겼다고 하자. 여기서 나오는 쌍은 $$\binom{3}{2}=3$$개다.

| 쌍     | 승자 $$y_w$$ | 패자 $$y_l$$ |
| ------ | ------------ | ------------ |
| (B, A) | B            | A            |
| (B, C) | B            | C            |
| (A, C) | A            | C            |

순진한 방식이면 이 3개 쌍을 서로 다른 배치(혹은 다른 스텝)에 흩어 넣는다. 각 쌍마다 응답 두 개를 인코딩해야 하니 총 $$3 \times 2 = 6$$번 인코딩하고, 응답 B는 (B,A)와 (B,C) 두 쌍에, A는 (B,A)와 (A,C) 두 쌍에 등장해 각각 서로 다른 스텝에서 두 번씩 학습된다.

InstructGPT의 방식은 이렇다. 먼저 A, B, C를 **한 번씩만 인코딩**해 세 스칼라를 얻는다.

$$s_A = r_\theta(x,A), \quad s_B = r_\theta(x,B), \quad s_C = r_\theta(x,C)$$

그다음 3개 쌍의 손실을 이 세 값의 뺄셈만으로 만든다.

$$\ell_{BA} = -\log\sigma(s_B - s_A), \quad \ell_{BC} = -\log\sigma(s_B - s_C), \quad \ell_{AC} = -\log\sigma(s_A - s_C)$$

이 셋의 평균이 프롬프트 $$x$$ 하나의 손실이고, 여기서 역전파를 **한 번** 돈다. 인코딩 6번 → 3번($$K-1=2$$배 절감), 그리고 B의 두 번의 등장이 별개 스텝이 아니라 **같은 스텝 안에서 $$s_B$$ 하나로 합쳐진다.** $$s_B$$가 받는 gradient는 $$\ell_{BA}$$와 $$\ell_{BC}$$ 양쪽에서 흘러온 값을 더한 것이라, 순진한 방식처럼 "같은 응답을 두 번 따로 학습"하는 일이 생기지 않는다.

### 손실 함수

RM의 손실 함수는 다음과 같다.

$$
L(\theta) = -\frac{1}{\binom{K}{2}} \, \mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\left(\sigma\left(r_\theta(x,y_w) - r_\theta(x,y_l)\right)\right)\right]
$$

기호를 하나씩 풀면 다음과 같다.

- $$r_\theta(x, y)$$: 파라미터 $$\theta$$를 가진 RM이 프롬프트 $$x$$와 응답 $$y$$에 대해 출력하는 스칼라 보상.
- $$y_w$$: 라벨러가 더 선호한(winner) 응답, $$y_l$$: 덜 선호한(loser) 응답.
- $$\sigma$$: 시그모이드 함수. $$r_\theta(x,y_w) - r_\theta(x,y_l)$$을 확률로 변환한다.
- $$D$$: 사람이 매긴 비교 데이터셋 전체.
- $$\binom{K}{2}$$로 나누는 것은 같은 프롬프트 내 쌍 개수로 정규화해, 각 배치(=각 프롬프트)가 손실에 동등한 크기로 기여하게 만드는 장치다.

직관은 단순하다. $$r_\theta(x,y_w) - r_\theta(x,y_l)$$이 클수록(승자를 패자보다 훨씬 높게 평가할수록) $$\sigma(\cdot)$$는 1에 가까워지고 $$-\log \sigma(\cdot)$$는 0에 가까워져 손실이 줄어든다. 즉 이 손실은 **"보상 점수의 차이가 사람이 표시한 선호의 log odds를 재현하도록"** RM을 학습시킨다. 학습이 끝난 뒤에는 RM 손실이 절대적인 보상 크기가 아니라 차이에만 의존하므로(shift-invariant), 라벨러 시범 데이터의 평균 점수가 0이 되도록 bias를 더해 정규화한다.

## Step 3 — Reinforcement Learning with PPO

### KL penalty — RM을 속이지 못하게 묶어두는 고무줄

SFT 모델을 초기 정책으로 삼아, RM이 주는 스칼라 보상을 최대화하도록 PPO(Schulman et al., 2017)로 미세조정한다. 환경은 단순한 bandit이다 — 프롬프트가 주어지면 정책이 응답을 생성하고, RM이 점수를 매기고, 에피소드가 끝난다.

문제는 RM을 그대로 최대화하면 정책이 "RM이 좋아하는 텍스트"를 향해 SFT 분포에서 점점 멀어질 수 있다는 것이다. RM은 사람 선호를 근사한 모델일 뿐 완벽한 참값이 아니므로, 정책이 RM의 빈틈을 찾아 파고들면(over-optimization) RM 점수는 오르지만 실제 사람이 보기엔 이상한 텍스트가 나올 수 있다. 이를 막기 위해 매 토큰마다 SFT 모델과의 **KL divergence penalty**를 보상에서 빼준다.

비유하면, 정책을 SFT라는 말뚝에 고무줄로 묶어둔 것과 같다. RM이라는 먹이를 향해 달려가되, 고무줄이 늘어날수록(=SFT에서 멀어질수록) 당기는 힘이 세져서 너무 멀리 가지 못하게 막는다.

### PPO-ptx — 목적함수에 사전학습 손실을 섞다

KL penalty만으로 정책을 SFT 근처에 묶어도, RLHF로 미세조정한 모델은 SQuAD, DROP, HellaSwag, WMT'15 불어-영어 번역 같은 공개 NLP 벤치마크에서 GPT-3보다 성능이 떨어지는 현상이 관찰됐다. 논문은 이를 **alignment tax**라고 부른다 — 정렬을 얻는 대가로 치르는 추가 비용이다.

이를 상쇄하기 위해 논문은 PPO gradient에 **사전학습 데이터의 log-likelihood gradient를 함께 섞어** 넣는다. 이렇게 얻은 모델을 **PPO-ptx**라 부르고, 별다른 언급이 없는 한 논문에서 "InstructGPT"는 이 PPO-ptx를 가리킨다. 전체 목적함수는 다음과 같다.

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}}\left[r_\theta(x,y) - \beta \log\left(\pi^{RL}_\phi(y \mid x) / \pi^{SFT}(y \mid x)\right)\right] + \gamma \, \mathbb{E}_{x\sim D_{pretrain}}\left[\log\left(\pi^{RL}_\phi(x)\right)\right]
$$

- $$\pi^{RL}_\phi$$: 학습 중인 RL 정책. $$\pi^{SFT}$$: 1단계에서 얻은 SFT 모델(고정).
- 첫 항 $$r_\theta(x,y) - \beta \log(\pi^{RL}_\phi(y \mid x)/\pi^{SFT}(y \mid x))$$: RM 보상에서 KL penalty를 뺀 것. $$\beta$$가 KL의 세기를 조절한다.
- 둘째 항 $$\gamma \, \mathbb{E}_{x\sim D_{pretrain}}[\log(\pi^{RL}_\phi(x))]$$: 사전학습 분포 $$D_{pretrain}$$에서 뽑은 데이터에 대한 언어모델링 log-likelihood. $$\gamma$$가 이 항의 세기를 조절한다.
- "PPO" 모델은 $$\gamma=0$$(사전학습 항 없음), "PPO-ptx" 모델은 $$\gamma$$를 0이 아닌 값으로 설정한다.

여기서 헷갈리기 쉬운 지점을 못 박아두자. **섞어 넣는 것은 사전학습 loss이지 SFT loss가 아니다.** 수식 안에 SFT가 이미 다른 역할로 한 번 등장하기 때문에 둘이 뒤섞이기 쉽다.

| 등장하는 것                            | 수식에서의 위치     | 역할                                                 |
| -------------------------------------- | ------------------- | ---------------------------------------------------- |
| $$\pi^{SFT}$$ — SFT **모델**           | KL 항의 **기준점**  | "SFT에서 너무 멀어지지 마라" → **행동**을 묶는다     |
| $$D_{pretrain}$$ — 사전학습 **데이터** | ptx 항              | "사전학습으로 얻은 걸 잊지 마라" → **능력**을 지킨다 |
| SFT 시범 **데이터**(약 13k)            | **등장하지 않는다** | 1단계에서 이미 다 쓰였다                             |

`ptx`라는 이름 자체가 **p**re**t**raining mi**x**의 약자다. 왜 SFT loss가 아니어야 하는지는 두 가지로 설명된다. 첫째, SFT는 이미 KL 항의 분모로 붙잡고 있으므로 SFT loss를 또 더하면 같은 방향으로 두 번 당기는 중복이다. 둘째, **애초에 망가진 것이 SFT로 배운 능력이 아니다.** 회귀가 난 벤치마크는 SQuAD, DROP, HellaSwag, 번역 — 독해·수치 추론·상식·번역이고, 이건 인터넷 코퍼스 수백 GB에서 생긴 능력이지 라벨러가 쓴 1만 3천 개 지시 시범에서 나온 능력이 아니다. 거기 없던 것을 다시 먹인다고 돌아올 리 없다.

실제 값은 $$\beta = 0.02$$, $$\gamma = 27.8$$이다. 흥미로운 실험 결과는 **KL 계수 $$\beta$$만 키우는 것으로는 이 회귀를 되돌릴 수 없었다**는 점이다 — 기본값의 100배인 $$\beta=2.0$$까지 올려도 SQuAD·DROP 성능 저하는 완전히 회복되지 않았고, 오히려 validation reward만 크게 깎였다. 반면 $$\gamma \geq 20$$(1.3B 모델 기준) 수준으로 사전학습 항을 섞으면 회귀가 상당 부분 복구됐다. 이 실험이 곧 위 표의 증거다. **"SFT 쪽으로 당기기"를 극단까지 밀어도 안 됐다**는 것은, 문제가 "SFT에서 멀어진 것"이 아니라 "사전학습 능력을 잃은 것"이었으므로 당기는 방향 자체가 틀렸다는 뜻이다. 한 문장으로 줄이면 — **KL은 "어떻게 말하는지"를 붙잡고, ptx는 "무엇을 아는지"를 붙잡는다.** 둘은 다른 것이라 각각의 신호가 따로 필요하다.

PPO 학습은 256k episode(약 31k개의 고유 프롬프트를 반복 샘플링), 배치 크기 512, 미니배치 64, PPO clip ratio 0.2로 진행됐고, value function은 6B RM으로 초기화됐다.

## 40명의 라벨러 — 누구의 선호를 배우는가

이 모든 데이터의 출발점은 **약 40명의 계약직 라벨러**다(Upwork와 Scale AI를 통해 고용). 이들은 무작위로 뽑힌 게 아니라 스크리닝 테스트를 통과한 사람들이다.

| 스크리닝 기준         | 내용                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 민감 발화 판별 일치도 | 연구진이 직접 라벨링한 민감 콘텐츠(유해·성적·폭력·정치적 등) 데이터에 대해, 라벨러 판단이 연구진과 얼마나 일치하는지 측정 |
| 순위 매기기 일치도    | API 프롬프트와 여러 모델 응답에 대해, 라벨러의 전체 품질 순위가 연구진 라벨과 얼마나 일치하는지 측정                      |
| 민감 시범 작성 품질   | 민감한 프롬프트에 뉘앙스 있게 응답하는 시범을 작성시키고, 1-7 Likert로 평가                                               |

이렇게 선발된 라벨러들의 **라벨러 간 일치율(inter-annotator agreement)**은 훈련 라벨러끼리 $$72.6 \pm 1.5\%$$, 학습 데이터를 만들지 않은 held-out 라벨러 기준으로는 $$77.3 \pm 1.3\%$$였다. 참고로 Stiennon et al.(2020)의 요약 태스크에서는 연구자-연구자 간 일치율이 $$73 \pm 4\%$$였으니, InstructGPT의 훨씬 넓고 open-ended한 태스크에서도 비슷한 수준의 일치도가 나온 셈이다. 뒤집어 말하면, 아무리 스크리닝을 거쳐도 사람 4명 중 1명 이상은 "정답"에 동의하지 않는다는 뜻이기도 하다.

그렇다면 이 40명은 어떤 사람들인가. 논문 부록이 공개한 국적 분포는 다음과 같다.

| 국적                                                                          | 비중  |
| ----------------------------------------------------------------------------- | ----- |
| Filipino                                                                      | 22%   |
| Bangladeshi                                                                   | 22%   |
| American                                                                      | 17%   |
| Albanian / Brazilian / Canadian / Colombian / Indian / Uruguayan / Zimbabwean | 각 5% |

40명 중 절반 가까이가 필리핀·방글라데시 국적이고, 미국은 17%에 불과하다. 연령대는 25-34세가 47.4%로 가장 많고, 인종적으로는 동남아시아계가 52.6%로 다수를 차지한다. 이 분포 자체가 좋다 나쁘다를 판단할 문제는 아니지만, **"InstructGPT가 정렬된 대상은 인류 보편의 가치가 아니라 이 특정 40명(과 OpenAI 연구진)의 선호"**라는 사실은 분명해진다. 논문도 이 점을 스스로 인정한다 — "이 절차는 GPT-3의 행동을 특정 집단의 명시된 선호에 맞추는 것이지, 더 넓은 의미의 '인간의 가치'에 맞추는 것이 아니다." 그리고 한계 섹션에서는 "대부분의 비교가 라벨러 1명에게만 라벨링되었다"는 점, "라벨러 간 의견이 갈릴 때 평균 선호에 맞추는 것이 항상 바람직한 것은 아니다"라는 점을 직접 지적한다.

# Experiments

## 핵심 결과: 1.3B가 175B를 이긴다

<p align="center"><img src="/assets/post/image/instructgpt/fig1-winrate-by-modelsize.png" width="75%"></p>

위 그림은 175B SFT 모델을 기준(win rate 0.5)으로 놓고, 각 모델이 그 기준을 얼마나 자주 이기는지를 모델 크기별로 그린 것이다. 세 가지가 눈에 띈다.

- **PPO-ptx(빨강)와 PPO(주황)는 모델 크기와 무관하게 GPT-3 계열(파랑 계열)을 압도적으로 이긴다.** 심지어 가장 작은 1.3B PPO-ptx조차 175B GPT-3(few-shot 프롬프팅 포함)보다 선호도가 높다.
- **SFT(초록)만으로도 GPT-3보다는 낫지만, PPO만큼은 아니다.** RM으로 한 번 더 걸러내는 단계가 확실히 추가 이득을 준다.
- **GPT-3에 few-shot prefix를 줘서 "지시를 따르는 척" 시켜도(연한 파랑)** 기본 GPT-3보다는 낫지만 여전히 SFT 175B에도 못 미친다.

정량적으로 보면 다음과 같다.

| 비교                                               | Win rate                                   |
| -------------------------------------------------- | ------------------------------------------ |
| 175B InstructGPT vs 175B GPT-3 (기본)              | $$85 \pm 3\%$$                             |
| 175B InstructGPT vs 175B GPT-3 (few-shot prompted) | $$71 \pm 4\%$$                             |
| 1.3B InstructGPT vs 175B GPT-3                     | 175B GPT-3보다 선호 (100배 이상 작은 모델) |

파라미터를 100배 늘리는 것보다, 13k개의 시범과 33k개의 비교로 사람 피드백을 주는 쪽이 "사용자가 원하는 답"에 훨씬 더 큰 영향을 미쳤다는 뜻이다.

## Truthfulness와 Toxicity

| 지표                            | GPT-3 | InstructGPT        | 비고                                |
| ------------------------------- | ----- | ------------------ | ----------------------------------- |
| TruthfulQA 정답률               | 기준  | 약 2배             | 진실하고 유익한 답변 빈도           |
| Closed-domain hallucination율   | 41%   | 21%                | 입력에 없는 내용을 지어내는 비율    |
| RealToxicityPrompts 독성 출력   | 기준  | 약 25% 감소        | "정중하게 답하라"고 프롬프팅했을 때 |
| Winogender / CrowS-Pairs (편향) | 기준  | 유의미한 개선 없음 | bias는 별도 문제로 남음             |

truthfulness와 toxicity는 확실히 개선됐지만 **편향(bias)은 거의 그대로**라는 점이 중요하다. RLHF가 "사람이 명시적으로 싫어하는 것(거짓말, 욕설)"은 잘 억제하지만, "라벨러가 딱히 지적하지 않은 은근한 편향"까지 잡아내지는 못한다는 뜻이다. 이는 결국 RM이 라벨러가 실제로 표시한 선호의 함수라는 한계에서 나온다 — 라벨러가 신경 쓰지 않은 축은 RM도 학습하지 않는다.

## Alignment tax와 PPO-ptx의 효과

| 모델                                 | SQuAD/DROP 등 공개 NLP 벤치마크            | 라벨러 선호 점수         |
| ------------------------------------ | ------------------------------------------ | ------------------------ |
| PPO ($$\gamma=0$$)                   | 유의미한 성능 회귀 발생                    | 높음                     |
| PPO-ptx ($$\gamma=27.8$$)            | 회귀 대부분 복구                           | PPO와 거의 동등하게 높음 |
| KL 계수만 100배 증가 ($$\beta=2.0$$) | 회귀 여전히 미복구, validation reward 급락 | —                        |

PPO-ptx가 중요한 이유는 "정렬 성능은 유지하면서 alignment tax만 없앤다"는 것을 보여줬기 때문이다. 목적함수에 사전학습 로그우도 항을 더하는 이 간단한 트릭 하나로, RLHF가 "범용 언어모델 능력을 깎아먹는 대가로 정렬을 얻는다"는 우려를 상당 부분 해소했다.

# Conclusion

InstructGPT의 메시지를 한 줄로 요약하면 이렇다. **정렬은 스케일의 문제가 아니라 목적함수의 문제다.** 100배 작은 1.3B 모델이 사람 시범과 사람 선호로 학습한 보상 하나만으로 175B GPT-3를 이겼다는 사실이 그 증거다. 그리고 이 논문이 남긴 진짜 유산은 결과 자체보다 **SFT → RM → PPO라는 재사용 가능한 레시피**다. 6B RM으로 175B 정책을 채점하는 비대칭 설계, $$K$$개 응답을 한 배치로 묶어 overfitting을 막는 트릭, KL penalty로 RM 과최적화를 억제하고 PPO-ptx로 그 대가(alignment tax)를 되갚는 구조 — 이 조합은 이후 거의 모든 RLHF 파이프라인의 기본값이 됐다.

동시에 이 논문은 스스로 다음 세대 연구가 풀어야 할 숙제를 남겼다.

- **Reward hacking의 씨앗**: RM은 사람 선호를 근사한 대리 신호일 뿐이다. KL penalty로 정책이 SFT에서 너무 멀어지지 않게 막았지만, 이는 완화책이지 근본 해법이 아니다. "RM 점수를 얼마나 밀어붙이면 실제 품질과 괴리되기 시작하는가"는 이 논문에서 다뤄지지 않았다 — 이 질문은 이후 overoptimization scaling law 연구(3부)로 이어진다.
- **길이·형식 편향의 가능성**: 이 논문은 응답 길이를 별도로 통제하거나 분석하지 않는다. RM이 "더 길고 상세한 답"을 "더 유용한 답"과 혼동했을 가능성은 열려 있는 채로 남았다 — 이 질문 역시 후속 연구(3부)에서 정량적으로 다뤄진다.
- **"누구의 선호인가"라는 질문**: 정렬의 기준이 된 것은 인류 보편의 가치가 아니라 40명 계약직 라벨러(그중 다수가 필리핀·방글라데시 국적)와 OpenAI 연구진의 판단이다. 저자들 스스로 이 점을 한계로 명시했고, 이 질문은 helpful·harmless를 분리해 학습하는 다음 논문(HH-RLHF)에서 다시 등장한다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 두 번째 글이다.

**1부. 지형도**

1. [Deep RL from Human Preferences (Christiano 2017)](/blog/2026/deep-rl-human-preferences/) — 선호로 보상을 배우는 원형
2. **(현재 글)** InstructGPT (Ouyang 2022) — RLHF 3단계 표준 레시피
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

22. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
23. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
24. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling
25. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
26. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

# 참고 문헌

- Ouyang et al., 2022. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
- [NeurIPS 2022 Proceedings: InstructGPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
- Stiennon et al., 2020. [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325). NeurIPS 2020. (InstructGPT 파이프라인의 직접적인 방법론적 원형)
- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). (시리즈 1부)
- Askell et al., 2021. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861). (Helpful·Honest·Harmless 프레임의 출처)
- Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Brown et al., 2020. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). (GPT-3 원 논문)
