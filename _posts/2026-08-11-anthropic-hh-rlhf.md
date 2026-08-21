---
layout: post
title: "HH-RLHF: helpfulness와 harmlessness는 왜 충돌하는가"
date: 2026-08-11 09:03:00 +0900
description: "RLHF Reward 설계 시리즈 #3 — preference model의 스케일링과 helpful·harmless 텐션 (Bai et al., Anthropic, arXiv 2022)"
categories: [paper]
tags: [rlhf, reward-model, preference-model, safety, alignment, paper]
giscus_comments: true
related_posts: true
---

> [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (Bai et al., Anthropic, arXiv 2022)

# Introduction

이 시리즈 [#1 Christiano 글](/blog/2026/deep-rl-human-preferences/)과 [#2 InstructGPT 글](/blog/2026/instructgpt/)은 "선호로부터 보상을 배운다"는 RLHF의 뼈대를 다뤘다. 사람이 두 응답 중 더 나은 것을 고르면, 그 선택으로부터 스칼라 reward를 뽑아내고, 그 reward로 정책을 최적화한다. 이 파이프라인 자체는 이제 표준이다.

문제는 "더 나은 것"이 하나의 기준이 아니라는 데 있다. 어시스턴트는 도움이 되어야(helpful) 하고, 동시에 해롭지 않아야(harmless) 한다. 이 둘을 하나의 숫자로 합쳐서 reward로 쓰면 무슨 일이 벌어질까? Anthropic의 HH-RLHF 논문이 실제로 겪은 실패 사례가 있다. 연구 초반 RLHF 정책들이 사용자가 조금이라도 불만을 표하면 거의 항상 "전문가와 상담해보세요" 같은 천편일률적인 답을 내놓았다. 위험을 피하는 데는 성공했지만, 쓸모는 거의 없는 모델이었다. 저자들은 이를 harmlessness는 과최적화되고 helpfulness는 과소최적화된 상태라고 진단한다.

이 글은 InstructGPT처럼 "RLHF 파이프라인 자체"가 아니라, **reward를 어떻게 설계했는가**에 집중한다. HH-RLHF 논문의 핵심 기여 세 가지가 여기 해당한다.

1. **helpful·harmless preference model(PM)의 스케일링 법칙** — 모델 크기와 데이터가 늘어날수록 PM 정확도가 어떻게 오르는가.
2. **helpfulness와 harmlessness를 하나의 PM에 욱여넣었을 때의 충돌** — 두 목표가 서로를 얼마나 갉아먹는지, 그리고 모델 크기가 이 충돌을 어떻게 완화하는지.
3. **iterated online RLHF** — PM을 한 번 학습하고 끝내는 게 아니라, 매주 정책을 다시 배포해 새 데이터를 모으고 PM을 다시 학습하는 루프.

미리 결론을 요약하면 이렇다. PM 정확도는 파라미터 수·데이터 양 모두에 대해 대략 로그 선형으로 오른다. 작은 모델은 정렬 학습으로 성능이 깎이는 'alignment tax'를 겪지만, 13B·52B급에서는 오히려 성능이 오르는 'alignment bonus'로 반전된다. 그리고 helpfulness와 harmlessness는 100% 서로 반대 방향으로 학습시키면 상대 축의 정확도가 **찍기(chance)보다 낮아질 정도**로 강하게 충돌하는데, 모델이 커질수록 이 충돌에 덜 민감해진다.

이 텐션은 이 시리즈 전체를 관통하는 주제이기도 하다. 하나의 스칼라 reward에 여러 목표를 섞는 방식의 한계는 이후 [#7 ArmoRM](/blog/2026/armorm/)이 다목적 분해로, [#8 Llama 2](/blog/2026/llama2-rlhf/)가 helpfulness RM과 safety RM을 아예 분리하는 방식으로 답한다. HH-RLHF는 그 문제가 **왜 생기는지**를 가장 먼저, 가장 정량적으로 보여준 논문이다.

참고로 이 논문이 만든 red-teaming 데이터셋 자체(필드 구조, 라벨링 방식, 활용법)는 [이전 글](/blog/2026/hh-rlhf-red-team/)에서 다뤘다. 이 글은 그 데이터가 **어떻게 preference model의 reward로 바뀌고, helpful 데이터와 섞였을 때 무슨 일이 일어나는지**에 집중한다. 즉 이전 글이 "광석의 성분표"였다면, 이 글은 "그 광석으로 만든 저울(reward)이 왜 한쪽으로 기우는가"를 다룬다.

# Background

## Preference Model 한 줄 복습

PM은 대화 하나에 스칼라 점수 $$r_{PM}$$을 매기는 모델이다. 두 응답 A, B 중 사람이 A를 선호했다면, PM은 $$r_{PM}(A) > r_{PM}(B)$$가 되도록 Bradley-Terry 스타일로 학습된다. 이 변환의 수학적 근거는 [#4 Rethinking Bradley-Terry 글](/blog/2026/bradley-terry-rethinking/)에서 더 깊게 다룰 예정이니, 여기서는 "두 응답의 점수 차이가 클수록 사람이 그 응답을 선호했을 확률도 높다"는 직관만 잡고 넘어간다.

## 데이터를 어떻게 모았는가 — 인터페이스 설계가 곧 reward 설계

HH-RLHF의 첫 번째 설계 결정은 **helpfulness와 harmlessness를 완전히 별도의 태스크로 수집한 것**이다. 크라우드워커는 둘 중 하나의 역할만 맡는다.

- **Helpfulness 태스크**: 모델에게 질문·글쓰기·계획 상담 등을 요청하고, 두 응답 중 **더 도움이 되는(better) 쪽**을 고른다. 대화는 갈수록 더 나은 방향으로 움직인다.
- **Harmlessness(red-teaming) 태스크**: 모델을 적대적으로 찔러 유해한 응답을 끌어내려 시도하고, 두 응답 중 **더 해로운(worse) 쪽**을 고른다. 대화는 갈수록 더 나쁜 방향으로 움직인다.

이 비대칭이 나중에 중요한 문제를 만든다(Experiments 절에서 자세히 다룬다). 저자들 스스로도 "harmlessness 데이터는 모델에게 무엇을 하지 말아야 하는지만 알려줄 뿐, 무엇을 해야 하는지는 알려주지 않는다"고 명시적으로 지적한다.

데이터는 세 갈래로 수집됐다.

| 데이터 소스                        | Helpfulness 비교 수 | Harmlessness(red-team) 비교 수 | 비고                           |
| ---------------------------------- | ------------------- | ------------------------------ | ------------------------------ |
| Base (context-distilled 52B 모델)  | 44k                 | 42k                            | 대화 1개 ≈ 비교 4개            |
| RS (rejection sampling, k=16)      | 52k                 | 2k                             | base로 학습한 PM으로 샘플 선별 |
| Online (주 단위 RLHF 정책, 약 5주) | 22k                 | 0                              | red-team 데이터 없음           |

base+RS를 합쳐 **static 데이터셋**(helpful 96k, harmless 44k)이라 부르고, 여기에 online 22k를 더한 것이 **online 데이터셋**이다. 총합은 helpful 약 118k, harmless 약 44k 비교로, helpfulness 쪽이 harmlessness보다 약 2.7배 많다. 이 비대칭 때문에 뒤에서 다룰 손실 가중치 실험이 필요해진다.

크라우드워커는 미국 거주 MTurk·Upwork 작업자였고, 그중 가장 활발한 약 20명이 전체 데이터의 80%를 만들었다. 흥미로운 지표 하나: Anthropic 연구자와 크라우드워커 사이의 평균 합의율은 약 **63%**에 불과했다. 즉 "어느 응답이 더 나은가"라는 판단 자체가 상당히 주관적이라는 뜻이다. 이 라벨 노이즈는 PM 학습에도, 뒤에 나올 calibration 결과 해석에도 배경으로 깔려 있다.

## 토이 예제 — PM 점수는 어떻게 Elo·선호 빈도로 바뀌는가

논문은 PM 점수 차이를 사람이 실제로 느끼는 선호 빈도(Elo, win rate)로 변환하는 공식을 준다.

$$
\Delta(\text{Elo Score}) \approx 174 \times \Delta(\text{PM Score})
$$

$$
\text{Win Fraction} = \frac{1}{1 + 10^{\Delta(\text{Elo Score})/400}}
$$

기호를 하나씩 풀면, $$\Delta(\text{PM Score})$$는 두 모델 응답의 PM 점수 차이, $$\Delta(\text{Elo Score})$$는 그 차이를 사람이 체감하는 Elo 점수 격차로 환산한 값, Win Fraction은 두 모델을 나란히 붙였을 때 진 쪽이 이긴 것으로 관찰될 확률이다.

숫자로 따라가보자. 두 정책 A, B의 PM 점수 차이가 $$\Delta(\text{PM Score}) = 2$$라고 하자.

1. Elo 격차: $$\Delta(\text{Elo}) \approx 174 \times 2 = 348$$점.
2. 진 쪽(B)이 이긴 것으로 관찰될 확률: $$10^{348/400} \approx 7.5$$이므로 $$\text{Win Fraction} = 1/(1+7.5) \approx 0.12$$, 즉 A가 약 88%의 빈도로 선호된다.

이 계산이 장난감 숫자만은 아니다. 논문 Figure 1이 정확히 이 환산을 실제 모델에 적용한 결과다.

읽기 전에 두 축이 무엇인지부터 정리하자. 둘 다 **baseline(context-distilled 52B)과 1:1로 붙였을 때**의 결과다.

- **helpfulness Elo**: 위 식으로 환산한 Elo 격차. 0점이면 baseline과 동률이다.
- **harmlessness 선호 빈도**: 사람에게 두 응답을 나란히 보여주고 **덜 유해한 쪽**을 고르게 했을 때 그 모델이 뽑힌 비율. **50%면 baseline과 동률**이고, 50% 아래면 baseline보다 더 유해하다는 뜻이다.

이 기준으로 두 모델을 비교하면 이렇다.

| 52B 모델                | helpfulness Elo        | harmlessness 선호 빈도 |
| ----------------------- | ---------------------- | ---------------------- |
| online **HH** RLHF      | 약 +250점 (≈ 81% 선호) | 약 **80%**             |
| online **helpful-only** | 약 +330점 (≈ 87% 선호) | 약 **27%**             |

`helpful-only`는 harmlessness 데이터를 빼고 helpfulness만으로 학습한 모델이다. Elo 괄호 안 숫자는 위 Win Fraction 식에 그대로 넣은 값이다 — 250점이면 $$10^{250/400} \approx 4.2$$이므로 baseline이 이길 확률이 $$1/(1+4.2) \approx 19\%$$, 즉 이 모델이 약 81% 빈도로 선호된다.

두 행을 나란히 놓으면 대가가 선명하다. helpful-only는 helpfulness에서 HH 모델을 앞질러 **전문 작가(Professional Writers) 밴드에 근접**한다. 그런데 harmlessness는 **27%** — 50%가 동률이므로, 이 모델은 아무것도 정렬하지 않은 baseline보다도 **더 유해해졌다.** 도움이 되도록만 밀어붙이면 "요청받은 건 뭐든 잘 해주는" 방향으로 가고, 그 안에는 하면 안 되는 요청도 포함된다. 논문 표현으로는 **레드팀 공격이 훨씬 쉬운 모델**이 된 것이다.

아래 그림이 그 대비다.

<p align="center"><img src="/assets/post/image/anthropic-hh-rlhf/fig1-elo-comparison.png" width="90%"></p>

- 왼쪽 패널(helpfulness Elo, x축 파라미터 수): context-distilled → static HH RLHF → online HH RLHF → online helpful-only RLHF 순으로 점수가 오르고, helpful-only 모델은 전문 작가(Professional Writers) 밴드를 넘어선다.
- 오른쪽 패널(52B harmlessness 선호 빈도): online HH RLHF는 baseline보다 훨씬 안전해지지만(약 80%), online helpful-only RLHF는 오히려 baseline보다 위험해진다(약 27%).

# Method

## PM 스케일링 — 모델이 커질수록, 데이터가 많아질수록

PM 학습에는 13M부터 52B까지, 약 4배씩 증가하는 7개 모델 크기를 썼다. 학습 전 [Askell et al., 2021](https://arxiv.org/abs/2112.00861)에서 제안한 preference model pretraining(PMP)을 거치고, 사람 선호 데이터로는 딱 1 epoch만 파인튜닝한다. 1 epoch만 돌리기 때문에 학습 곡선 자체가 곧 "데이터 양에 따른 스케일링"을 보여준다.

<p align="center"><img src="/assets/post/image/anthropic-hh-rlhf/fig7-pm-scaling.png" width="95%"></p>

- 왼쪽(학습 곡선): x축은 학습에 쓴 비교(comparison) 수, 색은 모델 크기. 데이터가 $$10^3$$에서 $$10^5$$개로 늘어날 때 정확도가 대략 로그 선형으로 오른다 — 가장 작은 모델은 0.53 근처에서 0.64 근처로, 가장 큰 색상 라인은 0.58 근처에서 0.72 근처로 움직인다.
- 오른쪽(모델 크기 의존성): 13M(약 $$10^7$$)에서 52B(약 $$5\times10^{10}$$)로 갈수록 helpful·harmless·mixture 세 곡선 모두 test accuracy가 약 0.61~0.66에서 약 0.72로 오른다. 재미있는 점은 아주 작은 모델에서는 harmless 단일 데이터 정확도(약 0.61)가 helpful 단일 데이터 정확도(약 0.65)보다 낮지만, 모델이 커지면서 이 격차가 사라지고 세 곡선이 약 0.72 근처로 수렴한다는 것이다.

이 "일단 크게 만들면 웬만한 노이즈는 무마된다"는 패턴은 뒤에서 다룰 helpful·harmless 혼합 실험에서도 반복된다.

## Calibration — PM 점수 차이가 곧 확률이 되는가

RLHF에서 PM 점수를 reward로 쓰려면, 점수 차이가 실제 선호 확률과 일치해야 신뢰할 수 있다. 논문은 이를 다음 식으로 검증한다.

$$
P(A \succ B) = \frac{1}{1 + e^{r_{PM}(B) - r_{PM}(A)}}
$$

여기서 $$r_{PM}(A) - r_{PM}(B)$$가 클수록 A가 선호될 확률이 1에 가까워진다. 실측 정확도를 이 곡선에 겹쳐 그렸을 때, helpfulness 단일 데이터로 학습한 PM은 거의 완벽하게 이 곡선을 따라갔지만, helpful+harmless 혼합 데이터로 학습한 PM은 살짝 **과소확신(under-confident)** 이었다. 두 데이터 분포를 하나의 헤드로 압축하는 대가로 calibration이 약간 흐려진다는 뜻이다.

독립적인 검증도 있다. Anthropic이 이전 논문에서 만든 HHH(helpful·honest·harmless) 평가셋에서 52B static PM은 **86%** 정확도를 기록했다. 같은 평가셋에서 Google의 Pathways(PaLM) 팀이 보고한 사람 평균 정확도는 **75%**였으니, PM이 평균적인 사람보다도 이 판단을 더 잘한다는 뜻이다. 다만 예외도 있다 — PM이 자신 있게 틀리는 소수 사례는 전부 "정직한 대신 불친절한 답"보다 "그럴듯하지만 미묘하게 틀린 답"을 선호하는 정직성(honesty) 실패였다.

| 평가 대상                                                                         | HHH 평가 정확도 |
| --------------------------------------------------------------------------------- | --------------- |
| 52B static PM (본 논문)                                                           | 86%             |
| 사람 평균 (Pathways/PaLM 보고치)                                                  | 75%             |
| 논문 이전 prompted 모델들 [Askell et al., 2021](https://arxiv.org/abs/2112.00861) | PM보다 낮음     |

# Experiments

## 하나의 스칼라에 두 목표를 넣으면 생기는 일

이 절이 이 글의 핵심이다. RL reward는 다음과 같이 정의된다.

$$
r_{total} = r_{PM} - \lambda_{KL} D_{KL}(\pi \Vert \pi_0)
$$

$$\pi$$는 현재 정책, $$\pi_0$$는 초기 정책(context-distilled LM), $$\lambda_{KL}$$은 KL 페널티 가중치다. 논문은 $$\lambda_{KL} = 0.001$$이라는 아주 작은 값을 써서, 사실상 대부분의 학습 구간에서 KL 페널티가 거의 작동하지 않는다(학습 중 $$D_{KL}$$이 보통 100 미만이라 페널티가 무시할 만한 크기다). 즉 reward는 사실상 $$r_{PM}$$ 그 자체다 — helpfulness와 harmlessness가 하나의 숫자로 압축된다는 뜻이다.

문제는 여기서 시작한다. 아래 그림은 52B PM이 매긴 helpful·harmless 비교 데이터 분포(왼쪽)와, 그 PM으로 학습한 RLHF 정책이 실제로 도달한 점수(오른쪽)를 겹쳐 보여준다.

<p align="center"><img src="/assets/post/image/anthropic-hh-rlhf/fig14-overoptimization.png" width="95%"></p>

- 왼쪽: harmless 비교 데이터(빨강)는 대부분 PM 점수 −2~0 구간에 몰려 있다.
- 오른쪽: 그런데 학습된 정책의 harmlessness 점수(빨강)는 훈련이 진행되며 점점 올라가 왼쪽 그래프의 **상위 꼬리(약 2점)**까지 도달한다. 반대로 helpfulness 점수(파랑)는 왼쪽 helpful 데이터 분포의 평균(약 4.7점) 근처에 머문다.

즉 **harmlessness는 데이터의 상위 극단까지 밀어붙여 과최적화된 반면, helpfulness는 여전히 과소최적화** 상태다. 왜 이런 비대칭이 생길까? 저자들의 설명은 직관적이다. red-team 프롬프트에서 아주 높은 harmlessness 점수를 받는 가장 쉬운 방법은 "답변할 수 없습니다"라고 거절하는 것이다. 이건 유해 요청을 분류하는 능력만 있으면 되므로, 정교한 도움을 주는 것보다 훨씬 배우기 쉽다.

**일상 비유**: 보안 요원에게 "방문객을 잘 응대하되 위험한 사람은 반드시 막아라"는 지침만 주면 어떻게 될까? 가장 쉬운 전략은 **아무도 들여보내지 않는 것**이다. 위험 관리 점수는 만점이지만, 안내 데스크로서는 낙제점이다. RLHF 정책이 "모르겠어요"만 반복하게 된 것도 정확히 같은 이유다 — 논문은 이런 실패를 피하고, 위험한 요청조차 정중히 설명하며 응대하는 모델을 **'인질 협상가(hostage negotiator)'** 모델이라 부른다.

여기서 앞서 짚은 데이터 수집 비대칭이 다시 등장한다. harmlessness 데이터는 대화가 갈수록 **더 나쁜 방향**으로 움직이도록 수집했기 때문에, 모델은 "무엇을 하면 안 되는가"만 배우고 "위험한 요청에 어떻게 잘 대응해야 하는가"는 배울 기회가 없다. 저자들은 이를 향후 harmlessness 데이터도 "가장 좋은 응답을 고르는" 방식으로 재수집해야 한다고 명시적으로 제안한다.

## 데이터 비율을 바꿔보면 — 반비례하는 두 축

정말로 두 목표가 서로 반대 방향인지 직접 검증한 실험도 있다. helpful 데이터 비율을 0%에서 100%까지 10% 단위로 바꿔가며 PM을 학습했다.

<p align="center"><img src="/assets/post/image/anthropic-hh-rlhf/fig19-mixing-tradeoff.png" width="95%"></p>

- 위 왼쪽: harmlessness 데이터 100%로만 학습하면 helpfulness test accuracy가 약 0.35~0.40까지 떨어진다. 이진 분류 chance level(0.5)보다 **낮다** — 그냥 찍는 것보다 못한 정책을 학습한다는 뜻이다.
- 위 오른쪽: 반대로 helpfulness 데이터 100%로만 학습하면 harmlessness test accuracy가 약 0.32~0.35까지 떨어진다. 마찬가지로 chance보다 낮다.
- 아래 두 그래프(정규화 버전): 모델 크기별로 최댓값 대비 정규화하면, **큰 모델(노란색)일수록 극단 비율에서도 정확도가 덜 떨어진다.** 즉 대형 PM은 데이터 비율에 훨씬 강건하다.

| 학습 데이터 구성           | Helpfulness Test Acc      | Harmlessness Test Acc |
| -------------------------- | ------------------------- | --------------------- |
| Helpful 100% / Harmless 0% | 정상(~0.72)               | chance 이하(~0.32)    |
| Helpful 0% / Harmless 100% | chance 이하(\~0.35\~0.40) | 정상(~0.75)           |
| 적절한 혼합 (52B)          | ~0.72                     | ~0.72                 |

손실 가중치($$\lambda$$)를 바꾸는 실험도 같은 결론을 가리킨다. $$L_{Total} = L_{Helpfulness} + \lambda \cdot L_{Harmlessness}$$에서 $$\lambda$$를 1에서 10으로 올리면, 13M 모델은 helpfulness 정확도가 **7.4%p** 떨어지지만 52B 모델은 **1.5%p**만 떨어진다. 모델이 커질수록 두 목표를 동시에 잘 배우는 능력 자체가 커진다는 뜻이다.

## Alignment Tax와 Alignment Bonus

정렬 학습이 범용 성능을 깎아먹지는 않을까? 이 우려에 대한 답이 아래 그림이다.

<p align="center"><img src="/assets/post/image/anthropic-hh-rlhf/fig3-alignment-tax-bonus.png" width="95%"></p>

MMLU, Lambada, HellaSwag, OpenBookQA, ARC-Easy/Challenge, TriviaQA 7개 벤치마크의 평균 정확도를 파라미터 수에 따라 그린 그래프다. 작은 모델(왼쪽)에서는 RLHF 곡선(주황)이 plain LM 곡선(파랑)보다 아래에 있다 — **alignment tax**다. 그런데 파라미터가 커질수록 두 곡선이 만나고, 13B·52B 구간에서는 zero-shot에서 RLHF가 plain LM을 역전한다 — **alignment bonus**다. few-shot에서는 두 곡선이 거의 겹쳐, 최소한 손해는 없다.

| 모델 크기        | Zero-shot               | Few-shot                   |
| ---------------- | ----------------------- | -------------------------- |
| 소형 (\~10M\~1B) | RLHF < Plain LM (tax)   | RLHF < Plain LM (약한 tax) |
| 13B              | RLHF > Plain LM (bonus) | RLHF ≈ Plain LM            |
| 52B              | RLHF > Plain LM (bonus) | RLHF ≈ Plain LM            |

왜 크기에 따라 부호가 뒤집힐까. 열쇠는 **RLHF가 새 지식을 넣어주지 않는다**는 데 있다. 이 벤치마크들이 묻는 상식·독해·사실은 전부 사전학습에서 온 것이고, RLHF는 그걸 더 넣어주는 게 아니라 **이미 가진 것을 어떻게 꺼내 쓸지를 바꾼다.**

그러면 두 크기에서 벌어지는 일이 갈린다.

- **작은 모델**: 애초에 꺼낼 것이 별로 없다. 그런데 "도움이 되게, 해롭지 않게, 어시스턴트답게"라는 행동 제약은 똑같이 걸린다. 즉 **얻는 것 없이 제약만 지불**하니 순손실이다.
- **큰 모델**: 아는 건 많은데 raw 언어모델은 그걸 안정적으로 꺼내주지 않는다. 인터넷 텍스트를 흉내내는 모델은 상황에 따라 아무 저자나 될 수 있어서, 헛소리하는 사람도 전문가도 흉내낼 수 있다. RLHF는 이 모델을 **"성실하게 답하는 어시스턴트" 하나의 모드로 고정**시키는데, 공교롭게도 그 모드가 벤치마크 문제를 푸는 데 유리한 모드다.

즉 큰 모델에서 RLHF는 **능력을 주입하는 게 아니라 끌어내는(elicitation) 장치**로 작동한다. 이미 있던 것이 더 자주 표면에 나오니 점수가 오른다.

이 해석을 뒷받침하는 게 위 표의 **few-shot 열**이다. few-shot에서는 bonus가 사라지고 두 곡선이 거의 겹친다. few-shot 예시 자체가 이미 "이런 식으로 답하라"고 모드를 지정해주기 때문이다. **RLHF가 해주던 일을 예시가 대신 해주니 추가 이득이 없다.** 반대로 zero-shot에서는 모드를 잡아줄 장치가 없으므로 RLHF의 효과가 그대로 드러난다.

저자들은 이 결과에서 중요한 함의를 하나 짚는다. **작은 모델만 가지고 정렬 연구를 하면, 큰 모델에 그대로 외삽했을 때 틀린 결론에 도달할 수 있다.** "정렬은 성능을 깎는다"는 통념이 애초에 작은 모델 실험에서 나온 것일 수 있다는 경고다. 실제로 [#2 InstructGPT](/blog/2026/instructgpt/)는 이 tax를 상쇄하려고 PPO-ptx라는 별도 장치(사전학습 loss를 목적함수에 섞기)를 도입했는데, HH-RLHF의 결과는 **충분히 큰 모델에서는 그런 장치 없이도 tax가 저절로 사라질 수 있음**을 시사한다.

같은 맥락에서 alignment는 특화 스킬과도 잘 섞인다. 요약(summarization) 데이터를 HH 데이터와 섞어 PM을 학습해도 요약 성능·HH 성능 둘 다 저하가 없었고, 코드로 파인튜닝된 모델에 자연어 HH RLHF를 추가로 적용하면 오히려 프로그래밍 평가 성능이 올라갔다(아마 일반적인 지시 따르기 능력이 좋아진 덕분으로 추정된다).

## Iterated Online RLHF — 매주 시험지를 새로 낸다

앞서 calibration 절에서 짚었듯, PM은 고득점 구간에서 점점 신뢰도가 떨어진다. 원인은 단순하다 — 애초에 그 구간에 사람이 남긴 비교 데이터 자체가 부족하기 때문이다. 이 문제를 풀기 위해 저자들이 제안한 방법이 **iterated online RLHF**다.

1. 현재 가진 최선의 PM으로 RLHF 정책을 학습한다.
2. 그 정책을 크라우드워커 인터페이스에 배포해 새 비교 데이터를 모은다. 정책이 PM 점수를 최대화하도록 학습됐으므로, 이 데이터는 자연히 점수 분포의 상위 꼬리를 채운다.
3. 새 데이터를 기존 데이터와 합쳐 PM을 다시 학습한다.
4. 그 새 PM으로 RLHF 정책을 다시 학습한다. 1로 돌아가 반복한다.

이 과정을 약 5주간 주 단위로 돌렸다. 여기서 논문이 쓰는 'online'이라는 용어는 흔한 online RL(같은 모델을 계속 업데이트)과 다르다는 점을 명시한다 — 매 iteration마다 **새 모델을 처음부터 다시 학습**한다.

**일상 비유**: 학원이 매주 같은 모의고사를 낸다고 하자. 학생들이 그 시험에 익숙해지면 다들 만점을 받아서, 누가 더 실력이 좋은지 변별이 안 된다. 그래서 학원은 매주 학생들의 현재 실력에 맞춰 **더 어려운 시험지**를 새로 낸다. Online RLHF도 같은 이유로 동작한다 — 정책이 좋아질수록 PM이 변별해야 할 데이터도 그만큼 어려워져야 한다.

효과를 검증하는 방법도 흥미롭다. 최종 online PM으로 base·RS·online 세 데이터 분포 각각의 test set 정확도를 측정했더니 **74%, 70%, 67%**로 순서대로 낮아졌다. 즉 데이터 품질이 올라갈수록(base → RS → online) 고품질 샘플들 사이에서 "어느 쪽이 더 나은가"를 구분하기가 더 어려워진다 — 변별력이 필요한 지점이 계속 위로 밀려 올라간다는 뜻이다.

| 테스트 데이터 분포       | Online PM 정확도 | 해석                              |
| ------------------------ | ---------------- | --------------------------------- |
| Base (context-distilled) | 74%              | 상대적으로 쉬운 변별              |
| RS (rejection sampling)  | 70%              | 중간 난이도                       |
| Online (iterated RLHF)   | 67%              | 고품질 샘플 간 구분이 가장 어려움 |

데이터셋 크기·하이퍼파라미터 차이가 결과를 왜곡했을 가능성을 배제하기 위한 통제 실험도 진행했다. base 데이터만 44k로 학습한 PM과, base·RS·online에서 각각 약 15k씩(합쳐서 동일하게 44k) 뽑은 혼합 데이터로 학습한 PM을 같은 조건에서 비교했다. 크라우드워커 평가 결과 혼합 데이터로 학습한 정책이 명확히 더 선호됐다 — 즉 online 데이터의 이득은 단순히 "데이터가 더 많아서"가 아니라 **분포가 상위 꼬리를 채워서** 생긴 것임을 확인했다.

## $$\sqrt{D_{KL}}$$과 reward의 선형 관계

부가적인 발견 하나: RLHF 학습 곡선을 $$\sqrt{D_{KL}(\pi \Vert \pi_0)}$$ 대 reward 평면에 그리면, 학습의 상당 구간에서 거의 **직선**이 나온다. KL은 정책이 초기 정책에서 얼마나 멀어졌는지를 재는 값이니, 이 관계는 "reward를 더 뽑아내려면 정책을 얼마나 바꿔야 하는가"에 대한 예산표 같은 역할을 한다. 비유하자면 헬스장에서 벤치프레스 중량을 늘릴 때, 초반에는 자세(KL)를 조금만 바꿔도 중량이 쉽게 오르지만, 어느 순간부터는 자세를 더 크게 바꿔야 같은 폭의 중량 증가를 얻는 것과 비슷하다. 다만 저자들은 이 구간에서는 오히려 거의 선형이라는 점 자체를 강조하며, 이 관계가 모델 크기 간 RLHF의 "실효 크기 이득"을 설명한다고 본다 — 그림 1에서 RLHF 곡선과 context-distilled 곡선이 거의 평행한 이유도 이 선형 관계 때문이다. 자세한 RL 최적화 메커니즘은 [#20 PPO 글](/blog/2026/ppo/)에서 더 다룬다.

# Conclusion

HH-RLHF의 메시지를 한 줄로 요약하면 이렇다. **helpfulness와 harmlessness를 하나의 스칼라 preference model에 담으면, 두 목표는 서로를 거의 반비례로 갉아먹는다. 모델을 키우면 이 충돌은 완화되지만 사라지지는 않는다.**

정리하면,

1. **PM 스케일링**: 정확도는 모델 크기·데이터 양 모두에 대해 로그 선형으로 오른다. 52B PM은 HHH 평가에서 86%로 사람 평균(75%)을 넘어선다.
2. **텐션**: helpful 100% 또는 harmless 100%로만 학습하면 반대 축 정확도가 chance보다 낮아진다. 데이터 수집 인터페이스 자체(harmlessness를 "더 나쁜 쪽 선택"으로 수집한 결정)가 이 텐션을 키운 원인 중 하나였다. 큰 모델일수록 이 비율 변화에 덜 민감하다.
3. **Alignment tax → bonus**: 작은 모델은 정렬로 성능이 깎이지만, 13B·52B에서는 오히려 오른다.
4. **Online RLHF**: PM을 한 번 학습하고 끝내지 않고, 주 단위로 정책을 재배포해 상위 꼬리 데이터를 계속 채워 넣는다. 정적 데이터 대비 명확히 더 나은 정책을 만든다.

한계도 분명하다. 저자들 스스로 인정하듯 harmlessness 데이터가 "무엇을 하지 말아야 하는가"만 가르치는 구조적 비대칭은 이 논문에서 완전히 해결되지 않았다. 크라우드워커-연구자 합의율이 63%에 그칠 만큼 라벨 자체가 주관적이라는 점, 크라우드워커 구성이 프로젝트 기간 내내 고정되지 않아 online 학습 효과와 뒤섞였을 가능성도 남아 있다. 그리고 무엇보다, 이 논문이 보여준 "하나의 스칼라로는 두 목표를 완전히 화해시킬 수 없다"는 관찰이 이후 연구가 reward를 어떻게 쪼개는지—[ArmoRM의 다목적 분해](/blog/2026/armorm/), [Llama 2의 RM 이원화](/blog/2026/llama2-rlhf/)—를 이해하는 출발점이 된다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 세 번째 글이다.

**1부. 지형도**

<ol start="1">
  <li><a href="/blog/2026/deep-rl-human-preferences/">Deep RL from Human Preferences (Christiano 2017)</a> — 선호로 보상을 배우는 원형</li>
  <li><a href="/blog/2026/instructgpt/">InstructGPT (Ouyang 2022)</a> — RLHF 3단계 표준 레시피</li>
  <li><strong>(현재 글)</strong> HH-RLHF (Bai 2022) — helpful·harmless preference model</li>
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

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="39">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="44">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어의 helpfulness reward 설계</a> — 열한 개 모델이 능력 축에서 택한 것</li>
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 46편으로 구성된다.

# 참고 문헌

- Bai et al., 2022. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862).
- Askell et al., 2021. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861). (PM pretraining, context distillation, Elo 비교 방법론의 원출처)
- Stiennon et al., 2020. [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325). (KL 페널티 reward 설계, summarization 특화 스킬 비교 대상)
- 장원범, 2026. [HH-RLHF Red-Team Attempts: Anthropic의 38,961건 레드팀 대화 데이터셋](/blog/2026/hh-rlhf-red-team/). (같은 논문의 red-team 데이터셋 구조·활용 관점)
- [HuggingFace: Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GitHub: anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf)
