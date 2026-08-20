---
layout: post
title: "RLHF는 정말 좋아진 걸까, 길어진 걸까"
date: 2026-08-11 09:11:00 +0900
description: "RLHF Reward 설계 시리즈 #11 — reward 향상의 대부분이 길이로 설명된다는 불편한 결과"
categories: [paper]
tags: [rlhf, reward-model, reward-hacking, length-bias, evaluation, paper]
giscus_comments: true
related_posts: true
---

> [A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716) (Singhal et al., UT Austin, COLM 2024)

# Introduction

[#10 글](/blog/2026/reward-model-overoptimization/)은 "RLHF로 학습한 reward는 어느 지점부터 실제 품질과 어긋나기 시작한다"는 걸 KL 거리 함수로 정량화했다. Goodhart의 법칙 — "측정치가 목표가 되는 순간 그 측정치는 더 이상 좋은 측정치가 아니다" — 을 스케일링 법칙으로 보여준 것이다. 그런데 그 글은 정작 "policy가 정확히 무엇을 파고들어 reward를 뻥튀기하는가"는 답하지 않는다. 과최적화가 일어난다는 건 알겠는데, 그래서 policy는 구체적으로 뭘 하고 있는 걸까?

이 논문은 그 질문에 아주 구체적인 답 하나를 던진다. **길이다.** 저자들은 WebGPT, Stack(StackExchange), RLCD 세 데이터셋에서 PPO 전후의 reward 개선을 뜯어봤고, 그 개선의 대부분이 "같은 길이대에서 더 나은 답을 쓰게 됐다"가 아니라 "그냥 더 긴 답을 쓰게 됐다"로 설명된다는 걸 보여준다. WebGPT는 이 문제가 특히 심각해서, PPO가 얻은 reward 개선분 중 **딱 2.0%만**이 길이와 무관한 진짜 개선이었다. 나머지 98%는 순전히 "버킷을 옮긴" 효과다.

더 인상적인 건 이 다음이다. 저자들은 reward model을 아예 없애고 "목표 길이에 가까울수록 높은 점수"라는 순수 길이 reward만으로 PPO를 돌렸다. 그런데 이 길이만 보는 PPO(LPPO)가 WebGPT에서 56% 승률, RLCD에서 64% 승률을 냈다 — 진짜 reward model로 학습한 표준 PPO의 58%, 63%와 사실상 동급이다. **reward model 전체가 하는 일의 상당 부분을, 토큰 개수 세기가 대신할 수 있다는 뜻이다.**

이 글은 다음 순서로 이 결과를 뜯어본다.

1. **길이가 얼마나 설명하는가**: Non-Length Reward Gain(NRG)이라는 지표로 "진짜 개선"과 "길이 재배치"를 분리하고, 데이터셋별 수치를 표로 비교한다.
2. **왜 길이가 hacking 축이 되는가**: 사람 선호 데이터 자체에 길이 편향이 있고, reward model이 그중에서도 가장 쉬운 지름길을 학습한다는 training dynamics 분석.
3. **저자들이 시도한 7가지 개입**: PPO 쪽 4개, reward model 쪽 3개 — 각각 어디까지 통하고 어디서 실패하는지.
4. **평가의 함정**: 흔히 쓰는 win-rate 지표가 길이 편향을 얼마나 가리는지.
5. **실무 체크리스트**: 자기 RM에 길이 편향이 있는지 스스로 진단하는 법.

# Background

## RLHF 파이프라인 한 줄 복습

이 시리즈 [#1](/blog/2026/deep-rl-human-preferences/), [#3](/blog/2026/anthropic-hh-rlhf/)에서 다룬 구조 그대로다. 선호 쌍 데이터 $$(x, y^+, y^-)$$로 Bradley-Terry 목적함수를 최소화해 스칼라 reward model $$R(x, y)$$를 학습하고, 그 reward로 PPO가 정책을 업데이트한다.

$$R_{ppo}(x, y) = R(x, y) - \lambda D_{KL}(\pi_\theta^{RL}(y \mid x) \Vert \pi_\theta^{SFT}(y \mid x))$$

- $$R(x, y)$$: 학습된 reward model이 매기는 점수.
- $$\lambda$$: KL 페널티 강도. 크면 SFT 분포에서 멀어지지 않으려 하고, 작으면 reward를 더 적극적으로 좇는다.
- $$D_{KL}(\pi_\theta^{RL} \Vert \pi_\theta^{SFT})$$: 현재 정책이 SFT 정책에서 얼마나 벗어났는지.

기본값은 $$\lambda = 0.04$$다. 이 논문은 이 파이프라인 자체를 바꾸지 않는다. 대신 파이프라인을 그대로 돌리면서 **"reward가 오를 때 policy가 정확히 뭘 바꾸는지"**를 현미경으로 들여다본다.

## 세 데이터셋 — 라벨 출처가 다르면 편향도 다를까

저자들은 라벨 출처가 서로 다른 세 "helpfulness" 데이터셋을 고른다. 라벨이 사람인지, 자동 집계인지, 합성인지에 따라 길이 편향의 정도가 달라지는지 보기 위해서다.

| 데이터셋              | 태스크                         | 선호 라벨 출처             | 규모    | 응답 평균 길이 |
| --------------------- | ------------------------------ | -------------------------- | ------- | -------------- |
| WebGPT                | 개방형 장문 QA                 | 사람이 직접 비교           | 19.6K   | 169 토큰       |
| Stack (StackExchange) | 기술 QA                        | 업보트 수로 자동 산출      | 100K 쌍 | 236 토큰       |
| RLCD                  | 멀티턴 대화 (Helpful/Harmless) | 프롬프트 휴리스틱으로 합성 | 40K     | 45 토큰        |

베이스 모델은 세 설정 모두 **LLaMA-7B**, PPO는 LoRA(rank 16) + 8-bit 양자화로 학습했다. SFT 체크포인트는 WebGPT·RLCD는 AlpacaFarm SFT, Stack은 TRL SFT를 그대로 썼다. 평가는 (1) PPO에 쓰인 reward model 점수 자체, (2) AlpacaFarm의 시뮬레이션 선호(SIM PREF) — GPT-4-0314, GPT-3.5-turbo-0301, text-davinci-003 세 모델을 "평가자 12명"처럼 굴려 500개 held-out 예제에 대해 win rate를 매기는 방식 — 두 가지를 쓴다.

## PPO는 실제로 길이를 늘린다

<p align="center"><img src="/assets/post/image/rlhf-length-correlations/fig2_length_histograms.png" width="85%"></p>

세 설정 모두 PPO(주황) 이후 출력 길이 분포가 SFT(파랑)보다 오른쪽으로 크게 이동한다. WebGPT는 평균이 약 100 → 230 토큰으로, Stack은 203 → 257 토큰으로, RLCD는 59 → 94 토큰으로 늘어난다. 이 자체는 이미 Stiennon et al.(2020), Nakano et al.(2021) 등 여러 선행 연구가 관찰했던 현상이다. 다만 그동안은 "PPO의 부산물" 정도로 취급되고 깊게 파헤쳐지지 않았다. 이 논문의 기여는 여기서 멈추지 않고 "그래서 이게 reward 개선의 진짜 원인인가"를 직접 측정한 것이다.

# Method

## Non-Length Reward Gain (NRG) — 길이를 고정하고 잰다

가장 직접적인 질문은 이렇다. **같은 길이대의 출력끼리만 비교해도 PPO가 SFT보다 reward가 높은가?** 이를 재려고 저자들은 20토큰 단위로 버킷을 나누고, 각 버킷 안에서 SFT와 PPO의 평균 reward 차이를 구한 뒤, 버킷 크기로 가중평균한다. 이렇게 얻은 값이 **Non-Length Reward Gain(NRG)**이다. 전체 reward 개선폭 $$\Delta R$$(PPO 전체 평균 − SFT 전체 평균) 대비 NRG의 비율이, "길이 재배치가 아닌 진짜 개선"이 차지하는 몫이다.

다이어트에 비유하면 이해가 빠르다. 체중계 숫자가 5kg 줄었다고 해서 그게 곧 "건강해졌다"는 뜻은 아니다. 근육이 늘고 지방이 준 것인지, 그냥 물을 뺀 것인지 구분하려면 체지방률 구간을 고정하고 그 안에서 체중이 어떻게 변했는지 따로 봐야 한다. NRG가 하는 일이 정확히 이거다 — "길이"라는 몸무게 총량이 아니라, 같은 길이 구간 안에서 reward가 실제로 올랐는지를 잰다.

## 장난감 예제: 버킷을 옮기기만 해도 reward는 오른다

숫자로 감을 잡아보자. 길이 버킷을 세 개로 나누고, 버킷 자체의 reward 수준(길이가 길수록 reward가 높다는 Figure 1의 경향)과 SFT·PPO의 버킷별 비중을 다음과 같이 가정한다.

| 버킷             | 버킷 내 평균 reward | SFT 비중 | PPO 비중 |
| ---------------- | ------------------- | -------- | -------- |
| 짧음 (~50 토큰)  | 0.0                 | 70%      | 10%      |
| 중간 (~100 토큰) | 0.5                 | 20%      | 20%      |
| 긺 (~150 토큰)   | 1.0                 | 10%      | 70%      |

버킷 안에서는 SFT와 PPO의 평균이 완전히 같다고 가정하자(=진짜 개선은 0, 순수 재배치만 발생).

- SFT 전체 평균: $$0.7 \times 0.0 + 0.2 \times 0.5 + 0.1 \times 1.0 = 0.2$$
- PPO 전체 평균: $$0.1 \times 0.0 + 0.2 \times 0.5 + 0.7 \times 1.0 = 0.8$$
- $$\Delta R = 0.8 - 0.2 = 0.6$$
- NRG = 0 (버킷 내부에서 SFT와 PPO 평균이 동일하므로)
- ratio $$= \text{NRG} / \Delta R = 0\%$$

버킷 내부 개선이 정확히 0인데도 전체 reward는 3배 뛴 것처럼 보인다. 순전히 짧은 버킷에서 긴 버킷으로 표본이 옮겨갔기 때문이다. 이제 버킷마다 진짜로 0.05씩 개선됐다고 가정을 살짝 바꾸면 — SFT 짧음 0.0 → PPO 짧음 0.05, 중간 0.5 → 0.55, 긺 1.0 → 1.05 — NRG는 0.05, $$\Delta R$$은 0.65 근처로 커지고 ratio는 약 7.7%가 된다. 진짜 개선이 있어도, 재배치 효과가 워낙 크면 ratio는 여전히 낮게 나온다는 뜻이다. 이 장난감 계산이 정확히 WebGPT의 실제 패턴이다.

## 실제 수치: Table 1

<p align="center"><img src="/assets/post/image/rlhf-length-correlations/fig3_length_bucketed_reward.png" width="90%"></p>

위 그림은 실제 20토큰 버킷별 reward를 찍은 것이다(High-λ PPO 기준). 검은 점이 SFT, 화살표 끝이 PPO다. WebGPT와 RLCD는 버킷 내부의 화살표 길이(=진짜 개선)가 짧고, 대신 점들이 오른쪽(긴 길이)으로 몰려 있다는 게 한눈에 보인다.

|              | WGPT (표준) | WGPT (High-λ) | Stack (표준) | Stack (High-λ) | RLCD (표준) | RLCD (High-λ) |
| ------------ | ----------- | ------------- | ------------ | -------------- | ----------- | ------------- |
| $$\Delta R$$ | 0.82        | 0.20          | 0.89         | 0.67           | 0.94        | 0.61          |
| NRG          | 0.02        | 0.03          | 0.48         | 0.37           | 0.25        | 0.12          |
| ratio        | **2.0%**    | 15.1%         | **53.4%**    | 56.5%          | **27.2%**   | 19.1%         |

WebGPT는 표준 PPO 기준 reward 개선의 98%가 길이 재배치다. RLCD는 27.2%만 진짜 개선이고, Stack이 그나마 53.4%로 가장 낫다. 저자들은 Stack이 상대적으로 나은 이유를, SFT 출력이 이미 최대 길이 근처라 더 늘어날 여지가 적었고, 기술 QA 특성상 길이 외 신호(코드 정확성 등)에 기댈 여지가 더 많았기 때문이라고 설명한다.

## 순수 길이만으로 보상을 주면? LPPO

reward model을 아예 지우고 길이만 보는 reward로 바꾸면 어떻게 될까. 저자들은 목표 길이 $$L$$에 얼마나 가까운지로 점수를 주는 **LPPO**를 정의한다.

$$R^*(y) = 1 - \left\lvert \frac{\text{len}(y)}{L} - 1 \right\rvert$$

- $$\text{len}(y)$$: 실제 생성된 응답의 토큰 길이.
- $$L$$: 목표 길이 하이퍼파라미터(WebGPT 156, RLCD 120, Stack 250 — 원하는 길이 증가폭을 실험으로 찾아 정한 값).
- $$\text{len}(y)/L = 1$$일 때, 즉 정확히 목표 길이일 때 $$R^*=1$$로 최댓값. 짧든 길든 목표에서 멀어질수록 선형으로 감점된다.

이 reward에는 콘텐츠에 대한 정보가 전혀 없다. 오직 "몇 토큰을 썼는가"만 본다.

## 저자들이 시도한 7가지 개입

<p align="center"><img src="/assets/post/image/rlhf-length-correlations/fig4_interventions_pipeline.png" width="95%"></p>

길이 의존을 줄이기 위한 개입을 파이프라인의 두 지점에 나눠 걸었다.

| 개입                                   | 위치         | 아이디어                                                                |
| -------------------------------------- | ------------ | ----------------------------------------------------------------------- |
| I.1 Length Balancing (BAL)             | 선호 데이터  | 선호-비선호 쌍의 길이 차이 분포를 10토큰 단위로 대칭화                  |
| I.2 Reward Data Augmentation (R-DA)    | 선호 데이터  | 서로 다른 프롬프트의 응답을 무작위로 짝지어 "비선호"로 라벨링(25% 추가) |
| I.3 Confidence-based Truncation (C-TR) | RM 학습      | 저신뢰(confidence 낮은) 예제만 골라 재학습                              |
| I.4 Omit Long Outputs                  | PPO 롤아웃   | 길이 임계값을 넘는 출력을 배치에서 제외하고 무작위 샘플로 교체          |
| I.5 Penalize Length                    | reward 점수  | 긴 출력에 스칼라 페널티 부여                                            |
| I.6 Reward Scaling                     | reward 점수  | 배치 정규화 방식으로 reward 스케일 고정                                 |
| I.7 High-λ KL                          | PPO 목적함수 | KL 페널티를 0.04 → 0.12로 키워 초기 분포에 더 묶어둠                    |

I.5, I.6의 수식은 다음과 같다.

$$R' = R + \left(1 - \frac{\text{len}(y)}{N}\right)\sigma$$

- $$N$$: PPO가 넘지 않길 바라는 최대 길이.
- $$\sigma$$: 배치 reward 표준편차의 이동평균.
- 길이가 $$N$$에 가까울수록 보너스 항이 0에 수렴하고, 짧을수록 보너스가 커진다.

$$R' = \frac{R - \mu}{\sigma}$$

- $$\mu, \sigma$$: 최근 배치들에 대한 reward의 이동평균과 이동표준편차.
- 배치 정규화와 같은 발상으로, 원래는 과최적화로 인한 학습 변동을 잡으려는 목적([Zheng et al., 2023 — Secrets of RLHF I](/blog/2026/secrets-rlhf-ppo/))이지 길이를 직접 겨냥한 개입은 아니다.

# Experiments

## LPPO는 진짜 PPO를 거의 따라잡는다

<p align="center"><img src="/assets/post/image/rlhf-length-correlations/fig1_reward_length_heatmap.png" width="70%"></p>

위 그림(WebGPT)은 SFT 출력의 길이-reward 히트맵이다. 별표로 찍힌 SFT 예시(59 토큰, reward ≈ −0.9)는 왼쪽 아래에 있고, 200토큰을 넘는 지점은 대부분 +0.5~+1.0 사이에 몰려 있다. 오른쪽 패널은 실제 사례 하나 — "어른들은 왜 자다가 침대에서 안 떨어지나?"라는 질문에 SFT는 59 토큰으로 짧게 답했고, RLHF 이후 모델은 243 토큰으로 같은 내용을 훨씬 장황하게 풀어 썼다. 두 응답의 정보량 차이는 크지 않은데, 히트맵상 reward 차이는 1점 가까이 벌어진다.

|                         | W-GPT |          | Stack |          | RLCD |          |
| ----------------------- | ----- | -------- | ----- | -------- | ---- | -------- |
|                         | 길이  | SIM PREF | 길이  | SIM PREF | 길이 | SIM PREF |
| SFT                     | 100   | 50%      | 203   | 50%      | 59   | 50%      |
| 표준 PPO                | 230   | 58%*     | 257   | 58%*     | 94   | 63%*     |
| SFT-LONG(8개 중 최장)   | 141   | 48%      | 249   | 57%*     | 117  | 52%      |
| **LPPO**(길이만 최적화) | 118   | 56%*     | 252   | 59%*     | 98   | 64%*     |
| LPPO ($$\lambda=0$$)    | 167   | 53%      | 248   | 58%*     | 163  | 51%      |

(*는 SFT 대비 통계적으로 유의한 차이, $$p<0.05$$)

핵심은 LPPO 행이다. WebGPT에서 118 토큰짜리 LPPO가 56% 승률로, 230 토큰까지 늘어난 표준 PPO(58%)와 거의 같은 성능을 낸다. RLCD에서는 LPPO(64%)가 오히려 표준 PPO(63%)를 근소하게 앞선다. **reward model이 학습한 것 중 상당 부분을, "목표 길이에 맞춰라"는 한 줄짜리 규칙이 대신할 수 있다는 뜻이다.** 흥미로운 점 하나 더: LPPO(118 토큰)는 자신보다 더 긴 SFT-LONG(141 토큰, SFT에서 8개 뽑아 가장 긴 것)보다도 승률이 높다. 순수하게 길게 쓰는 것과 "적당한 목표 길이 + KL 제약 속에서 길게 쓰는 것"은 다르다는 신호다. 저자들은 KL 항이 반복적이고 병적인(repetitive, pathological) 출력을 억제해, 길이만 최적화하는 와중에도 어느 정도 서술적인 문장을 유지하게 만든다고 해석한다.

## 평가의 함정: win rate만 보면 안 되는 이유

이 표가 남기는 가장 불편한 시사점은 **AlpacaFarm 같은 시뮬레이션 선호 평가 자체가 길이에 취약할 수 있다**는 것이다. 저자들도 이를 인정한다 — "이 지표 자체가 길이 편향을 가질 수 있다"고 명시적으로 밝히며, 그래서 Table 1의 NRG처럼 다른 지표를 더 신뢰한다고 적는다. 실무에서 "우리 모델이 RLHF 후 win rate가 올랐다"는 결과를 볼 때 반드시 물어야 할 질문은 이거다: **평가자(사람이든 LLM judge든)가 길이 자체에 낚인 건 아닌가?** 길이를 통제한 채(같은 길이대끼리만, 혹은 LPPO 같은 순수 길이 베이스라인과) 다시 재보지 않으면, "개선"이 진짜인지 판단할 수 없다. [#9 RewardBench 2 글](/blog/2026/rewardbench-2/)이 다루는 "RM을 어떻게 평가할 것인가"라는 질문도 결국 이 함정을 피하기 위한 것이다.

## PPO 쪽 개입: 길이는 줄지만 reward도 같이 깎인다

|                   | W-GPT |        |          | Stack |        |          | RLCD |        |          |
| ----------------- | ----- | ------ | -------- | ----- | ------ | -------- | ---- | ------ | -------- |
|                   | 길이  | reward | SIM PREF | 길이  | reward | SIM PREF | 길이 | reward | SIM PREF |
| SFT               | 100   | −0.45  | 42%*     | 203   | 0.05   | 42%*     | 59   | 4.40   | 37%*     |
| 표준 PPO          | 230   | 0.25   | 50%      | 257   | 0.74   | 50%      | 94   | 5.50   | 50%      |
| Reward Scale      | 128   | −0.05  | 49%      | 249   | 0.40   | 46%*     | 82   | 5.00   | 41%*     |
| Penalize Length   | −     | −      | −        | −     | −      | −        | 72   | 5.20   | 44%*     |
| High-λ            | 120   | −0.06  | 45%*     | 250   | 0.30   | 45%*     | 97   | 5.20   | 43%*     |
| Omit Long Outputs | 127   | −0.13  | 48%      | −     | −      | −        | −    | −      | −        |

(SIM PREF는 표준 PPO 대비. "−"는 학습이 수렴하지 않아 결과가 없는 경우)

패턴이 뚜렷하다. **모든 개입이 길이를 표준 PPO보다 줄이는 데는 성공한다.** 하지만 예외 없이 reward와 SIM PREF도 함께 떨어진다. Penalize Length와 Omit Long Outputs는 대부분의 설정에서 아예 수렴에 실패한다(표의 "−"). 즉 "길이를 억지로 누르면 reward 자체를 학습하는 능력도 같이 무너진다"는 것 — 이는 reward model이 길이 신호에 얼마나 깊이 의존하고 있는지를 역설적으로 보여준다.

## Reward Model 쪽 개입: 데이터를 고쳐도 안 사라지는 편향

|                           | WGPT ACC | WGPT CORR | Stack ACC | Stack CORR | RLCD ACC | RLCD CORR |
| ------------------------- | -------- | --------- | --------- | ---------- | -------- | --------- |
| RAND (기준선)             | 50%      | 0         | 50%       | 0          | 50%      | 0         |
| STND (표준 RM)            | 61.5%    | 0.72      | 70%       | 0.55       | 80%      | 0.67      |
| BAL (길이 균형)           | 52.6%    | −0.13     | 61.9%     | −0.09      | 73.1%    | 0.62      |
| C-TR (저신뢰 재학습)      | 58.8%    | 0.67      | 59.5%     | 0.31       | 77.2%    | 0.57      |
| R-DA (무작위 페어링 증강) | 62.5%    | 0.35      | 72.6%     | 0.37       | 80%      | 0.43      |

ACC는 held-out 정확도, CORR은 같은 프롬프트에서 뽑은 8개 생성물 내부의 길이-reward Pearson 상관이다. 참고로 "무조건 더 긴 쪽을 고른다"는 단순 휴리스틱만으로도 WebGPT 55.7%, Stack 59.6%, RLCD 63.1% 정확도가 나온다(표준 RM보다 크게 낮은 정도가 아니다) — **선호 데이터 자체에 길이 쪽으로 기운 편향이 이미 있다는 뜻**이다.

BAL(길이 균형)은 상관을 거의 0 또는 음수로 낮추는 데 성공하지만, 그 대가로 정확도가 함께 무너진다(WebGPT 61.5% → 52.6%, 거의 랜덤 수준). RLCD는 BAL을 적용해도 상관이 0.62로 여전히 높게 남는다 — "균형을 맞춰도 안 사라지는" 가장 고집 센 사례다. R-DA만 유일하게 상관을 낮추면서 정확도를 오히려 소폭 끌어올린다(WebGPT 61.5% → 62.5%).

다운스트림 결과도 함께 보면 개입의 실효성이 더 명확해진다.

|      | WGPT 길이 | WGPT SIM PREF | Stack 길이 | Stack SIM PREF | RLCD 길이 | RLCD SIM PREF |
| ---- | --------- | ------------- | ---------- | -------------- | --------- | ------------- |
| SFT  | 100       | 42%*          | 203        | 42%*           | 59        | 37%*          |
| STND | 230       | 50%           | 257        | 50%            | 94        | 50%           |
| BAL  | −         | −             | 148        | 57%*           | 82        | 44%*          |
| R-DA | 139       | 49%           | 256        | 58%*           | 112       | 44%*          |
| C-TR | 141       | 44%*          | 244        | 44%*           | 97        | 50%           |

Stack에서 BAL은 인상적이다. 길이를 SFT(203)보다도 짧은 148 토큰으로 줄이면서 동시에 SIM PREF를 57%로 끌어올린다 — 길이를 늘리지 않고도 표준 PPO에 준하는 개선이 가능함을 보여주는 유일한 사례다. 저자들은 Stack에 길이 외에도 배울 만한 "쉬운" 신호(코드 스니펫 정확성 등)가 있어서라고 추정한다. 반대로 WebGPT는 BAL 자체가 수렴하지 않고, RLCD는 BAL을 해도 짧아진 만큼 SIM PREF가 오히려 SFT에 가깝게 떨어진다(44%). **"어떤 개입도 모든 데이터셋에 통하지 않는다"**는 게 이 절 전체의 결론이다.

## 왜 길이가 hacking 축이 되는가: "쉬운" 예제가 학습을 지배한다

<p align="center"><img src="/assets/post/image/rlhf-length-correlations/fig5_confidence_length_heuristic.png" width="85%"></p>

BAL로 데이터를 균형 잡아도 RLCD에서 상관이 안 사라지는 이유를 저자들은 training dynamics로 파고든다. reward model을 여러 epoch 학습시키며 각 학습 예제에 대해 "confidence" — 선호 응답과 비선호 응답의 reward 차이 $$R(x_i, y_i^+) - R(x_i, y_i^-)$$ — 를 epoch마다 기록한다. 이 confidence를 구간별로 나누고, 각 구간에서 "무조건 긴 쪽을 고른다"는 길이 휴리스틱이 얼마나 잘 맞는지(길이 휴리스틱 정확도)를 그린 게 위 그림이다.

패턴이 놀랍도록 깔끔하다. confidence가 0 근처(모델이 확신하지 못하는 대다수 예제)에서는 길이 휴리스틱 정확도도 낮거나 중간이지만, confidence가 극단으로 갈수록(모델이 강하게 확신하는 소수 예제) 길이 휴리스틱 정확도가 거의 1.0에 수렴한다. 즉 **reward model이 "확신을 갖고" 학습하는 예제 대부분이 바로 길이 휴리스틱이 통하는 예제다.** 대부분의 학습 데이터에 대해서는 사실 모델이 이렇다 할 확신을 갖지 못하는데, 그 와중에 몇 안 되는 "쉬운" 신호 — 길이 — 가 학습을 지배해버리는 것이다. 흥미롭게도 이 패턴이 가장 뚜렷한 데이터셋이 WebGPT이고, WebGPT는 Table 1에서 NRG 비율이 가장 낮았던(2.0%) 데이터셋과 정확히 일치한다.

일상 비유로 옮기면 이렇다. 채점자가 리포트 100장을 눈 코 뜰 새 없이 채점해야 하는데, 대부분의 리포트는 내용이 애매해서 판단이 잘 안 선다. 그런데 유독 "페이지 수가 확 차이 나는" 몇 장만큼은 채점자가 자신 있게 판단을 내린다 — "이게 더 두꺼우니 더 성의 있겠지." 채점자의 최종 채점 기준은 결국 이 "쉬운 몇 장"에서 배운 지름길, 즉 페이지 수에 수렴해버린다. reward model도 마찬가지다. 대부분의 선호 쌍에서 헤매다가, 확신을 가질 수 있는 몇 안 되는 사례에서 길이라는 지름길을 배우고, 그 지름길을 전체 데이터에 일반화해버린다.

## 곁다리 증거들: DPO도, 큰 모델도, 다른 목적함수도

부록 실험 세 가지가 본문 결과의 일반성을 보강한다.

| 실험                                  | 결과                                                                                                                                                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DPO로 바꿔도 동일 (Appendix C.1)      | RLCD/Stack/WebGPT 모두 DPO 후에도 길이가 늘어난다(59→68, 203→248, 100→164). RM 정확도는 오히려 명시적 reward model보다 살짝 낮거나 비슷하다 — DPO가 reward model을 없앤다고 이 문제를 피해가는 건 아니다. |
| RM을 7B → 13B로 키워도 (Appendix C.2) | 정확도가 소폭만 오른다(WebGPT 61.5%→64.5%, Stack 70%→71.3%, RLCD 80%→81.2%). 스케일이 근본 원인이 아니라는 뜻이다.                                                                                        |
| Harmlessness 목적함수 (Appendix C.3)  | 정확도는 비슷하게 낮지만(68%) 길이 상관은 오히려 음수(약 −0.3)로 나온다. PPO로 학습해도 길이가 늘지 않는다. 대신 다른 방식으로 "이상해지는" 출력이 나온다.                                                |

마지막 행이 특히 중요한 균형추다. **길이 편향이 모든 objective에 보편적인 건 아니다.** Harmlessness는 오히려 "짧게 회피하는 답"이 안전한 경우가 많아서 방향이 반대로 나타난다. 다만 저자들은 이 경우도 reward model이 얕은 특징에 기대는 근본 패턴 자체는 동일하다고 지적한다 — 그 얕은 특징이 helpfulness에서는 길이였을 뿐이다. 즉 "길이"는 이 문제의 원인이 아니라, **가장 흔하게 관찰되는 증상**이다.

# Conclusion

한 줄로 요약하면: **RLHF가 만들어내는 reward 개선의 상당 부분(WebGPT는 98%, RLCD는 73%, Stack은 47%)이 콘텐츠 개선이 아니라 단순히 더 긴 출력을 만든 결과이며, 이는 reward model이 선호 데이터의 소수 "쉬운" 예제에서 길이라는 지름길을 학습하기 때문이다.**

실무에서 자기 RM에 이 문제가 있는지 진단하려면 다음을 확인하면 된다. 이 표의 항목들은 모두 이 논문이 실제로 쓴 진단 도구다.

| 진단 방법                           | 무엇을 보는가                                              | 위험 신호                                                |
| ----------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Within-batch 길이-reward 상관       | 같은 프롬프트에서 뽑은 여러 생성물 내부의 Pearson 상관     | 0.5 이상이면 위험 (WebGPT 0.72, RLCD 0.67)               |
| Length heuristic accuracy           | "무조건 긴 쪽이 선호"라는 규칙만으로 예측한 정확도         | 랜덤(50%)보다 유의하게 높으면 데이터 자체에 편향 존재    |
| Length-stratified reward gain (NRG) | 길이 버킷을 고정하고 그 안에서 reward가 실제로 오르는지    | ratio가 낮을수록 개선이 길이 재배치로 설명됨             |
| LPPO 대조군                         | reward model 대신 순수 길이 reward로 PPO 시행 후 승률 비교 | 표준 PPO와 승률이 비슷하면 RM이 길이 이상을 못 준다는 뜻 |
| SFT-LONG 베이스라인                 | SFT에서 8개 샘플 중 가장 긴 것 선택                        | 이보다 짧으면서도 이기면 길이 외 신호가 있다는 증거      |
| 개입 후 재검증                      | 길이 균형 등 개입 후에도 상관이 남는지                     | RLCD처럼 balancing해도 안 없어지면 구조적 문제           |

이 논문이 남긴 부채는 명확하다. 저자들 스스로도 인정하듯, 7가지 개입 중 어느 것도 "길이도 짧고, reward도 유지되고, 모든 데이터셋에 통하는" 조합을 만들지 못했다. Reward Scale이나 High-λ 같은 개입은 데이터를 건드리지 않는 사후 처방이라 근본적인 한계가 있고, BAL·R-DA 같은 데이터 개입은 데이터셋마다 정반대 결과를 낸다. 이 부채는 이후 두 방향으로 이어진다. [#12 ODIN 글](/blog/2026/odin-disentangled-reward/)은 이 문제를 데이터 개입이 아니라 **아키텍처**로 풀려는 시도다 — reward 자체를 길이 성분과 콘텐츠 성분으로 명시적으로 분리하는 head를 둔다. [#7 ArmoRM 글](/blog/2026/armorm/)은 좀 더 넓은 각도에서, 길이를 포함한 여러 목적을 애초에 하나의 스칼라로 뭉개지 않고 다목적으로 분해해 MoE 게이팅으로 조합하는 방식으로 이 문제를 완화한다. 두 접근 모두 이 글의 결론 — "reward model은 얕은 지름길에 취약하고, 그 대표 사례가 길이다" — 을 전제로 삼는다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열한 번째 글이다.

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
  <li><strong>(현재 글)</strong> Length Correlations in RLHF (2023) — 성능 향상의 얼마가 길이인가</li>
  <li><a href="/blog/2026/odin-disentangled-reward/">ODIN (2024)</a> — 길이를 reward에서 분리</li>
  <li><a href="/blog/2026/warm-weight-averaged-reward/">WARM (2024)</a> — weight averaging으로 hacking 방어</li>
</ol>

**4부. 안전성 정렬**

<ol start="14">
  <li><a href="/blog/2026/safe-rlhf/">Safe RLHF (2023)</a> — 안전성을 reward가 아니라 제약으로</li>
  <li><a href="/blog/2026/rule-based-rewards/">Rule-Based Rewards (2024)</a> — 안전 규칙을 reward로 직접 번역</li>
  <li><a href="/blog/2026/deliberative-alignment/">Deliberative Alignment (2024)</a> — 안전 명세를 모델의 추론 안으로</li>
  <li><a href="/blog/2026/shallow-safety-alignment/">Shallow Safety Alignment (2024)</a> — 정렬은 첫 몇 토큰에만 얹혀 있다</li>
  <li><a href="/blog/2026/or-bench/">OR-Bench (2024)</a> — 과잉 거절을 어떻게 측정할 것인가</li>
</ol>

**5부. reward를 정책으로**

<ol start="19">
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

<ol start="30">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="33">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="38">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="43">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어의 helpfulness reward 설계</a> — 열한 개 모델이 능력 축에서 택한 것</li>
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 45편으로 구성된다.

# 참고 문헌

- Singhal et al., 2023/2024. [A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716). Published as a conference paper at COLM 2024.
- [arXiv HTML: A Long Way to Go (v2)](https://arxiv.org/html/2310.03716v2) — 본문 그림 원본.
- [GitHub: PrasannS/rlhf-length-biases](https://github.com/PrasannS/rlhf-length-biases) — 저자 공개 코드, reward/policy 체크포인트.
- [OpenReview: A Long Way to Go (COLM 2024)](https://openreview.net/forum?id=G8LaO1P0xv) — 리뷰 스레드.
- Nakano et al., 2021. [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332). (WebGPT 데이터셋 원 논문)
- Yang et al., 2023. [RLCD: Reinforcement Learning from Contrast Distillation for Language Model Alignment](https://arxiv.org/abs/2307.12950). (RLCD 데이터셋 원 논문)
- Dubois et al., 2023. [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://arxiv.org/abs/2305.14387). (SIM PREF 평가 지표)
- Swayamdipta et al., 2020. [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://arxiv.org/abs/2009.10795). (confidence 기반 분석 방법론)
