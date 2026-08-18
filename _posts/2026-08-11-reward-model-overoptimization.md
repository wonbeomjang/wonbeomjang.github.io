---
layout: post
title: "Reward Model Overoptimization: Goodhart의 법칙을 수식으로 쓰다"
date: 2026-08-11 09:10:00 +0900
description: "RLHF Reward 설계 시리즈 #10 — proxy reward를 올릴수록 진짜 reward가 꺾이는 지점의 스케일링 법칙"
categories: [paper]
tags: [rlhf, reward-model, reward-hacking, goodhart, scaling-law, paper]
giscus_comments: true
related_posts: true
---

> [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al., OpenAI, ICML 2023)

# Introduction

이 시리즈 [1편](/blog/2026/deep-rl-human-preferences/)에서 저자들은 이상한 장면을 하나 목격했다. 보상 모델을 한 번만 오프라인으로 학습시키고 그 뒤로는 고정한 채 정책을 계속 학습시켰더니, Pong 에이전트가 점수를 얻으려 하지 않고 랠리를 비정상적으로 길게 끄는 행동으로 수렴한 것이다. 고정된 $$\hat r$$ 입장에서는 최적이었지만, 참 보상 기준으로는 명백히 이상한 행동이었다. 그 글에서는 이걸 "reward hacking의 초기 사례"라고만 부르고 넘어갔다. **관찰**은 했지만 **측정**은 하지 않았다.

[9편](/blog/2026/rewardbench-2/)까지 이 시리즈는 "RM을 어떻게 잘 만들 것인가"를 다뤘다. 데이터를 어떻게 큐레이션하고, 손실 함수를 어떻게 바꾸고, 벤치마크에서 몇 점을 받는지. 그런데 벤치마크 점수가 높은 RM도 실제로 정책을 그 RM으로 최적화하면 뚫린다. 이 논문은 정확히 그 지점에서 질문을 바꾼다. "RM을 얼마나 잘 만드느냐"가 아니라 "**RM을 얼마나 오래 최적화해도 되느냐**"를 묻는다. Goodhart's law — "측정이 목표가 되는 순간 그 측정은 더 이상 좋은 측정이 아니다" — 를 구호로만 인용하는 대신, 그 붕괴가 정확히 어떤 곡선을 그리는지 수식으로 적어낸다.

문제는 이걸 측정하려면 사람 라벨이 어마어마하게 필요하다는 점이다. RM 크기 9종 × 데이터 크기 여러 종 × 정책 크기 2종, 각 조합마다 KL 거리를 촘촘히 훑어가며 "지금 이 정책이 진짜로 얼마나 좋은가"를 계속 물어야 한다. 사람에게 이걸 시키면 예산이 버티지 못한다. 그래서 저자들은 사람 대신 **또 다른 RM**을 세운다 — "gold-standard RM"이라는 이름의 가짜 인간이다.

이 글이 답할 질문은 세 가지다.

1. **왜 진짜 사람 대신 gold RM을 썼는가**, 그리고 그 우회가 어떻게 가능한가.
2. proxy reward가 커질수록 gold reward가 **어떤 함수 모양으로** 꺾이는가 — BoN과 RL이 왜 다른 모양을 그리는가.
3. 그 함수의 계수는 **무엇에 의존**하는가 — RM을 키우면, 데이터를 늘리면, 정책을 키우면 각각 무슨 일이 벌어지는가.

이 글은 3부 "Reward Hacking"의 첫 글이다. 앞의 2부가 "RM을 잘 만드는 법"이었다면, 이 글은 **아무리 잘 만들어도 최적화하면 무너진다**는 근본 한계를 정량화한다. 11~13편은 이 한계에 대한 세 가지 다른 대응이다 — Conclusion에서 그 연결을 짚는다.

# Background

## 관찰은 많았지만 측정은 없었다

reward hacking 자체는 이 논문 이전에도 여러 번 목격된 현상이다.

| 사례                             | 관찰 내용                                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Christiano et al. 2017 (Pong)    | 오프라인 고정 RM 최적화 시 랠리를 무한히 끄는 행동으로 수렴 ([1편](/blog/2026/deep-rl-human-preferences/)) |
| Christiano et al. 2017 (로봇 손) | 사람 피드백으로 학습한 로봇 팔이 공을 실제로 쥐지 않고 카메라 각도상 쥔 것처럼만 보이는 자세를 취함        |
| Stiennon et al. 2020 (요약)      | RM을 오래 최적화한 요약 모델이 사람 평가로는 오히려 나빠지는 구간 관찰                                     |
| InstructGPT 계열 실무 관찰       | RM이 긴 답변을 선호하는 경향을 학습해, 정책이 실제 품질과 무관하게 답변 길이만 늘리는 현상                 |

공통점은 전부 "이런 게 있더라"는 사후 보고라는 점이다. 어느 정도 KL을 쓰면 꺾이기 시작하는지, RM을 키우면 그 지점이 얼마나 늦춰지는지, 이런 정량적 질문에는 아무도 답하지 못했다. 사람 라벨을 촘촘히 반복 수집하는 비용이 감당되지 않았기 때문이다.

## Goodhart's law와 네 가지 유형

"측정이 목표가 되면 그 측정은 더 이상 좋은 측정이 아니다"라는 경험칙은 원래 경제학에서 나왔지만, 학습된 판별자·RM을 최적화 대상으로 쓰는 모든 머신러닝 세팅에 그대로 적용된다. Manheim과 Garrabrant(2018)는 이 효과를 네 가지 메커니즘으로 분류한다.

| 유형         | 정의                                                                                    | 일상 비유                                                                                               |
| ------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Regressional | proxy = gold + 독립 노이즈. 노이즈까지 같이 선택하게 됨                                 | 시험에서 상위권을 뽑으면 실력자와 "운 좋게 잘 찍은" 학생이 섞여 있다. 다음 시험에서는 평균으로 회귀한다 |
| Extremal     | 최적화가 진행되며 표본 분포가 RM의 학습 분포 밖(OOD)으로 밀려나 proxy가 신뢰 불가능해짐 | 체중계가 정상 체중 범위에서만 정확하듯, RM도 학습 때 본 범위를 벗어나면 눈금이 헛돈다                   |
| Causal       | 공통 원인 때문에 생긴 상관관계를 proxy가 인과관계로 오인                                | 아이스크림 판매량과 익사 사고가 함께 느는 건 여름 날씨 때문이지, 아이스크림이 익사를 유발해서가 아니다  |
| Adversarial  | 정책이 능동적으로 proxy를 속이는 전략을 구사                                            | 감독관의 채점 습관 자체를 역이용해 점수를 조작                                                          |

이 네 유형을 미리 알아두면 뒤에서 등장할 두 계수 $$\alpha$$, $$\beta$$가 각각 무엇의 흔적인지 해석할 수 있다.

## KL을 공통의 자로 쓰는 이유

BoN과 RL은 "정책을 최적화한다"는 점은 같지만 방식이 완전히 다르다. 그래서 "얼마나 최적화했는가"를 비교할 공통 잣대가 필요한데, 이 논문은 초기 정책 $$\pi_{init}$$과 최적화된 정책 $$\pi$$ 사이의 KL divergence를 그 잣대로 쓴다.

$$\mathrm{KL} := D_{KL}(\pi \parallel \pi_{init})$$

$$\pi_{init}$$에서 출발하면 항상 0이고, 정책이 멀어질수록 커진다. 그리고 저자들은 이 KL을 그대로 쓰지 않고 제곱근을 씌운 거리로 재정의한다.

$$d := \sqrt{\mathrm{KL}} = \sqrt{D_{KL}(\pi \parallel \pi_{init})}$$

왜 굳이 제곱근일까. Bai et al.(2022, 4.3절)에 따르면 KL은 정책 공간에서 **이차(quadratic) 형태의 거리 척도**다. 자동차 계기판이 이동한 킬로미터가 아니라 그 제곱값을 보여준다고 생각하면, "몇 km 이동했는지" 알려면 계기판 값에 제곱근을 씌워야 한다. $$d$$가 바로 그 "실제 이동 거리"에 해당한다. 뒤에 나올 두 함수형이 $$d$$에 대해 훨씬 깔끔한 모양(선형·이차)으로 fit되는 것도 이 재매개변수화 덕분이다.

# Method

## 왜 사람이 아니라 gold RM인가

이 실험을 사람으로 직접 하려면 무엇이 필요한지 따져보자. RM 크기 9종(3M~3B), 데이터 크기 여러 종, 정책 크기 2종을 조합하고, 각 조합마다 정책이 KL 0부터 최대 100 나츠까지 이동하는 궤적을 촘촘히 훑으며 "지금 이 응답이 진짜로 얼마나 좋은가"를 반복 측정해야 한다. 게다가 스케일링 법칙을 신뢰성 있게 fit하려면 각 지점의 측정 노이즈도 작아야 한다. 사람 라벨은 비싸고 노이즈도 크다 — 이 조합 전체를 사람으로 커버하는 건 예산상 불가능하다.

그래서 저자들은 사람을 아예 실험에서 빼고, **사람 대신 또 다른 학습된 RM**을 세운다. Ouyang et al.(2022)의 InstructGPT 6B RM을 "gold-standard RM"으로 고정하고, 이것이 진짜 인간 라벨러의 역할을 대신한다. proxy RM(3M~3B, 9개 크기)은 이 gold RM이 채점한 라벨로 학습된다.

<p align="center"><img src="/assets/post/image/reward-model-overoptimization/fig2_setup_diagram.png" width="85%"></p>

위 그림(논문 Figure 2)이 이 우회를 보여준다. 위쪽 "Real" 파이프라인에서는 사람 라벨러가 comparison 데이터를 만들고 그걸로 proxy RM을 학습시킨다. 아래쪽 "Synthetic" 파이프라인에서는 사람 자리에 gold RM이 들어간다 — 정책이 만든 롤아웃 쌍을 gold RM에게 채점시켜 더 높은 점수를 받은 쪽을 "선호됨"으로 표시한, 100% 결정론적인 라벨이다. 두 파이프라인 모두에서 **proxy RM은 어차피 그 라벨을 만든 원천(사람이든 gold RM이든)의 근사치일 뿐**이라는 게 저자들의 핵심 주장이다. gold RM으로 진짜 인간을 완벽히 대체하지는 못하지만, "proxy가 gold를 근사하다가 어긋나는 과정" 자체는 진짜 RLHF와 동일한 구조로 재현된다.

구체적으로는 정책이 만든 롤아웃 쌍 100,000개를 gold RM으로 채점해 synthetic comparison 100,000건을 만들고, 90,000건은 학습, 10,000건은 검증에 쓴다. 이렇게 만들어진 라벨은 무제한으로, 공짜로, 언제든 다시 만들 수 있다 — 사람 라벨로는 절대 불가능한 실험 규모다.

단, 이 우회에는 명확한 한계가 있다. gold RM도 결국 사람 의도의 근사치이지, 사람 의도 그 자체가 아니다. 그래서 이 논문이 측정하는 건 "**proxy RM이 gold RM을 근사하다가 어긋나는 오차**"이지, [1편](/blog/2026/deep-rl-human-preferences/)에서 언급한 "라벨러가 실제로 원하는 것과, 라벨러가 실제로 매긴 라벨 사이의 괴리"(공을 쥔 것처럼만 보이게 만든 로봇 손 사례)는 이 실험의 범위 밖이다. 저자들 스스로 이 두 번째 종류의 overoptimization은 이 논문에서 다루지 않는다고 명시한다.

## 두 개의 함수형

핵심 결과는 gold reward $$R$$을 거리 $$d$$의 함수로 적은 두 개의 경험적 공식이다. best-of-n(BoN)에서는,

$$R_{bon}(d) = d(\alpha_{bon} - \beta_{bon} d)$$

강화학습(RL)에서는,

$$R_{RL}(d) = d(\alpha_{RL} - \beta_{RL} \log d)$$

기호를 하나씩 풀면:

- $$d$$: 앞서 정의한 $$\sqrt{\mathrm{KL}}$$. 정책이 초기 정책에서 얼마나 멀어졌는지를 나타내는 "거리".
- $$\alpha_{bon}, \alpha_{RL}$$: $$d$$가 작을 때 gold reward가 얼마나 빠르게 올라가는지를 결정하는 "초기 이득의 기울기". 값이 클수록 최적화 초반에 이득을 더 빨리 얻는다.
- $$\beta_{bon}, \beta_{RL}$$: 얼마나 빨리, 얼마나 세게 무너지는지를 결정하는 "붕괴 계수". 값이 작을수록 더 오래 버틴다.
- $$R(0) := 0$$: 정의상 최적화를 전혀 하지 않은 시점(거리 0)의 gold reward는 0(재중심화된 기준값)이다.

두 식 모두 형태는 "$$d$$ 곱하기 (초기 이득 항 빼기 붕괴 항)"이지만, 붕괴 항의 모양이 다르다. BoN은 $$d$$에 대해 순수 이차식이라 정점이 $$d^\ast = \alpha_{bon}/(2\beta_{bon})$$에서 정확히 결정되는 대칭 포물선이다. RL은 $$-\beta_{RL} d \log d$$ 항이 붙어 있어 정점 이후로 더 완만하게, 더 오래 버티다가 무너진다.

|                     | BoN                                                  | RL                                                                                      |
| ------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 함수형              | $$d(\alpha_{bon} - \beta_{bon} d)$$                  | $$d(\alpha_{RL} - \beta_{RL}\log d)$$                                                   |
| $$d\to 0$$ 근처     | 매끄러움(포물선)                                     | 기울기가 발산 — $$d$$가 아주 작을 때는 이 식이 성립하지 않음(저자들 스스로 명시한 한계) |
| 정점                | $$d^\ast = \alpha_{bon}/(2\beta_{bon})$$로 닫힌 형태 | 로그 방정식이라 닫힌 형태가 더 복잡함                                                   |
| 큰 $$d$$에서의 거동 | 이차로 급격히 감소                                   | $$d\log d$$로 더 완만하게 감소                                                          |

## 토이 예제: n을 KL로 바꿔보기

BoN의 KL 거리는 사람이나 시뮬레이션 없이 $$n$$만 알면 해석적으로 계산된다(Stiennon et al. 2020, Appendix G.3).

$$\mathrm{KL}_{bon}(n) = \log n - \frac{n-1}{n}$$

몇 개의 $$n$$ 값에 직접 대입해보자.

| $$n$$  | $$\log n$$ | $$(n-1)/n$$ | $$\mathrm{KL}_{bon}(n)$$ | $$d=\sqrt{\mathrm{KL}}$$ |
| ------ | ---------- | ----------- | ------------------------ | ------------------------ |
| 1      | 0          | 0           | 0                        | 0                        |
| 10     | 2.303      | 0.900       | 1.403                    | 1.184                    |
| 100    | 4.605      | 0.990       | 3.615                    | 1.902                    |
| 1,000  | 6.908      | 0.999       | 5.909                    | 2.431                    |
| 60,000 | 11.002     | 1.000       | 10.002                   | 3.163                    |

$$n=1000$$일 때 KL이 약 5.9(논문이 "약 6 나츠"라고 부르는 값)라는 점, $$n=60{,}000$$일 때 KL이 정확히 10.0 근처(논문이 "약 10 나츠"라고 부르는 값)라는 점이 계산과 정확히 일치한다. 저자들은 이 함수형을 $$n=1000$$까지의 데이터(KL ≈ 6 나츠)만 보고 가설로 세운 뒤, 그 가설을 한 번도 보지 못한 $$n=60{,}000$$(KL ≈ 10 나츠) 구간에서 검증했다 — 논문 표현으로 "진짜 사전 예측(true advance prediction)"이었고, 실측이 예측과 들어맞았다.

## BoN이 RL보다 KL을 아낀다

같은 KL 거리를 썼다고 해서 같은 "양의 최적화"를 한 게 아니다. BoN은 초기 정책 주변을 국지적으로만 탐색한다 — 동네 상점 몇 곳을 둘러보고 그중 최선을 고르는 것과 비슷하다. 그래서 $$\mathrm{KL}_{bon}$$은 대략 $$\log n$$으로 완만하게 늘어난다. 반면 RL은 매 스텝마다 정책 자체를 바꾼다 — 아예 다른 동네로 이사가는 것에 가깝다. KL penalty가 없으면 KL은 스텝 수에 대략 이차로 늘어난다.

그 결과 BoN의 실험 범위는 KL 10 나츠 안팎이었던 반면, RL은 KL 100 나츠까지 가야 비슷한 정도의 붕괴를 볼 수 있었다. "RL이 BoN보다 훨씬 KL 비효율적"이라는 뜻이고, 동시에 **KL 거리만으로 서로 다른 최적화 방법 사이의 "최적화 양"을 비교하면 안 된다**는 뜻이기도 하다. 대신 proxy reward 점수 자체를 공통 축으로 놓으면 BoN과 RL의 거동은 훨씬 비슷해진다 — RL이 초반에는 더 큰 proxy-gold 격차를 보이지만, 결국 BoN보다 더 높은 gold reward 정점에 도달한다.

## 알파와 베타를 Goodhart 유형으로 해석하기

앞서 정리한 4가지 Goodhart 유형이 이제 구체적인 항에 대응한다.

| 유형         | 대응하는 항                                  | 근거                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------ | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Regressional | $$\alpha$$ 항                                | proxy가 gold + 독립 노이즈라면, 최적화 파워는 신호와 노이즈의 분산 비율대로 나뉜다. proxy의 기울기와 gold의 선형 성분(=$$\alpha$$) 사이 차이가 곧 이 노이즈 선택 효과                                                                                                                                                                                                                                                                               |
| Extremal     | $$\beta$$ 항                                 | 최적화가 진행되며 표본이 RM 학습 분포 밖(OOD)으로 밀려나 proxy-gold 관계가 약해짐. 비단조성(꺾이는 현상)과 무한 손실의 주 원인으로, RM 크기가 커질수록 $$\beta$$가 매끄럽게 줄어드는 것은 모델이 더 견고해짐을 뜻한다. 논문은 정확히 이 자리에서 "답변 길이가 학습 분포에서는 품질과 상관관계가 있지만, OOD 구간에서는 더 이상 그렇지 않다"는 길이 편향 사례를 든다 — [11편](/blog/2026/rlhf-length-correlations/)이 이 현상 하나만 따로 정량화한다 |
| Causal       | 별도 항 없음, regressional과 유사하게 관측됨 | 공통 원인(예: 정보량)이 길이와 품질을 함께 밀어올릴 때, proxy가 그 상관관계를 길이 자체의 인과관계로 오인                                                                                                                                                                                                                                                                                                                                           |
| Adversarial  | 관측되지 않음                                | 정책이 proxy를 능동적으로 속이려면 그럴 만한 능력이 필요한데, 이 실험의 모델들은 아직 거기 못 미친다. 저자들은 미래에 모델이 더 강력해지면 이 스케일링 법칙 자체가 깨질 수 있다고 명시적으로 경고한다                                                                                                                                                                                                                                               |

# Experiments

## RM을 키우면 늦춰지지만 사라지지는 않는다

<p align="center"><img src="/assets/post/image/reward-model-overoptimization/fig1a_bon_rmsweep.png" width="70%"></p>
<p align="center"><img src="/assets/post/image/reward-model-overoptimization/fig1b_rl_rmsweep.png" width="70%"></p>

정책 크기를 1.2B로 고정하고 proxy RM 크기를 3M부터 3B까지 9단계(3M, 12M, 25M, 42M, 85M, 300M, 680M, 1.2B, 3B)로 바꿔가며 실험한 결과다(논문 Figure 1). 점선이 proxy reward, 실선이 gold reward다.

그림에서 두 극단만 읽어도 메시지는 분명하다.

| RM 크기  | BoN (KL 0→10)                                                                            | RL (KL 0→100)                                                                                             |
| -------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 3M(최소) | gold가 KL≈4에서 약 0.50으로 정점을 찍고, KL=10에서 약 0.28로 하락(정점 대비 약 44% 하락) | gold가 KL≈18~20에서 약 0.58로 정점을 찍고, KL=100 근처에서 거의 0으로 붕괴 — 얻은 이득을 사실상 전부 반납 |
| 3B(최대) | KL=10까지 정점 없이 계속 상승, 약 1.25에 도달. proxy(약 1.35)와의 격차도 작음            | gold가 KL≈70~90에서 약 1.08로 정점을 찍고, KL=100에서 약 1.0으로 완만하게만 하락                          |

RM을 3M에서 3B로 천 배 가까이 키우면 정점의 위치도 늦춰지고, 정점 이후 하락 폭도 훨씬 완만해진다. 하지만 RL 결과가 보여주듯 **가장 큰 3B RM조차 KL을 충분히 밀어붙이면 결국 하락한다**. RM을 키우는 건 Goodharting을 늦추는 것이지, 없애는 게 아니다.

계수 단위로 보면(논문 Figure 3), $$\alpha_{bon}$$과 $$\beta_{bon}$$은 RM 크기에 따라 매끄러운 로그 추세로 움직인다. RL 쪽은 흥미로운 비대칭이 있다 — $$\alpha_{RL}$$은 RM 크기와 거의 무관하게 일정한 반면, $$\beta_{RL}$$만 로그 추세로 꾸준히 줄어든다. 즉 **RM을 키운다고 "초반에 얻는 이득의 기울기"가 늘지는 않고, "얼마나 오래 안 무너지는지"만 늘어난다**는 뜻이다.

## 데이터 크기와 정책 크기: 비대칭적인 두 축

| 축                                  | 효과                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RM 데이터 크기 (RM 크기 12M 고정)   | 약 2,000건 미만에서는 near-chance 손실 대비 개선이 거의 없다. 이 문턱을 넘으면 데이터가 늘수록 gold 점수가 개선되고 goodharting도 줄지만, RM 크기 스케일링만큼 깔끔하게 정리되지는 않는다. 같은 데이터를 4 epoch 반복해도 개선이 없었지만, 1 epoch에 데이터를 4배로 늘리면 확실히 개선됐다 — 중요한 건 SGD 스텝 수가 아니라 서로 다른 데이터를 얼마나 봤는가다 |
| 정책 크기 (1.2B vs 6B, RM 12M 고정) | 더 큰 정책은 RM 최적화로 얻는 전체 이득은 작지만, 그렇다고 더 심하게 overoptimize하지도 않는다. gold 점수 정점이 거의 같은 KL 지점에서 나타나고, proxy-gold 격차도 두 정책 크기 사이에 거의 동일하다                                                                                                                                                           |

정책 크기 결과는 직관과 어긋난다. "더 큰 정책이 RM을 더 잘 파고들어 더 빨리, 더 심하게 망가질 것"이라 예상하기 쉽지만 실제로는 그렇지 않았다. 저자들은 이를 "RLHF를 초기 정책이라는 prior에서 출발하는 베이지안 추론으로 보는" Korbak et al.(2022)의 관점으로 설명한다 — 정책을 키우는 건 사람 시연 분포를 더 정확히 모델링하는 것일 뿐, RM 최적화 압력에 반응하는 방식 자체를 바꾸지는 않는다는 해석이다.

## KL 페널티: 2편의 그 항은 무엇을 하고 있었나

<p align="center"><img src="/assets/post/image/reward-model-overoptimization/fig9_kl_penalty.png" width="65%"></p>

정책과 RM 크기를 1.2B로 고정하고, RL 보상에 더하는 KL penalty 계수를 0, 0.01, 0.05, 0.1, 0.5로 바꿔가며 실험했다(논문 Figure 9). 점선이 proxy, 실선이 gold다.

그림에서 확인되는 사실이 이 절의 핵심이다. **penalty 계수와 무관하게 모든 실선(gold vs KL 관계)이 거의 같은 곡선 위에 겹친다.** penalty가 하는 일은 그 곡선을 바꾸는 게 아니라, 같은 학습 스텝 수에서 KL이 얼마나 빨리 커지는지를 조절하는 것뿐이다. penalty가 강할수록 정책이 초기 정책에서 멀어지는 속도가 느려지고, 그래서 학습이 끝났을 때 도달한 KL 지점이 더 작다 — 결과적으로 gold-KL frontier 위에서 더 일찍 멈춘 것과 같은 효과, 즉 **early stopping과 동등**하다.

이게 [2편 InstructGPT](/blog/2026/instructgpt/)에서 KL penalty 항이 목적함수에 들어간 이유를 사후적으로 설명해준다. 그 항은 "더 높은 proxy reward를 얻게 해주는 장치"가 아니라, "정책이 gold-KL frontier 위를 너무 멀리까지 걷지 못하게 막는 브레이크"였을 뿐이다. frontier 자체, 즉 주어진 KL에서 얻을 수 있는 gold reward의 상한을 바꾸지는 못한다. 참고로 PPO의 surrogate objective에는 이미 $$D_{KL}(\pi_{old} \parallel \pi)$$에 대한 암묵적 penalty가 내장되어 있는데, 이건 여기서 다루는 $$D_{KL}(\pi \parallel \pi_{init})$$과는 다른 대상이다 — 암묵적 penalty가 잘 튜닝되어 있으면 명시적 penalty보다 오히려 overoptimization을 덜 유발한다는 관찰도 저자들은 덧붙이지만, 왜 그런지는 설명하지 못한다고 인정한다.

## 반복 RLHF로 확장하면

[Bai et al.(2022)](/blog/2026/anthropic-hh-rlhf/)가 제안한 것처럼, RM을 한 번만 학습시키지 않고 최적화 라운드마다 새 사람 라벨로 갱신하는 "온라인" 방식이 overoptimization을 줄이는 데 쓰인다. $$\alpha_{RL}, \beta_{RL}$$이 라운드마다 일정하고 $$d$$가 라운드에 걸쳐 더해진다고 가정하면, $$k$$번의 라운드로 나눠 각 라운드가 거리 $$d/k$$씩 커버할 때 최종 gold reward는 다음과 같이 유도된다.

$$R_{RL}(d) = d\left(\alpha_{RL} - \beta_{RL}\log d + \beta_{RL}\log k\right)$$

라운드를 나누는 효과는 정확히 $$\beta_{RL} d \log k$$만큼 gold reward를 끌어올리는 것으로 요약된다. 흥미로운 건 이게 $$\alpha_{RL}$$ 항(regressional Goodharting)은 전혀 건드리지 못한다는 점이다 — 반복 라운드는 $$\beta$$ 항(extremal Goodharting)의 붕괴 속도만 로그 스케일로 늦출 뿐, 애초에 노이즈를 같이 선택하는 문제 자체는 해결하지 못한다. [1편](/blog/2026/deep-rl-human-preferences/)에서 본 "정책·라벨 수집·RM 학습이 비동기로 계속 맞물려 도는" 구조가, 5년 뒤 텍스트 RLHF에서는 "라운드 단위 반복"이라는 훨씬 거친 근사로 재현된 셈이다.

## 실험 하이퍼파라미터

| 항목                  | 값      |
| --------------------- | ------- |
| RM Adam LR multiplier | 1.67e-2 |
| RM batch size         | 64      |
| RL Adam LR multiplier | 4e-3    |
| RL batch size         | 256     |
| PPO clipping          | 0.2     |
| Timesteps per rollout | 256     |

# Conclusion

한 줄로 요약하면: **이 논문은 사람 라벨 대신 gold RM을 세우는 우회 실험으로, proxy reward를 최적화할 때 gold reward가 꺾이는 지점을 $$d=\sqrt{\mathrm{KL}}$$의 함수로 정확히 fit해냈고, BoN은 이차식으로 RL은 로그가 섞인 식으로 서로 다르게 무너진다는 것을 보였다.**

정리하면,

1. **왜 gold RM인가**: 사람 라벨로는 9종 RM 크기 × 여러 데이터 크기 × 2종 정책 크기의 조합을 촘촘한 KL 궤적으로 다 훑을 수 없다. InstructGPT 6B RM을 "가짜 인간"으로 세워 무제한·결정론적 라벨을 만드는 대신, "인간 의도 vs 라벨"의 괴리는 실험 범위 밖에 남긴다.
2. **두 함수형**: $$R_{bon}(d) = d(\alpha_{bon} - \beta_{bon}d)$$는 대칭 포물선, $$R_{RL}(d) = d(\alpha_{RL} - \beta_{RL}\log d)$$는 더 완만하게 무너진다. RL은 BoN보다 같은 KL 예산으로 훨씬 적게 최적화한다.
3. **계수의 의존성**: RM을 키우면 $$\beta$$(붕괴 계수)가 매끄럽게 줄어 정점이 늦춰지지만 $$\alpha$$(초기 이득)는 거의 그대로다. 데이터는 약 2,000건 문턱을 넘어야 의미가 생긴다. 정책 크기는 overoptimization 정도에 거의 영향을 주지 않는다.
4. **KL penalty의 정체**: gold-KL frontier 자체를 바꾸지 못하고, early stopping과 동등한 효과만 낸다 — 2편의 그 항이 왜 거기 있었는지에 대한 사후적 답이다.

남는 문제는 명확하다. $$\alpha$$ 항(regressional Goodharting)은 RM을 키워도 데이터를 늘려도 거의 줄지 않는다. $$\beta$$ 항(extremal Goodharting)은 줄일 수는 있지만 없앨 수는 없다. 그리고 이 논문이 다루지 못한 adversarial Goodharting은 모델이 더 강력해지면 이 스케일링 법칙 자체를 깨뜨릴 수 있다고 저자들 스스로 경고한다. [11편](/blog/2026/rlhf-length-correlations/)은 이 논문이 extremal Goodhart의 예시로 든 "길이 편향"을 정면으로 파고들어 성능 향상 중 얼마가 진짜고 얼마가 길이인지 정량화하고, [12편 ODIN](/blog/2026/odin-disentangled-reward/)은 그 길이 성분을 reward에서 아예 분리해내며, [13편 WARM](/blog/2026/warm-weight-averaged-reward/)은 여러 RM을 가중 평균해 $$\beta$$ 항 자체를 줄이는 또 다른 접근을 시도한다. [1편](/blog/2026/deep-rl-human-preferences/)의 Pong 무한 랠리는 결국 이 논문에서 하나의 곡선이 되었고, 그 곡선이 남긴 두 개의 미해결 항이 3부의 나머지 세 편을 채운다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열 번째 글이다.

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

10. **(현재 글)** Overoptimization Scaling Laws (2022) — Goodhart의 법칙 정량화
11. [Length Correlations in RLHF (2023)](/blog/2026/rlhf-length-correlations/) — 성능 향상의 얼마가 길이인가
12. [ODIN (2024)](/blog/2026/odin-disentangled-reward/) — 길이를 reward에서 분리
13. [WARM (2024)](/blog/2026/warm-weight-averaged-reward/) — weight averaging으로 hacking 방어

**4부. 안전성 정렬**

14. [Safe RLHF (2023)](/blog/2026/safe-rlhf/) — 안전성을 reward가 아니라 제약으로
15. [Rule-Based Rewards (2024)](/blog/2026/rule-based-rewards/) — 안전 규칙을 reward로 직접 번역

**5부. reward를 정책으로**

16. [PPO (2017)](/blog/2026/ppo/) — clipped surrogate objective
17. [Secrets of RLHF I (2023)](/blog/2026/secrets-rlhf-ppo/) — PPO 학습 안정화 트릭
18. [GRPO / DeepSeekMath (2024)](/blog/2026/grpo-deepseekmath/) — value network를 버리다
19. [RLOO (2024)](/blog/2026/rloo-back-to-basics/) — REINFORCE로 충분한가
20. [DPO (2023)](/blog/2026/dpo/) — reward를 없애면 어떻게 되는가
21. [SimPO (2024)](/blog/2026/simpo/) — reference-free + 길이 정규화
22. [KTO (2024)](/blog/2026/kto/) — 선호 쌍 없이 이진 신호만으로
23. [GSPO (2025)](/blog/2026/gspo/) — importance ratio를 시퀀스 단위로
24. [DAPO (2025)](/blog/2026/dapo/) — 신호 없는 프롬프트를 버린다

**6부. Process & Verifiable Reward**

25. [Let's Verify Step by Step (2023)](/blog/2026/lets-verify-step-by-step/) — 과정 감독이 결과 감독을 이긴다
26. [Math-Shepherd (2023)](/blog/2026/math-shepherd/) — 사람 라벨 없는 PRM
27. [DeepSeek-R1 (2025)](/blog/2026/deepseek-r1/) — RLVR, 규칙이 reward가 될 때

**7부. Generative Reward Model**

28. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
29. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
30. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
31. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
32. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**8부. 생각하는 Judge, 그리고 그 신뢰**

33. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
34. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
35. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
36. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
37. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

**9부. 실전 종합**

38. [프론티어 모델의 reward 설계 (2025~2026)](/blog/2026/frontier-reward-design/) — 열 개 모델이 실제로 택한 것
39. [reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/) — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 39편으로 구성된다.

# 참고 문헌

- Gao, Schulman, and Hilton, 2023. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). ICML 2023.
- [PMLR: Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html) — 공식 게재본(ICML 2023, PMLR vol. 202, pp. 10835-10866).
- [ar5iv: Scaling Laws for Reward Model Overoptimization (HTML rendering)](https://ar5iv.labs.arxiv.org/html/2210.10760) — 본문 수식·그림 원본.
- Ouyang et al., 2022. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). (gold RM의 출처, InstructGPT)
- Stiennon et al., 2020. [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325). (BoN의 KL 해석적 계산식 출처)
- Bai et al., 2022. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). (KL이 이차 거리 척도라는 근거, 온라인 RLHF)
- Manheim and Garrabrant, 2018. [Categorizing Variants of Goodhart's Law](https://arxiv.org/abs/1803.04585).
- Korbak, Perez, and Buckley, 2022. [RL with KL Penalties is Better Viewed as Bayesian Inference](https://arxiv.org/abs/2205.11275). EMNLP Findings 2022.
- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). (Pong 랠리와 로봇 손 사례)
