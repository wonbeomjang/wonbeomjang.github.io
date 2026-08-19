---
layout: post
title: "Rethinking Bradley-Terry: 왜 이 식으로 reward를 만드는가"
date: 2026-08-11 09:04:00 +0900
description: "RLHF Reward 설계 시리즈 #4 — BT 모델의 이론적 근거와 order consistency, 그리고 대안 목적함수"
categories: [paper]
tags: [rlhf, reward-model, bradley-terry, theory, alignment, paper]
giscus_comments: true
related_posts: true
---

> [Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives](https://arxiv.org/abs/2411.04991) (Sun et al., University of Cambridge, ICLR 2025)

# Introduction

[1편](/blog/2026/deep-rl-human-preferences/)과 [2편](/blog/2026/instructgpt/)은 둘 다 같은 식을 아무 의심 없이 썼다. 두 궤적(또는 두 응답)의 예측 보상을 지수함수에 태우고 소프트맥스를 취해 "사람이 어느 쪽을 고를 확률"을 만드는 식이다.

$$\hat P[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat r(\sigma^1)}{\exp \sum \hat r(\sigma^1) + \exp \sum \hat r(\sigma^2)}$$

이 식은 Bradley-Terry(BT) 모델(1952)이다. 그런데 이 식은 원래 **무엇을 위해 만들어졌을까.** Ralph Bradley와 Milton Terry가 1952년에 이 식을 제안했을 때 문제는 "리그전에서 여러 시즌 동안 반복 대결한 팀들의 실력을 매기는 법"이었다. 한 팀은 시즌 내내 수십 번 경기를 뛴다. 그런데 LLM reward modeling에서는 사정이 다르다. 프롬프트 $$x$$에 대한 응답 $$y_1, y_2$$ 한 쌍은 **딱 한 번** 비교되고 그걸로 끝이다. 같은 응답 쌍이 다시 비교되는 일은 거의 없다. 축구 리그 순위표를 매기던 도구를, 평생 단 한 번 마주친 두 사람의 매력을 비교하는 데 그대로 갖다 쓰는 셈이다. 이 전용(轉用)이 왜 아무 문제 없이 성립하는지는 지금까지 그 누구도 정식으로 따져보지 않았다.

Hao Sun(University of Cambridge), Yunyi Shen(MIT), Jean-Francois Ton(ByteDance Research)의 이 논문은 정확히 이 질문을 붙잡는다. 논문은 세 가지 질문을 던진다.

1. **플레이어 수가 비교 수보다 많은 상황(LLM 정렬의 전형적 상황)에서 BT 모델을 쓰는 게 이론적으로 정당한가?** 정당하다면 무엇이 그 성공을 뒷받침하는가?
2. **BT 말고 다른 선택지는 없는가?**
3. **관례적으로 같은 프롬프트의 응답끼리만 비교하는데, 다른 프롬프트의 응답끼리 비교하면 더 나은가?**

결론부터 미리 적는다. (1) BT는 임베딩 기반 신경망으로 구현할 때 실제로 수렴한다는 것을 이 논문이 최초로 증명했다 — 이론적으로 근거가 있다. 하지만 (2) 그 근거는 "BT만이 유일한 답"이라는 뜻이 아니다. reward 모델링에 정말 필요한 조건은 **order consistency**(순서 일관성)라는 훨씬 느슨한 성질이고, BT는 그 조건을 만족하는 여러 선택지 중 하나일 뿐이다. 저자들은 order consistency를 만족하는 **classification 기반 대안**을 제시하고, 6개 base LLM·2개 데이터셋·12,000개 이상의 실험 설정에서 이 대안이 BT를 이긴다는 걸 보인다. (3) 같은 프롬프트끼리만 비교하는 관행도 근거가 약하다 — 프롬프트를 섞어 비교하는 쪽이 이론적으로도 실증적으로도 낫다.

이 논문은 ICLR 2025에 **Oral**로 채택됐다(카메라레디 제목은 "Rethinking Reward Modeling in Preference-based Large Language Model Alignment"로 arXiv 버전과 살짝 다르다). 이 시리즈에서 이 글이 서 있는 위치는 명확하다 — 1편·2편이 전제로 깔았던 BT 손실의 **근거 자체**를 캐묻는 자리다.

# Background

## BT 모델과 Luce-Shephard 선택 규칙

BT 모델은 Luce-Shephard 선택 규칙의 특수 케이스다. 두 선택지 $$i, j$$ 중 $$i$$를 고를 확률이 각 선택지의 utility $$u(\cdot)$$에 비례한다고 가정하면,

$$P(i \succ j) = \frac{u(i)}{u(i) + u(j)} = \frac{\exp(r(i))}{\exp(r(i)) + \exp(r(j))} = \mathrm{softmax}(r(i), r(j))$$

가 나온다. 여기서 $$r(\cdot) = \log u(\cdot)$$는 log utility다. 이 식 자체는 낯설지 않다. 1편에서 이미 다뤘다. 이번 글이 파고드는 건 "이 식을 실제로 어떻게 쓰고 있는가"다.

## 두 가지 다른 용도 — 파라미터 추정 vs 예측

논문이 지적하는 첫 번째 사실은, **BT 모델의 고전적 용도와 reward modeling에서의 용도가 근본적으로 다르다**는 것이다. Chatbot Arena(Chiang et al., 2024)가 전형적인 고전적 용도다. 130개의 LLM을 플레이어로 놓고, 사람 투표를 경기 결과로 취급해 170만 건 이상의 비교를 모았다 — 모델 한 개당 평균 26,000경기다. 목표는 각 모델에 스칼라 점수 $$r(\cdot)$$ 하나를 매기는 것, 즉 **직접 파라미터 추정**이다. $$N$$명의 진짜 실력을 무작위 쌍대 비교로 추정하려면 이론적 하한이 $$\mathcal{O}(N\log N)$$ 번의 비교다(직관: 정렬 알고리즘 퀵소트가 평균 $$N\log N$$ 번의 비교로 끝나는 것과 같은 이유). 알려진 최선의 방법은 $$\mathcal{O}(N\log^3 N)$$ 비교면 충분하다(Han et al., 2020).

reward modeling은 이 조건을 하나도 만족하지 않는다. $$N$$개의 프롬프트-응답 쌍이 있다면 비교는 겨우 $$N/2$$건이다.

$$N/2$$가 어디서 나오는지는 한 줄이면 된다. 표준 선호쌍 데이터는 `(프롬프트, chosen, rejected)` 형태라 **비교 1건이 응답 2개를 소비한다.** 그러니 응답이 $$N$$개면 비교는 $$N/2$$건이고, 뒤집어 말하면 **응답 하나는 평생 딱 한 번만 비교에 등장한다.**

이 "1번"이라는 숫자가 문제의 전부다. 앞의 두 세팅을 플레이어 한 명 기준으로 환산해보자.

|                           | 플레이어 수  | 총 비교 수                 | **플레이어당 경기 수**      |
| ------------------------- | ------------ | -------------------------- | --------------------------- |
| Chatbot Arena             | 130개 모델   | 170만 건                   | 약 **26,000경기**           |
| BT reward modeling        | $$N$$개 응답 | $$N/2$$건                  | **1경기**                   |
| 이론적 하한이 요구하는 양 | $$N$$개      | $$\mathcal{O}(N\log N)$$건 | $$\mathcal{O}(\log N)$$경기 |

$$N=100{,}000$$이라면 하한은 응답 하나당 대략 열몇 경기를 요구하는데, 실제로 가진 것은 **1경기**다. 총량으로 봐도 $$N/2$$ 대 $$N\log N$$이니 $$2\log N$$배가 모자란다. 하한의 로그 배수는커녕 선형에도 못 미친다는 말이 이 뜻이다.

체스로 바꿔 말하면 이렇다. **10만 명이 참가한 대회에서 각자 딱 한 판만 두게 하고, 그 결과로 10만 명 전원의 레이팅을 매기라는 요구다.** 이긴 사람이 진 사람보다 세다는 것 말고는 아무것도 알 수 없다. 개인의 점수를 데이터만으로 확정하는 것은 원리적으로 불가능하다.

게다가 목표도 다르다. 학습에 쓰인 응답 쌍의 점수를 아는 것으로 끝나지 않고, **한 번도 본 적 없는 새 응답에 점수를 매겨야(예측)** 한다. 논문의 Table 1을 정리하면 다음과 같다.

| 항목           | LLM Arena                          | BT Reward Modeling         |
| -------------- | ---------------------------------- | -------------------------- |
| 목표           | 직접 파라미터 추정                 | 모델 파라미터화(함수 근사) |
| 예측 필요 여부 | 불필요                             | 필요                       |
| 비교 수        | 충분(N=130에 170만 건)             | 매우 희소(N/2건)           |
| BT 모델 종류   | 고전적 BT                          | BT-회귀(BT-regression)     |
| 요구조건       | 최소 $$\mathcal{O}(N\log N)$$ 비교 | covariate가 필요           |

비유하자면 이렇다. 챗봇 아레나는 시즌 내내 같은 팀들이 반복해서 맞붙는 **프로 축구 리그**다. 순위표는 실제 대결 결과가 충분히 쌓이면 저절로 안정된다. 반면 LLM reward modeling은 **평생 단 한 번 만난 소개팅 상대의 매력도**를 그 한 번의 만남만으로 추론하는 것과 같다. 재대결이 없으니 "그 사람 자체의 점수"를 직접 추정할 방법이 없다. 대신 쓸 수 있는 건 그 사람의 **특징(옷차림, 말투, 학력 같은 covariate)**뿐이고, 그 특징으로부터 매력도를 예측하는 함수를 배워야 한다. LLM에서 이 covariate 역할을 하는 게 바로 문장 임베딩이다.

## 사람 라벨은 어떤 확률 모델에서 나오는가

BT 손실은 "사람이 $$y_1$$을 고를 **확률**"을 예측한다. 그런데 이상하지 않은가. 두 응답 중 어느 쪽이 더 나은지가 정해져 있다면, 사람은 늘 그 쪽을 골라야 하는 것 아닌가. **왜 사람의 선택이 확률적인가?**

논문은 이 확률이 어디서 나오는지를 두 가정으로 분해한다.

**가정 1 — 진짜 품질은 흔들리지 않는다.** 응답 $$(x,y)$$에는 oracle utility $$r_{x,y}$$가 존재하고, 이 값은 **결정론적**이다. 즉 "이 답이 저 답보다 낫다"는 사실 자체는 고정되어 있다.

**가정 2 — 그런데 사람은 그 값을 직접 보지 못한다.** annotator $$A$$가 실제로 보는 것은 진짜 값에 **자기 개인의 편향** $$b(x,y,A)$$가 얹힌 값이다. 그리고 그 얹힌 값끼리 비교해 결정론적으로 고른다.

$$\mathbb{1}(y_1 \succ y_2 \mid x, A) = \mathbb{1}\bigl(r_{x,y_1} + b(x,y_1,A) > r_{x,y_2} + b(x,y_2,A)\bigr)$$

$$\mathbb{1}(\cdot)$$은 안의 조건이 참이면 1, 거짓이면 0인 indicator function이다. 이 식이 말하는 것은 **개별 annotator는 동전을 던지지 않는다**는 것이다. 그 사람은 자기가 보는 값을 기준으로 확실하게 고른다.

그러면 확률은 어디서 오는가. **누가 라벨링을 하느냐**에서 온다. 간결한 답을 좋아하는 사람과 자세한 답을 좋아하는 사람에게 같은 두 응답을 주면 서로 다른 답을 고른다. 라벨러 풀에서 한 명을 무작위로 뽑는 순간 **그 사람의 편향도 함께 뽑히고**, 그래서 바깥에서 보면 결과가 확률적으로 보인다.

이제 식을 정리해보자. $$\Delta r = r_{x,y_1} - r_{x,y_2}$$(진짜 품질 차이), $$\Delta b = b(x,y_2,A) - b(x,y_1,A)$$(편향 차이)로 두면

$$P(y_1 \succ y_2 \mid x) = P(\Delta b < \Delta r) = F_{\Delta b}(\Delta r)$$

가 된다. 즉 **선호 확률은 "편향 차이의 누적분포함수(CDF)에 진짜 품질 차이를 넣은 값"** 그 이상도 이하도 아니다. 그러니 CDF를 무엇으로 고르냐가 곧 모델을 무엇으로 고르냐다.

| 편향 차이 $$\Delta b$$의 분포 가정 | 그 분포의 CDF | 나오는 모델                       |
| ---------------------------------- | ------------- | --------------------------------- |
| 표준 로지스틱 분포 (가정 3)        | $$\sigma$$    | **Bradley-Terry**                 |
| 표준 정규분포 (가정 4)             | $$\Phi$$      | **Thurstonian** (Thurstone, 1927) |

여기서 BT 공식의 시그모이드가 어디서 왔는지가 드러난다. **시그모이드는 로지스틱 분포의 CDF다.** 우연히 생긴 편의상의 함수가 아니라, "편향 차이가 로지스틱 분포를 따른다"는 가정의 직접적인 결과다. 정규분포를 가정하면 자리에 $$\Phi$$가 들어와 다음 식이 된다.

$$P(y_1 \succ y_2 \mid x) = \Phi(r_{x,y_1} - r_{x,y_2})$$

**토이 예제**로 차이를 체감해보자. 두 응답의 참 품질 차이가 $$\Delta r = 1.0$$이라 하자.

| 노이즈 가정                | 식              | 계산             | $$P(y_1 \succ y_2)$$ |
| -------------------------- | --------------- | ---------------- | -------------------- |
| 로지스틱 차이(BT)          | $$\sigma(1.0)$$ | $$1/(1+e^{-1})$$ | 약 0.731             |
| 가우시안 차이(Thurstonian) | $$\Phi(1.0)$$   | 표준정규 CDF     | 약 0.841             |

같은 품질 차이 1.0인데 한쪽은 73%, 다른 쪽은 84%다. 차이가 나는 이유도 직관적이다. 표준 로지스틱 분포는 표준정규분포보다 **퍼짐이 크다**(표준편차 약 1.81 대 1.0). 편향이 더 널뛴다고 가정한 셈이니, 같은 품질 차이라도 "그 정도로는 확실하지 않다"고 보는 것이다. 반대로 편향이 얌전하다고 가정하면 같은 차이가 더 확실한 승부로 읽힌다.

정리하면 — **BT는 유일한 정답이 아니라, 사람의 판단 노이즈에 대한 분포 가정 하나를 고른 결과다.** 로지스틱을 골랐으니 시그모이드가 나왔을 뿐, 정규분포를 골랐으면 $$\Phi$$가 나왔을 것이다. 이 사실이 뒤에서 다룰 "BT는 선택이다"라는 주장의 첫 단서다.

# Method

## BT-회귀: 희소 비교를 우회하는 법

앞 절에서 막다른 길에 도달했다. 응답 하나가 평생 1경기만 뛰므로, **그 응답의 점수를 데이터에서 직접 확정할 방법이 없다.** 그럼 어떻게 해야 하나.

Springall(1973)의 답은 문제를 바꾸는 것이다. **응답마다 점수를 하나씩 두려 하지 말고, "특징을 넣으면 점수가 나오는 함수" 하나를 배우자.**

$$r(\text{응답}) \;=\; f\bigl(\text{그 응답의 covariate}\bigr)$$

이 한 줄이 두 가지를 동시에 푼다.

|                          | 응답마다 점수를 두는 방식  | covariate 함수 방식          |
| ------------------------ | -------------------------- | ---------------------------- |
| 배워야 할 것             | 점수 $$N$$개 (응답 수만큼) | 함수 $$f$$ 하나              |
| 비교 1건이 기여하는 대상 | 그 비교에 낀 응답 2개뿐    | **함수 $$f$$ 전체**          |
| 처음 보는 응답           | 점수를 알 방법이 없음      | covariate만 있으면 계산 가능 |

핵심은 가운데 행이다. 원래 방식에서는 비교 하나가 응답 두 개의 점수에만 정보를 준다. 그래서 "1경기"라는 예산이 치명적이었다. 그런데 함수를 배우는 방식에서는 **모든 비교가 같은 함수 $$f$$를 향해 쌓인다.** 응답 A와 B의 비교에서 배운 것이 나중에 응답 Z를 채점할 때도 쓰인다. 응답들이 더 이상 남남이 아니라 하나의 함수를 공유하기 때문이다.

앞 절의 소개팅 비유로 돌아가면 이렇다. 그 사람을 다시 만날 수는 없지만, **"어떤 특징을 가진 사람이 매력적인가"라는 감각**은 다른 소개팅에서 얻은 경험으로도 다듬을 수 있다. 그 감각이 $$f$$다.

LLM에서 covariate 역할을 하는 것은 프롬프트-응답 쌍 $$(x,y)$$의 문장 임베딩 $$\Psi(x,y) \in [0,1]^d$$다. 그리고 $$f$$는 MLP로 근사한다. 즉 $$\Psi(x,y) \mapsto \hat r(\Psi(x,y))$$다.

## 그러면 학습은 그냥 이진 분류가 된다

이렇게 놓고 나면 BT 학습의 정체가 드러난다. 응답 두 개가 주어지면 먼저 각각의 점수를 뽑고,

$$\hat r_1 = f(\Psi_1), \qquad \hat r_2 = f(\Psi_2)$$

이 둘에 softmax를 취한다. 그게 곧 "어느 쪽이 이길 확률"이다.

$$P(y_1 \succ y_2) = \mathrm{softmax}(\hat r_1, \hat r_2)_1 = \frac{e^{\hat r_1}}{e^{\hat r_1} + e^{\hat r_2}} = \sigma(\hat r_1 - \hat r_2)$$

**이건 클래스가 2개인 분류기와 정확히 같은 모양이다.** 입력은 응답 쌍, 출력은 "1번이 이긴다 / 2번이 이긴다"의 확률, 손실은 cross-entropy. BT 손실이라는 특별한 것을 쓰는 게 아니라, 우리가 아는 그 이진 분류 손실 그대로다.

다만 보통의 분류기와 다른 성질이 하나 붙는다. **anti-symmetric**(반대칭) 구조다. 두 입력의 순서를 바꾸면 답이 정확히 뒤집힌다.

$$P(y_2 \succ y_1) = \sigma(\hat r_2 - \hat r_1) = 1 - \sigma(\hat r_1 - \hat r_2) = 1 - P(y_1 \succ y_2)$$

예측이 **차이 $$\hat r_1 - \hat r_2$$에만 의존**하므로, 순서를 바꾸면 부호만 뒤집혀 이 성질이 공짜로 따라온다. 두 응답을 그냥 이어붙여 분류기에 넣는 방식이었다면 "A vs B"와 "B vs A"의 예측이 어긋날 수 있는데, 점수를 각각 매긴 뒤 빼는 구조는 그럴 여지가 없다.

**그런데 여기에 구멍이 있었다.** 오늘날 거의 모든 reward model이 이 방식(임베딩 + MLP + BT 손실)으로 학습된다. 그런데 **이 조합이 데이터가 늘어날수록 참값에 수렴한다는 증명은 아무도 한 적이 없었다.** 고전 BT 이론은 "선수 한 명이 수십 경기를 뛰는" 세팅을 다루지, "선수마다 1경기씩만 뛰되 모두가 하나의 함수를 공유하는" 세팅을 다루지 않기 때문이다. 잘 되는 건 알겠는데 왜 되는지는 모르는 상태였던 것이다. 다음 절이 이 구멍을 메운다.

## MLP 기반 BT reward model의 수렴률

이 논문의 첫 번째 핵심 기여가 여기서 나온다. Bos and Schmidt-Hieber(2022)의 truncated KL risk 프레임워크를 빌려, 저자들은 embedding 기반 MLP reward model이 실제로 참 확률(그리고 참 보상 차이)에 수렴한다는 것을 증명한다.

$$\phi_n := 2^{\frac{(1+\alpha)\beta + (3+\alpha)d}{(1+\alpha)\beta + d}} \, n^{-\frac{(1+\alpha)\beta}{(1+\alpha)\beta + d}}$$

식이 험해 보이지만 **실제로 볼 것은 $$n$$의 지수 하나뿐**이다. 앞의 $$2^{(\cdots)}$$ 덩어리는 $$n$$과 무관한 상수라 데이터를 늘려도 변하지 않는다. 즉 이 식의 뼈대는 이것이다.

$$\phi_n \;\propto\; n^{-\gamma}, \qquad \gamma = \frac{(1+\alpha)\beta}{(1+\alpha)\beta + d}$$

오차가 $$n^{-\gamma}$$로 줄어든다는 뜻이고, **$$\gamma$$가 클수록 빨리 줄어든다.** 그러니 $$\gamma$$를 키우거나 깎는 게 무엇인지만 보면 된다. 등장하는 기호는 넷이다.

| 기호       | 정체                      | 쉽게 말하면                                    | $$\gamma$$에 미치는 영향      |
| ---------- | ------------------------- | ---------------------------------------------- | ----------------------------- |
| $$n$$      | 선호 비교 데이터 개수     | 라벨을 몇 건이나 모았나                        | 많을수록 오차 ↓ (수렴의 원천) |
| $$\beta$$  | 참 reward 함수의 매끄러움 | 비슷한 응답에 비슷한 점수가 매겨지는가         | ↑ → $$\gamma$$ ↑ (빨라짐)     |
| $$d$$      | 임베딩 공간의 차원        | 응답 하나를 몇 개의 숫자로 표현하나            | ↑ → $$\gamma$$ ↓ (느려짐)     |
| $$\alpha$$ | 마진 조건                 | 애매한 비교(50:50에 가까운 쌍)가 얼마나 드문가 | ↑ → $$\gamma$$ ↑ (빨라짐)     |

$$\beta$$부터 보자. reward 함수가 **매끄럽다**는 건 임베딩 공간에서 가까운 두 응답이 비슷한 점수를 받는다는 뜻이다. 그래야 "A를 보고 배운 것"이 그 옆에 있는 B에도 통한다. 반대로 조금만 달라져도 점수가 널뛰는 함수라면 응답 하나하나를 따로 배워야 하니 데이터가 훨씬 많이 든다.

$$d$$는 정반대로 작용한다. 차원이 높을수록 같은 개수의 데이터가 공간에 더 성기게 흩어지므로, 새 응답 주변에 참고할 이웃이 없다. 이게 $$d$$가 분모에 그대로 들어가는 이유이고, 익숙한 **차원의 저주(curse of dimensionality)** 패턴이다.

$$\gamma$$가 실제로 얼마나 차이를 만드는지 숫자로 보자. $$\beta=1, \alpha=0$$으로 고정하고 차원만 바꾼다.

| 임베딩 차원 $$d$$ | $$\gamma = 1/(1+d)$$ | 오차를 **절반**으로 줄이려면 |
| ----------------- | -------------------- | ---------------------------- |
| 2                 | 약 0.33              | 데이터 **8배**               |
| 10                | 약 0.09              | 데이터 약 **2,000배**        |
| 50                | 약 0.02              | 데이터 약 **$$10^{15}$$배**  |

오차가 $$n^{-\gamma}$$이니 절반으로 줄이려면 $$n$$을 $$2^{1/\gamma}$$배 해야 한다. 차원이 조금만 올라가도 필요한 데이터가 폭발한다. 실제 LLM 임베딩은 수천 차원이므로, 이 정리를 액면 그대로 받으면 절망적으로 보인다. 다만 이건 최악의 경우를 가정한 상한(upper bound)이고, 실제 임베딩은 훨씬 낮은 차원의 구조 위에 놓여 있다고 보는 게 보통이다.

**Theorem 6**(정리, informal)은 적당한 매끄러움·정규성 가정 아래, MLP reward model의 truncated KL risk가

$$R_B(\boldsymbol p_0, \hat{\boldsymbol p}) \le C' B \phi_n L \log^2(n) \to 0$$

으로 0에 수렴함을 보인다. 여기서 $$\boldsymbol p_0$$는 참 선호확률, $$\hat{\boldsymbol p}$$는 모델의 예측 확률, $$L$$은 MLP의 깊이, $$B$$와 $$C'$$는 상수다. truncated KL risk는 두 확률분포가 얼마나 다른지를 재는 값이라고 보면 된다.

결국 이 부등식이 말하는 바는 하나다. **데이터가 늘어날수록($$n \to \infty$$) 모델의 예측 확률이 참 확률에 다가간다.** 앞 절에서 "잘 되는 건 알겠는데 왜 되는지 모른다"고 했던 그 구멍이 여기서 메워진다. 임베딩 + MLP + BT 손실이라는 조합에 처음으로 수렴 보장이 붙은 것이다.

## 확률 오차가 reward 오차로 번질 때 생기는 함정

**Corollary 7**은 이 확률 수렴 결과를 실제로 쓰고 싶은 보상 차이 오차로 옮긴다.

$$\lvert r(\Psi_1) - r(\Psi_2) - (\hat r(\Psi_1) - \hat r(\Psi_2)) \rvert \lesssim \frac{\lvert \sqrt{p_0} + \sqrt{\hat p} \rvert}{\tilde p (1 - \tilde p)} \sqrt{\phi_n L} \log(n) \to 0$$

왜 이 변환이 필요한가부터 짚자. Theorem 6이 보장한 것은 **확률** $$\hat p$$가 참값에 가까워진다는 것이다. 그런데 우리가 정작 쓰고 싶은 건 **reward 점수 차이**다. best-of-N으로 응답을 고를 때도, PPO의 보상으로 쓸 때도 필요한 건 확률이 아니라 점수다. 그러니 "확률이 정확하다"를 "점수 차이가 정확하다"로 옮겨야 하는데, **그 환산 과정에서 오차가 증폭된다.**

기호부터 정리하면, $$r(\Psi_1) - r(\Psi_2)$$가 참 점수 차이, $$\hat r(\Psi_1) - \hat r(\Psi_2)$$가 모델이 매긴 점수 차이이므로 좌변은 그 둘의 오차다. 우변의 $$\tilde p$$는 $$p_0$$(참 확률)와 $$\hat p$$(예측 확률) 사이 어딘가의 값이다.

핵심은 분모 $$\tilde p(1-\tilde p)$$다. 이 항은 $$\tilde p = 0.5$$에서 최대(0.25)이고, 0이나 1로 갈수록 급격히 작아진다. 분모가 작아지면 전체 식이 커지니, **확률 오차가 점수 오차로 번질 때 곱해지는 증폭 배율**인 셈이다.

왜 하필 이 모양인가. 확률과 점수 차이는 시그모이드로 연결되어 있고($$p = \sigma(\Delta r)$$), 그걸 거꾸로 풀면 $$\Delta r = \log\frac{p}{1-p}$$다. 이 역함수의 기울기가 정확히 $$1/[p(1-p)]$$다. 즉 **시그모이드가 평평한 구간일수록 역방향 환산이 불안정해진다.**

말로 하면 이렇다. 확률이 0.99 근처라는 건 시그모이드의 꼬리 부분이라는 뜻인데, 거기서는 점수 차이가 크게 달라져도 확률이 거의 안 변한다.

$$\sigma(3) = 0.953, \qquad \sigma(5) = 0.993, \qquad \sigma(7) = 0.999$$

점수 차이가 3에서 7로 **두 배 넘게** 벌어지는 동안 확률은 0.95에서 0.999로 겨우 움직인다. 뒤집어 말하면, 확률을 0.99로 정확히 맞혔다 해도 **그게 점수 차이 4인지 6인지는 알 길이 없다.** 반대로 확률이 0.5 근처면 시그모이드가 가장 가파르므로 확률 하나로 점수 차이가 좁게 특정된다.

**토이 예제**로 이 증폭이 얼마나 심한지 보자. $$1/[\tilde p(1-\tilde p)]$$ 값을 $$\tilde p$$별로 계산하면:

| $$\tilde p$$        | $$\tilde p(1-\tilde p)$$ | $$1/[\tilde p(1-\tilde p)]$$ |
| ------------------- | ------------------------ | ---------------------------- |
| 0.50 (거의 동률)    | 0.250                    | 4                            |
| 0.90 (꽤 확실)      | 0.090                    | 약 11                        |
| 0.99 (거의 확실)    | 0.0099                   | 약 101                       |
| 0.999 (사실상 확정) | 0.000999                 | 약 1,001                     |

동률에 가까운 비교(4배)와 사실상 결과가 정해진 비교(1,001배) 사이에 **250배** 차이가 난다. 결론은 명확하다. **오차 보장이 의미를 가지려면 비교하는 두 응답의 보상이 서로 가까워야 한다.** 압도적으로 한쪽이 나은 쌍을 비교 데이터로 아무리 많이 넣어도, 그 학습 신호가 실제 reward 오차를 줄인다는 보장은 약해진다. 이건 뒤에서 다룰 cross-prompt 비교의 한계를 이해하는 데도 쓰인다.

## Order Consistency: reward 모델링에 정말 필요한 조건

여기서부터 논문의 두 번째 핵심 주장이 시작된다. BT 손실은 사람의 선택 **확률**을 정확히 맞히는 걸 목표로 한다. 그런데 downstream에서 reward model을 쓰는 방식(best-of-N 샘플링, PPO 등)은 확률값 자체가 아니라 **"어느 응답이 더 나은가"라는 순서**만 있으면 충분하다. 순서가 같다면 $$\hat r = h(r)$$처럼 단조증가함수 $$h$$로 재조정된 reward라도 최적화 결과는 똑같다.

**Definition 8(Order Consistency)**은 이 요구조건을 정식화한다. 서로 다른 두 프롬프트-응답 쌍에 대해,

$$(\hat r(x_1,y_1) - \hat r(x_2,y_2)) \cdot (r(x_1,y_1) - r(x_2,y_2)) > 0$$

이면 $$\hat r$$은 order consistent다. 그리고 사람 라벨 자체도 노이즈가 있다는 걸 반영해, 학습 가능한 관측 손실 $$\mathcal{L}_{oc}$$을 최소화하면 참 오라클 순서와도 높은 확률로 일치한다는 게 **Proposition 9**의 내용이다.

비유하자면 이렇다. 대학 입시에서 중요한 건 "정확히 몇 점을 맞았는가"가 아니라 "합격선 위인가 아래인가, 몇 등인가"다. 원점수를 표준점수로, 표준점수를 등급으로 바꿔도 등수만 유지되면 입시 결과는 똑같다. reward model도 마찬가지다 — **확률을 정밀하게 맞히는 것과 순서를 맞히는 것은 다른 요구조건이고, 후자만으로 충분하다.**

## BT는 정답이 아니라 하나의 선택이다

BT 모델은 order consistency를 만족하는 방식 중 하나다. $$h=1$$(첫 응답 선호)일 확률을 $$\sigma(\hat r_{\text{BT}}(x_1,y_1) - \hat r_{\text{BT}}(x_2,y_2))$$로 모델링하고 교차 엔트로피로 학습하면,

$$\mathcal{L}_{\text{BT}} = \mathbb{E}\left[\mathbb{1}_{h=1}\sigma(\hat r_{\text{BT}}^1 - \hat r_{\text{BT}}^2) + \mathbb{1}_{h=-1}(1 - \sigma(\hat r_{\text{BT}}^1 - \hat r_{\text{BT}}^2))\right]$$

이 손실은 **비교 순서를 뒤집으면 예측도 정확히 뒤집히도록** 강제하는 비대칭(anti-symmetric) 구조를 갖는다. 이 구조 때문에 BT는 반드시 Siamese 네트워크(같은 파라미터로 두 입력을 각각 통과시키고 차이를 비교하는 구조)로 구현해야 하고, 사실상 MLP만 백본으로 쓸 수 있다.

## 비대칭 제약을 느슨하게 풀면 — Classification 대안

order consistency만 원한다면 이 anti-symmetry가 정말 필수일까?

여기서 한 번 짚고 갈 것이 있다. **order consistency는 확률을 맞히라고 요구하지 않는다.** 순서만 맞으면 된다. 그런데 BT는 "이 쌍에서 사람이 첫 응답을 고를 확률은 0.73"까지 맞히려 든다. 목표보다 훨씬 많은 것을 요구하고 있는 셈이다.

그래서 저자들은 요구를 낮춘다. **쌍이라는 구조를 아예 버리고, 응답 하나하나를 따로 놓고 "이건 좋은 응답인가, 나쁜 응답인가"만 맞히면 어떨까?**

주의할 점은 이것이 "둘 중 어느 쪽이 나은가"를 맞히는 분류가 아니라는 것이다. 선호쌍 $$(y_1 \succ y_2)$$를 학습 데이터 **두 건**으로 쪼갠다.

| 원래 데이터                     | 쪼갠 뒤                                |
| ------------------------------- | -------------------------------------- |
| $$(x, y_1 \succ y_2)$$ 비교 1건 | $$(x, y_1) \to$$ 라벨 **+1**(chosen)   |
|                                 | $$(x, y_2) \to$$ 라벨 **−1**(rejected) |

이렇게 하면 각 프롬프트-응답이 독립적인 분류 예제가 된다. 형식적으로는 두 응답에 대해 따로 $$+1/-1$$을 예측하는 모델 $$\hat H := (\hat H_1, \hat H_2)$$를 두는 것이고, BT가 강제하던 $$\hat H_1 = -\hat H_2$$ 제약은 걸지 않는다. 데이터가 충분하면 $$\hat H_1 \approx h$$, $$\hat H_2 \approx -h$$가 저절로 학습된다는 게 저자들의 논지다.

**그런데 순서가 보장되나?** 보장된다. 논리는 단순하다. $$y_1$$을 "+1"로 맞히고 $$y_2$$를 "−1"로 맞혔다면, 두 응답의 순서는 자동으로 맞다. 순서가 틀리려면 **둘 중 적어도 하나를 잘못 분류해야만** 한다. 그러니 순서를 틀릴 확률은 두 분류 오류 확률의 합을 넘지 못한다. 이게 union bound이고, 식으로 쓰면 이렇다.

$$\mathcal{L}_{\text{oc}} \le \mathcal{L}_{\text{clf}} := \mathbb{E}(h = \hat H_{\text{clf}}(x_1,y_1)) + \mathbb{E}(-h = \hat H_{\text{clf}}(x_2,y_2))$$

즉 $$\mathcal{L}_{\text{clf}}$$는 order consistency 손실의 **upper bound**다. 분류 오류를 줄이면 순서 오류는 그 아래로 눌린다. 목표를 직접 최적화하지 않고 **더 다루기 쉬운 상한을 대신 최적화**하는, 머신러닝에서 흔한 수법이다.

얻는 것은 자유다. 페어를 묶어 Siamese로 흘릴 필요가 없으니 **기성 이진 분류기를 아무거나 쓸 수 있다** — MLP든 LightGBM이든. 학습이 끝나면 분류기의 로짓(logit)을 reward 프록시로 쓴다. 로짓은 "좋은 응답일 확신도"이므로 클수록 좋은 응답이고, 그 순서가 곧 reward 순서가 된다.

대신 포기하는 것도 분명하다. **로짓은 선호 확률로 보정(calibrate)되어 있지 않다.** 순서를 쓰는 용도(best-of-N 재랭킹 등)에는 문제없지만, "사람이 이걸 고를 확률이 얼마인가"를 그대로 읽어야 하는 용도에는 쓸 수 없다.

### 그런데 "좋은 것들" 사이의 순서는 어떻게 나오나

여기서 자연스러운 의문이 생긴다. RL이나 best-of-N에서는 한 프롬프트에 대해 응답을 여러 개 뽑아 **줄을 세워야** 한다. 그런데 분류기는 좋다/나쁘다 두 갈래로만 나눈다. **"좋음"으로 분류된 것들끼리는 무슨 근거로 순위를 매기나?**

먼저 짚을 것은, 실제로 쓰는 값이 하드 라벨이 아니라 **로짓**이라는 점이다. 분류기는 $$P(\text{chosen} \mid x, y)$$를 예측하고, 그 연속값이 곧 reward다. 그러니 "좋음" 안에서도 값은 갈린다.

그럼 그 값이 무엇을 재고 있는가. 학습 데이터에서 응답 $$y$$가 "+1"이 되는 것은 **자기 상대를 이겼을 때**다. 따라서 이상적인 분류기가 수렴하는 값은 이렇게 쓸 수 있다.

$$P(\text{chosen} \mid x, y) = \mathbb{E}_{y' \sim q(\cdot \mid x)}\bigl[P(y \succ y')\bigr] = \mathbb{E}_{y'}\bigl[\sigma(r(y) - r(y'))\bigr]$$

여기서 $$q(\cdot \mid x)$$는 그 프롬프트에서 상대로 등장하는 응답들의 분포다. 즉 로짓은 **"이 응답이 임의의 상대를 이길 확률", 일종의 승률**이다. 그리고 이 값은 $$r(y)$$에 대해 **단조증가**한다 — 참 점수가 높으면 승률도 높다. 단조 변환이므로 **순서가 보존된다.** 이것이 "좋은 것들 사이의 순서"가 나오는 근거다.

**다만 조건이 하나 붙는다.** 위 식의 기댓값이 $$q(\cdot \mid x)$$에 의존한다는 점이다. 상대가 누구였느냐에 따라 같은 "+1"도 의미가 달라진다.

| 응답    | 어떻게 chosen이 됐나            | 진짜 실력 |
| ------- | ------------------------------- | --------- |
| $$y_a$$ | 형편없는 상대를 이겨서          | 보통      |
| $$y_b$$ | 막상막하인 상대를 간신히 이겨서 | 뛰어남    |

둘 다 라벨은 "+1"이고, 분류기는 상대가 누구였는지 모른다. 그래서 $$y_b$$가 실제로 더 좋은데도 $$y_a$$가 더 높은 로짓을 받을 수 있다. 앞서 본 union bound $$\mathcal{L}_{\text{oc}} \le \mathcal{L}_{\text{clf}}$$도 이 경우를 덮지 못한다. 그 부등식이 보장하는 것은 **데이터에 실제로 등장한 쌍**의 순서이지, 서로 붙어본 적 없는 두 chosen 응답의 상대 순서가 아니다.

실무에서 그럭저럭 굴러가는 이유는 사용 방식에 있다. RL이나 best-of-N은 보통 **하나의 프롬프트에 대해 한 정책이 뽑은 $$K$$개**를 정렬한다. 이때는 $$q(\cdot \mid x)$$가 사실상 같으므로 위 기댓값이 공통 기준이 되고, 로짓 순서가 곧 reward 순서가 된다. **같은 프롬프트 안에서는 비교적 안전하다.**

위험한 것은 **프롬프트를 가로질러** 점수를 비교할 때다. 여러 프롬프트의 응답을 한 배치에 섞어 advantage를 계산하는 식이면 $$q$$가 달라져 기준이 어긋난다. 바로 다음 절의 cross-prompt 논의가 이 문제를 정면으로 다루고, RM 점수를 프롬프트 간에 비교해도 되는가라는 질문은 [#9 RewardBench 2](/blog/2026/rewardbench-2/)에서 평가 방법론의 문제로 다시 나온다.

### 결국 둘 다 "스칼라 하나로 점수 매기기"다

여기까지 오면 한 가지가 분명해진다. **추론 시점에는 두 방식이 완전히 같다.** 둘 다 $$(x,y)$$ 하나를 넣어 스칼라 하나를 받고 그걸 reward로 쓴다. 아키텍처도 똑같이 "임베딩 → 스칼라 헤드"다. 다른 것은 **학습할 때 그 스칼라를 무엇과 비교하느냐**뿐이다.

|                      | BT                                           | Classification             |
| -------------------- | -------------------------------------------- | -------------------------- |
| 추론                 | 스칼라 하나 → reward                         | **동일**                   |
| 손실이 비교하는 대상 | $$\sigma(\hat r_1 - \hat r_2)$$ vs 선호 라벨 | $$\sigma(g)$$ vs $$\pm 1$$ |
| 기준점               | **상대 응답**(상대적)                        | **고정 라벨**(절대적)      |

그리고 앞에서 본 장단점이 전부 이 한 줄에서 파생된다. BT는 스칼라를 **짝의 스칼라와의 차이로만** 다루므로 절대 위치가 의미 없고(shift-invariant), 대신 짝을 묶어야 하니 Siamese 구조에 갇힌다.

Classification은 **"$$y$$가 $$y'$$를 이겼다"는 상대적 사실을 "$$y$$는 +1"이라는 절대 라벨로 납작하게** 만든다. 그래서 짝을 풀 수 있고 아무 분류기나 쓸 수 있게 되는데, **바로 그 납작하게 만드는 과정에서 "누구를 이겼는지"가 버려진다.** 방금 본 상대 분포 문제가 여기서 나온다. 편의와 약점이 같은 뿌리인 셈이다.

|                            | BT (BT-MLP)         | Classification (CLF)              |
| -------------------------- | ------------------- | --------------------------------- |
| 강제하는 구조              | anti-symmetry(엄격) | 없음(union bound로 근사)          |
| 필요한 데이터 형태         | 페어(Siamese 입력)  | 개별 프롬프트-응답 (분류 라벨)    |
| 쓸 수 있는 모델            | 사실상 MLP만        | MLP, LightGBM 등 기성 분류기 전부 |
| order consistency와의 관계 | 정확히 만족         | 상한(upper bound)으로 만족        |

두 방식 다 order consistency라는 같은 목표의 서로 다른 구현이다. BT가 "필연"이 아니라는 이 논문 전체의 주장이 이 비교표 한 줄로 요약된다.

## Cross-prompt 비교가 이론적으로 더 낫다

관례적으로 선호 annotation은 같은 프롬프트에서 나온 두 응답끼리만 비교한다. 저자들은 이 관례에도 근거가 약하다고 지적한다. 두 응답의 utility가 가우시안 $$\mathcal{N}(\mu_x, \sigma_x^2)$$을 따른다고 하면, annotation quality를 "노이즈가 있는 상황에서 평균적으로 올바른 순서를 맞힐 확률" $$\mathcal{Q}_{\text{pair}}(x) = \mathbb{E}[\sigma(\beta \lvert r(x,y_1) - r(x,y_2) \rvert)]$$로 정의할 수 있다.

**토이 예제**로 annotator 판별력 $$\beta$$와 응답 다양성 $$\sigma_x$$의 곱 $$\beta^2\sigma_x^2$$을 바꿔가며 계산한 결과가 논문에 실려 있다.

| $$\beta^2 \sigma_x^2$$ | $$\mathcal{Q}_{\text{pair}}$$(정답률) |
| ---------------------- | ------------------------------------- |
| 1                      | 약 0.6749                             |
| 2                      | 약 0.7251                             |
| 4                      | 약 0.7781                             |
| 10                     | 약 0.8428                             |

같은 annotator라도(같은 $$\beta$$) 응답들의 분산 $$\sigma_x^2$$이 클수록 정답률이 67.5%에서 84.3%까지 올라간다. 즉 **annotation 품질은 annotator의 실력과 응답 간 편차, 둘 다에 달려 있다.** 그런데 같은 프롬프트에서 나온 두 응답은 같은 LLM이 만든 것이라 서로 비슷하기 쉽다 — 편차가 작다. 반면 서로 다른 프롬프트의 응답을 무작위로 짝지으면 편차가 구조적으로 커진다. **Proposition 10**과 **Theorem 11**은 이를 정식화해, unimodal하고 대칭인 효용 분포에서는 cross-prompt 비교가 same-prompt 비교보다 기대 reward 차이가 항상 크거나 같다는 것을 증명한다.

$$\mathbb{E}_x \mathbb{E}_{y_1,y_2 \mid x}[\lvert r_{x,y_1} - r_{x,y_2} \rvert] \le \mathbb{E}_{x_1,x_2} \mathbb{E}_{y_1 \mid x_1, y_2 \mid x_2}[\lvert r_{x_1,y_1} - r_{x_2,y_2} \rvert]$$

비유하자면 이렇다. 같은 반 1등과 2등의 시험 점수를 비교하면 종이 한 장 차이라 우열을 가리기 어렵다. 반면 전교생 중 아무나 두 명을 무작위로 뽑아 비교하면 점수 차가 훨씬 크게 벌어지기 마련이라 우열이 뚜렷하다. Cross-prompt 비교는 "무작위로 아무나 뽑아 비교하기"에 해당한다.

# Experiments

## 실험 설계 — 12,000개 이상의 조합

저자들은 재현성·통제 가능성·계산 효율을 우선해 실험을 설계했다. PPO 대신 **Best-of-N(BoN) 샘플링**으로 reward model을 평가한다 — PPO는 설정마다 LLM을 새로 파인튜닝해야 해서 12,000개 조합을 전부 도는 건 계산상 불가능하기 때문이다. 실험 규모는 다음 여섯 개 축의 곱이다.

| 축                         | 값                                                                    |
| -------------------------- | --------------------------------------------------------------------- |
| base LLM                   | Gemma2b, Gemma7b, LLaMA3-8b(및 각 SFT 버전) 총 6개                    |
| 데이터셋                   | Anthropic-Harmless, Anthropic-Helpful                                 |
| 응답 샘플링 방법           | 3가지                                                                 |
| annotation 노이즈 수준     | 6단계($$\beta \in \{0.5, 0.7, 1.0, 3.0, 5.0, 10.0\}$$, 오답률 5%~38%) |
| reward model 구현체        | BT-MLP, CLF-MLP, CLF-LGB(LightGBM) 3종                                |
| annotation 가용량 시나리오 | 4단계(5,000 / 10,000 / 20,000 / 40,000건)                             |
| random seed                | 5개                                                                   |

이 조합을 전부 곱하면 약 12,960건이고, 논문은 "12,000개 이상의 실험 설정"으로 보고한다.

## BT vs Classification — 정면 대결

<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig1_harmless.png" width="85%"></p>
<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig1_helpful.png" width="85%"></p>

BoN N=500에서 base model 대비 golden reward 개선폭을 6개 base 모델 × 2개 데이터셋에서 비교한 결과다(논문 Figure 1). Harmless에서는 BT-MLP와 CLF 계열이 대체로 비슷하거나 CLF가 근소 우위였지만(예: LLaMA3-8b에서 BT 약 0.10 vs CLF 약 0.18~0.20), Helpful에서는 차이가 훨씬 극적이다.

| Base 모델(Helpful) | BT-MLP  | CLF-MLP | CLF-LGB |
| ------------------ | ------- | ------- | ------- |
| Gemma2b            | 약 1.77 | 약 2.98 | 약 2.87 |
| Gemma2b-SFT        | 약 0.13 | 약 1.51 | 약 1.75 |
| Gemma7b            | 약 2.08 | 약 3.17 | 약 3.03 |
| Gemma7b-SFT        | 약 0.48 | 약 1.86 | 약 2.01 |

Gemma2b-SFT에서는 BT가 사실상 붕괴(약 0.13)한 반면 classification 계열은 10배 넘는 개선(1.51~1.75)을 보였다. 저자들의 결론은 명확하다 — **classification reward model이 BT보다 일반적으로 더 나은 성능을 내면서, MLP에 갇히지 않고 LightGBM 같은 기성 분류기도 자유롭게 쓸 수 있다는 유연성까지 더 갖췄다.**

## Annotation 품질·양이 바뀌면 순위가 바뀐다

<p align="center"><img src="/assets/post/image/bradley-terry-rethinking/fig3_quantity.png" width="95%"></p>

annotation 개수를 5,000건에서 40,000건까지 늘려가며 본 결과(논문 Figure 3)다. classification 계열이 대체로 BT보다 높거나 같은 golden reward를 유지하며, 데이터가 늘어날수록 개선폭도 더 안정적으로 커진다.

annotation 품질(노이즈 수준 $$\beta$$)을 바꾼 실험에서는 흥미로운 교차가 나온다.

| 조건                                           | 우세한 방식                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| 노이즈가 낮음(오답률 10% 미만, $$\beta$$가 큼) | BT가 classification보다 근소 우세                    |
| 노이즈가 높음(오답률 10% 이상)                 | classification이 BT보다 견고 — 성능 하락폭이 더 작음 |
| annotation 개수 증가                           | classification이 일관되게 우세, 개선폭도 더 안정적   |

즉 **정답률이 아주 높은 소량의 고품질 annotation만 있다면 BT도 충분히 경쟁력 있지만, 실전처럼 노이즈가 섞이고 대량 annotation을 다루는 상황에서는 classification이 더 안전한 선택**이라는 것이다.

## Cross-prompt 비교의 실증 효과

이론이 예측한 대로, cross-prompt annotation은 same-prompt annotation보다 일관되게 BoN 성능을 높였다(2개 데이터셋 × 6개 base 모델 × 3개 reward model 구현체 전체에서). 저자들은 이 효과의 원인을 파고들기 위해 두 개의 합성 세팅을 추가로 설계했다.

| 세팅                   | 응답 쌍 구성                                                                        | cross-prompt 효과                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Similar Comparison     | 한 프롬프트당 생성된 10개 응답 중 golden reward 순위 중간 2개를 짝지음(다양성 최소) | same-prompt 비교는 유용한 reward model을 거의 못 만듦 — cross-prompt가 압도적으로 우세 |
| Diversified Comparison | 같은 10개 응답 중 최상위·최하위를 짝지음(다양성 최대)                               | cross-prompt의 이점이 크게 줄어듦(부정적 영향은 없음)                                  |

그리고 pairwise annotation의 평균 절대 보상 차이(응답 간 다양성의 대리 지표)와 cross-prompt 개선폭 사이에는 강한 상관관계가 있었다(논문 Figure 6). 특히 무작위로 응답 두 개를 뽑는 실제 상황(Random)의 평균 보상 차이가 인위적으로 다양성을 낮춘 Similar 세팅과 비슷한 수준이라는 관찰이 중요하다 — **실전에서 같은 프롬프트로 뽑은 두 응답은 생각보다 서로 비슷해서, cross-prompt 비교를 적용할 유인이 실제로도 크다**는 뜻이다.

# Conclusion

한 줄로 요약하면: **BT 모델은 임베딩 기반 신경망으로 구현할 때 실제로 참 보상에 수렴한다는 것이 이 논문에서 처음 증명됐지만, 그 증명이 "BT가 유일한 답"이라는 뜻은 아니다. reward 모델링에 정말 필요한 조건은 order consistency뿐이고, 그 조건을 만족하는 classification 기반 대안이 12,000개 이상의 실험에서 오히려 BT를 능가했다.**

정리하면,

1. **이론**: BT 모델을 LLM reward modeling에 쓰는 관행은 고전적 BT 세팅(반복 대결, 파라미터 직접 추정)과 다르다(희소 비교, 새 데이터 예측). 그럼에도 임베딩 기반 MLP-BT reward model이 참 확률·참 보상 차이에 수렴한다는 것을 truncated KL risk bound로 증명했다(Theorem 6, Corollary 7).
2. **대안**: reward 모델링의 진짜 목표는 order consistency이며, BT의 anti-symmetry 제약은 그걸 만족하는 여러 방법 중 하나일 뿐이다. anti-symmetry를 느슨하게 풀면 기성 이진 분류기(MLP, LightGBM)를 그대로 쓰는 classification reward model을 얻는다 — 이 손실이 order consistency 손실의 upper bound라는 것도 증명됐다(Proposition 9, Eq. 22).
3. **annotation 설계**: 같은 프롬프트끼리만 비교하는 관행에도 이론적 근거가 없다. cross-prompt 비교가 기대 보상 차이를 구조적으로 키워 annotation 품질을 높인다(Theorem 11)는 것이 이론과 실증 양쪽에서 확인됐다.

이 논문의 결론 — **BT는 필연이 아니라 하나의 선택지이며, "필요한 건 order consistency뿐"** — 은 이 시리즈의 다음 두 흐름을 정당화하는 근거가 된다. [7편 ArmoRM](/blog/2026/armorm/)이 스칼라 하나 대신 다목적 reward로 분해하는 것도, [6부의 GenRM](/blog/2026/generative-verifiers/)이 아예 reward를 확률 스칼라가 아니라 텍스트 생성으로 바꾸는 것도, 결국 "reward 모델링이 반드시 BT-스타일 스칼라 확률 모델일 필요는 없다"는 이 글의 결론 위에서 성립한다. 다음 글([5편 Secrets of RLHF II](/blog/2026/secrets-rlhf-reward-modeling/))은 이 order-consistent 목적함수들이 실제 노이즈 섞인 선호 데이터 앞에서 어떻게 무너지고 일반화되는지를 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 네 번째 글이다.

**1부. 지형도**

<ol start="1">
  <li><a href="/blog/2026/deep-rl-human-preferences/">Deep RL from Human Preferences (Christiano 2017)</a> — 선호로 보상을 배우는 원형</li>
  <li><a href="/blog/2026/instructgpt/">InstructGPT (Ouyang 2022)</a> — RLHF 3단계 표준 레시피</li>
  <li><a href="/blog/2026/anthropic-hh-rlhf/">HH-RLHF (Bai 2022)</a> — helpful·harmless preference model</li>
</ol>

**2부. 스칼라 RM 해부**

<ol start="4">
  <li><strong>(현재 글)</strong> Rethinking Bradley-Terry (2024) — reward 변환의 수학적 기반</li>
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
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열한 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 44편으로 구성된다.

# 참고 문헌

- Sun, Shen, Ton, 2024/2025. [Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives](https://arxiv.org/abs/2411.04991). arXiv:2411.04991 (ICLR 2025, Oral).
- [ar5iv/arXiv HTML: Rethinking Bradley-Terry Models...](https://arxiv.org/html/2411.04991v2) — 본문 수식·그림 원본.
- [ICLR 2025 Proceedings: Rethinking Reward Modeling in Preference-based Large Language Model Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7423902b5534e2b267438c85444a54b1-Abstract-Conference.html) — 카메라레디 버전(제목 변경).
- [GitHub: holarissun/RewardModelingBeyondBradleyTerry](https://github.com/holarissun/RewardModelingBeyondBradleyTerry) — 공식 구현.
- Bradley, R. A. and Terry, M. E., 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika. (BT 모델 원 논문)
- Chiang et al., 2024. [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132). (LLM Arena 비교 규모 인용원)
- Bos, T. and Schmidt-Hieber, J., 2022. Convergence rates for non-parametric classification with generalized quadratic loss. (truncated KL risk 프레임워크)
- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS 2017. (BT 손실을 RLHF에 처음 적용)
