---
layout: post
title: "Skywork-Reward: 데이터 큐레이션이 아키텍처를 이긴다"
date: 2026-08-11 09:06:00 +0900
description: "RLHF Reward 설계 시리즈 #6 — 80K 선별 데이터로 SOTA를 찍은 레시피, 그리고 V2의 40M 확장"
categories: [paper]
tags: [rlhf, reward-model, data-curation, skywork, paper]
giscus_comments: true
related_posts: true
---

> [Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451) (Liu et al., Skywork AI, Tech Report 2024)

# Introduction

[#5 Secrets of RLHF II 글](/blog/2026/secrets-rlhf-reward-modeling/)은 문제를 정확히 진단했다. 선호 데이터에는 애매하거나 틀린 쌍이 섞여 있고, 그 노이즈가 RM의 판별력을 갉아먹으며, RM은 학습 분포를 벗어난 프롬프트에 취약하다. 그 논문의 해법은 알고리즘 쪽이었다 — 여러 RM의 투표로 선호 강도를 재고, contrastive loss로 판별력을 올리고, meta-learning으로 OOD 적응력을 붙였다. 전부 **학습 방식을 바꿔서 노이즈를 견디는** 접근이다.

이 글이 다루는 Skywork-Reward는 정반대 방향에서 같은 문제에 답한다. 노이즈를 견디는 손실 함수를 설계하는 대신, **애초에 노이즈가 들어오지 못하게 데이터를 고른다.** 결과는 극단적이다. 기존 표준이던 70만 쌍짜리 Preference 700K로 학습한 RM은 RewardBench 평균 86.9\~88.1점에 머물렀는데, 저자들이 그 700K의 소스 7개를 다시 골라 **8만 쌍(80K)**으로 줄인 데이터로 학습한 RM은 92.5\~93.8점을 찍었다. 데이터를 8.75배 줄였는데 점수는 오히려 5~7점 올랐다. 모델 크기도, 아키텍처도 건드리지 않았다. 바뀐 건 오직 **무엇을 학습시킬지 고르는 방법**뿐이다.

이 글은 세 가지를 순서대로 따라간다.

1. **어떻게 골랐나**: 378K 원본에서 80K로 줄이는 필터링 파이프라인의 각 단계와 그 설계 의도.
2. **무엇으로 학습시켰나**: Bradley-Terry와 7가지 대안 손실 함수를 정면으로 비교한 실증 결과, 그리고 [#4 Rethinking Bradley-Terry 글](/blog/2026/bradley-terry-rethinking/)이 던진 이론적 질문과의 접점.
3. **얼마나 더 갈 수 있나**: 1년 뒤 나온 후속작 Skywork-Reward-V2가 이 레시피를 4천만 쌍 규모로 밀어붙인 결과.

그리고 마지막에는 이 모든 RewardBench 점수 경쟁이 갖는 한계 — 벤치마크 자체가 포화됐다는 문제를 짚고, [#9 RewardBench 2 글](/blog/2026/rewardbench-2/)로 넘긴다.

# Background

## "Bag of Tricks"라는 이름이 뜻하는 것

이 논문 제목의 "Bag of Tricks"는 겸손한 표현이 아니라 장르 선언에 가깝다. 새로운 아키텍처나 새로운 손실 함수를 제안하는 논문이 아니라, **기존 요소들을 어떻게 조합·선별하면 실전에서 잘 먹히는지**를 실증적으로 정리한 테크니컬 리포트다(2024년 10월 공개, 학회 게재 없이 arXiv 테크니컬 리포트로 남아 있다). 이런 계열의 논문은 이미지 분류에서도 있었다 — 같은 모델을 그대로 두고 데이터 증강·학습률 스케줄 같은 자잘한 트릭만 바꿔 정확도를 크게 올린 사례들이다. Skywork-Reward는 그 전통을 RM 학습에 그대로 옮긴다. 다만 이 논문에서 가장 큰 트릭 하나가 나머지 전부를 압도한다 — **데이터 선별**이다.

## RewardBench 700K 시대

2024년 상반기까지 오픈소스 RM 학습의 표준 레시피는 여러 선호 데이터셋을 있는 대로 모아 붙이는 것이었다. 대표적인 것이 Dong et al.(2024)의 **Preference 700K**로, HH-RLHF·SHP·UltraFeedback 등 다양한 소스를 합쳐 약 70만 쌍을 구성한다. "데이터는 많을수록 좋다"는 암묵적 전제가 있었다. Skywork-Reward는 이 전제를 정면으로 반박하는 사례를 만든다.

# Method

## 7개 데이터셋, 37.8만 쌍에서 시작한다

Skywork-Reward-Preference-80K는 처음부터 80K로 수집된 게 아니라, 7개 공개 선호 데이터셋을 합친 약 **378K(Preference 378K)**에서 걸러낸 결과다.

| 데이터셋                   | 원본 쌍 수  | 라벨 소스                                      |
| -------------------------- | ----------- | ---------------------------------------------- |
| Magpie Pro (Llama 3.1)     | 98,000      | Llama 3.1 70B Instruct 생성 + ArmoRM 채점      |
| Magpie Pro (Llama 3)       | 98,000      | Llama 3 70B Instruct 생성 + ArmoRM 채점        |
| Magpie Air                 | 98,000      | Llama 3 8B Instruct 생성 + ArmoRM 채점         |
| Magpie Ultra               | 50,000      | Llama 3.1 405B Instruct 생성 + ArmoRM 채점     |
| WildGuardMix               | 18,548      | 8개 LLM 응답 + 사람 라벨(유해/양호, 거부/이행) |
| OffsetBias                 | 8,504       | GPT-3.5/GPT-4/Claude 3 Opus 응답, GPT-4 채점   |
| HelpSteer2                 | 7,221       | 사람 + 6개 LLM, helpfulness 점수               |
| **합계 (Preference 378K)** | **378,273** | —                                              |

이 중 Magpie 계열(Pro·Air·Ultra)이 전체의 약 93%를 차지한다. 문제는 여기서 시작한다. 나머지 6개 데이터셋이 아무리 품질이 좋아도, 학습 배치에서 압도적 비중을 차지하는 Magpie가 그 영향력을 희석해버린다(dilution effect). 그래서 저자들은 Magpie와 WildGuardMix 두 곳에 집중적으로 필터링을 건다.

<p align="center"><img src="/assets/post/image/skywork-reward/data-composition-378k-to-80k.png" width="95%"></p>

위 그림이 필터링 전후 구성 비율을 보여준다. 왼쪽(378K)에서 Magpie 세 갈래가 각 25.9%씩 균등하게 눌러앉아 있던 것이, 오른쪽(80K)에서는 Magpie_Pro_Llama3.1(가장 강한 모델이 생성)이 36.2%로 오히려 비중이 늘고, WildGuardMix·OffsetBias·HelpSteer2 같은 상대적으로 소량이지만 신뢰도 높은 데이터의 비중이 커진다. 비유하자면 냉장고에 재료가 아무리 많아도 절반이 유통기한을 넘겼다면, 신선한 재료 몇 가지만 골라 요리하는 편이 낫다 — 이 그림은 정확히 그 재배열을 숫자로 보여준다.

## 필터링 전략 각론 — 왜 그렇게 골랐는가

### Magpie: 강한 모델을 우선하되, 점수의 편향을 보정한다

Magpie 계열은 각 프롬프트당 5개 응답이 딸려 있고, 각 응답에 ArmoRM 점수가 매겨져 있다. 저자들은 5개 중 최고점을 선택(chosen), 최저점을 거부(rejected)로 삼는다. 여기까지는 자연스럽다. 문제는 그다음이다 — **어떤 모델이 생성한 데이터를 우선할 것인가**다.

직관적으로는 더 강한 모델(Llama 3.1 70B)이 만든 응답이 더 좋은 학습 신호를 준다고 기대할 만하다. 그런데 실제 ArmoRM 점수를 확인해보니, Llama 3 8B(Air)가 만든 응답이 Llama 3/3.1 70B(Pro)가 만든 응답보다 오히려 더 높은 점수를 받는 역전 현상이 나타났다. 저자들은 이를 채점 모델(ArmoRM) 자체의 편향 — 강한 모델의 응답 분포가 ArmoRM 학습 분포에서 벗어나 있어 저평가되는 현상 — 으로 해석한다. 그래서 원래 점수를 그대로 쓰지 않고 **Air 점수에서 0.1, Pro(Llama 3) 점수에서 0.05를 일괄로 빼서** Pro(Llama 3.1) > Pro(Llama 3) > Air 순서가 되도록 인위적으로 재정렬한다. "최적의 보정값을 더 탐색하지는 않았다"고 저자들 스스로 인정하는, 다소 거친 수작업 보정이지만 실제로 효과가 있었다.

두 번째 축은 **작업 카테고리(task category)**다. Magpie 데이터는 수학·코딩·일상 대화 등으로 카테고리가 나뉘어 있는데, 저자들은 수학과 코딩 카테고리에서는 각각 상위 30%만, 나머지 카테고리에서는 상위 10%만 남긴다. 왜 수학·코딩만 비율을 높였을까. 이 두 도메인은 정답 유무가 명확해 ArmoRM 점수의 신뢰도가 상대적으로 높고, 동시에 RM이 취약하기로 악명 높은 영역(추론 능력 평가, RewardBench의 Reasoning 카테고리)이기 때문이다. 실제로 최종 선별된 Magpie 쌍 중 수학이 49.81%를 차지할 만큼 압도적으로 편중됐다.

### WildGuardMix: 이미 아는 문제는 다시 풀지 않는다

WildGuardMix는 유해/양호 프롬프트와 거부/이행 응답의 조합으로 선호 쌍을 만든다. 유해한 프롬프트에는 거부가 선호, 양호한 프롬프트에는 이행이 선호가 된다("폭탄 만드는 법"에는 거부가 이기고, "케이크 만드는 법"에는 이행이 이긴다). 여기에 **2단계 필터링**을 건다.

1. **1단계**: 다른 데이터로 먼저 학습시킨 초기 RM을 WildGuardMix의 비-적대적(non-adversarial) 쌍에 돌려봤더니 이미 거의 다 맞혔다. 더 학습시켜도 얻을 게 없다는 뜻이므로, 이 쌍들은 통째로 제외하고 **적대적(adversarial) 쌍**에만 집중한다.
2. **2단계**: 그런데 적대적 쌍 전부를 넣고 학습시키자, 안전성 점수는 오르는데 그 대가로 일반 선호 판별력이 더 크게 떨어지는 트레이드오프가 나타났다. 그래서 **이전 버전 RM이 이미 정답을 맞혔던 적대적 쌍만** 남기는 재조정을 거친다.

이건 시험 대비에 비유하면 이해가 쉽다. 이미 백 점 맞는 문제 유형을 계속 풀리는 과외 선생님은 없다. 오답노트에서도, 너무 어려워서 아예 손도 못 대는 문제보다는 **한 끗 차이로 틀리는 문제**에 집중하는 편이 점수를 가장 효율적으로 올린다. WildGuardMix 필터링이 정확히 이 논리를 따른다.

### HelpSteer2: 단순하지만 명확한 기준

HelpSteer2는 원 논문의 방법론을 그대로 따라, chosen 응답의 helpfulness 점수가 rejected보다 높은 쌍만 사용한다. 세 데이터셋 중 가장 손을 적게 댄 경우다.

### 이 레시피는 다른 데이터셋에 옮겨지지 않는다

세 처방을 나란히 놓으면 공통점이 하나 보인다. **전부 그 데이터셋에 이미 붙어 있던 메타데이터에 기대고 있다.**

| 데이터셋     | 필터링이 의존하는 메타데이터           | 없으면?                     |
| ------------ | -------------------------------------- | --------------------------- |
| Magpie       | 응답별 ArmoRM 점수, 작업 카테고리 라벨 | 상위 30%/10%를 고를 수 없음 |
| WildGuardMix | adversarial 플래그, 유해/양호 라벨     | 1단계 분리 자체가 불가능    |
| HelpSteer2   | helpfulness 점수                       | 기준이 사라짐               |

새로 수집한 선호 데이터에는 이런 것들이 대체로 없다. 그러면 이 레시피 중 **적용 가능한 게 하나도 남지 않는다.** 게다가 Magpie의 보정값 0.1과 0.05는 저자들이 "더 탐색하지 않았다"고 인정한 수작업 상수이고, WildGuardMix 2단계는 **이미 쓸 만한 RM이 있어야** 돌릴 수 있다. 데이터를 고르려면 RM이 필요하고 RM을 만들려면 데이터가 필요한, 살짝 순환적인 구조다.

그러니 제목의 "Bag of Tricks"는 정확한 자기 규정이다. **일반적인 큐레이션 방법론이 아니라, 이 7개 데이터셋에 대해 통한 처방 모음**이다. 그 사실이 이 논문의 가치를 깎지는 않는다 — "데이터 선별이 아키텍처보다 중요하다"는 명제를 숫자로 못 박은 것만으로 충분하다. 다만 다음 질문이 자연스럽게 따라온다. **사람이 데이터셋마다 규칙을 손으로 짜지 않고도 같은 일을 할 수 있을까?**

1년 뒤 나온 후속작 Skywork-Reward-V2가 정확히 이 질문에 답한다.

## 손실 함수 비교 — Bradley-Terry는 왜 여전히 이기는가

데이터를 고른 다음 질문은 "무엇으로 학습시키는가"다. [#1 Christiano 2017](/blog/2026/deep-rl-human-preferences/)부터 이어져 온 표준은 Bradley-Terry(BT) 손실이다.

$$\mathcal{L}_{\text{BT}} = -\log\sigma(r_\theta(x,y_c) - r_\theta(x,y_r))$$

$$r_\theta(x,y_c)$$는 chosen 응답에 매긴 보상, $$r_\theta(x,y_r)$$는 rejected 응답에 매긴 보상, $$\sigma$$는 시그모이드다. 이 식은 두 보상의 차이가 클수록 손실이 0에 가까워지고, 역전되면 손실이 커진다.

[#4 Rethinking Bradley-Terry 글](/blog/2026/bradley-terry-rethinking/)이 던지는 질문은 이거다 — RM이 하는 일은 결국 "chosen이 rejected보다 순위가 높다"는 것만 보존하면 되므로, 참 보상의 단조 변환(monotonic transformation)이기만 하면 어떤 파라미터화든 이론적으로는 문제없다. 즉 BT가 **유일하게 필요한 선택은 아니다.** Skywork-Reward는 이 질문에 정면으로 실증 데이터를 던진다 — BT 대신 쓸 수 있는 대안 7가지를 전부 같은 데이터, 같은 모델(Gemma-2-27B)로 학습시켜 비교했다.

$$\Delta = r_\theta(x,y_c) - r_\theta(x,y_r)$$로 표기하면 대안들은 다음과 같다.

| 손실 함수       | 식                                                                                                    | 의도                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Focal           | $$\mathcal{L}_{\text{Focal}} = -\log\sigma(\Delta)\cdot(1-\sigma(\Delta))^\gamma$$                    | 이미 잘 분리된 쉬운 쌍의 손실을 죽이고 헷갈리는 쌍에 집중 |
| Focal+Penalty   | $$\mathcal{L}_{\text{FP}} = -\left(1-2\max(\sigma(\Delta)-0.5,\ 0)\right)^\gamma \log\sigma(\Delta)$$ | Focal에 확신 구간 페널티 추가                             |
| Hinge           | $$\mathcal{L}_{\text{Hinge}} = \max(0,\ m-\Delta)$$                                                   | 마진 $$m$$ 이상 벌어지면 손실 0, SVM식 마진 최대화        |
| Margin MSE      | $$\mathcal{L}_{\text{MMSE}} = (r_\theta(x,y_c)-(r_\theta(x,y_r)+m))^2$$                               | 확률이 아니라 보상 차이 자체를 회귀                       |
| Cross-Entropy   | $$\mathcal{L}_{\text{CE}} = -[\log\sigma(r_\theta(x,y_c))+\log(1-\sigma(r_\theta(x,y_r)))]$$          | 각 응답을 독립 이진 분류로 취급(쌍 비교 구조를 버림)      |
| Tempered Log BT | $$\mathcal{L}_{\text{Temp-Log}} = -\frac{1}{1-t}\left[(\sigma(\Delta))^{1-t}-1\right]$$               | $$t$$로 극단 확신에 대한 민감도 조절                      |
| Temperature BT  | $$\mathcal{L}_{\text{Temp}} = -\log\sigma(\Delta / T)$$                                               | $$T$$로 시그모이드 기울기(확신 강도) 조절                 |

### 토이 예제: Focal loss가 실제로 무엇을 하는가

식만 보면 감이 안 오니 숫자로 확인해보자. $$\gamma=2$$로 두고, 세 가지 $$\Delta$$ 값에서 BT 손실과 Focal 손실을 직접 계산한다.

| $$\Delta$$ | 상황                     | $$\sigma(\Delta)$$ | BT 손실   | Focal 손실($$\gamma=2$$) | 축소 배율 |
| ---------- | ------------------------ | ------------------ | --------- | ------------------------ | --------- |
| $$3.0$$    | 이미 확실히 맞힌 쉬운 쌍 | $$0.953$$          | $$0.049$$ | $$0.0001$$               | 약 445배  |
| $$0.0$$    | 반반, 애매한 쌍          | $$0.500$$          | $$0.693$$ | $$0.173$$                | 4배       |
| $$-1.0$$   | 모델이 틀린 쌍           | $$0.269$$          | $$1.313$$ | $$0.702$$                | 약 1.9배  |

같은 배율로 줄이는 게 아니라, **이미 잘 맞힌 쉬운 쌍일수록 훨씬 세게 죽인다.** 결과적으로 그래디언트는 헷갈리는 쌍·틀린 쌍에 훨씬 많이 실린다 — 시험에서 이미 백 점 맞는 유형은 아예 채점에서 가중치를 낮추고, 틀린 문제 위주로 재시험을 보는 셈이다.

### 결과: BT가 이긴다, 다만 "종합"에서

| 손실 함수         | Avg      | Chat     | Chat Hard | Safety   | Reasoning |
| ----------------- | -------- | -------- | --------- | -------- | --------- |
| **Bradley-Terry** | **93.8** | 95.8     | **91.4**  | 92.0     | 96.1      |
| Temperature BT    | 93.7     | 94.3     | 91.7      | 92.7     | 96.3      |
| Focal             | 93.6     | 94.3     | 91.8      | 92.0     | 96.5      |
| Focal+Penalty     | 93.4     | 93.9     | 91.5      | 92.0     | 96.5      |
| Hinge             | 93.3     | 94.1     | 90.2      | 92.6     | 96.3      |
| Tempered Log BT   | 92.9     | **96.4** | 87.4      | 91.8     | 96.2      |
| Margin MSE        | 92.3     | 90.2     | 89.0      | **93.3** | 96.7      |
| Cross-Entropy     | 87.6     | 74.9     | 87.3      | 94.0     | 94.5      |

숫자를 자세히 보면 흥미로운 그림이 나온다. Margin MSE는 Safety(93.3)에서 가장 강하고, Tempered Log BT는 Chat(96.4)에서 가장 강하다. **어느 대안도 전 영역에서 BT를 이기지 못하지만, 특정 영역 하나만 놓고 보면 BT를 이기는 대안이 항상 존재한다.** 이건 종합격투기 체육관과 비슷하다 — 타격 특화 선수, 그래플링 특화 선수가 각자의 무대에서는 올라운더를 이길 수 있지만, 종합 룰에서 가장 안정적으로 이기는 건 여전히 올라운더다. BT는 각 카테고리에서 1등을 한 번도 못 하고도(Chat Hard만 근소 우위) 평균 1위를 가져간다.

한 가지 예외가 눈에 띈다. Cross-Entropy만 확연히 낮다(87.6). 이건 쌍 비교 구조 자체를 버리고 chosen/rejected를 독립적인 이진 분류로 취급한 결과인데, Chat 점수가 74.9까지 떨어진 걸 보면 **비교 구조를 없애면 안 된다**는 것 자체는 명확한 결론이다.

정리하면 [#4 Rethinking Bradley-Terry 글](/blog/2026/bradley-terry-rethinking/)의 이론적 주장 — BT가 유일한 선택은 아니다 — 은 실증적으로도 맞다. 대안들은 실제로 작동하고, 특정 축에서는 BT보다 낫다. 그런데 "범용 RM 하나를 만들어야 한다"는 실무 제약 아래서는 **BT의 균형 잡힌 성능을 대체할 유인이 없다.** 이 결론이 얼마나 확고했는지는 후속작에서 드러난다 — Skywork-Reward-V2는 아예 손실 함수 비교 실험을 반복하지 않고, 처음부터 표준 BT 하나로 고정한 채 남은 노력 전부를 데이터에만 쏟는다.

## 오염 제거 — 벤치마크에도 노이즈가 있다

데이터를 다 고른 뒤에도 문제가 하나 더 남았다. RewardBench 관리팀이 저자들에게 Magpie Ultra의 약 5천 개 프롬프트가 RewardBench 평가셋과 겹칠 수 있다고 알려온 것이다. Magpie Ultra를 생성한 Llama-3.1-405B-Instruct가 RewardBench의 소스 데이터(Alpaca 등)로 학습됐을 가능성이 원인으로 지목됐다. 이건 학습 데이터 오염이면서 동시에 **RewardBench 평가셋 자체가 흔한 학습 데이터에서 유래한 프롬프트를 포함하고 있다**는 방증이기도 하다 — 뒤에서 다시 짚을 RewardBench의 구조적 약점이다.

| 데이터셋                    | 오염된 프롬프트 수 |
| --------------------------- | ------------------ |
| Preference 700K             | 15,349             |
| Skywork 80K v0.1(오염 포함) | 5,402              |
| Skywork 80K v0.2(제거 후)   | 445                |

RewardBench 팀이 제공한 스크립트로 오염된 프롬프트를 포함한 쌍을 Magpie Ultra 서브셋에서 제거해 v0.2를 만들었더니, 흥미롭게도 점수가 **떨어지지 않고 올랐다** — Gemma-2-27B는 93.8→94.3(+0.5), Llama-3.1-8B는 92.5→93.1(+0.6). 특히 Safety(+2.1/+1.0)와 Reasoning(+0.5~2.0)에서 개선 폭이 컸다. 오염된 데이터가 단순히 "정답 유출"이 아니라 품질 자체가 낮은 노이즈였다는 뜻이다.

# Experiments

## 80K가 378K와 700K를 모두 이긴다

핵심 비교표다. 같은 Llama-3.1-8B와 Gemma-2-27B 베이스에 데이터셋만 바꿔가며 RewardBench 평균을 측정했다.

| 데이터                        | 쌍 수      | Llama-3.1-8B | Gemma-2-27B |
| ----------------------------- | ---------- | ------------ | ----------- |
| Preference 700K               | 700,000    | 86.9         | 88.1        |
| Preference 378K               | 378,273    | 91.8         | 92.6        |
| **Skywork 80K (v0.1)**        | **80,000** | **92.5**     | **93.8**    |
| Skywork 80K (v0.2, 오염 제거) | 79,555     | 93.1         | 94.3        |

700K에서 378K로 줄였을 때 이미 +3.7~+5.9점이 올랐고, 378K에서 다시 80K로 줄였을 때 추가로 +0.7~+1.2점이 더 올랐다. 데이터를 계속 걷어냈는데 점수가 계속 오른다 — 이건 "데이터가 부족해서 성능이 떨어진다"는 통념과 정반대다. Gemma-2-27B-v0.2(94.3)는 이 논문이 나올 당시 RewardBench 리더보드 1위를 기록했다.

## V2: 어디까지 스케일할 수 있는가

> [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://arxiv.org/abs/2507.01352) (Liu et al., Skywork AI, ICLR 2026)

80K가 700K를 이긴다는 결과는 "작은 고품질 데이터가 이긴다"는 명제를 증명했지만, 동시에 질문 하나를 남겼다 — **이 큐레이션 방식이 스케일을 감당할 수 있는가?** 사람이 일일이 검수하는 방식은 80K에서는 가능해도 수천만 단위에서는 비용이 감당 안 된다. 2025년 7월 공개되고 2026년 ICLR에 채택된 후속작 Skywork-Reward-V2는 정확히 이 질문에 답한다.

### SynPref-40M: 사람과 LLM이 역할을 나눈다

저자들은 4천만 쌍짜리 선호 데이터 풀 **SynPref-40M**을 구축하고, 이 중 2,600만 쌍을 골라 최종 학습에 쓴다. 핵심은 사람과 LLM이 **같은 일을 나눠 하는 게 아니라 서로 다른 일을 맡는다**는 것이다. 사람은 검색엔진·최신 LLM 같은 외부 도구까지 동원해 소수의 쌍을 엄격하게 검증하고, LLM은 그 사람 라벨을 few-shot 예시로 삼아 대량의 나머지를 자동으로 채점한다.

<p align="center"><img src="/assets/post/image/skywork-reward/synpref-two-stage-pipeline.png" width="90%"></p>

파이프라인은 2단계로 나뉜다.

**Stage 1(소규모, 사람 주도, 반복 8회)**은 세 개의 데이터 뭉치를 오가며 돈다.

| 뭉치                            | 정체                                   | 규모      | 신뢰도 |
| ------------------------------- | -------------------------------------- | --------- | ------ |
| $$\mathcal{D}_{\text{un}}$$     | 아직 아무도 검증하지 않은 원본 풀      | 거대      | 모름   |
| $$\mathcal{D}_{\text{gold}}$$   | **사람이** 직접 검증한 것              | 아주 적음 | 최고   |
| $$\mathcal{D}_{\text{silver}}$$ | **LLM이** 사람 예시를 보고 라벨링한 것 | 중간      | 중간   |

여기서 사람이 하는 일이 단순 라벨링이 아니라는 점이 중요하다. 사람은 다섯 가지 속성을 함께 검증한다 — **작업 카테고리, 선호의 객관성, 논쟁성, 원하는 속성, 그리고 어노테이션 가이드라인 자체**다. 즉 "이 쌍은 $$y_1$$이 낫다"만 남기는 게 아니라 **"이런 종류의 쌍은 무엇을 기준으로 판단해야 하는가"**를 함께 남긴다. 그래야 LLM이 그 기준을 흉내낼 수 있기 때문이다.

그다음이 이 설계의 핵심인데, **학습과 평가에 서로 다른 뭉치를 쓴다.** $$\mathcal{D}_{\text{silver}}$$로 RM을 학습시키고, $$\mathcal{D}_{\text{gold}}$$의 검증 정확도로 가장 좋은 체크포인트를 고른다. 언뜻 반대로 해야 할 것 같지만 역할을 나눠 보면 자연스럽다.

- **학습에는 양이 필요하다.** gold는 규모가 작아 RM을 학습시키기엔 턱없이 부족하다.
- **평가에는 정확도가 필요하다.** 체크포인트를 고르는 기준이 틀리면 학습을 아무리 잘해도 잘못된 모델을 집는다.

교과서로 공부하고 기출 정답지로 채점하는 것과 같다. 정답지는 분량이 적어 공부 자료로는 못 쓰지만, 채점 기준으로는 그만한 것이 없다.

마지막으로 **적응적 검색(adaptive retrieval)**이 다음 라운드를 준비한다. 선택된 RM이 **틀리거나 확신이 낮은 쌍**을 골라낸 뒤, 그 쌍 각각에 대해 $$\mathcal{D}_{\text{gold}}$$에서 **비슷한 쌍(사람이 이미 판단해둔 것)을 검색해 few-shot 예시로 붙여** LLM에게 최종 판단을 맡긴다. 예측 확신도에 따라 검색 깊이도 조절한다.

판사가 애매한 사건을 만났을 때 유사 판례를 찾아보는 것과 같은 구조다. **사람의 판단을 라벨 하나로 쓰고 버리는 게 아니라, 비슷한 상황이 올 때마다 꺼내 쓰는 참조 자료로 재활용한다.** 이 과정을 8번 반복해 약 100만 쌍을 축적한다.

**Stage 2(대규모, 자동, 사람 개입 없음)**: 야생의(in-the-wild) 대량 풀 $$\mathcal{D}_{\text{wild}}$$에 Stage 1에서 만든 "현재 최고 RM"을 돌려 $$p = \sigma(r(y_w) - r(y_l))$$을 계산한다. $$p > 0.5$$로 확신에 찬 쌍은 그대로 채택하고, $$p$$가 0.5 근처거나 낮은 애매한 쌍만 $$\mathcal{D}_{\text{gold}}$$ 예시를 참고한 LLM 판정으로 넘긴다. 이렇게 나온 라벨을, 오직 사람 검증 데이터로만 학습된 별도의 "Gold RM"과 다시 대조해 **둘 다 동의하는 쌍만** 최종 풀에 남긴다.

여기에 더해 저자들은 버려지는 쌍도 그냥 버리지 않는다. 걸러진 쌍의 chosen/rejected를 **뒤집어서(flip)** 재활용하면 추가 성능 향상이 있었다고 보고한다 — 애매해서 버린 쌍도 "반대로 보면 확실한 신호"일 수 있다는 뜻이다.

이 구조는 신입 사원 교육과 닮았다. 처음(Stage 1)에는 사수가 신입이 처리한 건 하나하나를 다 들여다본다. 신입이 충분히 숙련되면(Stage 1의 RM이 좋아지면), 사수는 더 이상 전부를 보지 않고 **판단이 애매하거나 확신이 안 서는 케이스만** 골라 재검수한다(Stage 2). 전수 검사에서 표본 검사로 넘어가되, 표본을 무작위로 뽑지 않고 "틀리기 쉬운 지점" 위주로 뽑는 것이 이 파이프라인의 설계 의도다.

### 8개 모델, 0.6B부터 8B까지

Skywork-Reward-V2는 이 26M 쌍으로 Qwen3(0.6B/1.7B/4B/8B)와 Llama-3.1/3.2(1B/3B/8B) 계열 8개 모델을 학습시킨다. 전부 16K 토큰 컨텍스트, 전부 표준 BT 손실 하나로 통일했다.

| 모델                            | RewardBench | RewardBench v2 | 7개 벤치마크 평균 |
| ------------------------------- | ----------- | -------------- | ----------------- |
| Skywork-V2-Qwen3-0.6B           | 85.2        | 61.3           | 70.9              |
| Skywork-V2-Llama-3.2-1B         | 89.9        | 64.3           | 72.3              |
| Skywork-V2-Qwen3-1.7B           | 90.3        | 68.3           | 75.2              |
| Skywork-V2-Llama-3.2-3B         | 93.0        | 74.7           | 77.1              |
| Skywork-V2-Qwen3-4B             | 93.4        | 75.5           | 77.8              |
| Skywork-V2-Qwen3-8B             | 93.7        | 78.2           | 79.3              |
| Skywork-V2-Llama-3.1-8B         | 96.4        | 84.1           | 85.8              |
| **Skywork-V2-Llama-3.1-8B-40M** | **97.8**    | **86.5**       | **88.6**          |
| (비교) INF-ORM-Llama3.1-70B     | 95.1        | 76.5           | 73.8              |

작은 모델이 어디까지 가는지가 이 표의 진짜 메시지다. 1.7B짜리 Qwen3 모델(평균 75.2)이 70B짜리 INF-ORM-Llama3.1(평균 73.8)을 **RewardBench와 RewardBench v2를 제외한 나머지 5개 벤치마크 전부에서** 앞선다. 논문은 이를 "1.7B 규모의 RM이 70B 모델을 능가할 수 있다"고 직접 명시하며, **모델 크기 격차를 데이터 품질로 메운 사례**로 제시한다.

<p align="center"><img src="/assets/post/image/skywork-reward/v2-benchmark-comparison.png" width="95%"></p>

위 그림은 상위 3개 Skywork-V2 모델과 기존 SOTA급 RM(INF-ORM-70B, Llama-3.1-Nemotron-70B, Gemma-2-27B-v0.2)을 7개 벤치마크 전체에서 비교한다. 8B-40M 모델(진한 파랑)이 거의 모든 항목에서 막대 높이가 가장 높다. 특히 RM-Bench(96.0 vs 70B 모델들의 71\~75점대)와 PPE Correctness(87.2 vs 63\~64점대) 같은 "객관적 정답이 있는" 축에서 격차가 가장 크게 벌어진다.

### 품질 vs 규모: 1.8%로 이전 SOTA를 넘는다

<p align="center"><img src="/assets/post/image/skywork-reward/v2-quality-vs-quantity.png" width="65%"></p>

이 그래프가 이 논문의 결론을 한 장으로 압축한다. Llama-3.1-8B 기반 RM을 SynPref-40M의 필터링된 데이터로 학습시켰더니, **전체 학습 데이터의 1.8%(약 29만 쌍)만 쓰고도 이전 오픈소스 SOTA RM의 평균 점수(73점대)를 넘어섰다.** 나머지 98.2%의 데이터는 그 이후에 성능을 조금씩 더 끌어올리는 데 쓰인다. 반대로 Stage 2의 필터링을 거치지 않은 "원본(uncurated)" 데이터는 아무리 양을 늘려도 성능이 거의 늘지 않는 평평한 곡선을 그렸다(Figure 5 왼쪽 패널). 규모 자체가 아니라 **어떤 데이터로 채워진 규모인가**가 성능을 결정한다는 것을, 80K 논문의 결론을 4천만 쌍 규모에서 다시 확인한 셈이다.

사람 검증 단계가 얼마나 기여하는지도 별도로 측정했다. 시드 RM(71.0점) 대비, 원문 그대로만 보고 판정한 단순 사람 라벨링은 +0.4점, 선호 속성(작업 카테고리·논쟁성 등)까지 부여한 라벨링은 +1.1점, 외부 도구까지 동원한 전체 검증 프로토콜은 **+3.2점**을 기여했다. 반면 사람 없이 LLM만으로 큐레이션한 경우의 개선은 +0.1점에 그쳤다 — 이 논문에서도 "사람이 없으면 개선이 사실상 노이즈 수준"이라고 인정한다. 즉 V2의 스케일은 LLM이 만든 게 아니라, **소수의 엄격한 사람 검증을 앵커 삼아 LLM이 그 기준을 대량으로 복제한 결과**다.

## RewardBench의 한계 — 다음 글로 넘기는 부채

이 글 전체가 RewardBench 점수를 척도로 삼아왔다는 걸 짚고 넘어가야 한다. 그런데 V2 논문 스스로가 이 척도의 신뢰도에 의문을 던진다. 저자들은 오픈 RM 상위 20개 모델을 조사한 결과 **16개가 같은 베이스 모델을 쓰거나 비슷한 학습 데이터로 파인튜닝됐다**고 지적한다 — 사실상 2024년 9월 이후 RewardBench 상단이 정체돼 있다는 뜻이다. 또한 RewardBench 점수가 80점대에서 90점대로 오르는 구간에서, 이 논문이 새로 도입한 7개 벤치마크(PPE·RMB·RM-Bench·JudgeBench 등)와의 상관관계는 **약하거나 심지어 역상관**으로 나타났다. 앞서 살펴본 오염 프롬프트 사건도 같은 맥락이다 — RewardBench의 평가 프롬프트 자체가 흔히 쓰이는 학습 데이터 소스에서 유래해, 완전히 "본 적 없는" 평가라고 보장하기 어렵다.

이건 이 글의 결론을 무효화하는 게 아니라 **범위를 명확히 하는** 문제다. "데이터 큐레이션이 스코어를 밀어올린다"는 이 논문의 주장은 여러 벤치마크에서 반복 확인됐으므로 견고하다. 다만 "RewardBench 단일 숫자로 RM을 서열화할 수 있다"는 가정은 더 이상 성립하지 않는다. RM을 어떻게 평가해야 하는지 자체를 다시 설계하는 문제는 [#9 RewardBench 2 글](/blog/2026/rewardbench-2/)에서 이어간다.

# Conclusion

한 줄로 요약하면: **Skywork-Reward는 "모델을 키우지 않고도 데이터를 잘 고르는 것만으로 RM 성능을 지배할 수 있다"는 것을 80K 대 700K로 증명했고, V2는 같은 원리를 4천만 쌍 규모까지 사람-AI 협업 파이프라인으로 확장했다.**

정리하면,

1. **어떻게 골랐나**: Magpie는 강한 모델 우선 + 수학/코딩 상위 30% 집중 샘플링으로, WildGuardMix는 "이미 아는 문제는 다시 풀지 않는" 2단계 필터링으로, HelpSteer2는 단순 helpfulness 기준으로 각각 걸러 378K를 80K로 압축했다. 그 결과 700K(86.9\~88.1점) < 378K(91.8\~92.6점) < 80K(92.5~93.8점)라는, 데이터가 줄수록 성능이 오르는 역설이 나왔다.
2. **무엇으로 학습시켰나**: BT 손실은 8가지 대안 중 어느 하나도 전 영역에서 이기지 못하지만, 평균에서는 여전히 1위였다. [#4 글](/blog/2026/bradley-terry-rethinking/)의 "BT가 유일한 선택은 아니다"라는 이론적 주장은 맞지만, 범용 RM이라는 실무 제약에서는 BT를 대체할 유인이 없다는 게 실증적 결론이다.
3. **얼마나 더 갈 수 있나**: V2는 SynPref-40M과 2단계 human-AI 파이프라인으로 26M 쌍을 큐레이션했고, 1.8%의 데이터만으로 이전 SOTA를 넘었으며, 1.7B 모델이 70B 모델을 여러 벤치마크에서 앞섰다.

이 논문이 남긴 부채는 두 갈래다. 하나는 평가 쪽 — RewardBench가 포화·오염됐다는 문제로, [#9 RewardBench 2 글](/blog/2026/rewardbench-2/)이 이어받는다. 다른 하나는 방법론 쪽이다. 이 시리즈의 [#7 ArmoRM 글](/blog/2026/armorm/)은 "선호를 하나의 스칼라로 뭉개는 것 자체가 손실"이라는 다른 각도의 질문을 던진다. 데이터를 아무리 잘 고르고, 손실 함수를 아무리 잘 골라도, 애초에 스칼라 하나로 helpfulness·safety·정확성을 뭉뚱그리는 구조에서는 답할 수 없는 질문이 있다는 것이다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 여섯 번째 글이다.

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
  <li><strong>(현재 글)</strong> Skywork-Reward (2024) — 데이터 큐레이션이 아키텍처를 이긴다</li>
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
</ol>

**6부. Process & Verifiable Reward**

<ol start="28">
  <li><a href="/blog/2026/lets-verify-step-by-step/">Let's Verify Step by Step (2023)</a> — 과정 감독이 결과 감독을 이긴다</li>
  <li><a href="/blog/2026/math-shepherd/">Math-Shepherd (2023)</a> — 사람 라벨 없는 PRM</li>
  <li><a href="/blog/2026/deepseek-r1/">DeepSeek-R1 (2025)</a> — RLVR, 규칙이 reward가 될 때</li>
</ol>

**7부. Generative Reward Model**

<ol start="31">
  <li><a href="/blog/2026/prometheus-2/">Prometheus 2 (2024)</a> — 오픈 평가자 모델과 rubric 조건부 평가</li>
  <li><a href="/blog/2026/generative-verifiers/">Generative Verifiers (2024)</a> — reward를 next-token prediction으로</li>
  <li><a href="/blog/2026/generative-reward-models/">Generative Reward Models (2024)</a> — GenRM과 선호 학습의 결합</li>
  <li><a href="/blog/2026/self-taught-evaluators/">Self-Taught Evaluators (2024)</a> — 사람 라벨 없이 judge를 키우다</li>
  <li><a href="/blog/2026/deepseek-grm-spct/">DeepSeek-GRM / SPCT (2025)</a> — inference-time scaling</li>
</ol>

**8부. 생각하는 Judge, 그리고 그 신뢰**

<ol start="36">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 실전 종합**

<ol start="41">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어 모델의 reward 설계 (2025~2026)</a> — 열 개 모델이 실제로 택한 것</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 42편으로 구성된다.

# 참고 문헌

- Liu et al., 2024. [Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451). arXiv:2410.18451.
- [ar5iv: Skywork-Reward (HTML rendering)](https://arxiv.org/html/2410.18451) — 본문 그림·수식 원본.
- Liu et al., 2025. [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://arxiv.org/abs/2507.01352). ICLR 2026.
- [arXiv HTML v3: Skywork-Reward-V2](https://arxiv.org/html/2507.01352v3) — 본문 그림·표 원본.
- [OpenReview: Skywork-Reward-V2](https://openreview.net/forum?id=ofgxkMLqic) — ICLR 2026 리뷰·채택 정보.
- [GitHub: SkyworkAI/Skywork-Reward-V2](https://github.com/SkyworkAI/Skywork-Reward-V2) — 모델·데이터 저장소.
- [HuggingFace: Skywork/Skywork-Reward-Gemma-2-27B](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B)
- Sun, Shen, and Ton, 2024. [Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives](https://arxiv.org/abs/2411.04991). arXiv:2411.04991.
- 2024. [Secrets of RLHF in Large Language Models Part II: Reward Modeling](https://arxiv.org/abs/2401.06080). arXiv:2401.06080.
- Dong et al., 2024. [RLHF Workflow: From Reward Modeling to Online RLHF](https://arxiv.org/abs/2405.07863). (Preference 700K 출처)
- Lambert et al., 2024. [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787). arXiv:2403.13787.
- Malik et al., 2025. [RewardBench 2: Advancing Reward Model Evaluation](https://arxiv.org/abs/2506.01937). arXiv:2506.01937.
