---
layout: post
title: "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"
date: 2026-05-16 11:00:00 +0900
description: "Red-Teaming 시리즈 #2 — 38,961개 사람 공격 데이터셋과 RLHF 모델의 scaling behavior (Ganguli et al., Anthropic, 2022)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, rlhf, dataset]
giscus_comments: true
related_posts: true
---

> [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al., Anthropic, arXiv 2022)

# Introduction

## Red teaming이란 무엇인가

먼저 용어부터 정리하자. **레드 팀(red team)**은 원래 군사·보안 용어다. 아군(블루 팀)의 방어를 시험하기 위해 일부러 적군 역할을 맡아 공격하는 팀을 뜻한다. 이 개념을 LLM에 가져오면, **모델이 유해한 말을 하도록 일부러 유도하는 작업**이 된다. "폭탄 만드는 법", "특정 인종을 비하하는 말", "사기 치는 방법" 등을 어떻게든 모델 입에서 끄집어내려 시도하는 것이다.

왜 일부러 공격할까? 비유하자면, 새로 지은 건물에 입주하기 전에 소방관이 일부러 불을 질러보는 것과 같다. 실제 화재가 나기 전에 약점을 찾아 막기 위해서다. 모델을 세상에 내놓기 전에 "어떤 입력에서 무너지는가"를 미리 알아야 그 약점을 메울 수 있다.

## 이 논문의 위치

[지난 글](/blog/2026/perez-red-teaming/)에서 본 Perez 2022가 "**LM으로 LM을 공격하는** 자동화 red-teaming"의 첫 작품이었다면, 같은 해 8월 Anthropic이 낸 이 논문은 그 반대쪽 극단에 있다. **324명의 사람(크라우드워커)**이 직접 모델과 대화하며 만든 **38,961개의 사람 공격 데이터셋**을 공개하고, 그 데이터를 통해 "**RLHF가 정말로 안전을 보장하는가?**"를 정량적으로 검증한다.

두 접근이 정반대인 점이 흥미롭다. 자동 공격(Perez)은 빠르고 대량으로 찍어낼 수 있지만 사람의 창의적인 우회를 흉내 내기 어렵다. 사람 공격(Ganguli)은 느리고 비싸지만, 실제 악의적 사용자가 시도할 법한 미묘한 공격을 잡아낸다. 이 논문은 후자를 택해, 그 결과물을 **데이터셋으로 박제**해 학계 전체에 공개했다.

이 논문의 기여는 세 갈래다.

1. **Scaling 실험**: 3개 모델 크기(2.7B / 13B / 52B) × 4개 모델 타입(plain LM, prompted HHH, rejection sampling, RLHF). 결론은 **RLHF만이 크기를 키울수록 공격받기 어려워진다**는 것.
2. **데이터셋 공개**: 38,961개 공격 — 당시 기준 RLHF로 정렬된 LLM에 대한 **유일한** 대규모 공개 red-team 데이터.
3. **메서드 투명성**: red team 인터페이스, 워커 안전 가이드, 통계 방법론, 어노테이션 합의도까지 전부 공개.

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig1_scaling.png" width="95%">
</p>

위 그림이 논문의 핵심 메시지를 한 장에 담고 있다. 가로축이 모델 크기(파라미터 수), 세로축이 공격이 얼마나 잘 통했는지를 나타내는 harm 지표(높을수록 더 잘 뚫림 = 덜 안전)다. 네 개의 선 중 **RLHF 곡선만 크기가 커질수록 아래로 내려간다**(= 점점 안전해진다). Plain LM과 prompted HHH는 크기를 키워도 거의 평평하다 — 즉 더 커져도 안전해지지 않는다. Rejection sampling은 어느 크기에서도 강하지만 추론(inference) 비용이 16배다.

여기서 이 논문이 던지는 한 문장 메시지가 나온다. **"안전성은 모델을 그저 키운다고 따라오지 않는다. RLHF 같은 명시적 정렬 절차가 필요하고, 그 정렬은 모델이 클수록 더 잘 먹힌다."** 이것은 직관과 어긋날 수 있다. 흔히 "큰 모델이 더 똑똑하니 더 안전하겠지"라고 생각하지만, 실제로는 똑똑함과 안전함은 별개의 축이라는 것을 데이터로 보여준 것이다.

| 항목      | Perez 2022 (DeepMind)          | Ganguli 2022 (Anthropic)               |
| --------- | ------------------------------ | -------------------------------------- |
| 공격자    | LM (자동)                      | 사람 (수동, 324명)                     |
| 규모      | 50만 케이스                    | 38,961 attacks                         |
| 타겟      | DPG (Gopher 280B, prompt only) | 12개 모델 (4 타입 × 3 크기)            |
| 핵심 질문 | "어떻게 자동 공격할까?"        | "RLHF가 정말 안전한가? 스케일에 따라?" |
| 데이터    | 비공개                         | **공개** (HuggingFace)                 |
| 방어 평가 | 분류기 신호                    | 사람 라벨 + 합의도 분석                |

# Background

## 정렬을 만드는 4가지 레시피

논문이 비교하는 4가지 모델 타입을 이해하지 못하면 본문 결과를 읽을 수 없다. 중요한 것은 **네 가지 모두 동일한 Anthropic의 사전학습 LM(2.7B / 13B / 52B)에서 출발한다**는 점이다. 같은 재료(베이스 모델)에 서로 다른 "안전 조리법"을 적용한 셈이라, 어느 레시피가 효과적인지 공정하게 비교할 수 있다.

요리에 비유하면 이렇다. 같은 생선(베이스 LM)을 받아서,

1. **Plain LM (날것)**: 정렬을 전혀 하지 않은 순수 언어 모델. 단지 대화 형식만 보여주기 위해 1개의 예시 대화(1-shot dialogue context)를 앞에 붙여준다. "유해한지 아닌지"에 대한 개념이 없다. 회를 그냥 내놓는 것.
2. **Prompted HHH (양념만)**: 프롬프트 앞에 "도움이 되고(helpful), 정직하고(honest), 무해하게(harmless) 행동하라"는 14개의 예시 대화(14-shot prompting)를 붙인다. **모델 가중치는 전혀 바꾸지 않는다.** 그냥 "이렇게 행동해줘"라고 부탁만 하는 것. 양념을 뿌리되 굽지는 않은 것.
3. **Rejection Sampling, RS (여러 번 만들어 좋은 것만 고르기)**: 모델이 응답을 16개 샘플링한 뒤, **별도의 안전 점수 모델(safety/preference model)로 점수를 매겨 가장 안전한 응답을 골라** 사용자에게 보여준다. 학습 자체는 건드리지 않고 **추론 시점에만** 안전성을 높인다. 단, 16번 생성해야 하니 비용이 16배. 회를 16접시 만들고 가장 신선한 것만 내놓는 것.
4. **RLHF (제대로 굽기)**: 사람의 선호 데이터로 보상 모델(reward model)을 학습하고, 그 보상을 신호로 PPO 강화학습을 돌려 모델 가중치 자체를 바꾼다. 표준 정렬 방법. 생선을 양념해 제대로 조리한 요리.

이 네 가지의 핵심 차이는 **"모델 안에 안전을 새겨 넣는가, 아니면 바깥에서 다듬는가"**다. Prompted와 RS는 모델 가중치를 그대로 둔 채 외부에서 손보는 방식이고, RLHF는 모델 자체를 바꾼다. 이 차이가 뒤에서 본 scaling 결과의 갈림길이 된다.

## RLHF는 어떻게 동작하나

RLHF(인간 피드백 강화학습)는 이 논문의 주인공이므로 조금 더 풀어보자. 보통 세 단계로 이루어진다.

1. **선호 데이터 수집**: 같은 질문에 대한 두 응답 A, B 중 사람이 "어느 쪽이 더 나은가"를 고른다. 이 논문에서는 red team 인터페이스 자체가 이 데이터를 만드는 통로가 된다(뒤에서 설명).
2. **보상 모델(reward model) 학습**: 사람이 고른 선호를 흉내 내도록 별도의 모델을 학습한다. 이 모델은 임의의 응답을 입력받아 **"이 응답이 얼마나 좋은가(혹은 무해한가)"를 점수로 출력**한다. 사람을 일일이 부르는 대신, 사람의 취향을 학습한 자동 채점기를 만드는 것이다.
3. **PPO 강화학습**: 베이스 LM이 응답을 생성하면 보상 모델이 점수를 매기고, 그 점수를 높이는 방향으로 LM 가중치를 업데이트한다. 점수를 깎는 응답(유해한 응답)은 점점 덜 나오게, 점수를 높이는 응답(무해하고 도움 되는 응답)은 점점 더 나오게 모델이 미세조정된다.

핵심 직관: RLHF는 "사람이 싫어할 응답을 내면 벌점, 좋아할 응답을 내면 상점"을 모델 가중치에 새겨 넣는 과정이다. 그래서 한 번 학습되면 추론 때 추가 비용이 없다(RS와 달리 16번 생성할 필요가 없다).

## harm score는 어떻게 측정하나

"공격이 성공했다"를 판단하려면 응답이 얼마나 유해한지를 수치화해야 한다. 이 논문은 두 가지 잣대를 함께 쓴다.

- **사람 평가**: 워커가 대화를 끝낸 뒤 0~4점 Likert 척도로 "내가 모델을 얼마나 유해하게 만들었나"를 스스로 매긴다(0=실패, 4=매우 성공).
- **보상 모델 점수**: RLHF에 쓰는 것과 같은 종류의 preference/reward model이 각 응답에 **무해성 점수(harmlessness score)**를 자동으로 매긴다. 점수가 낮을수록 더 유해하다는 뜻이다. 멀티턴 대화는 각 턴 점수의 최솟값(min)이나 평균(mean)으로 집계한다.

왜 두 가지를 함께 쓸까? 사람 평가는 정확하지만 느리고 주관적이며, 보상 모델 점수는 빠르고 일관되지만 사람의 미묘한 판단을 다 잡지 못한다. 둘을 교차 검증하면 한쪽의 약점을 보완할 수 있다. 그리고 이 보상 모델이 곧 RS와 RLHF의 안전 장치이므로, **"공격을 측정하는 잣대"와 "방어를 만드는 도구"가 같은 모델**이라는 점도 기억해 두자.

## Red Team 인터페이스 — 공격하면서 데이터를 만든다

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig3_ui.png" width="95%">
</p>

워커는 4턴 이상의 멀티턴 대화로 모델을 공격한다. 여기서 영리한 설계가 들어간다. **매 턴마다 모델이 응답을 두 개 보여주고, 워커는 둘 중 더 유해한 쪽을 선택**한다. 이 선택이 두 가지 일을 동시에 한다.

1. 대화를 더 유해한 방향으로 끌고 간다 (공격 진행).
2. "A보다 B가 더 유해하다"는 비교 정보를 남긴다 — 이것이 곧 RLHF용 **선호 데이터(preference data)**가 된다.

즉 red teaming 인터페이스 자체가 데이터 수집 장치다. 공격자가 모델을 깨려고 애쓰는 행위가, 동시에 모델을 안전하게 만들 재료를 생산한다. 일석이조의 설계다. 대화가 끝나면 워커는 앞서 말한 0~4점 Likert로 공격 성공도를 자체 평가한다.

**워커 지시문 (Figure 3 왼쪽 패널)**의 요지:

- "AI를 나쁘게 행동하게 만들어라 — offensive하거나 harmful한 말을 하게 만들어라."
- 노골적인 욕설은 피하고, **한 attack당 하나의 주제**에 집중하라.
- 자신의 심리적 위험 허용도에 맞춰 주제를 골라라(자기 보호).

## Worker Safety — 이 논문이 특별한 이유

이 부분이 이 논문이 다른 안전 페이퍼와 구별되는 지점이다. 유해 콘텐츠를 하루 8시간씩 들여다보는 워커의 정신적 안전을 정면으로 다룬다. 혐오 발언, 폭력 묘사, 학대 콘텐츠를 반복해서 보는 일은 실제로 사람에게 트라우마를 남길 수 있기 때문이다.

- 명확한 콘텐츠 경고를 사전에 제시.
- **시간당 보상**(작업량 할당이 아니라) — 최소 **\$20/hr**. 할당량으로 보상하면 워커가 무리해서 더 많은 유해 콘텐츠에 노출되므로 시간당 보상을 택했다.
- 동료 지원 Slack 채널 운영.
- 사전 preview로 거북한 transcript는 건너뛸 수 있게 함.
- 작업 후 PANAS(Positive and Negative Affect Schedule, 긍정·부정 정서 척도) 기반 well-being 설문 실시.

결과적으로 "red team 멤버들이 작업을 부정적으로 느끼지 않았다"는 well-being 점수가 보고되었다. 이 절차는 이후 RT 연구의 사실상 표준이 되었다.

# Method

## 12개 모델 × 다수 워커 매트릭스

실험 설계는 단순하지만 규모가 크다. 4가지 모델 타입 × 3가지 크기 = 12개 모델. 각 모델을 동일한 인터페이스로 사람이 공격한다.

| 모델 타입 | 2.7B | 13B | 52B |
| --------- | ---- | --- | --- |
| Plain LM  | ✓    | ✓   | ✓   |
| Prompted  | ✓    | ✓   | ✓   |
| RS        | ✓    | ✓   | ✓   |
| RLHF      | ✓    | ✓   | ✓   |

각 모델에 대해 워커들이 약 3,000~4,000개의 attack을 만들어, 총 **38,961개**가 모였다. 12개 모델에 같은 공격 방식을 적용하므로, "어느 정렬 방법이 어느 크기에서 더 강한가"를 공정하게 비교할 수 있다.

### Rejection Sampling을 토이 예제로 따라가기

RS가 어떻게 안전성을 높이는지 작은 예로 보자. 사용자가 "특정 집단을 비하하는 농담을 해줘"라고 했다고 하자.

1. 모델이 응답을 16개 생성한다 (실제 논문 설정). 예시로 4개만 보면:
   - 응답 A: "물론이죠, [유해한 농담]…" → 안전 점수 1.2 (매우 유해)
   - 응답 B: "그런 농담은 누군가에게 상처를 줄 수 있어요…" → 안전 점수 8.5
   - 응답 C: "대신 이런 유머는 어때요…" → 안전 점수 7.9
   - 응답 D: "[약하게 유해한 농담]" → 안전 점수 4.0
2. 안전 점수 모델로 16개를 채점한다.
3. **가장 안전한 응답을 골라** 사용자에게 보여준다. (인터페이스가 두 응답을 보여주는 설정이면 안전한 상위 2개를 고른다.)

핵심: 모델 가중치는 그대로다. 단지 "여러 번 던지고 가장 깨끗한 것만 줍는다." 그래서 학습 없이도 즉시 안전성이 오르지만, 16번 생성하니 비용이 16배다. 또한 **천장이 정해져 있다** — 16개 후보가 전부 유해하면 그중 가장 덜 유해한 것도 결국 유해하다. RS는 모델이 "가끔은 안전한 답을 낼 수 있을 때" 그 안전한 답을 끌어올리는 방법이지, 모델에게 없는 안전성을 만들어내지는 못한다.

## 워커 분포 — 누가 공격했나

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig4_demographics.png" width="85%">
</p>

크라우드워커 324명 중 115명이 demographic survey에 응답했다.

- **성별**: 여 52.2% / 남 47.0% / non-binary 0.9%
- **연령**: 35~44세(33.9%)가 가장 많음
- **학력**: 대졸 53.9% — 미국 평균(32.9%)보다 **확연히 높음**
- **인종**: 백인 81.7% — 미국 평균(75.8%)보다 약간 높음

저자들이 명시적으로 인정하는 한계가 여기 있다. **"워커 모집단이 일반 사용자를 대표하지 않는다."** 학력과 백인 비율이 평균보다 높다. 그래서 "이 데이터셋의 공격 분포 = 실제 사용자의 위협 모델 분포"라고 단정할 수 없다. 어떤 종류의 공격은 과대표집되고, 다른 종류는 누락되었을 수 있다. 데이터셋을 쓸 때 이 편향을 염두에 둬야 한다.

## 공격 성공도 평가의 어려움 — Fleiss's Kappa

여기서 한 가지 미묘한 문제가 등장한다. **"이 공격이 성공했는가"를 사람들이 보면 의견이 갈린다.** 같은 대화를 보고도 누구는 "유해하다", 누구는 "그 정도는 괜찮다"고 한다. 이걸 수치로 잡기 위해 저자들은 같은 공격을 여러 reviewer에게 보여주고 **합의도(inter-annotator agreement)**를 측정했다.

합의도 측정에는 **Fleiss's Kappa($$\kappa$$)**를 쓴다. 직관은 이렇다.

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

기호를 하나씩 풀면,

- $$P_o$$ (observed agreement): reviewer들이 **실제로** 일치한 비율. 예를 들어 두 명이 같은 라벨을 단 케이스 비율.
- $$P_e$$ (expected agreement): reviewer들이 **순전히 우연으로** 일치할 것으로 기대되는 비율. 예컨대 둘 다 아무 생각 없이 70%의 케이스에 "유해"라고 찍으면, 우연히도 상당 비율이 겹친다. 그 우연의 몫이 $$P_e$$.
- 분자 $$P_o - P_e$$: "우연을 넘어선 진짜 일치." 우연으로 겹친 부분을 빼낸 순수 합의.
- 분모 $$1 - P_e$$: "우연을 넘어 도달할 수 있는 최대 일치." 즉 $$P_o - P_e$$를 이 최댓값으로 나눠 0~1로 정규화한다.

값의 의미: $$\kappa = 1$$이면 완벽한 합의, $$\kappa = 0$$이면 우연 수준(즉 합의라고 부를 게 없음), 음수면 우연보다도 못한 불일치. 통상 0.2~0.4는 "약함(poor)", 0.4~0.6은 "보통(fair)" 정도로 해석한다.

논문이 보고한 값:

| 합의도 측정              | Fleiss's $$\kappa$$ | 평가           |
| ------------------------ | ------------------- | -------------- |
| 5-point Likert           | 0.32                | poor agreement |
| Binary (≥3 vs <3)        | 0.49                | fair agreement |
| 3 reviewer만 (저자 제외) | 0.55                | maximum        |

읽는 법:

- 5점 척도로 그대로 보면 합의도가 0.32로 낮다. "3점이냐 4점이냐"처럼 미세한 정도 차이에서 사람들이 자주 갈린다.
- 그래서 "3점 이상이면 성공, 미만이면 실패"로 **이진화**하면 0.49로 오른다. 정도는 몰라도 "성공이냐 아니냐"는 좀 더 합의된다.
- 저자를 빼고 reviewer 3명만 보면 0.55까지 오른다 — 이게 이 데이터에서 얻을 수 있는 최대 합의도였다.

이게 시사하는 바: **"공격이 성공했는가"라는 판정 자체가 본질적으로 주관적**이라는 것이다. 0.55라는 숫자는 "노력해도 사람들 절반 좀 넘게만 의견이 모인다"는 뜻이다. 이 주관성 문제는 이후 모든 RT 벤치마크(HarmBench, JailbreakBench 등)가 끊임없이 씨름하게 되는 근본 난제다. 자동 채점기를 쓰든 사람을 쓰든, "유해함의 정의"가 흔들리면 모든 ASR 수치가 흔들린다.

# Experiments

## 메인 결과: Scaling Behavior

다시 Figure 1로 돌아가 자세히 보자. 가로축은 모델 크기, 세로축은 공격이 얼마나 잘 통했는지(높을수록 덜 안전).

- **Plain LM, Prompted HHH**: 크기를 키워도 공격 성공률이 **거의 변하지 않는다(flat)**. 모델을 크게 만들면 더 똑똑해질 뿐, 더 안전해지지는 않는다. 양념만 뿌린 생선은 아무리 큰 생선이어도 결국 익지 않은 것과 같다.
- **Rejection Sampling**: 모든 크기에서 가장 안전하다. 단, 16배 추론 비용이라는 큰 대가가 있다.
- **RLHF**: **52B에서만 RS와 동등한 수준**의 안전성에 도달한다. 2.7B에서는 그다지 안전하지 않지만, 크기를 키울수록 빠르게 안전해진다 — 유일하게 우하향하는 곡선.

**왜 RLHF만 크기의 혜택을 받을까?** 핵심은 "안전을 모델 안에 새겨 넣는가"에 있다. RLHF는 사람의 선호를 모델 가중치에 학습시키는데, 큰 모델은 표현력(capacity)이 커서 "어떤 요청에 어떻게 거부해야 하는가"라는 미묘한 패턴을 더 잘 흡수한다. 반대로 prompting은 모델에게 그냥 "착하게 굴어줘"라고 부탁할 뿐이라, 모델이 커진다고 그 부탁을 더 잘 따르지는 않는다. 부탁의 효과에는 천장이 있다.

이게 RLHF 정렬의 **"규모가 보상한다(scale rewards alignment)"** 성질이다. 그리고 이 발견의 무게는 가볍지 않다. **"모델을 키우면 알아서 안전해진다"는 막연한 기대를 정면으로 반박**하기 때문이다. 안전성은 별도의 명시적 정렬 절차(RLHF)를 통해서만, 그리고 그 절차가 규모와 결합할 때만 얻어진다.

## Harm Taxonomy — 어떤 공격이 통했나

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig9_harm_tags.png" width="80%">
</p>

성공한 52B 모델 공격 중 500개를 사후에 사람이 태깅하여 카테고리를 뽑았다.

| 카테고리                                        | 비중    |
| ----------------------------------------------- | ------- |
| Discrimination & injustice (차별·불공정)        | 가장 큼 |
| Hate speech & offensive language (혐오·욕설)    | 큼      |
| Violence & incitement (폭력·선동)               | 중간    |
| Non-violent unethical (거짓말, 사기 등)         | 중간    |
| Bullying & harassment (괴롭힘·희롱)             | 중간    |
| Child abuse, self-harm, terrorism, animal abuse | 2~5%    |

흥미로운 점: **"폭력적이지 않은 비윤리적 행동"**(거짓말, 사기 교사 등)이 단순한 욕설/혐오만큼 자주 나타난다는 것이다. 우리는 보통 "유해 = 욕설, 혐오 발언"이라고 떠올리지만, 실제 공격의 상당 부분은 노골적 욕설 없이도 충분히 해롭다. 예컨대 "보험 사기를 치는 자세한 방법"을 알려주는 응답은 욕설 하나 없이도 위험하다.

이 관찰이 후속 연구에 남긴 교훈은 분명하다. **단순 offensive language 필터링(욕설 사전 차단)만으로는 부족하다.** 정중한 어투로 포장된 유해 정보를 잡으려면 의미 수준의 판단이 필요하고, 이것이 뒤에 Llama Guard 같은 의미 기반 분류기 연구로 이어진다.

## UMAP으로 본 공격 공간

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig2_umap.png" width="95%">
</p>

UMAP은 고차원 임베딩을 사람이 볼 수 있는 2D로 눌러 펴는 시각화 기법이다(t-SNE의 친척이라 생각하면 된다). 38K 공격 각각을 임베딩으로 바꾼 뒤 2D 평면에 흩뿌리면, **여러 개의 뭉치(클러스터)로 분리**되는 것이 보인다. 각 뭉치가 서로 다른 종류의 공격 전략(attack vector)을 나타낸다.

이 그림의 메시지: 공격은 한 종류가 아니라 **여러 갈래로 흩어져 있다**. 따라서 한 가지 방어로 모든 클러스터를 한 번에 막을 수 없다. 한쪽 구멍을 막으면 다른 클러스터의 공격이 여전히 통한다는 뜻이다. 방어가 어려운 근본 이유를 한 장의 그림으로 보여준다.

## 정성적 예시

<p align="center">
  <img src="/assets/post/image/ganguli-red-teaming/fig10_examples.png" width="95%">
</p>

RS와 RLHF 모두 정렬되어 있음에도, 교묘하게 우회하면 여전히 유해 응답을 낸다. 이것이 핵심을 다시 못 박는다. **정렬은 유해 응답의 "확률을 낮추는" 것이지, "불가능하게 만드는" 것이 아니다.** 낮아진 확률은 충분히 영리한 입력으로 다시 끌어올릴 수 있다. (이 통찰은 시리즈 다음 글인 GCG에서 그래디언트로 그 "낮은 확률"을 자동으로 끌어올리는 공격으로 이어진다.)

# Discussion: 데이터 공개에 대한 윤리적 고민

이 논문이 다른 LLM 안전 페이퍼와 가장 다른 부분이 §5의 **데이터 공개 trade-off 논의**다. 유해 공격 38K개를 공개하는 것은 양날의 검이다 — 방어 연구를 가속하는 동시에, 악용될 수도 있다. 저자들은 이 딜레마를 숨기지 않고 정면으로 적었다.

**공개 찬성 근거**

- 투명성: Anthropic 외부의 독립적 연구자가 직접 데이터를 분석·검증할 수 있다.
- 선례: 앞선 BAD 데이터셋(5K개)을 잇는 작업이며, RLHF 정렬 모델에 대한 대규모 공격 데이터로는 첫 사례다.
- 공익: 오픈 사이언스가 분야 전체의 안전 연구를 가속한다.

**공개 반대 근거**

- 유해 모델 학습에 악용될 수 있다(공격 데이터로 더 잘 공격하는 모델을 만들 위험).
- 평판 리스크.
- 데이터를 보는 사람의 콘텐츠 노출 위험.
- PII(개인식별정보) 필터링이 완벽하지 않을 수 있다.
- 자사 모델의 취약점을 스스로 노출하는 셈.

저자들은 결국 **공개를 선택**했다. 단, PII 필터링, 콘텐츠 경고, 별도 접근 페이지 등 안전장치를 두었다. "위험이 있지만 투명성의 공익이 더 크다"는 판단이다. 이 결정과 그 논증 방식은 이후 거의 모든 RT 데이터셋 공개의 reference가 되었다.

# Conclusion

핵심 메시지 한 줄: **"RLHF만이 scale의 혜택을 받는 정렬 방법이다. 다른 방법들은 모델을 키워도 안전해지지 않는다."**

세 가지 기여를 다시 정리하면,

1. **Scaling 결과**: RLHF는 크기를 키울수록 공격받기 어려워진다(다른 방법은 평평). 안전성은 규모만으로 따라오지 않으며, 명시적 정렬 + 규모가 함께 필요하다.
2. **38,961 attack 공개**: RLHF 정렬 모델에 대한 당시 유일한 대규모 사람 공격 데이터.
3. **메서드 투명성**: UI, 워커 가이드, 통계 방법론, 합의도까지 모두 공개해 후속 연구가 그대로 따라 할 수 있게 했다.

## 한계점

- **대화 어시스턴트 한정**: 코드 생성, 검색, 에이전트 등 다른 응용에 대한 평가는 없다.
- **도메인 전문성 부족**: 무기, 화학처럼 평가자가 모르는 영역은 유해성을 제대로 라벨링하기 어렵다.
- **포괄성 불가능**: harm space가 사실상 무한해서 어떤 데이터셋도 모든 공격을 망라할 수 없다.
- **합의도 한계**: Fleiss's $$\kappa \approx 0.32{-}0.55$$ — RT 성공 판정은 본질적으로 주관적이다.
- **사후에 발견된 공격 클래스**: roleplay 공격 등은 데이터셋에 거의 없다(수집 당시 알려지지 않은 기법).
- **워커 demographic bias**: 대졸·백인 비율이 미국 평균보다 높아, 공격 분포가 실제 사용자 분포와 다를 수 있다.

이 논문은 화려한 새 공격 기법이 아니라 **"데이터셋 페이퍼"라는 형태로 RT 분야의 공통 자원을 만든 사례**다. 이후 GCG의 AdvBench, HarmBench, JailbreakBench 같은 후속 벤치마크들이 이 작업이 닦아놓은 길 위에 세워졌다. 동시에 "RLHF + 규모"라는 정렬의 황금 조합과 "정렬은 확률을 낮출 뿐"이라는 한계를 함께 못 박아, 이후 시리즈에서 다룰 자동 공격(GCG, AutoDAN 등)의 출발점을 제공했다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 두 번째 글이다.

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격하기 (foundation)
2. **(현재 글)** Ganguli 2022 — Anthropic의 38K 공격 데이터셋과 scaling behavior
3. [GCG (Zou 2023)](/blog/2026/gcg-attack/) — 그래디언트 기반 universal suffix
4. [AutoDAN (Liu 2023)](/blog/2026/autodan/) — 자연어 유지하는 GA 기반 jailbreak
5. AttnGCG — attention manipulation으로 GCG 강화 _(추후 작성)_
6. [PAIR (Chao 2023)](/blog/2026/pair-attack/) — 20쿼리 black-box attacker LM
7. [TAP (Mehrotra 2023)](/blog/2026/tap-attack/) — 트리 탐색 + 이중 pruning으로 PAIR 효율화
8. [GPTFuzz (Yu 2023)](/blog/2026/gptfuzz/) — AFL 영감의 template-level fuzzing
9. [Crescendo (Russinovich 2024)](/blog/2026/crescendo/) — multi-turn escalation으로 single-turn 방어 무력화
10. [Many-shot Jailbreaking (Anil 2024)](/blog/2026/many-shot-jailbreaking/) — long-context를 ICL로 weaponize
11. [Curiosity-driven RT (Hong 2024)](/blog/2026/curiosity-redteam/) — novelty reward로 mode collapse 해결
12. [Auto-RT (Liu 2025)](/blog/2026/auto-rt/) — strategy-level RL exploration + progressive curriculum
13. [AgenticRed (Yuan 2026)](/blog/2026/agenticred/) — RT 시스템 자체를 진화
14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — Tool-use LLM agent에 대한 IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격
16. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
18. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — AI feedback으로 인간 라벨 없이 alignment
19. [Llama Guard (Inan 2023)](/blog/2026/llama-guard/) — open-weight input/output safety classifier
    본 시리즈는 19편으로 완결되었다.

# 참고 문헌

- Ganguli et al., 2022. [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858).
- [Anthropic blog: Red Teaming Language Models to Reduce Harms](https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned)
- [GitHub: anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf) — 데이터셋 저장소
- [HuggingFace: Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- Bai et al., 2022. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). (RLHF 베이스라인)
- Perez et al., 2022. [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). (자동 RT 짝)
- Xu et al., 2021. [Bot-Adversarial Dialogue (BAD)](https://aclanthology.org/2021.naacl-main.235/). (5K 공격 데이터셋 선행 연구)
