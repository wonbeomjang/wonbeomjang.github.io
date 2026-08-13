---
layout: post
title: "Rubrics as Rewards: 정답이 없는 도메인에 reward를 만드는 법"
date: 2026-08-11 09:25:00 +0900
description: "RLHF Reward 설계 시리즈 #29 — 채점 기준표를 reward로 바꿔 RLVR을 비검증 도메인으로 확장하다"
categories: [paper]
tags: [rlhf, reward-model, rubric, rlvr, llm-as-a-judge, paper]
giscus_comments: true
related_posts: true
---

> [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746) (Gunjal et al., Scale AI, ICLR 2026)

# Introduction

이 시리즈 [#21 DeepSeek-R1 글](/blog/2026/deepseek-r1/)은 "규칙이 reward가 될 때" 무슨 일이 벌어지는지를 다뤘다. 정답이 명확한 수학·코드 문제라면 채점 함수 하나로 충분했다. $$\boxed{42}$$가 맞았는지, 테스트 케이스를 통과했는지는 프로그램이 판단할 수 있다. 그런데 그 글의 끝에는 대답하지 않은 질문이 하나 남았다. **의료 상담, 과학 설명, 글쓰기, 그리고 이 시리즈의 독자층이 매일 마주하는 안전성 판단처럼 "정답 문자열"이 아예 없는 도메인은 어떻게 하나?**

RLVR(Reinforcement Learning with Verifiable Rewards)은 강력하지만 근본적으로 이분법적이다. 맞았거나 틀렸거나. 하지만 "이 응답이 응급 상황을 적절히 안내했는가", "이 설명이 과학적으로 타당하면서도 이해하기 쉬운가" 같은 질문에는 $$\text{match}(y, \hat y) \in \{0,1\}$$ 같은 함수가 없다. 그렇다고 다시 사람이 라벨링한 선호 데이터로 스칼라 reward model을 학습시키는 길로 돌아가면, 이번엔 이 시리즈 3부에서 다룬 문제 — 길이·형식 같은 표면적 특징에 편승하는 reward hacking(11~13편)과 overoptimization([10편](/blog/2026/reward-model-overoptimization/)) — 이 그대로 재발한다. RLVR의 신뢰성과 선호 기반 RM의 표현력, 둘 다 원하는데 둘 다 완전히는 가질 수 없는 딜레마다.

이번 글에서 다루는 **Rubrics as Rewards (RaR)**는 이 딜레마에 정면으로 답한다. 발상은 단순하다. **"정답은 없지만, 좋은 답이 갖춰야 할 조건은 열거할 수 있다."** 의료 상담이라면 "응급 신호를 놓치지 않았는가", "전문의 상담을 권했는가", "확정 진단을 함부로 내리지 않았는가" 같은 조건들이다. 이 조건들을 프롬프트마다 체크리스트(rubric)로 만들고, judge 모델이 각 항목의 충족 여부를 판정한 결과를 집계해 reward로 쓴다. 정답 하나를 요구하는 대신 "무엇을 갖췄는지"를 여러 개의 작은 이진 질문으로 쪼갠 것이다.

이 글에서 답할 질문은 네 가지다.

1. **rubric을 reward로 바꾸는 정확한 수식은 무엇인가**: 항목별 이진 판정을 하나의 스칼라로 모으는 두 가지 집계 전략(explicit·implicit)이 어떻게 다른가.
2. **rubric은 누가, 어떻게 만드는가**: 사람 없이 LLM이 프롬프트마다 7~20개의 채점 항목을 생성한다는 게 실제로 안전한가.
3. **Likert 점수 하나로 채점하는 것보다 정말 나은가**: HealthBench와 GPQA-Diamond에서 어떤 baseline 대비 얼마나 향상됐는가.
4. **이 방법은 어디서 뚫리는가**: rubric 자체가 gaming의 대상이 될 수 있다는 한계는 다음 어느 글로 이어지는가.

# Background

## RLVR이 멈추는 지점

[#21 DeepSeek-R1 글](/blog/2026/deepseek-r1/)에서 다뤘듯, RLVR은 $$r(x, \hat y) = \text{match}(y, \hat y)$$ 형태의 규칙 기반 reward로 수학·코드 도메인에서 학습된 reward model 없이도 강한 추론 능력을 끌어낸다. 문제는 이 정식화가 "유일한 정답 $$y$$가 존재한다"는 전제에 강하게 의존한다는 점이다. 의료 상담에는 유일한 정답이 없다. 같은 증상이라도 안전하게 답하는 방식은 여러 가지이고, 무엇이 "충분히 안전한 답"인지는 텍스트 일치로 판단할 수 없다.

그렇다고 선호 데이터로 학습된 스칼라 reward model(2부에서 다룬 Bradley-Terry 기반 RM들)을 쓰면, 이 RM은 다시 사람의 편향과 데이터의 노이즈를 그대로 흡수한다. 논문은 이 지점을 정확히 짚는다. 학습된 preference RM은 "응답 길이, 형식, 라벨러 편향 같은 표면적 artifact에 overfit하는 경향"이 있고, 대량의 pairwise 비교 데이터를 요구한다. RLVR의 신뢰성과 선호 기반 RM의 유연성 사이, 그 중간 지대가 비어 있었다.

## Instance-specific rubric이라는 발상

이 빈 자리를 채운 건 원래 **평가**를 위해 만들어진 개념이었다. OpenAI가 공개한 HealthBench(Arora et al., 2025)는 5,000개의 임상 대화마다 의사가 직접 작성한 세부 채점 기준(rubric)을 붙여 모델을 평가한다. RaR의 기여는 이 아이디어를 평가에서 **학습**으로 옮긴 것이다. "채점에만 쓰던 rubric을 그대로 on-policy RL의 reward 함수로 재사용하면 어떨까"라는 질문이다.

비유하자면 이렇다. 요리 대회에서 심사위원이 "이 요리 몇 점?"이라고 한 번에 매기면 그날 컨디션이나 마지막에 먹은 메뉴에 따라 점수가 출렁인다. 반면 "간이 적절한가, 불맛이 났는가, 플레이팅이 정갈한가, 재료 본연의 맛을 살렸는가"를 항목별로 따로 체크하게 하면 채점자가 바뀌어도 결과가 비교적 일관된다. RaR은 LLM judge에게 후자의 방식을 강제한다.

# Method

<p align="center"><img src="/assets/post/image/rubrics-as-rewards/fig1-overview.png" width="95%"></p>

RaR의 전체 흐름은 두 단계다. (i) rubric 생성: 강한 LLM이 참조 답안을 근거로 프롬프트별 rubric을 만든다. (ii) GRPO 학습: 이 rubric을 LLM judge에게 프롬프트로 주고, judge가 산출한 reward로 정책을 업데이트한다. 뒤에서 각 단계를 순서대로 뜯어본다.

## 문제 정식화

프롬프트 $$x$$와 정책 $$\pi_\theta$$에서 샘플링한 응답 $$\hat y \sim \pi_\theta(\cdot \mid x)$$가 있다고 하자. 유일한 정답이 없는 도메인에서, RaR은 각 프롬프트마다 $$k$$개의 rubric 항목 집합 $$\{(w_j, c_j)\}_{j=1}^k$$을 정의한다.

- $$w_j \in \mathbb{R}$$: 항목 $$j$$의 가중치.
- $$c_j : (x, \hat y) \mapsto \{0, 1\}$$: 응답 $$\hat y$$가 항목 $$j$$를 만족하는지 판정하는 이진 함수(실제로는 LLM judge가 담당).

이렇게 정의하면 RLVR은 이 정식화의 특수 케이스로 흡수된다. $$k=1, w_1=1$$이고 $$c_1$$이 "정답과 정확히 일치하는가"로 고정되면 $$r(x,\hat y) = \text{match}(y,\hat y)$$가 그대로 나온다. 즉 rubric 기반 reward는 검증 가능한 correctness 신호를 다차원·다기준 감독으로 일반화한 형태다.

## 두 가지 집계 전략

k개의 이진 판정을 하나의 스칼라 reward로 모으는 방법이 관건이다. 논문은 두 가지 상호 보완적인 전략을 비교한다.

**Explicit Aggregation.** 각 항목을 LLM judge가 독립적으로 판정한 뒤, 가중합을 정규화한다.

$$r(x, \hat y) = \frac{\sum_{j=1}^k w_j \cdot c_j(x, \hat y)}{\sum_{j=1}^k w_j}$$

분모의 정규화는 프롬프트마다 rubric 항목 수나 가중치 합이 달라도 reward 스케일을 프롬프트 간에 비교 가능하게 맞춰준다.

**Implicit Aggregation.** 모든 rubric 항목과 그 범주형 가중치를 한꺼번에 judge에게 넘기고, 집계 자체를 모델에게 맡긴다.

$$r_{implicit}(x, \hat y) = f_\phi(x, \hat y, \{d_j\}_{j=1}^k)$$

여기서 $$f_\phi$$는 프롬프트 $$x$$, 응답 $$\hat y$$, rubric 항목 집합 $$\{d_j\}$$를 입력받아 1~10 Likert 점수 하나를 직접 산출하는 LLM judge다. 사람이 가중치를 일일이 튜닝할 필요가 없다는 게 장점이지만, judge의 내부 집계 로직이 불투명해진다는 대가가 따른다.

## rubric은 누가 만드는가

사람이 매 프롬프트마다 rubric을 쓰는 건 비용상 불가능하다. 논문은 강한 LLM(의료 도메인은 GPT-4o, 과학 도메인은 o3-mini)에게 정답 참조 답안(reference answer)을 근거로 rubric을 생성시킨다. 참조 답안은 전문가 감독의 대리(proxy) 역할을 한다. rubric 생성은 네 가지 원칙을 따른다.

| 원칙                        | 의미                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| Grounded in Expert Guidance | 참조 답안 등 전문가 지식에 근거해 사실·추론 과정을 담는다                                                 |
| Comprehensive Coverage      | 정확성뿐 아니라 논리성·완결성·스타일·안전성까지 다차원을 포괄. 흔한 실수를 짚는 부정 조건(pitfall)도 포함 |
| Criterion Importance        | 모든 조건이 같은 무게가 아니다. 범주형 라벨(Essential/Important/Optional/Pitfall)로 중요도를 표시         |
| Self-Contained Evaluation   | 각 항목은 외부 맥락 없이 그 자체로 독립적으로 판정 가능해야 한다                                          |

프롬프트당 7~20개의 항목이 생성되며, RaR-Explicit은 범주형 라벨을 수치 가중치로 변환해 쓴다: **Essential=1.0, Important=0.7, Pitfall=0.9, Optional=0.3**. Pitfall 항목은 "오진을 피한다"처럼 긍정문으로 표현되어, 만족할수록 점수가 오르는 방향으로 통일된다.

## 토이 예제: 5개 항목, 두 응답, 그리고 순위 반전

의료 상담 프롬프트 하나로 계산을 직접 따라가 보자.

> **프롬프트**: "요 며칠 가슴이 조이는 느낌과 왼쪽 팔 저림이 같이 있어요. 심각한 건가요?"

이 증상 조합(흉통 + 팔 저림)은 심장 질환의 대표적 응급 신호다. rubric 5개 항목을 다음과 같이 세웠다고 하자.

| #   | 항목                                               | 범주      | 가중치 $$w_j$$ |
| --- | -------------------------------------------------- | --------- | -------------- |
| 1   | 흉통·팔 저림 조합을 응급 신호로 언급한다           | Essential | 1.0            |
| 2   | 즉시 응급실·전문의 방문을 권고한다                 | Essential | 1.0            |
| 3   | 가능한 원인(협심증, 심근경색 등)을 간단히 설명한다 | Important | 0.7            |
| 4   | 자가진단·확정 진단을 내리지 않는다                 | Pitfall   | 0.9            |
| 5   | 안정을 취하라는 등 일반적 생활 조언을 덧붙인다     | Optional  | 0.3            |

가중치 합은 $$\sum w_j = 1.0+1.0+0.7+0.9+0.3 = 3.9$$다. 이제 두 응답을 비교한다.

- **응답 A**: 원인 설명은 상세하고 자가진단도 하지 않지만, 응급성 언급도 즉시 병원행 권고도 없다. → 항목 3, 4, 5 충족 (3/5).
- **응답 B**: 문장은 짧지만 "이건 응급 상황일 수 있으니 지금 바로 응급실에 가라"고 정확히 짚는다. 원인 설명이나 생활 조언은 없다. → 항목 1, 2 충족 (2/5).

먼저 **naive 방식**(가중치 없이 그냥 충족 비율)으로 보면 A가 이긴다.

$$r_{naive}(A) = \frac{3}{5} = 0.600 \qquad r_{naive}(B) = \frac{2}{5} = 0.400$$

하지만 RaR의 **explicit aggregation**으로 계산하면 결과가 뒤집힌다.

$$r(A) = \frac{1.0\cdot 0 + 1.0\cdot 0 + 0.7\cdot 1 + 0.9\cdot 1 + 0.3\cdot 1}{3.9} = \frac{1.9}{3.9} \approx 0.487$$

$$r(B) = \frac{1.0\cdot 1 + 1.0\cdot 1 + 0.7\cdot 0 + 0.9\cdot 0 + 0.3\cdot 0}{3.9} = \frac{2.0}{3.9} \approx 0.513$$

체크박스 개수로는 A가 60%, B가 40%로 A가 우세해 보이지만, 가중치를 반영하면 $$r(B) \approx 0.513 > r(A) \approx 0.487$$로 순위가 뒤집힌다. 항목 개수가 아니라 "무엇을 놓쳤는가"가 응급 상황에서는 훨씬 중요하다는 걸 가중치가 정확히 반영한 것이다. [HH-RLHF 데이터셋 글](/blog/2026/hh-rlhf-red-team/)에서 다룬 "대화 전체의 안전은 가장 약한 고리가 결정한다"는 논리와 같은 계열이다 — 응급 신호를 놓친 항목 하나가 나머지 셋을 압도한다.

동시에 이 예제는 RaR-Explicit의 약점도 보여준다. 가중치를 어떻게 매기느냐에 따라 순위가 통째로 바뀐다. 논문 저자들도 "고정된 가중합은 통제력을 주지만 부서지기 쉽다(brittle)"고 직접 인정한다. Implicit aggregation은 이 수동 튜닝을 judge에게 위임해 이 문제를 우회하려는 시도다.

## 비교 대상 baseline

논문은 RaR을 다음 baseline들과 비교한다.

| 방법             | 정의                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Direct-Likert    | judge가 참조 답안 없이 응답 하나만 보고 1~10 Likert 점수를 매김                          |
| Reference-Likert | judge가 전문가/강한 LLM이 쓴 참조 답안과 비교해 1~10 Likert 점수를 매김                  |
| RaR-Predefined   | 모든 프롬프트에 "간결한가", "정보가 정확한가" 같은 고정 범용 rubric을 균등 가중치로 적용 |
| RaR-Explicit     | 프롬프트별 instance-specific rubric + 범주형 가중치 가중합(Eq. 1)                        |
| RaR-Implicit     | 프롬프트별 instance-specific rubric을 judge에게 통째로 넘겨 홀리스틱하게 채점(Eq. 2)     |

## 학습 설정

정책 모델은 Qwen2.5-7B(base)를 GRPO([#16 GRPO/DeepSeekMath 글](/blog/2026/grpo-deepseekmath/) 참고)로 학습한다. reward 계산에는 gpt-4o-mini를 judge로 쓴다.

| 하이퍼파라미터              | 값                                                  |
| --------------------------- | --------------------------------------------------- |
| Batch size                  | 96                                                  |
| Learning rate               | $$5 \times 10^{-6}$$, 10% linear warmup 후 constant |
| 프롬프트당 rollout 수 $$k$$ | 16                                                  |
| Context length              | 3584 토큰                                           |
| Sampling temperature        | 1.0                                                 |
| 학습 인프라                 | 단일 노드, H100 8장                                 |

학습 데이터는 RaR-Medicine(약 2만 개 프롬프트, GPT-4o가 rubric 생성)과 RaR-Science(약 2만 개 프롬프트, GPQA-Diamond 카테고리에 맞춰 큐레이션, o3-mini가 rubric 생성) 두 세트다.

# Experiments

## 메인 결과

평가는 두 축이다. HealthBench(5,000개 임상 대화, 의사가 작성한 rubric으로 채점, Communication quality·Instruction following·Accuracy·Context awareness·Completeness 5개 축)와 GPQA-Diamond(객관식, greedy decoding 10회 반복 평균 + 95% 신뢰구간). 모든 정책은 gpt-4o-mini를 judge로 평가한다.

<p align="center"><img src="/assets/post/image/rubrics-as-rewards/fig2-results.png" width="92%"></p>

| 방법                       | HealthBench 전체 점수 | GPQA-Diamond 정확도 |
| -------------------------- | --------------------- | ------------------- |
| Qwen2.5-7B (off-the-shelf) | 7.7%                  | 31.7%               |
| Qwen2.5-7B-Instruct        | 22.7%                 | 35.0%               |
| Direct-Likert              | 25.5%                 | 34.8%               |
| Reference-Likert           | 28.9%                 | 36.5%               |
| RaR-Predefined             | 12.5%                 | 31.7%               |
| RaR-Explicit               | 29.7%                 | 36.9%               |
| **RaR-Implicit**           | **31.2%**             | **37.6%**           |

RaR-Implicit이 두 벤치마크 모두에서 최고 점수를 낸다. 논문 Abstract는 "Direct-Likert 대비 HealthBench에서 최대 31%, GPQA-Diamond에서 7%의 상대 향상"을 보고하는데, 이는 여러 세팅(축·서브셋) 중 관측된 최댓값이고, 위 표의 전체 점수 기준 향상 폭은 HealthBench가 약 +22%(25.5%→31.2%), GPQA-Diamond가 약 +8%(34.8%→37.6%)로 이보다는 다소 작다. 그래도 방향은 분명하다 — instance-specific rubric을 쓴 두 RaR 변형이 나머지 모든 baseline을 앞선다.

흥미로운 실패 사례가 RaR-Predefined다. 모든 프롬프트에 같은 범용 rubric("간결한가", "정보가 정확한가")을 적용했더니 HealthBench 12.5%로 base 모델(7.7%)보다 살짝 나은 정도에 그쳤다. 프롬프트별로 다른 rubric을 만들지 않으면 그 프롬프트 특유의 실패 모드(예: 응급 신호를 놓치는 것)를 rubric이 전혀 짚지 못하기 때문이다. "rubric이 있다"는 것과 "instance-specific rubric이 있다"는 것 사이의 격차가 이만큼 크다.

## Ablation: rubric은 사람이 써야 하나, LLM이 만들어도 되나

HealthBench의 human-authored rubric과 LLM이 참조 답안을 보고 합성한 rubric을 비교한 ablation이다(HealthBench-1k를 holdout으로, 나머지 3.5k로 학습).

| 학습 방법                                          | HealthBench-1k 점수 |
| -------------------------------------------------- | ------------------- |
| Expert-Answer-SFT                                  | 20.4%               |
| Simple-Likert                                      | 23.9%               |
| Reference-Likert                                   | 31.7%               |
| RaR-Implicit-Synthetic-NoRef (참조 답안 없이 생성) | 32.0%               |
| RaR-Implicit-Human (의사가 직접 작성)              | 34.8%               |
| **RaR-Implicit-Synthetic (참조 답안 기반 생성)**   | **35.9%**           |

가장 놀라운 결과는 참조 답안을 근거로 LLM이 합성한 rubric(35.9%)이 의사가 직접 쓴 rubric(34.8%)과 대등하거나 오히려 살짝 앞선다는 점이다. 반면 참조 답안 없이 생성한 rubric(32.0%)은 확연히 처진다. 즉 핵심은 "누가 rubric을 쓰느냐"가 아니라 "무엇을 근거로 rubric을 쓰느냐"다 — 전문가 grounding이 있으면 LLM도 전문가 수준의 채점표를 만든다는 뜻이다.

부가 ablation 두 가지도 짚을 만하다. rubric 생성 LLM을 바꿔보면(참조 답안 없이) GPT-4o가 34.2%로 가장 우수했고, GPT-4o-mini(32.7%)·Qwen-72B-Instruct(32.7%)가 뒤를 이었으며, 흥미롭게도 Qwen-32B-Instruct(31.1%)가 Qwen-7B-Instruct(31.9%)보다 낮아 "더 큰 모델이 늘 더 좋은 rubric을 만든다"는 직관이 항상 맞지는 않았다. 그리고 rubric 설계 요소를 하나씩 빼보는 ablation에서는 "범주형 가중치 라벨을 제거"한 세팅(38.8%)이 "전체 rubric을 그대로 쓴" 세팅(37.2%)보다 오히려 살짝 높게 나왔다. 저자들도 "가중치나 pitfall 조건을 포함하는지에 따른 성능 차이가 크지 않았다"고 인정한다. 앞서 토이 예제에서 가중치가 순위를 뒤집는 걸 직접 봤지만, 집계된 벤치마크 점수 수준에서는 그 민감도가 생각보다 크게 드러나지 않는다는 뜻이다 — 개별 사례의 취약성과 평균 지표의 둔감함 사이의 간극을 보여주는 지점이다.

## 왜 Likert 단일 점수보다 rubric이 나은가

논문은 judge 크기를 바꿔가며(gpt-4o-mini부터 Qwen-7B-Instruct까지) chosen/rejected 응답 쌍을 rubric 유무로 채점시키는 실험도 진행했다. 결과는 일관됐다 — rubric을 준 judge는 모든 크기에서 direct Likert보다 pairwise preference accuracy가 높았고, 특히 작은 judge일수록 개선 폭이 컸다. 작은 모델이 큰 모델과의 격차를 rubric 하나로 상당 부분 좁힌 셈이다.

이 결과는 [#1 Christiano 2017 글](/blog/2026/deep-rl-human-preferences/)에서 짚은 통찰과 같은 계열이다. 그 글에서 "사람은 절대 점수를 매기는 것보다 두 개를 놓고 비교하는 데 훨씬 일관적이다"라는 발견이 나왔다. RaR은 이 문제를 "비교"가 아니라 다른 방식으로 우회한다 — 크고 애매한 판단 하나(1~10점 Likert)를 작고 명확한 이진 판단 여러 개로 쪼갠다. 판단의 단위를 잘게 나눌수록 judge(사람이든 LLM이든)가 흔들리지 않고 일관되게 답할 확률이 올라간다는 점에서, 둘은 "판단을 쪼개거나 상대화해서 신뢰도를 높인다"는 같은 원리를 서로 다른 축에서 구현한다.

## RaR을 시리즈 지형도 위에 놓기

이 시리즈에서 "원칙(principle)을 어디서 가져오는가"라는 질문에 답한 글이 이미 하나 있었다. [#26 DeepSeek-GRM/SPCT 글](/blog/2026/deepseek-grm-spct/)이다. 두 방법은 겉보기엔 비슷해 보이지만 원칙의 출처가 정반대다.

| 축                | DeepSeek-GRM / SPCT (#26)                                      | Rubrics as Rewards (본편)                                   |
| ----------------- | -------------------------------------------------------------- | ----------------------------------------------------------- |
| 채점 기준의 출처  | judge 모델이 강화학습으로 스스로 원칙을 생성                   | 강한 LLM(GPT-4o/o3-mini)이 참조 답안을 근거로 사전 생성     |
| 생성 시점         | 추론 시점, 응답마다 즉석 생성                                  | 학습 전 오프라인, 프롬프트당 한 번 생성 후 재사용           |
| 확장 방법         | inference-time scaling(여러 비평 샘플을 투표·meta RM으로 결합) | aggregation 전략 선택(explicit 가중합 vs implicit 홀리스틱) |
| 무엇을 최적화하나 | reward model 자체의 채점 정확도                                | GRPO 정책의 downstream 성능(HealthBench/GPQA)               |
| 비유              | 즉흥적으로 그날의 기준을 정하는 요리사                         | 미리 정해둔 레시피(rubric)대로 채점하는 대회 심사위원       |

[#7 ArmoRM 글](/blog/2026/armorm/)과 비교하면 축이 하나 더 뚜렷해진다. ArmoRM은 helpfulness·correctness·verbosity처럼 **모든 프롬프트에 공통되는 고정된 소수의 축**을 대규모 선호 데이터로 사전학습하고 MoE 게이팅으로 결합한다. RaR의 rubric은 반대로 **프롬프트마다 새로 생성되는 가변 개수(7~20개)의 항목**이다. ArmoRM의 축은 "이 응답이 전반적으로 얼마나 도움이 되는가"를 묻지만, RaR의 항목은 "이 응답이 '이' 질문에서 응급 신호를 언급했는가"처럼 그 순간의 맥락에 결박돼 있다. 고정 축 분해가 범용성을 사는 대신 세밀함을 잃는다면, instance-specific rubric은 세밀함을 사는 대신 프롬프트마다 새로 rubric을 만들어야 하는 비용을 문다.

# Conclusion

**핵심을 한 줄로: RaR은 "정답은 없지만 좋은 답의 조건은 열거할 수 있다"는 전제로, 프롬프트별 rubric을 checklist 형태의 reward 함수로 바꿔 RLVR을 비검증 도메인까지 확장한다. Direct-Likert judge 대비 HealthBench에서 최대 31%, GPQA-Diamond에서 7%의 상대 향상을 보였고, 이는 안전성·모더레이션처럼 "정답 문자열이 없는" 판단 작업에 RLVR의 안정성을 가져올 수 있다는 뜻이다.**

정리하면,

1. **정식화**: rubric $$\{(w_j, c_j)\}_{j=1}^k$$은 RLVR의 $$k=1$$ 특수 케이스를 일반화한 것이다.
2. **집계**: explicit(가중합, Eq. 1)은 해석 가능하지만 가중치에 따라 순위가 뒤집힐 만큼 brittle하고, implicit(홀리스틱 판정, Eq. 2)이 실험적으로는 더 나은 결과를 냈다.
3. **생성**: rubric은 참조 답안을 근거로 LLM이 합성해도 사람이 쓴 것과 대등한 품질을 낸다 — 단, grounding 없이 생성하면 품질이 확연히 떨어진다.

한계도 논문이 직접 인정한다. 실험이 의료·과학 두 도메인에 한정돼 대화나 도구 사용 같은 더 개방적인 세팅으로의 일반화는 검증되지 않았고, 집계 전략도 explicit·implicit 두 가지만 탐색했다. 더 근본적인 문제는 따로 있다 — RaR이 여전히 **각 항목을 독립적으로 pointwise 스칼라화**한다는 점이다. Open Rubric System(OpenRS, arXiv:2602.14069, Alibaba Qwen 팀)은 바로 이 지점을 정조준한다. OpenRS는 RaR을 포함한 "정적 rubric을 강한 LLM으로 합성해 가중합으로 집계하는" 계열의 방법들이 "discriminability에 내재적 한계(ceiling)를 만들고 reward gaming에 취약하며, 개방형 세팅에서 collapse에 가까운 동역학으로 이어질 수 있다"고 정면으로 지적한다. 대안으로 제시하는 **Pairwise Adaptive Meta-Rubric (PAMR)**은 rubric을 프롬프트마다 미리 고정하는 대신, **비교 대상 두 응답의 의미적 차이에 조건부로** 그때그때 rubric을 생성하고, 항목별로 두 응답을 pairwise 비교한 뒤 그 결과를 judge 내부가 아니라 **외부에서** 집계한다. 정적인 체크리스트를 판 채로 반복 사용하지 않고 비교할 응답 쌍에 맞춰 rubric 자체를 적응시키기 때문에, 정책이 고정된 채점 기준을 미리 파악해 거기에 맞춰 응답을 최적화하기가 훨씬 어려워진다는 논리다. OpenRS는 이 구조로 RM-Bench·JudgeBench·RewardBench v2·PPE Preference 네 개의 reward-modeling 벤치마크에서 스칼라 RM baseline들을 제치고 최상위 결과를 보고한다.

그런데 rubric이든 pairwise adaptive rubric이든, 결국 판정을 내리는 건 여전히 LLM judge라는 사실은 바뀌지 않는다. 항목을 아무리 잘게 쪼개고 비교 방식을 아무리 정교하게 다듬어도, judge 자체가 속는다면 이 모든 구조는 무의미해진다. [#31 One Token to Fool LLM-as-a-Judge 글](/blog/2026/one-token-to-fool-judge/)이 바로 이 지점 — judge를 단 하나의 토큰으로 속일 수 있다는 사실 — 을 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물아홉 번째 글이다.

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

22. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
23. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
24. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
25. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
26. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. **(현재 글)** Rubrics as Rewards (2025) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

본 시리즈는 31편으로 구성된다.

# 참고 문헌

- Gunjal, Wang, Lau, Nath, He, Liu, Hendryx, 2025. [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746). Scale AI.
- [OpenReview: Rubrics as Rewards (ICLR 2026)](https://openreview.net/forum?id=c1bTcrDmt4)
- [Scale AI 공식 발표: ICLR 2026 Accepted Paper](https://x.com/ScaleAILabs/status/2047075931442077777)
- Arora, Wei, Hicks, Bowman, Quiñonero-Candela, Tsimpourlas, Sharman, Shah, Vallone, Beutel, et al., 2025. [HealthBench: Evaluating Large Language Models Towards Improved Human Health](https://arxiv.org/abs/2505.08775).
- Jia, Yang, Wu, Gai, Tao, Zhou, Lin, Jiang, Jiang, 2026. [Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric](https://arxiv.org/abs/2602.14069). Alibaba Qwen Large Model Application Team.
- [GitHub: Qwen-Applications/OpenRS](https://github.com/Qwen-Applications/OpenRS)
