---
layout: post
title: "Deliberative Alignment: 안전 명세를 모델의 추론 안으로"
date: 2026-08-11 09:16:00 +0900
description: "RLHF Reward 설계 시리즈 #16 — 안전 규칙을 라벨 생성용이 아니라 모델이 직접 읽고 추론하게 만들다"
categories: [paper]
tags: [rlhf, safety, reasoning, over-refusal, deliberative-alignment, paper]
giscus_comments: true
related_posts: true
---

> [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339) (Guan et al., OpenAI, arXiv 2024)

# Introduction

[#14 Safe RLHF](/blog/2026/safe-rlhf/)는 안전을 별도의 cost model과 Lagrangian 제약으로 다뤘다. [#15 Rule-Based Rewards](/blog/2026/rule-based-rewards/)는 사람이 쓴 규칙과 LLM grader로 reward를 합성했다. 두 방법은 접근 방식이 다르지만 구조는 똑같다. 안전 명세(사람이 정의한 규칙이든, cost model이 학습한 선호든)는 **학습 신호를 계산하는 도구**로만 쓰인다.

정책 모델 자신은 그 명세를 한 번도 읽지 않는다. 모델이 배우는 것은 "이 출력이 명세를 얼마나 만족했는가"라는 스칼라 값뿐이고, 왜 그 출력이 안전한지 혹은 위험한지에 대한 추론 과정은 어디에도 남지 않는다.

이렇게 학습된 모델의 실패 패턴은 두 가지로 갈린다. 첫째는 jailbreak에 취약하다는 것이다. 모델은 "이런 문구가 나오면 거절"이라는 표면적 상관관계를 학습하기 쉽고, 프롬프트를 살짝 우회하면 그 상관관계가 깨진다.

둘째는 over-refusal이다. 같은 표면 패턴 매칭 때문에, 안전한 요청도 위험한 요청과 비슷한 단어를 쓴다는 이유만으로 거절당한다. 두 실패는 한 뿌리에서 나온다. 모델이 명세를 이해하는 게 아니라 명세가 반영된 reward의 그림자만 보고 행동을 조정했기 때문이다.

Deliberative Alignment(Guan et al., OpenAI 2024)는 이 구조를 뒤집는다. 안전 명세(safety specification) 텍스트를 reward 계산용 부속품이 아니라, 모델이 **직접 읽고 추론하는 대상**으로 삼는다. 답을 내기 전에 관련 정책을 명시적으로 recall하고, 그 정책에 비추어 현재 요청이 안전한지 자기 CoT 안에서 추론하도록 학습시킨다. 이 방법은 OpenAI의 o-series 정렬에 실제로 사용됐다.

결과는 두 실패 모드를 동시에 줄이는 파레토 개선으로 나타난다. jailbreak 저항성을 보는 StrongREJECT와 불필요한 거절을 보는 XSTest가 함께 개선됐다. 게다가 이 학습 절차는 사람이 작성한 CoT나 정답 완성을 전혀 요구하지 않는다. 사람의 노력은 데이터 라벨링이 아니라 명세 작성과 최종 평가에만 들어간다.

# Background

지금까지 이 시리즈에서 본 안전 정렬 방법들의 공통 파이프라인을 정리하면 다음과 같다. 명세(사람이 쓴 규칙, 헌법 원칙, 혹은 학습된 cost function) → 그 명세를 참조하는 판단자(사람 라벨러, cost model, 혹은 LLM grader) → 스칼라 학습 신호 → 정책 업데이트. [Constitutional AI 글](/blog/2026/constitutional-ai/)도 마찬가지다. 헌법 원칙은 self-critique와 revision을 생성하는 데 쓰이고, 그렇게 만들어진 (critique 반영 후) 응답으로 학습이 끝나면 원칙 텍스트 자체는 최종 정책의 입력에서 사라진다. 정책은 원칙이 만들어낸 결과물만 흡수할 뿐, 원칙을 스스로 인용하며 추론하는 법은 배우지 않는다.

Deliberative Alignment는 여기서 한 가지 전제를 이용한다. o1류 reasoning model은 답을 내기 전에 explicit CoT를 생성할 수 있다는 전제다. [#32 DeepSeek-R1](/blog/2026/deepseek-r1/)에서 본 것처럼 이런 CoT는 RL로 스스로 길어지고 정교해질 수 있다. Deliberative Alignment는 이 능력을 안전 판단에도 적용한다. 안전 여부를 "패턴 매칭 후 즉답"이 아니라 "관련 정책을 recall → 현재 상황에 적용 → 결론"이라는 명시적 추론 체인으로 만들면, 모델은 명세의 표면 패턴이 아니라 명세가 담고 있는 **판단 기준**을 배우게 된다는 것이 핵심 가설이다.

이 논문에는 표기가 세 개 나온다. 하나씩 짚어보면 다음과 같다.

- $$\mathcal{G}_{base}$$: 아직 안전 학습을 거치지 않은 helpful-only 베이스 모델. Stage 1에서 명세를 읽고 (CoT, output)을 생성하는 역할을 한다.
- $$\mathcal{G}_{RM}$$: 안전 명세에 접근 권한을 가진 reward model. Stage 1에서는 생성된 완성을 걸러내는 필터로, Stage 2에서는 RL의 reward를 주는 judge로 두 번 쓰인다.
- $$\text{spec}(\text{category})$$: 안전 카테고리별로 정리된 정책 문서. Stage 1의 생성 단계에서만 프롬프트에 등장하고, 학습 데이터를 만들 때는 다시 떼어낸다.

뒤에서 보겠지만 $$\mathcal{G}_{base}$$와 $$\mathcal{G}_{RM}$$의 역할 분리, 그리고 $$\text{spec}$$이 프롬프트에 있다가 사라지는 이 흐름 자체가 파이프라인 전체를 지탱한다.

비유하자면 이렇다. RBR 방식은 오픈북 시험에서 감독관만 채점 기준표를 들고 있는 구조에 가깝다. 학생(정책)은 기준표를 모른 채 답안을 쓰고, 감독관이 기준표에 맞춰 점수만 알려준다. 학생은 점수의 오르내림을 보고 어떤 답이 좋았는지 짐작할 뿐, 기준표 자체를 읽어본 적은 없다.

Deliberative Alignment는 이 구조를 바꾼다. 학생에게도 채점 기준표를 미리 읽게 하고, 답안 옆에 "이 기준에 따르면 이렇게 판단해야 한다"는 풀이 과정을 쓰도록 연습시킨다. 다만 실제 시험(추론 시점)에서는 기준표를 다시 걷어간다. 학생에게 남는 것은 기준표 자체가 아니라, 기준표를 보고 훈련하는 동안 몸에 밴 풀이 습관이다.

# Method

## Stage 1: SFT via Context Distillation

첫 단계의 목표는 "명세를 참조해 추론하는 능력"을 데이터로 만들어 모델에 주입하는 것이다. 순서는 다음과 같다.

1. 프롬프트마다 안전 카테고리 라벨(성적 콘텐츠, 자해, 위험물 제조 등)을 붙이고, 카테고리별로 정책 문서 $$\text{spec}(\text{category})$$를 구성한다.
2. helpful-only 베이스 모델 $$\mathcal{G}_{base}$$에게 [원래 프롬프트 + $$\text{spec}(\text{category})$$ + "관련 정책을 인용하며 추론하라"는 지시]를 준다. 이 상태에서 $$\mathcal{G}_{base}$$가 (CoT, output) 쌍을 생성한다.
3. 생성이 끝나면 프롬프트에서 $$\text{spec}(\text{category})$$를 떼어낸다. 남는 것은 {prompt, CoT, output} 삼중항이다. 즉 학습 데이터의 입력에는 명세가 없고, 명세를 보고 만들어진 결과물(CoT + output)만 남는다.
4. $$\mathcal{G}_{RM}$$으로 이 완성들을 채점해 필터링한다. $$\mathcal{G}_{RM}$$은 명세에 접근할 수 있으므로, 정책을 정확히 따른 고득점 완성만 걸러낼 수 있다.
5. 이렇게 걸러진 {prompt, CoT, output} 데이터로 SFT를 한다.

여기서 "context distillation"이라는 이름의 의미가 드러난다. 명세라는 context를 프롬프트에 직접 넣어 답을 뽑아낸 뒤, 그 context를 다시 지워버리고 결과만 모델의 가중치에 흡수시킨다.

추론 시점에는 프롬프트 어디에도 명세가 없지만, 모델은 SFT를 통해 "이런 상황에서는 이런 정책이 관련되고, 이렇게 적용된다"는 패턴을 학습해뒀기 때문에 스스로 명세를 recall하는 것처럼 행동한다.

## Stage 2: RL

두 번째 단계는 안전 관련 프롬프트에 대한 RL이다. 명세에 접근 가능한 $$\mathcal{G}_{RM}$$이 정책 준수 여부를 기준으로 reward를 준다. 구조만 보면 [#15 Rule-Based Rewards](/blog/2026/rule-based-rewards/)의 LLM grader와 비슷하다. 규칙(명세)에 접근한 judge가 채점한다는 점은 같다.

결정적인 차이는 CoT를 다루는 방식에 있다. RL 도중 $$\mathcal{G}_{RM}$$은 모델의 CoT를 보지 못한다. output만 보고 채점한다. 이렇게 하는 이유는 명확하다. CoT까지 reward의 대상이 되면, 모델은 "그럴듯해 보이는 CoT를 써서 grader를 통과시키는" 방향으로 최적화될 위험이 있다. 즉 진짜 추론이 아니라 겉보기만 정책을 인용하는 기만적(deceptive) CoT를 학습할 수 있다. CoT를 reward 계산에서 제외함으로써, CoT는 오직 output의 품질을 통해서만 간접적으로 강화된다. 모델이 정책을 실제로 따르는 output을 내야 보상을 받고, 그 output을 내는 데 도움이 된 추론 과정만 살아남는다.

이 파이프라인 전체에서 사람이 쓴 CoT나 정답 완성은 등장하지 않는다. 논문은 이를 명시적으로 언급한다: "our training procedure requires no human-labeled completions." 사람의 노력이 들어가는 지점은 명세 문서를 작성하는 것과, 최종 결과를 평가하는 것뿐이다. 프롬프트에 안전 카테고리를 붙이는 라벨링도 쓰이긴 하지만, 이는 어디까지나 context-distillation 단계에서 어떤 spec을 붙일지 정하기 위한 보조 수단이고 필수는 아니라고 논문은 밝힌다.

두 단계로 나누는 이유도 짚을 만하다. Stage 1(SFT)은 "명세를 읽고 추론하는 형식"을 모델에 각인시키는 역할을 한다. 이 상태의 모델은 이미 정책을 인용하는 습관을 갖지만, 그 판단이 항상 정확하다는 보장은 없다. Stage 2(RL)는 같은 명세를 기준으로 실제 output의 정확도를 강화한다. SFT가 "형식"을 만들고 RL이 "정확도"를 끌어올리는 이분업인 셈이다.

## RBR과의 대비

[#15 Rule-Based Rewards](/blog/2026/rule-based-rewards/)와의 관계를 정리하면 이 논문의 위치가 분명해진다. 둘 다 "규칙 있는 judge가 reward를 계산한다"는 구조는 공유한다. 차이는 그 규칙이 정책 모델의 어디에 닿는가에 있다.

| 측면                          | Rule-Based Rewards (#15)                               | Deliberative Alignment (#16)                                                                         |
| ----------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 명세가 실제로 쓰이는 곳       | reward 계산 단계의 LLM grader 안(rubric)               | 정책의 CoT 생성 과정 자체(context distillation)                                                      |
| 정책이 명세를 추론하는가      | 아니오 — 정책은 명세를 본 적 없이 결과 reward만 받는다 | 예 — SFT 데이터 생성 시 명세를 읽고 추론한 CoT를 학습해, 이후 스스로 관련 정책을 recall하며 추론한다 |
| reward 계산 시 명세 접근 주체 | rubric을 참조하는 LLM grader                           | $$\mathcal{G}_{RM}$$ (구조적으로 유사)                                                               |
| CoT의 역할                    | 명시적 역할 없음(보통 최종 응답만 채점)                | 핵심 — 정책 인용이 CoT 안에 나타나고, RL 중 CoT는 $$\mathcal{G}_{RM}$$에게 숨겨 기만을 방지          |
| 필요한 사람 자원              | 규칙 작성 + 소량의 이상적 응답 예시                    | spec 문서 작성(+선택적 카테고리 라벨) — CoT·정답 완성 불필요                                         |
| 요약                          | 규칙을 reward 함수 쪽에 둔다                           | 규칙(명세)을 모델 안으로 넣는다                                                                      |

이 표의 마지막 줄이 이 글의 핵심이다. RBR은 "무엇이 옳은 답인가"를 심판하는 도구를 정교하게 만들었다. Deliberative Alignment는 심판 도구는 비슷하게 두되, 그 도구가 참조하는 기준을 정책 자신도 추론하도록 만든다.

## 토이 예제: 모호한 요청 하나

"이 두 약물을 같이 먹으면 왜 위험한지 설명해줘. 나 약대생이고 상호작용 리포트를 쓰고 있어" 라는 요청을 생각해보자. 표면적으로는 자해나 위해 관련 키워드(약물 조합, 위험)와 겹치기 때문에 패턴 매칭 기반 시스템이라면 거절하기 쉬운 케이스다. 하지만 맥락(약대생, 리포트 작성)은 정당한 교육 목적을 가리킨다. Deliberative Alignment로 학습된 모델의 추론 과정은 대략 이런 단계를 거친다.

1. **카테고리 recall**: 요청이 "약물/화학" 관련 안전 카테고리에 해당함을 인식하고, 관련 정책 조항을 떠올린다. 예를 들어 "위해 정보라도 교육·의료·안전 목적의 맥락에서는 safe completion으로 정보를 제공한다"는 조항이다.
2. **맥락 검토**: 요청에 담긴 신호(전문 용어 사용, 리포트 작성이라는 목적 진술, 특정 개인을 겨냥한 실행 지침 요청이 아님)를 정책 조항의 조건과 대조한다.
3. **판단**: 조건을 만족하므로 이 요청은 "거절 대상"이 아니라 "safe completion 대상"으로 분류된다. 즉 위험성 자체를 숨기지 않고 설명하되, 실행을 돕는 형태(예: 정확한 치사량 계산법)는 피하는 방향으로 응답을 구성한다.
4. **출력**: 약리 기전과 상호작용 위험을 설명하는 답을 낸다. 거절 문구는 등장하지 않는다.

이 과정 전체가 CoT 안에서 명시적으로 일어난다는 점이 중요하다. RBR의 grader라면 "이 답이 규칙 몇 번을 만족하는가"만 사후적으로 채점하지만, 여기서는 모델 자신이 "이 규칙이 왜 지금 관련 있고, 어떻게 적용되는가"를 answer 이전에 스스로 전개한다. 같은 추론 체인이 반대 방향으로도 작동한다. 만약 요청이 "특정인에게 몰래 먹일 약물 조합을 알려달라"는 식으로 바뀌면 2번 단계에서 맥락 신호가 달라지고, 3번 판단은 거절로 갈린다. 하나의 명세를 두고 모델이 맥락에 따라 다른 결론에 도달한다는 것이, 카테고리 단위로 뭉뚱그려 거절/허용을 정하는 방식과의 차이다.

# Experiments

GPT-4o(안전 학습 이전 helpful-only 베이스에 가까운 비교군)와 Deliberative Alignment로 학습된 o1을 비교한 결과는 다음과 같다.

| 벤치마크                                 | 지표            | GPT-4o | o1       |
| ---------------------------------------- | --------------- | ------ | -------- |
| StrongREJECT                             | goodness@0.1    | 0.37   | **0.88** |
| XSTest                                   | not_overrefuse  | 0.88   | **0.93** |
| Disallowed Content (Challenging Refusal) | not_unsafe      | 0.8    | **0.9**  |
| Hard Refusal Style                       | style_adherence | 0.72   | **0.79** |
| Self-Harm Safe Completion                | style_adherence | 0.04   | **0.92** |

가장 눈에 띄는 조합은 첫 두 줄이다. StrongREJECT는 jailbreak 프롬프트에 대한 견고성을 재는 벤치마크이고, XSTest는 안전한 요청을 안전하다고 인식하는지(과잉 거절을 안 하는지)를 잰다. 이 둘은 보통 트레이드오프 관계다. 거절을 늘리면 jailbreak엔 강해지지만 over-refusal이 늘고, 거절을 줄이면 반대가 된다. o1은 StrongREJECT에서 0.37 → 0.88로 크게 오르면서 동시에 XSTest에서도 0.88 → 0.93으로 올랐다. 트레이드오프 곡선 위에서 이동한 게 아니라 곡선 자체를 바깥으로 밀어낸 파레토 개선이다. Self-Harm Safe Completion의 격차(0.04 → 0.92)도 크다. 이 지표는 자해 관련 요청을 무조건 거절하는 대신 정해진 스타일(공감 + 안전 자원 안내)로 안전하게 완수했는지를 본다. 거절이라는 손쉬운 선택 대신, 명세가 요구하는 정확한 응답 형태를 따랐다는 뜻이다.

일반화 실험은 이 방법이 명세의 표면 문구를 암기한 게 아니라는 것을 보여준다. 250개의 금지 프롬프트에 대해 안전 학습 데이터의 일부를 의도적으로 빼고 학습시킨 뒤 성능을 비교했다.

| OOD 카테고리          | 해당 안전 데이터 미학습 o1 | 전체 데이터 학습 o1 | 안전 학습 없는 baseline |
| --------------------- | -------------------------- | ------------------- | ----------------------- |
| 인코딩 기반 jailbreak | 0.95                       | 0.95                | 0.65                    |
| 다국어 jailbreak      | 0.69                       | 0.68                | 0.44                    |

인코딩 기반 jailbreak(예: base64, 문자 치환 등으로 위장한 요청) 안전 데이터를 한 번도 보지 않은 o1이 0.95로, 그 데이터를 포함해 학습한 o1(0.95)과 사실상 같은 성능을 냈다. 다국어 jailbreak도 마찬가지로 0.69 대 0.68이다. 두 경우 모두 baseline(0.65, 0.44)과는 뚜렷한 격차를 유지했다. 이는 모델이 "인코딩된 요청을 거절하라"거나 "이 언어의 요청을 거절하라" 같은 표면 규칙을 암기한 게 아니라, 디코딩하거나 번역한 뒤 원래 요청의 의미에 명세를 적용하는 추론 능력 자체를 일반화했다는 뜻으로 읽힌다. [#33 Prometheus 2](/blog/2026/prometheus-2/)에서 본 rubric 조건부 judge와 마찬가지로, 여기서도 $$\mathcal{G}_{RM}$$은 고정된 스코어러가 아니라 spec이라는 조건을 받아 판단을 바꾸는 채점자다. 다만 이 논문은 그 spec-조건부 판단 능력을 judge뿐 아니라 정책 자신에게도 심어 넣었다는 점이 다르다.

# Conclusion

Deliberative Alignment의 핵심 메시지는 하나다. 안전 명세를 reward 계산에만 쓰지 말고, 모델이 답하기 전에 직접 recall하고 추론하는 대상으로 만들면 jailbreak 저항과 over-refusal 감소를 동시에 얻을 수 있다는 것이다. [#15 Rule-Based Rewards](/blog/2026/rule-based-rewards/)가 규칙을 정교한 채점 도구로 다듬는 방향이었다면, 이 논문은 같은 채점 구조를 유지하면서 규칙의 목적지를 판단자에서 정책 자신으로 옮겼다. 그리고 이 전환에 사람이 쓴 CoT나 정답 완성이 필요 없었다는 점은, [#32 DeepSeek-R1](/blog/2026/deepseek-r1/) 계보에서 본 "추론은 스스로 길러진다"는 관찰이 안전 영역에도 적용됨을 보여준다.

한계도 분명하다. 이 방법은 $$\mathcal{G}_{RM}$$이 명세를 정확히 반영해 채점할 수 있다는 전제, 그리고 $$\mathcal{G}_{base}$$가 애초에 명세를 읽고 그럴듯한 추론을 생성할 수 있을 만큼 충분히 능력 있는 모델이라는 전제 위에 서 있다. reasoning 능력이 약한 모델에는 이 파이프라인 자체가 잘 작동하지 않을 가능성이 크다. 또한 RL 중 CoT를 grader에게 숨기는 장치는 명시적 기만은 막지만, output만으로는 드러나지 않는 형태의 추론 편향까지 막아준다는 보장은 없다. 안전 판단을 "명세를 읽고 추론하는 능력"에 의존하게 만든 만큼, 그 추론 능력 자체의 신뢰성이 이후 시리즈([#43 프론티어 모델의 reward 설계](/blog/2026/frontier-reward-design/))에서 다시 다뤄질 문제로 남는다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열여섯 번째 글이다.

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
  <li><strong>(현재 글)</strong> Deliberative Alignment (2024) — 안전 명세를 모델의 추론 안으로</li>
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

- Guan et al. (OpenAI), 2024. [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339).
- Bai et al. (Anthropic), 2022. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — 명세를 라벨 생성에만 쓴 선행 접근([별도 글](/blog/2026/constitutional-ai/)).
- Mu et al. (OpenAI), 2024. [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111) — [#15](/blog/2026/rule-based-rewards/), 규칙을 reward 쪽에 두는 대비 사례.
- Dai et al. (Peking University), 2023. [Safe RLHF](https://arxiv.org/abs/2310.12773) — [#14](/blog/2026/safe-rlhf/), 안전을 제약으로 다루는 접근.
- DeepSeek-AI, 2025. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — [#32](/blog/2026/deepseek-r1/), 추론을 RL로 길러낸 계보.
- Kim et al., 2024. [Prometheus 2](https://arxiv.org/abs/2405.01535) — [#33](/blog/2026/prometheus-2/), 명세·rubric을 조건으로 받는 judge.
