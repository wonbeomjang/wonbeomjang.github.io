---
layout: post
title: "프론티어 모델은 harmlessness reward를 어떻게 설계했나"
date: 2026-08-11 09:45:00 +0900
description: "RLHF Reward 설계 시리즈 #45 — 안전성 reward의 실전 설계와 over-refusal 트레이드오프"
categories: [paper]
tags: [rlhf, safety, harmlessness, over-refusal, reward-model, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 프론티어 모델들의 공개 자료에서 **안전성 reward 설계**만 따로 떼어 비교한다. 능력(helpfulness) 축은 [#44](/blog/2026/frontier-reward-design/)에서 다뤘다.

# Introduction

[#44](/blog/2026/frontier-reward-design/)은 열한 개 프론티어 모델이 helpfulness reward를 어떻게 설계했는지 훑었다. 결론은 비교적 깔끔했다 — 검증 가능한 도메인(수학, 코드)에서는 규칙 기반 reward로 수렴하고, 검증 불가능한 도메인(대화, 글쓰기)에서는 GRM과 rubric judge로 갈렸다. 이 글은 같은 자료를 다시 펼치되, 이번엔 harmlessness(안전성) reward만 따로 뜯는다.

두 축을 한 글에 묶지 않은 데는 두 가지 이유가 있다.

첫째, **문서화되는 정도 자체가 다르다.** helpfulness reward는 대부분의 report가 수학·코드 벤치마크 표로 상세히 공개한다. 반면 안전성 reward는 report마다 격차가 극단적이다. Kimi K3의 47페이지 technical report에는 "safety"라는 단어가 단 한 번도 나오지 않는다. "refusal"은 다섯 번 등장하지만, 전부 다른 모델이 평가 도중 거절했다는 각주다 — 예컨대 "Claude Fable 5가 답변을 거부한 태스크 2건 포함"처럼, 벤치마크 채점 시 발생한 결측치를 설명하는 용도일 뿐 자기 모델의 안전 설계와는 무관하다. 정반대 끝에는 A.X K2가 있다. "safety" 43회, "over-refusal" 2회, "safe completion" 3회, "red-team" 6회. 같은 "technical report"라는 장르 안에 이 정도 편차가 있다는 것 자체가 이 글의 첫 번째 발견이다.

둘째, **안전 reward에는 helpfulness reward에 없는 고유한 실패 모드가 있다.** 유해한 요청을 걸러내도록 reward를 세게 걸면 무해한 요청까지 거절하는 over-refusal이 따라붙는다. "베이킹소다로 배수구 뚫는 법"이 "폭발물 제조법"과 표면적으로 비슷한 화학 키워드를 공유한다는 이유만으로 거절당하는 사례가 전형적이다. capability 쪽에는 없는, harmlessness reward 설계에서만 나타나는 트레이드오프다. 그래서 이 글은 순서대로 (1) helpful과 harmless를 왜 하나의 스칼라에 담을 수 없는지, (2) 프론티어 모델들이 실제로 무엇을 공개했는지, (3) 공개된 방법 중 실제로 안전과 과잉거절을 동시에 개선한 사례가 있는지를 짚는다.

미리 윤곽을 말하면 세 가지다.

1. 오픈 report를 냈다고 안전 설계를 공개하는 건 아니다. 능력·효율은 상세히 쓰면서 안전은 통째로 비우는 경우가 흔하다.
2. 공개된 설계들은 표현은 제각각이어도 공통된 방향을 향한다 — "거절 자체가 아니라 안전한 완수(safe completion)에 보상"한다.
3. 외부 벤치마크로 실측하면([#19 OR-Bench](/blog/2026/or-bench/)) 안전과 과잉거절을 실제로 동시에 개선한 사례는 드물다. 대부분의 모델이 하나의 트레이드오프 곡선 위에서 미끄러질 뿐이다.

먼저 이번에 다루는 자료들이 안전 설계를 얼마나, 어떻게 공개했는지부터 층위를 나눠보면 이렇다.

| 모델         | 문서화 정도              | 근거                                                                   |
| ------------ | ------------------------ | ---------------------------------------------------------------------- |
| A.X K2       | 상세 공개                | "safety" 43회, over-refusal 2회, safe completion 3회, red-team 6회     |
| K-EXAONE 2.0 | 별도 단계로 공개         | 능력 학습과 분리된 safety-aware preference 단계, taxonomy 226 → 296개  |
| DeepSeek-R1  | 파이프라인 단계로만 언급 | 마지막 all-scenario RL에서 helpfulness·harmlessness RM을 재호출        |
| Llama 4      | 결과만 공개, 방법 비공개 | 거절률 7% → 2% 미만이라는 수치는 있으나 데이터·reward 변경 내역은 없음 |
| Gemma 3/4    | 한 줄 표기               | reward 축 목록에 "minimizing model harmfulness" 한 줄                  |
| Magistral    | 범위 밖으로 명시         | 검증 가능한 수학·코드만 다루고 비검증 도메인 자체를 범위 밖에 둠       |
| Kimi K3      | 사실상 비공개            | "safety" 0회, "refusal"은 전부 타 모델 평가 각주                       |

아래에서 이 층위를 만드는 배경부터, 각 모델이 실제로 무엇을 했는지, 그리고 그 결과를 외부에서 재보면 무엇이 남는지 순서대로 짚는다.

# Background

## 왜 helpful과 harmless를 한 스칼라에 담을 수 없나

가장 단순한 설계는 "안전하지 않으면 감점, 도움이 되면 가점"을 하나의 reward로 합치는 것이다. 문제는 이 하나의 숫자 안에서 두 목표의 교환 비율이 암묵적으로 고정된다는 데 있다. 정책이 조금이라도 유해할 확률이 있는 응답을 피하려고 전면 거절 쪽으로 쏠리면, 그 쏠림을 되돌릴 손잡이가 reward 안에 따로 없다. [#8 Llama 2](/blog/2026/llama2-rlhf/)가 이 문제를 논문에서 직접 지적한다.

> "다른 연구들도 helpfulness와 safety가 종종 서로 트레이드오프 관계에 있다는 것을 발견했고, 이는 하나의 reward model이 둘 다에서 잘 작동하기 어렵게 만든다. 이를 해결하기 위해 우리는 두 개의 별도 reward model을 학습시킨다 — 하나는 helpfulness에, 다른 하나는 safety에 최적화된 모델이다."

Llama 2는 신호를 분리하는 데서 멈췄지만, [#15 Safe RLHF](/blog/2026/safe-rlhf/)는 한 걸음 더 나아가 이 분리를 최적화 목적식 자체에 박아 넣는다. helpfulness는 reward $$R$$로, harmlessness는 cost $$C$$로 완전히 분리하고, "cost가 threshold를 넘지 않는다"는 제약을 라그랑주 승수 $$\lambda$$로 건다.

- $$R$$: 응답이 얼마나 도움이 되는지 채점하는 helpfulness reward model의 출력
- $$C$$: 응답이 얼마나 유해한지 채점하는 cost model의 출력 (클수록 위험)
- $$\lambda$$: 제약이 깨졌을 때만 커지는 동적 계수. 안전 기준을 만족하면 다시 줄어든다

핵심은 $$\lambda$$가 고정 가중치가 아니라 **학습 중 데이터를 보고 스스로 조절되는 값**이라는 점이다. 정책이 아직 위험하면 $$\lambda$$가 커져 안전 쪽으로 강하게 밀고, 안전 기준을 넘어서면 $$\lambda$$가 줄어들어 helpfulness에 다시 여유를 준다. 안전을 reward에 섞을 또 하나의 항이 아니라 넘지 말아야 할 제약으로 두는 설계다. 이 결과는 Experiments에서 수치로 확인한다.

[#16 Rule-Based Rewards](/blog/2026/rule-based-rewards/)는 다른 방식으로 같은 원칙을 지킨다. helpfulness reward에 안전 규칙을 만족했는지 채점하는 $$r_{RBR}$$을 더하는 형태다.

$$r_{total} = r_{helpful} + r_{RBR}$$

여기서도 $$r_{helpful}$$과 $$r_{RBR}$$은 별개의 채점 결과이고, 최종 항에서만 더해진다. 두 방식 모두 "신호는 분리하되 최적화는 하나로" 돌린다는 점에서 같은 계열이다.

신호를 분리해야 hacking을 막을 수 있다는 원칙은 안전 도메인만의 이야기가 아니다. [#12 ODIN](/blog/2026/odin-disentangled-reward/)은 길이와 품질을 하나의 reward에 뭉쳐두면 정책이 길이만 늘려 점수를 딴다는 걸 보이고, 두 축을 별개 head로 분리해 막는다. Safe RLHF·RBR이 helpful과 harmless를 분리하는 것과 같은 문제의식이 다른 축에서도 반복해서 등장한다는 뜻이다.

## over-refusal은 reward 설계 문제다

harmlessness reward에만 있는 고유한 실패 모드가 over-refusal이다. 왜 안전 reward를 걸면 무해한 요청까지 거절하게 되는가. [#18 Shallow Safety Alignment](/blog/2026/shallow-safety-alignment/)는 이 현상과 jailbreak 취약성이 **같은 뿌리에서 나온다**고 진단한다.

안전 정렬이 응답의 첫 몇 토큰에만 얹히면(shallow), 모델이 실제로 배우는 건 "위험해 보이는 표면 단어가 나오면 거절 접두어를 붙여라"는 얕은 패턴이다. 이 패턴은 두 방향으로 문제를 일으킨다.

- 유해한 토큰 몇 개를 강제로 앞에 채워 넣는 prefilling 공격에는 거절 습관이 아예 발동하지 않아 쉽게 뚫린다 (취약성)
- 위험해 보이는 표면 단어가 들어간 무해한 요청에는 내용과 무관하게 거절 습관이 발동한다 (과잉거절)

두 실패가 대칭적으로 같은 얕음에서 나온다는 것이 이 글의 핵심이다. reward가 "거절 여부"라는 표면 행동만 채점하고 "거절이 타당한 근거로 이어지는가"를 채점하지 않으면, 정책은 가장 값싼 지름길 — 위험해 보이는 단어에 반사적으로 반응하는 것 — 을 학습한다.

처방도 이 진단에서 그대로 따라 나온다. 유해하게 시작했다가 거절로 복귀하는 safety recovery 예시를 학습에 넣으면, prefill 40토큰 공격 성공률이 57.0% → 4.5%로 떨어지면서도 AlpacaEval 승률 손실은 51.8% → 49.5%로 미미했다. 정렬을 응답 끝까지 깊게 박는 데 드는 비용이 생각보다 크지 않다는 뜻이다. 이 결과는 안전 reward를 설계할 때 "언제 거절 신호를 주는가"만큼 "거절 이후 응답이 어떻게 이어지는가"를 봐야 한다는 걸 보여준다.

# Method

배경을 놓고 보면, 실제 프론티어 모델들이 공개한 설계는 이 원칙(신호 분리, safe completion 지향)을 각자 다른 방식으로 구현한 결과에 가깝다. 먼저 한 장으로 정리한다.

| 모델                            | 안전 신호 조달처                                                                                           | 신호 결합 방식                                                        | 거절 자체를 보상하는가                                |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------- |
| A.X K2                          | red-team 프롬프트 + principle 기반 safety-preference + identity 데이터, LLM judge가 9개 안전 차원으로 채점 | 4그룹 mixture 안의 한 축(비율 고정 없이 동적 조정)                    | 아니다 — safe completion에 보상, 유해 응답에 벌       |
| K-EXAONE 2.0                    | 226 → 296개 위험 영역 taxonomy 기반 preference 데이터                                                      | 능력 학습과 분리된 별도 safety-aware 단계                             | 명시되지 않음                                         |
| DeepSeek-R1                     | helpfulness·harmlessness를 함께 보는 학습된 RM                                                             | 4단계 파이프라인의 마지막 all-scenario RL에서 규칙 reward 대신 재호출 | RM 채점 기준 비공개                                   |
| Llama 4                         | 비공개                                                                                                     | online RL + DPO, 방법은 비공개                                        | 비공개(결과만: 거절률 7% → 2% 미만)                   |
| OpenAI · RBR                    | 소량 예시로 정의한 규칙을 LLM이 채점                                                                       | $$r_{total} = r_{helpful} + r_{RBR}$$로 helpful reward에 가산         | 아니다 — 규칙 위반뿐 아니라 불필요한 거절도 감점      |
| OpenAI · Deliberative Alignment | 안전 스펙 자체를 모델이 CoT로 추론                                                                         | 안전 전용 SFT + RL, RL 중 CoT는 reward model에게 비공개               | 아니다 — jailbreak 견고성과 과잉거절 감소를 함께 채점 |
| Anthropic                       | 입출력 Constitutional Classifier                                                                           | 정책 reward와 별도로 분류기가 게이팅                                  | 거절 증가폭 자체를 +0.38%p로 상한을 두고 관리         |

## A.X K2

이번 비교에서 안전 설계를 가장 상세히 공개한 축이다. 학습 mixture를 instruction following, human preference, agentic tool use, safety 네 그룹으로 나누고 비율을 미리 고정하지 않는다. 중간 RL 체크포인트로 약점을 짚어 조정하는 control surface로 다룬다.

safety 데이터는 red-teaming 프롬프트, safety-preference 데이터, model-identity 데이터로 구성되고, red-team 응답은 LLM judge가 모욕·성인 콘텐츠·불법 행위·혐오·편향 등 harm 카테고리와 정보 관련 차원을 포함한 9개 안전 차원으로 채점한다. 핵심 문장은 report에 명시돼 있다.

> 정책은 안전한 완수(safe completion)에 보상을 받고 유해한 응답에 벌을 받는다 — 단순히 거절했다고 보상받는 것이 아니다.

safety-preference 데이터는 principle 기반 rubric으로 채점하되 데이터셋의 preferred·rejected 응답을 calibration reference로 쓰고, identity 데이터는 모델의 identity policy를 인코딩한 별도 rubric으로 채점한다. 필터링 rubric도 같은 철학을 따른다 — 사용자의 benign한 목적을 유해함 없이 충족할 수 있으면, 전면 거절 대신 유용한 안전 결과물로 우회하는 답을 선호 응답으로 삼는다.

## K-EXAONE 2.0

능력 학습과 별도의 safety-aware preference 최적화 단계를 둔다. 1.0 대비 안전 taxonomy가 226개에서 296개 위험 영역으로 확장됐다는 것 외에, 세부 reward 구성은 report에 상세히 나오지 않는다. taxonomy 확장 자체는 "안전이 커버해야 할 범위"에 대한 인식이 넓어졌다는 신호로는 읽을 수 있지만, 그 범위를 reward에 어떻게 반영했는지는 K-EXAONE 2.0만 봐서는 알 수 없다.

## DeepSeek-R1

4단계 파이프라인의 마지막 all-scenario RL 단계에서, 정답이 없는 일반 대화에 helpfulness·harmlessness를 보는 학습된 RM을 다시 불러온다. 앞 단계에서 수학·코드는 규칙 검증기로 처리했던 모델이, 안전이 걸린 도메인에서는 규칙만으로 부족하다는 걸 스스로 인정하고 학습된 RM으로 되돌아간 셈이다. RLVR로 검증 가능한 도메인을 완전히 대체할 수 있다는 인상과 달리, 안전은 여전히 학습된 RM의 자리로 남아 있다는 뜻이기도 하다.

## Llama 4

공식 블로그가 결과 수치를 낸 드문 사례다. 논쟁적 정치·사회 주제에 대한 거절률이 Llama 3.3의 7%에서 2% 미만으로 줄었고, 특정 관점에서만 불균등하게 거절하는 비율도 1% 미만이라고 밝힌다. 다만 데이터와 reward 중 무엇을 바꿨는지는 공개하지 않는다. 그리고 이 수치는 "덜 거절하기"만을 목표로 잡고 보고한 것이라, 같은 조치가 안전 축 자체(유해 프롬프트 거절률)에 어떤 영향을 줬는지는 함께 보고되지 않는다. 거절률이 줄었다는 사실만으로는 over-refusal이 개선된 건지 안전 기준 자체가 느슨해진 건지 구분할 수 없다는 문제를 그대로 남긴다.

## OpenAI: RBR와 Deliberative Alignment

제품 report 대신 방법론 논문으로 안전 reward 설계를 공개했다. [#16 RBR](/blog/2026/rule-based-rewards/)은 안전 분류 F1을 97.1까지 끌어올렸다(사람 피드백만 쓴 baseline은 91.7). F1은 precision과 recall을 함께 보는 지표이므로, 이 상승은 유해 응답을 더 잘 잡으면서 동시에 무해 응답을 덜 거절했다는 뜻이다.

[#17 Deliberative Alignment](/blog/2026/deliberative-alignment/)는 안전 스펙 자체를 모델이 CoT로 추론하게 만드는 안전 전용 SFT + RL 단계를 별도로 둔다. 설계상 중요한 디테일 하나는, RL 중 이 CoT를 reward model에게 숨긴다는 점이다. reward model이 CoT 내용을 보고 채점하면 모델이 "그럴듯해 보이는 CoT"를 최적화할 유인이 생기는데, 최종 출력만 채점 대상으로 남겨 기만적(deceptive) CoT를 최적화하지 않도록 막는다.

## Anthropic: Constitutional Classifiers

정책 reward 자체를 바꾸는 대신, 입출력에 별도의 분류기를 두고 이를 게이팅에 쓰는 접근이다. jailbreak 대부분을 막으면서도 무해 질의에 대한 거절률 증가를 +0.38%p로 억제했다고 보고한다. 대신 연산 비용은 +23.7% 늘었다. 안전 장치가 유발하는 over-refusal 비용을 스스로 정량화해 공개하는 것 자체가 드문 사례다.

2026년 1월 개정한 constitution은 규칙의 기계적 준수보다 인간의 실질적 이익을 우선하는 방향을 명시했다 — 예를 들어 의료 조언을 회피하라는 규칙 때문에 증상 설명 자체를 거절하는 문제를 경계 사례로 든다. reward 설계 관점에서 보면, 이는 "규칙을 지켰는가"라는 좁은 채점 기준 자체가 over-refusal을 유발할 수 있다는 걸 인정하고 채점 기준을 다시 넓힌 사례다.

# Experiments

## 외부 벤치마크가 본 실제 트레이드오프

지금까지는 각 모델이 자체 보고한 수치였다. [#19 OR-Bench](/blog/2026/or-bench/)는 25개 모델, 8개 모델 패밀리를 안전 프롬프트 거절률과 유해 프롬프트 거절률 두 축으로 함께 재서 외부에서 검증한다. 결과는 자체 보고와 결이 다르다. 안전 프롬프트 거절률과 유해 프롬프트 거절률 사이 Spearman 순위상관이 0.878로, 매우 강한 양의 상관을 보인다. 대부분의 모델이 이 두 축을 독립적으로 개선하지 못하고 하나의 트레이드오프 곡선 위에서 움직인다는 뜻이다. Claude는 가장 안전한 동시에 과잉거절도 가장 심하고, Mistral은 가장 많은 프롬프트를 수용한다. GPT-3.5-turbo는 버전이 올라가며 over-refusal은 개선됐지만 safety는 오히려 하락했다. 즉 자사 발표에서 "과잉거절을 줄였다"고 말하는 것과 별개로, 실제로 트레이드오프 곡선 자체를 옮긴 사례는 드물다.

이 현상은 [#10 Reward Model Overoptimization](/blog/2026/reward-model-overoptimization/)이 지적한 Goodhart 문제와 같은 구조다. 안전 프롬프트 거절률 하나만 밀어붙이면, 측정되지 않던 다른 축(유해 프롬프트 통과)이 조용히 무너진다. 안전 지표 단독 보고가 위험한 이유가 여기 있다 — 그 지표를 최적화한 대가를 다른 지표가 대신 치르고 있을 수 있는데, 그 다른 지표를 재지 않으면 대가 자체가 보이지 않는다.

## 파레토를 실제로 옮긴 사례

곡선 위에서 미끄러지는 게 기본값이라면, 곡선을 옮긴 사례는 무엇이 달랐는지가 중요하다. 이 시리즈가 다룬 방법론 논문 중 안전 지표와 과잉거절·능력 지표를 함께 보고하면서 둘 다 개선한 경우를 모으면 이렇다.

| 사례                       | 안전 지표                                                                      | 능력 · 과잉거절 지표                                                                     | 비고                                     |
| -------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------- |
| Safe RLHF                  | 유해 응답 비율 53.08% → 2.45%                                                  | helpfulness Elo +244.91, harmlessness Elo +268.31                                        | 고정 가중치가 아닌 동적 $$\lambda$$ 제약 |
| RBR                        | 안전 분류 F1 97.1 (사람 피드백 baseline 91.7)                                  | F1이 precision·recall을 함께 반영                                                        | 규칙 위반과 불필요한 거절을 함께 감점    |
| Deliberative Alignment     | StrongREJECT goodness@0.1 0.37 → 0.88, Disallowed Content not_unsafe 0.8 → 0.9 | XSTest not_overrefuse 0.88 → 0.93, Self-Harm Safe Completion style_adherence 0.04 → 0.92 | GPT-4o → o1, CoT를 RM에게 숨김           |
| Constitutional Classifiers | jailbreak 대부분 차단                                                          | 무해 질의 거절률 증가 +0.38%p로 상한                                                     | 연산 비용 +23.7%                         |

Safe RLHF는 유해 응답을 53.08%에서 2.45%로 낮추면서 helpfulness·harmlessness Elo를 동시에 244.91점, 268.31점 올렸다. 고정 가중치 reward-shaping이라면 보통 한쪽을 올리면 다른 쪽이 내려가는데, 제약을 동적으로 조절하는 설계 덕분에 둘 다 밀어 올렸다. Deliberative Alignment는 이 시리즈에서 확인한 것 중 가장 균형 잡힌 사례다. jailbreak 견고성(StrongREJECT goodness@0.1 0.37 → 0.88)과 과잉거절 감소(XSTest not_overrefuse 0.88 → 0.93)를 같은 학습 한 번으로 함께 얻었다. 게다가 이 개선이 특정 공격 유형을 외워서 나온 결과가 아니라는 것도 확인된다. 안전 학습 데이터에 인코딩 기반 jailbreak를 전혀 넣지 않은 o1도 해당 공격에서 0.95를 기록했다(전체 학습한 o1도 0.95, 안전 학습이 아예 없는 baseline은 0.65). 다국어 공격에서도 0.69(전체 학습 0.68, baseline 0.44)로 비슷하게 일반화됐다. 학습에 없던 도메인으로도 신호가 번진다는 뜻이다.

두 사례의 공통점은 안전 지표 하나만 보고하지 않았다는 것이다. Safe RLHF는 helpfulness Elo를, Deliberative Alignment는 XSTest not_overrefuse를 반드시 함께 실었다. 대조군 없이 "안전이 좋아졌다"는 수치만 내놓는 report(Llama 4가 그렇다)는 OR-Bench가 보여준 트레이드오프 구조를 감안하면 검증이 불가능하다.

# Conclusion

이 글은 harmlessness reward 설계를 프론티어 아홉 개 이상 자료에 걸쳐 훑었다. 네 가지로 정리된다.

첫째, **문서화 격차 자체가 신호다.** Kimi K3는 47페이지에서 "safety"를 한 번도 쓰지 않았고, Magistral은 비검증 도메인을 통째로 범위 밖에 뒀다. A.X K2, Deliberative Alignment, Constitutional Classifiers처럼 안전 reward를 상세히 공개한 쪽은 예외에 가깝다. 오픈 report를 냈다는 사실만으로 안전 설계까지 공개했다고 가정하면 안 된다.

둘째, **거절이 아니라 안전한 완수를 보상해야 한다.** A.X K2가 명시적으로 쓴 원칙이지만 RBR, Deliberative Alignment도 결과적으로 같은 방향이다. 거절 자체에 보상을 주는 순간 [#18 Shallow Safety](/blog/2026/shallow-safety-alignment/)가 지적한 얕은 패턴 학습이 시작되고, 그 패턴은 취약성과 과잉거절을 동시에 만든다.

셋째, **신호는 분리하되 최적화는 함께 돌려야 한다.** Llama 2의 두 RM, Safe RLHF의 $$R$$·$$C$$ 분리, RBR의 $$r_{helpful} + r_{RBR}$$, A.X K2의 4그룹 mixture — 표현은 다르지만 전부 helpful과 harmless를 별개의 채점 축으로 유지한다. 진짜 갈림길은 학습을 단계로 나누느냐가 아니라, 두 신호를 하나의 스칼라로 미리 뭉쳐버리느냐다. 뭉치는 순간 사후에 조절할 손잡이가 사라진다.

넷째, **측정에는 반드시 대조군이 있어야 한다.** OR-Bench의 Spearman 0.878은 안전 프롬프트 거절률만 보고하는 지표가 왜 위험한지를 보여준다. Deliberative Alignment와 Safe RLHF가 신뢰할 만한 이유는 안전 지표와 함께 helpfulness·과잉거절 지표를 나란히 실었기 때문이다. 안전 수치 하나만 내놓는 report는, 그게 진짜 파레토 개선인지 단순히 거절을 늘려 얻은 착시인지 구분할 수 없다.

이 넷을 관통하는 결론은 하나다. harmlessness reward 설계는 helpfulness reward 설계보다 어렵다 — 실패 모드가 하나 더 있고(과잉거절), 그 실패 모드는 안전 지표 하나만 봐서는 보이지 않기 때문이다. 다음 글 [#46](/blog/2026/reward-model-design/)에서는 이 시리즈 전체가 쌓은 결론 — helpfulness와 harmlessness를 아우르는 reward 설계 원칙 — 을 한 장으로 정리한다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 마흔다섯 번째 글이다.

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
  <li><strong>(현재 글)</strong> 프론티어의 harmlessness reward 설계 — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 46편으로 구성된다.

# 참고 문헌

- Dai et al. (PKU Alignment), 2023. [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773) — [#15](/blog/2026/safe-rlhf/)에서 다룬 제약 기반 안전 최적화.
- Mu et al. (OpenAI), 2024. [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111) — [#16](/blog/2026/rule-based-rewards/).
- Guan et al. (OpenAI), 2024. [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/abs/2412.16339) — [#17](/blog/2026/deliberative-alignment/).
- Qi et al. (Princeton University), 2024. [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) — [#18](/blog/2026/shallow-safety-alignment/).
- Cui et al., 2024. [OR-Bench: An Over-Refusal Benchmark for Large Language Models](https://arxiv.org/abs/2405.20947) — [#19](/blog/2026/or-bench/).
- Touvron et al. (Meta AI), 2023. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — [#8](/blog/2026/llama2-rlhf/).
- Meta AI, 2025. [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- LG AI Research, 2026. [K-EXAONE 2.0 Technical Report](https://arxiv.org/abs/2608.04505).
- SKT AI, 2026. [A.X K2 Technical Report](https://github.com/SKT-AI/A.X-K2).
- Kimi Team, 2026. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3).
- Gemma Team (Google DeepMind), 2025. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786).
- Anthropic, 2025. [Constitutional Classifiers: Defending Against Universal Jailbreaks](https://www.anthropic.com/research/constitutional-classifiers).
