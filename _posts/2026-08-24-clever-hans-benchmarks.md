---
layout: post
title: "표층 특징이 정답을 예측한다 — Clever Hans in LLM benchmarks"
date: 2026-08-24 09:05:00 +0900
description: "LLM 평가 체계 시리즈 #5 — 벤치마크 절반 가까이가 문항을 읽지 않고도 풀린다: n-gram만으로 정답을 예측하는 Clever Hans 효과의 실증 (Pacchiardi et al., Cambridge, arXiv 2024)"
categories: [paper]
tags: [evaluation, benchmark, clever-hans, shortcut-learning, nlp, llm, paper]
giscus_comments: true
related_posts: true
---

> [Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/abs/2410.11672) (Pacchiardi et al., University of Cambridge, arXiv 2024)

# Introduction

[#4](/blog/2026/benchmark-construct-validity/)에서 이 논문을 38줄로 요약하고 넘어갔다. 다섯 편의 논문을 한 글에 묶어야 했으니 어쩔 수 없는 선택이었지만, 다시 들여다보니 이 논문 하나만으로 독립된 글 한 편 분량의 실험이 들어 있었다. 19개 벤치마크, 44개 LLM, 13가지 표층 특징 벡터, 11개 모델 계열에 대한 t-검정 — 이 정도 규모의 실증을 각주처럼 다루는 건 아깝다. 그래서 이 논문만 떼어 다시 쓴다.

이 글이 다루는 질문은 하나다. **LLM 벤치마크의 정답을, 문항의 의미를 전혀 이해하지 않고도 예측할 수 있는가?** 저자들의 답은 명확하다 — 19개 벤치마크 중 9개(약 47%)에서 그렇다. 문항 텍스트에 어떤 단어가 몇 번 나오는지만 세는 로지스틱 회귀 분류기가, 지문을 읽지 않고도 우연 수준을 훌쩍 넘는 정확도로 정답을 맞힌다. 그리고 몇몇 LLM 계열은 바로 그 "쉽게 뚫리는" 문항에서 유독 더 잘 맞히는 경향을 보인다 — 즉 모델이 실제로 이 지름길을 쓰고 있을 가능성이 있다는 뜻이다.

[#1](/blog/2026/what-is-evaluation/)에서 세운 두 잣대를 다시 불러오면, 이 논문은 그중 **구성개념 무관 분산(construct-irrelevant variance)**의 가장 선명한 실증 사례집이다. 벤치마크가 재려는 것은 "추론 능력"인데, 점수의 상당 부분이 추론과 아무 상관 없는 "이 문항에 어떤 단어가 몇 번 나오는가"로 설명된다. 이 글은 1부의 마지막 편으로서 그 실증을 끝까지 따라간다.

이 글에서 답할 질문은 세 가지다.

1. **계보**: 이 발견은 2024년에 처음 나온 게 아니다. 2018년 NLI 연구부터 이어지는 "표층 신호가 정답을 흘린다"는 반복된 발견의 계보가 있다. 그 계보를 먼저 세운다.
2. **실험**: Pacchiardi et al.이 정확히 무엇을 측정했는가 — 어떤 특징을, 어떤 지표로, 어떻게 검증했는가.
3. **처방**: 이 문제를 피하려면 벤치마크 제작자가 공개 전에 무엇을 검사해야 하는가.

# Background

## Clever Hans — 조련사의 몸짓을 읽은 말

20세기 초 독일 베를린에 "영리한 한스(Kluger Hans)"라는 말이 있었다. 조련사 빌헬름 폰 오스텐이 발굽을 두드리는 방식으로 한스에게 산수를 가르쳤다고 주장했고, 실제로 한스는 "3 더하기 4는?" 같은 질문에 발굽을 일곱 번 두드려 정답을 맞혔다. 수년간 대중과 일부 과학자들이 이를 진짜 산술 능력으로 믿었다.

1907년 심리학자 오스카 풍스트가 이를 조사한 결과, 한스는 산수를 전혀 하지 못했다. 대신 질문자가 무의식적으로 보이는 미세한 신체 신호 — 정답 근처의 두드림 횟수에 다다르면 질문자의 자세나 호흡이 미묘하게 바뀌는 것 — 를 읽고 있었다. 질문자가 정답을 모르거나 한스가 질문자를 볼 수 없으면 한스는 정답을 맞히지 못했다. 즉 **한스가 실제로 학습한 것은 "산수"가 아니라 "질문자의 몸짓을 읽는 법"이었다.** 과제를 푸는 데 필요하다고 여겨진 능력(산수)과 실제로 사용된 단서(조련사의 몸짓)가 서로 다른 것 — 이것이 Clever Hans 효과다.

이 비유가 벤치마크에 대응하는 지점은 정확하다. 벤치마크 설계자는 "이 문항을 풀려면 인과 추론이 필요하다"고 믿고 문항을 만든다. 하지만 문항 텍스트 자체에 인과 추론과 무관한 통계적 규칙성 — 특정 단어의 등장, 문장 길이, 어휘 다양성 — 이 정답과 우연히 상관돼 있다면, 모델은 인과 추론 없이도 그 규칙성만으로 정답을 맞힐 수 있다. 조련사가 자기도 모르게 몸짓으로 답을 흘리듯, **벤치마크 제작자도 자기도 모르게 문항 텍스트에 답을 흘린다.**

## 계보 — 이 발견은 2024년이 처음이 아니다

Pacchiardi et al.의 실험은 새로운 질문이 아니라, 2018년부터 NLI(자연어 추론) 커뮤니티가 반복해서 확인해 온 현상을 현대 LLM 벤치마크 전반으로 확장한 것이다. 이 계보를 순서대로 짚어보면 왜 이 문제가 구조적인지 드러난다.

### Gururangan et al. (2018) — 가설 문장만 보고도 SNLI를 푼다

[Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/) (Gururangan et al., University of Washington / CMU / NYU, NAACL 2018)

NLI 과제는 전제(premise)와 가설(hypothesis) 두 문장을 보고 함의·중립·모순 중 하나를 고르는 것이다. 저자들은 **전제를 아예 주지 않고 가설 문장만** fastText 분류기에 입력했다. 결과는 다음과 같다.

| 데이터셋              | 가설만 본 분류기 | 다수결 기준선 |
| --------------------- | ---------------- | ------------- |
| SNLI                  | 67.0%            | 34.3%         |
| MultiNLI (matched)    | 53.9%            | 35.4%         |
| MultiNLI (mismatched) | 52.3%            | 35.2%         |

전제 문장을 단 한 글자도 보지 않고도 SNLI의 3분의 2를 맞힌다. 원인은 크라우드워커의 **작문 습관**이었다. 함의 가설은 "동물", "사람" 같은 일반화된 단어를 쓰는 경향이 있었고(전제의 "개"를 "동물"로 일반화), 중립 가설은 "가장", "매우" 같은 최상급·수식어와 목적절("~하려고")을 덧붙이는 경향이, 모순 가설은 "아니다", "결코", "없다" 같은 부정어를 쓰는 경향이 뚜렷했다. 게다가 **가설 길이 자체가 신호였다** — SNLI에서 중립 가설의 중위 길이는 9토큰인데, 함의 가설의 60%는 7토큰 이하였다. 즉 "가설이 짧으면 함의, 길면 중립일 확률이 높다"는 규칙만으로도 우연 이상의 성능이 나온다.

### Poliak et al. (2018) — 같은 현상을 10개 데이터셋에서 체계적으로

[Hypothesis Only Baselines in Natural Language Inference](https://arxiv.org/abs/1805.01042) (Poliak et al., Johns Hopkins University, *SEM 2018)

Gururangan et al.과 거의 동시에, 독립적으로 같은 실험을 열 개의 서로 다른 NLI류 데이터셋(SNLI, MultiNLI, SICK, SciTail, JOCI 등)에 걸쳐 수행했다. InferSent 기반 가설 전용(hypothesis-only) 모델을 학습시킨 결과, **10개 중 6개**에서 다수결 기준선을 유의하게 웃돌았다.

| 데이터셋              | 가설만 본 모델 (test) | 다수결 기준선 | 상대 개선폭 |
| --------------------- | --------------------- | ------------- | ----------- |
| SNLI                  | 69.00%                | 34.28%        | +101.28%    |
| MultiNLI (matched)    | 55.52%                | 35.45%        | +56.61%     |
| MultiNLI (mismatched) | 55.18%                | 35.22%        | +56.67%     |

사람이 직접 가설을 창작한("elicited") 데이터셋일수록 이 격차가 크게 벌어졌다 — SNLI에서는 다수결 대비 정확도가 두 배를 넘었다. 저자들의 결론은 이후 이 계보 전체를 관통하는 문장이다: **"이 발견은 NLI 데이터셋 자체의 구성 방식에 내재한 편향을 가리키며, 다수결 기준선이 과제의 난이도를 제대로 반영하지 못할 수 있다."**

### McCoy, Pavlick & Linzen (2019) — HANS, 휴리스틱을 진단하는 데이터셋

[Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://aclanthology.org/P19-1334/) (McCoy et al., Johns Hopkins University / Brown University, ACL 2019)

앞의 두 논문이 "표층 신호가 존재한다"를 보였다면, 이 논문은 한발 더 나아가 **모델이 정확히 어떤 휴리스틱을 채택했는지**를 진단하는 평가셋(HANS)을 설계했다. 세 가지 구문적 휴리스틱을 정의한다.

| 휴리스틱                   | 정의                                                   | 실패 예시                                                                            |
| -------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 어휘 중복(lexical overlap) | 가설의 모든 단어가 전제에 있으면 함의로 판단           | "The doctor was paid by the actor." → "The doctor paid the actor." (오답: 함의 아님) |
| 부분수열(subsequence)      | 전제의 모든 연속 부분열을 함의로 판단                  | "The doctor near the actor danced." → "The actor danced." (오답)                     |
| 구성요소(constituent)      | 전제의 구문 트리상 모든 완전한 하위 트리를 함의로 판단 | "If the artist slept, the actor ran." → "The artist slept." (오답)                   |

이 휴리스틱들이 통계적으로 유리한 이유는 MNLI 학습 데이터 자체의 불균형에 있다. 어휘 중복 휴리스틱이 맞는 사례가 2,158건인데 반박하는 사례는 261건뿐이다(부분수열 1,274 대 72, 구성요소 1,004 대 58). 즉 학습 데이터가 이 휴리스틱을 채택하도록 통계적으로 유도한다. 실제로 BERT를 포함한 MNLI 학습 모델들은 휴리스틱이 맞는 사례(함의)에서는 거의 항상 정답을 내지만, 휴리스틱이 틀리는 사례(비함의, 우연 수준이 50%)에서는 **대부분 10% 미만의 정확도**로 무너졌다. 이는 이 모델들이 진짜로 함의 관계를 판단하는 게 아니라 휴리스틱을 학습했다는 강력한 증거다.

### Niven & Kao (2019) — BERT가 "not" 하나로 논증을 "이해"한다

[Probing Neural Network Comprehension of Natural Language Arguments](https://aclanthology.org/P19-1459/) (Niven & Kao, National Cheng Kung University, ACL 2019)

ARCT(Argument Reasoning Comprehension Task)는 주장·근거가 주어졌을 때 두 개의 근거(warrant) 후보 중 논리적으로 타당한 쪽을 고르는 이지선다 과제다. BERT는 이 과제에서 **77%**의 정확도를 냈는데, 이는 훈련받지 않은 사람의 평균 정확도(79.8%)보다 겨우 3점 낮은 수치였다. 저자들은 이 놀라운 성능이 온전히 **통계적 단서 활용**으로 설명된다는 것을 보였다.

핵심 단서는 유니그램 "not"이었다. 데이터셋 전체에서 근거 문장에 "not"이 있는 쪽을 고르면 **61%가 정답이었고(productivity 0.61), 이 단서는 전체 문항의 64%에 적용 가능했다(coverage 0.64)**. "will not", "cannot" 같은 바이그램도 비슷하게 유효했다. 실제로 BERT에서 **근거(warrant) 문장만 보여주고** 주장·근거를 가려도 정확도는 71%까지만 떨어졌다 — 즉 77% 중 71%는 근거 문장의 단어 패턴만으로 설명되고, 나머지 6점만 주장·이유와의 관계에서 나왔다.

저자들은 이를 검증하기 위해 **적대적 데이터셋**을 만들었다. 원 데이터의 논리 구조상 "이유 ∧ 대안근거 → ¬주장"이 성립하므로, 각 문항의 주장을 부정하고 정답 레이블을 뒤집은 사본을 추가하면 통계적 단서의 분포가 두 레이블에 걸쳐 대칭이 된다. 이 적대적 데이터셋에서 BERT의 최고 성능은 **53%**로, 평균·중앙값은 **50%**(이지선다 우연 수준)로 떨어졌다. 즉 통계적 단서를 제거하자 BERT의 "논증 이해 능력"은 완전히 사라졌다.

### Kavumba et al. (2019) — "Clever Hans"라는 이름 자체의 NLP 선례

[When choosing plausible alternatives, Clever Hans can be clever](https://arxiv.org/abs/1911.00225) (Kavumba et al., Tohoku University, arXiv 2019)

"Clever Hans"라는 표현을 NLP 벤치마크에 처음 붙인 것은 Pacchiardi et al.이 아니다. 이 논문은 인과관계 이지선다 과제인 COPA에서, 개별 단어 하나가 이미 강력한 예측 신호가 된다는 것을 보이며 이 표현을 제목에 직접 썼다. Pacchiardi et al.도 관련 연구 절에서 이 논문을 직접 인용한다 — 즉 2024년 논문의 제목은 새 비유를 만든 게 아니라, 2019년부터 NLP 커뮤니티 안에서 이미 통용되던 이름을 현대 LLM 벤치마크 19개로 확장 적용한 것이다.

### Geirhos et al. (2020) — 이 현상을 일반화한 관점

[Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) (Geirhos et al., Nature Machine Intelligence 2020)

앞의 세 논문이 NLI라는 한 과제에 집중했다면, 이 논문은 시각·언어를 아우르는 딥러닝 전반에서 이 현상을 "지름길 학습(shortcut learning)"이라는 하나의 틀로 묶는다. 핵심 주장은 이렇다 — **지름길 규칙(shortcut)은 표준 벤치마크에서는 사람과 구별되지 않는 성능을 내지만, 그 지름길이 통하지 않는 조건(분포 밖 데이터, 적대적 예제)으로 옮기면 실패한다.** 개구리를 배경(연못)으로 분류하는 이미지 모델, 배경의 목초지로 소를 인식하는 모델처럼, 이 현상은 NLP에 국한되지 않는 딥러닝 전반의 특성이다.

## 이 계보가 말해주는 것 — 모델의 결함이 아니라 데이터셋의 결함

다섯 편(Gururangan, Poliak, McCoy et al., Niven & Kao, Geirhos et al.)을 관통하는 하나의 결론이 있다. **shortcut learning은 모델의 결함이 아니라 데이터셋 설계의 결함이다.** 모델은 손실 함수를 최소화하도록 학습될 뿐이고, 학습 신호 중 가장 예측력이 높은 것을 사용하는 것은 오히려 합리적인 행동이다. "not"이 논증 이해보다 훨씬 저렴하게 정답을 예측한다면, 경사하강법은 당연히 "not"을 학습한다. 문제는 모델이 게으른 게 아니라 **데이터셋이 그 게으름에 보상을 준다**는 데 있다. Pacchiardi et al. (2024)은 바로 이 문장을, 개별 NLI 데이터셋이 아니라 **현대 LLM 벤치마크 19개에 걸쳐** 재확인한 논문이다.

# Method — Pacchiardi et al. (2024)의 실험 설계

저자는 케임브리지대 Leverhulme Centre for the Future of Intelligence의 Lorenzo Pacchiardi, Marko Tešić, Lucy G. Cheke와, 같은 소속에 발렌시아 공대(VRAIN, Universitat Politècnica de València)를 겸한 José Hernández-Orallo다. 2024년 10월 arXiv에 공개된 프리프린트다.

## 무엇을 물었나

논문은 두 가지를 묻는다.

1. 유니그램·바이그램을 **조합**하면 현대 LLM 벤치마크의 정답 레이블을 예측할 수 있는가?
2. 실제 LLM들이 이 조합 패턴을 **활용해서** 벤치마크를 통과하고 있는가?

앞서 살펴본 계보와의 결정적 차이가 여기 있다 — Niven & Kao나 Kavumba et al.은 개별 단어("not", "a" 등) 하나가 유효한 단서임을 보였지만, Pacchiardi et al.은 **유니그램과 바이그램 수백~수천 개를 로지스틱 회귀로 조합**했을 때 어디까지 뚫리는지를 본다. 개별 신호가 아니라 신호들의 조합이 관건이라는 점에서 한 단계 더 나아간 질문이다.

## 19개 벤치마크

BIG-Bench, LegalBench, 그리고 독립 배포된 NLI·추론 데이터셋을 포함해 인과 추론·반사실 분석·도덕 판단·형식 추론·은유 이해·상식 추론·공간 추론·법적 추론·자연어 추론을 아우르는 19개의 객관식(multiple-choice) 벤치마크를 썼다. 모든 벤치마크는 인스턴스마다 선택지 수가 고정돼 있다.

| 데이터셋                             | 문항 수 | 선택지 수 | 출처       | κ > 0.2 돌파 | 최적 특징 벡터      |
| ------------------------------------ | ------- | --------- | ---------- | ------------ | ------------------- |
| Fantasy Reasoning                    | 197     | 2         | BIG-Bench  | ✓            | 1-gram TF, 단어수준 |
| NeuBAROCO                            | 363     | 3         | -          | ✓            | 2-gram TF, 단어수준 |
| Moral Permissibility                 | 338     | 2         | BIG-Bench  |              | 가독성·다양성 지표  |
| Causal Judgment                      | 184     | 2         | BIG-Bench  |              | 가독성·다양성 지표  |
| Metaphor Boolean                     | 676     | 2         | BIG-Bench  |              | 가독성·다양성 지표  |
| Commonsense QA 2.0                   | 2537    | 2         | -          |              | 가독성·다양성 지표  |
| SpaceNLI                             | 1600    | 3         | -          | ✓            | 2-gram TF, 단어수준 |
| ANLI                                 | 3196    | 3         | -          |              | 가독성·다양성 지표  |
| ART                                  | 364     | 2         | -          |              | 가독성·다양성 지표  |
| WANLI                                | 1000    | 3         | -          |              | 가독성·다양성 지표  |
| bAbI Task 16                         | 1000    | 4         | BIG-Bench  | ✓            | 2-gram TF, 단어수준 |
| Formal Fallacies Syllogisms Negation | 1000    | 2         | BIG-Bench  |              | 가독성·다양성 지표  |
| Abercrombie                          | 95      | 5         | LegalBench |              | 가독성·다양성 지표  |
| Corporate Lobbying                   | 3267    | 2         | LegalBench | ✓            | 1-gram TF, 단어수준 |
| Function of Decision Section         | 367     | 7         | LegalBench | ✓            | 2-gram TF, 토큰수준 |
| PROA                                 | 95      | 2         | LegalBench | ✓            | 1-gram TF, 토큰수준 |
| International Citizenship Questions  | 1000    | 2         | LegalBench | ✓            | 1-gram TF, 토큰수준 |
| CLadder                              | 8917    | 2         | -          |              | 2-gram TF, 단어수준 |
| ProntoQA                             | 7200    | 2         | -          | ✓            | 2-gram TF, 단어수준 |

"최적 특징 벡터" 열은 각 데이터셋에서 검증(validation) 분할 기준 정확도가 가장 높았던 특징 벡터를 뜻한다(뒤에서 다시 설명). 규모가 큰 데이터셋(CLadder, ProntoQA 등)은 비용을 줄이기 위해 무작위 부분집합만 사용했다 — 위 표의 "문항 수"가 바로 그 검사 대상 인스턴스 수다.

## 44개 LLM, 11개 계열

11개 계열에 걸친 44개 LLM의 인스턴스 단위 응답을 수집했다. LegalBench 계열 결과는 HELM 프로젝트가 공개한 것을, CLadder·ProntoQA는 각 논문이 공개한 것을 썼고, 나머지는 저자들이 직접 테스트했다.

| 계열        | 대표 모델                                       | 계열 내 모델 수 |
| ----------- | ----------------------------------------------- | --------------- |
| OpenAI      | GPT-4, GPT-3.5-turbo, davinci, curie, ada 등    | 18              |
| Meta        | Llama-1-7B, Llama-2-7B/13B/70B, Llama-65B       | 5               |
| Anthropic   | Claude-v1.3, Claude-instant-1.2, Claude-2.0/2.1 | 4               |
| Aleph Alpha | luminous-base/extended/supreme                  | 3               |
| Mistral AI  | Mistral-7B, Mixtral-8x7B                        | 2               |
| Writer      | Palmyra-X-v2/v3                                 | 2               |
| 01-AI       | Yi-6B, Yi-34B                                   | 2               |
| Google      | text-bison, text-unicorn                        | 2               |
| TII UAE     | Falcon-7B, Falcon-40B                           | 2               |
| AI21 Labs   | J2-grande, J2-jumbo                             | 2               |
| Cohere      | command, command-light                          | 2               |

주의할 점 — 모든 LLM이 모든 데이터셋에서 테스트된 건 아니다. 데이터셋과 LLM 조합마다 존재 여부가 다르며, 이는 뒤에서 볼 모델 계열별 통계 검정력에 직접 영향을 준다.

## 13개 표층 특징 벡터

핵심은 여기다 — **이 논문이 실제로 쓴 특징은 유니그램·바이그램의 빈도 통계와, 문항 전체의 가독성·다양성 지표뿐이다.** 문항 길이나 선택지 길이 차이, 정답 위치 통계를 직접 특징으로 넣지는 않았다(길이 정보는 가독성 지표에 간접적으로만 녹아 있다). 이 점은 계보의 Gururangan et al.이 가설 길이 자체를 명시적 신호로 짚었던 것과 대조된다 — Pacchiardi et al.은 더 좁은 특징 집합으로도 벤치마크 절반 가까이가 뚫린다는 것을 보인 셈이다.

n-gram 계열 특징은 두 축의 조합으로 열두 가지가 나온다.

| 축          | 선택지                                              |
| ----------- | --------------------------------------------------- |
| n-gram 종류 | 유니그램(1-gram) / 유니그램+바이그램(1+2-gram)      |
| 추출 단위   | 단어 수준(word-level) / 토큰 수준(GPT-2 토크나이저) |
| 가중치 방식 | TF / TF-IDF / Presence                              |

- **TF(Term Frequency)**: 문항에 그 n-gram이 등장한 원시 횟수.
- **TF-IDF**: 데이터셋 전체에서 그 n-gram이 얼마나 희소한지로 가중치를 보정한 값. 여러 문항에 흔하게 나오는 n-gram의 중요도를 낮춘다.
- **Presence**: 그 n-gram이 문항에 있으면 1, 없으면 0인 이진 지표.

2(n-gram 종류) × 2(추출 단위) × 3(가중치) = 12개 벡터. 여기에 **가독성·다양성 지표** 벡터 1개를 더해 총 13개 특징 벡터가 된다.

| 지표                | 무엇을 재나                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| Flesch Reading Ease | 문장 길이·음절 수 기반의 읽기 쉬움 점수(높을수록 쉬움)                    |
| Gunning Fog Index   | 복잡한(3음절 이상) 단어 비율과 문장 길이로 추정한 이해에 필요한 학년 수준 |
| SMOG Index          | 다음절 단어 수 기반의 읽기 난이도 추정치                                  |
| Yule's K            | 어휘 반복도 — 값이 클수록 같은 단어를 자주 반복(어휘 다양성이 낮음)       |

이 네 값을 이어 붙인 것이 13번째 특징 벡터다. 흥미롭게도 저자들은 TF-IDF가 대체로 TF·Presence보다 예측력이 낮다는 것을 관찰했다 — IDF 보정이 데이터셋 전체에 흔한(그래서 레이블과 강하게 상관될 가능성이 큰) n-gram의 가중치를 오히려 깎아내리기 때문이다.

## 평가 지표: Cohen's κ

벤치마크마다 선택지 수가 다르므로(2지선다부터 7지선다까지) 우연 수준 정확도 자체가 다르다. 이를 보정하기 위해 Cohen의 κ 계수를 썼다.

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $$P_o$$: 분류기의 예측과 정답이 실제로 일치한 비율(관측 정확도).
- $$P_e$$: 순전히 우연히 일치할 것으로 기대되는 비율.
- $$\kappa = 0$$이면 우연 수준, $$\kappa = 1$$이면 완벽 일치, 음수면 우연보다도 못한 체계적 불일치를 뜻한다.

전통적인 해석 기준(Landis & Koch, 1977)은 다음과 같다.

| κ 범위        | 해석                           |
| ------------- | ------------------------------ |
| κ ≤ 0.2       | 미미하거나 없음                |
| 0.2 < κ ≤ 0.4 | 작지만 탐지 가능(slight~fair)  |
| 0.4 < κ ≤ 0.6 | 상당한(fair-moderate) 일치     |
| κ > 0.6       | 상당히 강한(considerable) 일치 |

논문은 **κ > 0.2**를 "완전히 통제된 벤치마크라면 나타나서는 안 되는 최소 기준선"으로 채택한다. 완벽하게 통제된 벤치마크라면 문항의 한두 단어짜리 특징으로 정답을 예측할 수 없어야 하므로, 이 값은 이론적으로 0에 가까워야 한다. κ의 정확한 유도와 다중 평가자로의 확장은 [#17](/blog/2026/kappa-agreement/)에서 별도로 다룬다.

## 실험 파이프라인

각 데이터셋마다 다음 절차를 거쳤다.

1. **분할**: 데이터를 train/validation/test로 나눈다.
2. **학습**: train 분할에서 13개 특징 벡터 각각에 대해 로지스틱 회귀 분류기를 학습한다. 정규화는 L2(λ=1)와 L1(λ=1, λ=10) 세 가지 설정을 모두 시도한다.
3. **모델 선택**: validation 분할에서 정확도가 가장 높은 (특징 벡터, 정규화 설정) 조합을 고른다.
4. **평가**: 선택된 조합을 test 분할에 적용해 Cohen's κ를 계산한다.
5. **LLM 대조**: 같은 test 분할을 n-gram 분류기가 맞힌 부분집합과 못 맞힌 부분집합으로 나누고, 각 LLM이 두 부분집합에서 얼마나 다르게 수행하는지 비교한다.

코드와 데이터는 GitHub([Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability](https://github.com/Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability))에 공개돼 있다.

# Experiments

## 벤치마크별 예측력 — 19개 중 9개가 뚫린다

위 표에서 ✓ 표시된 9개 데이터셋(Fantasy Reasoning, NeuBAROCO, SpaceNLI, bAbI Task 16, Corporate Lobbying, Function of Decision Section, PROA, International Citizenship Questions, ProntoQA)이 κ > 0.2를 넘었다. **19개 중 9개, 약 47%다.** 이 중 대부분은 "fair to moderate"(0.2\~0.6) 구간에 있었지만, **Corporate Lobbying과 SpaceNLI는 κ > 0.6**로 "considerable agreement" 수준까지 올라갔다 — 즉 법적 로비 목적 판별이나 공간 추론 같은, 언뜻 문항을 꼼꼼히 읽어야만 풀릴 것 같은 과제조차, 문항에 어떤 단어가 몇 번 등장하는지만으로 상당히 신뢰도 높게 정답을 맞힐 수 있다.

몇 가지 패턴이 눈에 띈다.

- **LegalBench 계열이 특히 취약하다**: 5개 LegalBench 과제 중 4개(Corporate Lobbying, Function of Decision Section, PROA, International Citizenship Questions)가 뚫렸다. 법률 문서는 정형화된 문구(boilerplate)를 반복하는 경향이 있어, 어휘 패턴이 과제 범주와 우연히 강하게 얽혔을 가능성이 크다.
- **가독성·다양성 지표가 의외로 자주 최적이었다**: 19개 중 10개 데이터셋에서 n-gram이 아니라 가독성·다양성 지표(Flesch, Gunning Fog, SMOG, Yule's K)가 최적 특징으로 선택됐다. 다만 이들 대부분은 κ ≤ 0.2로, "최적이었지만 문턱을 넘지는 못한" 경우다.
- **CLadder는 뚫리지 않았다**: 8,917개 인스턴스라는 가장 큰 데이터셋인데도 최적 특징(2-gram TF)의 κ가 0.2를 넘지 못했다. 규모가 크다고 해서 자동으로 안전하지는 않지만, 적어도 이 경우는 표층 신호에 상대적으로 강건했다.

## 모델이 실제로 이 지름길을 쓰는가

특징만으로 정답이 예측된다는 사실 자체는 "벤치마크의 결함"을 보여줄 뿐, "LLM이 그 결함을 이용한다"는 것을 증명하지는 않는다. 저자들은 이를 확인하기 위해 κ > 0.2를 넘긴 9개 데이터셋에서, 각 LLM의 test 인스턴스를 n-gram 분류기가 **맞힌 부분집합**과 **못 맞힌 부분집합**으로 나누고, "LLM이 맞힌 부분집합에서 유독 더 잘하는가"를 모델 계열별로 단측 대응표본 t-검정했다. 다중비교는 Benjamini-Hochberg 절차로 보정했다.

| 모델 계열   | p-value  | 보정 p-value | 계열 내 모델 수 |
| ----------- | -------- | ------------ | --------------- |
| Meta        | 0.007342 | 0.059433     | 4               |
| OpenAI      | 0.010806 | 0.059433     | 18              |
| Mistral AI  | 0.027277 | 0.089484     | 2               |
| Writer      | 0.032540 | 0.089484     | 2               |
| Anthropic   | 0.071911 | 0.158205     | 4               |
| Aleph Alpha | 0.108315 | 0.198578     | 3               |
| 01-AI       | 0.336185 | 0.528291     | 2               |
| Google      | 0.393498 | 0.541060     | 2               |
| TII UAE     | 0.474044 | 0.579387     | 2               |
| AI21 Labs   | 0.678992 | 0.681576     | 2               |
| Cohere      | 0.681576 | 0.681576     | 2               |

**Meta·OpenAI·Mistral AI·Writer** 네 계열은 보정 p-value가 0.1 미만으로, "n-gram이 맞히는 문항에서 이 계열 모델도 유의하게 더 잘 맞힌다"는 중간 정도의 증거가 나왔다. Anthropic·Aleph Alpha는 보정 p-value 0.1\~0.2 사이로 약한 증거만 있었고, 나머지 계열은 증거가 없었다.

다만 저자들 스스로 이 결과를 신중하게 해석한다. 계열별 검정력 차이가 크기 때문이다 — OpenAI는 18개 모델이 12개 이상의 데이터셋에서 테스트됐지만, AI21 Labs나 Cohere 같은 계열은 모델 2개가 데이터셋 5개에서만 테스트됐다. OpenAI 수준의 효과 크기를 이런 작은 표본에서 탐지하려면 애초에 데이터가 더 필요하다 — 이는 [#21](/blog/2026/statistical-power/)에서 다룰 통계 검정력 문제 그 자체다. 저자들도 논문에 명시한다: **"제공한 증거가, 그 LLM 계열들이 실제로 성공을 위해 n-gram에 의존한다는 것을 결정적으로 입증하지는 않는다."** 다만 유의한 효과가 OpenAI·Meta처럼 규모가 크고 실제로 널리 쓰이는 계열에서 나왔다는 점은 무시하기 어렵다.

## 토이 예제 — "가장 긴 선택지가 정답"이라면

Pacchiardi et al.이 실제로 쓴 특징은 n-gram과 가독성 지표지만, "표층 특징이 정답을 흘린다"는 현상이 얼마나 쉽게 벤치마크를 오염시키는지 감을 잡기에는 훨씬 단순한 예제가 낫다. 4지선다 객관식 벤치마크 1,000문항을 가정하자. 문항 출제자가 정답 선택지를 쓸 때 무의식적으로 조건·단서를 더 자세히 적는 습관이 있다고 하자 — Gururangan et al.이 SNLI에서 발견한 "가설 길이가 레이블과 상관된다"는 현상과 정확히 같은 종류의 습관이다.

가정한 분포는 다음과 같다.

| 상황                            | 문항 비율 | 문항 수 | "가장 긴 선택지" 규칙의 성공 확률 |
| ------------------------------- | --------- | ------- | --------------------------------- |
| 정답이 유일하게 가장 긴 선택지  | 45%       | 450     | 1 (항상 성공)                     |
| 정답이 최장 길이 동률(2개 동률) | 15%       | 150     | 1/2 (동률 중 무작위 선택)         |
| 정답이 최장 길이가 아님         | 40%       | 400     | 0 (항상 실패)                     |

이 규칙의 기대 정답 수는 다음과 같다.

$$450 \times 1 + 150 \times \frac{1}{2} + 400 \times 0 = 450 + 75 = 525$$

즉 관측 정확도는 다음과 같다.

$$P_o = \frac{525}{1000} = 0.525$$

4지선다의 우연 수준은 $$P_e = 0.25$$이므로, "문항을 한 글자도 읽지 않고 가장 긴 선택지를 고르는" 규칙만으로 **52.5%** — 우연의 2.1배 — 를 맞힌다. Cohen's κ로 보면 다음과 같다.

$$\kappa = \frac{0.525 - 0.25}{1 - 0.25} = \frac{0.275}{0.75} \approx 0.37$$

이는 Pacchiardi et al.이 쓴 κ > 0.2 문턱을 가볍게 넘는다. 즉 이 가상의 벤치마크는 실험도 해보기 전에 "선택지 길이"라는 단 하나의 메타 특징만으로 논문의 진단 기준에 걸린다.

이것이 왜 성능 보고를 오염시키는지가 핵심이다. 실제 추론 능력이 있는 모델 A와, 사전학습 중 이런 유형의 벤치마크를 많이 접해 "길다=정답"이라는 표층 규칙을 체득한 모델 B가 있다고 하자. 모델 B는 지문을 전혀 이해하지 못해도 이 규칙 하나로 52.5%를 확보한다. 만약 모델 A의 실제 추론 정확도가 45%에 그친다면, 리더보드는 모델 B를 "더 나은 추론 모델"로 표시한다 — 실제로는 추론을 전혀 하지 않았는데도 말이다. Pacchiardi et al.의 Table 3이 보여준 것도 정확히 이 구조다. 모델 계열마다 이 지름길에 얼마나 의존하는지가 다르므로, 표층 신호는 점수를 균일하게 올리는 게 아니라 **순위 자체를 왜곡한다.**

## 진단 절차 — 새 벤치마크를 공개하기 전에

Pacchiardi et al.의 방법론은 그대로 하나의 사전 점검 절차로 일반화할 수 있다.

1. **메타 특징만 추출한다**: 문항이 무엇을 묻는지 이해할 필요가 없는 특징만 뽑는다 — n-gram 빈도, 가독성·다양성 지표, 그리고 원한다면 문항·선택지 길이, 정답 위치 히스토그램 같은 것도 추가할 수 있다(Pacchiardi et al.은 앞의 두 가지만 썼다).
2. **문항 텍스트의 의미는 건드리지 않는 단순 분류기를 학습한다**: 로지스틱 회귀처럼 표현력이 제한된 모델을 쓴다. 표현력이 큰 모델을 쓰면 우연이 아니라 과적합으로 정확도가 오를 위험이 있다.
3. **train/validation/test로 분할**하고, validation에서 하이퍼파라미터(정규화 방식·강도)를 고른다.
4. **test 분할에서 Cohen's κ를 계산**한다.
5. **κ가 우연보다 유의하게 큰지 판정**한다. 여기서 Pacchiardi et al.처럼 고정된 관습적 문턱(κ > 0.2)에만 기대는 것은 최소 기준일 뿐, 통계적으로 더 방어 가능한 방법은 κ의 표본분포에 대한 신뢰구간을 구성해 하한이 0을 넘는지 보거나([#19](/blog/2026/confidence-intervals/)), 귀무가설 "κ = 0"에 대한 유의성 검정을 수행하는 것이다([#20](/blog/2026/significance-testing/)). 논문 스스로도 고정 문턱이라는 다소 느슨한 기준에 의존했다는 한계가 있다 — 후속 벤치마크 제작자는 여기서 한 걸음 더 나아갈 수 있다.
6. **κ가 유의하게 0보다 크면**, 그 벤치마크는 최소한 검사한 특징 집합에 대해 Clever Hans 취약점이 있다는 뜻이다. 공개 전에 선택지 길이 정규화, 무작위 순서 재배치, 균형 잡힌 템플릿 재생성 같은 개입이 필요하다.

## 처방 — 저자들이 제안하는 통제 방법

논문 5.1절 "Controlling for Clever Hans"는 이 문제를 사람·동물 행동 실험이 오랫동안 다뤄온 문제의 연장선에 놓는다. 심리학 실험에서는 흔한 지혜다 — 이런 교란 요인은 피할 수 없으므로, 이를 찾아내고 제거하고 통제하는 데 상당한 노력을 들인다는 것이다.

| 분야           | 통제 기법                                                               |
| -------------- | ----------------------------------------------------------------------- |
| 동물 행동 실험 | 실험자가 답을 모르게 하는 불투명 고글, 사전 녹음된 음성 명령            |
| 의학 연구      | 이중맹검(double-blind) — 실험자도 어느 쪽이 처치군인지 모르게 함        |
| 인간 행동 연구 | 자극의 무작위화(randomisation)·변형(variation)·균형화(counterbalancing) |
| AI 평가(현재)  | 이런 관행이 최근에야, 그것도 일부 사례에서만 등장하기 시작함            |

저자들의 결론은 명확하다. **"완전히 통제된 벤치마크라면 문항의 한두 단어짜리 특징으로 정답을 예측할 수 있어서는 안 된다."** 자극의 무작위화·다양화·균형화가 AI 평가에서도 표준 관행이 되어야 한다는 것이다. 동시에 저자들은 스스로 연구의 한계를 인정한다 — 이번 실험은 유니그램·바이그램이라는 좁은 특징 집합만 봤을 뿐이고, 이는 모델이 활용할 수 있는 모든 표층 단서의 일부에 불과하다. 이 특정 특징들에서 신호가 안 보인다고 해서 그 벤치마크가 다른 종류의 지름길로부터 안전하다는 뜻은 아니다.

# 통계 요약

| 개념                             | 정의                                                     | 이 논문에서의 역할                                                     |
| -------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| Cohen's κ                        | $$\kappa = (P_o - P_e) / (1 - P_e)$$                     | 선택지 수가 다른 벤치마크 간에 우연 보정된 예측력을 비교하는 공통 척도 |
| κ 해석 문턱(Landis & Koch, 1977) | 0.2/0.4/0.6 구간 구분                                    | "탐지 가능한 수준의 예측력이 있는가"를 가르는 관습적 기준선            |
| 단측 대응표본 t-검정             | 같은 LLM의 두 부분집합(성공/실패 예측) 간 평균 차이 검정 | 모델이 n-gram 신호를 실제로 활용하는지에 대한 통계적 증거              |
| Benjamini-Hochberg 보정          | 다중비교 시 거짓 발견율(FDR) 통제                        | 11개 모델 계열에 대해 동시에 검정할 때 우연한 유의성 과다 검출 방지    |

# Conclusion

이 논문의 결론은 한 문장으로 압축된다. **벤치마크를 공개하기 전에 "문항을 읽지 않고도 풀리는가"를 검사해야 한다.** 19개 중 9개(47%)가 이 검사에 걸렸고, 그중 두 개(Corporate Lobbying, SpaceNLI)는 κ > 0.6이라는 "상당한 일치" 수준까지 갔다. 몇몇 주요 LLM 계열(OpenAI, Meta, Mistral AI, Writer)은 실제로 이 지름길을 어느 정도 활용하고 있다는 통계적 증거도 나왔다.

이 발견을 시리즈의 다른 편과 정확히 구분해 둘 필요가 있다.

- **[#9](/blog/2026/mcqa-fragility/) 위치 편향과의 관계**: [#9](/blog/2026/mcqa-fragility/)에서 다룰 위치 편향(모델이 특정 선택지 위치를 선호하는 경향)은 **모델 쪽** 인공물이다. 문항 자체는 아무 문제가 없어도, 모델이 "A를 고르는 습관"을 갖고 있으면 점수가 왜곡된다. 반면 Clever Hans는 **데이터 쪽** 인공물이다 — 문항 텍스트 자체에 정답과 상관된 표층 규칙성이 박혀 있다. 같은 무관 분산이라도 층위가 다르다. 위치 편향은 모델을 셔플하거나 프롬프트를 바꿔 완화할 수 있지만, Clever Hans는 벤치마크 자체를 다시 설계해야 없어진다.
- **[#24](/blog/2026/contamination-reproducibility/) 오염과의 관계**: 오염(contamination)은 "모델이 이 문항의 답을 사전학습 중에 이미 봤다"는 문제다. Clever Hans는 그보다 훨씬 약한 조건에서 발생한다 — **답을 한 번도 본 적이 없어도, 문항의 표층 패턴만으로 우연 이상을 맞힐 수 있다.** 둘 다 점수를 부풀리지만 처방이 다르다. 오염은 탈오염(decontamination)·홀드아웃·동적 벤치마크로 대응하고, Clever Hans는 자극의 무작위화·균형화로 대응한다. 데이터를 완전히 새로 만들어도(오염 없음을 보장해도) Clever Hans는 여전히 남을 수 있다 — 새 문항을 쓰는 사람이 여전히 같은 작문 습관을 가지고 있기 때문이다.

이 논문이 [#1](/blog/2026/what-is-evaluation/)의 구성개념 무관 분산에 기여하는 바는 이론이 아니라 **관측 가능한 실증**이라는 점이다. "벤치마크에 무관한 잡음이 섞여 있을 수 있다"는 우려는 추상적으로 듣기 쉽지만, 이 논문은 19개 벤치마크 중 9개에서 그 잡음의 크기를 κ 값으로 직접 잰다. Gururangan et al. (2018)부터 이어진 6년치 계보가 결국 도달한 지점은 같다 — **모델의 능력을 재고 싶다면, 먼저 그 문항이 능력 없이도 풀리지 않는지부터 확인해야 한다.**

# 참고 문헌

- Pacchiardi, Tešić, Cheke, Hernández-Orallo (University of Cambridge), 2024. [Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/abs/2410.11672) (arXiv:2410.11672).
- [GitHub: Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability](https://github.com/Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability) — 실험 재현 코드.
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith, 2018. [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/) (NAACL 2018).
- Poliak, Naradowsky, Haldar, Rudinger, Van Durme, 2018. [Hypothesis Only Baselines in Natural Language Inference](https://arxiv.org/abs/1805.01042) (*SEM 2018).
- McCoy, Pavlick, Linzen, 2019. [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://aclanthology.org/P19-1334/) (ACL 2019).
- Niven, Kao, 2019. [Probing Neural Network Comprehension of Natural Language Arguments](https://aclanthology.org/P19-1459/) (ACL 2019).
- Kavumba, Inoue, Heinzerling, Singh, Reisert, Inui, 2019. [When choosing plausible alternatives, Clever Hans can be clever](https://arxiv.org/abs/1911.00225) (arXiv:1911.00225).
- Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann, 2020. [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z). Nature Machine Intelligence.
- Landis, Koch, 1977. The Measurement of Observer Agreement for Categorical Data. Biometrics 33(1), 159-174.

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 다섯 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/everything-benchmark/">범용 벤치마크라는 주장</a> — Raji et al. — 모든 것을 잰다는 말</li>
  <li><a href="/blog/2026/fixing-nlu-benchmarking/">벤치마킹을 고치려면</a> — Bowman & Dahl의 네 기준</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
  <li><strong>(현재 글)</strong> 표층 특징이 정답을 예측한다 — Clever Hans, 데이터셋 인공물</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="6">
  <li><a href="/blog/2026/measurement-scales/">척도와 허용 연산</a> — Likert 평균을 내도 되는가</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><a href="/blog/2026/generation-metrics/">생성 지표와 그 타당도</a> — BLEU에서 COMET까지</li>
  <li><a href="/blog/2026/mcqa-fragility/">객관식 평가는 왜 흔들리나</a> — 위치 편향과 포맷 민감도</li>
</ol>

**3부. LLM 벤치마크 지형도**

<ol start="10">
  <li><a href="/blog/2026/knowledge-benchmarks/">지식과 추론 — MMLU 계열의 흥망</a> — MMLU·GPQA·BBH</li>
  <li><a href="/blog/2026/math-code-benchmarks/">검증 가능한 도메인 — 수학과 코드</a> — GSM8K·MATH·HumanEval·SWE-bench</li>
  <li><a href="/blog/2026/mt-bench-to-arena/">개방형 대화 — MT-Bench에서 Arena까지</a> — judge 기반 벤치의 등장</li>
  <li><a href="/blog/2026/capability-axes-benchmarks/">능력의 다른 축</a> — 지시따르기·긴 문맥·사실성</li>
  <li><a href="/blog/2026/korean-benchmarks/">한국어 벤치마크</a> — 번역이 아니라 원산, 그리고 문화 타당도</li>
  <li><a href="/blog/2026/helm-holistic-evaluation/">점수 하나가 아니라 행렬로</a> — HELM — 시나리오 × 지표</li>
</ol>

**4부. 사람이 읽는다 — 정성평가와 일치도**

<ol start="16">
  <li><a href="/blog/2026/human-evaluation-design/">사람 평가 설계</a> — 루브릭·Likert·pairwise·BWS</li>
  <li><a href="/blog/2026/kappa-agreement/">우연을 빼다 — κ 계열</a> — Cohen·Fleiss·weighted·Krippendorff</li>
  <li><a href="/blog/2026/kappa-paradox/">κ의 역설</a> — 일치율 90%인데 κ가 0.21</li>
</ol>

**5부. 차이는 진짜인가 — 정량평가의 통계**

<ol start="19">
  <li><a href="/blog/2026/confidence-intervals/">점수는 추정치다</a> — 이항비율 신뢰구간과 Wald의 실패</li>
  <li><a href="/blog/2026/significance-testing/">차이는 유의한가</a> — paired bootstrap·순열검정·McNemar</li>
  <li><a href="/blog/2026/statistical-power/">몇 개를 재야 하나</a> — 검정력·표본크기·다중비교</li>
  <li><a href="/blog/2026/error-bars-for-evals/">LLM eval의 통계 실무</a> — 클러스터 SE·IQM·분산 분해</li>
</ol>

**6부. 신뢰할 수 있는 평가 체계**

<ol start="23">
  <li><a href="/blog/2026/judge-statistics/">judge를 통계로 다루기</a> — 편향·Bradley-Terry·PPI</li>
  <li><a href="/blog/2026/contamination-reproducibility/">오염·재현성·효율</a> — 오염 검정·harness·IRT</li>
  <li><a href="/blog/2026/safety-evaluation-statistics/">안전 평가의 통계와 체계 설계</a> — 희귀사건·calibration·체크리스트</li>
</ol>

본 시리즈는 25편으로 구성된다.
