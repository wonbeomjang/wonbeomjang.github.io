---
layout: post
title: "표층 특징이 정답을 예측한다 — Clever Hans in LLM benchmarks"
date: 2026-08-24 09:05:00 +0900
description: "LLM 평가 체계 시리즈 #4 — 19개 벤치마크 중 9개에서 유니그램·바이그램 같은 표층 특징만으로 정답 레이블이 우연 이상으로 예측된다 (Pacchiardi et al., Cambridge, arXiv 2024)"
categories: [paper]
tags: [evaluation, benchmark, clever-hans, shortcut-learning, nlp, llm, paper]
giscus_comments: true
related_posts: true
---

> [Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/abs/2410.11672) (Pacchiardi et al., University of Cambridge, arXiv 2024)

# Introduction

벤치마크가 재려는 능력을 전혀 사용하지 않고도 정답을 예측할 수 있다면, 높은 점수는 무엇을 의미할까?

Pacchiardi et al.은 이 질문을 현대 LLM 벤치마크 19개에 실제로 던졌다. 문항 텍스트에서 유니그램·바이그램의 빈도와 가독성·다양성 지표만 뽑아 로지스틱 회귀 분류기를 학습시킨 뒤, 그 분류기가 정답 레이블을 얼마나 맞히는지 쟀다. 결과는 이렇다 — **19개 벤치마크 중 9개에서, 유니그램·바이그램 같은 단순한 표층 특징만으로도 정답 레이블을 우연 수준보다 체계적으로 예측할 수 있었다**(Cohen's $$\kappa > 0.2$$). 그리고 일부 LLM 계열은 바로 그 문항들에서 상대적으로 더 잘 맞히는 경향을 보였다.

여기서 표현을 정확히 해 둘 필요가 있다. 이 분류기는 텍스트를 안 읽는 것이 아니다. **텍스트의 표면 통계를 읽되, 문항의 의미를 해석하지 않는다.** 그래서 이 글은 "문항을 읽지 않고"가 아니라 "문항의 의미를 해석하지 않고"라고 쓴다.

[#1](/blog/2026/what-is-evaluation/)에서 세운 두 잣대 중 이 논문이 겨냥하는 것은 **구성개념 무관 분산(construct-irrelevant variance)**이다. 다만 여기서 한 단계를 반드시 나눠야 한다.

**표층 특징이 정답을 예측한다는 사실은 벤치마크에 구성개념과 무관한 대안적 풀이 경로가 존재한다는 뜻이다. 그것이 실제 모델 점수에 구성개념 무관 분산으로 들어갔다고 말하려면, 모델이 그 경로를 실제로 사용한다는 추가 증거가 필요하다.**

이 구분을 사슬로 그리면 네 칸이 된다.

$$\text{표층 단서 존재} \rightarrow \text{대안 풀이 경로} \rightarrow \text{모델이 그 경로를 사용} \rightarrow \text{점수에 구성개념 무관 성분}$$

앞 화살표를 확인했다고 뒤가 자동으로 따라오지는 않는다. 뒤에서 보겠지만 이 논문은 첫 두 칸을 강하게 보이고, 세 번째 칸에 대해서는 시사적인 증거(suggestive evidence)를 제시한다. 이 글은 그 경계를 흐리지 않는 것을 목표로 한다.

이 글에서 답할 질문은 세 가지다.

1. **계보**: 이 발견은 2024년에 처음 나온 것이 아니다. 2018년 NLI 연구부터 이어지는 계보를 먼저 짧게 세운다.
2. **실험**: Pacchiardi et al.이 정확히 무엇을 측정했는가 — 어떤 특징을, 어떤 지표로, 어디까지 검증했는가.
3. **처방**: 벤치마크 제작자가 공개 전에 무엇을 검사하고, 무엇을 찾았을 때 무엇을 고쳐야 하는가.

# Background

## Clever Hans — 조련사의 몸짓을 읽은 말

20세기 초 독일 베를린에 "영리한 한스(Kluger Hans)"라는 말이 있었다. 조련사 빌헬름 폰 오스텐이 발굽을 두드리는 방식으로 한스에게 산수를 가르쳤다고 주장했고, 실제로 한스는 "3 더하기 4는?" 같은 질문에 발굽을 일곱 번 두드려 정답을 맞혔다. 수년간 대중과 일부 과학자들이 이를 진짜 산술 능력으로 믿었다.

1907년 심리학자 오스카 풍스트가 이를 조사한 결과, 한스는 산수를 하지 못했다. 대신 질문자가 무의식적으로 보이는 미세한 신체 신호 — 정답 근처의 두드림 횟수에 다다르면 질문자의 자세나 호흡이 미묘하게 바뀌는 것 — 를 읽고 있었다. 질문자가 정답을 모르거나 한스가 질문자를 볼 수 없으면 한스는 정답을 맞히지 못했다. 즉 **한스가 실제로 사용한 것은 "산수"가 아니라 "질문자의 몸짓"이었다.** 과제를 푸는 데 필요하다고 여겨진 능력과 실제로 사용된 단서가 서로 다른 것 — 이것이 Clever Hans 효과다.

이 비유가 벤치마크에 대응하는 지점은 정확하다. 벤치마크 설계자는 "이 문항을 풀려면 인과 추론이 필요하다"고 믿고 문항을 만든다. 하지만 문항 텍스트 자체에 인과 추론과 무관한 통계적 규칙성 — 특정 단어의 등장, 문장 길이, 어휘 다양성 — 이 정답과 상관돼 있다면, 그 규칙성만으로도 정답을 맞힐 수 있는 경로가 생긴다. 조련사가 자기도 모르게 몸짓으로 답을 흘리듯, **벤치마크 제작자도 자기도 모르게 문항 텍스트에 답을 흘릴 수 있다.**

## 계보 — 네 단계로 압축한 6년

Pacchiardi et al.의 실험은 새로운 질문이 아니라, 2018년부터 NLP 커뮤니티가 반복해서 확인해 온 현상을 현대 LLM 벤치마크 전반으로 확장한 것이다. 이 논문의 novelty를 이해하는 데 필요한 계보는 네 단계면 충분하다.

### 2018 — 주석 인공물: 필요한 입력을 가려도 답이 예측된다

[Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/) (Gururangan et al., NAACL 2018)

NLI 과제는 전제(premise)와 가설(hypothesis) 두 문장을 보고 함의·중립·모순 중 하나를 고르는 것이다. 저자들은 **전제를 아예 주지 않고 가설 문장만** fastText 분류기에 입력했다.

| 데이터셋              | 가설만 본 분류기 | 다수결 기준선 |
| --------------------- | ---------------- | ------------- |
| SNLI                  | 67.0%            | 34.3%         |
| MultiNLI (matched)    | 53.9%            | 35.4%         |
| MultiNLI (mismatched) | 52.3%            | 35.2%         |

전제 문장을 보지 않고도 SNLI의 3분의 2를 맞힌다. 원인은 크라우드워커의 **작문 습관**이었다. 함의 가설은 "동물", "사람" 같은 일반화된 단어를 쓰는 경향이 있었고, 중립 가설은 최상급·수식어와 목적절을 덧붙이는 경향이, 모순 가설은 부정어를 쓰는 경향이 뚜렷했다. 게다가 **가설 길이 자체가 신호였다** — SNLI에서 중립 가설의 중위 길이는 9토큰인데, 함의 가설의 60%는 7토큰 이하였다.

같은 해 Poliak et al.의 [Hypothesis Only Baselines in Natural Language Inference](https://arxiv.org/abs/1805.01042)(\*SEM 2018)는 거의 동시에, 독립적으로 같은 실험을 열 개의 NLI류 데이터셋에 걸쳐 수행해 **10개 중 6개**에서 가설 전용 모델이 다수결 기준선을 유의하게 웃돈다는 것을 보였다. 사람이 직접 가설을 창작한(elicited) 데이터셋일수록 격차가 컸다.

### 2019 — 진단과 적대적 검증: 모델이 그 단서를 실제로 쓴다는 증거

2018년 연구가 "표층 단서가 존재한다"를 보였다면, 2019년의 세 논문은 한 발 더 나아가 **모델이 그 단서에 실제로 의존하는지**를 검증하는 방법을 만들었다. 사슬의 두 번째 칸에서 세 번째 칸으로 넘어가려는 첫 시도다.

| 논문                                                                     | 과제           | 검증 방법                               | 핵심 결과                                                       |
| ------------------------------------------------------------------------ | -------------- | --------------------------------------- | --------------------------------------------------------------- |
| [McCoy, Pavlick & Linzen](https://aclanthology.org/P19-1334/) (ACL 2019) | NLI (HANS)     | 구문 휴리스틱을 겨냥한 진단 평가셋 설계 | 휴리스틱이 틀리는 사례(우연 50%)에서 정확도 대부분 10% 미만     |
| [Niven & Kao](https://aclanthology.org/P19-1459/) (ACL 2019)             | 논증 이해 ARCT | 입력 일부 가리기 + 적대적 데이터셋 구성 | warrant만 줘도 71%(원래 77%), 적대적 데이터에서 평균·중앙값 50% |
| [Kavumba et al.](https://arxiv.org/abs/1911.00225) (arXiv 2019)          | 인과 COPA      | 개별 어휘 단서의 예측력 측정            | 단어 하나가 이미 강한 예측 신호 — "Clever Hans effect"라 명명   |

세 결과의 해석에는 각각 주의할 점이 있다. HANS는 매우 강한 진단 증거지만, 모델 내부의 모든 추론 기제가 휴리스틱 하나뿐임을 증명하지는 않는다 — **표준 MNLI 성능만으로는 모델이 일반적인 함의 관계를 학습했다고 보기 어렵고, 적어도 HANS가 겨냥한 사례에서는 특정 구문 휴리스틱에 크게 의존하고 있음이 드러났다**는 것이 정확한 진술이다. Niven & Kao에서 warrant만 보여준 71%는 원래 77% 중 71%p가 인공물이고 6%p가 추론이라는 뜻이 아니다 — 두 조건에서 맞힌 문항이 같지 않고, 성능은 가법적으로 분해되지 않는다. 이 결과는 **원래 성능의 상당 부분이 관계적 논증 구조 없이도 얻어질 수 있음**을 보여줄 뿐이고, 더 결정적인 증거는 표층 단서를 무력화한 적대적 데이터에서 성능이 우연 수준으로 떨어졌다는 쪽이다. Kavumba et al.은 2019년에 이미 COPA의 표층 단서 문제를 "Clever Hans effect"라 불렀고, Pacchiardi et al.도 관련 연구 절에서 이 논문을 인용한다.

### 2020 — 지름길 학습: 개별 과제를 넘어선 일반 현상

[Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) (Geirhos et al., Nature Machine Intelligence 2020)

앞선 연구들이 개별 NLP 데이터셋에서 표층 단서와 휴리스틱을 발견했다면, Geirhos et al.은 이를 시각과 언어를 아우르는 **지름길 학습(shortcut learning)**이라는 더 일반적인 관점으로 묶었다. 핵심 주장은 이렇다 — 지름길 규칙은 표준 벤치마크에서는 사람과 구별되지 않는 성능을 내지만, 그 지름길이 통하지 않는 조건(분포 밖 데이터, 적대적 예제)으로 옮기면 실패한다. 개구리를 배경인 연못으로 분류하는 이미지 모델처럼, 이 현상은 NLP에 국한되지 않는다.

### 2024 — Pacchiardi: 현대 LLM 벤치마크 19개로

Niven & Kao나 Kavumba et al.이 개별 단어 하나가 유효한 단서임을 보였다면, Pacchiardi et al.은 **유니그램과 바이그램 수백\~수천 개를 로지스틱 회귀로 조합**했을 때 어디까지 예측되는지를, 특정 과제가 아니라 현대 LLM 벤치마크 19개에 걸쳐 체계적으로 검사한다. 개별 신호가 아니라 신호들의 조합이 관건이라는 점에서 한 단계 더 나아간 질문이다.

## 이 계보가 말해주는 것 — 모델만 탓해서는 안 된다

모델이 표층 단서를 활용한다는 사실만으로 모델을 "게으르다"고 볼 수는 없다. 학습 데이터에서 그 단서가 정답을 안정적으로 예측한다면, 손실을 줄이는 모델이 그것을 배우는 것은 자연스럽다. 따라서 문제의 일부는 **그런 지름길이 높은 점수로 보상되도록 데이터와 평가를 설계했다**는 데 있다.

다만 이 진단을 "지름길 학습은 데이터셋의 결함일 뿐"으로 환원하지는 말아야 한다. Geirhos et al.도 지름길 학습을 단일한 데이터셋 버그로 축소하지 않는다 — 지름길은 데이터 분포, 학습 목적함수, 모델의 귀납 편향이 **함께** 만들어내는 현상이다. 특히 Pacchiardi et al.이 다루는 LLM들은 해당 벤치마크의 train 분할로 직접 미세조정된 모델이 아니다. 그러므로 "경사하강법이 이 벤치마크의 not을 학습한다"는 NLI 시대의 논리를 2024년 LLM 평가에 그대로 옮겨서는 안 된다. 데이터·목적함수·모델 중 어느 하나만 지목하는 대신, 평가 설계가 통제해야 할 교란 요인이 무엇인지를 묻는 편이 생산적이다.

# Method — Pacchiardi et al. (2024)의 실험 설계

저자는 케임브리지대 Leverhulme Centre for the Future of Intelligence의 Lorenzo Pacchiardi, Marko Tešić, Lucy G. Cheke와, 같은 소속에 발렌시아 공대(VRAIN, Universitat Politècnica de València)를 겸한 José Hernández-Orallo다. 2024년 10월 arXiv에 공개된 프리프린트다.

## 두 개의 질문

논문은 두 가지를 묻는다. 앞서 세운 사슬에서 각각 어느 칸에 해당하는지가 중요하다.

1. 유니그램·바이그램을 **조합**하면 현대 LLM 벤치마크의 정답 레이블을 예측할 수 있는가? (사슬의 1\~2번째 칸)
2. 실제 LLM들이 그런 문항에서 **더 잘하는가**? (사슬의 3번째 칸에 대한 간접 증거)

두 질문을 분리해서 물었다는 점이 이 논문의 방법론적 핵심이다. 첫 질문의 답이 "그렇다"여도 두 번째 질문의 답이 자동으로 정해지지는 않는다.

## 19개 벤치마크

BIG-Bench, LegalBench, 그리고 독립 배포된 NLI·추론 데이터셋을 포함해 인과 추론·반사실 분석·도덕 판단·형식 추론·은유 이해·상식 추론·공간 추론·법적 추론·자연어 추론을 아우르는 19개의 객관식(multiple-choice) 벤치마크를 썼다. 모든 벤치마크는 인스턴스마다 선택지 수가 고정돼 있다.

| 데이터셋                             | 문항 수 | 선택지 수 | 출처       | κ > 0.2 | 최적 특징 벡터      |
| ------------------------------------ | ------- | --------- | ---------- | ------- | ------------------- |
| Fantasy Reasoning                    | 197     | 2         | BIG-Bench  | ✓       | 1-gram TF, 단어수준 |
| NeuBAROCO                            | 363     | 3         | -          | ✓       | 2-gram TF, 단어수준 |
| Moral Permissibility                 | 338     | 2         | BIG-Bench  |         | 가독성·다양성 지표  |
| Causal Judgment                      | 184     | 2         | BIG-Bench  |         | 가독성·다양성 지표  |
| Metaphor Boolean                     | 676     | 2         | BIG-Bench  |         | 가독성·다양성 지표  |
| Commonsense QA 2.0                   | 2537    | 2         | -          |         | 가독성·다양성 지표  |
| SpaceNLI                             | 1600    | 3         | -          | ✓       | 2-gram TF, 단어수준 |
| ANLI                                 | 3196    | 3         | -          |         | 가독성·다양성 지표  |
| ART                                  | 364     | 2         | -          |         | 가독성·다양성 지표  |
| WANLI                                | 1000    | 3         | -          |         | 가독성·다양성 지표  |
| bAbI Task 16                         | 1000    | 4         | BIG-Bench  | ✓       | 2-gram TF, 단어수준 |
| Formal Fallacies Syllogisms Negation | 1000    | 2         | BIG-Bench  |         | 가독성·다양성 지표  |
| Abercrombie                          | 95      | 5         | LegalBench |         | 가독성·다양성 지표  |
| Corporate Lobbying                   | 3267    | 2         | LegalBench | ✓       | 1-gram TF, 단어수준 |
| Function of Decision Section         | 367     | 7         | LegalBench | ✓       | 2-gram TF, 토큰수준 |
| PROA                                 | 95      | 2         | LegalBench | ✓       | 1-gram TF, 토큰수준 |
| International Citizenship Questions  | 1000    | 2         | LegalBench | ✓       | 1-gram TF, 토큰수준 |
| CLadder                              | 8917    | 2         | -          |         | 2-gram TF, 단어수준 |
| ProntoQA                             | 7200    | 2         | -          | ✓       | 2-gram TF, 단어수준 |

"최적 특징 벡터" 열은 각 데이터셋에서 검증(validation) 분할 기준 정확도가 가장 높았던 특징 벡터를 뜻한다. 규모가 큰 데이터셋(CLadder, ProntoQA 등)은 비용을 줄이기 위해 무작위 부분집합만 사용했다 — 위 표의 "문항 수"가 그 검사 대상 인스턴스 수다.

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

주의할 점 — 모든 LLM이 모든 데이터셋에서 테스트된 것은 아니다. 데이터셋과 LLM 조합마다 존재 여부가 다르며, 이는 뒤에서 볼 모델 계열별 통계 검정력에 직접 영향을 준다.

## 13개 표층 특징 벡터

핵심은 여기다 — **이 논문이 실제로 쓴 특징은 유니그램·바이그램의 빈도 통계와, 문항 전체의 가독성·다양성 지표뿐이다.** 문항 길이나 선택지 길이 차이, 정답 위치 통계를 직접 특징으로 넣지는 않았다(길이 정보는 가독성 지표에 간접적으로만 녹아 있다). 이 점은 Gururangan et al.이 가설 길이 자체를 명시적 신호로 짚었던 것과 대조된다 — Pacchiardi et al.은 더 좁은 특징 집합으로도 절반 가까이에서 예측력이 검출된다는 것을 보인 셈이다.

n-gram 계열 특징은 세 축의 조합으로 열두 가지가 나온다.

| 축          | 선택지                                              |
| ----------- | --------------------------------------------------- |
| n-gram 종류 | 유니그램(1-gram) / 유니그램+바이그램(1+2-gram)      |
| 추출 단위   | 단어 수준(word-level) / 토큰 수준(GPT-2 토크나이저) |
| 가중치 방식 | TF / TF-IDF / Presence                              |

- **TF(Term Frequency)**: 문항에 그 n-gram이 등장한 원시 횟수.
- **TF-IDF**: 데이터셋 전체에서 그 n-gram이 얼마나 희소한지로 가중치를 보정한 값.
- **Presence**: 그 n-gram이 문항에 있으면 1, 없으면 0인 이진 지표.

2(n-gram 종류) × 2(추출 단위) × 3(가중치) = 12개 벡터. 여기에 **가독성·다양성 지표** 벡터 1개를 더해 총 13개 특징 벡터가 된다.

| 지표                | 무엇을 재나                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| Flesch Reading Ease | 문장 길이·음절 수 기반의 읽기 쉬움 점수(높을수록 쉬움)                    |
| Gunning Fog Index   | 복잡한(3음절 이상) 단어 비율과 문장 길이로 추정한 이해에 필요한 학년 수준 |
| SMOG Index          | 다음절 단어 수 기반의 읽기 난이도 추정치                                  |
| Yule's K            | 어휘 반복도 — 값이 클수록 같은 단어를 자주 반복(어휘 다양성이 낮음)       |

이 네 값을 이어 붙인 것이 13번째 특징 벡터다. 저자들은 TF-IDF가 대체로 TF·Presence보다 예측력이 낮다는 것을 관찰했다 — IDF 보정이 데이터셋 전체에 흔한(그래서 레이블과 강하게 상관될 가능성이 큰) n-gram의 가중치를 오히려 깎아내리기 때문이다.

## 평가 지표: Cohen's κ

데이터셋마다 레이블 수와 레이블 분포가 달라 단순 accuracy를 바로 비교하기 어렵다. 저자들은 **예측과 정답의 주변분포에서 기대되는 우연 일치를 제거한** Cohen's $$\kappa$$를 공통 척도로 사용했다.

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $$P_o$$: 분류기의 예측과 정답이 실제로 일치한 비율(관측 정확도).
- $$P_e$$: 우연히 일치할 것으로 기대되는 비율. 여기서 $$P_e$$는 일반적으로 단순한 $$1/K$$가 아니라 **예측 레이블과 실제 레이블의 주변분포로부터** 계산된다. 두 주변분포가 모두 균등할 때에만 $$1/K$$와 같아진다.
- $$\kappa = 0$$이면 우연 수준, $$\kappa = 1$$이면 완벽 일치, 음수면 우연보다도 못한 체계적 불일치를 뜻한다.

논문이 참고한 관습적 해석 기준(Landis & Koch, 1977)은 다음과 같다.

| κ 범위        | 해석                           |
| ------------- | ------------------------------ |
| κ ≤ 0.2       | 미미하거나 없음                |
| 0.2 < κ ≤ 0.4 | 작지만 탐지 가능(slight\~fair) |
| 0.4 < κ ≤ 0.6 | 상당한(fair-moderate) 일치     |
| κ > 0.6       | 상당히 강한(considerable) 일치 |

논문은 $$\kappa > 0.2$$를 채택하는데, 이는 통계적 유의성 기준이 아니라 **이 논문이 사용하는 선별(screening) 문턱**이자 관습적인 효과 크기 기준이다. $$\kappa > 0.2$$를 넘겼다는 것이 곧 "취약점 확정"을 뜻하지는 않으며, 반대로 넘기지 못했다는 것이 "안전 확인"을 뜻하지도 않는다. $$\kappa$$의 정확한 유도와 다중 평가자로의 확장은 [#16](/blog/2026/kappa-agreement/)에서 별도로 다룬다.

## 실험 파이프라인

각 데이터셋마다 다음 절차를 거쳤다.

1. **분할**: 데이터를 train/validation/test로 나눈다.
2. **학습**: train 분할에서 13개 특징 벡터 각각에 대해 로지스틱 회귀 분류기를 학습한다. 정규화는 L2(λ=1)와 L1(λ=1, λ=10) 세 가지 설정을 모두 시도한다.
3. **모델 선택**: validation 분할에서 정확도가 가장 높은 (특징 벡터, 정규화 설정) 조합을 고른다.
4. **평가**: 선택된 조합을 test 분할에 적용해 Cohen's κ를 계산한다.
5. **LLM 대조**: 같은 test 분할을 n-gram 분류기가 맞힌 부분집합과 못 맞힌 부분집합으로 나누고, 각 LLM이 두 부분집합에서 얼마나 다르게 수행하는지 비교한다.

코드와 데이터는 GitHub([Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability](https://github.com/Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability))에 공개돼 있다.

# Experiments

## 결과 1 — 레이블 예측 가능성: 19개 중 9개에서 κ > 0.2

위 표에서 ✓ 표시된 9개 데이터셋(Fantasy Reasoning, NeuBAROCO, SpaceNLI, bAbI Task 16, Corporate Lobbying, Function of Decision Section, PROA, International Citizenship Questions, ProntoQA)에서 **단순 특징으로 학습한 분류기의 Cohen's $$\kappa$$가 0.2를 넘었다.** 19개 중 9개, 약 47%다. 이 중 대부분은 0.2\~0.6 구간에 있었지만, **Corporate Lobbying과 SpaceNLI는 $$\kappa > 0.6$$**까지 올라갔다 — 법적 로비 목적 판별이나 공간 추론처럼 문항을 꼼꼼히 읽어야 풀릴 것 같은 과제에서도, 표층 특징의 예측력이 상당한 수준으로 검출된다.

몇 가지 패턴이 눈에 띈다.

- **LegalBench 계열에서 검출률이 높다**: 5개 LegalBench 과제 중 4개(Corporate Lobbying, Function of Decision Section, PROA, International Citizenship Questions)가 문턱을 넘었다. 한 가지 가능한 설명은 법률 문서의 반복적인 정형 표현(boilerplate)이다. 다만 이 논문은 LegalBench의 높은 예측력이 실제로 boilerplate에서 비롯됐는지를 별도로 검증하지 않았다 — 이는 이 글의 해석이지 논문이 확인한 인과적 설명이 아니다.
- **가독성·다양성 지표가 의외로 자주 최적이었다**: 19개 중 10개 데이터셋에서 n-gram이 아니라 가독성·다양성 지표(Flesch, Gunning Fog, SMOG, Yule's K)가 최적 특징으로 선택됐다. 다만 이들 대부분은 κ ≤ 0.2로, "최적이었지만 문턱을 넘지는 못한" 경우다.
- **CLadder에서는 신호가 검출되지 않았다**: 8,917개 인스턴스라는 가장 큰 데이터셋인데도 최적 특징(2-gram TF)의 κ가 0.2를 넘지 못했다. 정확히 말하면 **이 논문이 검사한 n-gram·가독성 특징에서는 $$\kappa > 0.2$$ 신호가 검출되지 않았다**는 것이다. 저자도 "적어도 n-gram 특징에 대해서는"이라고 명확히 제한한다. 다른 종류의 지름길에 대해 안전하다는 뜻은 아니다.

## 결과 2 — 모델이 그 문항에서 더 잘하는가

표층 특징만으로 정답이 예측된다는 사실은 사슬의 두 번째 칸까지만 보여준다. 세 번째 칸 — 모델이 그 경로를 실제로 사용하는가 — 을 보기 위해 저자들은 $$\kappa > 0.2$$를 넘긴 9개 데이터셋에서, 각 LLM의 test 인스턴스를 n-gram 분류기가 **맞힌 부분집합**과 **못 맞힌 부분집합**으로 나누고, "맞힌 부분집합에서 유독 더 잘하는가"를 모델 계열별로 단측 대응표본 t-검정했다. 다중비교는 Benjamini-Hochberg 절차로 보정했다.

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

**보정 후 $$p < .05$$인 계열은 없었다.** 가장 작은 보정 p-value도 0.0594다. 저자들은 $$q < .1$$이 나온 네 계열(Meta·OpenAI·Mistral AI·Writer)을 **중간 정도의 증거(moderate evidence)**로, $$q < .2$$인 두 계열(Anthropic·Aleph Alpha)을 **더 약한 증거(weaker evidence)**로 해석한다. 통상적인 5% 기준을 넘긴 계열은 하나도 없으므로, 이 표를 "유의하게 더 잘 맞힌다"로 읽어서는 안 된다.

저자들 스스로도 이 결과를 신중하게 해석한다. 계열별 검정력 차이가 크기 때문이다 — OpenAI는 18개 모델이 12개 이상의 데이터셋에서 테스트됐지만, AI21 Labs나 Cohere 같은 계열은 모델 2개가 데이터셋 5개에서만 테스트됐다. 즉 상위권에 OpenAI 계열이 올라온 것은 표본이 가장 많아 검정력이 높았다는 사실과 분리하기 어렵다. 같은 크기의 효과를 작은 표본에서 탐지하려면 애초에 데이터가 더 필요하다 — 이는 [#20](/blog/2026/statistical-power/)에서 다룰 통계 검정력 문제 그 자체다. 저자들도 논문에 명시한다: **"제공한 증거가, 그 LLM 계열들이 실제로 성공을 위해 n-gram에 의존한다는 것을 결정적으로 입증하지는 않는다."**

## Limitations — 이 논문이 보이지 않은 것

세 가지 한계를 분명히 해 두는 것이 이 논문을 정확히 인용하는 데 중요하다.

1. **검사한 표층 단서가 제한적이다.** 유니그램·바이그램과 네 개의 가독성·다양성 지표뿐이다. 문항 길이, 선택지 길이 차이, 정답 위치 히스토그램, 템플릿 구조 같은 다른 표층 단서는 검사되지 않았다. 신호가 검출되지 않은 10개 벤치마크는 "이 특징 집합에서 검출되지 않았다"는 뜻이지 "지름길이 없다"는 뜻이 아니다.
2. **9/19라는 판정이 관습적 문턱에 의존한다.** $$\kappa > 0.2$$는 Landis & Koch의 관습적 구간에서 온 효과 크기 기준이지, 사전에 정당화된 실질적 기준이나 통계적 유의성 기준이 아니다. 문턱을 0.15나 0.25로 옮기면 개수가 달라진다.
3. **모델이 실제로 n-gram을 사용한다는 분석은 결론적이지 않다.** 보정 후 $$p < .05$$인 계열이 없고, 계열별 모델 수와 데이터셋 수가 2\~18개로 크게 불균형해 검정력이 계열마다 다르다. 또한 t-검정이 잡아내는 것은 "n-gram이 맞히는 문항과 LLM이 맞히는 문항의 상관"이지, 인과적 의존이 아니다 — 두 부분집합이 문항 난이도에서도 다를 수 있기 때문이다.

## 토이 예제 — "가장 긴 선택지가 정답"이라면

Pacchiardi et al.이 실제로 쓴 특징은 n-gram과 가독성 지표지만, 표층 특징이 정답을 흘리는 현상 자체는 훨씬 단순한 예제로 감을 잡을 수 있다. 4지선다 객관식 벤치마크 1,000문항을 가정하자. 출제자가 정답 선택지를 쓸 때 무의식적으로 조건·단서를 더 자세히 적는 습관이 있다고 하자 — Gururangan et al.이 SNLI에서 발견한 "가설 길이가 레이블과 상관된다"는 현상과 같은 종류의 습관이다.

| 상황                            | 문항 비율 | 문항 수 | "가장 긴 선택지" 규칙의 성공 확률 |
| ------------------------------- | --------- | ------- | --------------------------------- |
| 정답이 유일하게 가장 긴 선택지  | 45%       | 450     | 1 (항상 성공)                     |
| 정답이 최장 길이 동률(2개 동률) | 15%       | 150     | 1/2 (동률 중 무작위 선택)         |
| 정답이 최장 길이가 아님         | 40%       | 400     | 0 (항상 실패)                     |

이 규칙의 기대 정답 수는 다음과 같다.

$$450 \times 1 + 150 \times \frac{1}{2} + 400 \times 0 = 450 + 75 = 525$$

즉 관측 정확도는 $$P_o = 525/1000 = 0.525$$다. 문항의 의미를 해석하지 않고 가장 긴 선택지를 고르는 규칙만으로 **52.5%** — 우연 수준 25%의 2.1배 — 를 맞힌다.

이를 $$\kappa$$로 환산하려면 가정을 하나 더 명시해야 한다. 앞서 봤듯 Cohen's $$\kappa$$의 기대 일치율은 실제 정답 위치와 규칙이 고른 위치의 **주변분포**에 따라 달라지기 때문이다. 정답 위치와 "가장 긴 선택지"의 위치가 A\~D에 각각 균등하게 분포한다고 추가로 가정하자. 그러면 주변분포에 따른 우연 일치율은 $$P_e = 0.25$$이고,

$$\kappa = \frac{0.525 - 0.25}{1 - 0.25} = \frac{0.275}{0.75} \approx 0.37$$

이 되어 논문의 선별 문턱 0.2를 넘는다. 즉 이 가상의 벤치마크는 실험 전에 "선택지 길이"라는 메타 특징 하나만으로 진단 기준에 걸린다.

이것이 왜 성능 보고를 오염시킬 수 있는지가 핵심이다. 실제 추론 능력이 있는 모델 A와, 사전학습 중 이런 유형을 많이 접해 "길다=정답"이라는 표층 규칙을 체득한 모델 B가 있다고 하자. 모델 B는 지문을 이해하지 못해도 이 규칙 하나로 52.5%를 확보한다. 모델 A의 실제 추론 정확도가 45%에 그친다면, 리더보드는 모델 B를 더 나은 추론 모델로 표시한다.

이런 현상이 모델마다 다른 정도로 나타난다면, 단순히 모든 모델 점수를 같은 만큼 올리는 데 그치지 않고 **모델 간 비교에도 영향을 줄 수 있다.** Pacchiardi et al.의 Table 3은 계열마다 이 경향의 강도가 다르다는 것을 보여주므로 그 가능성을 뒷받침하는 간접 증거다. 다만 Table 3이 특정 두 모델의 순위 역전을 직접 보여준 실험은 아니라는 점은 분명히 해 둔다.

## 진단 절차 — 새 벤치마크를 공개하기 전에

Pacchiardi et al.의 방법론은 그대로 하나의 사전 점검 절차로 일반화할 수 있다.

1. **메타 특징만 추출한다**: 문항이 무엇을 묻는지 해석할 필요가 없는 특징만 뽑는다 — n-gram 빈도, 가독성·다양성 지표, 그리고 원한다면 문항·선택지 길이, 정답 위치 히스토그램, 템플릿 식별자도 추가한다(Pacchiardi et al.은 앞의 두 가지만 썼다).
2. **표현력이 제한된 단순 분류기를 학습한다**: 로지스틱 회귀 정도가 적절하다. 표현력이 큰 모델을 쓰면 표층 신호가 아니라 과적합으로 정확도가 오를 위험이 있다.
3. **train/validation/test로 분할**하고, validation에서 하이퍼파라미터(정규화 방식·강도)를 고른다.
4. **test 분할에서 Cohen's κ를 계산**한다.
5. **점추정치와 불확실성을 함께 본다**: $$\kappa$$의 점추정치와 신뢰구간을 함께 보고([#18](/blog/2026/confidence-intervals/)), **사전에 정한 실질적 효과 크기 기준**을 넘는지 확인한다. 표본이 아주 크면 $$\kappa = 0.01$$도 통계적으로 유의할 수 있지만 실질적으로는 무시할 만하다. 반대로 표본이 작으면 큰 $$\kappa$$도 신뢰구간이 넓다. "유의한가"만 묻는 것([#19](/blog/2026/significance-testing/))으로는 부족하다.
6. **여러 특징 집합이나 여러 벤치마크를 동시에 검사한다면 다중비교를 고려한다**: 13개 특징 벡터에 대해 각각 검정하면 우연히 하나가 걸릴 확률이 올라간다([#20](/blog/2026/statistical-power/)).
7. **기준을 넘는 신호가 나오면, 어떤 단서였는지에 맞춰 처방한다**: 탐지된 교란의 종류와 무관한 개입은 효과가 없다. 아래 표가 그 대응이다.

| 탐지된 교란    | 대응                                                    |
| -------------- | ------------------------------------------------------- |
| 위치 단서      | 정답 위치 counterbalancing                              |
| 길이 단서      | 선택지 길이 균형화                                      |
| 특정 어휘 단서 | 그 단서를 레이블 사이에 counterbalance한 대조 문항 제작 |
| 템플릿 단서    | 패러프레이즈·다중 템플릿 생성                           |
| 출처 단서      | 출처별 층화(stratification)와 held-out source 평가      |

여기서 흔한 실수 하나를 짚어둘 만하다. **선택지 순서 재배치는 위치 편향에는 유효하지만, "not이 모순을 예측한다" 같은 어휘 인공물은 전혀 없애지 못한다.** 문항 텍스트 안에 박힌 단서는 순서를 바꿔도 그대로 남는다. 탐지와 처방을 짝지어 생각해야 하는 이유다.

## 저자들이 제안하는 통제 방법

논문 5.1절 "Controlling for Clever Hans"는 이 문제를 사람·동물 행동 실험이 오랫동안 다뤄온 문제의 연장선에 놓는다. 이런 교란 요인은 피할 수 없으므로, 찾아내고 제거하고 통제하는 데 상당한 노력을 들인다는 것이 그 분야의 표준 관행이다.

| 분야           | 통제 기법                                                               |
| -------------- | ----------------------------------------------------------------------- |
| 동물 행동 실험 | 실험자가 답을 모르게 하는 불투명 고글, 사전 녹음된 음성 명령            |
| 의학 연구      | 이중맹검(double-blind) — 실험자도 어느 쪽이 처치군인지 모르게 함        |
| 인간 행동 연구 | 자극의 무작위화(randomisation)·변형(variation)·균형화(counterbalancing) |
| AI 평가(현재)  | 이런 관행이 최근에야, 그것도 일부 사례에서만 등장하기 시작함            |

저자들의 결론은 명확하다. **"완전히 통제된 벤치마크라면 문항의 한두 단어짜리 특징으로 정답을 예측할 수 있어서는 안 된다."** 자극의 무작위화·다양화·균형화가 AI 평가에서도 표준 관행이 되어야 한다는 것이다.

# 통계 요약

| 개념                             | 정의                                                     | 이 논문에서의 역할                                                     |
| -------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| Cohen's κ                        | $$\kappa = (P_o - P_e) / (1 - P_e)$$                     | 레이블 수·분포가 다른 벤치마크 간에 우연 보정된 예측력을 비교하는 척도 |
| κ 해석 문턱(Landis & Koch, 1977) | 0.2/0.4/0.6 구간 구분                                    | 논문이 참고한 관습적 효과 크기 기준 — 유의성 기준이 아님               |
| 단측 대응표본 t-검정             | 같은 LLM의 두 부분집합(성공/실패 예측) 간 평균 차이 검정 | 모델이 n-gram 신호를 활용할 가능성에 대한 간접 증거                    |
| Benjamini-Hochberg 보정          | 다중비교 시 거짓 발견율(FDR) 통제                        | 11개 모델 계열을 동시에 검정할 때 우연한 유의성 과다 검출 방지         |

# Conclusion

이 논문의 결론은 한 문장으로 압축된다. **벤치마크를 공개하기 전에 "목표 능력 없이도 정답이 예측되는가"를 검사해야 한다.** 19개 중 9개에서 단순 표층 특징의 $$\kappa$$가 0.2를 넘었고, 그중 두 개(Corporate Lobbying, SpaceNLI)는 0.6을 넘었다. 그리고 일부 모델 계열이 이러한 단서를 활용하고 있을 가능성을 시사하는 증거가 나왔다 — 다만 보정 후 5% 기준을 넘은 계열은 없으며, 저자들도 결론적 증거는 아니라고 명시한다.

이 발견을 시리즈의 다른 편과 정확히 구분해 둘 필요가 있다.

- **[#8](/blog/2026/mcqa-fragility/) 위치 편향과의 관계**: 위치 편향은 모델이 선택지 위치에 민감하게 반응하는 **모델·평가 프로토콜 상호작용**이고, Clever Hans는 문항 내용에 정답과 연결된 불필요한 단서가 존재하는 **데이터 설계 문제**다. 둘 다 목표 능력 외의 경로가 점수에 들어온다는 점에서는 같다. 다만 위치 편향을 순수한 모델 인공물로만 보아서는 안 된다 — 모델의 위치 선호는 정답 위치가 불균형한 데이터셋과 결합될 때 비로소 점수를 왜곡하므로, 실제로는 모델 × 프로토콜 × 데이터의 상호작용이다. 처방이 다른 이유도 여기 있다: 위치 편향은 위치 counterbalancing과 프롬프트 변형으로 완화되지만, 어휘 단서는 문항 자체를 다시 설계해야 없어진다.
- **[#23](/blog/2026/contamination-reproducibility/) 오염과의 관계**: 오염(contamination)은 "모델이 이 문항의 답을 사전학습 중에 이미 봤다"는 문제다. Clever Hans는 그보다 약한 조건에서 발생한다 — 답을 본 적이 없어도, 문항의 표층 패턴만으로 우연 이상을 맞힐 수 있는 경로가 존재한다. 둘 다 점수를 부풀릴 수 있지만 처방이 다르다. 오염은 탈오염·홀드아웃·동적 벤치마크로 대응하고, Clever Hans는 자극의 무작위화·균형화로 대응한다. 데이터를 새로 만들어 오염이 없음을 보장해도 Clever Hans는 남을 수 있다 — 새 문항을 쓰는 사람이 여전히 같은 작문 습관을 가지고 있기 때문이다.

이 논문이 [#1](/blog/2026/what-is-evaluation/)의 구성개념 무관 분산에 기여하는 바는 이론이 아니라 **관측 가능한 실증**이라는 점이다. 다만 무엇을 실증했는지는 정확히 말해야 한다 — 이 논문은 잡음의 크기를 잰 것이 아니라, **19개 중 9개에서 목표 능력 없이도 사용할 수 있는 표층 신호의 예측력을 $$\kappa$$로 정량화했다.** 그 신호가 실제 모델 점수에 얼마나 섞여 들어갔는지는 별개의, 아직 결론 나지 않은 질문이다.

Gururangan et al. (2018)부터 이어진 6년치 계보가 도달한 지점은 같다 — **모델의 능력을 재고 싶다면, 먼저 그 문항이 능력 없이도 풀리지 않는지부터 확인해야 한다.**

# 참고 문헌

- Pacchiardi, Tešić, Cheke, Hernández-Orallo (University of Cambridge), 2024. [Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/abs/2410.11672) (arXiv:2410.11672).
- [GitHub: Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability](https://github.com/Kinds-of-Intelligence-CFI/benchmark-ground-truth-predictability) — 실험 재현 코드.
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith, 2018. [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/) (NAACL 2018).
- Poliak, Naradowsky, Haldar, Rudinger, Van Durme, 2018. [Hypothesis Only Baselines in Natural Language Inference](https://arxiv.org/abs/1805.01042) (\*SEM 2018).
- McCoy, Pavlick, Linzen, 2019. [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://aclanthology.org/P19-1334/) (ACL 2019).
- Niven, Kao, 2019. [Probing Neural Network Comprehension of Natural Language Arguments](https://aclanthology.org/P19-1459/) (ACL 2019).
- Kavumba, Inoue, Heinzerling, Singh, Reisert, Inui, 2019. [When choosing plausible alternatives, Clever Hans can be clever](https://arxiv.org/abs/1911.00225) (arXiv:1911.00225).
- Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann, 2020. [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z). Nature Machine Intelligence.
- Landis, Koch, 1977. The Measurement of Observer Agreement for Categorical Data. Biometrics 33(1), 159-174.

---

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 네 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/everything-benchmark/">범용 벤치마크라는 주장</a> — Raji et al. — 모든 것을 잰다는 말</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
  <li><strong>(현재 글)</strong> 표층 특징이 정답을 예측한다 — Clever Hans, 데이터셋 인공물</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="5">
  <li><a href="/blog/2026/measurement-scales/">척도와 허용 연산</a> — Likert 평균을 내도 되는가</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><a href="/blog/2026/generation-metrics/">생성 지표와 그 타당도</a> — BLEU에서 COMET까지</li>
  <li><a href="/blog/2026/mcqa-fragility/">객관식 평가는 왜 흔들리나</a> — 위치 편향과 포맷 민감도</li>
</ol>

**3부. LLM 벤치마크 지형도**

<ol start="9">
  <li><a href="/blog/2026/knowledge-benchmarks/">지식과 추론 — MMLU 계열의 흥망</a> — MMLU·GPQA·BBH</li>
  <li><a href="/blog/2026/math-code-benchmarks/">검증 가능한 도메인 — 수학과 코드</a> — GSM8K·MATH·HumanEval·SWE-bench</li>
  <li><a href="/blog/2026/mt-bench-to-arena/">개방형 대화 — MT-Bench에서 Arena까지</a> — judge 기반 벤치의 등장</li>
  <li><a href="/blog/2026/capability-axes-benchmarks/">능력의 다른 축</a> — 지시따르기·긴 문맥·사실성</li>
  <li><a href="/blog/2026/korean-benchmarks/">한국어 벤치마크</a> — 번역이 아니라 원산, 그리고 문화 타당도</li>
  <li><a href="/blog/2026/helm-holistic-evaluation/">점수 하나가 아니라 행렬로</a> — HELM — 시나리오 × 지표</li>
</ol>

**4부. 사람이 읽는다 — 정성평가와 일치도**

<ol start="15">
  <li><a href="/blog/2026/human-evaluation-design/">사람 평가 설계</a> — 루브릭·Likert·pairwise·BWS</li>
  <li><a href="/blog/2026/kappa-agreement/">우연을 빼다 — κ 계열</a> — Cohen·Fleiss·weighted·Krippendorff</li>
  <li><a href="/blog/2026/kappa-paradox/">κ의 역설</a> — 일치율 90%인데 κ가 0.21</li>
</ol>

**5부. 차이는 진짜인가 — 정량평가의 통계**

<ol start="18">
  <li><a href="/blog/2026/confidence-intervals/">점수는 추정치다</a> — 이항비율 신뢰구간과 Wald의 실패</li>
  <li><a href="/blog/2026/significance-testing/">차이는 유의한가</a> — paired bootstrap·순열검정·McNemar</li>
  <li><a href="/blog/2026/statistical-power/">몇 개를 재야 하나</a> — 검정력·표본크기·다중비교</li>
  <li><a href="/blog/2026/error-bars-for-evals/">LLM eval의 통계 실무</a> — 클러스터 SE·IQM·분산 분해</li>
</ol>

**6부. 신뢰할 수 있는 평가 체계**

<ol start="22">
  <li><a href="/blog/2026/judge-statistics/">judge를 통계로 다루기</a> — 편향·Bradley-Terry·PPI</li>
  <li><a href="/blog/2026/contamination-reproducibility/">오염·재현성·효율</a> — 오염 검정·harness·IRT</li>
  <li><a href="/blog/2026/safety-evaluation-statistics/">안전 평가의 통계와 체계 설계</a> — 희귀사건·calibration·체크리스트</li>
</ol>

본 시리즈는 24편으로 구성된다.
