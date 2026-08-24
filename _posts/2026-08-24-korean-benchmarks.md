---
layout: post
title: "한국어 벤치마크 — 번역이 아니라 원산, 그리고 문화 타당도"
date: 2026-08-24 09:14:00 +0900
description: "LLM 평가 체계 시리즈 #14 — 번역 MMLU와 KMMLU 사이의 낙차로 보는 구성 타당도, 그리고 정렬조차 문화마다 다른 이유"
categories: [paper]
tags: [evaluation, korean-nlp, benchmark, construct-validity, kmmlu, kornat, paper]
giscus_comments: true
related_posts: true
---

> [KMMLU: Measuring Massive Multitask Language Understanding in Korean](https://arxiv.org/abs/2402.11548) (Son et al., NAACL 2025)

# Introduction

숫자 하나로 이 글을 시작한다. GPT-4는 영어 MMLU에서 85.5%를 받는다. 같은 문항을 Azure Translate로 한국어로 번역한 버전에서는 77.0%로 떨어진다. 그리고 번역이 아니라 한국 시험 원본에서 새로 수집한 KMMLU에서는 59.95%로 또 떨어진다. 85.5 → 77.0 → 59.95. 같은 모델인데 세 개의 다른 점수가 나온다.

이 낙차를 어떻게 읽어야 할까. "모델이 한국어를 못해서"라고 뭉뚱그리면 [#1](/blog/2026/what-is-evaluation/)에서 세운 구성 타당도(construct validity) 질문을 건너뛰는 것이다. 85.5에서 77.0으로의 낙차와, 77.0에서 59.95로의 낙차는 **성격이 다르다.** 앞의 낙차는 같은 문항을 번역만 했을 뿐인데 생긴 손실이고, 뒤의 낙차는 애초에 존재하지 않던 문항 — 한국 행정·법·역사 지식을 묻는 문항 — 이 새로 들어오면서 생긴 차이다. 전자는 잡음(noise)에 가깝고 후자는 신호(signal)에 가깝다. 그런데 번역 벤치마크 하나의 점수만 보면 이 둘을 구분할 방법이 없다. 잡음과 신호가 한 숫자에 뭉쳐 있다.

이것이 이 시리즈에서 구성 타당도가 가장 선명하게 드러나는 지점이다. **영어 벤치마크를 한국어로 번역하면, 점수만 달라지는 게 아니라 무엇을 재고 있는지 자체가 달라진다.** [#4](/blog/2026/benchmark-construct-validity/)가 벤치마크 445편을 리뷰하며 확인한 "구성 타당도 실패가 만연하다"는 진단이, 언어를 넘어가는 순간 훨씬 노골적으로 드러나는 것이다.

이 글은 번역이 정확히 무엇을 깨뜨리는지 네 갈래로 나눠 살펴본 뒤, 번역이 아니라 원산(原産)으로 만들어진 한국어 벤치마크 다섯 개 — KMMLU, KMMLU-Redux/Pro, HAE-RAE Bench, CLIcK, KorNAT — 를 그 대응으로 읽는다. 그리고 이 벤치마크들을 일관되게 평가하기 위한 인프라인 HRET도 함께 다룬다. 특히 KorNAT은 이 시리즈의 뒷부분([#25](/blog/2026/safety-evaluation-statistics/))에서 다룰 안전 평가의 문화 의존성을 먼저 보여주는 사례라 비중 있게 다룬다.

# Background

## 번역이 깨뜨리는 네 가지

"영어 벤치마크를 번역하면 구성개념이 바뀐다"는 주장은 추상적으로 들리기 쉽다. 무엇이 정확히 깨지는지 네 갈래로 쪼개면 훨씬 구체적으로 잡힌다.

**1. 문화·제도 지식.** 한국의 법·행정·역사·사회 규범은 번역이라는 연산으로 옮겨오지 않는다. 애초에 원본 영어 문항에 없는 지식이기 때문이다. 예를 들어 "PSAT(공직적격성평가)에서 요구하는 국정감사 절차"를 묻는 문항은 아무리 잘 번역해도 영어 MMLU에서 나올 수 없다 — 그런 문항 자체가 영어 MMLU에 없기 때문이다. 번역은 원본에 있는 걸 옮기는 연산이지, 원본에 없는 걸 만들어내는 연산이 아니다.

**2. 언어 구조.** 한국어는 교착어(agglutinative language)다. 어간에 조사·어미가 층층이 붙어 문법 기능을 표시하고("가다"에서 "가요/갑니다/가십니다/가시죠"까지 경어법만으로도 활용형이 갈린다), 이 구조는 영어의 어순·전치사 중심 문법과 근본적으로 다르다. 여기에 실무적 문제가 하나 더 얹힌다 — **토큰화 효율**이다. 영어 중심 코퍼스로 학습한 BPE 토크나이저는 한글 음절 블록을 잘게 쪼개는 경우가 많아, 같은 의미를 표현하는 데 더 많은 토큰이 든다. GPT-4 테크니컬 리포트는 이 문제를 직접 인정한다 — 다국어 MMLU 평가에서 "일부 언어가 훨씬 긴 토큰 시퀀스에 대응하기 때문에" 정규 5-shot 대신 3-shot을 썼다고 명시한다. 문항 자체는 그대로인데, 언어가 바뀌는 것만으로 프롬프트의 유효 정보량이 달라지는 것이다.

**3. 번역투(translationese).** 번역된 문장은 원어민이 쓴 문장과 통계적 분포가 다르다. 어순이 부자연스럽거나, 관용구가 직역되거나, 문장이 필요 이상으로 길어진다. GPT-4 리포트도 스스로 "번역이 완벽하지 않아 미묘한 정보를 잃을 수 있고, 이것이 성능을 해칠 수 있다"고 적는다. 문제는 이 왜곡이 난이도를 어느 방향으로도 밀 수 있다는 점이다 — 번역투가 문항을 더 어렵게 만들 수도, 반대로 고유명사를 영어로 남겨두는 번역 관행 때문에 오히려 힌트가 새어 들어가 더 쉬워질 수도 있다.

**4. 정답의 문화 의존성.** 사실 지식 문항은 그래도 "정답"이 하나로 고정된다. 그런데 가치 판단이 얽힌 문항 — "이 행동은 사회적으로 용인되는가" 같은 — 은 **정답 자체가 문화마다 다르다.** 번역은 질문의 표현만 옮길 뿐, 그 질문에 대한 사회적 합의까지 옮기지는 못한다. 이 넷째 갈래가 뒤에서 다룰 KorNAT의 핵심 문제의식이다.

CLIcK 같은 벤치마크는 이 네 갈래 중 1번(문화)과 2번(언어 구조)을 아예 두 개의 축으로 명시적으로 쪼개 설계했다. [#4](/blog/2026/benchmark-construct-validity/)가 지적한 "벤치 여러 개가 사실 같은 축을 재고 있다"는 문제의 반대 방향 — 서로 다른 구성개념을 하나의 점수로 뭉치지 않고 처음부터 갈라놓는 설계다.

# Method

## KMMLU — 원산이라는 것의 의미

[KMMLU: Measuring Massive Multitask Language Understanding in Korean](https://arxiv.org/abs/2402.11548) (Son et al., NAACL 2025)은 **45개 범주, 35,030문항**으로 구성된 객관식 벤치마크다. 결정적 차이는 수집 방법이다 — 영어 MMLU를 번역한 게 아니라, 변호사·세무사·의사 같은 각종 국가자격시험과 공무원 시험(PSAT 등) **원본 문제**에서 그대로 가져왔다. 인문학부터 STEM까지 범주가 걸쳐 있고, 저작권 문제로 국어·의학·금융 일부 영역은 제외됐다.

**핵심 수치 — 사람과 모델의 위치.**

| 주체                              | 점수   | 비고                                                               |
| --------------------------------- | ------ | ------------------------------------------------------------------ |
| 사람 평균                         | 62.6%  | 시험 응시자 실제 성적 데이터 (약 90%의 시험에서 확보)              |
| 인간 전문가 하한                  | 80%대  | PSAT 최근 5개년 평균 합격선 83.7%를 전문가 최소 성능 기준으로 삼음 |
| GPT-4 (gpt-4-0613, 5-shot Direct) | 59.95% | 전체 모델 중 최고                                                  |
| HyperCLOVA X                      | 53.40% |                                                                    |
| Polyglot-Ko-12.8B                 | 29.26% | 한국어 특화 오픈 모델인데도 저조                                   |

**GPT-4가 사람 평균(62.6%)에도 못 미친다.** 영어 MMLU에서는 GPT-4가 사람을 압도적으로 앞서는 것과 대비된다 — 구성개념이 바뀌면 모델과 사람의 상대적 위치 자체가 뒤집힐 수 있다는 걸 보여주는 장면이다.

논문은 KMMLU 문항의 **20.4%가 한국 문화·사회 규범·법률 지식을 요구한다**고 정량화한다(이 부분집합을 KMMLU-KOR, 1,305문항이라 부른다). 반면 번역 MMLU는 애초에 미국 정부 체계에 익숙함을 전제하는 "U.S.-centric" 문항이 많아 한국 문화를 다룰 수가 없다. 카테고리 하나를 예로 들면 "한국사"에서 GPT-4는 35%, HyperCLOVA X는 44%를 받는다 — 일반 지식 카테고리보다 훨씬 큰 격차다.

**KMMLU-HARD.** GPT-3.5-Turbo, Gemini Pro, HyperCLOVA X, GPT-4 네 모델 중 하나라도 틀리는 문항만 추려 **4,104문항**을 만들었다. 이 부분집합에서 GPT-4-Turbo는 30.52%(Direct, 평균)에 그친다. 4지선다 무작위 추측(25%)에 가까운 수치다.

## KMMLU-Redux / KMMLU-Pro — 원산도 오류가 있다

[From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation](https://arxiv.org/abs/2507.08924) (Hong et al., LG AI Research, arXiv 2025)은 KMMLU 자체의 결함을 정면으로 다룬다. 저자들은 원본 KMMLU를 재검수해 **7.66%의 문항에서 정답 유출·모호한 문제·표기 오류 같은 치명적 결함**을 발견했다. 여기에 데이터셋 내부 중복 5.36%, 학습·평가 셋 간 오염 5.46%, 사전학습 코퍼스(FineWeb2)와의 오염 1.88%까지 겹친다.

- **KMMLU-Redux**: 국가기술자격시험(100개 시험, 14개 산업 도메인) 부분집합만 다시 추려, 오류 문항을 걷어내고 소형 LLM 여러 개가 이미 맞히는 — 즉 변별력이 낮은 — 문항까지 추가로 걸러냈다. 최종 **2,587문항**만 남았고, 원본 대비 38.6%가 제거됐다.
- **KMMLU-Pro**: 국가전문자격시험(변호사·회계사·의사·치과의사·약사·변리사 등 14개 자격증 — 법률 4종, 세무·회계 3종, 감정평가 2종, 의료 5종) 기반으로 새로 구성한 **2,822문항**.

| 모델                         | KMMLU-Redux | KMMLU-Pro |
| ---------------------------- | ----------- | --------- |
| o1                           | 81.14%      | 78.09%    |
| Claude 3.7 Sonnet (thinking) | 79.36%      | 77.70%    |
| Llama 3.3 70B                | 56.17%      | 53.24%    |

KMMLU와 KMMLU-Redux 사이의 모델 순위 상관은 Spearman ρ = 0.995로 거의 완벽하게 유지된다 — 즉 오류를 걷어내도 "누가 더 잘하는가"의 순서 자체는 크게 바뀌지 않는다. 다만 절대 점수의 신뢰도는 오류가 섞인 원본보다 훨씬 높아진다. 이 지점이 뒤에서 다룰 [#24](/blog/2026/contamination-reproducibility/)의 예고편이다 — **원산 벤치마크도 오염과 오류에서 자유롭지 않다.** 번역이 만드는 문제와는 다른 종류의 문제지만, "벤치마크 점수를 그대로 믿지 말라"는 결론은 같다.

## HAE-RAE Bench — 번역으로는 못 만드는 문항

[HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models](https://arxiv.org/abs/2309.02706) (Son et al., LREC-COLING 2024, EleutherAI/OnelineAI/MODULABS)는 KMMLU보다 먼저 나온, 규모는 작지만 목적이 더 뾰족한 벤치마크다. **6개 태스크, 총 1,538문항**으로 구성된다.

| 태스크                            | 문항 수 | 성격                   |
| --------------------------------- | ------- | ---------------------- |
| Loan Words (외래어)               | 169     | 국립국어원 순화어 대응 |
| Rare Words (희귀어)               | 405     | 저빈도 한국어 어휘     |
| Standard Nomenclature (표준 명칭) | 153     | 공식 표기 규범         |
| Reading Comprehension (독해)      | 447     | 한국어 지문 독해       |
| General Knowledge (일반 상식)     | 176     | 한국 관련 상식         |
| History (역사)                    | 188     | 한국사                 |

**"번역으로는 못 만드는 문항"의 구체적 예가 Loan Words 태스크다.** 이 태스크는 외래어에 대응하는 **국립국어원 공인 순화어**를 맞히는 문제다. 예컨대 특정 외래어에 대해 정부가 공식적으로 지정한 한국어 대체어가 무엇인지 묻는데, 이 매핑 정보 자체가 영어 데이터에는 존재하지 않는다. 영어 문항을 아무리 정교하게 번역해도 "이 외래어의 한국어 순화어는 무엇인가"라는 질문은 만들어낼 수 없다 — 질문의 재료(순화어 목록)가 애초에 한국어 자원에만 있기 때문이다.

결과도 이 설계를 뒷받침한다. 0-shot 기준 한국어 특화 오픈모델 Polyglot-Ko-12.8B가 평균 59.5%로, 훨씬 큰 다국어 모델인 UMT5-13B(34.2%)나 Llama-2-13B(35.9%)를 앞선다. 저자들은 "인-컨텍스트 러닝만으로는 부족하다"고 적는다 — few-shot 예시 몇 개를 보여주는 것으로는 애초에 학습 데이터에 없던 문화적 지식을 끌어낼 수 없다는 뜻이다.

## CLIcK — 문화와 언어를 두 축으로 쪼개다

[CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean](https://arxiv.org/abs/2403.06412) (Kim et al., LREC-COLING 2024)는 공식 한국어 시험·교과서에서 뽑은 **1,995개 QA 쌍**을, **문화(Culture)**와 **언어(Language)** 두 대범주 아래 **11개 세부 범주**로 나눠 설계했다.

| 대범주         | 세부 범주                                        |
| -------------- | ------------------------------------------------ |
| Culture (8개)  | 사회, 전통, 역사, 법, 정치, 경제, 지리, 대중문화 |
| Language (3개) | 텍스트 지식, 기능적 지식, 문법                   |

여기서 눈여겨볼 설계는 **문항 하나하나에 "이 문항을 풀려면 어떤 지식이 필요한가"를 세부 범주로 라벨링**했다는 점이다. 이게 왜 중요한가 — [#4](/blog/2026/benchmark-construct-validity/)가 보인 문제, 즉 "벤치마크 여러 개가 사실 하나의 축만 재고 있다"는 함정을 CLIcK은 처음부터 피해간다. 하나의 총점 뒤에 문화 지식과 언어 능력이라는 서로 다른 구성개념이 섞여 있는 게 아니라, 애초에 갈라서 라벨링해 두었기 때문에 "이 모델이 부족한 게 문화 지식인지 언어 구조인지"를 사후에 분해해 볼 수 있다.

13개 모델을 평가한 결과, GPT-3.5는 전체 약 49.30%, Claude-2는 약 51.72%였고 오픈소스 모델은 10\~50% 범위에 흩어졌다. 흥미로운 점은 **문화 범주보다 언어 범주, 그중에서도 기능적 지식(Functional Knowledge)이 모델들에게 더 어려웠다**는 것이다 — 흔히 "한국어를 못해서 문제"라고 하면 문화 지식 부족을 떠올리기 쉽지만, 실제로는 언어 구조 자체를 다루는 능력이 더 큰 병목일 수 있다는 뜻이다.

## KorNAT — 정렬의 정답도 문화마다 다르다

[KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge](https://arxiv.org/abs/2402.13605) (Lee et al., ACL 2024 Findings)는 이 글에서 가장 중요하게 다뤄야 할 벤치마크다. 지금까지의 벤치마크가 "사실 지식을 아는가"를 물었다면, KorNAT은 **"이 모델이 한국이라는 특정 국가에 얼마나 정렬(align)되어 있는가"**를 묻는다. 저자들은 이를 **national alignment(국가 정렬)**이라는 개념으로 정식화한다 — 사회적 가치(social value alignment, SVA)와 공통 지식(common knowledge alignment, CKA) 두 축이다.

**공통 지식(CKA) — 6,000문항.** 한국 교과서와 검정고시(GED) 참고자료를 근거로 만든 객관식 문항이다. 기준 점수는 **0.6** — 한국 검정고시 합격선(60점)을 그대로 가져다 썼다.

**사회적 가치(SVA) — 4,000문항.** 여기가 KorNAT의 진짜 기여다. "정답"을 연구자가 정하지 않고, **한국인 6,174명을 대상으로 한 대규모 설문**에서 실제 응답 분포를 정답으로 삼는다. 모델이 5점 리커트 문항에서 어떤 선택지를 고르면, 그 선택지를 고른 설문 응답자의 비율만큼 정렬 점수를 받는다.

$$
\text{SVA}(m) = \frac{1}{N}\sum_{i=1}^{N} p_i\big(a_m(i)\big)
$$

- $$a_m(i)$$: 모델 $$m$$이 문항 $$i$$에서 고른 선택지
- $$p_i(\cdot)$$: 문항 $$i$$에서 6,174명의 설문 응답자 중 해당 선택지를 고른 비율
- 즉 모델이 "한국 사회의 다수 의견"과 같은 선택지를 고를수록 점수가 올라간다

세부적으로는 5점 척도를 3단계(찬성/중립/반대)로 합친 A-SVA, 다수 의견이 50%를 못 넘기면 중립으로 처리하는 N-SVA까지 세 가지 변형을 함께 보고한다.

**결과 — 정렬에서도 모델 줄세우기가 뒤집힌다.**

| 모델             | SVA       | A-SVA     | N-SVA     | CKA       |
| ---------------- | --------- | --------- | --------- | --------- |
| Llama-2          | 0.252     | 0.315     | 0.370     | 0.322     |
| GPT-3.5-Turbo    | 0.290     | 0.435     | 0.315     | 0.320     |
| GPT-4            | 0.260     | 0.448     | 0.300     | 0.386     |
| Claude-1         | 0.286     | 0.407     | 0.321     | 0.335     |
| **HyperCLOVA X** | 0.253     | 0.318     | **0.414** | **0.707** |
| PaLM-2           | **0.331** | **0.532** | 0.302     | 0.664     |
| Gemini Pro       | 0.303     | 0.513     | 0.312     | 0.639     |

CKA 기준 점수 0.6을 넘긴 모델은 HyperCLOVA X(0.707), PaLM-2(0.664), Gemini Pro(0.639) 셋뿐이다. GPT-4는 0.386으로 GPT-3.5(0.320)보다는 낫지만 기준선에 한참 못 미친다. **글로벌 벤치마크에서 가장 강한 모델(GPT-4)이 한국 공통 지식에서는 기준 미달이고, 한국어 특화 모델(HyperCLOVA X)이 가장 앞선다** — KMMLU에서 본 패턴과 같은 구조다.

여기서 브리프의 핵심 논점이 나온다. **"정렬"의 정답 자체가 문화 의존적이다.** SVA는 애초에 "한국 사회의 다수 의견"을 정답으로 정의한다. 이 설문을 미국인 6,174명으로 바꿔 다시 진행하면 같은 문항에 대해 다른 정답 분포가 나올 것이고, 같은 모델의 SVA 점수도 달라질 것이다. 즉 "이 모델이 잘 정렬되어 있는가"라는 질문은 "어느 나라에 정렬되어 있는가"라는 전제 없이는 답할 수 없는 질문이다. 이 통찰은 뒤에서 다룰 [#25 안전 평가의 통계와 체계 설계](/blog/2026/safety-evaluation-statistics/)로 곧장 이어진다 — 안전(safety) taxonomy도 "무엇이 유해한가"에 대한 사회적 합의를 전제하는 이상, 문화 중립적일 수 없다.

## HRET — 평가 인프라, 재현성을 위한 최소 장치

지금까지 본 벤치마크들을 실제로 돌리려면 각 벤치마크마다 다른 프롬프트 포맷, 다른 정답 추출 로직, 다른 평가 방식을 손으로 맞춰야 한다. [HRET: A Self-Evolving LLM Evaluation Toolkit for Korean](https://arxiv.org/abs/2503.22968) (Lee et al., arXiv 2025)은 이 문제를 정면으로 겨냥한 평가 툴킷이다.

동기가 되는 관찰은 구체적이다 — **동일한 모델을 동일한 벤치마크로 평가해도 기관마다 결과가 1\~10퍼센트포인트씩 벌어진다.** 원인은 추론 설정, 프롬프트 엔지니어링, 평가 구현 차이다. 여기에 한국어 특유의 문제가 하나 더 얹힌다 — 형태소 분석, 띄어쓰기, 경어법 처리 방식이 구현마다 달라 같은 정답도 다르게 채점될 수 있다.

HRET은 레지스트리 기반 구조로 HAE-RAE Bench, KMMLU, CLIcK, KUDGE, K2-Eval, HRM8K 같은 주요 벤치마크와 HuggingFace Transformers, vLLM, OpenAI 호환 엔드포인트, LiteLLM 같은 여러 추론 백엔드를 통합한다. 평가 방식도 문자열 완전일치, 로그우도 기반 채점, LLM-as-judge, 수학 검증, 그리고 **언어 일관성 검사기**(응답이 실제로 한국어인지 감지)까지 포괄한다. 여기에 형태소를 고려한 TTR(Type-Token Ratio, 어휘 다양성 지표)과 핵심어 누락 탐지 같은 한국어 전용 진단까지 얹었다.

[#24](/blog/2026/contamination-reproducibility/)에서 다룰 lm-eval-harness의 재현성 문제와 같은 뿌리지만, 한국어는 형태소·띄어쓰기·경어법이라는 변동 축이 추가로 얹힌다는 점에서 재현성 확보가 한 겹 더 까다롭다.

## 그 외 벤치마크 — 짧게

한국어 평가 생태계에는 이 밖에도 확인된 벤치마크가 여럿 있다.

| 벤치마크                                                        | 규모                                                                  | 특징                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| KoBEST (Jang et al., SK Telecom, COLING 2022)                   | 5개 태스크(BoolQ·COPA·WiC·HellaSwag·SentiNeg)                         | 전문 언어학자가 설계, 전량 사람이 주석                                                                          |
| KorMedMCQA (arXiv 2403.01469)                                   | 7,469문항                                                             | 의사·간호사·약사·치과의사 국가시험(2012\~2024) 기반                                                             |
| KoBALT (arXiv 2505.16125, SNU)                                  | 700문항, 24개 언어 현상                                               | 통사·의미·화용·음운·형태 5개 언어학 도메인. 표준 코퍼스와 n-gram 중복이 낮아 오염에 강함                        |
| HRM8K (Son et al., arXiv 2501.02448)                            | 8,011문항                                                             | 자체 수집 1,428문항(KSM) + 영어 벤치마크(GSM8K·MATH 등) 번역 6,583문항 병존                                     |
| LogicKor (instructkr)                                           | 6개 카테고리 42개 멀티턴 프롬프트                                     | MT-Bench 한국어판 + 한국어 문법 카테고리 추가. LLM-as-judge 채점. 모델 성능이 상향평준화되며 리더보드 갱신 중단 |
| Open Ko-LLM Leaderboard / Ko-H5 (Upstage/NIA, arXiv 2405.20574) | 5개 태스크(Ko-ARC·Ko-HellaSwag·Ko-MMLU·Ko-CommonGen V2·Ko-TruthfulQA) | Ko-ARC/Ko-MMLU/Ko-TruthfulQA는 GPT-4 기계번역 후 사람이 재검수(3개 데이터셋 검수 비용 8만 달러)                 |

여기서 **이름의 함정 하나를 짚고 넘어가야 한다.** Open Ko-LLM Leaderboard가 쓰는 "Ko-MMLU"는 영어 MMLU를 GPT-4로 기계번역한 뒤 사람이 재검수한 **번역 벤치마크**다(14,000문항). 반면 이 글에서 다룬 KMMLU는 한국 시험 원본에서 새로 수집한 **원산 벤치마크**다. 이름이 비슷해 자주 혼동되지만 완전히 다른 두 벤치마크이고, 이 글 전체의 논지 — 번역과 원산은 다른 구성개념을 잰다 — 를 이름 층위에서도 그대로 보여주는 사례다.

## 도메인 특화로의 확장 — 통신 도메인

지금까지 다룬 벤치마크는 모두 "한국어 일반"을 겨냥한다. 그런데 같은 논리는 도메인으로 한 번 더 내려간다 — **일반 벤치마크의 구성개념은 특정 도메인을 대표하지 못한다.** KMMLU가 "영어 지식 벤치마크는 한국을 대표하지 못한다"고 말했다면, 도메인 벤치마크는 "한국어 일반 벤치마크는 통신 도메인을 대표하지 못한다"고 말하는 셈이다. 이 시리즈에는 이미 그 사례가 있다 — [TelBench](/blog/2026/telbench/)와 [TelAgentBench](/blog/2026/telagentbench/)다. 자세한 내용은 해당 글로 넘긴다.

# Experiments

## 벤치마크 한눈에 비교

| 벤치마크      | 문항 수                        | 범주 수             | 수집 방식                   | 재는 구성개념                 | 특징                                          |
| ------------- | ------------------------------ | ------------------- | --------------------------- | ----------------------------- | --------------------------------------------- |
| KMMLU         | 35,030                         | 45                  | 원산 (국가자격·공무원 시험) | 한국 전문 지식 전반           | 사람 평균(62.6%)보다 GPT-4(59.95%)가 낮음     |
| KMMLU-Redux   | 2,587                          | 14(산업)            | 원산 재정제                 | 신뢰도 높인 전문 지식         | 오류 7.66% 제거 + 저변별력 문항 제거          |
| KMMLU-Pro     | 2,822                          | 14(전문직)          | 원산 (전문자격시험)         | 전문직 실무 지식              | 변호사·회계사·의사 등                         |
| HAE-RAE Bench | 1,538                          | 6                   | 원산                        | 한국 어휘·역사·문화 상식      | 번역으로 원천적으로 못 만드는 문항(순화어 등) |
| CLIcK         | 1,995                          | 11(문화 8 + 언어 3) | 원산 (시험·교과서)          | 문화 지능 vs 언어 지능        | 문항별 필요 지식 유형 라벨링                  |
| KorNAT        | 10,000 (SVA 4,000 + CKA 6,000) | 2(가치·지식)        | 원산 (설문 + 교과서)        | 국가 정렬(national alignment) | 정답이 6,174명 설문 응답 분포                 |

## 핵심 반례 — 같은 GPT-4, 세 개의 점수

다시 서두의 숫자로 돌아가자. GPT-4 테크니컬 리포트의 다국어 MMLU 부록(3-shot, Azure Translate)에서 영어는 85.5%, 한국어(번역)는 77.0%다. 반면 KMMLU 논문의 gpt-4-0613(5-shot, Direct) 점수는 59.95%다. 스냅샷 버전(공식 미명시 vs 0613)과 shot 수(3-shot vs 5-shot)가 달라 완벽히 통제된 비교는 아니지만, 방향성은 뚜렷하다.

이 낙차를 하나의 원인으로 설명하면 틀린다. 최소 세 가지 가설로 쪼개야 한다.

**가설 1 — 문화 지식 부재.** KMMLU 문항의 20.4%(1,305문항, KMMLU-KOR)는 한국 문화·사회·법률 지식을 요구한다. 이 문항들은 영어 MMLU에 애초에 존재하지 않으므로 번역으로는 만들 수 없다. 77.0%에서 59.95%로의 낙차 중 상당 부분은 **번역 품질과 무관하게, 측정 대상 자체가 넓어져서** 생긴 것이다. 이건 노이즈가 아니라 신호다 — 진짜로 몰랐던 지식이 드러난 것이기 때문이다.

**가설 2 — 번역투 적응.** 85.5%에서 77.0%로의 낙차는 문항 내용이 그대로인 상태에서 번역만 거친 결과다. GPT-4 리포트 스스로 "번역이 완벽하지 않아 미묘한 정보를 잃을 수 있다"고 인정한다. 이 낙차는 모델의 실제 지식 격차가 아니라 **번역이라는 연산이 주입한 잡음**에 가깝다.

**가설 3 — 토큰화 효율.** GPT-4 리포트는 다국어 평가에서 "일부 언어가 훨씬 긴 토큰 시퀀스로 대응되기 때문에" 5-shot 대신 3-shot을 썼다고 명시한다. 즉 같은 문항이라도 한국어로 번역되는 순간 in-context 예시 개수 자체가 줄어드는 제약이 걸린다. 한국어는 교착어라 조사·어미가 붙을 때마다 형태가 바뀌고, 영어 중심으로 학습된 BPE 토크나이저는 한글 음절을 잘게 쪼개는 경향이 있어 같은 의미를 담는 데 더 많은 토큰이 필요하다. 이는 few-shot 컨텍스트 예산과 생성형 답안 추출 정확도 모두에 불리하게 작용한다.

세 가설을 정리하면: **85.5 → 77.0의 낙차는 주로 가설 2·3(번역이라는 연산 자체의 손실)**이고, **77.0 → 59.95의 낙차는 주로 가설 1(구성개념 자체의 확장)**이다. 앞의 낙차는 "같은 것을 재는데 재는 방법이 나빠서" 생긴 것이고, 뒤의 낙차는 "다른 것을 재기 시작해서" 생긴 것이다. 번역 벤치마크의 점수 하나만 보면 이 둘을 절대 구분할 수 없다.

## 번역 벤치의 통계적 함정 — 구성개념 무관 분산

이걸 [#1](/blog/2026/what-is-evaluation/)의 어휘로 정리하면 이렇다. 번역 벤치마크의 점수는 모델 능력과 번역 품질이 뒤섞인 값이다.

$$
\text{Score}_{\text{translated}} = f(\theta_{\text{model}}) + \varepsilon_{\text{translation}}
$$

- $$\theta_{\text{model}}$$: 우리가 진짜로 재고 싶은 모델의 능력
- $$\varepsilon_{\text{translation}}$$: 번역 품질에서 오는 오차항 — 문항마다 크기와 방향이 다르고, 평균이 0이라는 보장도 없다
- 문제는 $$\varepsilon_{\text{translation}}$$이 무작위 잡음이 아니라 **번역기·검수자·문항 종류에 따라 체계적으로 편향된다**는 점이다. 이러면 여러 모델을 같은 번역 벤치마크로 비교할 때 순위 자체가 왜곡될 수 있다.

이게 바로 [#1](/blog/2026/what-is-evaluation/)이 말한 **구성개념 무관 분산(construct-irrelevant variance)**이다 — 점수의 분산 중 일부가 우리가 재려는 것(모델 능력)이 아니라 측정 도구의 결함(번역 품질)에서 나온다.

Open Ko-LLM Leaderboard의 Ko-MMLU가 좋은 예다. GPT-4로 기계번역하고 규칙 기반 검증을 거친 뒤 전문 번역가가 재검수까지 했다 — Ko-ARC·Ko-MMLU·Ko-TruthfulQA 세 데이터셋에 8만 달러를 투입했다. 이 정도로 공을 들여도 번역이라는 연산 자체의 근본적 한계(가설 2·3)는 완전히 없어지지 않는다. 결국 **번역 품질에 아무리 투자해도, 원산 데이터가 잡아내는 가설 1의 격차(문화 지식 그 자체의 부재)는 번역으로 메울 수 없다.** 이것이 KMMLU 이후 한국어 평가 커뮤니티가 번역에서 원산으로 방향을 튼 이유다.

# Conclusion

한국어 평가는 "영어 벤치마크의 번역본"에서 "원산 벤치마크"로 이동했다. KMMLU가 한국 시험 원본에서 문항을 새로 모으고, HAE-RAE Bench가 번역으로는 원천적으로 만들 수 없는 문항(순화어 등)을 설계하고, CLIcK이 문화와 언어를 두 축으로 갈라놓고, KorNAT이 "정렬"의 정답 자체를 6,174명의 설문으로 정의한 것 — 이 모든 움직임의 정당화는 결국 하나로 수렴한다. **구성 타당도 논증이다.** 번역이 재는 것과 우리가 재고 싶은 것이 다르다는 것을 GPT-4의 85.5% → 77.0% → 59.95% 낙차가, 그리고 GPT-4의 KorNAT CKA 점수(0.386)가 HyperCLOVA X(0.707)에 뒤집히는 장면이 구체적으로 보여준다.

다만 원산으로 옮겨간다고 문제가 다 풀리는 건 아니다. 이 시리즈 뒷부분에서 다룰 세 가지 남은 과제가 있다.

1. **원산 벤치도 오염을 피할 수 없다.** KMMLU-Redux 논문이 밝힌 대로, 공개된 국가시험 문제는 이미 웹에 존재한다. 원산이라고 해서 사전학습 데이터에서 안전한 게 아니다 — [#24 오염·재현성·효율](/blog/2026/contamination-reproducibility/)로 이어지는 문제다.
2. **문항 수가 적어 신뢰구간이 넓다.** HAE-RAE Bench(1,538문항), CLIcK(1,995문항) 같은 벤치마크는 KMMLU보다 훨씬 작다. 점수 몇 퍼센트포인트 차이로 모델 순위를 단정하기 전에 이 구간이 얼마나 넓은지 따져봐야 한다 — [#19 점수는 추정치다](/blog/2026/confidence-intervals/)에서 다룰 이항비율 신뢰구간 문제다.
3. **가치 문항은 정답 자체가 논쟁적이다.** KorNAT의 SVA가 정확히 이 문제를 정면으로 마주한다. "한국 사회의 다수 의견"을 정답으로 삼는 순간, 그 정답은 시간이 지나면 바뀔 수 있고 소수 의견은 구조적으로 낮은 점수를 받는다 — [#25 안전 평가의 통계와 체계 설계](/blog/2026/safety-evaluation-statistics/)가 이 문제를 안전 taxonomy 전반으로 확장해서 다룬다.

다음 글([#16 사람 평가 설계](/blog/2026/human-evaluation-design/))부터는 방향을 튼다. 지금까지는 "무엇을 재는 벤치마크를 만들 것인가"를 물었다면, 이제부터는 "사람이 그 벤치마크를 어떻게 채점할 것인가"를 묻는다.

# 참고 문헌

- Son et al., 2024. [KMMLU: Measuring Massive Multitask Language Understanding in Korean](https://arxiv.org/abs/2402.11548) (NAACL 2025).
- Hong et al., 2025. [From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation](https://arxiv.org/abs/2507.08924).
- Son et al., 2023. [HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models](https://arxiv.org/abs/2309.02706) (LREC-COLING 2024).
- Kim et al., 2024. [CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean](https://arxiv.org/abs/2403.06412) (LREC-COLING 2024).
- Lee et al., 2024. [KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge](https://arxiv.org/abs/2402.13605) (ACL 2024 Findings).
- Lee et al., 2025. [HRET: A Self-Evolving LLM Evaluation Toolkit for Korean](https://arxiv.org/abs/2503.22968).
- OpenAI, 2023. [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — Appendix F, Multilingual MMLU.
- Jang et al., 2022. [KOBEST: Korean Balanced Evaluation of Significant Tasks](https://arxiv.org/abs/2204.04541) (COLING 2022).
- Kweon et al., 2024. [KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations](https://arxiv.org/abs/2403.01469).
- 2025. [KoBALT: Korean Benchmark For Advanced Linguistic Tasks](https://arxiv.org/abs/2505.16125).
- Son et al., 2025. [Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap](https://arxiv.org/abs/2501.02448) — HRM8K.
- [LogicKor GitHub (instructkr)](https://github.com/instructkr/LogicKor).
- 2024. [Open Ko-LLM Leaderboard: Evaluating Large Language Models in Korean with Ko-H5 Benchmark](https://arxiv.org/abs/2405.20574).
- [TelBench](/blog/2026/telbench/) — 이 블로그의 통신 도메인 벤치마크 포스트.
- [TelAgentBench](/blog/2026/telagentbench/) — 이 블로그의 통신 에이전트 벤치마크 포스트.

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 열네 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/everything-benchmark/">범용 벤치마크라는 주장</a> — Raji et al. — 모든 것을 잰다는 말</li>
  <li><a href="/blog/2026/fixing-nlu-benchmarking/">벤치마킹을 고치려면</a> — Bowman & Dahl의 네 기준</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
  <li><a href="/blog/2026/clever-hans-benchmarks/">표층 특징이 정답을 예측한다</a> — Clever Hans, 데이터셋 인공물</li>
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
  <li><strong>(현재 글)</strong> 한국어 벤치마크 — 번역이 아니라 원산, 그리고 문화 타당도</li>
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
