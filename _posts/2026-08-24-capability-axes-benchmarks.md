---
layout: post
title: "능력의 다른 축 — 지시따르기·긴 문맥·사실성"
date: 2026-08-24 09:13:00 +0900
description: "LLM 평가 체계 시리즈 #12 — IFEval·RULER·TruthfulQA·FActScore·SimpleQA로 보는, 정확도 평균이 숨기는 세 가지 실패"
categories: [paper]
tags: [evaluation, instruction-following, long-context, factuality, hallucination, benchmark, paper]
giscus_comments: true
related_posts: true
---

> [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) (Zhou et al., Google, arXiv 2023)

> [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (Hsieh et al., NVIDIA, COLM 2024)

> [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) (Lin et al., Oxford/OpenAI, ACL 2022)

> [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251) (Min et al., University of Washington, EMNLP 2023)

> [Measuring short-form factuality in large language models](https://arxiv.org/abs/2411.04368) (Wei et al., OpenAI, arXiv 2024)

# Introduction

[#9](/blog/2026/knowledge-benchmarks/)은 지식과 추론을, [#10](/blog/2026/math-code-benchmarks/)은 수학·코드처럼 정답이 프로그램으로 검증되는 영역을, [#11](/blog/2026/mt-bench-to-arena/)는 개방형 대화 품질을 다뤘다. 이 세 편을 합치면 "이 모델은 얼마나 똑똑하고, 얼마나 대화를 잘하는가"에는 어느 정도 답할 수 있다. 그런데 실무에서 모델을 배포하기 전에 물어야 할 질문은 따로 있다.

- 사용자가 "400단어 이내로 요약해줘"라고 했을 때 실제로 400단어 이내로 쓰는가?
- 계약서 50페이지를 던져줬을 때, 3페이지째에 있는 조항을 25페이지 뒤에서도 기억하는가?
- 모르는 걸 아는 척 지어내지는 않는가?

이 세 질문은 지식([#9](/blog/2026/knowledge-benchmarks/))도, 수학·코드([#10](/blog/2026/math-code-benchmarks/))도, 대화 선호([#11](/blog/2026/mt-bench-to-arena/))도 아니다. **지시 따르기(instruction following), 긴 문맥(long context), 사실성(factuality)** 이라는 별도의 능력 축이다. 이 글의 핵심 주장은 하나다 — 이 세 축은 **평균 정확도라는 단일 숫자로는 절대 드러나지 않는 실패 양상**을 공유한다는 것이다.

- 지시 따르기는 "무엇을 검증 가능한가"를 좁혀서 이 문제를 우회한다(IFEval).
- 긴 문맥은 "평균은 멀쩡한데 특정 위치·특정 과제에서 무너진다"(RULER).
- 사실성은 "대부분 맞는데 가끔 확신에 차서 틀린다. 그리고 모른다고 말하지 않는다"(TruthfulQA, FActScore, SimpleQA).

세 축 모두 결국 같은 곳으로 수렴한다. 안전하고 신뢰할 수 있는 모델을 말하려면 **평균이 아니라 분포와 꼬리(tail)를 봐야 한다**는 것, 그리고 이 시리즈의 마지막 부인 [#24](/blog/2026/safety-evaluation-statistics/)에서 통계적으로 정식화할 주제다. 이 글은 그 예고편이자, [#1](/blog/2026/what-is-evaluation/)에서 세운 구성개념(construct)·조작화(operationalization) 틀을 세 벤치마크 각각에 적용해보는 사례집이다.

# Background

[#1](/blog/2026/what-is-evaluation/)에서 정리했듯, 벤치마크를 만드는 일은 항상 두 단계를 거친다. 먼저 재고 싶은 추상적 능력(구성개념)을 정한다 — "지시를 잘 따른다", "긴 문맥을 이해한다", "사실만 말한다"처럼. 그다음 그것을 채점 가능한 구체적 과제로 좁힌다(조작화). 문제는 이 좁히는 과정에서 원래 구성개념의 상당 부분이 잘려 나간다는 것이다.

이 글에서 다루는 세 축은 조작화의 방식이 서로 다르다.

1. **지시 따르기**는 구성개념을 아예 **"프로그램으로 검증 가능한 지시"** 로 좁혀버린다. 그 대가로 judge도 사람도 필요 없는 완전 자동 채점을 얻는다.
2. **긴 문맥**은 구성개념을 "정보를 찾아온다(retrieval)"로 좁혔다가, 그것만으로는 부족함을 깨닫고 다시 넓히는 중이다.
3. **사실성**은 구성개념 자체가 너무 커서(모든 발화의 참·거짓) 여러 벤치가 서로 다른 조각만 담당한다 — TruthfulQA는 "흔한 착각", FActScore는 "긴 글의 원자 사실", SimpleQA는 "짧은 사실 질의와 기권".

세 축 모두 [#10](/blog/2026/math-code-benchmarks/)이 다룬 "검증 가능한 도메인"과 비교하면 흥미롭다. 수학·코드는 정답이 하나로 고정되고 프로그램(테스트 케이스, 수치 비교)이 채점한다. 지시 따르기(IFEval)는 이 아이디어를 **개방형 생성 과제에 억지로 이식**한 것이고, 긴 문맥(RULER의 NIAH류)도 마찬가지로 정답 문자열을 정확히 매칭하는 규칙 채점이다. 반면 사실성은 정답이 하나가 아니라 **긴 글 안에 흩어진 여러 개의 참·거짓 명제**이므로, 검색 대조나 judge가 다시 필요해진다. 즉 이 세 축은 "judge 없이도 잴 수 있는가"라는 기준으로 한 줄로 세울 수 있다 — 지시 따르기(완전 규칙) → 긴 문맥(대부분 규칙) → 사실성(규칙+검색+judge 혼합).

# Method

## 축 1: 지시 따르기 — 검증 가능한 것만 잰다

### IFEval — 채점을 프로그램으로 되돌리는 영리한 우회

"이 응답이 지시를 잘 따랐는가?"를 사람이나 judge에게 물으면 항상 같은 문제가 반복된다. "재치있는 톤으로 써줘", "과하게 설명하지 말고 답해줘" 같은 지시는 **무엇을 만족이라 할지 기준 자체가 불명확**하다. IFEval(Zhou et al., 2023)의 발상은 이 논쟁을 정면 돌파하지 않고 **비켜간다**. 애초에 프로그램으로 자동·객관적으로 판정 가능한 지시만 모으는 것이다 — "450~500단어로 써라", "키워드 AI를 최소 3번 언급해라", "전체를 JSON으로 출력해라"처럼.

저자들은 이런 지시를 **검증 가능한 지시(verifiable instructions)** 라 부르고, 정확히 **25종**을 정의해 **9개 그룹**으로 묶었다.

| 그룹                  | 포함된 지시 종류 (개수)                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------- |
| Keywords              | Include Keywords, Keyword Frequency, Forbidden Words, Letter Frequency (4)                  |
| Language              | Response Language (1)                                                                       |
| Length Constraints    | Number Paragraphs, Number Words, Number Sentences, 문단 수+각 문단 첫 단어 지정 (4)         |
| Detectable Content    | Postscript, Number Placeholder (2)                                                          |
| Detectable Format     | Number Bullets, Title, Choose From, 강조 표시 최소 개수, Multiple Sections, JSON Format (6) |
| Combination           | Repeat Prompt, Two Responses (2)                                                            |
| Change Cases          | All Uppercase, All Lowercase, 전대문자 단어 빈도 (3)                                        |
| Start with / End with | End Checker, Quotation (2)                                                                  |
| Punctuation           | No Commas (1)                                                                               |

이 25종 지시를 하나 이상 조합해 총 **541개 프롬프트**를 만들었다. 같은 지시 종류라도 파라미터(예: "450\~500단어" vs "350\~400단어")와 표현("write 450 to 500 words" vs "your response must contain 450 to 500 words")을 다양화해, 모델이 특정 문구 패턴만 외운 게 아닌지도 확인할 수 있게 했다.

**채점: strict vs loose.** 응답 $$resp$$와 지시 $$inst$$가 주어졌을 때, 가장 단순한 채점은 이렇다.

$$
\text{is\_followed}(resp, inst) = \begin{cases} \text{True}, & \text{instruction이 지켜졌으면} \\ \text{False}, & \text{그렇지 않으면} \end{cases}
$$

이걸 그대로 쓴 것이 **strict-accuracy**다. 그런데 여기에 함정이 있다. 지시가 "P.S. I do like the cake로 끝내라"인데 모델이 마크다운 볼드를 써서 "P.S. **I do like the cake**"로 끝내면, 순수 문자열 매칭은 이걸 실패로 잘못 판정한다(거짓 음성). 이런 오탐을 줄이려고 **loose-accuracy**를 추가로 정의한다.

$$
\text{is\_followed}_{\text{loose}}(resp, inst) = \text{Any}\big(\text{is\_followed}(\text{transform}_t(resp), inst)\ \text{for}\ t = 1, 2, \ldots \big)
$$

여기서 $$\text{transform}_t$$는 응답을 살짝 변형하는 함수다. 논문은 세 가지 기본 변형 — (1) 마크다운 기호(`*`, `**`) 제거, (2) 첫 줄 제거(모델이 흔히 붙이는 "네, 알겠습니다" 같은 서두 제거), (3) 마지막 줄 제거(마무리 인사 제거) — 을 정의하고, 이 셋을 단독·둘씩 조합·셋 다 조합·아무것도 적용 안 함(항등변환)까지 합쳐 **총 8가지 변형** 중 하나라도 통과하면 loose-accuracy에서는 성공으로 센다. 대신 이 관대함은 거짓 양성을 늘릴 수 있어, 논문 스스로도 loose를 strict의 **보완 지표**로만 쓰라고 못 박는다.

최종적으로 4가지 점수를 낸다 — 프롬프트 단위 strict, 지시 단위 strict, 프롬프트 단위 loose, 지시 단위 loose. 논문이 베이스라인으로 평가한 두 모델의 결과다.

| 모델         | 프롬프트-strict | 지시-strict | 프롬프트-loose | 지시-loose |
| ------------ | --------------- | ----------- | -------------- | ---------- |
| GPT-4 (2023) | 76.89%          | 83.57%      | 79.30%         | 85.37%     |
| PaLM 2 S     | 43.07%          | 55.76%      | 46.95%         | 59.11%     |

여기서 프롬프트 단위 점수가 지시 단위 점수보다 항상 낮다는 점이 흥미롭다. 한 프롬프트에 지시가 3개 있으면 3개를 **전부** 지켜야 그 프롬프트가 성공으로 카운트되기 때문이다(AND 조건). 지시 하나하나는 80%대로 잘 지키는 모델도, 프롬프트 전체 성공률은 그보다 눈에 띄게 낮아진다 — 여러 제약을 동시에 만족시키는 일이 각 제약을 따로 만족시키는 일보다 항상 어렵다는, 당연하지만 숫자로 확인하면 새삼스러운 사실이다.

**한계 — 구성개념 부족.** IFEval의 영리함은 동시에 가장 큰 한계다. "지시를 잘 따른다"는 구성개념의 대부분은 애초에 **검증이 불가능하다**. 논문 자신도 서론에서 "재치있는 톤으로 써라", "상세히 추론하되 과하게 설명하지 마라" 같은 지시는 "기준 자체가 매우 불분명하다(greatly unclear)"고 인정한다. 즉 IFEval이 재는 것은 지시 따르기 능력 전체가 아니라 **그중 프로그램으로 걸러낼 수 있는 좁은 부분집합**이다. 의도 파악, 암묵적 제약(맥락상 당연히 지켜야 할 것들), 사용자의 숨은 의도를 읽는 능력은 이 벤치의 사각지대에 그대로 남는다. [#10](/blog/2026/math-code-benchmarks/)에서 수학·코드가 "판정 프로그램이 있어 논쟁의 여지가 없다"는 장점을 가졌던 것과 같은 구조인데, 그 대가로 도메인 자체가 좁아진 것도 똑같다.

### FollowBench와 InFoBench — 검증 불가능한 지시로 확장

IFEval이 남긴 사각지대를 메우려는 후속 벤치 두 개만 짧게 짚는다.

- **FollowBench**(Jiang et al., ACL 2024, arXiv 2310.20410)는 Content·Situation·Style·Format·Example 5종의 **세밀한 제약(fine-grained constraints)** 을 다루고, 한 지시에 제약을 한 단계씩 누적해 붙이는 **다단계(multi-level)** 설계로 "몇 단계에서 무너지는가"를 짚어낸다. 채점은 강한 LLM을 judge로 쓴다 — IFEval의 프로그램 판정을 포기하는 대신 검증 불가능한 스타일·상황 제약까지 다룰 수 있게 됐다.
- **InFoBench**(Qin et al., ACL 2024 Findings, arXiv 2401.03601)는 복잡한 지시 하나를 여러 개의 단순한 예/아니오 질문으로 쪼개는 **DRFR(Decomposed Requirements Following Ratio)** 지표를 제안한다. 500개 지시를 2,250개의 분해된 질문으로 쪼갰고, GPT-4를 채점자로 써도 사람과 신뢰도가 크게 다르지 않음을 보였다.

두 벤치 모두 judge를 다시 불러들였다는 공통점이 있다. IFEval이 얻은 "judge 없는 완전 자동 채점"이라는 장점을 포기하는 대가로 구성개념을 넓힌 것이다.

## 축 2: 긴 문맥 — "주장하는 길이"와 "쓸 수 있는 길이"

### Needle-in-a-Haystack — 왜 이것만으로는 부족한가

긴 문맥 평가의 원형은 단순하다. 긴 텍스트(건초더미, haystack) 어딘가에 관련 없는 문장 하나("샌프란시스코에서 가장 좋은 일은 화창한 날 공원에서 샌드위치를 먹는 것이다" 같은)를 심어 놓고, 그 문장에 대한 질문을 던져 모델이 찾아내는지 본다. Greg Kamradt가 2023년에 만든 이 비형식적 테스트가 지금도 널리 인용된다.

문제는 이 과제가 **너무 쉽다**는 것이다. 요구되는 능력은 사실상 **정확 일치 검색(exact-match retrieval)** 뿐이다. 관련 없는 문장 하나가 도드라지게 삽입돼 있으면, attention 메커니즘 입장에서는 이 문장이 통계적으로 눈에 띄기 때문에 찾기가 상대적으로 쉽다. 실제로 최근 모델 대부분이 이 vanilla NIAH에서 거의 만점(90%대 후반)을 받는다. 그런데 그 만점이 "이 모델이 긴 문맥을 이해한다"를 뜻하지는 않는다. 긴 문서 전체에 흩어진 여러 단서를 **엮어서 추론**하거나, 문맥 전체에서 **정보를 집계**하거나, 여러 홉을 거쳐 **개체를 추적**하는 일은 전혀 다른 능력이고, NIAH는 이걸 전혀 재지 못한다.

### RULER — 주장하는 문맥 길이는 스펙이지 능력이 아니다

RULER(Hsieh et al., NVIDIA, COLM 2024)는 이 간극을 메우려고 NIAH를 확장하고, 검색이 아닌 새 과제 범주 두 개를 추가한다. 총 **4개 과제 범주**다.

| 범주               | 과제                                                                                                                 | 무엇을 테스트하나                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Retrieval          | S-NIAH(단일 needle), MK-NIAH(다중 키, 방해 정보 다수), MV-NIAH(같은 키에 여러 값), MQ-NIAH(여러 개의 키를 동시 질의) | 정보를 정확히 찾아오는가                                                  |
| Multi-hop Tracing  | Variable Tracking (VT)                                                                                               | 변수 대입 체인을 따라가며 개체를 추적하는가(공지시 해소의 최소 대리 과제) |
| Aggregation        | Common Words Extraction (CWE), Frequent Words Extraction (FWE)                                                       | 문맥 전역에서 빈출 정보를 집계하는가(요약의 대리 과제)                    |
| Question Answering | QA (SQuAD·HotpotQA를 방해 문단으로 늘림)                                                                             | 방해 정보 속에서 실제 질의응답을 하는가                                   |

이 4개 범주에서 파생된 변형(needle 종류·개수, 체인 길이, 방해 요소 강도 등)을 조합해 **총 13개 세부 과제**를 만들고, 17개 모델을 4K/8K/16K/32K/64K/128K 여섯 길이에서 평가했다.

핵심 설계는 **효과 문맥 길이(effective context length)** 다. 기준선을 Llama2-7B가 4K에서 낸 평균 점수(85.6%)로 잡고, 13개 과제 평균 점수가 이 기준선을 넘는 **가장 긴** 길이를 그 모델의 효과 길이로 정의한다. "주장하는 길이(claimed length)"는 모델 카드에 적힌 스펙이고, "효과 길이"는 실제로 그 스펙만큼 성능을 유지하는 길이다. 둘을 나란히 두면 이렇다.

| 모델                    | 주장 길이 | 효과 길이 |
| ----------------------- | --------- | --------- |
| Gemini-1.5-Pro          | 1M        | >128K     |
| GPT-4                   | 128K      | 64K       |
| Llama3.1 (70B)          | 128K      | 64K       |
| GLM4 (9B)               | 1M        | 64K       |
| Qwen2 (72B)             | 128K      | 32K       |
| Command-R-plus (104B)   | 128K      | 32K       |
| Llama3.1 (8B)           | 128K      | 32K       |
| Yi (34B)                | 200K      | 32K       |
| Mistral-v0.2 (7B)       | 32K       | 16K       |
| GradientAI/Llama3 (70B) | 1M        | 16K       |
| DBRX (36B/132B)         | 32K       | 8K        |
| LongChat (7B)           | 32K       | <4K       |

이 표 하나가 [#1](/blog/2026/what-is-evaluation/)의 구성타당도 실패를 가장 선명하게 보여준다. "128K 문맥"이라는 스펙 숫자가 실제 능력을 대표하지 못한다 — GPT-4는 절반(64K)에서 이미 성능이 무너지고, Yi-34B는 주장한 200K 중 32K(16%)만 쓸 수 있으며, LongChat-7B 같은 일부 모델은 **주장한 32K는커녕 RULER가 평가한 가장 짧은 길이인 4K에서도 이미 기준선 밑으로 떨어진다.** 논문 본문의 표현을 그대로 옮기면, 32K 이상을 주장하는 모델들 중에서도 **절반만이 실제로 32K에서 만족스러운 성능을 유지한다.** 자동차 카탈로그가 "8인승"이라고 광고하지만 실제로 뒷좌석 세 명이 다 타면 무릎이 앞좌석에 닿는 것과 같다 — 정원(定員) 스펙과 실사용 정원이 다른 것이다.

한 가지 더: 이 표는 큰 모델이 항상 이긴다는 뜻도 아니다. 학습 시점부터 128K로 브루트포스 학습한 Llama3.1과, 32K로 학습한 뒤 추론 시점에 길이를 외삽한 Qwen2가 나란히 최상위권에 있다. 반면 GradientAI/Llama3처럼 1M까지 문맥을 늘려 학습했지만 효과 길이는 16K에 불과한 경우도 있다 — **학습 문맥 길이를 늘리는 것이 곧 실사용 능력 향상을 보장하지 않는다**는 뜻이다.

### LongBench·∞Bench·HELMET — 더 현실적인 과제로

RULER는 순수 합성 과제라 판정이 쉬운 대신 실제 응용과의 상관을 검증하기 어렵다는 한계를 스스로 인정한다. 세 후속 벤치가 이 간극의 다른 조각을 메운다.

- **LongBench**(Bai et al., 2023, arXiv 2308.14508)는 영어·중국어 이중언어로 **21개 데이터셋, 6개 과제 범주**(단일/다중 문서 QA, 요약, few-shot 학습, 합성 과제, 코드 완성)를 묶은, 실제 텍스트 기반 벤치다. 영어 평균 길이 약 6,711단어, 중국어 약 13,386자로 RULER보다 훨씬 "현실적"이지만 그만큼 판정이 사람·judge에 다시 의존한다.
- **∞Bench**(Zhang et al., ACL 2024, arXiv 2402.13718)는 100K 토큰을 넘는 초장문에 초점을 맞춰 **12개 과제, 5개 도메인**(검색·코드·수학·소설·대화)을 다룬다. 기존 벤치의 10배에 달하는 길이를 요구한다.
- **HELMET**(Yen et al., Princeton, arXiv 2410.02694)은 **7개 응용 중심 범주**로 128K까지 통제된 길이를 평가하며, 59개 모델을 비교한 결과 **NIAH 점수가 실제 응용 성능을 신뢰성 있게 예측하지 못한다**는 것, 그리고 오픈소스와 폐쇄형 모델의 격차가 문맥이 길어질수록 벌어진다는 것을 보였다. RULER의 발견("주장 길이 ≠ 효과 길이")을 한 단계 더 밀어붙여 "합성 과제 성능 ≠ 실제 응용 성능"까지 확장한 셈이다.

## 축 3: 사실성과 환각 — 맞았다는 것과 확신한다는 것

### TruthfulQA — 사람의 착각을 그대로 흉내내는지 본다

TruthfulQA(Lin, Hilton & Evans, ACL 2022)의 설계는 다른 사실성 벤치와 결이 다르다. 무작위로 사실 질문을 모은 게 아니라, **사람들이 흔히 잘못 알고 있는 것**을 일부러 골랐다 — 건강·법률·금융·정치·음모론·픽션 등 **38개 범주에 걸친 817개 질문**이다. "손가락 관절을 꺾으면 관절염에 걸리나요?"처럼, 답이 널리 퍼진 통념과 반대인 질문들이다. 이런 질문을 만든 이유는 명확하다 — 언어모델은 인터넷 텍스트를 학습하는데, 통념 자체가 인터넷에 널리 퍼져 있다면 모델이 사실이 아니라 **통념을 모방(imitate)** 하도록 학습될 위험이 크다. 저자들은 이를 **모방적 거짓(imitative falsehood)** 이라 부른다.

가장 놀라운 결과는 **역방향 스케일링(inverse scaling)** 이다. GPT-3, GPT-Neo/J, GPT-2, UnifiedQA 네 계열 모두에서, **같은 계열 안에서 모델이 커질수록 TruthfulQA 점수가 대체로 낮아졌다.** 다른 거의 모든 NLP 과제에서 스케일이 커지면 성능이 오르는 것과 정반대다. 대조군으로 함께 측정한 "통제 삼시 문제(control trivia questions, 통념과 무관한 평범한 사실 질문)"에서는 반대로 **큰 모델이 더 잘했다** — 즉 큰 모델이 무능해진 게 아니라, **통념을 더 그럴듯하게 재현하는 능력이 늘어난 것**이 원인이라는 뜻이다. 가장 성능이 좋았던 모델(GPT-3-175B, "helpful" 프롬프트)조차 진실한 답변 비율이 **58%** 였고, 사람 기준선은 **94%** 였다. 더 나쁜 신호는 따로 있다 — 이 최상위 모델은 "거짓이면서 그럴듯하게 정보를 담은" 답변을 **42%** 의 비율로 냈는데, 사람은 이 비율이 **6%** 에 불과했다. 정보가 많아 보이는 거짓 답변일수록 사람을 속이기 쉽다는 점에서 이 42%가 특히 위험하다.

자동 채점도 흥미롭다. 저자들은 사람이 매긴 참/거짓 라벨로 GPT-3를 파인튜닝해, 처음 본 모델의 답변에도 **90~96%의 정확도**로 참/거짓을 예측하는 분류기를 만들었다 — 지금 시리즈에서 반복적으로 등장할 "judge를 학습시켜 채점을 대신한다"는 아이디어의 초기 사례([#22](/blog/2026/judge-statistics/)에서 이 계보를 정식으로 다룬다)다.

**중요한 한계 — 과대해석 주의.** TruthfulQA의 점수를 "이 모델은 일반적으로 얼마나 사실적인가"로 읽으면 안 된다. 애초에 이 벤치의 817문항은 **무작위 표본이 아니라, 사람이 흔히 틀리는 지점만 적대적으로 골라낸 부분집합**이다. 즉 조작화 단계에서 이미 "쉬운 사실 질문"을 걸러내고 "함정 질문"만 남긴 것이다. 이런 표본으로 낮은 점수가 나왔다고 "이 모델은 사실에 약하다"고 일반화하면, 표본 선택 편향을 무시한 과대해석이 된다. TruthfulQA는 "특정 종류의 함정에 얼마나 잘 안 걸리는가"를 재는 것이지, 사실성 전반의 대푯값이 아니다.

### FActScore — 원자 사실 단위로 쪼개서 대조한다

TruthfulQA가 짧은 답변의 참/거짓을 보는 반면, FActScore(Min et al., EMNLP 2023)는 **긴 글**의 사실성을 잰다. 문제는 긴 글은 문장 하나에도 참인 정보와 거짓인 정보가 섞여 있다는 것이다 — 논문은 ChatGPT가 생성한 문장의 **40%가 참·거짓이 섞인 혼합 문장**이라고 보고한다. 그러니 문장 단위, 심지어 문단 단위로 "참/거짓"을 매기는 이분법 채점은 애초에 정보 손실이 크다.

FActScore의 해법은 생성문을 **원자 사실(atomic fact)** — 한 문장에 정보 하나만 담은 최소 단위 — 로 쪼갠 뒤, 각각을 신뢰할 수 있는 지식원(위키피디아)에 대조하는 것이다. 모델 $$\mathcal{M}$$이 프롬프트 $$x$$에 응답 $$y$$를 냈고 그 원자 사실 집합이 $$A_y$$일 때,

$$
f(y) = \frac{1}{|A_y|} \sum_{a \in A_y} \mathbb{I}[a\ \text{가}\ \mathcal{C}\ \text{에 의해 지지됨}]
$$

$$
\text{FActScore}(\mathcal{M}) = \mathbb{E}_{x \sim \mathcal{X}}\big[ f(\mathcal{M}_x) \mid \mathcal{M}_x\ \text{가 응답함} \big]
$$

$$f(y)$$는 응답 하나의 원자 사실 중 지지되는 비율, $$\text{FActScore}$$는 그걸 프롬프트 전체에 대해 평균한 값이다. 마지막 조건 "$$\mathcal{M}_x$$가 응답함"에 주의해야 한다 — **FActScore는 정밀도(precision)만 재고 재현율(recall)은 재지 않는다.** 즉 짧고 안전한 문장만 생성해 사실을 적게 담아도 점수는 높게 나올 수 있다. 저자들도 이를 한계로 명시한다.

사람 손으로 세 상용 모델의 인물 전기 생성문을 평가한 결과다.

| 모델                     | FActScore |
| ------------------------ | --------- |
| InstructGPT              | 42%       |
| ChatGPT                  | 58%       |
| PerplexityAI (검색 결합) | 71%       |

세 모델 모두 "완벽과는 거리가 멀다"는 게 핵심 메시지다. 더 흥미로운 발견은 **개체의 희귀도에 따른 붕괴**다 — 같은 ChatGPT라도 잘 알려진 인물(위키피디아 조회수 상위)에서는 FActScore 약 80%가 나오지만, 희귀한 인물로 갈수록 **16%까지 떨어진다.** "평균 FActScore 58%"라는 숫자 하나만 보면 이 80%→16%의 붕괴는 완전히 가려진다.

사람 평가는 정확하지만 비싸다. 저자들은 검색과 강한 언어모델을 결합한 **자동 추정기**를 만들어, 사람 평가 대비 **오차율 2% 미만**으로 FActScore를 근사할 수 있음을 보였다. 이 추정기 덕분에 13개 최신 모델의 생성문 **6,500건**을 평가했는데, 같은 규모를 사람이 했다면 **\$26,000**가 들었을 것이라 추산한다. 자동 추정기가 정확도를 크게 희생하지 않으면서 비용을 극적으로 낮춘 사례다.

### SimpleQA — 모르면 모른다고 하는가

SimpleQA(Wei et al., OpenAI, 2024)는 다시 짧은 답으로 돌아온다. 다만 설계 목표가 분명하다 — **채점하기 쉽게** 만드는 것이다. 질문마다 **단 하나의 반박 불가능한 정답**만 있도록 설계했고(예: "오바마 부부가 어디서 만났나"처럼 여러 답이 가능한 질문은 "어느 도시"로 범위를 좁힌다), 정답이 시간이 지나도 바뀌지 않도록 골랐다. 총 **4,326개** 질문이며, GPT-4 계열 네 응답 중 최소 하나는 틀리게 만들도록 적대적으로 수집해 최신 모델에도 도전적이도록 했다.

채점은 프롬프트된 ChatGPT 분류기가 **correct / incorrect / not attempted** 세 등급으로 나눈다. 정답을 완전히 포함하고 모순이 없으면 correct, 조금이라도 모순되면(헤지해서 말해도) incorrect, 정답을 제시하지 않고 회피하면 not attempted다. 사람이 100개씩 재검토한 결과 자동 채점기와의 불일치는 단 2건뿐이었다.

여기서 SimpleQA가 이 시리즈에 중요한 이유가 나온다 — **정답률(overall correct)만으로는 절대 부족하다**는 것을 논문 스스로 설계에 반영했다. "correct given attempted"(시도한 것 중 정답률, 정밀도에 가깝다)를 별도로 계산하고, 이 둘의 조화평균으로 F-score를 만든다. 프론티어 모델들의 결과다.

| 모델              | 정답  | 미시도(기권) | 오답  | 시도 중 정답률 | F-score |
| ----------------- | ----- | ------------ | ----- | -------------- | ------- |
| Claude-3-haiku    | 5.1%  | 75.3%        | 19.6% | 20.6%          | 8.2     |
| Claude-3-sonnet   | 5.7%  | 75.0%        | 19.3% | 22.9%          | 9.2     |
| Claude-3-opus     | 23.5% | 39.6%        | 36.9% | 38.8%          | 29.3    |
| Claude-3.5-sonnet | 28.9% | 35.0%        | 36.1% | 44.5%          | 35.0    |
| GPT-4o-mini       | 8.6%  | 0.9%         | 90.5% | 8.7%           | 8.6     |
| GPT-4o            | 38.2% | 1.0%         | 60.8% | 38.6%          | 38.4    |
| o1-mini           | 8.1%  | 28.5%        | 63.4% | 11.3%          | 9.4     |
| o1-preview        | 42.7% | 9.2%         | 48.1% | 47.0%          | 44.8    |

가장 먼저 눈에 띄는 숫자는 **o1-preview의 정답률이 42.7%에 불과하다**는 것이다. 프론티어라 불리는 모델이 짧고 명확한 사실 질문에서 절반도 못 맞힌다 — 저자들이 이 벤치를 "프론티어 모델에게도 도전적"이라 부른 이유다.

두 번째로, 더 중요한 관찰이 있다. **GPT-4o는 거의 기권하지 않는다(1.0%)** 반면 **Claude-3-sonnet은 4분의 3 이상을 기권한다(75.0%).** GPT-4o-mini는 기권을 0.9%만 하고 나머지를 거의 다 시도하는데, 그중 90.5%가 틀린다 — 이건 "모른다"고 말할 자리에서 계속 찍는다는 뜻이다. 반대로 Claude-3-sonnet은 모르면 대체로 기권하지만, 그 결과 raw 정답률(5.7%)은 낮아 보인다. **정답률 하나만 보면 Claude-3-sonnet이 GPT-4o-mini보다 무능해 보이지만, 실제로는 기권 전략이 다를 뿐이다.** correct-given-attempted(시도한 것 중 정답률)를 보면 Claude-3-sonnet(22.9%)이 GPT-4o-mini(8.7%)보다 오히려 낫다. F-score가 이 둘을 절충한 값이다.

### HaluEval — 환각을 인식할 수 있는가

HaluEval(Li et al., EMNLP 2023, arXiv 2305.11747)은 방향이 조금 다르다. 모델이 **직접 생성**할 때의 사실성이 아니라, **환각이 섞인 텍스트를 주고 그걸 알아채는지** 본다. 사람이 라벨링한 ChatGPT 응답 5,000건과, QA·지식 기반 대화·요약 세 과제에 대해 자동 생성한 30,000건, 총 **35,000개 샘플**로 구성된다. "생성"과 "탐지"는 다른 능력이라, 이 벤치는 사실성 축의 또 다른 조각 — **모델이 자기 자신의(혹은 남의) 환각을 스스로 식별할 수 있는가** — 를 담당한다.

# Experiments

## 세 축을 한 표에

세 축을 채점 방식과 사각지대 기준으로 정리하면 이렇다.

| 축          | 대표 벤치            | 채점 방식                             | 무엇을 못 잡나                                                       |
| ----------- | -------------------- | ------------------------------------- | -------------------------------------------------------------------- |
| 지시 따르기 | IFEval               | 규칙(프로그램 검증)                   | 검증 불가능한 지시 대부분(의도 파악, 암묵적 제약, 톤)                |
| 긴 문맥     | RULER                | 규칙(정답 문자열 매칭)                | 실제 응용에서 필요한 이해·추론(HELMET이 지적), 위치별 세부 붕괴 지점 |
| 사실성      | FActScore / SimpleQA | 검색 대조 + judge(자동 추정기·분류기) | 재현율(누락된 사실), 기권을 반영하지 않으면 찍기가 유리해짐          |

세 축이 공유하는 패턴은 이렇다 — **채점을 규칙으로 자동화할수록 구성개념이 좁아지고, 구성개념을 넓힐수록 다시 judge가 필요해진다.** IFEval → RULER → FActScore/SimpleQA로 갈수록 이 트레이드오프가 뚜렷해진다.

## 평균이 숨기는 것: 위치와 꼬리

세 축 모두 "평균 정확도"라는 단일 숫자를 내면 그 안에 있는 실패가 사라진다는 공통점이 있다.

- **긴 문맥**: RULER에서 GPT-4는 128K까지 어느 정도 점수를 유지하지만, 유효 길이는 64K다. "128K에서 평균 66.6%"라는 숫자 하나만 보면 "그럭저럭 쓸 만하다"로 읽히지만, 실제로는 특정 과제(변수 추적, 다중 값 검색)에서 성능이 급격히 무너지는 구간이 있고, 그 구간이 평균에 희석돼 안 보인다. Yi-34B 분석에서 저자들은 방해 요소(distractor)가 늘어날수록 256K에서 약 40점이나 떨어진다는 것도 보였다 — "평균은 완만한 하락"이지만 "특정 조건에서는 절벽"인 것이다.
- **사실성**: FActScore에서 ChatGPT의 "평균 58%"는 유명 인물(약 80%)과 희귀 인물(16%)을 뭉갠 값이다. 실무에서 위험한 것은 평균이 아니라 **꼬리** — 희귀하고 확인하기 어려운 질의에서 모델이 얼마나 자신 있게 틀리는가 — 다. TruthfulQA의 "정보성 있는 거짓 답변 42%"도 마찬가지다. 이런 답변은 흔한 오답보다 사람을 속이기 쉬운, 분포의 위험한 꼬리에 속한다.

이것이 [#24](/blog/2026/safety-evaluation-statistics/)에서 본격적으로 다룰 주제의 예고다 — 안전·신뢰성 평가는 평균값 하나가 아니라 **분포 전체, 특히 최악의 꼬리**를 봐야 한다. 희귀사건(rare event) 추정, 분위수 보고, calibration 같은 도구가 그래서 필요해진다.

## 기권을 지표에 넣으면 무슨 일이 생기는가

SimpleQA의 correct/incorrect/not-attempted 3분류는 겉보기엔 단순한 채점 디테일이지만, 사실은 **평가 지표 설계가 모델 행동에 어떤 유인을 만드는가**라는 근본적 질문을 담고 있다. 이걸 직접 유도해보자.

정답이면 $$+1$$, 오답이면 $$0$$, 기권도 $$0$$을 주는 가장 단순한 채점(=SimpleQA의 "overall correct")을 생각하자. 모델이 스스로 "내가 맞을 확률"을 $$p$$로 믿는다고 하면,

$$
\mathbb{E}[\text{시도}] = p \cdot 1 + (1-p) \cdot 0 = p, \qquad \mathbb{E}[\text{기권}] = 0
$$

$$p > 0$$이기만 하면 시도의 기댓값이 기권의 기댓값보다 항상 크다. 즉 **확신이 1%밖에 없어도 찍는 것이 언제나 이득**이다. 이 채점 방식 아래에서는 "모른다"고 말하는 모델이 구조적으로 손해를 본다 — **정답률만 보상하는 지표는 환각을 장려한다.**

이 문제를 고치려면 오답에 **벌점**을 줘야 한다. 정답 $$+1$$, 기권 $$0$$, 오답 $$-p_{\text{penalty}}$$로 바꾸면,

$$
\mathbb{E}[\text{시도}] = p \cdot 1 + (1-p)\cdot(-p_{\text{penalty}}) = p(1 + p_{\text{penalty}}) - p_{\text{penalty}}
$$

이 값이 기권의 기댓값 $$0$$과 같아지는 지점을 구하면,

$$
p^{*} = \frac{p_{\text{penalty}}}{1 + p_{\text{penalty}}}
$$

즉 $$p > p^{*}$$일 때만 시도하는 것이 합리적이다. SimpleQA 논문은 벌점 $$p_{\text{penalty}} = 9$$인 예시를 든다. 위 식에 대입하면 $$p^{*} = 9/10 = 0.9$$ — 정확히 논문이 서술한 "적어도 90% 확신이 있어야 시도하는 게 이득"이라는 결과와 일치한다. 벌점을 키울수록 $$p^{*}$$가 1에 가까워져, 모델은 확실할 때만 답하도록 유도된다.

이게 바로 **선택적 예측(selective prediction)** 의 핵심 아이디어다 — 정답률 하나가 아니라 "얼마나 커버(coverage)하면서 얼마나 정확(risk)한가"를 함께 보는 것. [#24](/blog/2026/safety-evaluation-statistics/)에서 risk-coverage 곡선과 AURC(Area Under the Risk-Coverage curve)로 이를 정식화할 것이다. SimpleQA의 세 등급 채점, FActScore가 재현율을 일부러 재지 않은 선택, TruthfulQA가 "정보성 있는 거짓"을 따로 집계한 것 모두 결국 **"모른다고 말하는 것에 어떻게 보상을 매길 것인가"** 라는 같은 질문의 다른 답이다.

# 통계 요약

이 글에서 나온 채점 장치를 표로 모으면 다음과 같다.

| 장치                    | 정의                                                                                                        | 어디서 쓰였나                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------ |
| strict/loose accuracy   | $$\text{Any}(\text{transform}_t(resp)\ \text{판정})$$ — 응답 변형 중 하나라도 통과하면 성공                 | IFEval                         |
| 효과 문맥 길이          | 기준선(예: 4K 성능)을 넘는 최대 길이                                                                        | RULER                          |
| FActScore               | $$\frac{1}{\lvert A_y \rvert}\sum_{a \in A_y} \mathbb{I}[a\ \text{지지됨}]$$의 프롬프트 평균, 정밀도만 측정 | FActScore                      |
| correct given attempted | 시도한 것 중 정답 비율, precision에 대응                                                                    | SimpleQA                       |
| 기권 임계값 $$p^{*}$$   | $$p_{\text{penalty}}/(1+p_{\text{penalty}})$$, 오답 벌점이 클수록 1에 수렴                                  | SimpleQA, selective prediction |

# Conclusion

이 글의 메시지를 한 줄로 요약하면 — **지시 따르기·긴 문맥·사실성은 "정확도가 몇 점인가"로 요약할 수 없는, 서로 다른 방식으로 평균을 배신하는 세 축이다.** IFEval은 애초에 검증 가능한 부분집합만 재는 대신 완전 자동 채점을 얻었고, RULER는 "주장하는 길이"와 "실효 길이" 사이의 간극을 드러내 스펙 숫자가 능력을 대표하지 못함을 보였으며, TruthfulQA·FActScore·SimpleQA는 각각 통념·긴 글·짧은 사실 질의라는 다른 조각에서 "대부분 맞지만 가끔 확신에 차서 틀린다"는 패턴을 확인했다.

한계도 분명하다. 세 축은 서로 겹치지 않는 완전한 커버리지를 이루지 못한다 — 긴 문맥 안에서의 지시 따르기, 사실성과 지시 따르기가 충돌하는 상황(예: "간결하게 답하라"는 지시와 "출처를 다 밝혀라"는 사실성 요구) 같은 조합적 실패는 이 벤치들 어디에도 없다. 그리고 이 시리즈가 [#9](/blog/2026/knowledge-benchmarks/)부터 반복해온 경고가 여기서도 반복된다 — 벤치 하나의 점수를 그 벤치가 실제로 조작화한 좁은 구성개념 이상으로 확대 해석하지 말 것.

다음 편 [#13](/blog/2026/korean-benchmarks/)은 이 지형도를 한국어로 옮긴다 — 번역이 아니라 원산 벤치마크를 만들 때 구성개념이 어떻게 다시 흔들리는지 볼 것이다.

# 참고 문헌

- Zhou et al., 2023. [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) (IFEval).
- Jiang et al., 2024. [FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models](https://arxiv.org/abs/2310.20410) (ACL 2024).
- Qin et al., 2024. [InFoBench: Evaluating Instruction Following Ability in Large Language Models](https://arxiv.org/abs/2401.03601) (ACL 2024 Findings).
- Hsieh et al., 2024. [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (COLM 2024).
- Bai et al., 2023. [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508).
- Zhang et al., 2024. [∞Bench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718) (ACL 2024).
- Yen et al., 2024. [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/abs/2410.02694).
- Lin, Hilton & Evans, 2022. [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) (ACL 2022).
- Min et al., 2023. [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251) (EMNLP 2023).
- Wei et al., 2024. [Measuring short-form factuality in large language models](https://arxiv.org/abs/2411.04368) (SimpleQA, OpenAI).
- Li et al., 2023. [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747) (EMNLP 2023).

---

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 열두 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/everything-benchmark/">범용 벤치마크라는 주장</a> — Raji et al. — 모든 것을 잰다는 말</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
  <li><a href="/blog/2026/clever-hans-benchmarks/">표층 특징이 정답을 예측한다</a> — Clever Hans, 데이터셋 인공물</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="5">
  <li><a href="/blog/2026/measurement-scales/">1~5점 평가는 평균내도 되는가</a> — 척도와 허용 연산</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><a href="/blog/2026/generation-metrics/">생성 지표와 그 타당도</a> — BLEU에서 COMET까지</li>
  <li><a href="/blog/2026/mcqa-fragility/">객관식 평가는 왜 흔들리나</a> — 위치 편향과 포맷 민감도</li>
</ol>

**3부. LLM 벤치마크 지형도**

<ol start="9">
  <li><a href="/blog/2026/knowledge-benchmarks/">지식과 추론 — MMLU 계열의 흥망</a> — MMLU·GPQA·BBH</li>
  <li><a href="/blog/2026/math-code-benchmarks/">검증 가능한 도메인 — 수학과 코드</a> — GSM8K·MATH·HumanEval·SWE-bench</li>
  <li><a href="/blog/2026/mt-bench-to-arena/">개방형 대화 — MT-Bench에서 Arena까지</a> — judge 기반 벤치의 등장</li>
  <li><strong>(현재 글)</strong> 능력의 다른 축 — 지시따르기·긴 문맥·사실성</li>
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
