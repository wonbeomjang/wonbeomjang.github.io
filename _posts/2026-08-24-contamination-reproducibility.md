---
layout: post
title: "오염·재현성·효율 — 이 점수는 무엇을 재고 있나"
date: 2026-08-24 09:20:00 +0900
description: "LLM 평가 체계 시리즈 #20 — 벤치마크 점수를 무너뜨리는 세 가지: 오염(교환가능성 검정), 재현성(lm-evaluation-harness가 드러낸 채점 방식 격차), 효율(문항반응이론 기반 tinyBenchmarks)"
categories: [paper]
tags: [evaluation, contamination, reproducibility, item-response-theory, tinybenchmarks, paper]
giscus_comments: true
related_posts: true
---

> [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623) (Oren, Meister, Chatterji, Ladhak, Hashimoto, ICLR 2024)

# Introduction

지금까지 이 시리즈는 "이 점수를 어떻게 해석하는가"를 물었다. κ로 라벨 일치도를 재고([#13](/blog/2026/kappa-agreement/)), 이항비율로 신뢰구간을 씌우고([#15](/blog/2026/confidence-intervals/)), 짝지은 검정으로 차이의 유의성을 따지고([#16](/blog/2026/significance-testing/)), judge의 편향까지 통계로 보정했다([#19](/blog/2026/judge-statistics/)). 이 모든 논의는 암묵적으로 하나를 전제한다 — **점수가 재려는 것을 실제로 재고 있다는 것.**

이 글은 그 전제 자체를 문제 삼는다. 세 가지가 그 전제를 무너뜨린다.

1. **오염(contamination)**: 벤치마크 문항이 모델의 사전학습 데이터에 이미 들어가 있었다면, 높은 점수는 능력이 아니라 암기를 반영한다.
2. **비재현성(non-reproducibility)**: 같은 이름의 벤치마크를 두 그룹이 돌렸는데 숫자가 다르다면, "이 모델이 MMLU에서 70점을 받았다"는 문장 자체가 정의되지 않은 문장이다.
3. **비용(cost)**: 벤치마크가 수만 문항으로 커지고 모델은 수백 개로 늘어나면, 모든 모델을 모든 문항에 온전히 돌려볼 예산이 없다. 결국 무언가를 덜 재게 되는데, 그 "덜 재는 방법"이 통계적으로 정당화되지 않으면 이번에는 예산이 타당도를 깎아먹는다.

세 문제는 서로 다른 얼굴을 하고 있지만 뿌리는 같다. [#1](/blog/2026/what-is-evaluation/)에서 정의한 어휘로 말하면, 오염은 **구성개념 무관 분산(construct-irrelevant variance)**이 가장 극단적인 형태로 나타난 경우다 — 점수를 흔드는 요인(암기)이 재려는 구성개념(일반화 능력)과 완전히 무관하다. 비재현성은 **신뢰도**의 실패다 — 같은 대상을 같은 방식으로 잰다고 믿었는데, "같은 방식"이라는 전제 자체가 그룹마다 달랐다. 비용은 새로운 문제가 아니라 [#15](/blog/2026/confidence-intervals/)·[#17](/blog/2026/statistical-power/)의 문항 표집 문제가 예산 제약 아래서 다시 등장한 것이다. 이 글은 이 세 얼굴을 각각 어떻게 탐지하고, 회피하고, 다루는지를 순서대로 본다.

# Background — 왜 이 셋이 하나의 문제인가

**오염**부터 보자. 벤치마크의 존재 이유는 **일반화 주장**이다 — "이 모델은 MMLU에서 88점을 받았으니, MMLU가 대표하는 지식 영역에서 이 정도 능력이 있다고 볼 수 있다"는 추론이 성립해야 벤치마크가 의미를 가진다. 그런데 문항이 사전학습 데이터에 그대로 들어 있었다면 이 추론은 끊어진다. 모델은 "지식 영역에서 잘한다"가 아니라 "이 특정 문자열을 본 적이 있다"는 사실만 보여준 것이다. [#1](/blog/2026/what-is-evaluation/)이 코딩 능력 사례로 보인 구도를 그대로 가져오면, 오염은 점수를 재려는 구성개념과 무관한 요인(암기)이 분산을 지배하는 상태이며, 그 분산이 너무 커지면 점수와 구성개념의 상관 자체가 사라진다. 무섭게도 이 실패는 **조용히** 일어난다 — 오염된 모델은 리더보드에서 오히려 더 높은 순위를 받는다. 실패가 점수를 낮추는 게 아니라 점수를 부풀리기 때문에, 별도의 탐지 절차 없이는 아무도 알아채지 못한다.

**비재현성**은 다른 각도에서 같은 신뢰를 깨뜨린다. 두 논문이 "우리 모델은 MMLU에서 70%를 받았다"고 쓴다. 그런데 한쪽은 정답 선택지를 로그우도로 직접 비교했고, 다른 쪽은 "A", "B", "C", "D" 토큰 자체의 우도를 비교했다. 두 방법이 재는 것은 미묘하게 다른 대상이며, 실제로 어느 쪽이 더 관대한지는 모델마다 방향이 다르다(Part 2에서 정확한 수치로 확인한다). 이 상태에서 "MMLU 70%"라는 숫자는 채점 규칙을 명시하지 않으면 재현 불가능한, 사실상 정의되지 않은 양이다. [#6](/blog/2026/mcqa-fragility/)에서 본 프롬프트 포맷 민감도가 "같은 모델, 같은 벤치마크인데 포맷 때문에 점수가 흔들린다"는 문제였다면, 여기서는 그 포맷 선택 하나하나가 논문마다 암묵적으로 달라서, 벤치마크 이름과 숫자만으로는 비교가 성립하지 않는다는 더 근본적인 문제다.

**비용**은 통계적으로 가장 익숙한 얼굴을 하고 있다. [#15](/blog/2026/confidence-intervals/)는 문항 $$n$$개를 표집해 정확도의 신뢰구간을 구했고, 그 신뢰구간의 폭은 $$n$$이 커질수록 좁아진다. 문제는 요즘 벤치마크가 수만 문항, 평가 대상 모델이 수백 개인 상황에서 모든 조합을 온전히 돌리는 비용이 감당하기 어려워진다는 것이다. 예산이 없으니 결국 문항 일부만 표집하게 되는데, [#15]의 이항 모형은 문항을 전부 동등하게(같은 정보량을 가진 것으로) 취급한다. 실제로는 문항마다 "얼마나 모델을 구별해주는가"가 다르고, 이 차이를 무시하고 무작위로 줄이면 같은 예산에서 훨씬 부정확한 추정을 얻는다. 세 문제 모두 결국 하나의 질문으로 모인다 — **이 점수가 지금 무엇을 재고 있는지 우리가 확신할 수 있는가?**

# Method

## Part 1 — 오염: 모델이 이미 답을 봤다

### Oren, Meister, Chatterji, Ladhak, Hashimoto (2023) — 교환가능성 검정

_[Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623) (Oren et al., ICLR 2024)_

이 파트의 중심 논문이고, 접근 자체가 통계적으로 우아하다. 핵심 관찰은 이것이다 — 벤치마크 문항의 순서는 원래 **교환 가능(exchangeable)** 해야 한다. 즉 문항을 어떤 순서로 늘어놓아도 데이터의 결합분포는 바뀌지 않아야 한다(문항 순서 자체가 의미를 갖는 벤치마크가 아니라면). 형식적으로 쓰면 임의의 순열 $$\pi$$에 대해

$$p(x_1, \ldots, x_n) = p(x_{\pi(1)}, \ldots, x_{\pi(n)})$$

가 성립해야 한다. 그런데 모델이 사전학습 중 이 벤치마크를 **원래 순서 그대로** 봤다면, 언어모델은 문서 내 순서를 기억하는 경향이 있으므로, 원래(canonical) 순서로 배열했을 때의 로그우도가 무작위로 섞은 순서들의 로그우도보다 유독 높게 나온다. 오염되지 않았다면 원래 순서든 섞은 순서든 로그우도가 통계적으로 구별되지 않아야 한다 — 이것이 귀무가설이다.

**귀무가설**: 모델 $$\theta$$는 데이터셋 $$X$$와 독립이다(오염되지 않았다면, 벤치마크 순서는 교환 가능하다).

논문은 이를 두 가지 검정통계량으로 정식화한다.

1. **순열 검정(permutation test)**: 원래 순서의 로그우도 $$\log p_\theta(\text{seq}(X))$$를, 무작위로 섞은 순열 $$X_{\pi_1}, \ldots, X_{\pi_m}$$의 로그우도들과 비교한다. 원래 순서가 섞은 순서들보다 유독 높은 순위를 차지하면 오염을 의심한다.
2. **샤딩 검정(sharded test, 논문이 권장하는 방법)**: 데이터셋을 $$r$$개의 조각(shard)으로 나눈 뒤, 조각 $$i$$마다

   $$s_i = \log p_\theta(\text{seq}(X_i)) - \text{Mean}_\pi\big[\log p_\theta(\text{seq}(X_{i,\pi}))\big]$$

   를 계산한다(원래 순서 로그우도에서, 그 조각을 섞었을 때의 평균 로그우도를 뺀 값). $$s_i$$를 조각별로 모아 $$\bar{s} = \frac{1}{r}\sum_i s_i$$를 구하고, $$E[s_i] > 0$$을 대립가설로 하는 단측 $$t$$-검정을 적용한다.

3. **몬테카를로 $$p$$값**: 이론적 분포 대신 실제로 $$m$$개의 무작위 순열을 만들어

   $$\hat{p} = \frac{1 + \#\{\, j : \log p_\theta(\text{seq}(X)) < \log p_\theta(\text{seq}(X_{\pi_j})) \,\}}{m+1}$$

   로 계산한다. 분자에 $$1$$을 더하는 것은 원래 순서 자체를 하나의 "표본"으로 포함시켜 $$\hat p$$가 절대 $$0$$이 되지 않게 하는 표준적인 보정이다. $$\hat{p} \le \alpha$$이면 귀무가설(오염 없음)을 기각한다.

이 방법의 가장 큰 강점은 **사전학습 데이터에 접근하지 않고도(black box)** 성립한다는 것이다 — 필요한 것은 모델에 로그우도를 질의할 수 있다는 것뿐이다. 순열 검정은 유한 표본에서도 정확히 유효하고, 샤딩 검정은 조각들이 독립이고 분산이 유한하다는 가정 아래 중심극한정리로 $$r \to \infty$$일 때 위양성률이 명목 수준 $$\alpha$$로 수렴하는 **점근적** 유효성을 갖는다(논문이 명시적으로 "asymptotic"이라 표현한다 — 유한 표본에서 엄밀히 보장되는 것은 아니다).

실험에서는 LLaMA2-7B, Mistral-7B, Pythia-1.4B, GPT-2-XL과 음성 대조군(오염될 수 없는) BioMedLM을 여러 벤치마크에 대해 검사했다. 대표적인 결과:

| 모델             | 벤치마크 | $$\hat p$$ | 판정                   |
| ---------------- | -------- | ---------- | ---------------------- |
| Mistral-7B       | AI2-ARC  | 0.001      | 유의(오염 의심)        |
| Mistral-7B       | MMLU     | 0.011      | 유의(오염 의심)        |
| LLaMA2-7B        | MMLU     | 0.014      | 유의(오염 의심)        |
| LLaMA2-7B        | AI2-ARC  | 0.318      | 비유의                 |
| Pythia-1.4B      | MMLU     | 0.362      | 비유의                 |
| BioMedLM(대조군) | 전체     | \>0.05     | 비유의(정상 작동 확인) |

논문의 결론은 "이 모델들에서 **광범위한** 오염의 증거는 많지 않다"는 것이다(GPT-3.5·GPT-4 같은 폐쇄형 모델은 감사 대상에 포함되지 않았다). 검정력을 확인하기 위한 합성 주입 실험에서는, 자체 1.4B 모델의 학습 데이터에 문항을 인위적으로 중복 삽입해가며 검정력을 측정했는데, **중복 4회 정도부터 탐지 가능**해지기 시작했고, 10회 중복이면 순열 검정이 $$\hat p = 0.009$$를, 중복 횟수를 더 늘리면 샤딩 검정이 $$p \approx 10^{-38}$$ 같은 극단적으로 작은 값을 낸다.

한계도 논문이 명시한다. 첫째, 다중 검정 보정을 적용하지 않았다(비교 대상 벤치마크·모델 조합이 몇 개인지 정의하기 애매하다는 이유). 둘째, 어떤 벤치마크가 "진짜로" 교환 가능한지는 데이터 생성 과정을 모르고는 증명할 수 없다. 셋째, 이 검정은 **원래 순서를 그대로 기억하는 형태의(verbatim, order-preserving) 오염**만 잡는다 — 문항이 순서가 흐트러진 채 섞여 들어갔거나 의역·재구성되어 들어갔다면 탐지력이 급격히 떨어진다. 단일 중복(딱 한 번만 포함된 경우) 수준의 오염을 안정적으로 잡는 검정은 아직 열린 문제로 남아 있다.

### Golchin & Surdeanu (2023) — Time Travel in LLMs

_[Time Travel in LLMs: Tracing Data Contamination in Large Language Models](https://arxiv.org/abs/2308.08493) (Golchin & Surdeanu, ICLR 2024)_

접근이 완전히 다르다. **guided instruction**이라는 프롬프팅 기법을 쓴다 — 데이터셋 이름과 분할(split) 종류, 그리고 참조 인스턴스의 앞부분(길이를 무작위로 자른 일부)을 프롬프트에 넣고, 모델에게 나머지를 이어 완성하게 시킨다. 모델이 실제 참조 인스턴스의 뒷부분과 정확히 또는 근접하게 일치하는 텍스트를 만들어내면 오염을 의심한다(데이터셋 이름을 주지 않은 일반 instruction과 비교해, 순전히 그럴듯한 텍스트를 생성한 것인지와 구별한다). 사람 전문가의 수동 평가와 비교했을 때 7개 데이터셋에서 탐지 정확도 92\~100%를 달성했고, 이 방법으로 GPT-4가 AG News, WNLI, XSum에서 오염된 것으로 표시됐다.

### Sainz et al. (2023) — NLP Evaluation in trouble

_[NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark](https://aclanthology.org/2023.findings-emnlp.722/) (Sainz et al., EMNLP Findings 2023)_

이 논문은 새 검정을 제안하기보다 문제의 심각성을 지적한다. 가장 심한 형태의 오염 — 벤치마크의 테스트 분할 자체를 학습에 써버린 경우 — 을 표준적으로 측정하는 방법이 아직 없다는 것이다. 저자들은 오염이 모델 성능을 실제보다 부풀린다는 점을 지적하며, 커뮤니티가 자동·반자동 오염 탐지 방법을 개발하고, 오염이 확인된 논문의 결론에 플래그를 다는 관행을 세워야 한다고 제안한다. 부산물로 알려진 오염 사례를 수작업으로 모은 "LM Contamination Index" 데이터베이스도 함께 공개했다.

### n-gram 중복 검사 — 가장 오래되고 가장 단순한 방법

사전학습 데이터에 접근할 수 있는 개발사가 실제로 쓰는 방법은 훨씬 소박하다. GPT-3 논문(Brown et al., 2020)은 평가 문항과 학습 문서 사이에 8\~13-gram(문항 길이의 5분위수를 기준으로, 최소 8·최대 13 단어로 정함) 이상 겹치는 부분이 있으면 오염된 것으로 간주하고, 겹치는 n-gram과 그 주변 200자를 학습 문서에서 잘라낸다. 10개가 넘는 학습 문서에서 동시에 나타나는 n-gram은 관용구나 법률 상용구일 가능성이 높다고 보고 제외한다.

GPT-4 기술 보고서(OpenAI, 2023)는 더 단순한 규칙을 쓴다 — 공백과 기호를 지운 뒤, 평가 문항마다 무작위로 50자 길이 부분문자열 3개를 뽑아(문항이 50자보다 짧으면 문항 전체를 쓴다), 그 부분문자열이 학습 데이터 안에 그대로 있으면 오염으로 표시한다. 문항의 질문·맥락만 검사하고 정답·선택지는 제외하는데, 이는 오히려 위양성을 늘리는 방향으로 작용한다고 보고서가 스스로 인정한다. GPT-4의 경우 이렇게 찾아낸 오염을 제거하고 다시 채점해도 "결과에 미치는 영향은 매우 작았다"고 밝힌다.

n-gram 중복 검사의 한계는 명확하다. **표층 문자열이 그대로 겹칠 때만** 잡는다 — 의역, 번역, 재구성을 거친 의미상 동일한 오염은 놓친다. 그리고 정의상 **사전학습 데이터 자체에 접근**해야 하므로, 외부 연구자가 폐쇄형 모델을 감사할 때는 쓸 수 없다. 바로 이 지점이 Oren et al.의 블랙박스 검정이 갖는 가치다 — 데이터에 접근할 수 없는 제3자도 검증할 수 있다.

### Zhou et al. (2023) — benchmark leakage

_[Don't Make Your LLM an Evaluation Benchmark Cheater](https://arxiv.org/abs/2311.01964) (Zhou et al., arXiv 2023)_

짧게만 짚는다. 이 논문은 "벤치마크 유출(benchmark leakage)"이라는 용어로, 평가 데이터와 관련된 데이터가 의도치 않게(또는 의도적으로) 학습에 섞여 들어가는 상황이 성능을 크게 부풀릴 수 있다고 경고한다. 구체적인 부풀림 폭에 대한 검증된 수치까지는 확인하지 못했지만, LLM 개발자와 벤치마크 관리자 양쪽을 향한 실천 지침을 제안한다는 점에서 Sainz et al.과 결이 같다.

### 다섯 가지 탐지 방법, 한눈에 놓고 비교하기

지금까지 본 방법들은 서로 대체재가 아니라 **요구하는 접근 권한과 잡아내는 대상이 다른** 보완재다.

| 방법                                                 | 필요한 접근 권한             | 잡아내는 오염                         | 이 글에서의 강점/한계                                                                               |
| ---------------------------------------------------- | ---------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 교환가능성 검정 (Oren et al.)                        | 로그우도 질의만(블랙박스)    | 원래 순서를 보존한 verbatim 오염      | 제3자도 감사 가능하지만, 순서가 흐트러지거나 의역된 오염은 놓친다                                   |
| Guided instruction (Golchin & Surdeanu)              | 텍스트 생성 질의만(블랙박스) | 모델이 이어쓰기로 재현할 수 있는 오염 | 탐지 정확도는 높지만(92\~100%), 참조 인스턴스의 앞부분을 실제로 알고 있어야 프롬프트를 만들 수 있다 |
| n-gram·부분문자열 중복 검사                          | 사전학습 데이터 접근 필요    | 표층 문자열이 그대로 겹치는 오염      | 개발사 스스로는 정밀하게 쓸 수 있지만, 외부 감사에는 원천적으로 못 쓴다                             |
| LM Contamination Index 등 커뮤니티 DB (Sainz et al.) | 공개된 사례 취합             | 이미 보고된 오염 사례                 | 새로운 오염을 탐지하는 방법이 아니라, 알려진 사례를 추적하는 아카이브다                             |

이 표가 말하려는 것은 "어느 방법이 최선인가"가 아니라, 블랙박스 검정과 데이터 접근 기반 검사가 **서로 다른 틈을 막는다**는 것이다 — 폐쇄형 모델을 외부에서 감사할 때는 전자만 쓸 수 있고, 개발사가 스스로 점검할 때는 후자가 훨씬 정밀하다.

### 오염을 피하는 벤치 설계

탐지 대신 설계로 문제를 우회하는 접근도 있다.

**LiveBench** — _[LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314) (White, Dooley et al., ICLR 2025 Spotlight)_. 최근 수학 경시대회, arXiv 논문, 뉴스 기사, 새로 나온 데이터셋에서 문항을 계속 새로 만들어낸다. 수학·코딩·추론·언어·지시따르기·데이터 분석 6개 범주로 구성되며, 매달 전체 문항의 약 6분의 1을 새 문항으로 교체해 전체가 약 6개월 주기로 갈아치워진다. 정답이 객관적으로 검증 가능하게 설계되어 LLM judge가 필요 없다는 것도 설계상의 장점이다. 공개 당시 최상위 모델(o1-preview-2024-09-12)의 점수도 64.7%에 그쳐, 선두 모델 전부가 70% 아래였다 — 오염되지 않은 새 문항 앞에서는 벤치마크가 여전히 어렵다는 뜻이다.

**LiveCodeBench**는 같은 시점 기반 수집 전략을 코드 도메인에 적용한 사례로, [#8](/blog/2026/math-code-benchmarks/)에서 이미 다뤘으므로 링크만 건다.

**Chatbot Arena**도 원리적으로 정적 오염에서 자유롭다 — 사용자가 계속 새로운 프롬프트를 실시간으로 채워 넣기 때문에, 모델이 "정답"을 사전학습에서 미리 봤을 수 있는 고정된 문항 집합 자체가 존재하지 않는다([#9](/blog/2026/mt-bench-to-arena/)). 다만 이것이 모든 오염 위험을 없애는 것은 아니다 — 개발사가 Arena류의 선호 패턴 자체를 목표로 최적화하면, 문항 단위 오염과는 다른 종류의 Goodhart 위험이 남는다.

**비공개 홀드아웃 + 제출 서버** 방식(정답을 공개하지 않고, 참가자는 모델이나 예측 결과만 제출해 서버가 채점)은 장단이 뚜렷하다. 장점은 문항이 한 번도 공개되지 않으므로 직접적인 스크래핑을 통한 오염이 원천적으로 막힌다는 것이다. 단점은 두 가지다 — 반복 제출을 허용하면 홀드아웃 자체에 암묵적으로 과적합되는(리더보드를 향한 다중비교 문제, [#17](/blog/2026/statistical-power/)) 위험이 생기고, 무엇보다 외부 연구자가 채점 과정을 스스로 재현할 수 없다는 점에서 Part 2가 요구하는 재현성(정확한 문항·채점 코드 공개)과 정면으로 충돌한다. 오염을 막는 방법이 재현성을 깎아먹는 이 긴장은 이 글 전체가 반복해서 마주치는 트레이드오프다.

## Part 2 — 재현성: 같은 벤치가 다른 숫자를 낸다

### Biderman et al. (2024) — lm-evaluation-harness의 교훈

_[Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (Biderman et al., arXiv 2024)_

lm-evaluation-harness를 3년 넘게 개발·운영한 팀이 실무에서 마주친 재현성 붕괴 원인을 정리한 논문이다. 논문이 짚는 원인 목록은 이렇다.

- **프롬프트 템플릿** — few-shot 예시를 어떤 문구·구분자로 감싸는지.
- **정규화 방식** — 로그우도를 답 길이로 나누는지(length normalization), 조건 없는 확률로 나누는지(unconditional normalization).
- **정답 추출 방식** — 선택지 각각의 로그우도를 직접 비교하는지, 모델이 생성한 자유 텍스트를 파싱해서 정답을 추출하는지.
- **few-shot 예시 선택과 순서** — 어떤 예시를 뽑는지, 어떤 순서로 배열하는지가 성능을 눈에 띄게 바꾼다.
- **토크나이저 차이**, **배치 크기·패딩**, **부동소수점 비결정성**.

이 중 가장 극적인 수치 증거는 **정답 추출 방식**이다. 같은 모델, 같은 문항인데 "cloze"(정답 선택지 자체의 로그우도를 직접 비교) 방식과 "MMLU-style"(정답을 가리키는 기호/문자 "A", "B", "C", "D" 토큰의 우도를 비교) 방식으로 채점을 바꾸면 이렇게 벌어진다.

| 모델         | ARC-C (cloze) | ARC-C (MMLU-style) | 차이                       | MMLU (hybrid) | MMLU (MMLU-style) | 차이   |
| ------------ | ------------- | ------------------ | -------------------------- | ------------- | ----------------- | ------ |
| GPT-NeoX-20B | 38.0%         | 26.6%              | 11.4%p (cloze가 높음)      | 27.6%         | 24.5%             | 3.1%p  |
| Llama-2-7B   | 43.5%         | 42.8%              | 0.7%p                      | 39.8%         | 41.3%             | 1.5%p  |
| Falcon-7B    | 40.2%         | 25.9%              | 14.3%p (cloze가 높음)      | 29.1%         | 25.4%             | 3.7%p  |
| Mistral-7B   | 50.1%         | 72.4%              | 22.3%p (MMLU-style가 높음) | 48.3%         | 58.6%             | 10.3%p |
| Mixtral-8x7B | 56.7%         | 81.3%              | 24.6%p (MMLU-style가 높음) | 59.7%         | 67.1%             | 7.4%p  |

핵심은 **어느 방식이 더 관대한지의 방향이 모델마다 뒤집힌다**는 것이다 — GPT-NeoX-20B와 Falcon-7B는 cloze 채점이 훨씬 높게 나오고, Mistral-7B와 Mixtral-8x7B는 정반대로 MMLU-style 채점이 훨씬 높게 나온다. 이는 [#6](/blog/2026/mcqa-fragility/)에서 다룬 객관식 평가의 포맷 민감도와 정확히 같은 종류의 문제이지만, 여기서는 포맷이 아니라 **채점 규칙 자체**가 흔들리는 원인이라는 점이 다르다. 방향이 모델마다 다르다는 사실은 "한 방식에서 다른 방식으로 옮길 때 몇 %p를 더하면 된다"는 식의 사후 보정이 불가능하다는 뜻이기도 하다. 논문은 더 나아가 MMLU를 구현한 세 가지 독립적인 코드베이스(HELM, lm-eval-harness, MMLU 원 저자 코드)를 비교했을 때 "결과가 크게 다르고, 심지어 모델 간 순위까지 바뀐다"고 밝힌다.

논문 부록의 **모범 사례 체크리스트**는 다섯 줄로 요약된다 — (1) 정확한 프롬프트를 항상 공개하라, (2) 다른 구현체의 결과를 그대로 복사해 비교하지 말라, (3) 모델 출력 자체를 항상 제공하라, (4) 정성적 분석을 수행하라, (5) 통계적 유의성 검정을 수행하라. 이 목록은 아래 "통계 요약" 표의 처방 칸에도 그대로 들어간다.

이 문제는 [#18](/blog/2026/error-bars-for-evals/)이 다룬 분산원의 목록과는 층이 다르다. [#18]은 **채점 규칙이 고정되어 있다는 전제 아래** 문항 표집·클러스터 구조·디코딩 랜덤성이 표준오차를 얼마나 키우는지를 다뤘다. 여기서 Biderman et al.이 보여주는 것은 그 전제 자체가 흔들린다는 것이다 — 채점 규칙(정답 추출 방식, 정규화)을 어떻게 정하느냐에 따라 점수가 수 %p에서 20%p 넘게 갈라지고, 그 갈라지는 방향조차 모델마다 다르다. 표준오차를 아무리 정확히 계산해도, 애초에 "어떤 채점 규칙으로 나온 점수인가"를 밝히지 않으면 그 오차 막대가 어떤 양에 붙어 있는지조차 알 수 없다.

### 생성 지표에서 먼저 터진 문제

"같은 이름, 다른 숫자"는 사실 이 시리즈에서 처음 나온 현상이 아니다. [#5](/blog/2026/generation-metrics/)에서 다룬 Post (2018)의 sacreBLEU는 정확히 같은 문제를 BLEU에서 먼저 지적했다 — 토큰화·정규화 설정을 밝히지 않으면 "BLEU 32.1"이라는 숫자는 재현할 수 없는 숫자다. Biderman et al.이 lm-evaluation-harness에서 겪은 것은 그 문제가 생성 지표뿐 아니라 객관식 정답률에도 똑같이 적용된다는 것을 보여준 사례다.

### 프롬프트 민감도와 재현성

프롬프트 포맷 하나만 바꿔도 성능이 크게 흔들린다는 것은 이미 [#6](/blog/2026/mcqa-fragility/)에서 Sclar et al. (2024)의 FormatSpread와 Alzahrani et al. (2024)의 리더보드 순위 변동으로 상세히 다뤘으므로 여기서 다시 서술하지 않는다. 이 글의 관점에서 덧붙일 것은 하나뿐이다 — 프롬프트 민감도가 존재한다는 사실 자체가, harness가 프롬프트를 **임의로 고르는 대신 고정하고 명시적으로 보고해야 하는** 이유다. 프롬프트가 재현성의 원인 목록에 오르는 것은 우연이 아니라, 흔들림의 크기가 크기 때문이다.

### HRET — 한국어 평가의 재현성 인프라

한국어 벤치마크에서도 같은 문제가 반복된다. [#11](/blog/2026/korean-benchmarks/)에서 다룬 [HRET](https://arxiv.org/abs/2503.22968) (Lee et al., arXiv 2025)은 여러 한국어 벤치마크의 프롬프트 포맷·정답 추출 방식·평가 백엔드를 레지스트리 구조로 표준화한 툴킷이다. 여기서는 링크만 걸어둔다 — 결국 이 툴킷이 하는 일도 Biderman et al.의 체크리스트를 한국어 벤치마크 생태계에 그대로 적용한 것이다.

## Part 3 — 효율: 적게 재고 많이 알기

### tinyBenchmarks — 문항반응이론(IRT)으로 벤치마크를 압축하다

_[tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992) (Maia Polo, Weber, Choshen, Sun, Xu, Yurochkin, ICML 2024)_

이 시리즈에서 **문항반응이론(Item Response Theory, IRT)**을 처음 소개하므로 제대로 짚고 간다. IRT는 원래 교육·심리 측정에서 나온 이론으로, "문항마다 난이도와 변별력이 다르다"는 상식을 정식 통계 모형에 넣는다. 가장 기본적인 형태인 **2모수 로지스틱 모형(2PL)**은 이렇게 쓴다.

$$P(\text{정답} \mid \theta) = \frac{1}{1 + e^{-a(\theta - b)}}$$

기호를 하나씩 풀면:

- $$\theta$$ — 모델(피험자)의 **잠재 능력(latent ability)**. 직접 관측되지 않고 응답 패턴에서 추정해야 하는 값이다.
- $$b$$ — 문항의 **난이도**. $$\theta = b$$일 때 정답률이 정확히 $$0.5$$가 되도록 정의된다.
- $$a$$ — 문항의 **변별도(discrimination)**. $$\theta$$가 $$b$$를 지나며 커질 때 정답 확률이 얼마나 급하게 올라가는지를 정한다.

변별도가 왜 핵심인지는 극단값을 넣어보면 바로 보인다. $$a \to 0$$이면 지수 $$-a(\theta-b) \to 0$$이 되어 $$P(\text{정답}\mid\theta) \to 1/2$$, 즉 $$\theta$$가 무엇이든 정답률이 항상 $$0.5$$로 고정된다 — 능력이 높은 모델이든 낮은 모델이든 이 문항을 맞힐 확률이 똑같다는 뜻이다. 이런 문항은 **아무리 많이 모아도 모델을 구별하는 데 아무 정보를 주지 못한다.** 반대로 $$a$$가 크면 $$\theta$$가 $$b$$를 살짝만 넘어도 정답률이 급격히 $$1$$에 가까워진다 — 능력 있는 모델과 없는 모델을 예리하게 갈라준다.

이 직관을 정량화한 것이 **정보함수(information function)** 다. 일반적인 IRT 이론에서 정보함수는

$$I(\theta) = \frac{[P'(\theta)]^2}{P(\theta)\big(1 - P(\theta)\big)}$$

로 정의된다. 2PL의 $$P(\theta)$$는 로지스틱 함수라 $$P'(\theta) = a\,P(\theta)\big(1-P(\theta)\big)$$이 성립하고(로지스틱 함수의 표준적인 미분 성질), 이를 대입하면

$$I(\theta) = a^2\, P(\theta)\big(1 - P(\theta)\big)$$

로 깨끗하게 정리된다. 이 식은 두 가지를 바로 보여준다. 첫째, $$a=0$$이면 $$\theta$$와 무관하게 $$I(\theta)=0$$이다 — 앞서 말한 "변별도 0인 문항은 정보가 없다"가 수식으로 확인된다. 둘째, 정보량은 $$a$$의 **제곱**에 비례하므로 변별도가 조금만 높아져도 정보량은 훨씬 빠르게 커진다. $$\theta = b$$인 지점(정답률이 정확히 $$0.5$$인 곳)에서 $$P(1-P)$$가 최댓값 $$0.25$$를 가지므로, 문항이 가장 많은 정보를 주는 지점은 $$I(b) = a^2/4$$다 — 그 문항의 난이도에 정확히 걸맞은 능력을 가진 모델을 지나갈 때 가장 예리하게 구별해준다는 뜻이다.

tinyBenchmarks는 이 모형을 다차원으로 확장해서 쓴다. 모델 $$l$$이 문항 $$i$$를 맞힐 확률을

$$p_{il} = P(Y_{il}=1 \mid \theta_l, \alpha_i, \beta_i) = \frac{1}{1 + e^{-(\alpha_i^\top \theta_l - \beta_i)}}$$

로 놓고(위 단일 변수 버전의 $$a, b$$가 여기서는 벡터 $$\alpha_i$$와 스칼라 $$\beta_i$$로 일반화된다), **395개 모델**의 실제 응답으로 이 모형을 학습시킨다. 그런 다음 문항 파라미터 추정치 $$(\hat\alpha_i, \hat\beta_i)$$ 공간에서 $$k$$-평균 군집화를 적용해, 난이도·변별도 공간을 고르게 대표하는 소수의 **앵커 문항(anchor items)** — 벤치마크마다 100개 — 을 골라낸다. 무작위로 100개를 뽑는 대신 이렇게 고르면, 서로 겹치는 정보를 주는 쉬운 문항들이 자리를 낭비하지 않고, 소수의 문항이 파라미터 공간 전체를 효율적으로 커버한다.

앵커 문항의 응답만으로 전체 벤치마크 점수를 추정하는 세 가지 추정량을 제안한다 — (1) **IRT**: 앵커 응답으로 $$\theta_l$$을 추정한 뒤, 학습된 문항 은행 전체에 대해 정답률을 예측, (2) **p-IRT**: 앵커 문항에서 실제 관측한 정답률과 IRT 예측을 함께 반영, (3) **gp-IRT**: 두 추정량을 상황에 맞게 가중 결합한 일반화 버전. 결과는 인상적이다 — **문항 100개**로 MMLU(14,000문항), HELM Lite, AlpacaEval 2.0(805문항), Open LLM Leaderboard(약 29,000문항, 6개 시나리오 각 100개)의 전체 성능을, 평균 추정 오차 **1\~3%p 수준**으로 복원한다(공개된 결과 기준 MMLU 약 1.6\~2.4%p, GSM8K 약 2.0\~2.9%p 범위). 같은 예산으로 무작위·계층 표집을 했을 때보다 오차가 뚜렷하게 크다는 것도 함께 보고한다.

[#15](/blog/2026/confidence-intervals/)의 이항 모형과 비교하면 이 접근의 의의가 분명해진다. [#15]의 이항 신뢰구간은 문항 $$n$$개가 전부 동등한 정보를 준다고(같은 $$p$$를 가진 독립 베르누이 시행이라고) 가정한다. IRT는 그 가정을 깨고 **문항마다 난이도와 변별도가 다르다는 것을 모형 안에 명시적으로 넣는다.** 그 결과 "어떤 문항이 예산을 쓸 가치가 있는가"라는 질문에 답할 수 있게 된다 — 이것이 무작위로 문항을 줄이는 것과 tinyBenchmarks가 다른 지점이다.

### Rodriguez et al. (2021) — 리더보드에 IRT를 적용하다

_[Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?](https://aclanthology.org/2021.acl-long.346/) (Rodriguez, Barrow, Hoyle, Lalor, Jia, Boyd-Graber, ACL-IJCNLP 2021)_

tinyBenchmarks보다 먼저, IRT를 NLP 리더보드에 직접 적용한 논문이다. 잠재 능력(모델)과 문항 난이도를 함께 추정하는 베이지안 IRT 모형을 세워, 리더보드 순위의 신뢰성을 점검하고, 라벨링 오류로 의심되는 문항을 찾아내고, 특정 모델이 벤치마크에 과적합된 흔적을 탐지하고, 어떤 문항이 실제로 모델을 구별하는 데 유익한지를 짚어낸다. tinyBenchmarks가 "몇 문항으로 줄여도 되는가"라는 효율의 질문에 IRT를 썼다면, 이 논문은 그보다 앞서 "지금 있는 문항 전체 중 어떤 것이 정보를 주고 있는가"라는 진단의 질문에 IRT를 썼다.

### Perlitz et al. (2024) — Efficient Benchmarking

_[Efficient Benchmarking (of Language Models)](https://arxiv.org/abs/2308.11696) (Perlitz, Bandel, Gera, Arviv, Ein-Dor, Shnarch, Slonim, Shmueli-Scheuer, Choshen, NAACL 2024)_

HELM을 테스트베드로 삼아, 벤치마크 설계 선택(문항 수, few-shot 샷 수, 비교 대상 모델 집합 등)이 "이 벤치마크로부터 내리는 결론"의 신뢰성에 어떤 영향을 주는지를 정량화한다. 이를 위해 **DIoR(Decision Impact on Reliability)** 라는 지표를 새로 제안한다. 발견 중 하나가 흥미롭다 — HELM의 1위 모델이 **순위가 낮은 모델 하나를 비교 대상에서 빼는 것만으로도 바뀔 수 있다**는 것이다. 이는 문항이나 채점 방식이 전혀 바뀌지 않아도, 단지 "누구와 비교하는가"라는 비교 집합 구성 자체가 순위를 불안정하게 만든다는 뜻이다. 또한 평가 문항의 **일부만으로도 올바른 벤치마크 순위를 복원할 수 있다**는 것을 관찰하고, 이 발견에 기반한 평가 알고리즘을 HELM에 적용해 **계산량을 100배 이상 절감**하면서도 신뢰성 손실을 최소화했다고 보고한다.

### Item Response Theory for AI Safety (2026)

_[Item Response Theory for AI Safety](https://arxiv.org/abs/2608.05086) (Fonseca Rivera, Shah, Africa, Voudouris, arXiv 2026)_

가장 최근 사례로 짧게 짚는다. 안전 벤치마크 8개, 언어모델 192개에 IRT 모형을 적합한, 저자들 표현으로 "지금까지 가장 큰 규모의 LLM 안전 평가 심리계량 분석"이다. 배경 문제의식은 이렇다 — 안전 벤치마크들은 서로 겹치는 내용을 재고, 벤치마크 간 상관이 지나치게 높으며, 모델이 평가받고 있음을 감지하면 일부러 못하는 척(sandbagging)할 수도 있다. 이런 상황에서 벤치마크 점수를 단순 합산한 값은 신뢰하기 어렵다. 저자들은 IRT를 적용해 모델 간 차이의 대부분이 **거부 엄격성(refusal strictness), 진실성(truthfulness), 맥락적 위해성(contextual harm)** 이라는 해석 가능한 세 개의 잠재 요인으로 설명된다는 것을 보인다. 안전 평가에 IRT를 효율화 목적으로 적용한 인접 연구로 [Efficient Safety Benchmarking via Item Response Theory](https://arxiv.org/abs/2606.20626) (arXiv 2026)도 있다.

# Experiments

## 1. 교환가능성 검정의 $$\hat p$$를 손으로 계산해보기

Oren et al.의 순열 검정이 어떻게 작동하는지, 가상의 숫자로 직접 따라가보자. 문항 열 하나를 원래 순서 그대로 모델에 넣었을 때 로그우도가 $$-120.0$$이 나왔다고 하자. 이제 이 문항들을 무작위로 섞은 순열을 $$m=9$$개 만들어 각각의 로그우도를 구했더니, 원래 순서보다 낮은 로그우도가 나온 순열이 $$8$$개, 원래 순서보다 높게 나온 순열이 $$1$$개였다고 하자(즉 원래 순서가 거의 항상 "더 그럴듯한" 순서로 나왔다). 공식에 넣으면

$$\hat p = \frac{1 + 1}{9+1} = \frac{2}{10} = 0.2$$

이다. 통상적인 $$\alpha=0.05$$ 기준으로는 오염을 기각하지 못한다. 이제 같은 상황에서 $$m$$을 $$99$$개로 늘리고, 그중 원래 순서보다 높은 로그우도가 나온 순열이 여전히 딱 $$1$$개뿐이라면

$$\hat p = \frac{1+1}{99+1} = \frac{2}{100} = 0.02$$

로 떨어진다. 같은 "원래 순서가 거의 항상 이긴다"는 신호라도, 순열 수 $$m$$을 늘려야 그 신호가 통계적으로 유의한 수준까지 정밀해진다 — [#17](/blog/2026/statistical-power/)에서 다룬 표본 크기와 검정력의 관계가 여기서도 그대로 작동한다는 뜻이다.

## 2. 변별도가 다른 세 문항의 정보량 비교

$$b=0$$으로 난이도를 맞춘 세 문항이 변별도만 $$a=0.3$$(낮음), $$a=1.0$$(중간), $$a=2.5$$(높음)로 다르다고 하자. 각 문항이 자신의 난이도 지점($$\theta=b=0$$, 이 때 $$P=0.5$$)에서 주는 정보량 $$I(0) = a^2 \times 0.25$$는 다음과 같다.

| 변별도 $$a$$ | $$I(0) = a^2 \times 0.25$$ |
| ------------ | -------------------------- |
| 0.3          | 0.0225                     |
| 1.0          | 0.25                       |
| 2.5          | 1.5625                     |

변별도가 높은 문항($$a=2.5$$)은 낮은 문항($$a=0.3$$)보다 같은 지점에서 약 **69배** 많은 정보를 준다($$1.5625 / 0.0225 \approx 69.4$$). tinyBenchmarks가 100개의 앵커 문항을 무작위가 아니라 파라미터 공간에서 골라내는 이유가 바로 이것이다 — 변별도가 낮은 문항을 아무리 많이 모아도, 변별도가 높은 문항 몇 개가 주는 정보량을 따라가지 못한다.

# 통계 요약

| 문제                                 | 진단 방법                                                                   | 처방                                                                                                  | 대표 문헌                                                             |
| ------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 오염 (사전학습 데이터에 문항이 포함) | 교환가능성 검정(순열/샤딩), guided instruction, n-gram·부분문자열 중복 검사 | 블랙박스 감사 실행, 시점 기반 신규 문항으로 벤치마크 설계, 오염 검사 결과와 방법을 공개               | Oren et al. (2023), Golchin & Surdeanu (2023), LiveBench (2024)       |
| 비재현성 (같은 벤치가 다른 숫자)     | 서로 다른 구현체(정답 추출·정규화·프롬프트) 간 점수 비교                    | harness와 정확한 설정(프롬프트·정답 추출 방식·few-shot 구성)을 명시, 모델 출력 공개, 유의성 검정 수행 | Biderman et al. (2024), Post (2018)                                   |
| 비용 (예산 안에서 다 못 잰다)        | IRT 문항 파라미터(난이도 $$b$$·변별도 $$a$$) 추정, 정보함수, DIoR           | 변별도 높은 문항으로 구성된 앵커 집합 선정, 추정량과 오차를 함께 보고                                 | tinyBenchmarks (2024), Rodriguez et al. (2021), Perlitz et al. (2024) |

# Conclusion

이 글에서 다룬 세 문제 — 오염, 비재현성, 비용 — 은 표면적으로 다른 도구(통계 검정, 소프트웨어 엔지니어링, 심리계량 모형)를 쓰지만 결국 하나의 질문으로 되돌아온다. **이 점수가 지금 무엇을 재고 있는지 우리가 확신할 수 있는가?** 오염은 그 확신을 조용히, 그리고 점수를 올리는 방향으로 무너뜨린다. 비재현성은 "이 벤치마크의 이 숫자"라는 문장 자체가 설정을 명시하지 않으면 정의되지 않는다는 것을 보여준다. 비용은 덜 재야 하는 상황에서 무엇을 덜 재도 되는지에 대한 통계적 근거 없이 임의로 줄이면, 예산 제약이 그대로 타당도의 손실로 이어진다는 것을 보여준다.

실무 체크리스트로 정리하면 이렇다.

1. 새 벤치마크를 도입하거나 SOTA를 주장할 때는 최소한 블랙박스 오염 검정(교환가능성 검정) 하나는 실행한다.
2. 평가 결과를 보고할 때는 harness 이름과 정확한 설정(정답 추출 방식·정규화·few-shot 구성)을 명시한다.
3. 가능하면 프롬프트 자체를 공개한다.
4. 문항 수를 줄여야 한다면 무작위가 아니라 문항의 정보량(변별도)을 고려해서 줄인다.
5. 가능하면 시점 기반으로 계속 새로 만들어지는 벤치마크를 함께 쓴다.

그리고 이 세 문제는 안전 평가에서 훨씬 더 심각해진다. [#21](/blog/2026/safety-evaluation-statistics/)에서 다룰 이야기를 미리 예고하면 — 안전 벤치마크는 대개 문항 수가 적고(통계적 정밀도가 낮다), 공개되는 즉시 다음 세대 모델의 사전학습 데이터에 흘러들어가기 쉽고(오염 위험이 훨씬 크다), 그리고 재려는 사건(치명적 실패) 자체가 애초에 희귀해서 "위반 0건 관찰"이 "안전하다"를 뜻하지 않는다. 이 글의 세 문제가 안전 평가라는 무대에서 한꺼번에, 그리고 더 날카롭게 다시 등장한다.

# 참고 문헌

- Oren, Meister, Chatterji, Ladhak, Hashimoto, 2023. [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623) (ICLR 2024).
- Golchin & Surdeanu, 2023. [Time Travel in LLMs: Tracing Data Contamination in Large Language Models](https://arxiv.org/abs/2308.08493) (ICLR 2024).
- Sainz, Campos, García-Ferrero, Etxaniz, Lopez de Lacalle, Agirre, 2023. [NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark](https://aclanthology.org/2023.findings-emnlp.722/) (EMNLP Findings 2023).
- Zhou et al., 2023. [Don't Make Your LLM an Evaluation Benchmark Cheater](https://arxiv.org/abs/2311.01964) (arXiv 2023).
- Brown et al., 2020. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (OpenAI, NeurIPS 2020).
- OpenAI, 2023. [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774).
- White, Dooley et al., 2024. [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314) (ICLR 2025 Spotlight).
- Biderman et al., 2024. [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (arXiv 2024).
- Lee et al., 2025. [HRET: A Self-Evolving LLM Evaluation Toolkit for Korean](https://arxiv.org/abs/2503.22968).
- Post, 2018. [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) (WMT 2018).
- Maia Polo, Weber, Choshen, Sun, Xu, Yurochkin, 2024. [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992) (ICML 2024).
- Rodriguez, Barrow, Hoyle, Lalor, Jia, Boyd-Graber, 2021. [Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?](https://aclanthology.org/2021.acl-long.346/) (ACL-IJCNLP 2021).
- Perlitz, Bandel, Gera, Arviv, Ein-Dor, Shnarch, Slonim, Shmueli-Scheuer, Choshen, 2024. [Efficient Benchmarking (of Language Models)](https://arxiv.org/abs/2308.11696) (NAACL 2024).
- Fonseca Rivera, Shah, Africa, Voudouris, 2026. [Item Response Theory for AI Safety](https://arxiv.org/abs/2608.05086) (arXiv 2026).

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 스무 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="3">
  <li><a href="/blog/2026/measurement-scales/">척도와 허용 연산</a> — Likert 평균을 내도 되는가</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><a href="/blog/2026/generation-metrics/">생성 지표와 그 타당도</a> — BLEU에서 COMET까지</li>
  <li><a href="/blog/2026/mcqa-fragility/">객관식 평가는 왜 흔들리나</a> — 위치 편향과 포맷 민감도</li>
</ol>

**3부. LLM 벤치마크 지형도**

<ol start="7">
  <li><a href="/blog/2026/knowledge-benchmarks/">지식과 추론 — MMLU 계열의 흥망</a> — MMLU·GPQA·BBH·HELM</li>
  <li><a href="/blog/2026/math-code-benchmarks/">검증 가능한 도메인 — 수학과 코드</a> — GSM8K·MATH·HumanEval·SWE-bench</li>
  <li><a href="/blog/2026/mt-bench-to-arena/">개방형 대화 — MT-Bench에서 Arena까지</a> — judge 기반 벤치의 등장</li>
  <li><a href="/blog/2026/capability-axes-benchmarks/">능력의 다른 축</a> — 지시따르기·긴 문맥·사실성</li>
  <li><a href="/blog/2026/korean-benchmarks/">한국어 벤치마크</a> — 번역이 아니라 원산, 그리고 문화 타당도</li>
</ol>

**4부. 사람이 읽는다 — 정성평가와 일치도**

<ol start="12">
  <li><a href="/blog/2026/human-evaluation-design/">사람 평가 설계</a> — 루브릭·Likert·pairwise·BWS</li>
  <li><a href="/blog/2026/kappa-agreement/">우연을 빼다 — κ 계열</a> — Cohen·Fleiss·weighted·Krippendorff</li>
  <li><a href="/blog/2026/kappa-paradox/">κ의 역설</a> — 일치율 90%인데 κ가 0.21</li>
</ol>

**5부. 차이는 진짜인가 — 정량평가의 통계**

<ol start="15">
  <li><a href="/blog/2026/confidence-intervals/">점수는 추정치다</a> — 이항비율 신뢰구간과 Wald의 실패</li>
  <li><a href="/blog/2026/significance-testing/">차이는 유의한가</a> — paired bootstrap·순열검정·McNemar</li>
  <li><a href="/blog/2026/statistical-power/">몇 개를 재야 하나</a> — 검정력·표본크기·다중비교</li>
  <li><a href="/blog/2026/error-bars-for-evals/">LLM eval의 통계 실무</a> — 클러스터 SE·IQM·분산 분해</li>
</ol>

**6부. 신뢰할 수 있는 평가 체계**

<ol start="19">
  <li><a href="/blog/2026/judge-statistics/">judge를 통계로 다루기</a> — 편향·Bradley-Terry·PPI</li>
  <li><strong>(현재 글)</strong> 오염·재현성·효율 — 오염 검정·harness·IRT</li>
  <li><a href="/blog/2026/safety-evaluation-statistics/">안전 평가의 통계와 체계 설계</a> — 희귀사건·calibration·체크리스트</li>
</ol>

본 시리즈는 21편으로 구성된다.
