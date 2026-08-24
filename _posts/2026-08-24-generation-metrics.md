---
layout: post
title: "생성 지표와 그 타당도 — BLEU에서 COMET까지"
date: 2026-08-24 09:05:00 +0900
description: "LLM 평가 체계 시리즈 #5 — n-gram 겹침 지표부터 학습된 지표까지, 그리고 그 타당도가 어디서 깨지는가"
categories: [paper]
tags: [evaluation, nlg, machine-translation, bleu, comet, paper]
giscus_comments: true
related_posts: true
---

> [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) (Papineni et al., IBM, ACL 2002)

# Introduction

[#4](/blog/2026/classification-metrics/)에서 분류 지표를 다뤘다. 분류는 편하다 — 정답 레이블이 하나로 정해져 있고, 모델의 출력을 그 레이블과 맞대어 보면 맞았는지 틀렸는지가 나온다. 생성(generation)은 이 전제가 통째로 무너진다. "이 문장을 한국어로 번역하라"는 요청에 정답은 하나가 아니다. "고양이가 매트 위에 앉아 있다"와 "매트 위에 고양이가 앉았다"는 둘 다 맞는 번역이고, 순서도 어휘도 다르다. 요약도 마찬가지다 — 같은 기사를 세 명이 요약하면 세 개의 서로 다른, 그러나 모두 타당한 요약이 나온다.

그래서 생성 평가 지표들은 전부 같은 질문에 서로 다른 답을 내놓는 시도였다. **"정답이 여러 개일 때, 후보 출력이 그 중 하나와 얼마나 가까운가를 어떻게 재는가?"** 이 글은 그 계보를 훑는다 — n-gram이 겹치는 정도를 세던 BLEU에서, 사람 판정으로 직접 학습시킨 COMET까지. 다만 이 시리즈의 정체성대로, "무엇을 하는 지표인가"보다 **"각 지표가 어떤 가정을 깔고 있고, 그 가정이 어디서 깨졌는가"**를 축으로 삼는다.

결론을 먼저 제시한다. n-gram 겹침 지표(BLEU, ROUGE)는 "표층 형태가 비슷하면 의미도 비슷하다"는 가정에 기대는데, 이 가정은 동의어·의역·어순 변화 앞에서 쉽게 깨진다. Callison-Burch et al. (2006)은 이 균열을 실제 시스템 비교에서 잡아냈고, Reiter (2018)는 284개의 상관계수를 모아 "BLEU는 MT 시스템 단위 비교에서만 어느 정도 타당하다"는 좁은 결론을 냈다. 학습된 지표(BERTScore, BLEURT, COMET)는 의미 유사도를 직접 학습해 이 문제를 상당 부분 완화했지만, 그 대가로 **지표 자체가 블랙박스가 되었고, 적대적 공격에 뚫리며, 학습 도메인에 갇힌다.** 그리고 재현성 문제(Post, 2018)는 지표가 "숫자 하나"가 아니라 "설정에 따라 달라지는 숫자들의 집합"이라는 것을 드러낸다. 이 모든 균열의 종착점에서 등장하는 것이 참조 문장 자체가 필요 없는, 혹은 사람 대신 또 다른 LLM이 판정하는 접근이다 — [#9](/blog/2026/mt-bench-to-arena/)에서 다룰 LLM-as-a-Judge다.

# Background

## 생성 평가의 근본 문제 — 정답이 하나가 아니다

측정론적으로 말하면, 생성 품질이라는 구성개념(construct, [#1](/blog/2026/what-is-evaluation/) 참고)을 조작화(operationalize)하는 방법은 크게 두 갈래다.

1. **참조와의 유사도로 근사한다.** 사람이 미리 써 둔 정답(참조, reference)과 모델 출력이 얼마나 가까운지를 잰다. 이 접근의 근본 가정은 "참조가 정답 공간을 대표한다"는 것이다. 그런데 참조는 보통 하나 혹은 소수이고, 정답 공간은 사실상 무한하다. 그래서 이 가정은 처음부터 근사에 불과하다.
2. **유사도를 어떻게 잴 것인가.** 참조와 후보가 정해지면, 그 둘을 비교하는 함수가 필요하다. 여기서 다시 두 갈래로 갈린다 — 표층 형태(단어·문자의 겹침)를 잴 것인가, 아니면 의미(임베딩 공간에서의 거리, 혹은 사람 판정에 직접 회귀시킨 점수)를 잴 것인가.

이 글에서 다루는 모든 지표는 이 두 축의 조합이다. BLEU와 ROUGE는 "참조 기반 + 표층 겹침"이다. METEOR와 chrF는 같은 축이지만 겹침의 정의를 조금 더 관대하게(동의어, 문자 단위) 만든 변형이다. BERTScore·BLEURT·COMET은 "참조 기반 + 학습된 의미 유사도"로 축 하나를 이동한다. 그리고 이 글의 끝에서 예고하는 LLM-as-a-Judge는 아예 첫 번째 축("참조가 있어야 한다")까지 버린다.

## 지표를 판정하는 잣대 — 사람과의 상관

지표 자체는 숫자를 뱉을 뿐이다. 그 숫자가 "쓸모 있다"고 말하려면, 사람이 매긴 품질 점수와 상관이 있어야 한다. 표준적인 검증 절차는 이렇다.

1. 여러 시스템(혹은 여러 문장)의 출력을 모은다.
2. 자동 지표로 각각 점수를 매긴다.
3. 사람 평가자가 같은 출력들에 유창성(fluency)·충실도(adequacy)·정보성(informativeness) 등을 채점한다.
4. 두 점수열 사이의 상관(Pearson, Spearman, 또는 Kendall)을 계산한다.

이 상관이 높으면 지표를 "사람 평가의 대리(proxy)"로 믿고 쓸 수 있다. 낮으면 그 지표로 시스템을 고르는 행위 자체가 위험하다. Reiter(2018)는 이것을 임상의학의 "대리 종점(surrogate endpoint)" 개념에 빗댄다 — AIDS 치료제를 "환자가 더 오래 사는가"로 재는 대신 "바이러스 수치가 낮아지는가"로 재는 것과 같은 구조다. 대리 지표는 그것이 진짜 결과와 상관관계가 있다는 **검증 연구**가 있을 때만 의미가 있다. 이 글의 절반은 그 검증 연구들이 무엇을 발견했는지에 관한 것이다.

# Method

## BLEU — 정밀도와 길이의 타협

[BLEU](https://aclanthology.org/P02-1040/)(Papineni et al., IBM, ACL 2002)는 기계번역 평가의 원형이다. 핵심 아이디어는 단순하다 — **후보 번역의 n-gram이 참조 번역에도 등장하는 비율**을 잰다.

### 수정된 n-gram 정밀도

가장 단순한 형태는 유니그램 정밀도, 즉 "후보의 단어 중 참조에도 있는 단어의 비율"이다. 그런데 이 정의는 쉽게 뚫린다. 후보가 그냥 "the the the the the the the"라고 내놓았다고 하자. 참조 문장 어딘가에 "the"가 있다면, naive 정밀도는 $$7/7 = 1.0$$이 되어버린다 — 완전히 무의미한 출력에 만점을 주는 것이다.

Papineni et al.은 이 문제를 **클리핑(clipping)**으로 막는다. n-gram 하나가 후보에서 몇 번 등장하든, 그 n-gram이 참조에서 등장하는 최대 횟수를 넘어서는 카운트는 인정하지 않는다.

$$
Count_{clip}(w) = \min\big(Count_{cand}(w),\ \max_{Ref} Count_{Ref}(w)\big)
$$

- $$Count_{cand}(w)$$: n-gram $$w$$가 후보 문장에 등장한 횟수
- $$\max_{Ref} Count_{Ref}(w)$$: $$w$$가 (여러 참조 중) 어느 한 참조에 등장한 최대 횟수
- 둘 중 작은 값만 인정한다 — "the"가 참조에 두 번밖에 없다면, 후보가 그 단어를 일곱 번 반복해도 두 번만 쳐준다.

이 클리핑을 코퍼스 전체에 대해 합산한 것이 **수정된 n-gram 정밀도**다.

$$
p_n = \frac{\sum_{S \in C} \sum_{ngram \in S} Count_{clip}(ngram)}{\sum_{S \in C} \sum_{ngram \in S} Count(ngram)}
$$

- $$C$$: 후보 문장들의 코퍼스 전체
- 분자: 클리핑된 매치 수를 모든 문장, 모든 n-gram에 대해 더한 값
- 분모: 후보 n-gram의 (클리핑 없는) 총 개수

앞의 "the" 예시를 다시 계산하면, $$Count_{clip}(\text{the}) = \min(7, 2) = 2$$이므로 $$p_1 = 2/7 \approx 0.286$$이 된다 — 훨씬 합리적인 점수다. 보통 $$n=1,2,3,4$$까지 계산하고, 이 넷을 가중 기하평균으로 합친다.

### 브레비티 페널티

정밀도만으로는 여전히 구멍이 있다. 후보가 짧을수록 정밀도는 유리해진다 — 확실히 맞는 단어 두 개만 내놓으면 정밀도는 쉽게 1.0에 가까워진다. 그런데 BLEU는 재현율(recall)을 직접 계산하지 않는다(참조가 여러 개일 수 있어 재현율의 분모를 정의하기 애매하기 때문이다). 대신 **브레비티 페널티(brevity penalty, BP)**로 우회한다.

$$
BP = \begin{cases} 1 & \text{if } c > r \\ e^{1-r/c} & \text{if } c \le r \end{cases}
$$

- $$c$$: 후보 코퍼스 전체 길이(토큰 수)
- $$r$$: 유효 참조 코퍼스 길이(각 문장마다 후보 길이에 가장 가까운 참조를 골라 합산한 값)

이 함수의 방향을 직접 확인해보자. $$c$$가 $$r$$보다 작아질수록(후보가 짧아질수록) $$r/c$$는 커지고, $$1 - r/c$$는 더 음수가 되며, $$BP = e^{1-r/c}$$는 0에 가까워진다 — 짧은 번역일수록 더 세게 벌점을 받는다. 반대로 $$c = r$$이면 $$BP = e^0 = 1$$로 페널티가 없다. $$c > r$$(후보가 더 길면)이면 페널티를 아예 주지 않는다 — 이미 정밀도가 긴 문장에 불리하게 작동하므로 별도 페널티가 필요 없다는 논리다.

### 최종 점수와 왜 기하평균인가

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

- $$w_n$$: n-gram 차수별 가중치, 보통 $$N=4$$까지 균등하게 $$w_n = 1/4$$
- $$\exp(\sum w_n \log p_n)$$는 $$p_n$$들의 가중 **기하평균**과 같다. 산술평균이 아니라 기하평균을 쓰는 이유는, 어느 한 $$p_n$$이라도 0에 가까우면(고차 n-gram이 하나도 안 맞으면) 전체 점수를 강하게 끌어내리기 위해서다 — 산술평균이면 $$p_1$$이 높은 것만으로 낮은 $$p_4$$를 희석해버릴 수 있다. 기하평균은 곱셈적이라 "모든 차수에서 골고루 맞아야" 높은 점수가 나온다.

**BLEU의 암묵적 가정**은 두 가지로 요약된다. 첫째, **참조가 정답 공간을 대표한다** — 참조와 다른 표현은 곧 틀린 표현으로 취급된다(동의어·의역을 반영할 방법이 없다). 둘째, **표층 겹침이 품질에 비례한다** — n-gram이 많이 겹칠수록 더 좋은 번역이라고 본다. 이 두 가정이 어디서 깨지는지는 뒤에서 Callison-Burch et al.의 실측으로 확인한다.

## ROUGE — recall 지향과 LCS

[ROUGE](https://aclanthology.org/W04-1013/)(Lin, ISI, 2004)는 요약(summarization) 평가를 위해 만들어졌다. BLEU와 뼈대는 같다(n-gram 겹침) — 하지만 방향이 다르다. **BLEU는 정밀도(precision) 지향, ROUGE는 재현율(recall) 지향**이다.

이유는 과제의 성격 차이에서 나온다. 번역은 "후보가 참조에 없는 말을 지어내지 않았는가"(정밀도)가 중요하다. 요약은 "참조 요약이 담고 있는 핵심 내용을 후보가 놓치지 않았는가"(재현율)가 중요하다. 당시 DUC(Document Understanding Conference) 평가에서는 후보 요약의 길이를 어느 정도 통제했기 때문에, 정밀도보다 재현율이 더 변별력 있는 신호였다.

### ROUGE-N

$$
\text{ROUGE-N} = \frac{\sum_{S \in Ref} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in Ref} \sum_{gram_n \in S} Count(gram_n)}
$$

- 분모가 **참조** 쪽 n-gram 총수라는 점이 BLEU의 $$p_n$$과 정반대다. BLEU는 분모가 후보(candidate) n-gram 수였다 — 즉 "후보가 낸 말 중 몇 개가 맞았는가"를 물었다. ROUGE-N은 "참조가 담은 내용 중 몇 개를 후보가 회수했는가"를 묻는다.

### ROUGE-L — LCS 기반

ROUGE-L은 n-gram 대신 **최장 공통 부분수열(Longest Common Subsequence, LCS)**을 쓴다. LCS는 두 문장에서 순서를 지키되 연속하지 않아도 되는 가장 긴 공통 단어열이다.

$$
R_{lcs} = \frac{LCS(H,R)}{len(R)}, \qquad P_{lcs} = \frac{LCS(H,R)}{len(H)}
$$

$$
F_{lcs} = \frac{(1+\beta^2)\, R_{lcs}\, P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}
$$

- $$H$$: 후보(hypothesis), $$R$$: 참조
- $$R_{lcs}$$: LCS 길이를 참조 길이로 나눈 값(재현율), $$P_{lcs}$$: LCS 길이를 후보 길이로 나눈 값(정밀도)
- $$\beta$$: 재현율에 얼마나 가중치를 줄지 정하는 값. ROUGE는 요약 평가라는 목적에 맞게 보통 $$\beta$$를 크게 잡아 재현율 쪽으로 기울인다.
- LCS는 굳이 연속된 n-gram이 아니어도 "같은 순서로 등장하는 단어들"을 잡아내므로, 고정된 n-gram 길이(BLEU의 $$n=1,2,3,4$$)보다 문장 구조의 변화에 조금 더 관대하다.

## METEOR와 chrF — 형태론을 배려한 대안

BLEU와 ROUGE는 표층 형태가 정확히 일치해야만 매치로 인정한다. [METEOR](https://aclanthology.org/W05-0909/)(Banerjee & Lavie, CMU, ACL Workshop 2005)와 [chrF](https://aclanthology.org/W15-3049/)(Popović, WMT 2015)는 이 엄격함을 서로 다른 방식으로 완화한다.

### METEOR — 동의어·어간까지 매칭한다

METEOR는 유니그램 정렬(alignment)을 만들되, 정확히 같은 단어뿐 아니라 **어간(stem)이 같은 단어, WordNet 동의어**까지 단계적으로 매칭을 허용한다. 정렬이 끝나면 정밀도 $$P$$와 재현율 $$R$$을 계산하고, 재현율에 아홉 배 가중치를 준 조화평균을 쓴다.

$$
F_{mean} = \frac{10\, P\, R}{R + 9P}
$$

여기에 "매칭된 단어들이 참조와 얼마나 같은 순서로 놓여 있는가"를 보는 단편화(fragmentation) 페널티를 곱해 최종 점수를 만든다. 순서가 심하게 뒤섞여 있으면 개별 단어는 다 맞아도 점수가 깎인다. METEOR의 강점은 "will not attend"와 "would boycott"처럼 표층은 다르지만 의미가 겹치는 표현을 일부 잡아낸다는 것이다. 약점은 이 매칭이 WordNet 같은 외부 언어 자원에 의존한다는 점 — 그 자원이 부실한 언어에서는 METEOR도 BLEU와 다를 바 없어진다.

### chrF — 문자 단위로 내려가면 왜 한국어에 유리한가

chrF는 아예 단어 단위를 포기하고 **문자 n-gram**(보통 1~6그램)의 정밀도·재현율을 잰다.

$$
chrF_\beta = \frac{(1+\beta^2)\cdot chrP \cdot chrR}{\beta^2 \cdot chrP + chrR}
$$

- $$chrP$$: 문자 n-gram 정밀도, $$chrR$$: 문자 n-gram 재현율
- 원 논문은 재현율에 세 배 가중치를 준 $$\beta=3$$(chrF3)을 기본값으로 제안했다.

왜 이것이 한국어처럼 형태론이 풍부한(교착어) 언어에 유리한가. 한국어는 어간에 조사·어미가 붙어 단어 형태가 계속 바뀐다 — "책을 읽었다"와 "책을 읽는다"는 어간 "읽"은 같지만 어미가 달라 완전히 다른 토큰이 된다. 단어 단위 n-gram 지표(BLEU, ROUGE)는 이 둘을 전혀 다른 단어로 취급해 매치를 놓친다. 반면 문자 n-gram은 "읽었"과 "읽는"이 공유하는 "읽" 부분 문자열을 여전히 겹침으로 잡아낸다. 즉 형태소 경계를 알 필요 없이, 어미 변화가 있어도 부분적인 겹침 신호가 살아남는다. 또한 chrF는 토큰화 방식 자체에 의존하지 않는다는 부수적 장점이 있다 — 이는 뒤에서 다룰 재현성 문제(토크나이저 차이로 점수가 달라지는 문제)에서 자유롭다는 뜻이기도 하다.

## 지표 타당도 비판 — 상관은 어디서 깨지는가

지금까지의 지표들은 전부 "표층 겹침이 품질을 대변한다"는 가정 위에 서 있다. 이 절은 그 가정을 실측으로 검증한 세 편의 연구를 다룬다. 세 논문 모두 결론의 결이 다르다 — Callison-Burch et al.은 "구체적 반례"를, Reiter는 "체계적 리뷰"를, Novikova et al.은 "NLG로 확장했을 때의 붕괴"를 보여준다.

### Callison-Burch et al. (2006) — BLEU를 다시 본다

[Re-evaluating the Role of BLEU in Machine Translation Research](https://aclanthology.org/E06-1032/)(Callison-Burch, Osborne & Koehn, University of Edinburgh, EACL 2006)는 "BLEU 점수 개선이 실제 번역 품질 개선과 어긋나는" 두 가지 실측 사례를 제시한다.

**사례 1 — 2005 NIST MT Eval.** 아랍어→영어 번역 시스템 평가에서, **사람 평가 1위 시스템이 BLEU로는 6위**였다. 저자들이 일곱 개 참가 시스템의 평균 사람 점수와 BLEU 점수를 산점도로 그려보니, 이상치(outlier) 하나가 전체 상관을 망가뜨리고 있었다 — 그 이상치를 포함하면 충실도(adequacy) 상관 $$R^2 = 0.14$$, 유창성(fluency) 상관은 $$R^2 = 0.002$$까지 떨어진다. 그런데 이 이상치를 제외하면 각각 $$R^2 = 0.87$$, $$R^2 = 0.742$$로 급격히 올라간다. 문제의 이상치는 완전 자동 번역이 아니라 **사람이 후보정한(post-edited) 시스템**이었다 — 나머지 여섯 개는 모두 같은 병렬 코퍼스로 학습한 구문 기반 통계적 기계번역(phrase-based SMT) 시스템이었다. 저자들은 이 표를 통해 하나의 논문 속 사례를 구체적으로 제시한다 — 같은 정보를 담은 두 후보 번역의 n-gram 매치 수는 비슷했지만(27유니그램/20바이그램/15트라이그램/10 4그램 대 24/19/15/12), 사람 점수는 크게 갈렸다(충실도 3,2 대 5,4 / 유창성 3,2 대 5,4). 차이의 원인은 동의어였다 — 낮은 점수를 받은 문장은 "would boycott"를 "will not attend"로, "meddling"을 "interfering"으로 바꿔 쓴 문장을 제대로 인정받지 못했다.

**사례 2 — 규칙 기반 대 통계 기반.** 저자들은 직접 실험을 설계했다. 규칙 기반 번역기 Systran과, Europarl 코퍼스로 학습한 두 개의 통계적 기계번역 시스템을 프랑스어→영어 300문장에 대해 세 명의 평가자가 유창성·충실도를 채점하게 했다. 그림상 Systran의 BLEU 점수는 세 시스템 중 **가장 낮았지만**(0.2 안팎), 사람 평가에서는 유창성·충실도 모두 **가장 높았다**. 반대로 훈련 데이터의 $$1/64$$만으로 학습한 SMT 시스템은 Systran보다 BLEU가 살짝 높았지만 사람 평가는 확연히 낮았다. 결론은 분명하다 — **BLEU는 서로 다른 번역 전략(규칙 기반 대 통계 기반)을 쓰는 시스템 사이에서는 비교 근거로 쓰기 위험하다.**

이 논문은 또 하나 흥미로운 이론적 관찰을 남긴다. BLEU가 허용하는 "동점 후보"의 수가 어마어마하게 많다는 것이다. 바이그램 불일치 지점을 기준으로 문장을 쪼개 순서를 섞어도 BLEU 점수는 그대로다 — 저자들이 예시로 든 문장 하나는 최소 40,320가지의 순열이 같은 점수를 받았고, 2005 NIST 평가의 어떤 문장은 가능한 순열이 $$10^{73}$$가지를 넘었다. 이렇게 많은 동점 후보가 전부 사람 눈에 동등한 품질일 리 없다는 것이 저자들의 핵심 논증이다.

### Reiter (2018) — 284개 상관계수의 구조적 리뷰

[A Structured Review of the Validity of BLEU](https://aclanthology.org/J18-3002/)(Reiter, University of Aberdeen, Computational Linguistics 2018)는 개별 반례가 아니라 **체계적 문헌 리뷰**로 접근한다. ACL Anthology를 정해진 검색어로 훑어 34개 논문에서 **284개의 BLEU–사람 상관계수**를 모았다.

Reiter는 임상시험의 대리 종점 평가 기준을 빌려와 상관 강도를 네 단계로 분류한다.

| 등급     | 상관계수 범위 |
| -------- | ------------- |
| High     | 0.85 이상     |
| Medium   | 0.70 \~ 0.85  |
| Low      | 0 \~ 0.70     |
| Negative | 0 미만        |

수집된 상관계수를 이 기준으로 나눠보면 패턴이 뚜렷하다. **MT 시스템 단위(system-level) 상관은 대체로 Medium\~High**였지만, **MT 문장 단위(text-level) 상관은 낮았고, NLG(자연어 생성 전반) 상관은 시스템 단위·문장 단위 가릴 것 없이 낮았다.** 즉 BLEU–사람 상관이 그나마 신뢰할 만한 유일한 조합은 "기계번역 + 시스템 단위 비교"뿐이었다.

WMT 지표 공유 과제(2007\~2016) 10개 논문만 따로 보면 더 흥미롭다. 같은 언어쌍(독일어↔영어), 같은 도메인(뉴스)에서 매년 반복된 평가인데도 상관계수가 해마다 크게 요동쳤다 — 독일어→영어는 0.12(WMT08)부터 0.90(WMT13)까지, 영어→독일어는 -0.43(WMT09, 음의 상관)부터 0.83까지 널뛰었다. 조건을 이렇게 통제해도 상관이 안정되지 않는다는 것은, BLEU의 타당도가 **평가 대상 시스템·정확한 텍스트·평가 프로토콜의 세부 사항에 강하게 의존**한다는 뜻이다. 다음 새로운 상황에서 BLEU가 사람 평가와 상관이 있을지 예측할 방법이 없다.

Reiter의 최종 결론은 원문 그대로 옮길 가치가 있다: **"증거는 BLEU를 MT 시스템의 진단적 평가(diagnostic evaluation)에 쓰는 것은 지지하지만, MT 바깥의 다른 NLP 시스템 평가, 개별 텍스트 평가, 과학적 가설 검증에 쓰는 것은 지지하지 않는다."** 특히 마지막 지점이 뼈아프다 — 학회 논문에서 "우리 방법이 BLEU를 몇 점 올렸다"는 주장을, 이 리뷰는 **과학적 증거로 쓰기에 부적절하다**고 지적한다. 이유는 세 가지다. 상관의 편차가 비슷한 과제에서도 크고, 검증 연구에 쓰인 사람 평가 자체가 실제 배포 환경의 결과가 아니라 인위적 실험실 평가였으며, 신경망 기반 시스템에 대해 BLEU가 공정한지조차 알려진 바가 없다.

### Novikova et al. (2017) — NLG에는 새 지표가 필요하다

[Why We Need New Evaluation Metrics for NLG](https://aclanthology.org/D17-1238/)(Novikova, Dušek, Cercas Curry & Rieser, Heriot-Watt University, EMNLP 2017)는 Reiter의 리뷰가 시사한 "NLG에서는 상관이 나쁘다"는 결론을 직접 실측으로 확인한다.

저자들은 대화 시스템 발화 생성 과제에서 세 개의 데이터셋(BAGEL, SFHOTEL, SFREST)과 세 개의 종단간(end-to-end) NLG 시스템(RNNLG, TGen, LOLS)을 조합해 **총 2,460개의 시스템 출력**을 모았다. 여기에 BLEU·ROUGE·METEOR·CIDEr를 포함한 **21개의 자동 지표**를 계산하고, 크라우드워커 세 명이 각 출력에 정보성(informativeness)·자연스러움(naturalness)·품질(quality)을 6점 리커트 척도로 채점하게 했다(평가자 간 신뢰도 ICC $$=0.45$$, 중간 수준의 일치도).

결과는 단호했다. **"데이터셋, 시스템, 사람 평가의 어떤 측면을 보더라도, 어떤 지표도 사람 평가와 중간 이상의 상관을 보이지 않았다."** 비교를 위해 무작위로 생성한 가짜 점수를 기준선으로 넣었는데, 이 무작위 점수조차 사람 평가와 거의 상관이 없었다(최고 $$\rho = 0.09$$) — 그런데 실제 자동 지표들의 상관도 이 무작위 기준선과 별로 멀지 않았다. 상대적 순위를 맞히는 과제(두 출력 중 어느 쪽이 사람 평가에서 더 높은가)로 완화해도 지표들의 정확도는 30.6\~49.8%였고, 무작위 기준선은 25.4\~44.5%였다 — 지표가 무작위보다 나은 것은 맞지만 그 차이가 크지 않다.

Reiter의 리뷰와 마찬가지로, Novikova et al.도 **시스템 단위에서는 지표가 어느 정도 쓸모 있다**(성능이 눈에 띄게 나쁜 시스템을 걸러내는 데는 유효하다)는 점은 인정한다. 문제는 문장 단위, 그리고 "중간 정도로 괜찮은" 출력들 사이의 미세한 차이다 — 지표는 명백히 나쁜 출력은 잡아내지만, 중간·상 품질 출력들 사이의 순위는 사람과 거의 무관하게 매긴다.

## 학습된 지표 — 표층에서 의미로

세 논문의 공통된 처방은 같다. **표층 형태 대신 의미를 직접 비교해야 한다.** 2019년 이후 등장한 지표들은 사전학습된 언어모델의 임베딩을 빌려 이 처방을 구현한다.

### BERTScore (Zhang et al., ICLR 2020)

[BERTScore](https://arxiv.org/abs/1904.09675)(Zhang, Kishore, Wu, Weinberger & Artzi, Cornell/ASAPP, ICLR 2020)는 발상이 단순하다 — 정확히 같은 단어를 세는 대신, **BERT가 만든 문맥 임베딩끼리 코사인 유사도**를 재고, 각 토큰을 상대 문장에서 가장 유사한 토큰과 그리디(greedy)하게 매칭한다.

$$
R_{BERT} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} x_i^\top \hat{x}_j, \qquad P_{BERT} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_j \in \hat{x}} \max_{x_i \in x} x_i^\top \hat{x}_j
$$

$$
F_{BERT} = 2\,\frac{P_{BERT} \cdot R_{BERT}}{P_{BERT} + R_{BERT}}
$$

- $$x$$: 참조 문장의 토큰열, $$\hat{x}$$: 후보 문장의 토큰열 (사전정규화된 임베딩이라 코사인 유사도가 내적 $$x_i^\top \hat{x}_j$$과 같아진다)
- $$R_{BERT}$$: 참조의 각 토큰이 후보에서 가장 비슷한 토큰과 얼마나 가까운지(재현율)
- $$P_{BERT}$$: 후보의 각 토큰이 참조에서 가장 비슷한 토큰과 얼마나 가까운지(정밀도)
- 여기에 선택적으로 역문서빈도(idf) 가중을 더한다.

$$
idf(w) = -\log \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}[w \in x^{(i)}]
$$

흔한 단어(관사, 조사)보다 희귀한 단어(내용어)에 더 큰 가중치를 준다는 뜻이다 — BLEU가 모든 단어를 동등하게 취급하는 것과 대비된다. 저자들이 논문 서두에서 든 예시가 이 지표의 존재 이유를 잘 설명한다. BLEU와 METEOR는 참조 "people like foreign cars"에 대해 "people like visiting places abroad"(의미가 다름)에 "consumers prefer imported cars"(동의어로 재구성된 정답)보다 **더 높은 점수**를 준다 — 표층 겹침만 보기 때문이다. BERTScore는 문맥 임베딩으로 이 오류를 상당 부분 바로잡는다.

### BLEURT (Sellam et al., ACL 2020)

[BLEURT](https://aclanthology.org/2020.acl-main.704/)(Sellam, Das & Parikh, Google Research, ACL 2020)는 한 걸음 더 나아간다 — 유사도를 계산하는 대신, **사람이 매긴 품질 점수 자체에 회귀(regression)**시킨다. BERT의 `[CLS]` 벡터 위에 선형층을 얹어 점수를 예측한다.

$$
\hat{y} = f(x, \tilde{x}) = W v_{[CLS]} + b
$$

- $$x$$: 참조, $$\tilde{x}$$: 후보, $$v_{[CLS]}$$: BERT가 만든 `[CLS]` 표현
- 지도학습 손실은 단순한 회귀 손실이다: $$\ell_{supervised} = \frac{1}{N}\sum_{n=1}^{N} \lVert y_n - \hat{y}_n \rVert^2$$

문제는 사람이 매긴 학습 데이터가 겨우 수천 개 수준이라는 것이다. 이 정도로는 BERT를 안정적으로 파인튜닝하기 어렵다. BLEURT의 핵심 기여는 **본 학습 전에 합성 데이터로 워밍업**시키는 프리트레이닝 단계다. 위키피디아 문장 180만 개를 마스크 채우기(mask-filling)·역번역(backtranslation)·단어 삭제로 변형해 650만 개의 합성 문장쌍을 만들고, 여기에 BLEU·ROUGE·BERTScore 점수, 역번역 가능성, 텍스트 함의(entailment) 여부 등 **9가지 신호**를 다중 과제 손실로 함께 학습시킨다. 즉 사람 라벨이 부족한 문제를, "사람 라벨과 상관관계가 있을 법한 자동 신호들"을 대량으로 미리 학습해 완화하는 전략이다. WMT17\~19 지표 공유 과제에서 이 사전학습 유무를 비교한 결과, 사전학습을 거친 BLEURT가 대부분의 언어쌍에서 Kendall 상관이 더 높았고 특히 학습 데이터가 적거나 분포가 다른(domain-shifted) 상황에서 격차가 컸다.

### COMET (Rei et al., EMNLP 2020)

[COMET](https://aclanthology.org/2020.emnlp-main.213/)(Rei, Stewart, Farinha & Lavie, Unbabel AI, EMNLP 2020)은 BERTScore·BLEURT가 놓친 한 가지를 더한다 — **원문(source)까지 함께 본다.** 지금까지의 모든 지표는 "후보가 참조와 얼마나 비슷한가"만 봤다. COMET은 "후보가 원문의 의미를 얼마나 보존했는가"까지 신호로 쓴다.

구조는 이렇다. 다국어 사전학습 인코더(XLM-RoBERTa)로 원문 $$s$$, 후보 $$h$$, 참조 $$r$$을 각각 독립적으로 인코딩해 문장 임베딩을 얻는다. 이 세 임베딩을 조합해 하나의 벡터로 만든다.

$$
x = [\,h;\ r;\ h \odot r;\ \lvert h - r \rvert\,]
$$

- $$h \odot r$$: 원소별 곱(element-wise product) — 두 벡터가 같은 방향으로 큰 값을 가지는지
- $$\lvert h-r \rvert$$: 원소별 절대 차이 — 두 벡터가 벡터 공간에서 얼마나 떨어져 있는지
- 이 결합 벡터를 순전파(feed-forward) 회귀기에 넣어 사람 판정(DA, HTER, MQM 등)에 회귀시킨다.

흥미로운 점은, 저자들이 원문 임베딩 $$s$$ 자체를 이 결합 벡터에 직접 넣는 실험도 했지만 효과가 미미해 최종 모델에서는 뺐다는 것이다(대신 원문은 학습 데이터 구성 단계와 랭킹 모델 쪽에서 활용한다). WMT19 지표 공유 과제 결과가 그 효과를 보여준다. 영어→X 여덟 개 언어쌍의 Kendall 상관에서, COMET 계열 모델은 BLEU를 큰 폭으로 앞섰다 — 예컨대 영어→독일어에서 BLEU $$\tau = 0.248$$인 반면 COMET-RANK는 $$\tau = 0.427$$이었고, 여덟 개 언어쌍 중 일곱 개에서 COMET-RANK가 비교 대상 지표(BLEU, chrF, YiSi-1, BERTScore) 전부를 앞질렀다.

### 학습된 지표의 새 문제

의미를 직접 학습한 대가로, 이 지표들은 n-gram 지표에는 없던 새로운 약점을 얻는다.

- **적대적 취약성.** 지표 자체가 신경망이므로, 신경망의 일반적인 약점을 그대로 물려받는다. [Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks](https://arxiv.org/abs/2311.00508)(Huang & Baldwin, Findings of EMNLP 2023)는 BERTScore·BLEURT·COMET에 단어·문자 수준의 적대적 변형을 가했다. 그 결과 이 지표들은 사람이라면 크게 벌점을 주지 않을 미세한 변형에도 과도하게 벌점을 주거나, 반대로 의미가 훼손된 문장에 비일관적인 점수를 매기는 사례가 확인됐다 — 지표를 최적화 목표로 직접 쓸 경우([reward hacking](/blog/2026/reward-model-design/)과 같은 구조의 문제가) 발생할 수 있다는 뜻이다.
- **학습 도메인 종속.** BLEURT의 사전학습이 위키피디아·WMT 뉴스 도메인의 변형 패턴에 맞춰져 있듯, 학습된 지표는 학습 데이터의 언어·도메인·시대에 갇힌다. n-gram 지표는 규칙 기반이라 이런 종속이 없다(대신 의미를 전혀 못 본다는 다른 대가를 치른다).
- **해석 불가능.** BLEU의 $$p_n=0.6$$은 "n-gram의 60%가 겹쳤다"는 뜻이 분명하다. BERTScore의 코사인 유사도 0.92, BLEURT의 회귀값 0.7은 그 자체로 사람에게 뭘 의미하는지 설명하기 어렵다. 점수가 왜 그렇게 나왔는지 되짚어볼 방법이 마땅치 않다.

## 재현성 문제 — 같은 이름, 다른 숫자

지금까지는 "지표가 사람과 얼마나 상관 있는가"를 물었다. 이 절의 문제는 다르다 — **같은 지표, 같은 이름인데도 숫자가 다르게 나오는** 문제다.

[A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/)(Post, Amazon Research, WMT 2018)는 BLEU가 "하나의 지표가 아니라 여러 파라미터로 구성된 하나의 계열"이라는 점을 지적한다. 참조 개수, 최대 n-gram 길이, 0-카운트에 대한 스무딩(smoothing), 그리고 무엇보다 **토큰화·정규화 방식**이 전부 결과에 영향을 준다. 그런데 논문들은 이 설정을 잘 밝히지 않는다.

저자는 WMT 2017의 한 시스템(online-B) 출력을 고정한 채, 참조 문장의 전처리 방식만 네 가지로(사용자 지정 토큰화(basic), 복합어 분리(split), 미등록어 처리(unk), WMT 공식 지표 내부 토큰화(metric)) 바꿔 BLEU를 다시 계산했다. **같은 시스템 출력, 같은 참조 텍스트인데도 처리 방식만 바뀌었을 뿐인데 점수가 최대 1.8점까지 벌어졌다**(언어쌍 평균으로는 약 1.0점). 대소문자 처리(cased/uncased) 여부까지 합치면 그 폭은 더 커진다. 참조 개수도 무시할 수 없는 변수다 — WMT 2017 영어→핀란드어에서 같은 online-B 시스템은 참조 하나로는 BLEU 22.04, 참조 둘로는 25.25를 받는다. 논문 개선폭이 흔히 1점 안팎인 것을 생각하면, 이 정도 변동은 "우리 방법이 더 낫다"는 주장 전체를 무력화할 수 있는 크기다.

Post는 대안으로 **SacreBLEU**를 제안한다 — 참조 전처리를 사용자에게 맡기지 않고 도구 내부에서 표준화해서 처리하고, 재현에 필요한 모든 파라미터를 담은 **서명 문자열**(참조 개수, 대소문자 처리, 토큰화 방식, 스무딩, 버전 정보를 인코딩한 문자열)을 함께 출력하도록 만든 파이썬 패키지다. 논문에 "BLEU 32.1을 얻었다"고만 쓰는 대신, 이 서명까지 같이 적으면 다른 연구자가 정확히 같은 설정을 재현할 수 있다.

이 문제는 이 시리즈에서 다시 만난다. "같은 이름의 벤치마크·지표가 논문마다 다른 숫자를 낸다"는 현상은 오염(contamination)·재현성 문제의 더 일반적인 형태이며, [#20](/blog/2026/contamination-reproducibility/)에서 벤치마크 전반의 재현성 위기로 확장해서 다룬다.

## 평가 방식의 전환 — 참조 기반에서 judge 기반으로

이 글에서 훑은 지표들의 궤적을 한 줄로 요약하면, "참조와의 겹침"에서 "참조와의 의미 유사도"로, 그리고 "원문과의 의미 보존"(COMET이 QE 방향으로 원문을 활용하는 실험)으로 조금씩 참조 문장에 대한 의존을 줄여 온 과정이다.

이 방향이 끝까지 가면 참조 없는(reference-free) 평가에 도달한다. 왜 이 방향으로 갈 수밖에 없는가 — 애초에 참조 자체를 만들기 어렵거나, 참조라는 개념 자체가 성립하지 않는 과제들이 있기 때문이다. 기계번역이나 요약은 그나마 "정답에 가까운 것"을 사람이 미리 써 둘 수 있다. 하지만 **개방형 대화**는 다르다. "오늘 저녁 뭐 먹을까?"라는 질문에 참조 답안이 있을 수 없다 — 좋은 답은 상황과 취향에 따라 무한히 다양하고, 그중 하나를 "정답"으로 못박는 순간 나머지 전부가 부당하게 낮은 점수를 받는다. 이 경우 지표가 참조 문장과 겹침을 재는 방식 자체가 성립하지 않는다.

이 막다른 길에서 나온 답이 **LLM을 판정자(judge)로 쓰는 것**이다. 참조 문장 대신, 강력한 LLM에게 "이 두 응답 중 어느 쪽이 더 나은가" 혹은 "이 응답이 얼마나 좋은가"를 직접 판단시킨다. 참조가 없어도 되고, 사람이 채점 기준을 미리 규칙으로 못박지 않아도 어느 정도 유연하게 판단한다. 물론 이것으로 문제가 끝나는 것은 아니다 — judge 역시 편향(위치, 장황함, 자기선호)을 갖고, judge의 판정이 진짜 사람 판정을 얼마나 대체할 수 있는지가 새로운 검증 대상이 된다. 이 전환이 실제로 어떻게 일어났는지, 그리고 judge 기반 평가가 어떤 문제를 새로 만드는지는 [#9 MT-Bench에서 Arena까지](/blog/2026/mt-bench-to-arena/)에서 자세히 다룬다.

# Experiments

## 토이 예제: BLEU를 손으로 계산한다

참조 문장 하나와 후보 문장 하나를 놓고 BLEU를 처음부터 끝까지 손으로 따라가 본다. 단순화를 위해 $$n=1,2$$까지만, 가중치는 $$w_1=w_2=0.5$$로 균등하게 쓴다.

- **참조(Reference)**: "the cat sat on the mat" (토큰 6개: the, cat, sat, on, the, mat — "the"가 두 번 등장)
- **후보(Candidate A)**: "the cat is sitting on the mat" (토큰 7개: the, cat, is, sitting, on, the, mat)

**1단계 — 유니그램 정밀도 $$p_1$$.** 참조의 유니그램 카운트는 the:2, cat:1, sat:1, on:1, mat:1이다. 후보 A의 유니그램 카운트는 the:2, cat:1, is:1, sitting:1, on:1, mat:1이다. 각 후보 유니그램의 클리핑된 매치 수를 구하면 다음과 같다.

| 후보 유니그램 | 후보 카운트 | 참조 카운트 | 클리핑된 매치 |
| ------------- | ----------- | ----------- | ------------- |
| the           | 2           | 2           | 2             |
| cat           | 1           | 1           | 1             |
| is            | 1           | 0           | 0             |
| sitting       | 1           | 0           | 0             |
| on            | 1           | 1           | 1             |
| mat           | 1           | 1           | 1             |

클리핑된 매치의 합은 $$2+1+0+0+1+1=5$$, 후보의 총 유니그램 수는 7이다. 따라서 $$p_1 = 5/7 \approx 0.714$$.

**2단계 — 바이그램 정밀도 $$p_2$$.** 참조의 바이그램은 (the,cat), (cat,sat), (sat,on), (on,the), (the,mat) 다섯 개(모두 서로 다른 타입)다. 후보 A의 바이그램은 (the,cat), (cat,is), (is,sitting), (sitting,on), (on,the), (the,mat) 여섯 개다. 참조와 겹치는 것은 (the,cat), (on,the), (the,mat) 세 개뿐이다. 따라서 $$p_2 = 3/6 = 0.5$$.

**3단계 — 브레비티 페널티.** 후보 길이 $$c=7$$, 참조 길이 $$r=6$$이므로 $$c>r$$이다. 정의에 따라 $$BP=1$$(페널티 없음).

**4단계 — 최종 BLEU.**

$$
\text{BLEU} = 1 \cdot \exp\big(0.5 \log 0.714 + 0.5 \log 0.5\big) = \exp(-0.168 - 0.347) = \exp(-0.515) \approx 0.598
$$

과거형("sat")을 현재진행형("is sitting")으로 바꿔 쓴, 사람이 보기에 완전히 무난한 번역/생성이 BLEU $$0.598$$을 받았다. 이 숫자 자체는 낮지도 높지도 않은 애매한 값이다 — 이것이 다음 반례에서 문제가 된다.

## 핵심 반례: 높은 BLEU·틀린 의미 대 낮은 BLEU·완벽한 의역

같은 참조 "the cat sat on the mat"에 대해 두 개의 다른 후보를 더 만들어 비교한다.

**후보 B (의미가 뒤바뀐 문장)**: "the mat sat on the cat" — "고양이"와 "매트"의 위치를 바꿔치기했다. 매트가 고양이 위에 앉았다는, 원문과 정반대의 터무니없는 문장이다.

유니그램은 참조와 완전히 같은 집합(the, mat, sat, on, cat)이므로 클리핑된 매치 수는 $$2+1+1+1+1=6$$, 총 유니그램 수도 6이다. $$p_1 = 6/6 = 1.0$$ — **완벽한 유니그램 정밀도**다. 바이그램은 (the,mat), (mat,sat), (sat,on), (on,the), (the,cat) 다섯 개 중 (mat,sat)만 참조에 없고 나머지 네 개는 전부 참조와 일치한다. $$p_2 = 4/5 = 0.8$$. $$c=r=6$$이므로 $$BP=1$$.

$$
\text{BLEU}_B = \exp(0.5\log 1.0 + 0.5\log 0.8) = \exp(-0.112) \approx 0.894
$$

**후보 C (완벽한 의역, 그러나 어휘가 전혀 다름)**: "a feline was resting atop the rug" — "고양이"를 "feline"으로, "매트 위에 앉았다"를 "깔개 위에서 쉬고 있었다"로, 뜻은 정확히 같지만 단어를 전부 바꿔 쓴 문장이다.

참조와 겹치는 유니그램은 "the" 하나뿐이다(클리핑 매치 $$=1$$). 총 유니그램 수는 7. $$p_1 = 1/7 \approx 0.143$$. 바이그램은 참조와 단 하나도 겹치지 않는다 — $$p_2 = 0$$.

$$
\text{BLEU}_C = \exp(0.5\log 0.143 + 0.5\log 0) = \exp(-\infty) = 0
$$

세 후보를 나란히 놓으면 이 지표의 문제가 선명해진다.

| 후보                                   | 의미                              | $$p_1$$ | $$p_2$$ | BLEU      |
| -------------------------------------- | --------------------------------- | ------- | ------- | --------- |
| A: "the cat is sitting on the mat"     | 정확 (시제만 변형)                | 0.714   | 0.5     | 0.598     |
| B: "the mat sat on the cat"            | **정반대 (고양이·매트가 뒤바뀜)** | 1.0     | 0.8     | **0.894** |
| C: "a feline was resting atop the rug" | **정확 (완벽한 의역)**            | 0.143   | 0       | **0.0**   |

의미가 완전히 뒤집힌 후보 B가 가장 높은 점수(0.894)를 받고, 의미가 완벽히 보존된 의역 후보 C는 최저점(0.0)을 받는다. 원인은 명확하다 — BLEU는 **단어 집합의 겹침**만 보지, **단어가 문장 내에서 어떤 역할을 하는지(주어인지 목적어인지)나 단어의 의미적 등가성**은 전혀 보지 않는다. 이것이 Callison-Burch et al.과 Zhang et al.(BERTScore)이 실제 시스템 비교에서 반복해서 관찰한 현상의 축소판이다.

## 토크나이저를 바꾸면 점수가 바뀐다

Post(2018)가 보인 실측을 다시 짚는다. 같은 WMT 2017 시스템(online-B)의 같은 출력에 대해, 참조 문장을 처리하는 방식만 바꿔 BLEU를 다시 계산한 결과다(cased 기준, 일부 발췌).

| 언어쌍          | basic 토큰화 | split(복합어 분리) | metric(WMT 공식) | 최대 편차 |
| --------------- | ------------ | ------------------ | ---------------- | --------- |
| 영어→체코어     | 20.7         | 20.7               | 20.1             | 0.6       |
| 영어→라트비아어 | 16.9         | 17.0               | 17.9             | 1.0       |
| 영어→러시아어   | 33.3         | 33.3               | 32.0             | 1.3       |
| 영어→터키어     | 18.5         | 18.7               | 19.9             | 1.4       |
| 독일어→영어     | 31.2         | 31.7               | 33.0             | **1.8**   |

모델도, 학습 데이터도, 심지어 시스템 출력 자체도 전혀 바뀌지 않았다. 오직 참조 텍스트를 어떤 규칙으로 토큰화했는지만 바뀌었을 뿐인데, 영어→러시아어처럼 형태론이 복잡한 언어에서는 점수가 1점 이상 벌어진다. 이것이 chrF처럼 토큰화에 의존하지 않는 지표가 갖는 실질적 이점이다 — 그리고 왜 "BLEU 32.1"이라는 숫자 하나만으로는 논문 간 비교가 원칙적으로 불가능한지를 보여준다.

# 통계 요약

| 지표                                                       | 참조 필요        | 무엇을 재는가                                   | 알려진 실패 모드                                                              | 해석 가능성                                    |
| ---------------------------------------------------------- | ---------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------- |
| BLEU                                                       | 필요(다중 권장)  | 표층 n-gram 정밀도(clipped)                     | 동의어·어순 변화 무시, 토크나이저 의존으로 재현성 낮음, 문장 단위 신뢰 어려움 | 높음(정밀도 값 자체는 명확)                    |
| ROUGE-N / ROUGE-L                                          | 필요             | 표층 n-gram 재현율 / LCS 기반 재현율            | ROUGE-N과 유사, 요약 길이에 편향                                              | 높음                                           |
| METEOR                                                     | 필요             | 유니그램 정밀·재현 + 동의어·어간 매칭           | 외부 언어 자원(WordNet) 의존, 저자원 언어에 불리                              | 중간                                           |
| chrF                                                       | 필요             | 문자 n-gram 정밀·재현                           | 문자 겹침이 의미 변화를 못 잡음, 형태소 경계 무시                             | 높음(토큰화 불필요, 재현성 좋음)               |
| BERTScore                                                  | 필요             | 문맥 임베딩 코사인 유사도(greedy matching)      | 사전학습 도메인·언어 의존, 적대적 변형에 취약                                 | 낮음(유사도 값의 절대적 의미 불명확)           |
| BLEURT                                                     | 필요             | 사람 판정에 회귀 학습된 유사도                  | 학습 도메인·시대에 종속, 합성 프리트레이닝 품질에 민감, 블랙박스              | 낮음                                           |
| COMET                                                      | 필요(+원문 활용) | 원문·후보·참조를 함께 인코딩해 학습한 사람 선호 | 도메인 종속, 해석 불가, MQM 등 채점 체계 정의에 의존                          | 낮음                                           |
| LLM-as-a-Judge (예고, [#9](/blog/2026/mt-bench-to-arena/)) | 불필요 가능      | LLM이 직접 내린 판단(pairwise 혹은 절대 점수)   | 위치·장황함·자기선호 편향                                                     | 매우 낮음(설명은 있으나 신뢰성 별도 검증 필요) |

# Conclusion

이 글이 훑은 계보는 결국 하나의 질문을 계속 다르게 되묻는 과정이었다 — **"정답이 하나가 아닐 때, 후보 출력이 얼마나 좋은지를 어떻게 숫자로 만들 것인가?"** BLEU와 ROUGE는 "표층이 겹치면 의미도 겹친다"고 가정했고, 이 가정은 Callison-Burch et al.(2006)의 Systran 실험과 동의어 사례, Reiter(2018)의 284개 상관계수, Novikova et al.(2017)의 21개 지표 실측에서 반복적으로 깨졌다. 특히 세 논문이 공통으로 남긴 메시지는 "시스템 단위 비교에서는 그럭저럭 쓸 만하지만, 문장 단위·NLG 전반으로 일반화하면 무너진다"는 것이었다.

BERTScore·BLEURT·COMET은 표층 대신 의미를 직접 학습해 이 문제를 크게 완화했다. 하지만 그 대가로 지표 자체가 신경망이 되면서 적대적 취약성·도메인 종속·해석 불가능성이라는 새 문제를 얻었다. 그리고 Post(2018)가 보인 재현성 문제는, 지표가 완벽하다 해도 "같은 이름으로 다른 숫자를 보고하는" 실무적 함정은 별개로 존재한다는 것을 일깨운다.

이 모든 흐름의 끝에서, 참조 문장이라는 전제 자체가 성립하지 않는 개방형 대화가 등장하면서 LLM을 판정자로 쓰는 접근이 나왔다. 다만 이것이 이야기의 끝은 아니다 — judge 역시 학습된 모델이므로 학습된 지표들이 앓았던 문제(편향, 해석 불가능성)를 새로운 형태로 물려받는다. 그 이야기는 [#9](/blog/2026/mt-bench-to-arena/)에서 이어진다.

# 참고 문헌

- Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/). ACL 2002.
- Lin, C.-Y. (2004). [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/). ACL Workshop 2004.
- Banerjee, S., & Lavie, A. (2005). [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909/). ACL Workshop 2005.
- Popović, M. (2015). [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/). WMT 2015.
- Callison-Burch, C., Osborne, M., & Koehn, P. (2006). [Re-evaluating the Role of BLEU in Machine Translation Research](https://aclanthology.org/E06-1032/). EACL 2006.
- Reiter, E. (2018). [A Structured Review of the Validity of BLEU](https://aclanthology.org/J18-3002/). Computational Linguistics, 44(3).
- Novikova, J., Dušek, O., Cercas Curry, A., & Rieser, V. (2017). [Why We Need New Evaluation Metrics for NLG](https://aclanthology.org/D17-1238/). EMNLP 2017.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675). ICLR 2020.
- Sellam, T., Das, D., & Parikh, A. P. (2020). [BLEURT: Learning Robust Metrics for Text Generation](https://aclanthology.org/2020.acl-main.704/). ACL 2020.
- Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/). EMNLP 2020.
- Post, M. (2018). [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/). WMT 2018.
- Huang, Y., & Baldwin, T. (2023). [Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks](https://arxiv.org/abs/2311.00508). Findings of EMNLP 2023.

---

# LLM 평가 체계 시리즈

이 글은 LLM 평가 체계 시리즈의 다섯 번째 글이다.

**1부. 평가란 무엇인가**

<ol start="1">
  <li><a href="/blog/2026/what-is-evaluation/">측정으로서의 평가</a> — 구성개념·조작화·타당도·신뢰도</li>
  <li><a href="/blog/2026/benchmark-construct-validity/">벤치마크는 무엇을 재고 있나</a> — 벤치 445편 구성타당도 리뷰</li>
</ol>

**2부. 무엇을 숫자로 만드나 — 평가 metric**

<ol start="3">
  <li><a href="/blog/2026/measurement-scales/">척도와 허용 연산</a> — Likert 평균을 내도 되는가</li>
  <li><a href="/blog/2026/classification-metrics/">분류 지표</a> — accuracy의 함정부터 PR-AUC까지</li>
  <li><strong>(현재 글)</strong> 생성 지표와 그 타당도 — BLEU에서 COMET까지</li>
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
  <li><a href="/blog/2026/contamination-reproducibility/">오염·재현성·효율</a> — 오염 검정·harness·IRT</li>
  <li><a href="/blog/2026/safety-evaluation-statistics/">안전 평가의 통계와 체계 설계</a> — 희귀사건·calibration·체크리스트</li>
</ol>

본 시리즈는 21편으로 구성된다.
