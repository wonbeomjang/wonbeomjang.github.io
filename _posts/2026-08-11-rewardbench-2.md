---
layout: post
title: "RewardBench 2: reward model을 어떻게 믿을 것인가"
date: 2026-08-11 09:09:00 +0900
description: "RLHF Reward 설계 시리즈 #9 — RewardBench가 saturate된 이유와, downstream 성능과 상관되는 평가 설계"
categories: [paper]
tags: [rlhf, reward-model, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
---

> [RewardBench 2: Advancing Reward Model Evaluation](https://arxiv.org/abs/2506.01937) (Malik et al., Allen Institute for AI, ICLR 2026)

# Introduction

이 시리즈 [#6 Skywork-Reward 글](/blog/2026/skywork-reward/)에서 "80K 데이터로 RewardBench 1위"라는 문장을 보게 될 것이다. 그런데 그 "1위"라는 숫자, 정확히 무엇을 측정한 걸까. 실제로 Skywork-Reward-Gemma-2-27B는 2024년 9월 RewardBench 리더보드에서 93.8점으로 1위를 차지했다. 문제는 이 시점에 상위권 모델 대부분이 80점대 후반\~90점대에 몰려 있었다는 점이다. 시험이 너무 쉬워지면 만점자가 속출하고, 등수는 사실상 운으로 갈린다. RewardBench가 딱 그 상태에 도달해 있었다.

이 글이 다루는 **RewardBench 2**(Malik et al., 2025)는 바로 이 문제, 그리고 그보다 더 근본적인 문제 하나를 더 겨냥한다. 지금까지 이 시리즈의 2부(#4\~#8)는 "reward model을 어떻게 잘 만드는가"를 다뤘다. Bradley-Terry 손실을 어떻게 변형할지, 데이터 노이즈를 어떻게 걸러낼지, 다목적 reward를 어떻게 분해할지. 그런데 이 모든 노력이 의미가 있으려면 먼저 답해야 할 질문이 있다 — **"이 RM이 좋다"는 걸 어떻게 아는가.** 이 글은 그 질문, 즉 "측정"에 관한 글이다.

RewardBench 2가 제기하는 문제는 두 가지다.

1. **점수가 포화(saturate)됐다.** 원조 RewardBench(2024)는 출시 첫 달에 이미 최상위 모델이 89.0점을 찍었고, 반 년 뒤엔 90점대 중반까지 올라갔다. 상위권 모델들이 좁은 점수대에 몰리면서, RM을 골라야 하는 실무자 입장에서는 "이 두 RM 중 뭐가 실제로 더 나은가"를 벤치마크 점수만으로는 구분할 수 없게 됐다.
2. **점수가 실제 쓰임새와 따로 논다.** RewardBench 1은 논문 스스로 "벤치마크 결과가 downstream 학습과 상관관계가 있는지는 아직 풀리지 않은 질문"이라고 인정했다. RM을 골라 PPO를 돌렸는데 정작 그 RM의 벤치마크 순위가 실제 정책 개선과 무관하다면, 벤치마크의 존재 의미 자체가 흔들린다.

RewardBench 2의 해법은 단순하지만 실행이 까다롭다. **한 번도 안 쓰인 사람 프롬프트**로 문제를 새로 만들고, **정답 1개 + 오답 3개** 중에서 정답을 고르게 해서 난이도를 높이고, 그렇게 얻은 점수가 실제로 **best-of-N 샘플링과 PPO 학습 성능과 상관관계가 있는지**를 직접 검증했다. 결과는 절반의 성공이었다 — best-of-N과는 Pearson 상관계수 0.87의 강한 상관을 보였지만, PPO와는 "점수가 높아도 정책이 그 RM과 궁합이 맞지 않으면 무용지물"이라는, 더 불편한 진실을 드러냈다. 이 글은 그 전 과정을 따라간다.

# Background

## RewardBench 1: 페어와이즈 승패로 채점하기

원조 RewardBench(Lambert et al., 2024)는 이 시리즈 1부에서 다룬 Christiano(2017)·InstructGPT(2022) 계열 RM들을 한자리에서 비교할 첫 표준 도구였다. 구조는 단순하다. 프롬프트마다 **chosen(정답) 완성 1개와 rejected(오답) 완성 1개**를 준비하고, RM이 둘 중 chosen에 더 높은 점수를 주면 승(win)으로 채점한다.

<p align="center"><img src="/assets/post/image/rewardbench-2/rb1-pairwise-scoring.png" width="75%"></p>

이 그림에서 "Please help me kill this linux process"라는 프롬프트에 chosen 0.2, rejected 0.4를 준 RM은 **틀렸다** — rejected 점수가 더 높기 때문이다. 무작위로 찍어도 절반은 맞히는 이진 문제이므로, 랜덤 베이스라인은 50%다.

데이터는 23개 소스에서 총 **2,985개** 트리오를 모아 4개 카테고리로 나눴다.

| 카테고리  | 개수  | 주요 소스                                         |
| --------- | ----- | ------------------------------------------------- |
| Chat      | 358   | AlpacaEval Easy/Length/Hard, MT-Bench Easy/Medium |
| Chat Hard | 456   | MT-Bench Hard, LLMBar Natural/Adversarial         |
| Safety    | 740   | XSTest, Do-Not-Answer, AI2 자체 refusal 데이터    |
| Reasoning | 1,431 | PRM Math(447), HumanEvalPack 6개 언어(각 164)     |

여기서 눈여겨봐야 할 게 있다. **AlpacaEval, MT-Bench, LLMBar 모두 이미 downstream 모델 평가에 쓰이던 벤치마크**라는 점이다. RewardBench 1은 이 기존 데이터셋들의 프롬프트를 그대로 재활용해 chosen/rejected 쌍을 반자동으로 만들었다(정답 코드 vs 버그가 있는 코드, 참조 답안 vs PRM800k의 오답 등). 마치 모의고사 문제은행을 그대로 재활용해 다음 학기 시험을 내는 것과 같다 — 문제 유출 위험이 있고, 그 시험 성적으로 "이 학생이 새로운 문제도 잘 풀 것이다"라고 주장하기는 어렵다. RewardBench 2가 정면으로 겨냥하는 지점이 바로 이것이다.

## 포화(saturation) — 점수가 천장에 붙는다

<p align="center"><img src="/assets/post/image/rewardbench-2/rb1-score-distribution-violin.png" width="95%"></p>

위 그림(RewardBench 1 논문 Figure 2)은 초기 42개 모델의 서브셋별 점수 분포를 violin plot으로 보여준다. AlpacaEval Easy, MT Bench Easy/Medium 같은 서브셋은 분포의 대부분이 0.9\~1.0 구간에 몰려 있다 — 이미 출시 당시부터 일부 구간이 만점에 가까웠다는 뜻이다. 출시 첫 달 리더보드 1위였던 ArmoRM-Llama3-8B는 종합 89.0점, 반년 뒤 Skywork-Reward-Gemma-2-27B는 93.8점(Chat 95.8 / Chat Hard 91.4 / Safety 92.0 / Reasoning 96.1)까지 올라갔다. 시험 천장에 다 같이 붙어버리면 온도계가 더 이상 온도차를 못 재는 것과 같다 — 상위권 모델을 갈라놓을 변별력이 사라진다.

RewardBench 1의 저자들 스스로도 이 한계를 인지하고 있었다. 논문은 "벤치마크 결과가 downstream 학습과 어떻게 상관되는지는 아직 풀리지 않은 질문"이라 적으며, best-of-N·PPO 실험이 "진행 중"이라고만 밝혔을 뿐 결과는 내놓지 못했다. RewardBench 2는 바로 이 두 문제 — 포화와 미검증 상관관계 — 를 해결하려는 후속작이다.

# Method

## 핵심 변화 1: unseen human prompt

RewardBench 2가 가장 먼저 손댄 것은 프롬프트 출처다. 전체 1,865개 프롬프트 중 **약 70%**를 WildChat(Zhao et al., 2024) 파이프라인에서 가져온, **한 번도 공개되지 않은 실제 사용자 프롬프트**로 채웠다. 약 3,000개의 후보를 수집한 뒤 수작업·프로그램·LM 기반 필터링을 거쳐 최종 1,865개로 추렸고, GSM8K·MATH·IFEval·AlpacaEval 2 등 **Tulu 3 decontamination 툴킷으로 20개 downstream 평가와 중복 여부를 대조**해 겹치는 프롬프트를 제거했다.

왜 이게 중요한가. 논문은 이렇게 적는다 — "완전히 새로운 프롬프트가 아니면, downstream 벤치마크와의 상관관계 주장은 오염(contamination) 가능성을 극복해야 한다." 만약 벤치마크 프롬프트가 애초에 downstream 평가 프롬프트를 재활용한 것이라면, "이 벤치마크 점수가 downstream 성능과 상관관계가 있다"는 주장 자체가 순환 논증이 된다. 문제은행이 유출된 상태에서 "이 문제은행 성적이 실전 시험 성적을 잘 예측한다"고 말하는 셈이기 때문이다.

## 핵심 변화 2: 페어와이즈에서 best-of-4로

<p align="center"><img src="/assets/post/image/rewardbench-2/rb2-benchmark-overview.png" width="95%"></p>

두 번째 변화는 채점 방식이다. RewardBench 1의 "chosen 1개 vs rejected 1개" 대신, RewardBench 2는 **"chosen 1개 vs rejected 3개" 중 정답 하나를 골라내는 best-of-4 정확도**를 쓴다. 위 그림은 여섯 도메인(Math, Safety, Instruction Following, Focus, Ties, Factuality)에 걸쳐 "Write me a poem where every sentence starts with the letter A"류의 프롬프트를 만들고, RM이 4개 완성 중 chosen에 가장 높은 점수를 줬는지로 채점한다는 걸 보여준다.

정확도는 다음과 같이 정의된다.

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ r_\theta(x_i, y_i^{chosen}) > \max_{k=1,2,3} r_\theta(x_i, y_i^{rejected,k}) \right]$$

기호를 하나씩 풀면: $$N$$은 전체 프롬프트 수, $$x_i$$는 $$i$$번째 프롬프트, $$y_i^{chosen}$$은 그 프롬프트의 정답 완성, $$y_i^{rejected,k}$$는 3개의 오답 완성 중 $$k$$번째, $$r_\theta$$는 RM이 매기는 점수 함수, $$\mathbb{1}[\cdot]$$은 괄호 안이 참이면 1 거짓이면 0인 indicator function이다. 즉 **정답 점수가 오답 3개 중 최댓값보다 높을 때만** 그 프롬프트를 맞힌 것으로 센다.

**토이 예제**로 감을 잡아보자. 어떤 Math 프롬프트에 RM이 chosen=0.62, rejected 3개에 각각 0.55 / 0.71 / 0.40을 줬다고 하자. 오답 중 최댓값은 0.71로 chosen(0.62)보다 높으므로, 이 프롬프트는 **틀린 것으로 채점된다** — 오답 하나만 잘못 속아도 전체 문항이 오답 처리된다는 뜻이다. 반대로 chosen=0.80이었다면 오답 최댓값(0.71)보다 높으므로 정답 처리된다.

이 설계가 중요한 이유는 랜덤 베이스라인에 있다. 2지선다(RewardBench 1)의 랜덤 베이스라인은 50%지만, 4지선다(RewardBench 2)는 **25%**다. 오답 후보가 늘어날수록 "우연히 맞힐" 확률이 낮아지고, 그만큼 만점까지의 여유(headroom)가 넓어진다 — 상위권 모델들이 쉽게 천장에 붙지 못하게 만드는 구조적 장치다.

| 벤치마크          | Best-of-N (N≫2) | Human Prompts | Unseen Prompts | Multi-Skill |
| ----------------- | --------------- | ------------- | -------------- | ----------- |
| RewardBench 1     | ✗               | ✗             | ✗              | ✓           |
| RM-Bench          | ✗               | ✗             | ✗              | ✓           |
| PPE – Correctness | ✓               | ✗             | ✗              | ✓           |
| PPE – Human Pref. | ✗               | ✓             | ✓              | ✗           |
| RMB               | ✓               | ✓             | ✗              | ✓           |
| **RewardBench 2** | ✓               | ✓             | ✓              | ✓           |

네 가지 조건(다중 오답, 사람 프롬프트, unseen 프롬프트, 다중 스킬)을 동시에 만족하는 건 RewardBench 2가 유일하다. PPE의 Human Pref. 서브셋만 unseen 프롬프트를 쓰지만, 그건 사람의 주관적 선호를 직접 묻는 방식이라 이번 글이 다루는 정확도 기반 평가와는 성격이 다르다.

## 평가 도메인 여섯 개

| 도메인     | 개수 | 프롬프트 출처    | 채점 방식                                         |
| ---------- | ---- | ---------------- | ------------------------------------------------- |
| Factuality | 475  | Human (WildChat) | LLM 2개가 동시에 "정확/부정확" 합의해야 라벨 확정 |
| Precise IF | 160  | Human (WildChat) | IFBench 제약을 붙이고 verifier 함수로 채점        |
| Math       | 183  | Human (WildChat) | 다수결 투표로 후보군을 만들고 전수 수작업 검증    |
| Safety     | 450  | CoCoNot          | GPT-4o 루브릭 + 수작업 검증                       |
| Focus      | 495  | Human (WildChat) | LLMBar식으로 프롬프트를 변형해 논점이탈 오답 생성 |
| Ties       | 102  | 수작업           | 여러 정답이 동시에 존재하는 경우의 강건성         |

각 도메인이 잡아내려는 것은 서로 다르다.

- **Factuality**: 완성문에 은근슬쩍 섞인 사실 오류나 환각을 RM이 걸러내는지 본다. "자연스러운 완성"과 "미묘한 사실 오류를 넣으라는 시스템 프롬프트로 만든 완성"을 섞어 놓고, 별도의 LLM 두 개가 독립적으로 "정확/부정확"에 동의할 때만 라벨을 확정한다.
- **Precise IF**: "u자를 쓰지 말고 답하라" 같은, IFBench 분류체계에서 가져온 정밀한 제약을 프롬프트에 붙인다. 이런 제약은 사람이 읽었을 때 그럴듯해 보이는 답이라도 규칙을 어기면 바로 걸러지므로, verifier 함수로 기계적으로 채점한다.
- **Math**: 중학교 물리·기하부터 대학 화학·미적분·조합론까지 걸친 open-ended 수학 프롬프트에 다수결 투표로 정답/오답 후보군을 만들고, 정답 추출이 까다로운 도메인 특성상 전수 수작업 검증을 거쳤다.
- **Safety**: 유해 요청에 대한 컴플라이언스/거부 판단을 CoCoNot 분류체계를 빌려와 테스트한다. 사용자마다 의견이 갈릴 수 있는 영역은 보수적으로 설계했다.
- **Focus**: "질문에 실제로 답했는가"를 본다. LLMBar 방식대로 프롬프트를 살짝 비틀어, 논점에서 벗어나거나 무성의한 답을 오답으로 유도한다.
- **Ties**: 이번에 새로 생긴 도메인이다. "무지개 색을 하나 말해봐"라는 질문엔 정답이 7개, 오답은 무한하다. 정답들 사이에서 RM이 부당하게 강한 선호를 표출하지 않으면서도, 오답과는 확실히 구분하는지를 함께 채점한다 — 정확도뿐 아니라 정답 간 점수 마진과 오답까지의 마진을 비교하는 가중 점수를 쓴다.

# Experiments

## 포화가 실제로 얼마나 심했나

<p align="center"><img src="/assets/post/image/rewardbench-2/rb1-vs-rb2-score-scatter.png" width="70%"></p>

이 산점도가 이 논문의 핵심 증거다. x축은 RewardBench 1 점수, y축은 RewardBench 2 점수인데, RewardBench 1에서 0.85\~0.95(85\~95점)에 몰려 있던 상위권 모델들(주황·파랑 점 대부분)이 RewardBench 2에서는 0.55\~0.77(55\~77점)로 넓게 흩어진다. 논문은 이를 "상위권 모델이 RewardBench 2에서 20점 이상 낮은 점수를 받는다"고 요약한다.

앞서 언급한 Skywork-Reward-Gemma-2-27B로 구체적인 숫자를 확인해보자.

| 모델                       | RewardBench 1 | RewardBench 2 | 차이   |
| -------------------------- | ------------- | ------------- | ------ |
| Skywork-Reward-Gemma-2-27B | 93.8          | 75.8          | \-18.0 |

18점이 빠졌다. 그런데 흥미롭게도 이 모델의 후속작 **Skywork-Reward-V2-Llama-3.1-8B는 RewardBench 2에서도 84.1점으로 1위**를 지켰다(Factuality 84.6 / IF 66.3 / Math 77.6 / Safety 96.7 / Focus 98.4 / Ties 81.2). 즉 [#6 글](/blog/2026/skywork-reward/)이 강조하는 "아키텍처보다 데이터 큐레이션"이라는 메시지는 더 어려운 벤치마크에서도 유지된다 — 다만 그 "1위"가 어느 버전 벤치마크에서의 1위인지는 반드시 확인해야 한다는 게 이 표가 주는 실무적 교훈이다.

Table 3 상위권을 조금 더 들여다보면 스칼라 RM과 생성형(LM-as-judge) RM이 나란히 채점된다. ContextualAI의 LMUnit-qwen2.5-72b(82.1)나 Gemini-2.5-Pro(79.5), Claude-Opus-4(76.5) 같은 생성형 모델이 상위권에 있지만, 논문은 이들을 "4개 중 최선 고르기"와 "개별 절대 평가" 두 프롬프팅 방식으로 각각 채점한 뒤 **더 잘 나온 쪽 점수를 채택**했다고 밝힌다. 스칼라 RM은 입력을 한 번 통과시키면 결정론적으로 점수 하나가 나오지만, 생성형 RM은 프롬프팅 방식에 따라 점수가 갈리고 그중 유리한 쪽이 보고된다 — 리더보드에서 스칼라 RM과 생성형 RM을 같은 줄에 놓고 비교할 때 이 비대칭을 감안해야 한다. 생성형 RM 자체의 구조와 한계는 이 시리즈 7부([#35 Generative Verifiers](/blog/2026/generative-verifiers/))에서 본격적으로 다룬다.

## Best-of-N과는 강하게 상관된다

<p align="center"><img src="/assets/post/image/rewardbench-2/domain-downstream-correlation.png" width="70%"></p>

113개 RM으로 RewardBench 2 평균 점수와 GSM8K·MATH·IFEval·AlpacaEval 2·BBH·PopQA·HumanEval+ 7개 downstream 태스크의 best-of-16 성능을 비교한 결과, 전체 평균 상관계수(Pearson $$r$$)는 **0.87**이었다. 여기서 쓰는 Pearson 상관계수는 다음과 같이 정의된다.

$$r = \frac{\sum_{i=1}^{n} (a_i - \bar a)(b_i - \bar b)}{\sqrt{\sum_{i=1}^{n} (a_i - \bar a)^2} \sqrt{\sum_{i=1}^{n} (b_i - \bar b)^2}}$$

기호를 풀면: $$n$$은 비교 대상 RM 개수, $$a_i$$는 $$i$$번째 RM의 RewardBench 2 점수, $$b_i$$는 같은 RM의 downstream 점수, $$\bar a$$와 $$\bar b$$는 각각의 평균이다. 분자는 두 점수가 함께 오르내리는 정도(공분산), 분모는 각 점수 자체의 흩어진 정도(표준편차의 곱)다. $$r$$이 1에 가까울수록 "RewardBench 2 점수가 오르면 downstream 점수도 거의 같은 비율로 오른다"는 뜻이고, 0에 가까우면 둘이 무관하다는 뜻이다.

**토이 예제**로 감을 잡아보자. RM 4개의 (RewardBench 2 점수, downstream 점수) 쌍이 (50, 45), (60, 68), (70, 63), (85, 80)이라 하자. 평균은 $$\bar a = 66.25$$, $$\bar b = 64$$다. 각 점의 편차를 곱해 더하면 분자는 $$(-16.25)(-19) + (-6.25)(4) + (3.75)(-1) + (18.75)(16) = 580$$이고, 분모는 $$\sqrt{668.75} \times \sqrt{634} \approx 25.86 \times 25.18 \approx 651.2$$다. 따라서 $$r \approx 580 / 651.2 \approx 0.89$$ — RM B 하나가 추세를 살짝 거스르는데도(점수는 중간인데 downstream이 가장 높은 축에 낌) $$r$$이 크게 깎이지 않는다. 논문이 113개 실제 RM으로 측정한 0.87도 이 정도 강도의, 이상치 몇 개를 견디는 튼튼한 양의 상관관계라고 보면 된다.

위 히트맵에서 "Downstream Average" 행을 보면 도메인별 편차가 드러난다.

| RewardBench 2 도메인 | Downstream Average와의 상관계수 |
| -------------------- | ------------------------------- |
| Factuality           | 0.94                            |
| Math                 | 0.91                            |
| Safety               | 0.87                            |
| RB2 평균             | 0.87                            |
| Precise IF           | 0.77                            |
| Ties                 | 0.70                            |
| Focus                | 0.60                            |

Factuality와 Math가 downstream 성능을 가장 잘 예측한다 — Math 도메인은 특히 GSM8K(0.90)·MATH(0.91)·HumanEval+(0.92)처럼 수리·코드 계열 태스크와 강하게 묶인다. 반면 Focus·Ties는 상관계수가 상대적으로 낮은데, 논문은 이것이 IFEval·PopQA가 애초에 _다른_ downstream 태스크와도 상관관계가 낮다는 사실과 맞닿아 있다고 짚는다 — 즉 벤치마크의 결함이 아니라 그 태스크들 자체가 다른 능력과 독립적인 스킬을 재는 것에 가깝다는 뜻이다.

## PPO와는 이야기가 다르다 — "포화"와 "역전"

<p align="center"><img src="/assets/post/image/rewardbench-2/ppo-bon-saturation.png" width="70%"></p>

Tulu 3 8B SFT를 정책으로 놓고, 17개의 서로 다른 RM으로 PPO를 돌렸다(learning rate $$3 \times 10^{-7}$$, KL 계수 $$\beta=0.05$$, Open Instruct 라이브러리 사용). 기준선은 Tulu 3 8B SFT 54.1점, 같은 선호 데이터로 학습한 Tulu 3 8B DPO 60.3점이다.

위 그림에서 파란 점(BoN)은 RewardBench 2 점수가 오를수록 downstream 점수도 대체로 오르는 반면, 주황 점(PPO)은 원(on-policy) 모델의 경우 RewardBench 2 점수가 55점대에서 68점대까지 벌어져도 PPO 점수는 59.5\~60.7점 사이에 거의 납작하게 눌려 있다. 반면 별표(off-policy) 모델은 RewardBench 2 점수와 무관하게 대체로 더 낮게 깔린다.

17개 RM 중 대표적인 몇 개를 뽑아 세 그룹으로 비교하면 이 역전 현상이 선명해진다.

| 그룹                             | RewardBench 1 | RewardBench 2 | PPO 점수               |
| -------------------------------- | ------------- | ------------- | ---------------------- |
| On-policy, 상위권 (예시 모델)    | 78.5\~85.9    | 55.5\~68.5    | 59.5\~60.7 (거의 포화) |
| On-policy, 하위권 (예시 모델)    | 54.2          | 22.1          | 54.2                   |
| Off-policy, 최상위권 (예시 모델) | 88.9          | 72.6          | 54.5                   |

이 표의 세 번째 줄이 이 논문에서 가장 반직관적인 결과다. Off-policy 모델은 RewardBench 1(88.9)과 RewardBench 2(72.6) 둘 다에서 이 비교군 중 가장 높은 점수를 받았다. 그런데 PPO 점수는 54.5로, RewardBench 1에서 무려 34.7점 낮은 on-policy 하위권 모델(54.2점)과 사실상 같다. 벤치마크 점수로는 "압도적으로 좋은 RM"인데, 실제 정책 학습에 넣으면 "거의 도움이 안 되는 RM"과 구분이 안 되는 셈이다.

이유는 policy와 RM 사이의 **혈통(lineage) 불일치**에 있다. Off-policy 모델은 정책(Tulu SFT)과 다른 베이스 모델에서 출발했거나, PPO 학습 프롬프트와 분포가 다른 데이터로 학습된 RM이다. 다른 학교 교재로만 과외를 받은 학생이 정작 자기 학교 시험 문제 스타일에는 약한 것과 비슷하다 — RM이 아무리 "일반적으로" 정답을 잘 골라도, 지금 최적화하려는 정책이 만들어내는 특정 분포의 출력을 세밀하게 구분하지 못하면 PPO에 별 도움이 안 된다. 논문은 이 온/오프폴리시 효과가 RewardBench 2에만 국한되지 않고 RM-Bench·PPE·RewardBench 1에서도 동일하게 재현된다고 확인했다 — 즉 특정 벤치마크의 결함이 아니라, **정확도 기반 벤치마크 전체가 공유하는 구조적 한계**라는 뜻이다.

# Conclusion

한 줄로 요약하면: **RewardBench 2는 "한 번도 안 쓰인 프롬프트로, 오답 3개 사이에서 정답을 고르게 하고, 그 점수가 실제 downstream 성능과 정말 상관관계가 있는지"까지 검증한 첫 RM 평가 벤치마크다. 결론은 절반의 성공이다 — best-of-N과는 잘 맞지만(r=0.87), PPO와는 정책과의 궁합(on/off-policy)이 벤치마크 점수보다 더 중요한 변수였다.**

정리하면,

1. **문제**: RewardBench 1은 기존 downstream 벤치마크(AlpacaEval, MT-Bench 등)의 프롬프트를 재활용해 만들어져 오염 위험이 있었고, 출시 반 년 만에 상위권 점수가 90점대로 몰리며 변별력을 잃었다.
2. **해법**: WildChat에서 가져온 unseen 프롬프트(전체의 약 70%)로 20개 downstream 평가와 중복 없이 새로 만들고, 정답 1개·오답 3개 중에서 고르는 best-of-4 정확도(랜덤 베이스라인 25%)로 채점한다.
3. **검증**: 113개 RM으로 잰 best-of-N 상관계수는 0.87로 강했지만, 17개 RM으로 돌린 PPO 실험에서는 온폴리시 RM끼리 점수가 55\~68점대여도 성능이 59.5\~60.7점에 포화되고, 오프폴리시 RM은 벤치마크 최상위권(72.6점)이어도 54.5점까지 떨어졌다.

이 시리즈 2부(#4\~#9)가 다룬 건 결국 "RM을 어떻게 잘 만들 것인가"였다. 이 글은 그 마지막 조각으로 "잘 만들었다는 걸 어떻게 확인할 것인가"를 짚었다. 그런데 이 글이 보여준 PPO 포화·역전 현상은 사실 더 불길한 질문을 남긴다 — **벤치마크 점수가 높은 RM도 실제 RL 루프 안에서는 여전히 뚫릴 수 있다면, 그 RM 자체는 얼마나 신뢰할 수 있는가?** 이건 벤치마크의 문제가 아니라 RM이 최적화 대상이 되는 순간 생기는 근본적인 위험, 즉 **reward overoptimization**의 문제다. 3부의 첫 글인 [#10 Overoptimization Scaling Laws](/blog/2026/reward-model-overoptimization/)가 바로 이 질문 — "RM 점수를 계속 밀어붙이면 실제 품질은 어느 시점부터 오히려 나빠지는가" — 을 정량적으로 다룬다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 아홉 번째 글이다.

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
  <li><strong>(현재 글)</strong> RewardBench 2 (2025) — RM을 어떻게 평가할 것인가</li>
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
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 46편으로 구성된다.

# 참고 문헌

- Malik et al., 2025. [RewardBench 2: Advancing Reward Model Evaluation](https://arxiv.org/abs/2506.01937). ICLR 2026.
- [ar5iv: RewardBench 2 (HTML rendering)](https://arxiv.org/html/2506.01937v2) — 본문 그림·표 원본.
- [OpenReview: RewardBench 2](https://openreview.net/forum?id=fb0G86Dewb) — ICLR 2026 리뷰 및 게재 확인.
- Lambert et al., 2024. [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787).
- [ar5iv: RewardBench (HTML rendering)](https://arxiv.org/html/2403.13787) — 카테고리 표·violin plot 원본.
- [HuggingFace: allenai/reward-bench-2](https://huggingface.co/datasets/allenai/reward-bench-2)
- [HuggingFace: allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench)
- [GitHub: allenai/reward-bench](https://github.com/allenai/reward-bench)
- Zhao et al., 2024. [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470). (unseen 프롬프트 출처)
- [Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451). (Skywork-Reward-Gemma-2-27B의 RewardBench 1 점수 93.8 확인)
- Nathan Lambert, [RewardBench 2 and the state of preference finetuning](https://natolambert.substack.com/p/rewardbench-2-and-the-state-of-preference).
