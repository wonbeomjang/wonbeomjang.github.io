---
layout: post
title: "Generative Verifiers: reward를 분류가 아니라 생성으로 풀다"
date: 2026-08-11 09:22:00 +0900
description: "RLHF Reward 설계 시리즈 #23 — next-token prediction으로 학습한 verifier가 CoT와 test-time compute를 얻는 법"
categories: [paper]
tags: [rlhf, reward-model, genrm, verifier, llm-as-a-judge, paper]
giscus_comments: true
related_posts: true
---

> [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) (Zhang et al., Google DeepMind, ICLR 2025)

# Introduction

이 시리즈의 2부(4\~9편)는 스칼라 reward model(RM)을 해부하는 데 다섯 편을 썼다. [#4 Rethinking Bradley-Terry](/blog/2026/bradley-terry-rethinking/)는 "선호를 스칼라 점수로 접는 변환이 Bradley-Terry(BT)만 있는 게 아니다"라는 수학적 균열을 열었고, [#9 RewardBench 2](/blog/2026/rewardbench-2/)는 "판별형(discriminative) RM과 생성형(generative) judge를 같은 벤치마크로 비교하는 것 자체가 문제"라는 평가 방법론의 틈을 짚었다. 3부(10\~13편)는 그 스칼라 점수가 얼마나 쉽게 hacking당하는지를 보였다. 이 모든 균열이 가리키는 방향은 하나였다. **reward를 스칼라 하나로 접어야 할 이유가 애초에 없다.**

이번 글부터 시작하는 6부가 그 답이다. 판별 RM은 사전학습으로 얻은 LLM의 **텍스트 생성 능력을 전혀 쓰지 않는다.** 수십억 토큰으로 "그럴듯한 다음 토큰을 생성하는 법"을 배운 모델을 가져다가, 맨 끝에 classification head 하나만 새로 붙이고 그 능력을 통째로 버린다. Generative Verifiers(이하 GenRM)는 이 낭비를 정면으로 지적한다. 논문은 이렇게 말한다.

> "Discriminative LLM-based verifiers do not utilize the text generation capabilities of pretrained LLMs... As a result, discriminative RMs miss out on the inherent strengths of generative LLMs, such as unified instruction tuning, chain-of-thought reasoning, and utilizing additional inference-time computation."

GenRM의 핵심은 단순하다. "이 답이 맞습니까?"라는 질문에 **Yes 또는 No를 다음 토큰으로 예측**하게 학습시키고, reward는 그 `Yes` 토큰의 확률로 정의한다. 검증(verification)을 분류가 아니라 **생성**으로 재정의하는 것이다.

이 전환이 중요한 이유는 세 가지다. (a) 판별 head 없이 instruction tuning 파이프라인에 그대로 올라탄다. (b) Yes/No를 뱉기 전에 **검증 근거(rationale)를 먼저 생성**할 수 있다 — chain-of-thought(CoT) 검증이다. (c) 이 rationale을 여러 번 샘플링해서 **test-time compute를 reward 품질에 직접 투입**할 수 있다. 이 시리즈에서 (c)는 처음 등장하는 축이다. 지금까지의 RM은 학습이 끝나면 추론 비용이 고정된 결정론적 함수였다. GenRM은 "얼마나 많이 계산할 것인가"를 reward 품질의 손잡이로 바꿔놓는다.

결과는 명확하다. Best-of-N 선택에서 알고리즘 태스크는 5% → 45.3%, GSM8K는 73% → 93.4%, MATH로의 전이는 28% → 44.6%까지 오른다. 다만 이 모든 이득에는 값이 매겨져 있다. 검증마다 텍스트를 생성해야 하므로 스칼라 RM보다 훨씬 느리다. 이 비용 문제를 정면으로 다루는 것이 [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)다.

# Background

## 판별 RM이 버리는 것

지금까지 이 시리즈에서 다룬 스칼라 RM은 전부 같은 골격이었다. Pretrained LLM의 마지막 hidden state에 classification head를 붙이고, 선호 쌍 $$(y^+, y^-)$$에 대해 이진 분류로 학습한다.

$$\mathcal{L}(\theta, D_{RM}) = -\mathbb{E}_{(x,y^+)\sim D_{correct}}[\log r_\theta(x,y^+)] - \mathbb{E}_{(x,y^-)\sim D_{incorrect}}[\log(1-r_\theta(x,y^-))]$$

$$r_\theta(x,y) = \text{sigmoid}(z_{cls}), \qquad z_{cls} = \text{logit}_\theta(cls \mid y, x)$$

기호를 하나씩 풀면, $$x$$는 문제, $$y^+/y^-$$는 정답/오답 풀이, $$z_{cls}$$는 새로 얹은 classification head가 내놓는 로짓 하나, $$r_\theta$$는 그걸 sigmoid로 눌러 만든 스칼라 점수다. [#4 글](/blog/2026/bradley-terry-rethinking/)에서 다룬 BT 변환도 결국 이 $$r_\theta$$ 하나를 어떻게 정의하느냐의 문제였다.

문제는 이 구조가 **LLM이 원래 잘하는 일 — 다음 토큰을 생성하는 일 — 을 단 한 번도 쓰지 않는다**는 데 있다. 비유하자면, 소설을 수백 권 읽고 문장을 만드는 훈련을 받은 작가를 건물 경비원으로 앉혀놓고 "이 사람 들여보내도 돼요?"에 고개만 끄덕이거나 젓게 하는 것과 같다. 그 작가가 "왜 이 사람을 들여보내면 안 되는지" 설명하는 능력은 통째로 사장된다. 판별 RM의 classification head가 바로 이 역할이다 — 생성 능력을 가진 모델에게 끄덕임 하나만 요구한다.

[#9 RewardBench 2](/blog/2026/rewardbench-2/)가 지적한 것도 이 지점과 맞닿아 있다. 판별 RM(스칼라 점수)과 LLM-as-a-Judge(생성된 판정)를 같은 벤치마크 표에 나란히 올리는 순간, 사실은 **서로 다른 종류의 출력을 비교하는 것**이 된다. GenRM은 이 둘을 애초에 같은 축 위에 놓는다 — 둘 다 "생성"이기 때문이다.

## LLM-as-a-Judge는 왜 부족한가

검증을 생성으로 하는 아이디어 자체는 새롭지 않다. 이미 존재하는 off-the-shelf LLM에게 "이 풀이가 맞습니까? 단계별로 생각해보세요"라고 프롬프트만 주는 LLM-as-a-Judge가 그 원형이다. 이 논문의 실험에서도 baseline으로 등장하는데, 파인튜닝 없이 zero-shot CoT 프롬프트만으로 32개의 검증 rationale을 뽑는다.

그런데 이 baseline은 실험 전반에서 가장 낮은 성능대에 머문다. 검증 능력을 명시적으로 가르치지 않았기 때문이다. GenRM이 하는 일은 "검증을 생성으로 표현할 수 있다"는 관찰에서 한 걸음 더 나아가, **그 생성 능력을 검증 태스크에 맞춰 직접 파인튜닝**하는 것이다.

# Method

## GenRM-Direct: Yes/No를 다음 토큰으로

가장 단순한 형태부터 보자. Instruction-tuning과 동일한 SFT 목적함수를 그대로 쓴다.

$$\mathcal{L}_{SFT}(\theta, D) = -\mathbb{E}_{(x,y)\sim D}\left[\sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})\right]$$

여기에 검증용 데이터셋을 이렇게 구성한다.

$$D_{Direct} = \{(x, y^+, I),\ \text{Yes}\} \cup \{(x, y^-, I),\ \text{No}\}$$

$$x$$는 문제, $$y^+/y^-$$는 정답/오답 풀이, $$I$$는 "정답이 맞습니까 (Yes/No)?"라는 고정 지시문이다. 이 데이터로 파인튜닝한 뒤, reward는 생성 시퀀스가 아니라 다음 토큰 분포에서 바로 읽는다.

$$r_{Direct}(x,y) = p_\theta(\text{Yes} \mid x, y, I)$$

즉 classification head를 새로 붙이지 않는다. 모델이 원래 가지고 있던 vocabulary 위의 다음 토큰 확률 중 `Yes`라는 토큰 하나의 확률을 reward로 그대로 쓴다. Head가 사라졌으니 instruction-tuned 모델과 아키텍처가 완전히 같고, 학습 파이프라인도 그대로 재사용된다 — 이것이 앞서 말한 첫 번째 이득이다.

## GenRM-CoT: 판정 전에 근거를 먼저 쓴다

GenRM-Direct는 여전히 "판정 하나"만 출력한다. 여기서 한 걸음 더 나아가 판정 전에 **검증 rationale $$v_{CoT}$$를 먼저 생성**하게 만든 것이 GenRM-CoT다.

$$D_{CoT} = \{(x, y^+, I_{CoT}),\ (v_{CoT}, I, \text{Yes})\} \cup \{(x, y^-, I_{CoT}),\ (v_{CoT}, I, \text{No})\}$$

$$I_{CoT}$$는 "단계별로 검증해보세요" 같은 지시문이고, $$v_{CoT}$$는 모델이 생성한 검증 근거(어느 단계에서 계산이 틀렸는지, 논리가 끊기는지 등을 설명하는 텍스트)다. 학습은 이 검증 생성과 원래의 정답 풀이 생성을 함께 최적화한다.

$$\mathcal{L}_{GenRM}(\theta, D_{verify}) = \mathcal{L}_{SFT}(\theta, D_{verify}) + \lambda\, \mathcal{L}_{SFT}(\theta, D_{correct})$$

$$D_{verify}$$는 검증 데이터, $$D_{correct}$$는 원래의 solution-generation 데이터, $$\lambda$$는 두 loss의 균형 계수다. 이렇게 둘을 함께 학습하는 이유는, 검증 능력만 파고들다 보면 원래의 문제 풀이 생성 능력이 망가지기 쉽기 때문이다(논문은 이를 unified generation-verification synergy로 부른다). 단일 rationale로 채점할 때 reward는 다음과 같다.

$$r_{CoT}(x,y) = p_\theta(\text{Yes} \mid x, y, I_{CoT}, v_{CoT}, I)$$

<p align="center"><img src="/assets/post/image/generative-verifiers/x2.png" width="70%"></p>

위 그림이 GenRM(위)과 GenRM-CoT(아래)의 구조 차이를 보여준다. GenRM은 "정답입니까?" 질문 뒤 바로 `Yes`/`No` 토큰 확률을 읽는다. GenRM-CoT는 "단계별로 검증해보자"는 지시 뒤 여러 개의 검증 CoT를 생성하고, 각각의 `Yes` 확률을 평균 내 최종 $$r$$을 만든다.

한 가지 실무적인 문제가 남는다 — "왜 틀렸는지"를 설명하는 rationale은 사람이 대량으로 라벨링해줄 수 없다. 논문은 이를 합성으로 해결한다. 정답 참조 풀이(reference solution)를 모델에게 함께 보여주고 검증 rationale을 샘플링한 뒤, 최종 판정이 실제 정답 라벨과 일치하는 rationale만 학습 데이터로 남긴다. GSM8K에서 이 참조 유도(reference guidance) 없이 생성한 rationale의 정답 일치율은 87.8%였는데, 참조를 주고 유도하면 91.7%까지 올라간다. 근거 없이 아무 CoT나 만들면 판정 자체가 부정확해지므로, 이 필터링 단계가 GenRM-CoT 데이터 품질의 핵심이다.

## Majority voting: 계산을 더 쓰면 reward가 좋아진다

GenRM-CoT는 rationale을 **여러 번** 샘플링할 수 있다는 점에서 판별 RM과 근본적으로 다르다. $$K$$개의 독립적인 검증 rationale $$v_{CoT}^{(i)}$$을 뽑고, 각각의 `Yes` 확률을 평균한다.

$$r_{MajV@K}(x,y) = \frac{1}{K}\sum_{i=1}^{K} p_\theta(\text{Yes} \mid x, y, I_{CoT}, v_{CoT}^{(i)}, I)$$

$$K$$는 샘플링한 rationale 개수, $$v_{CoT}^{(i)}$$는 $$i$$번째로 독립 샘플링한 검증 근거다. 서로 다른 관점에서 풀이를 다시 짚어본 $$K$$명의 채점자에게 각자 Yes 확신도를 묻고 평균 내는 것과 같다.

### 토이 예제: 세 개의 rationale, 하나의 reward

어떤 수학 문제의 풀이 $$y$$ 하나를 GenRM-CoT로 채점한다고 하자. 독립적으로 3개의 검증 rationale을 샘플링했고, 각 rationale이 끝난 뒤의 `Yes` 토큰 확률이 다음과 같이 나왔다.

| 샘플 $$i$$ | 검증 rationale 요약                         | $$p_\theta(\text{Yes})$$ |
| ---------- | ------------------------------------------- | ------------------------ |
| 1          | "3단계 나눗셈 계산을 다시 확인 — 이상 없음" | 0.9                      |
| 2          | "단위 변환에서 자릿수 하나 놓친 듯"         | 0.3                      |
| 3          | "최종 답 대입 검산 — 일치"                  | 0.8                      |

$$r_{MajV@3}(x,y) = \frac{0.9 + 0.3 + 0.8}{3} = \frac{2.0}{3} \approx 0.667$$

세 번째 rationale까지 반영한 최종 reward는 0.667이다. 여기서 중요한 건 숫자 자체가 아니라 **한 번 더 검증할수록 정보가 늘어난다**는 점이다. 만약 $$K=1$$이었고 하필 2번 샘플(0.3)만 뽑았다면 이 풀이는 낮은 점수를 받았을 것이다. $$K$$를 늘릴수록 우연히 뽑힌 rationale 하나에 reward가 휘둘릴 확률이 줄어든다.

이제 같은 풀이를 판별 RM으로 채점했다면 어떻게 될까. 판별 RM은 $$(x, y)$$를 한 번 forward pass에 통과시켜 $$r_\theta(x,y) = \text{sigmoid}(z_{cls})$$ 값 하나 — 예를 들어 0.62 — 를 내놓는다. 이 값은 **결정론적**이다. 같은 입력을 100번 다시 넣어도 정확히 같은 0.62가 나온다. 채점을 몇 번 더 시킨다고 정보가 늘어나지 않는다. 판별 RM에게 "한 번 더 봐줘"라고 요청하는 것 자체가 의미가 없는 것이다. 반면 GenRM-CoT는 매번 다른 각도에서 rationale을 새로 쓰기 때문에, 계산을 더 쓸수록 실제로 더 나은 판정에 수렴한다. 이것이 "test-time compute를 reward 품질로 바꾼다"는 말의 정확한 의미다.

## DPO를 검증기로 쓰는 방법과의 대비

논문의 또 다른 baseline은 DPO다. [#18 DPO 글](/blog/2026/dpo/)에서 다룬 것처럼, DPO는 명시적 reward model 없이 정책을 직접 최적화한다. 그런데 DPO로 학습한 정책은 참조 모델 대비 log-probability 비율이라는 **암묵적 reward**를 갖고 있고, Hosseini et al.(V-STaR, 2024)은 이 암묵적 reward를 그대로 verifier로 재활용한다. 즉 DPO verifier는 "reward를 없앤" 노선(#18)의 부산물을 다시 판정 점수로 끌어오는 방식이다. GenRM-CoT는 이와 정반대 방향이다 — reward를 없애는 대신, reward를 **생성 그 자체**로 확장한다. 실험에서 DPO verifier는 판별 RM보다도 낮은 성능대에 머무는데, 이는 애초에 판정을 위해 설계된 신호가 아니라 정책 최적화의 부산물이기 때문으로 보인다.

# Experiments

## Best-of-N: 다섯 가지 검증 방법의 대결

논문은 다섯 가지 검증 방법 — LLM-as-a-Judge, DPO verifier, Discriminative RM, GenRM(Direct), GenRM-CoT — 을 세 도메인에서 Best-of-N 정확도로 비교한다. 방법별 차이를 먼저 정리하면 다음과 같다.

| 방법              | 검증 신호                 | 학습 방식                               | 출력 형태               | test-time compute 활용             |
| ----------------- | ------------------------- | --------------------------------------- | ----------------------- | ---------------------------------- |
| LLM-as-a-Judge    | 없음 (off-the-shelf)      | 학습 없음, zero-shot CoT 프롬프트       | Yes/No 토큰 확률        | 가능하나 파인튜닝 없이 정확도 낮음 |
| DPO verifier      | 정답/오답 선호 쌍         | DPO objective (V-STaR 방식)             | 정책의 암묵적 reward    | 불가능                             |
| Discriminative RM | 정답/오답 이진 라벨       | cls head + BCE                          | 스칼라 점수 하나        | 불가능 (결정론적 단일 forward)     |
| GenRM (Direct)    | 정답/오답 이진 라벨       | next-token prediction (SFT)             | Yes/No 토큰 확률        | 제한적 (rationale 없음)            |
| GenRM-CoT         | 정답/오답 + CoT rationale | next-token prediction + 생성 joint 학습 | rationale → Yes/No 확률 | 가능 (majority voting)             |

<p align="center"><img src="/assets/post/image/generative-verifiers/x1.png" width="85%"></p>

세 도메인의 결과는 다음과 같다.

- **알고리즘 추론(2개 태스크, Best-of-32, Gemma-2B, length generalization)**: 5.0% → 45.3%
- **GSM8K(Best-of-16, Gemini 1.0 Pro가 생성한 풀이를 Gemma2-9B verifier가 채점)**: 73.0% → 93.4%
- **MATH로의 전이(Best-of-32, GSM8K로만 학습한 verifier를 MATH에 그대로 적용)**: 28.0% → 44.6%

세 그래프 모두에서 순위는 대체로 일관된다. **LLM-as-a-Judge가 가장 낮고, Discriminative RM과 GenRM(Direct)이 비슷한 성능대를 이루며, GenRM-CoT가 그 위에서 가장 높다.** 특히 세 번째 그래프(MATH 전이)에서 GenRM-CoT의 우위가 가장 크게 벌어지는데, 이는 학습 때 보지 못한 도메인일수록 "판정 하나"보다 "근거를 먼저 써보는 것"이 더 크게 도움이 된다는 뜻으로 읽힌다.

## MMLU: 훈련하지 않은 도메인으로의 전이

GSM8K로만 학습한 verifier를 MMLU의 수학 하위 태스크에 그대로 적용한 easy-to-hard 전이 실험도 있다. Best-of-N 기준 결과는 다음과 같다.

| MMLU 하위 태스크        | Base(Pass@1) | Discriminative RM | GenRM-CoT | GenRM-CoT 개선폭 |
| ----------------------- | ------------ | ----------------- | --------- | ---------------- |
| elementary_mathematics  | 80.1%        | 90.6%             | 91.1%     | +0.5%p           |
| high_school_mathematics | 52.2%        | 74.8%             | 76.1%     | +1.3%p           |
| college_mathematics     | 47.6%        | 53.0%             | 56.1%     | +3.1%p           |
| abstract_algebra        | 37.9%        | 50.0%             | 53.5%     | +3.5%p           |

패턴이 뚜렷하다. **문제가 쉬울수록(elementary) GenRM-CoT의 추가 이득은 작고, 어려울수록(abstract_algebra) 이득이 커진다.** 판별 RM도 이미 쉬운 문제는 잘 채점하기 때문에 CoT rationale이 더할 정보가 적지만, 추상대수처럼 검증 자체에 여러 단계의 추론이 필요한 문제에서는 "판정 전에 근거를 써보는" 것이 실질적인 차이를 만든다.

## 계산을 더 쓸수록 reward가 좋아지는지 확인하기

<p align="center"><img src="/assets/post/image/generative-verifiers/x9.png" width="85%"></p>

이 그림은 GSM8K에서 majority voting 샘플 수 $$K$$를 1에서 32까지 늘렸을 때의 Best-of-16 정확도를, Gemma-2B/7B/9B 세 모델 크기에서 각각 보여준다. 두 가지가 확인된다.

1. **GenRM-CoT(teal 실선)는 $$K$$가 늘어날수록 세 모델 크기 모두에서 단조롭게 좋아진다.** 이는 앞서 토이 예제에서 본 majority voting의 효과가 실제 벤치마크에서도 재현된다는 뜻이다.
2. **LLM-as-a-Judge(노란 실선)도 $$K$$를 늘리면 좋아지지만, 훨씬 낮은 성능대에 머무른다.** 파인튜닝 여부가 test-time compute를 아무리 투입해도 메워지지 않는 격차를 만든다는 뜻이다.

논문은 여기에 더해, weighted self-consistency(자기 일관성에 verifier 점수를 가중치로 결합하는 기법)에 GenRM-CoT를 쓰면 판별 RM과 동등한 성능에 도달하는 데 필요한 solution 샘플 수가 약 2.5배 줄고, MATH easy-to-hard 설정에서는 최대 6.4배 적은 샘플로 판별 RM 수준에 도달한다고 보고한다. Reward 품질이 좋아지면 그만큼 다른 축(생성 샘플 수)의 계산을 아낄 수 있다는 것이다.

## 한계: 이 모든 이득의 대가는 추론 비용

판별 RM은 $$(x, y)$$ 한 쌍마다 forward pass 한 번이면 끝난다. GenRM-CoT는 $$N$$개의 후보 풀이 각각에 대해 $$K$$개의 검증 rationale을 **새로 생성**해야 한다. 즉 채점 비용이 $$O(N)$$에서 $$O(N \times K)$$로 늘어나고, 그마저도 각 rationale이 스칼라 하나가 아니라 수십~수백 토큰짜리 텍스트다. 논문 자체는 이 비용을 정량적으로 다루지 않지만, 구조적으로 명백하다 — 검증이 생성인 이상, 검증은 생성만큼 느리다.

이 트레이드오프 — reward 품질은 좋아지는데 추론은 비싸진다 — 를 어떻게 다룰 것인가가 이후 GenRM 계열 연구의 핵심 과제로 넘어간다. [#26 DeepSeek-GRM/SPCT](/blog/2026/deepseek-grm-spct/)가 바로 이 inference-time scaling 문제를 정면으로 다룬다.

# Conclusion

GenRM의 메시지를 한 줄로 요약하면, **reward는 스칼라 분류가 아니라 next-token prediction으로도 충분히, 오히려 더 잘 만들 수 있다**는 것이다. 판별 RM이 버렸던 생성 능력을 되찾아오면 세 가지가 따라온다 — instruction tuning과의 매끄러운 통합, 판정 전에 근거를 쓰는 CoT 검증, 그리고 계산을 더 쓸수록 좋아지는 test-time compute 활용이다. Best-of-N에서 알고리즘 태스크 5% → 45.3%, GSM8K 73% → 93.4%, MMLU abstract algebra 37.9% → 53.5%라는 수치가 이를 뒷받침한다.

다만 이 이득은 공짜가 아니다. 판정마다 텍스트를 생성해야 하므로 판별 RM보다 훨씬 느리다. [#21 DeepSeek-R1](/blog/2026/deepseek-r1/)에서 본 규칙 기반 reward는 검증 가능한 도메인 안에서는 이 문제 자체가 없었지만, 검증이 어려운 일반 도메인으로 갈수록 "생성으로 검증한다"는 이 접근이 불가피해진다. 다음 글(#23)은 GenRM을 선호 학습(preference learning)과 결합하는 후속 연구를, 그다음(#24)은 이 추론 비용 문제를 inference-time scaling으로 다루는 DeepSeek-GRM/SPCT를 살펴본다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스물세 번째 글이다.

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
23. **(현재 글)** Generative Verifiers (2024) — reward를 next-token prediction으로
24. [Generative Reward Models (2024)](/blog/2026/generative-reward-models/) — GenRM과 선호 학습의 결합
25. [Self-Taught Evaluators (2024)](/blog/2026/self-taught-evaluators/) — 사람 라벨 없이 judge를 키우다
26. [DeepSeek-GRM / SPCT (2025)](/blog/2026/deepseek-grm-spct/) — inference-time scaling

**7부. 생각하는 Judge, 그리고 그 신뢰**

27. [ReasonGRM (2025)](/blog/2026/reasongrm/) — reasoning 능력을 judge에 이식
28. [J1 (2025)](/blog/2026/j1-thinking-judge/) — RL로 judge를 생각하게 만들기
29. [Rubrics as Rewards (2025)](/blog/2026/rubrics-as-rewards/) — 비검증 도메인으로
30. [CriticEval (2024)](/blog/2026/criticeval/) — judge 자체를 어떻게 평가하나
31. [One Token to Fool LLM-as-a-Judge (2025)](/blog/2026/one-token-to-fool-judge/) — GenRM도 뚫린다

**8부. 실전 종합**

32. [프론티어 모델의 reward 설계 (2025~2026)](/blog/2026/frontier-reward-design/) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것
33. [reward를 어떻게 설계할 것인가](/blog/2026/reward-model-design/) — 시리즈를 관통한 RM 설계 원칙 한 장

본 시리즈는 33편으로 구성된다.

# 참고 문헌

- Zhang et al., 2024. [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) (arXiv:2408.15240, ICLR 2025).
- [ICLR 2025 OpenReview: Generative Verifiers](https://openreview.net/forum?id=Ccwp4tFEtE)
- Rafailov et al., 2023. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).
- Hosseini et al., 2024. [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457).
