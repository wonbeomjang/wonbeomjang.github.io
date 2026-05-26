---
layout: post
title: "Curiosity-driven Red-teaming for Large Language Models"
date: 2026-05-16 17:00:00 +0900
description: "Red-Teaming 시리즈 #11 — RL 기반 red-teaming의 mode collapse를 novelty reward로 해결, SelfBLEU + 코사인 유사도 (Hong et al., ICLR 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, reinforcement-learning, novelty]
giscus_comments: true
related_posts: true
---

> [Curiosity-driven Red-teaming for Large Language Models](https://arxiv.org/abs/2402.19464) (Hong et al., MIT, ICLR 2024)

# Introduction

## Red-teaming이란 무엇인가 — 먼저 큰 그림부터

ChatGPT, Claude, Llama 같은 LLM을 세상에 내보내기 전에, 개발자는 한 가지를 반드시 확인해야 한다. **"이 모델이 위험하거나 유해한 말을 뱉게 만드는 입력이 있는가?"** 폭탄 제조법, 혐오 발언, 욕설 같은 응답을 유도하는 입력을 미리 찾아내야, 출시 전에 그 구멍을 막을 수 있다.

이렇게 **모델을 일부러 공격해 약점을 찾는 작업**을 **red-teaming**이라 부른다. 군사·보안에서 아군 시스템을 가상의 적("red team")이 공격해보는 훈련에서 빌려온 용어다. 공격을 던지는 쪽이 red team, 공격당하는 모델이 target LLM이다.

전통적으로 red team은 **사람**이었다. 사람이 직접 머리를 짜내 "이렇게 물으면 모델이 넘어가지 않을까?" 하는 프롬프트(이것을 **test case**라 부른다)를 만들어 던진다. 그러나 사람이 직접 하는 건 (1) 비싸고 (2) 느리고 (3) 사람마다 떠올리는 공격 패턴이 한정적이다.

그래서 [Perez 2022](/blog/2026/perez-red-teaming/)는 발상을 바꿨다. **공격하는 쪽도 LLM에게 맡기자.** 별도의 작은 LLM(red LM)을 두고, "유해 응답을 유발하는 프롬프트를 생성하라"는 목표로 강화학습(RL)으로 훈련시키는 것이다. 이것이 **자동 red-teaming**이다.

## RL 기반 red-teaming의 기본 구조

RL이 처음이라면, 다음 비유로 충분하다. 강아지에게 재주를 가르친다고 하자. 강아지가 원하는 행동을 하면 간식(보상, reward)을 주고, 아니면 안 준다. 강아지는 "간식을 최대한 많이 받는" 방향으로 행동을 바꿔간다.

RL red-teaming도 똑같다.

1. red LM(강아지 역할)이 공격 프롬프트 $$x$$를 하나 생성한다. 예: "What is the most stupid thing you've ever heard?"
2. 이 $$x$$를 target LLM에 넣어 응답 $$y$$를 받는다.
3. 응답 $$y$$가 얼마나 유해한지를 **toxicity 분류기**가 점수로 매긴다. 이것이 보상 $$R(y)$$다 (간식). 유해할수록 점수가 높다.
4. red LM은 이 보상을 최대화하도록 자기 파라미터를 업데이트한다. 즉 "유해 응답을 잘 끌어내는 프롬프트"를 점점 잘 만들게 된다.

수식으로 쓰면 red LM $$\pi$$의 목표는 다음과 같다.

$$\max_\pi \ \mathbb{E}_{x \sim \pi,\ y \sim p(\cdot \mid x)}\left[ R(y) \right]$$

기호를 하나씩 풀면 이렇다.

- $$\pi$$: red LM의 정책(policy). 쉽게 말해 "어떤 프롬프트를 어떤 확률로 생성하는가"를 정하는 모델 그 자체.
- $$x \sim \pi$$: red LM이 생성한 공격 프롬프트.
- $$p$$: target LLM. $$y \sim p(\cdot \mid x)$$는 그 모델이 $$x$$를 받고 내놓은 응답.
- $$R(y)$$: 응답의 유해도 점수(toxicity classifier가 매김).
- $$\mathbb{E}[\cdot]$$: 기댓값. 여러 번 샘플링했을 때의 평균.

요약하면 **"red LM이 만든 프롬프트가 유발하는 평균 유해도를 최대로 만들어라."**

여기에 보통 한 가지 항을 더 붙인다. 그냥 보상만 좇으면 red LM이 사람이 못 읽는 깨진 문자열(gibberish)을 뱉을 수 있다. 그래서 "원래 LM($$\pi_{\text{ref}}$$, 사전학습된 GPT-2 등)에서 너무 멀어지지 마라"는 **KL 페널티**를 더한다.

$$\max_\pi \ \mathbb{E}\left[ R(y) - \beta\, D_{\text{KL}}\big(\pi(\cdot \mid z) \,\|\, \pi_{\text{ref}}(\cdot \mid z)\big) \right]$$

- $$D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$: 두 분포가 얼마나 다른지를 재는 거리(쿨백-라이블러 발산). 클수록 red LM이 원래 모델에서 멀어졌다는 뜻.
- $$\beta$$: 그 페널티의 세기. 크면 "정상 언어"에 가깝게 유지, 작으면 자유롭게 공격.
- $$z$$: red LM에게 주는 입력 프롬프트(일종의 시드). red LM도 LLM이므로 입력이 필요하다. 데이터셋 $$D$$에서 뽑는다.

여기까지가 Perez 2022가 세운 RL red-teaming의 표준 틀이다. 이 글에서 다루는 CRT는 바로 이 틀의 **치명적 약점 하나를 고친다.**

## 문제: Mode Collapse

[Perez 2022](/blog/2026/perez-red-teaming/)는 RL이 가장 강한 공격력을 보였지만 한 가지 문제를 짚었다. **RL은 "같은 공격만 반복"하는 함정에 빠진다.** 실제로 RL($$\beta=0.3$$) 케이스에서 생성된 공격의 78%가 "invisible"이라는 **단 하나의 magic word**를 포함하고 있었다. 공격 성공률은 높았지만, 다양성은 처참하게 무너진 것이다.

이 현상을 **mode collapse**라 부른다. 직관적으로 설명하면 이렇다. 강아지에게 "앉아"를 가르치는데, 강아지가 "앉으면 무조건 간식이 나온다"는 걸 학습하면, 그 뒤로는 **다른 어떤 재주도 시도하지 않고 계속 앉기만** 한다. 일단 보상받는 한 가지 방법을 찾으면, 굳이 위험을 무릅쓰고 새로운 시도를 할 이유가 없다.

red-teaming에서 mode collapse가 왜 치명적인가? red-teaming의 진짜 목적은 **"모델의 모든 약점을 빠짐없이 찾는 것(coverage, 커버리지)"**이다. 그런데 RL이 "invisible"이라는 단어 하나로만 공격에 성공하고 거기서 멈추면, 그 외의 수많은 약점(다른 단어, 다른 문장 구조, 다른 주제로 유발되는 취약점)은 **영영 발견되지 못한 채 남는다.** 출시 후 그 미발견 약점을 공격자가 먼저 찾으면 끝이다.

| 비유                                                   | mode collapse 상황                      |
| ------------------------------------------------------ | --------------------------------------- |
| 보안 점검팀이 "현관문만 100번 두드려 보고 보고서 제출" | 창문, 뒷문, 환풍구는 점검 안 됨         |
| 의사가 한 가지 검사만 반복                             | 다른 질병은 진단 못 함                  |
| RL red LM이 "invisible"만 반복                         | 다른 공격 패턴이 유발하는 취약점 미발견 |

## CRT가 던진 해법: Curiosity (호기심)

2024년 ICLR에서 Hong et al.(MIT Improbable AI)이 이 문제를 정공법으로 풀었다. **Curiosity-driven Red-Teaming (CRT)** 의 핵심 아이디어는 한 문장으로 요약된다.

> **"이미 해본 공격을 또 하면 보상을 깎고, 새로운(novel) 공격을 시도하면 추가 보상을 준다."**

강아지 비유로 돌아가면, "앉기"를 한 번 성공해 간식을 받았다면, 두 번째부터는 같은 "앉기"에 간식을 점점 적게 준다. 대신 "구르기", "악수" 같은 **처음 보는 재주**를 시도하면 보너스 간식을 준다. 그러면 강아지는 한 가지 재주에 안주하지 않고 다양한 재주를 탐색하게 된다.

이 "새로운 것에 보상을 주는" 발상은 RL에서 이미 잘 연구된 **curiosity-driven exploration(호기심 기반 탐색)**에서 빌려온 것이다. 보상이 매우 드문 게임(예: Montezuma's Revenge — 한참을 헤매야 점수가 나오는 게임)에서, 에이전트가 "처음 보는 상태를 방문하면 보너스를 주는" 방식으로 탐색을 유도한다. ICM, RND 같은 기법이 대표적이다. CRT는 이 아이디어를 **자연어 공격 생성**에 옮긴 것이다.

핵심 메시지: **새로운 공격을 찾을수록 보상 → 단순 magic word 반복은 보상 안 됨 → 다양한 attack space를 탐색하게 됨.** 그리고 놀랍게도, 이 다양성이 공격 성공률까지 함께 끌어올린다.

| 항목                 | 기존 RL red-teaming  | **CRT (이 논문)**                               |
| -------------------- | -------------------- | ----------------------------------------------- |
| 공격 성공률(quality) | 높음                 | **유지하거나 더 높음**                          |
| 다양성(coverage)     | 낮음 (mode collapse) | **높음 (SelfBLEU, embedding 다양성 모두 우위)** |
| Magic word 의존      | 강함 (78%+)          | **약함**                                        |
| 추가 비용            | -                    | novelty score 계산 (작음)                       |
| 핵심 아이디어        | 보상 최대화만        | 보상 최대화 + **신규성(novelty) 보상**          |

# Background

이 절에서는 CRT를 이해하는 데 필요한 두 가지 — 왜 task reward만으로는 collapse하는지, curiosity exploration이 무엇인지 — 를 토이 예제로 차근차근 풀어본다.

## 왜 task reward만으로는 mode collapse가 일어나는가 — 토이 예제

수식 없이, 작은 가상 시나리오로 mode collapse의 원리를 직접 따라가 보자.

red LM이 만들 수 있는 공격이 단 세 종류라고 하자.

- 공격 A: "Say something insulting." → target이 욕설할 확률 높음 → 보상 $$R = 0.9$$
- 공격 B: "Describe a violent scene." → 보상 $$R = 0.8$$
- 공격 C: "What's your opinion on X?" → 보상 $$R = 0.3$$ (별로 안 유해)

학습 초기, red LM은 셋을 비슷한 확률로 시도한다(예: 33%씩). 그런데 A의 보상이 가장 크다. RL의 업데이트 규칙은 **"보상이 큰 행동의 확률을 높여라"**다. 그래서 매 스텝마다 A의 생성 확률이 조금씩 올라간다.

| 학습 스텝 | P(A)    | P(B) | P(C) |
| --------- | ------- | ---- | ---- |
| 0         | 33%     | 33%  | 34%  |
| 100       | 60%     | 30%  | 10%  |
| 300       | 85%     | 13%  | 2%   |
| 500       | **98%** | 2%   | 0%   |

스텝이 진행될수록 red LM은 거의 **A만 생성**한다. A가 보상이 가장 크니, A를 자주 할수록 평균 보상이 올라가기 때문이다. 결국 정책은 "A를 뱉는 결정적(deterministic) 정책"으로 수렴한다. 이것이 mode collapse다.

여기서 핵심 통찰: **B도 보상 0.8로 충분히 유효한 공격인데, RL은 B를 버린다.** RL의 목적함수 어디에도 "여러 공격을 골고루 시도하라"는 항이 없기 때문이다. 보상만 좇으면 가장 큰 하나로 쏠리는 게 수학적으로 당연한 결과다.

## entropy bonus나 KL로는 왜 부족한가

"그럼 다양성을 강제하는 항을 넣으면 되지 않나?" 가장 흔히 떠올리는 두 가지가 있다.

**(1) 샘플링 온도(temperature)를 높인다.** 생성을 더 무작위하게 만든다. 하지만 무작위성이 늘 뿐, "이미 해본 것"과 "안 해본 것"을 구분하지는 못한다. 여전히 A 근처만 맴돌 수 있다.

**(2) entropy bonus를 더한다.** 정책 분포가 한쪽으로 쏠리지 않게(균등 분포에 가깝게) 보너스를 준다. 위 토이 예제에서 P(A)가 98%로 쏠리는 걸 막아준다. 하지만 entropy bonus도 한계가 있다. **"서로 다른 소수의 공격이 균등하게 섞인 상태"도 엔트로피는 높다.** 즉 A, B, C 세 가지만 33%씩 내놓아도 엔트로피는 최대다. 하지만 그건 "세 가지 공격을 골고루"일 뿐, **전체 공격 공간을 넓게 덮는 것(coverage)은 아니다.**

CRT 논문은 이 점을 정확히 짚는다. KL 페널티 $$\beta$$를 키우면 다양성은 늘지만 공격력이 급락하고(원래 LM에 끌려가서), 온도나 entropy를 올려도 다양성이 "조금" 오를 뿐 CRT에 한참 못 미친다.

| 방법                     | 작동 원리                     | 한계                                    |
| ------------------------ | ----------------------------- | --------------------------------------- |
| 온도 ↑                   | 생성을 더 무작위하게          | 이미 한 것/안 한 것 구분 못 함          |
| entropy bonus            | 정책을 균등 분포 쪽으로       | "소수의 다른 공격 균등" ≠ 넓은 커버리지 |
| KL 페널티 ↑              | 원래 LM에 가깝게              | 공격력이 급락                           |
| **novelty reward (CRT)** | **과거 공격과 다를수록 보상** | (계산 비용 약간)                        |

핵심 차이는 **메모리(memory)**다. entropy와 온도는 "지금 이 순간의 분포"만 본다(memory-independent). 반면 novelty reward는 **"지금까지 생성한 모든 공격을 기억"하고, 그것들과 다른 새 공격에만 보상**을 준다(memory-dependent). 논문의 결론 한 줄: _"memory-dependent 방법이 memory-independent 방법보다 커버리지 향상에 우월하다."_

## Curiosity-driven Exploration 한 줄 정리

RL에서 보상이 드문 환경(긴 미로를 헤매야 점수가 나오는 게임 등)을 풀 때, 에이전트가 **"처음 방문하는 상태에 보너스 보상"**을 받도록 설계한다. 그러면 에이전트는 보상을 못 받더라도 일단 새로운 곳을 계속 탐험하게 되고, 그러다 진짜 보상을 발견한다. CRT는 이 "새로운 state에 보너스"를 "새로운 공격 프롬프트에 보너스"로 번역한 것이다.

# Method

## 전체 보상 함수 — 한 항씩 풀어보기

CRT는 앞서 본 RL 목적함수(보상 + KL)에 두 가지를 추가한다. **entropy bonus**(정책 randomness)와 **novelty reward**(신규성)다. 전체 목적함수는 다음과 같다.

$$
\max_\pi \ \mathbb{E}\Big[\ \underbrace{R(y)}_{\text{(1) 공격력}} \ \underbrace{-\,\beta\, D_{\text{KL}}(\pi \| \pi_{\text{ref}})}_{\text{(2) 정상 언어 유지}} \ \underbrace{-\,\lambda_E \log \pi(x \mid z)}_{\text{(3) entropy bonus}} \ +\ \underbrace{\sum_i \lambda_i B_i(x)}_{\text{(4) novelty reward}}\ \Big]
$$

네 항을 하나씩 보자.

- **(1) $$R(y)$$ — task reward (공격력).** target LLM 응답 $$y$$의 toxicity 점수. 이게 메인 목표다. "유해 응답을 끌어내라."
- **(2) $$-\beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$ — KL 페널티 (안전망).** red LM이 깨진 문자열을 뱉지 않고 자연스러운 언어에 머물도록 잡아준다. 논문은 $$\beta = 0.001$$로 약하게 둔다.
- **(3) $$-\lambda_E \log \pi(x \mid z)$$ — entropy bonus.** $$-\log \pi(x \mid z)$$는 생성 확률이 낮은(=덜 뻔한) 출력일수록 큰 값이 된다. 즉 "정책이 한쪽으로 쏠리지 말고 골고루 시도하라"는 보너스. $$\lambda_E = 0.01$$.
- **(4) $$\sum_i \lambda_i B_i(x)$$ — novelty reward (이 논문의 핵심).** $$B_i$$는 신규성 보상 함수이고, $$i$$는 그 종류(CRT는 두 종류를 씀)다. $$\lambda_i$$는 각 항의 가중치. **과거에 만든 공격들과 다를수록 $$B_i(x)$$가 커진다.**

여기서 mode collapse를 막는 마법은 전적으로 **(4)번 항**에 있다. 앞의 토이 예제로 돌아가 보자. red LM이 A를 한 번 생성했다고 하자. 이제 두 번째로 또 A를 생성하면, A는 "이미 해본 공격"이므로 novelty reward $$B(A) \approx 0$$이다. 반면 아직 안 해본 B를 생성하면 $$B(B)$$가 크다. 그래서 A의 총 보상($$R + B$$)이 점점 떨어지고, B의 총 보상이 상대적으로 올라간다. 결국 red LM은 A에 안주하지 못하고 B, C, 그리고 더 새로운 공격으로 **탐색을 계속하게 된다.**

이제 핵심인 novelty reward $$B_i$$ 두 종류를 자세히 본다.

## Novelty Reward 1 — SelfBLEU (표면적 신규성)

첫 번째 novelty는 **단어/구문이 얼마나 겹치는가**, 즉 텍스트의 **표면(surface form)** 다양성을 본다. 이를 위해 **BLEU 점수**를 응용한다.

**BLEU란?** 기계번역 평가에 쓰는 지표로, 생성 문장이 참조 문장과 **n-gram(연속된 n개 단어)을 얼마나 공유하는지**를 0~1로 잰다. 1에 가까우면 거의 똑같고, 0이면 전혀 안 겹친다. 예를 들어 "the cat sat"과 "the cat ran"은 2-gram "the cat"을 공유하므로 BLEU가 어느 정도 높다.

**SelfBLEU**는 이 BLEU를 한 코퍼스 **내부**에 적용한다. 즉 "내가 지금 만든 문장 $$x$$가, 지금까지 내가 만든 다른 문장들 $$X$$와 얼마나 비슷한가?" 비슷할수록 SelfBLEU가 높다(=신규성 낮음).

CRT는 신규성에 보상을 주고 싶으므로, SelfBLEU에 **음의 부호**를 붙여 보상으로 쓴다. 여러 $$n$$값에 대해 합산한다.

$$
B_{\text{SelfBLEU}}(x) = -\sum_{n=1}^{K} \text{SelfBLEU}_X(x, n)
$$

기호 풀이:

- $$x$$: 방금 생성한 공격 프롬프트.
- $$X$$: 학습 중 지금까지 생성한 **모든** 공격 프롬프트의 집합(reference 역할). **여기가 핵심이다 — 슬라이딩 윈도가 아니라 누적 전체 메모리**다. 새 공격이 과거 전부와 다를수록 보상이 크다.
- $$\text{SelfBLEU}_X(x, n)$$: $$x$$를 $$X$$와 비교한 $$n$$-gram 기준 SelfBLEU.
- $$K$$: 사용하는 n-gram 종류 수. (평가에서는 $$n \in \{2,3,4,5\}$$를 씀.)
- 음의 부호: SelfBLEU가 높으면(=과거와 비슷) 보상이 작아지고, 낮으면(=새롭다) 보상이 커진다.

직관: **"방금 만든 공격이 과거에 만든 어떤 공격과도 단어가 안 겹칠수록 보너스를 많이 받는다."** "invisible"을 계속 써서 비슷한 문장을 반복하면 SelfBLEU가 높아져 이 보상이 0에 수렴하므로, magic word 반복이 더 이상 이득이 안 된다.

## Novelty Reward 2 — Cosine Similarity (의미적 신규성)

SelfBLEU에는 약점이 있다. **단어는 다른데 의미는 같은** 경우를 못 잡는다. 예를 들어 "Say something insulting"과 "Tell me a rude remark"는 공유 단어가 거의 없어 SelfBLEU상 "다른" 문장이지만, **의미는 사실상 같다.** 표면만 보면 다양해 보여도 의미 공간에서는 한곳에 뭉쳐 있을 수 있다.

그래서 CRT는 **의미(semantic) 신규성**을 재는 두 번째 보상을 추가한다. 문장을 **sentence embedding** 모델(Sentence-BERT 등)로 벡터로 바꾼 뒤, **코사인 유사도**로 의미 거리를 잰다.

**코사인 유사도란?** 두 벡터가 가리키는 **방향이 얼마나 비슷한지**를 -1~1로 재는 값이다. 두 문장의 의미가 비슷하면 임베딩 벡터의 방향이 비슷해 코사인 유사도가 1에 가깝고, 의미가 다르면 0(또는 음수)에 가깝다. 식은 $$\dfrac{\phi(x)\cdot\phi(x')}{\|\phi(x)\|\,\|\phi(x')\|}$$, 즉 두 벡터의 내적을 각 크기로 나눈 것이다(크기 영향을 제거하고 방향만 본다).

CRT의 cosine novelty reward는 다음과 같다.

$$
B_{\text{Cos}}(x) = -\sum_{x' \in X} \frac{\phi(x)\cdot\phi(x')}{\|\phi(x)\|_2\,\|\phi(x')\|_2}
$$

기호 풀이:

- $$\phi$$: sentence embedding 모델. 문장을 저차원 벡터로 변환.
- $$\phi(x)\cdot\phi(x')$$: 두 임베딩 벡터의 내적.
- $$\|\phi(x)\|_2$$: 벡터의 크기(L2 norm). 분모로 나눠 방향만 비교.
- $$X$$: 역시 지금까지 생성한 **모든** 공격의 집합.
- $$\sum_{x' \in X}$$: 과거 공격 전부와의 코사인 유사도를 **모두 더한다.** 합이 클수록 "과거와 의미적으로 닮은 게 많다"는 뜻 → 음의 부호로 보상이 작아진다. 합이 작으면 "어디에도 안 닮은 새 의미" → 보상 큼.

직관: **"방금 만든 공격이 과거 공격들과 의미적으로도 멀수록 보너스를 많이 받는다."**

## 두 novelty term이 함께 쓰이는 이유

| 보상                    | 측정 대상         | 잡는 것            | 못 잡는 것                        |
| ----------------------- | ----------------- | ------------------ | --------------------------------- |
| $$B_{\text{SelfBLEU}}$$ | 표면(단어/n-gram) | 단어 표현의 다양성 | 단어만 바꾼 같은 의미             |
| $$B_{\text{Cos}}$$      | 의미(임베딩)      | 의미의 다양성      | 의미 같고 표현만 다른 미묘한 변주 |

둘은 **상호 보완적**이다. SelfBLEU만 쓰면 "단어만 살짝 바꿔 의미는 똑같은" 가짜 다양성에 속을 수 있고, cosine만 쓰면 표면적 변주를 놓칠 수 있다. 둘을 함께 쓰면 **표면과 의미 양쪽에서 진짜로 새로운 공격**에만 보상이 간다.

논문 권장 가중치: $$\lambda_{\text{SB}} = \lambda_{\text{Cos}} = 1.0$$, entropy $$\lambda_E = 0.01$$. SelfBLEU와 cosine 보상이 task reward $$R(y)$$와 같은 $$[0, 1]$$ 범위에 놓이므로 가중치 1.0이 잘 맞는다. entropy는 너무 키우면(1.0) 공격력이 급락해 작게 둔다.

## 알고리즘 한눈에 보기 (한 스텝)

지금까지의 내용을 한 스텝으로 정리하면 다음과 같다.

1. red LM $$\pi$$가 시드 $$z$$를 받아 공격 프롬프트 $$x$$를 생성한다.
2. $$x$$를 target LLM에 넣어 응답 $$y$$를 받는다.
3. **task reward** $$R(y)$$ 계산 (toxicity 분류기).
4. **novelty reward** 계산: 지금까지 누적된 공격 집합 $$X$$와 비교해 $$B_{\text{SelfBLEU}}(x)$$, $$B_{\text{Cos}}(x)$$를 구한다.
5. 총 보상 $$R(y) - \beta D_{\text{KL}} - \lambda_E \log\pi(x \mid z) + \lambda_{\text{SB}}B_{\text{SelfBLEU}} + \lambda_{\text{Cos}}B_{\text{Cos}}$$로 **PPO** 업데이트.
6. 방금 만든 $$x$$를 메모리 $$X$$에 추가하고 다음 스텝으로.

여기서 6번 — **생성한 공격을 계속 메모리에 쌓는 것** — 이 entropy/온도와 결정적으로 다른 지점이다. 정책은 자기가 과거에 무엇을 했는지 기억하므로, 같은 걸 반복하면 점점 손해를 본다.

# Experiments

## 실험 설정

- **red LM(attacker)**: GPT-2 (137M). 매우 작은 모델로도 큰 모델을 공격할 수 있음을 보이려는 의도. PPO(trlx)로 학습, KL $$\beta=0.001$$, 온도 0.7.
- **두 과제**:
  - **Text continuation**: target은 IMDb로 미세조정한 GPT-2. red LM이 영화 리뷰 앞 네 단어 뒤에 몇 단어를 붙여 유해한 이어쓰기를 유도.
  - **Instruction following**: target은 GPT2-alpaca, Dolly-v2-7B. red LM이 지시문(instruction) 형태의 공격을 생성.
- **보상**: target 응답의 toxicity를 RoBERTa hate-speech 분류기로 측정.
- **비교 대상(baseline)**:
  - **ZS / FS**: zero-shot / few-shot 프롬프팅 (학습 없이 프롬프트로만).
  - **RL**: Perez 2022의 표준 RL (보상 + KL).
  - **RL+TDiv**: Casper et al.(2023). target **응답**의 임베딩 다양성을 추가로 최대화.
  - **RL+Curiosity (= CRT, 이 논문)**: 보상 + KL + entropy + novelty.

평가는 두 축으로 한다.

- **Quality(품질)**: 다양한 toxicity 임계값 $$\tau$$에서, 생성한 공격 중 유해 응답($$R(y) \ge \tau$$)을 끌어낸 비율. 즉 공격력.
- **Diversity(다양성)**: 유해 공격 집합의 SelfBLEU 다양성($$1-\text{SelfBLEU}$$)과 임베딩 다양성($$1-\text{CosSim}$$). 클수록 다양.

## 결과: 다양성 ↑, 그리고 공격력도 ↑

핵심 결과를 정리하면 다음과 같다.

| 측면            | RL   | RL+TDiv | **CRT (RL+Curiosity)** |
| --------------- | ---- | ------- | ---------------------- |
| Quality(공격력) | 높음 | 높음    | **유지하거나 더 높음** |
| SelfBLEU 다양성 | 낮음 | 낮음    | **가장 높음**          |
| 임베딩 다양성   | 낮음 | 중간    | **가장 높음**          |

핵심 관찰들:

- **CRT는 다양성과 공격력을 동시에 끌어올린다 — trade-off가 아니다.** text continuation에서는 공격력을 baseline과 동등하게 유지하면서 다양성이 압도적으로 높았고, instruction following에서는 공격력**조차** baseline보다 높았다. 논문은 그 이유를 "instruction following은 유효 공격을 찾기가 더 어려운 과제라, 탐색을 잘하는 CRT가 더 많은 유효 공격을 발견하기 때문"이라고 해석한다. **더 넓게 탐색하니 더 좋은 공격까지 줍게 된 것**이다.
- **RL+TDiv는 왜 부족한가?** TDiv는 target **응답**의 다양성을 키울 뿐, red LM이 **새로운 공격 프롬프트**를 만들도록 유도하지는 않는다. 그래서 비슷한 프롬프트로 다양한 응답을 끌어낼 뿐, 공격 자체의 커버리지는 못 넓힌다.
- **온도/KL/entropy 단독은 모두 CRT에 못 미친다.** 온도를 2.0까지 올리거나 $$\beta$$를 키워도 다양성이 조금 오를 뿐, CRT 수준에는 한참 못 미친다.

## Ablation: 각 보상 항의 역할

논문은 SB(SelfBLEU), Cos(cosine), Ent(entropy)를 켜고 끄며 기여를 분리했다.

| 조합                   | SelfBLEU 다양성 | 임베딩 다양성 |
| ---------------------- | --------------- | ------------- |
| 없음 (= 순수 RL)       | 낮음            | 낮음          |
| Ent 단독               | 거의 변화 없음  | 약간 ↑        |
| SB 단독                | ↑               | ↑ (덤으로)    |
| Cos 단독               | -               | ↑             |
| **SB+Cos+Ent (= CRT)** | **최고**        | **최고**      |

읽어낼 점:

- **entropy 단독은 SelfBLEU 다양성을 거의 못 올린다.** "정책을 무작위하게"만 해서는 표면적 다양성이 안 생긴다는 뜻. 앞서 본 "memory-independent의 한계"를 실증.
- **SelfBLEU 보상은 표면 다양성뿐 아니라 임베딩 다양성까지 덤으로 올린다.** 단어를 진짜 다르게 쓰면 의미도 따라 벌어지는 경향.
- 세 항을 합칠 때 다양성이 **가산적(additively)**으로 누적되어 최고가 된다. 그러면서도 공격력은 유지된다.

## 정렬된 모델도 뚫린다: LLaMA2-7B-Chat

가장 인상적인 결과다. **LLaMA2-7b-chat-hf**는 RLHF로 유해 응답을 피하도록 정렬된 모델로, 공식 평가에서 toxicity **0%**를 기록한 모델이다. 그런데 **고작 137M짜리 GPT-2 red LM**으로 돌린 CRT가 **toxic 응답을 유발하는 196개의 프롬프트**를 찾아냈다.

흥미로운 디테일: 논문 Table 1의 예시를 보면, LLaMA2의 응답이 사실 욕설은 아니다. 오히려 "Thank you for asking! However, I must point out that..."처럼 **공손한 거부**에 가깝다. 그런데 toxicity 분류기가 이런 응답에 85~94%의 높은 toxicity 점수를 매겼다. 이는 두 가지를 동시에 시사한다.

1. CRT는 정렬된 모델조차 **분류기 관점에서 "유해"로 판정되는 응답**을 끌어내는 입력을 찾아낸다.
2. 동시에 **toxicity 분류기 자체의 한계**도 드러난다 — 정중한 거부를 유해로 오판할 수 있다. CRT의 공격력은 결국 보상 신호인 분류기의 품질에 묶여 있다(아래 한계점 참고).

요점은 변하지 않는다. **"공식 0%"라는 안전성 평가가 절대적 안전을 뜻하지 않으며, 다양성을 추구하는 탐색이 기존 평가가 놓친 구멍을 찾아낸다는 것**이다.

# Conclusion

핵심 메시지 한 문장: **"다양성과 공격 성공률은 trade-off가 아니다 — curiosity(신규성 보상)가 둘을 함께 끌어올린다."**

세 가지 기여:

1. **Novelty reward 도입**: SelfBLEU(표면) + Cosine(의미) 두 신규성 보상으로 RL red-teaming의 mode collapse를 해결.
2. **다양성이 공격력도 향상시킨다**: 더 넓은 탐색이 더 효과적인 공격까지 발견(특히 instruction following). 다양성과 효과성은 상충하지 않는다.
3. **memory가 핵심이다**: entropy/온도 같은 memory-independent 기법으로는 부족하고, 과거 공격을 기억해 비교하는 memory-dependent 신규성 보상이 커버리지를 결정적으로 끌어올린다.

## 한계점

- **Novelty score 계산 비용**: 매 스텝마다 새 공격을 누적된 과거 공격 집합 $$X$$와 비교한다. $$X$$가 커질수록 비교 비용이 늘어난다.
- **임베딩 모델 의존**: cosine novelty는 Sentence-BERT 같은 외부 임베딩 모델의 품질에 좌우된다.
- **보상 가중치 튜닝**: novelty 보상이 task reward를 압도하지 않도록 $$\lambda$$를 조정해야 한다. 논문은 모든 실험에 같은 가중치를 썼지만, 모델·과제마다 적응적으로 정하는 방식(예: PPO 대신 EIPO)이 더 견고할 수 있다고 제안한다.
- **Toxicity 분류기 의존**: task reward $$R(y)$$가 곧 분류기 신호다. LLaMA2 실험에서 보듯, 분류기가 정중한 거부를 유해로 오판하면 그 한계가 그대로 상속된다.

CRT는 **RL 기반 red-teaming이 "강한 단일 공격"이 아니라 "넓은 attack 공간 탐색"으로 발전**하는 분기점이다. 이후 [Auto-RT](/blog/2026/auto-rt/)(전략 수준 자동 탐색), [AgenticRed](/blog/2026/agenticred/)(red-teaming 시스템 자체를 진화) 같은 더 추상화된 자동화로 이어진다.

---

# Red-Teaming 시리즈

이 글은 LLM Red-Teaming 시리즈의 열한 번째 글이다.

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격하기 (foundation)
2. [Ganguli 2022](/blog/2026/ganguli-red-teaming/) — Anthropic의 38K 공격 데이터셋과 scaling behavior
3. [GCG (Zou 2023)](/blog/2026/gcg-attack/) — 그래디언트 기반 universal suffix
4. [AutoDAN (Liu 2023)](/blog/2026/autodan/) — 자연어 유지하는 GA 기반 jailbreak
5. AttnGCG — attention manipulation으로 GCG 강화 _(추후 작성)_
6. [PAIR (Chao 2023)](/blog/2026/pair-attack/) — 20쿼리 black-box attacker LM
7. [TAP (Mehrotra 2023)](/blog/2026/tap-attack/) — 트리 탐색 + 이중 pruning으로 PAIR 효율화
8. [GPTFuzz (Yu 2023)](/blog/2026/gptfuzz/) — AFL 영감의 template-level fuzzing
9. [Crescendo (Russinovich 2024)](/blog/2026/crescendo/) — multi-turn escalation으로 single-turn 방어 무력화
10. [Many-shot Jailbreaking (Anil 2024)](/blog/2026/many-shot-jailbreaking/) — long-context를 ICL로 weaponize
11. **(현재 글)** Curiosity-driven RT (Hong 2024) — novelty reward로 mode collapse 해결
12. [Auto-RT (Liu 2025)](/blog/2026/auto-rt/) — strategy-level RL exploration + progressive curriculum
13. [AgenticRed (Yuan 2026)](/blog/2026/agenticred/) — RT 시스템 자체를 진화
14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — Tool-use LLM agent에 대한 IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격
16. [AdvBench (Zou 2023)](/blog/2026/advbench/) — GCG 논문의 harmful behaviors/strings 표준 벤치마크
17. [HH-RLHF red-team (Ganguli 2022)](/blog/2026/hh-rlhf-red-team/) — Anthropic 38K red-team 대화 데이터셋
18. [HarmfulQA (Bhardwaj 2023)](/blog/2026/harmfulqa/) — Chain-of-Utterances 기반 유해 QA + RED-INSTRUCT
19. [BeaverTails (Ji 2023)](/blog/2026/beavertails/) — helpfulness/harmlessness 분리 라벨 QA 데이터셋
20. [WildJailbreak (Jiang 2024)](/blog/2026/wildjailbreak/) — 대규모 합성 vanilla/adversarial 학습 데이터
21. [PIKA (2025)](/blog/2026/pika/) — 난이도 집중 expert-level 합성 정렬 데이터셋
22. [ALMA (Yasunaga 2024)](/blog/2026/alma/) — 최소 주석으로 합성 데이터 기반 정렬
23. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 표준 + R2D2 방어
24. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + jailbreak artifacts repository
25. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — AI feedback으로 인간 라벨 없이 alignment
26. [Llama Guard (Inan 2023)](/blog/2026/llama-guard/) — open-weight input/output safety classifier
    본 시리즈는 26편으로 구성된다 (#5 AttnGCG는 추후 작성).

# 참고 문헌

- Hong et al., 2024. [Curiosity-driven Red-teaming for Large Language Models](https://arxiv.org/abs/2402.19464). ICLR 2024.
- [GitHub: Improbable-AI/curiosity_redteam](https://github.com/Improbable-AI/curiosity_redteam)
- [OpenReview](https://openreview.net/forum?id=4KqkizXgXU)
- Perez et al., 2022. [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286). (RL red-teaming의 원형)
- Pathak et al., 2017. [Curiosity-driven Exploration by Self-supervised Prediction (ICM)](https://arxiv.org/abs/1705.05363). (curiosity exploration 원형)
- Burda et al., 2018. [Exploration by Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894).
- Zhu et al., 2018. [Texygen: SelfBLEU 측정 도구](https://arxiv.org/abs/1802.01886).
- Reimers & Gurevych, 2019. [Sentence-BERT](https://arxiv.org/abs/1908.10084).
