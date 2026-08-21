---
layout: post
title: "안전 정렬은 첫 몇 토큰에만 얹혀 있다"
date: 2026-08-11 09:18:00 +0900
description: "RLHF Reward 설계 시리즈 #18 — shallow safety alignment가 prefilling·fine-tuning 공격을 한 번에 설명한다"
categories: [paper]
tags: [rlhf, safety, alignment, fine-tuning-attack, robustness, paper]
giscus_comments: true
related_posts: true
---

> [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) (Qi et al., Princeton University, ICLR 2025)

# Introduction

[#15 Safe RLHF](/blog/2026/safe-rlhf/), [#16 Rule-Based Rewards](/blog/2026/rule-based-rewards/), [#17 Deliberative Alignment](/blog/2026/deliberative-alignment/)까지 세 편은 전부 같은 질문을 다뤘다. 안전 reward를 **어떻게 설계할까**. Cost model을 따로 두거나(Safe RLHF), rubric 기반 reward를 쓰거나(RBR), 정책 자체가 안전 스펙을 추론하게 만들거나(deliberative alignment) — 방식은 다르지만 전부 "무엇을 거절하게 만들 것인가"에 집중했다.

이번 글이 묻는 건 다른 질문이다. 그렇게 설계한 정렬이 모델 안에 **얼마나 깊이** 박히는가. ICLR 2025 Outstanding Paper Award를 받은 이 논문은, 지금 쓰이는 safety alignment 방법들(RLHF든 SFT든) 대부분이 지름길을 탄다고 주장한다. 모델은 유해한 질문에 어떻게 답해야 하는지를 배우는 게 아니라, 응답을 **"I cannot"**처럼 시작하는 법만 배운다는 것이다. 논문은 이 현상을 **shallow safety alignment**라고 부른다.

이 관점 하나가 꽤 많은 걸 설명한다. adversarial suffix 공격(GCG), prefilling 공격, decoding parameter 조작, 그리고 최근 API로 열린 fine-tuning 공격까지 — 겉보기엔 전혀 다른 네 가지 공격이 사실 같은 구조적 약점을 찌르고 있다. 이 블로그의 red-teaming 시리즈가 공격 기법 자체를 다룬다면, 이 글은 **왜 이런 공격들이 한결같이 통하는지**, 그리고 **정렬을 더 깊게 박으려면 무엇을 바꿔야 하는지**에 집중한다.

# Background

먼저 지금 쓰이는 안전 정렬이 어떤 그림인지 짚고 가자. [#3 HH-RLHF](/blog/2026/anthropic-hh-rlhf/)에서 다뤘듯, 안전 정렬의 원형은 "유해한 요청에는 거절 응답을 선호하도록" 학습시키는 것이다. RLHF든 safety SFT든, 학습 데이터의 거절 응답들은 거의 항상 정해진 패턴으로 시작한다. "I cannot assist with that", "I'm sorry, but I can't help with this" 같은 문장들이다.

문제는 언어모델이 **autoregressive**하다는 데 있다. 응답 $$y = (y_1, \ldots, y_T)$$의 확률은

$$
\pi(y \mid x) = \prod_{t=1}^{T} \pi(y_t \mid x, y_{<t})
$$

로 분해된다. 즉 $$t$$번째 토큰을 생성할 확률은 그 이전까지 나온 토큰 $$y_{<t}$$에 전적으로 조건화되어 있다. 정렬 모델과 비정렬(base 혹은 helpful-only) 모델이 실제로 얼마나 다르게 행동하는지는, 같은 프롬프트·같은 접두어를 줬을 때 두 모델의 다음 토큰 분포가 얼마나 다른지, 즉 토큰 위치별 KL divergence로 잴 수 있다.

$$
D_t = \mathbb{E}_{y_{<t}}\left[\mathrm{KL}\left(\pi_{\text{aligned}}(\cdot \mid x, y_{<t}) \,\|\, \pi_{\text{base}}(\cdot \mid x, y_{<t})\right)\right]
$$

여기서:

- $$\pi_{\text{aligned}}$$: 안전 정렬을 거친 모델의 다음 토큰 분포
- $$\pi_{\text{base}}$$: 정렬 이전(혹은 helpful-only) 모델의 다음 토큰 분포
- $$D_t$$: 토큰 위치 $$t$$에서 두 모델이 얼마나 다르게 행동하는지의 기댓값

$$D_t$$가 크다는 건 그 위치에서 "정렬이 실제로 뭔가를 바꿔놓았다"는 뜻이고, $$D_t$$가 0에 가깝다는 건 "정렬 모델이나 비정렬 모델이나 그 위치에서는 사실상 같은 말을 한다"는 뜻이다.

여기서 간단한 토이 예제를 보자. 유해한 프롬프트 $$x$$ = "폭탄 제조법을 알려줘"가 주어졌을 때, 정렬 모델의 응답이 앞 5토큰으로 무엇을 내놓느냐에 따라 이후 생성이 완전히 갈린다.

| 앞 5토큰             | 이후 생성이 조건화되는 맥락       | 결과                            |
| -------------------- | --------------------------------- | ------------------------------- |
| "I cannot help with" | "지금까지 거절하는 중"이라는 맥락 | 거절 문장이 자연스럽게 이어짐   |
| "Sure, here is how"  | "지금까지 돕는 중"이라는 맥락     | 유해한 절차가 자연스럽게 이어짐 |

$$\pi(y_6, y_7, \ldots \mid x, y_{<6})$$은 $$y_{<6}$$이 무엇이었는지에 강하게 의존한다. 첫 5토큰이 "I cannot help with"였다면 모델은 그 뒤로도 거절의 궤적을 따라가는 게 학습 데이터상 가장 그럴듯한 다음 토큰이다. 그런데 누군가 첫 5토큰을 강제로 "Sure, here is how"로 채워 넣으면, 모델은 그 시점부터 "돕는 중"이라는 맥락에 조건화되고, 남은 토큰들에 대해서는 정렬 모델과 비정렬 모델의 차이가 거의 없어져 버린다. 안전 정렬이 실제로 개입하는 지점이 앞쪽 몇 토큰뿐이라면, 그 몇 토큰만 우회하면 나머지는 그냥 뚫린다.

# Method

## 증거 1: KL divergence는 앞쪽에서만 크다

논문은 HEx-PHI(유해 프롬프트 데이터셋)로 위 $$D_t$$를 실제로 측정했다. 결과는 토이 예제에서의 직관을 그대로 확인해준다. 토큰 위치를 앞에서부터 훑으면 KL divergence가 처음 몇 토큰에서 현저히 높다가, 뒤로 갈수록 급격히 감소해 거의 0에 수렴한다. 이 패턴은 Llama-2-7B와 Gemma-7B 두 모델 계열 모두에서 동일하게 나타난다. 논문은 이걸 "안전 정렬의 KL 예산(budget) 대부분이 응답 맨 앞 접두어에 소진된다"고 표현한다. 정렬 학습이 모델에게 실제로 가르친 건 유해한 요청을 다루는 법이 아니라, 유해한 요청 앞에서 "I cannot"으로 시작하는 습관이었다는 뜻이다.

## 증거 2: prefilling 공격

이 가설이 맞다면, 응답의 앞부분만 강제로 유해하게 채워 넣어도(prefilling) 정렬이 무너져야 한다. 실제로 Llama-2-7B-Chat에 대해 강제 주입한 유해 토큰 수를 늘려가며 공격 성공률(ASR, Attack Success Rate)을 측정한 결과는 다음과 같다.

| 강제 주입한 유해 토큰 수 | ASR         |
| ------------------------ | ----------- |
| 5                        | 42.1% ± 0.9 |
| 10                       | 51.5% ± 1.6 |
| 20                       | 56.1% ± 2.5 |
| 40                       | 57.0% ± 0.4 |

토큰 5개만 강제로 채워도 ASR이 42%를 넘고, 20개 근처에서 이미 56%까지 포화한다. 안전 정렬이 KL 예산을 앞쪽 몇 토큰에 다 써버린 게 사실이라면, 그 몇 토큰만 우회하면 이후는 사실상 비정렬 모델과 다를 바 없다는 뜻이고, 위 표가 정확히 그걸 보여준다.

## 통합 설명: 왜 여러 공격이 다 같은 구멍을 찌르는가

shallow safety alignment라는 관점 하나로 서로 무관해 보이던 공격들이 다 같은 이야기가 된다.

- **Prefilling 공격**: 앞 토큰을 직접 덮어써서 KL 예산이 소진된 구간을 건너뛴다.
- **Adversarial suffix(GCG)**: 프롬프트 뒤에 붙는 접미사를 최적화해, 모델이 첫 토큰부터 거절 대신 순응 쪽을 고르도록 유도한다. 목표 지점은 다르지만 결국 "앞쪽 몇 토큰의 선택"을 흔드는 공격이다.
- **Decoding parameter 공격**: temperature나 top-p를 조작해 앞쪽 토큰에서 거절 토큰이 뽑힐 확률을 낮춘다. 역시 앞쪽 몇 토큰의 분포를 흔드는 것으로 귀결된다.
- **Fine-tuning 공격**: 소량의 유해 예시로 파인튜닝하면, gradient가 가장 크게 움직이는 지점이 다름 아닌 응답의 첫 몇 토큰이다. 정렬이 애초에 그 몇 토큰에만 걸려 있었으니, 파인튜닝이 그 몇 토큰의 분포만 바꿔놔도 정렬 전체가 무너진다. 왜 몇 스텝, 몇 개 안 되는 예시만으로 안전장치가 통째로 풀리는지가 여기서 설명된다.

네 공격 모두 방법은 다르지만 표적은 같다. 모델이 "정렬되어 있다"고 실제로 다르게 행동하는 그 좁은 창(window)이다.

## 해법 1: safety recovery examples로 데이터 증강

가장 직접적인 해법은 학습 데이터 자체를 바꾸는 것이다. 지금까지의 거절 데이터는 전부 "처음부터 끝까지 거절"이었다. 논문은 여기에 새로운 패턴을 추가한다. **유해한 응답으로 시작했다가 도중에 스스로 멈추고 거절로 돌아오는 예시**다. 예컨대 "Sure, here's how to..."로 몇 문장을 이어가다가 "Wait, I shouldn't provide this. I cannot assist with this request." 로 꺾이는 응답을 학습 데이터에 섞는다.

이렇게 하면 정렬 모델이 배우는 신호가 더 이상 "첫 토큰에서 거절 접두어를 고르라"에 그치지 않는다. 응답 중간, 심지어 유해한 문맥이 이미 형성된 뒤에도 "지금이라도 거절로 돌아와야 한다"는 신호가 걸린다. 그 결과 정렬 모델과 비정렬 모델 사이의 divergence $$D_t$$가 앞쪽 몇 토큰을 넘어 훨씬 깊은 위치까지 유지된다.

## 해법 2: token-wise constrained objective로 fine-tuning 공격 막기

데이터 증강은 원래 정렬 단계에서 쓰는 처방이다. 그런데 fine-tuning 공격은 상황이 다르다. 사용자가 API로 직접 파인튜닝을 걸기 때문에, 서비스 제공자는 그 파인튜닝 목적함수 자체에 제약을 걸어야 한다. 논문이 제안하는 건 토큰 위치마다 서로 다른 강도로 "정렬 모델에서 너무 멀어지지 말라"는 제약을 거는 것이다.

$$
\min_\theta\ \mathbb{E}_{(x,y) \sim D}\left[-\sum_t \frac{2}{\beta_t} \log \sigma\left(\beta_t \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{aligned}}(y_t \mid x, y_{<t})}\right)\right]
$$

기호를 하나씩 풀면:

- $$\pi_\theta$$: 파인튜닝 중인 모델
- $$\pi_{\text{aligned}}$$: 파인튜닝 시작점인, 이미 안전 정렬된 모델
- $$\beta_t$$: 토큰 위치 $$t$$마다 다르게 주는 제약 강도 파라미터
- $$\sigma$$: sigmoid 함수

이 형태는 낯설지 않다. log-ratio를 sigmoid에 넣고 음의 로그를 취하는 구조는 [#24 DPO](/blog/2026/dpo/)의 목적함수와 뼈대가 같다. 다른 점은 DPO가 응답 쌍 전체에 하나의 $$\beta$$를 쓰는 반면, 여기서는 **같은 응답 안에서도 토큰 위치마다 $$\beta_t$$를 다르게** 준다는 것이다.

### $$\beta_t$$가 큰 쪽이 강한 제약이다

부호를 헷갈리기 쉬우니 미분해서 확인하자. $$u_t = \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{aligned}}(y_t \mid x, y_{<t})}$$ 로 두면 토큰 하나의 손실은 $$f(u) = \frac{2}{\beta}\log(1 + e^{-\beta u})$$ 이고, 기울기는 이렇게 된다.

$$
f'(u) = -2\big(1 - \sigma(\beta u)\big)
$$

- $$u < 0$$(정렬 모델보다 확률이 낮음)이면 기울기 크기가 최대 2까지 커져 **되돌리는 힘**이 작동한다.
- $$u > 0$$(정렬 모델보다 확률을 더 밀어올림)이면 $$\sigma(\beta u) \to 1$$ 이 되면서 기울기가 **0으로 사라진다.** 즉 정렬 모델을 넘어서 더 밀어붙일 이유가 없어진다.

여기서 $$\beta$$의 역할이 드러난다. **$$\beta$$가 클수록 $$\sigma(\beta u)$$가 빨리 포화**하므로, $$u$$가 조금만 양수가 돼도 미는 힘이 사라진다 — 정렬 모델 바로 옆에 단단히 묶이는 것이다. 반대로 $$\beta$$가 작으면 넓은 $$u$$ 구간에서 기울기가 거의 그대로 유지돼, 사실상 보통의 cross-entropy처럼 확률을 계속 밀어올린다.

숫자로 보면 분명하다. $$u = 1$$ 일 때 미는 힘의 크기는

| $$\beta$$ | $$2(1-\sigma(\beta u))$$ at $$u=1$$ | 해석                      |
| --------- | ----------------------------------- | ------------------------- |
| 2         | 약 0.24                             | 거의 멈춤 → **강한 제약** |
| 0.1       | 약 0.95                             | 거의 그대로 → **느슨**    |

즉 **큰 $$\beta_t$$ = 그 위치를 강하게 붙잡아 둔다**가 맞다.

### 그래서 설정을 읽으면

논문이 쓴 값은 $$\beta_1 = 0.5$$, $$t \in [2,5]$$ 에서 $$\beta_t = 2$$, $$t > 5$$ 에서 $$\beta_t = 0.1$$ 이다. 위 해석을 적용하면 이렇게 읽힌다.

| 토큰 위치       | $$\beta_t$$ | 제약 강도     | 의도                                        |
| --------------- | ----------- | ------------- | ------------------------------------------- |
| $$t \in [2,5]$$ | 2           | **가장 강함** | 정렬이 실제로 얹혀 있는 구간을 잠근다       |
| $$t = 1$$       | 0.5         | 중간          | 첫 토큰도 붙잡되 약간의 여지                |
| $$t > 5$$       | 0.1         | **거의 자유** | 정상 파인튜닝이 task에 적응할 여지를 남긴다 |

설계 의도가 앞의 진단과 정확히 맞물린다. **정렬이 앞 몇 토큰에만 얕게 얹혀 있다면, 파인튜닝 공격이 노리는 곳도 바로 그 앞 몇 토큰이다.** 그러니 그 좁은 창을 강하게 잠그고, 나머지 구간은 풀어줘서 **정상적인 파인튜닝 성능은 그대로 살린다.** 얕음이라는 약점을 없애는 대신, 그 얕은 구간을 방어선으로 삼아 지키는 전략이다.

실제로 이 절충이 성립한다는 게 결과로 확인된다 — 파인튜닝 공격 ASR은 88.9%에서 4.6%로 떨어지는데, 정상 파인튜닝의 유용성은 SQL Create Context 99.1% → 98.5%, Samsum ROUGE-1 51.7 → 50.1처럼 거의 유지된다.

# Experiments

## 데이터 증강의 효과

Llama-2-7B-Chat에 safety recovery examples를 섞어 재정렬한 뒤, 세 가지 공격에 대한 ASR을 다시 측정했다.

| 공격                                   | 증강 이전 | 증강 이후 |
| -------------------------------------- | --------- | --------- |
| Prefilling (40 토큰)                   | 57.0%     | **4.5%**  |
| GCG (AdvBench)                         | 65.6%     | **19.0%** |
| Decoding parameter (MaliciousInstruct) | 84.3%     | **1.0%**  |

세 공격 모두 큰 폭으로 꺾인다. 특히 decoding parameter 공격은 84.3%에서 1.0%로, prefilling 공격은 57.0%에서 4.5%로 떨어진다. GCG는 상대적으로 덜 줄지만(65.6% → 19.0%) 여전히 3배 넘게 개선된다. 유용성 손실은 미미하다. AlpacaEval 승률이 51.8%에서 49.5%로, 오차 범위 안에서 소폭 하락하는 데 그친다. 안전을 깊게 박는 데 드는 비용이 생각보다 크지 않다는 뜻이다.

## Token-wise constrained objective의 효과

파인튜닝 공격에 대해서는, 표준 SFT로 파인튜닝했을 때와 앞서의 제약 목적함수로 파인튜닝했을 때를 비교한다.

| 공격 유형              | 표준 SFT | 제약 SFT  |
| ---------------------- | -------- | --------- |
| Harmful Examples       | 88.9%    | **4.6%**  |
| Identity Shifting      | 79.5%    | **8.1%**  |
| Backdoor (트리거 포함) | 90.9%    | **10.9%** |

세 가지 공격 시나리오 모두 표준 SFT에서는 ASR이 80\~90%대로 사실상 정렬이 완전히 무력화된다. 반면 제약을 건 SFT는 5\~11% 수준으로 억제된다. 그리고 이 억제가 정상적인 파인튜닝 용도까지 망가뜨리는 건 아닌지가 관건인데, 결과는 다음과 같다.

| 벤치마크           | 표준 SFT | 제약 SFT |
| ------------------ | -------- | -------- |
| SQL Create Context | 99.1%    | 98.5%    |
| GSM8K              | 41.7%    | 37.4%    |
| Samsum (ROUGE-1)   | 51.7%    | 50.1%    |

세 벤치마크 모두 성능 하락이 1\~5%p 안쪽이다. GSM8K가 상대적으로 가장 많이 떨어지긴 하지만, 정렬을 우회하는 통로를 막는 대가로는 감수할 만한 수준이다.

# Conclusion

안전 정렬은 기본적으로 얕다. 거절 예시로만 SFT를 걸면, 모델은 유해한 내용을 다루는 법이 아니라 응답을 거절 접두어로 시작하는 습관만 배운다. 이 하나의 사실이 adversarial suffix, prefilling, decoding parameter, fine-tuning 네 가지 공격을 전부 설명한다. 그리고 정렬을 깊게 만드는 처방도 이 진단에서 그대로 따라 나온다. (a) 유해한 응답으로 시작했다가 거절로 되돌아오는, 응답 뒷부분까지 안전 신호가 걸리는 데이터를 학습에 넣어야 하고, (b) 파인튜닝 API를 외부에 열어줄 경우에는 앞쪽 토큰을 특히 강하게 보호하는 위치별 제약을 파인튜닝 목적함수 자체에 걸어야 한다. [#44 프론티어 모델의 reward 설계](/blog/2026/frontier-reward-design/)에서 다루듯, 실제 서비스에서 파인튜닝 API를 제공하는 순간 이 문제는 이론이 아니라 운영 리스크가 된다.

다만 이 논문의 처방이 모든 공격을 원천 봉쇄하는 건 아니다. 데이터 증강과 제약 목적함수 모두 논문이 테스트한 공격 종류에 한정된 결과이고, 더 정교하게 설계된 새로운 공격이 같은 구멍을 다른 방식으로 다시 찌를 가능성은 남아 있다. shallow safety alignment는 진단이지 만병통치약이 아니다. "정렬이 얼마나 깊이 박혀 있는가"를 측정하고 개선하는 하나의 축일 뿐, 안전성 자체를 보장하는 개념은 아니다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 열여덟 번째 글이다.

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
  <li><strong>(현재 글)</strong> Shallow Safety Alignment (2024) — 정렬은 첫 몇 토큰에만 얹혀 있다</li>
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

- Qi et al. (Princeton University), 2024. [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) (ICLR 2025 Outstanding Paper).
- Bai et al. (Anthropic), 2022. [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) — [#3](/blog/2026/anthropic-hh-rlhf/), 거절 위주 안전 학습의 원형.
- Dai et al. (Peking University), 2023. [Safe RLHF](https://arxiv.org/abs/2310.12773) — [#15](/blog/2026/safe-rlhf/).
- Mu et al. (OpenAI), 2024. [Rule Based Rewards for Language Model Safety](https://arxiv.org/abs/2411.01111) — [#16](/blog/2026/rule-based-rewards/).
- Guan et al. (OpenAI), 2024. [Deliberative Alignment](https://arxiv.org/abs/2412.16339) — [#17](/blog/2026/deliberative-alignment/).
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — [#24](/blog/2026/dpo/), 제약 목적함수와 닮은 log-ratio·sigmoid 구조.
