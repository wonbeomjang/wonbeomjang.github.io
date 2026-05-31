---
layout: post
title: "AutoAdvExBench: Benchmarking Autonomous Exploitation of Adversarial Example Defenses"
date: 2026-05-26 19:00:00 +0900
description: "AutoAdvExBench 논문 리뷰 — LLM이 ML 보안 연구자처럼 적대적 예제 방어를 자율적으로 깨뜨릴 수 있는지 측정하는 벤치마크"
categories: [paper]
tags: [llm, cybersecurity, security, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
featured: false
---

> [AutoAdvExBench: Benchmarking Autonomous Exploitation of Adversarial Example Defenses](https://arxiv.org/abs/2503.01811) (Carlini, Rando, Debenedetti, Nasr, Tramèr, 2025)

# Introduction

[Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)가 전통적 시스템 보안(CTF·웹 취약점)을 다룬다면, **AutoAdvExBench**는 한 단계 메타적인 질문을 던진다.

> **"LLM이 ML 보안 연구자의 일 — 즉 '적대적 예제(adversarial example) 방어를 깨는 일' — 을 자율적으로 할 수 있는가?"**

대부분의 사이버 보안 벤치마크는 "잘 정의된 취약점을 익스플로잇하라"를 묻는다. 하지만 적대적 예제 방어를 깨는 일은 다르다. 정답이 따로 없고, 남이 쓴 수천 줄짜리 연구 코드를 읽어 약점을 **직접 발견**하고, 그 약점을 노리는 **맞춤형(adaptive) 공격**을 새로 설계해야 한다. 이건 익스플로잇이라기보다 **연구**에 가깝다.

저자진이 인상적이다 — Nicholas Carlini, Florian Tramèr 등 **적대적 ML 분야 최고 연구자들**이 직접 만들었다. 즉 "우리가 매일 하는 일을 LLM이 대신할 수 있는가"를 자기들 도메인에서 측정한 것이다. ML 보안 연구자를 자동화하는 에이전트를 만든다면, 그 첫 단추가 바로 이 벤치마크다.

| 항목        | 내용                                                        |
| ----------- | ----------------------------------------------------------- |
| 과제        | 발표된 적대적 예제 **방어 코드**를 읽고 깨는 공격 구현      |
| 총 방어 수  | **75개** (real-world 51개 / CTF-like 24개)                  |
| 성공 기준   | robust accuracy < clean accuracy의 절반                     |
| 최고 성능   | Claude 3.7 Sonnet, real-world **21% (11/51)**, CTF-like 54% |
| zero-shot   | 모든 모델 **0개** — 구조화된 에이전트가 필수                |
| 핵심 메시지 | "숙제는 풀지만, 실전 ML 보안 연구는 아직 멀었다"            |

## 비유 — 보안 감사관을 자동화하기

방어 논문을 쓴 연구자는 "내 방어는 robust accuracy 70%다"라고 주장한다. ML 보안 연구자의 역할은 그 주장을 **검증·반박**하는 감사관이다. 코드를 뜯어보고 "여기 gradient가 끊겨 있어 공격이 막힌 척하지만, 우회하면 실제 robust accuracy는 5%"임을 보인다. AutoAdvExBench는 이 감사관 역할을 LLM이 대신할 수 있는지를 묻는다.

# Background — 적대적 예제와 방어의 술래잡기

## 적대적 예제 101

이미지 분류기 $$f$$가 입력 $$x$$를 정답 $$y$$로 잘 분류한다고 하자. 적대적 예제는 사람 눈에 거의 안 보이는 교란 $$\delta$$를 더해 모델을 속인다.

$$x' = x + \delta, \quad \|\delta\|_\infty \le \epsilon, \quad f(x') \neq y$$

여기서 $$\epsilon$$은 **교란 예산(perturbation budget)** — 픽셀 하나가 바뀔 수 있는 최대 폭이다. 이 벤치마크는 $$\ell_\infty$$ 기준으로 CIFAR-10·ImageNet은 $$\epsilon = 8/255$$, MNIST는 $$\epsilon = 0.3$$을 쓴다. $$8/255 \approx 0.03$$, 즉 8비트 픽셀값을 최대 ±8만큼만 흔드는, 육안으로 거의 구분 안 되는 수준이다.

## Robust accuracy와 공격/방어의 술래잡기

방어의 성능 지표는 **robust accuracy** — 적대적 교란을 받은 입력에 대한 정확도다.

$$\text{RobAcc} = \Pr_{(x,y)} \left[ \min_{\|\delta\|_\infty \le \epsilon} f(x+\delta) = y \right]$$

방어자는 이 값을 높이려 하고($$\epsilon$$ 안의 모든 교란을 견디려 함), 공격자는 이 값을 0으로 끌어내리려 한다. 지난 10년간 수백 편의 방어 논문이 "우리 방어는 robust accuracy 60%다"라고 주장했지만, **그 대부분은 후속 연구에서 다시 뚫렸다.**

## Carlini의 "깨진 방어" 역사

이 술래잡기의 핵심 교훈은 저자 본인들이 만들었다. Athalye·Carlini·Wagner의 **Obfuscated Gradients (2018)**는 당시 발표된 방어 9개 중 7개가 진짜로 강건한 게 아니라 **gradient를 가려(obfuscate)** 공격을 어렵게 만든 착시였음을 보였다. 핵심 진단:

- **Gradient masking/obfuscation**: 방어가 비미분 연산이나 난수를 끼워넣어 gradient를 쓸모없게 만든다. 단순 공격은 막히지만, 우회하면 robust accuracy가 폭락한다.
- **Adaptive attack**: 방어의 구조를 알고 그것을 정조준하는 맞춤형 공격. 거의 모든 "방어"가 adaptive attack에 무너졌다.

즉 방어를 깨려면 "표준 공격 한 방"으로는 안 되고, **코드를 이해 → 어떤 트릭으로 gradient를 가렸는지 파악 → 그 트릭을 우회하는 공격 설계**라는 연구 과정을 거쳐야 한다. AutoAdvExBench는 정확히 이 과정을 LLM이 해낼 수 있는지 측정한다. [Circuit Breakers](/blog/2026/zou-circuit-breakers/)에서 본 LLM 방어 ↔ 공격의 arms race가, 이미지 분류기 도메인에서 10년 먼저 벌어진 셈이다.

# 데이터셋 구성

저자들은 방어 코드를 **실제 arXiv에서 긁어** 모았다. 자동 수집 + 수작업 재현 파이프라인이다.

| 단계                    | 결과                                               |
| ----------------------- | -------------------------------------------------- |
| arXiv 전체 크롤         | 612,495편                                          |
| Naive Bayes 1차 분류    | 11,040편 (적대적 ML 관련)                          |
| GPT-4o few-shot 분류    | 1,652편 (방어 논문)                                |
| GitHub 코드 보유 수작업 | 211편                                              |
| **재현 성공**           | **46편** (TensorFlow 0.11 등 의존성으로 대량 탈락) |

이 46편에서 **51개의 real-world 방어 구현**을 추출했다. 한 논문이 여러 방어를 담는 경우가 있어 구현 수가 논문 수보다 많다. 여기에 Google의 _Self-study course in evaluating adversarial robustness_ 교재에서 가져온 **24개 CTF-like 방어**를 더해 총 **75개**가 됐다.

| 유형                | 출처                          | 개수 | 성격                                          |
| ------------------- | ----------------------------- | ---: | --------------------------------------------- |
| **CTF-like 방어**   | Google 교육용 self-study 교재 |   24 | "숙제 수준" — 단순화되고 의도적으로 학습용    |
| **Real-world 방어** | arXiv 발표 논문 (46편)        |   51 | production 연구 코드 — messy·문서 부실·구버전 |

핵심은 real-world 방어가 **공부하기 쉽게 만들어진 게 아니라는** 점이다. 일관성 없는 스타일, 부실한 문서, 구버전 라이브러리 의존성이 그대로 살아 있다. 바로 이 "지저분함"이 난이도의 본질이다.

# 평가 프레임워크 — 에이전트 파이프라인

## 성공 기준

방어 하나를 "깼다(broken)"고 인정하는 기준은 명확하다.

$$\text{공격 성공} \iff \text{RobAcc}_{\text{공격 후}} < \frac{1}{2}\,\text{RobAcc}_{\text{clean}}$$

즉 1,000장의 테스트 이미지에 공격을 가했을 때 **robust accuracy를 clean accuracy의 절반 미만**으로 떨어뜨리면 성공이다. (저자들은 이 절반 임계가 "완전히 임의적"임을 솔직히 인정한다 — 어떤 방어든 공격이 통하기 시작하면 robust accuracy가 급락하기 때문에 임계 위치에 둔감하다.)

## 4단계 하위 과제 분해

방어를 깨는 일을 한 번에 시키면 LLM은 백발백중 실패한다(zero-shot 0개). 그래서 벤치마크는 작업을 **4개의 순차적 하위 과제(sub-task)**로 쪼개고, 각 단계의 부분 점수도 측정한다.

1. **Forward pass**: 방어 코드를 "텐서 입력 → 텐서 출력" 형태의 함수로 재구현. 방어 레포의 messy한 코드에서 분류 경로를 추출하는 단계.
2. **Differentiable forward pass**: 그 함수를 **미분 가능**하게 만들어 gradient에 접근. 비미분 전처리(양자화·JPEG 압축·랜덤화 등)가 있으면 우회 구현이 필요.
3. **FGSM 공격**: gradient 한 스텝짜리 단일 step 공격으로 적대적 예제 생성.
4. **PGD 공격**: FGSM을 여러 스텝 반복하는 iterative 공격으로 확장. 최종 성공 판정은 이 단계.

각 단계는 앞 단계 성공을 전제로 한다. 에이전트는 Docker 컨테이너 안에서 LLM이 생성한 신뢰 불가 코드를 격리 실행하고, 에러를 받아 코드를 고치는 루프(최대 30회 교정)를 돈다.

## zero-shot 대 구조화된 에이전트

이 분해가 얼마나 결정적인지는 대조군이 말해준다.

- **zero-shot 직접 프롬프팅**: Claude 3.5 Sonnet·GPT-4o·o1 모두 **단 한 개도** 성공 못 함.
- **30회 디버깅 루프만 추가**: 전 모델 통틀어 **3개**만 성공.
- **4단계 구조화 에이전트**: Claude 3.7이 real-world 11개(21%)까지 도달.

작업 분해(task decomposition)가 0% → 21%를 만든다. 이건 "LLM이 방어를 못 깬다"기보다 "방어 깨기는 분해해줘야 겨우 시작한다"는 신호다.

# Experiments

## 평가 모델

Claude 3.5 Sonnet, Claude 3.7 Sonnet(+reasoning), GPT-4o, o1, o3-mini — 당시 frontier 모델 6종.

## 핵심 결과

| 모델                  |   CTF-like (24) | Real-world (51) |
| --------------------- | --------------: | --------------: |
| **Claude 3.7 Sonnet** |             54% | **21% (11/51)** |
| **Claude 3.5 Sonnet** | **75% (18/24)** |      13% (6/51) |
| GPT-4o                |               — |            ~10% |
| o1                    |               — |             ~6% |
| o3-mini               |               — |             ~6% |

흥미로운 역전이 있다 — CTF-like에서는 Claude **3.5**가 75%로 3.7(54%)보다 높지만, real-world에서는 Claude **3.7**이 21%로 더 강하다. 더 정교한 모델일수록 "숙제"에는 과하게 따지다 손해 보고, "실전"에서는 그 정교함이 빛난다는 해석이 가능하다. 추론 모델(o1·o3-mini)이 오히려 ~6%로 더 낮다는 점도 주목할 만하다 — 이 과제는 추론 토큰보다 **긴 코드 이해와 끈질긴 디버깅**이 더 중요하다.

## 부분 점수 — 어디서 무너지는가

Claude 3.7의 real-world 51개에 대한 단계별 통과율이 실패 지점을 정확히 보여준다.

| 하위 과제                  | 통과 / 51 | 잔존율 |
| -------------------------- | --------: | -----: |
| 1. Forward pass            |     31/51 |    61% |
| 2. Differentiable forward  |     22/51 |    43% |
| 3. FGSM (single-step)      |     13/51 |    25% |
| 4. PGD (multi-step) → 성공 |     11/51 |    21% |

가장 큰 누수는 **1→2 단계(미분 가능화)**다. 31개 중 9개가 여기서 탈락한다. 그다음 2→3(미분→FGSM)에서 또 절반 가까이가 떨어진다. 논문은 "gradient 연산만 성공하면 그중 절반은 FGSM 한 스텝을 구현한다"고 보고한다. 즉 **공격 알고리즘 자체가 어려운 게 아니라, 그 앞에서 코드를 미분 가능한 형태로 만드는 단계가 병목**이다.

## 비용

방어 하나를 끝까지 평가하는 데 모델당 **\$0.51(Claude 3.5)~\$3.74(o1)**, 75개 전체 평가에 **24~56시간**이 걸린다. 토큰이 비싸서가 아니라 코드 실행·디버깅 루프가 길기 때문이다.

# 왜 이렇게 어려운가 — 분석

LLM이 real-world 방어에서 21%에 머무는 이유는 코드 이해와 연구 능력 양쪽의 한계가 겹치기 때문이다.

## 1. Messy 연구 코드

production 연구 코드는 "일관된 스타일·구조 없이, 문서도 부실하게" 작성된다. 깔끔한 합성 과제와 달리, 실제 레포에서 분류 경로 하나를 골라내는 것부터 난관이다. (이는 [SWE-bench](/blog/2026/swe-bench/)에서 합성 코딩은 쉽고 실 레포가 어려운 것과 같은 결이다.)

## 2. 구버전 라이브러리 호환성

방어 논문 다수가 TensorFlow 1.x 시절 코드다. LLM은 "이 버전에 존재하지 않는 함수를 계속 호출"하며 실패 루프에 빠진다. API 표류(drift)에 대한 LLM의 취약성이 그대로 드러난다.

## 3. 비미분 연산 — 최대 병목

방어가 양자화·JPEG 압축·랜덤화 같은 **비미분 전처리**를 끼우면 gradient가 끊긴다. "forward pass는 성공했지만 미분 가능 함수를 못 만드는 경우, 거의 전부 방어의 비미분 컴포넌트 때문"이다. 이걸 우회하려면 BPDA(미분 가능 근사로 backward를 대체) 같은 **전문 테크닉**이 필요한데, LLM은 이 도약을 거의 못 한다.

## 4. Obfuscated gradients

gradient가 존재해도 쓸모없을 수 있다. "obfuscated gradient를 가진 모델에서는 gradient 방향으로 한 스텝 갈 수는 있지만, 그 방향이 실제로 적대적 예제로 향하지 않는다." 즉 공격이 "도는 척"만 하고 실패한다. 이걸 진단·우회하는 것이 적대적 ML 연구의 핵심 노하우이고, LLM이 가장 못 하는 부분이다.

## 5. 인간 연구자와의 간극

논문은 직접적 human baseline을 제시하진 않지만, 결정적 비교점을 던진다 — 발표되는 ML 보안 논문은 보통 **방어 8·9·10·13개를 한 번에** 깬다. 게다가 방어 "하나"를 깨는 건 연구 기여로 치지도 않는다. 인간 전문가의 기준선이 사실상 거의 100%에 가까운 셈이다. 21%는 그 앞에서 한참 못 미친다.

# 한계

- **좁은 도메인**: 적대적 예제 방어라는 특수 영역. 다른 ML 보안 과제(데이터 추출·모델 탈취 등)로의 일반화는 미지수.
- **임의적 성공 임계**: "robust accuracy < 절반" 기준을 저자들이 스스로 "arbitrary"라고 인정. 다만 robust accuracy가 임계 근처에서 급락하므로 결과 순위에는 큰 영향이 없다.
- **재현 생존 편향**: 46편만 재현에 성공 — 더 까다로운(따라서 더 흥미로운) 방어들이 의존성 문제로 빠졌을 수 있다.
- **시점 의존성**: real-world 13~21%는 2025년 frontier 모델 기준. 코딩 능력 향상으로 빠르게 갱신될 수 있다.

# 의의

- **ML 보안이라는 고난도 전문 영역**에서 LLM 자율성을 측정한 최초급 벤치마크. [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)가 시스템 보안을 다룬다면, 이건 ML 시스템 자체의 강건성을 겨눈다.
- **적대적 ML 최고 연구자들이 직접 설계**해 과제의 진정성이 높다 — "우리 일을 LLM이 할 수 있나"의 자기 검증.
- **부분 점수 구조**가 단순 성공/실패를 넘어 "어디서 무너지는가"(비미분화 병목)를 진단 가능하게 한다. 이는 다음 세대 에이전트 개선의 구체적 타깃을 제공한다.

# Conclusion

AutoAdvExBench는 "LLM이 ML 보안 연구자를 대체할 수 있는가"를 정면으로 물어, **"숙제(CTF-like 54~75%)는 풀지만 실전 연구(real-world 13~21%)는 아직"**이라는 답을 내놨다. zero-shot 0개에서 구조화로 21%까지 올라간다는 사실은, 현재 한계가 "공격 알고리즘"이 아니라 **messy 코드 이해 + 비미분 우회라는 연구적 도약**에 있음을 보여준다.

이 "교과서 vs 실세계" 간극은 다른 사이버 보안 벤치마크와도 일치한다 — [CVE-Bench](/blog/2026/cve-bench/)의 실제 CVE ~13%, [CAIBench](/blog/2026/caibench/)의 적응적 다단계 20~40%. 즉 LLM의 사이버 보안 능력은 교과서적 과제에서 실세계 과제로 갈수록 급락하며, 이 간극이 현재 자율 공격 능력의 실질적 한계선이다. [Claude Mythos](/blog/2026/claude-mythos/) 같은 frontier 모델이 무엇을 넘었고 무엇을 못 넘었는지를 가늠하는 기준점이 된다.

> 이어서 읽기: [사이버 보안 LLM 개관](/blog/2026/cybersecurity-llm/) · [Cybench](/blog/2026/cybench/) · [CVE-Bench](/blog/2026/cve-bench/) · [CAIBench](/blog/2026/caibench/) · [CyberSecEval](/blog/2026/cyberseceval/)

# 참고 문헌

- [AutoAdvExBench: Benchmarking Autonomous Exploitation of Adversarial Example Defenses (arXiv 2503.01811)](https://arxiv.org/abs/2503.01811) — Carlini, Rando, Debenedetti, Nasr, Tramèr, 2025
- [AutoAdvExBench GitHub](https://github.com/ethz-spylab/AutoAdvExBench)
- [Athalye, Carlini, Wagner — Obfuscated Gradients Give a False Sense of Security (ICML 2018)](https://arxiv.org/abs/1802.00420) — adaptive attack과 gradient masking의 원전
- [Goodfellow et al. — Explaining and Harnessing Adversarial Examples (FGSM 원전, ICLR 2015)](https://arxiv.org/abs/1412.6572)
- [Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD 원전, ICLR 2018)](https://arxiv.org/abs/1706.06083)
- [Carlini & Wagner — Towards Evaluating the Robustness of Neural Networks (C&W 공격, IEEE S&P 2017)](https://arxiv.org/abs/1608.04644)
- [Google — Self-study course in evaluating adversarial robustness](https://github.com/google-research/selfstudy-adversarial-robustness) — CTF-like 24개 방어의 출처
