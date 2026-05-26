---
layout: post
title: "에이전트란 무엇인가: 지능형 에이전트의 고전 정의부터 LLM 에이전트까지"
date: 2026-05-25 23:00:00 +0900
description: "agent 벤치마크 시리즈의 도입부 — Russell & Norvig의 지능형 에이전트 정의(합리성, 기대효용, PEAS, MDP/POMDP, 5유형, 환경 속성)부터 Lilian Weng·Anthropic의 LLM 에이전트 해부까지, 수식과 함께"
categories: [paper]
tags: [llm, agent, concept, intelligent-agent, benchmark, paper]
giscus_comments: true
related_posts: true
featured: false
---

> 이 글은 agent 평가 벤치마크 시리즈([AgentBench](/blog/2026/agentbench/), [GAIA](/blog/2026/gaia/), [SWE-bench](/blog/2026/swe-bench/), [TravelPlanner](/blog/2026/travelplanner/), [MedAgentBench](/blog/2026/medagentbench/), [OSWorld](/blog/2026/osworld/))의 도입부다. 논문 리뷰로 들어가기 전에 **"에이전트란 무엇인가", "지능형 에이전트란 무엇인가"**라는 일반론을, 가능한 한 형식적으로(수식과 함께) 정리한다.

# Introduction

요즘 "AI 에이전트"라는 말은 너무 흔해져서 오히려 정의가 흐릿해졌다. 챗봇도 에이전트라 부르고, 코드를 짜주는 도구도 에이전트라 부르고, 자율주행차도 에이전트라 부른다. 도대체 무엇이 에이전트이고 무엇이 아닌가? 그리고 "**지능형**"이라는 수식어는 정확히 무엇을 더하는가?

다행히 이 질문에는 30년 가까이 다듬어진 **표준 답안**이 있다. 1995년 Russell과 Norvig의 교과서 _Artificial Intelligence: A Modern Approach_(AIMA)가 제시한 **지능형 에이전트(intelligent agent)** 프레임워크다. 그 핵심 주장은 한 문장으로 요약된다.

> **지능 = 합리성(rationality) = 기대효용의 최대화.**

흥미로운 점은, 이 고전적 정의가 2023년 이후의 **LLM 에이전트**에도 거의 그대로 적용된다는 것이다. 바뀐 것은 의사결정을 수행하는 "두뇌"가 명시적 탐색·계획 알고리즘에서 LLM으로 교체되었을 뿐, **"환경을 지각하고, 기대 성과를 최대화하도록 행동한다"**는 골격은 동일하다.

이 글의 흐름은 다음과 같다. 먼저 에이전트를 형식적으로 정의하고(1장), 그 위에서 "합리성"을 기대효용으로 수식화한다(2장). 행동이 미래에 영향을 주는 순차적 환경을 다루기 위해 MDP·POMDP와 Bellman 방정식을 도입하고(3장), 이 이론 위에서 AIMA의 5가지 에이전트 유형을 재해석한다(4장). 이어 환경의 속성(5장)을 정리한 뒤, 두뇌가 LLM으로 교체되는 과정(6~8장)과 완벽한 합리성이 불가능한 이유(9장)를 거쳐, 결국 벤치마크가 무엇을 측정하는지(10장)로 시리즈에 연결한다.

# 1. 에이전트의 형식적 정의

AIMA의 정의는 한 문장으로 시작한다.

> "An agent is **anything that perceives its environment** (through sensors) **and acts upon that environment** (through actuators)."

즉 에이전트의 본질은 **지각(perceive) → 행동(act)의 순환 루프**다. 인간(눈·귀 → 손·발), 로봇(카메라 → 모터), 소프트웨어(파일·패킷 입력 → 화면·네트워크 출력) 모두 같은 틀로 설명된다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_1.svg" width="100%">
</p>

## 지각, 지각 이력, 에이전트 함수

매 시점 센서가 받는 입력 하나를 **지각(percept)**이라 하고, 시작부터 현재까지 받은 지각의 전체 나열을 **지각 이력(percept sequence)**이라 한다. 에이전트의 행동은 원리적으로 이 지각 이력 전체의 함수로 표현된다. 이를 **에이전트 함수(agent function)**라 부른다.

$$f: P^{*} \to A$$

여기서 $$P$$는 가능한 지각의 집합, $$P^{*}$$는 임의 길이의 지각 이력 전체 집합(클레이니 스타), $$A$$는 행동 집합이다. 중요한 점은 행동이 "지금 보이는 것"이 아니라 **"지금까지 본 모든 것"**의 함수라는 것이다. 이 덕분에 기억·추론이 정의에 자연스럽게 포함된다.

에이전트 함수는 "에이전트가 **무엇을** 해야 하는가"를 정의하는 이상(理想)이고, 이를 실제로 구현한 것이 **에이전트 프로그램(agent program)**이다. AIMA는 이를 다음과 같이 정리한다.

$$\text{에이전트} = \text{아키텍처(architecture)} + \text{프로그램(program)}$$

## 왜 테이블로는 안 되는가

에이전트 함수를 "지각 이력 → 행동" 테이블로 그냥 저장하면 안 될까? 길이 $$T$$까지의 지각 이력만 따져도 경우의 수는

$$\sum_{t=1}^{T} \lvert P \rvert^{\,t}$$

로 $$T$$에 대해 지수적으로 폭발한다. 체스만 해도 지각 이력의 수가 천문학적이라 테이블은 물리적으로 불가능하다. 그래서 우리는 **이력을 압축적으로 요약하는 프로그램**(상태, 모델, 정책)을 만들어야 한다. 이 "압축의 방식"이 곧 뒤에서 볼 5가지 에이전트 유형의 차이를 만든다.

# 2. 합리성이란 무엇인가 — 기대효용 최대화

단순히 지각하고 행동한다고 "지능형"인 것은 아니다. 온도계도 지각하고 반응하지만 우리는 그것을 지능적이라 하지 않는다. AIMA가 더하는 결정적 조건은 **합리성(rationality)**이다.

> "For each possible percept sequence, a rational agent selects an action **expected to maximize its performance measure**, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has."

핵심 단어는 **성과 척도(performance measure)**와 **기대(expected)**다. 합리성은 결과론적 성공이 아니라, **주어진 정보 하에서 기대 성과를 최대화하는 의사결정**이다.

## 기대효용과 MEU 원리

행동 $$a$$가 어떤 결과 상태 $$s'$$로 이어질지에 불확실성이 있다고 하자. 증거 $$e$$(지각 이력)가 주어졌을 때, 행동 $$a$$의 **기대효용(expected utility)**은 가능한 결과들에 대한 효용의 기대값이다.

$$EU(a \mid e) = \sum_{s'} P\big(\text{Result}(a) = s' \mid a, e\big)\, U(s')$$

여기서 $$U(\cdot)$$는 결과 상태에 매기는 **효용 함수(utility function)**, $$P(\cdot \mid a, e)$$는 행동의 결과에 대한 믿음이다. 합리적 에이전트는 기대효용을 최대화하는 행동을 고른다. 이것이 **최대 기대효용 원리(Maximum Expected Utility, MEU)**다.

$$a^{*} = \arg\max_{a \in A} \; EU(a \mid e)$$

이 한 줄이 이 글 전체를 관통한다. 뒤에서 utility-based agent, MDP의 최적 정책, 그리고 LLM 에이전트의 벤치마크 success rate가 모두 이 식의 변주다.

## 효용은 왜 존재하는가 — von Neumann–Morgenstern

"그런데 왜 하필 기대값을 최대화해야 합리적인가? 효용 함수라는 게 임의적인 것 아닌가?"라는 의문이 자연스럽다. von Neumann–Morgenstern 효용 정리는 이에 답한다. 에이전트의 선호(preference)가 몇 가지 합리성 공리 — **완전성(completeness), 이행성(transitivity), 연속성(continuity), 독립성(independence)** — 를 만족하면, 그 선호를 표현하는 효용 함수 $$U$$가 존재하며, **합리적 선택은 곧 기대효용 최대화와 동치**임이 증명된다.

$$
a \succeq b \;\;\Longleftrightarrow\;\; EU(a) \ge EU(b)
$$

즉 "효용을 최대화하라"는 것은 임의의 규칙이 아니라, **일관된 선호를 가진다는 가정으로부터 따라 나오는 정리**다. 이행성이 깨진 에이전트(예: A>B, B>C, C>A)는 돈을 무한히 잃는 "Dutch book"에 걸리므로 비합리적이다.

## 합리성 ≠ 전지·완벽

오해를 막기 위한 AIMA의 강조점 하나. 합리성은 **전지(omniscience)도, 완벽한 결과도, 미래 예지도 아니다.** 합리성은 4가지에만 의존한다.

1. 성과 척도 (무엇이 좋은가의 정의)
2. 에이전트의 사전 지식
3. 에이전트가 취할 수 있는 행동
4. 지금까지의 지각 이력

길을 건너다 하늘에서 떨어진 운석에 맞았다면, 그 행동은 **불운**했을 뿐 비합리적이었던 것은 아니다. 이 구분은 뒤에서 LLM 에이전트가 "운/노이즈로 실패한 것"과 "비합리적으로 실패한 것"을 가르는 데에도 그대로 쓰인다.

## PEAS — 에이전트를 설계하는 4요소

합리적 에이전트를 설계하려면 먼저 성과 척도와 환경을 명세해야 한다. AIMA는 이를 **PEAS**로 정리한다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_2.svg" width="42%">
</p>

자율주행차를 예로 들면 다음과 같다.

| 요소            | 자율주행차 예시                     |
| --------------- | ----------------------------------- |
| **Performance** | 안전, 속도, 법규 준수, 승차감, 연비 |
| **Environment** | 도로, 다른 차량·보행자, 신호, 날씨  |
| **Actuators**   | 핸들, 가속·제동 페달, 방향지시등    |
| **Sensors**     | 카메라, LiDAR, GPS, 속도계          |

성과 척도(P)가 곧 2장의 효용 $$U$$이고, 환경(E)이 결과 확률 $$P(s' \mid a, e)$$를 정의한다. 즉 PEAS는 MEU 식을 현실 문제에 끼워 넣기 위한 슬롯이다.

# 3. 순차적 의사결정: MDP와 POMDP

지금까지는 "한 번의 행동"을 다뤘다. 그러나 대부분의 에이전트는 **여러 스텝에 걸쳐** 행동하고, 한 행동이 다음 상태를 바꾼다. 이 **순차적 의사결정**을 형식화하는 표준 도구가 **마르코프 결정 과정(MDP)**이다.

## MDP

MDP는 다음 5-튜플로 정의된다.

$$\mathcal{M} = (S,\, A,\, P,\, R,\, \gamma)$$

- $$S$$: 상태 집합
- $$A$$: 행동 집합
- $$P(s' \mid s, a)$$: 상태 $$s$$에서 $$a$$를 했을 때 $$s'$$로 갈 전이 확률
- $$R(s, a, s')$$: 보상 함수 (성과 척도의 스텝별 신호)
- $$\gamma \in [0, 1]$$: 미래 보상의 할인율

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_3.svg" width="59%">
</p>

에이전트의 행동 규칙은 **정책(policy)** $$\pi(a \mid s)$$ — 상태에서 행동으로의 (확률적) 사상이다. 에이전트가 추구하는 것은 할인된 누적 보상, 즉 **return**의 기대값이다.

$$G_t = \sum_{k=0}^{\infty} \gamma^{k}\, R_{t+k+1}$$

## 가치 함수와 Bellman 방정식

정책 $$\pi$$의 좋음은 **가치 함수(value function)**로 측정한다. 상태가치와 행동가치는 각각 다음과 같다.

$$V^{\pi}(s) = \mathbb{E}_{\pi}\!\left[\, G_t \mid S_t = s \,\right], \qquad Q^{\pi}(s, a) = \mathbb{E}_{\pi}\!\left[\, G_t \mid S_t = s,\, A_t = a \,\right]$$

이들은 자기참조적 재귀식 — **Bellman 기대 방정식**을 만족한다.

$$V^{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\Big[\, R(s, a, s') + \gamma\, V^{\pi}(s') \,\Big]$$

"지금의 가치 = 즉시 보상 + 할인된 다음 상태의 가치"라는 직관이다. 합리적(=최적) 에이전트는 가치를 최대화하는 정책을 찾으며, 이는 **Bellman 최적 방정식**의 해다.

$$V^{*}(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\Big[\, R(s, a, s') + \gamma\, V^{*}(s') \,\Big]$$

$$\pi^{*}(s) = \arg\max_{a}\, Q^{*}(s, a)$$

이 $$\arg\max$$가 바로 2장 MEU 식의 **순차적(다스텝) 버전**임에 주목하자. 한 스텝 효용 $$U(s')$$가 "즉시 보상 + 미래 가치"로 확장되었을 뿐, **"기대값을 최대화하라"**는 본질은 동일하다.

## 부분관측: POMDP와 믿음 상태

현실의 에이전트는 상태 $$s$$를 직접 보지 못하고 **관측 $$o$$**만 받는다. 이를 다루는 것이 **POMDP**로, MDP에 관측 모델 $$O(o \mid s', a)$$가 추가된다. 상태를 모르므로 에이전트는 상태에 대한 확률분포 — **믿음 상태(belief state)** $$b(s)$$ — 를 유지하고, 매 스텝 베이즈 규칙으로 갱신한다.

$$b'(s') = \eta \; O(o \mid s', a) \sum_{s} P(s' \mid s, a)\, b(s)$$

여기서 $$\eta$$는 정규화 상수다. 즉 부분관측 환경의 합리적 에이전트는 **"보이지 않는 진짜 상태를 확률적으로 추론하며 행동"**한다. 이 믿음 상태 개념은 6장에서 LLM 에이전트의 "컨텍스트"를 해석하는 열쇠가 된다.

# 4. 지능형 에이전트의 5가지 유형

AIMA는 에이전트를 지능 수준에 따라 5가지로 분류한다. 위의 MDP/POMDP 틀로 보면, 이 5유형은 **"이력을 얼마나 정교하게 요약하고, 얼마나 멀리 내다보며 최적화하는가"의 사다리**다.

## (1) Simple Reflex Agent

**현재 지각만** 보고 `if 조건 then 행동` 규칙으로 반응한다. 정책이 $$\pi(a \mid o_t)$$로, 과거 이력과 상태 추정이 없다. 환경이 **완전관측(fully observable)**일 때만 제대로 동작한다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_4.svg" width="75%">
</p>

예: "장애물이 앞에 있으면 멈춘다." POMDP에서는 믿음 상태가 없으니 쉽게 실패한다.

## (2) Model-based Reflex Agent

세계가 어떻게 돌아가는지에 대한 **내부 모델(world model)**, 즉 전이·관측 모델을 갖고 **믿음 상태 $$b(s)$$를 유지**한다. 정책이 $$\pi(a \mid b)$$가 되어 **부분관측(partially observable)** 환경에 대응한다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_5.svg" width="46%">
</p>

## (3) Goal-based Agent

조건-행동 규칙을 넘어, **목표(goal)**를 명시하고 거기에 도달하는 행동열을 **탐색·계획(search & planning)**으로 찾는다. 목표는 "보상이 목표 상태에서만 1"인 특수한 보상 함수로 볼 수 있다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_6.svg" width="49%">
</p>

예: "공항에 가야 한다"는 목표를 두고 여러 경로를 시뮬레이션해 도달 가능한 행동열을 고른다.

## (4) Utility-based Agent

목표 달성/미달성의 이분법을 넘어, 상태마다 **효용 $$U(s)$$**(또는 가치 $$V(s)$$)를 매겨 **기대효용을 최대화**한다. "도착하기만 하면 된다"가 아니라 "더 빠르고 안전하고 싸게 도착한다"를 구분한다. 2장의 MEU 식 $$a^{*} = \arg\max_a EU(a \mid e)$$, 3장의 $$\pi^{*}(s) = \arg\max_a Q^{*}(s,a)$$가 바로 이 유형의 정의다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_7.svg" width="38%">
</p>

## (5) Learning Agent

위 어느 유형이든 **학습 능력**을 더할 수 있다. 전이 $$P$$, 보상 $$R$$, 가치 $$Q$$를 미리 모를 때, 경험으로부터 이를 추정해 정책을 개선한다 — 이것이 바로 **강화학습(RL)**이다. AIMA의 학습 에이전트는 4개 부품으로 구성된다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_8.svg" width="63%">
</p>

- **Performance element**: 실제 행동을 고르는 부분 (앞의 1~4유형)
- **Critic**: 외부 성과 기준에 비추어 잘했는지 평가 (RL의 reward/value 신호)
- **Learning element**: critic의 피드백으로 performance element를 개선 (RL의 policy/value update)
- **Problem generator**: 새로운 경험을 위해 탐험적 행동을 제안 (exploration)

이 critic ↔ learning element 구조가 강화학습의 **actor–critic**, 그리고 [Reflexion](/blog/2026/travelplanner/) 같은 LLM self-reflection 기법의 직접적 조상이다.

# 5. 환경의 7가지 속성 — 무엇이 task를 어렵게 만드는가

같은 에이전트라도 환경의 성격에 따라 난이도가 천차만별이다. AIMA는 task environment를 7가지 축으로 분류하며, 각 축은 위의 형식론과 정확히 대응한다.

| 축              | 쉬운 쪽  | 어려운 쪽  | 형식적 의미                                                   |
| --------------- | -------- | ---------- | ------------------------------------------------------------- |
| **관측성**      | 완전관측 | 부분관측   | $$o_t = s_t$$ (MDP) vs $$o_t \sim O(\cdot \mid s_t)$$ (POMDP) |
| **에이전트 수** | 단일     | 다중       | 단일 $$P$$ vs 게임이론적 상호작용                             |
| **결정성**      | 결정론적 | 확률론적   | $$P(s' \mid s,a) \in \{0,1\}$$ vs 일반 분포                   |
| **에피소드성**  | episodic | sequential | $$\gamma = 0$$ (1스텝) vs $$\gamma > 0$$ (다스텝)             |
| **동역학**      | 정적     | 동적       | 사고 중 $$s$$ 불변 vs 시간에 따라 변화                        |
| **상태/시간**   | 이산     | 연속       | $$\lvert S \rvert$$ 유한 vs 연속 공간                         |
| **사전 지식**   | known    | unknown    | $$P, R$$ 기지 (계획) vs 미지 (학습 필요)                      |

이 표가 중요한 이유는, **agent 벤치마크들이 정확히 "어려운 쪽" 끝을 측정**하기 때문이다.

- [OSWorld](/blog/2026/osworld/): 부분관측(스크린샷) · sequential · 동적 · unknown
- [SWE-bench](/blog/2026/swe-bench/): 거대한 unknown 코드베이스 · sequential
- [TravelPlanner](/blog/2026/travelplanner/): 다수 제약의 sequential planning

즉 "지능형 에이전트가 얼마나 합리적인가"를 묻는 일은, 곧 **이 어려운 POMDP에서 기대 성과를 얼마나 높이는가**를 재는 일이다.

# 6. 고전에서 LLM으로 — 두뇌의 교체

LLM 에이전트는 새 패러다임처럼 보이지만, 골격은 위의 이론 그대로다. 달라진 것은 **정책 $$\pi$$를 구현하는 "두뇌"가 LLM이 되었다**는 점이다.

| 고전 AI 에이전트          | LLM 에이전트                          |
| ------------------------- | ------------------------------------- |
| 두뇌 = 탐색·계획 알고리즘 | 두뇌 = LLM (controller)               |
| sensors = 카메라·센서     | 관측 $$o_t$$ = 텍스트·이미지·스크린샷 |
| actuators = 모터·신호     | 행동 $$a_t$$ = tool call / API / 코드 |
| belief state $$b(s)$$     | 컨텍스트 $$c_t$$ (대화·관측 이력)     |
| $$\pi^{*}$$ = MEU/Bellman | $$\pi_\theta$$ = 사전학습된 LLM 정책  |

형식적으로 LLM 에이전트는 다음 정책이다.

$$a_t \sim \pi_{\theta}(a_t \mid c_t), \qquad c_t = \big(g,\; o_1, a_1,\; \dots,\; o_{t-1}, a_{t-1},\; o_t\big)$$

여기서 $$g$$는 목표/지시문, $$c_t$$는 누적된 컨텍스트다. 그리고 행동 $$a_t$$ 자체가 토큰열이므로, 정책은 토큰 단위로 자기회귀적으로 분해된다.

$$\pi_{\theta}(a_t \mid c_t) = \prod_{i=1}^{\lvert a_t \rvert} p_{\theta}\big(w_i \mid c_t,\, w_{<i}\big)$$

핵심 통찰: **컨텍스트 $$c_t$$가 POMDP의 믿음 상태 $$b(s)$$의 역할**을 한다. LLM은 명시적으로 $$b(s)$$를 계산하지 않지만, 관측 이력을 컨텍스트에 쌓아 암묵적으로 상태를 추정하며 행동한다. 그래서 "perceive → act 루프"라는 본질이 1995년부터 변하지 않았다고 말할 수 있다.

# 7. LLM 에이전트의 해부

LLM 에이전트의 구성요소에 대한 가장 널리 인용되는 정리는 Lilian Weng의 2023년 글 *"LLM Powered Autonomous Agents"*이다. 그는 에이전트를 다음과 같이 분해한다.

> **Agent = LLM (brain) + Planning + Memory + Tool use**

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_9.svg" width="100%">
</p>

## Planning — 탐색의 LLM 버전

복잡한 작업을 **작은 sub-goal로 분해(task decomposition)**하고, 그 sub-goal들을 순차/트리 형태로 풀어간다. 이는 3장의 계획 탐색을 LLM의 생성으로 구현한 것이다.

- **Chain-of-Thought / Tree-of-Thoughts**: 추론을 사슬/트리로 펼쳐 탐색 공간을 넓힌다
- **ReAct**: 추론(Reasoning)과 행동(Acting)을 한 루프에서 교차
- **Reflexion**: 실패를 언어적 피드백으로 기억해 다음 시도를 개선 — 4장 learning agent의 critic 구조와 동형

## Memory — 믿음 상태의 근사

- **단기 기억(short-term)**: 컨텍스트 윈도우 안의 in-context 정보. 단 길이 한계 $$L$$이 있어 $$\lvert c_t \rvert \le L$$.
- **장기 기억(long-term)**: 외부 vector store에 저장하고 검색(retrieval)으로 불러오는 정보. 유한한 컨텍스트로 사실상 무한한 이력을 근사하는 장치다.

즉 메모리는 6장의 믿음 상태 $$b(s)$$를 유한 자원으로 근사하는 메커니즘이다.

## Tool use — 행동 공간의 확장

도구는 LLM의 행동 집합 $$A$$를 외부 세계로 확장한다(검색, 코드 실행, API). Chip Huyen은 이를 *"도구가 base 모델을 유능한 에이전트로 바꾸는 force multiplier"*라고 표현하며, 그의 정의 역시 AIMA와 같다.

> "An agent is anything that can perceive its environment and act upon that environment." — 에이전트는 **환경**과 **행동(도구)** 두 축으로 규정되며, **planning이 모델을 agentic하게 만든다.**

## ReAct 루프 — 한 스텝의 형식화

대부분의 LLM 에이전트는 **Thought → Action → Observation** 루프를 반복한다. 스텝 $$t$$에서 LLM은 사고 $$\tau_t$$와 행동 $$a_t$$를 함께 생성하고, 환경이 관측 $$o_t$$를 돌려주며, 컨텍스트가 누적된다.

$$(\tau_t,\, a_t) \sim \pi_{\theta}(\cdot \mid c_t), \qquad c_{t+1} = c_t \,\oplus\, (\tau_t,\, a_t,\, o_t)$$

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_10.svg" width="64%">
</p>

[AgentBench](/blog/2026/agentbench/)의 `Thought:` + `Action:` 포맷, [OSWorld](/blog/2026/osworld/)의 pyautogui 액션 + 스크린샷 관측이 모두 이 루프의 구체화다.

# 8. Workflow vs Agent — 어디까지가 에이전트인가

Anthropic은 2024년 *"Building Effective Agents"*에서 중요한 구분을 제시했다. 모든 LLM 시스템이 "에이전트"는 아니라는 것이다. 형식적으로 말하면, **제어 흐름(control flow)을 LLM 정책 $$\pi$$에 얼마나 위임하는가**가 기준이다.

먼저 기본 빌딩블록은 **augmented LLM** — 검색·도구·메모리로 증강된 LLM이다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_11.svg" width="71%">
</p>

이 블록을 어떻게 엮느냐에 따라 두 갈래로 나뉜다.

**Workflow** — LLM과 도구가 **사전 정의된 코드 경로**로 오케스트레이션된다. 상태 전이가 개발자의 코드로 고정되어 예측 가능하다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_12.svg" width="90%">
</p>

**Agent** — LLM이 **스스로 프로세스와 도구 사용을 동적으로 결정**한다. 즉 다음 행동도, 종료 시점도 정책 $$\pi$$가 정한다. 매 스텝 환경에서 "ground truth" 피드백을 받아 진행을 평가한다.

<p align="center">
    <img src="/assets/post/image/what-is-an-agent/fig_13.svg" width="87%">
</p>

| 구분        | Workflow              | Agent                         |
| ----------- | --------------------- | ----------------------------- |
| 제어 흐름   | 사전 정의된 코드 경로 | LLM 정책 $$\pi$$가 동적 결정  |
| 예측 가능성 | 높음                  | 낮음 (유연함)                 |
| 적합한 경우 | well-defined task     | open-ended, 스텝 수 예측 불가 |
| trade-off   | 빠르고 저렴·일관      | 느리고 비싸지만 유연          |

Anthropic의 권고는 명확하다.

> "Find the **simplest solution possible**, and only increase complexity when needed."

자율성(autonomy)은 공짜가 아니라 **latency·cost와의 trade-off**다. 고정 workflow ↔ 완전 자율 agent 스펙트럼에서 "어디에 위치시킬 것인가"가 실무의 핵심 질문이다.

# 9. 제한적 합리성 (Bounded Rationality)

2~3장에서 본 "기대효용/가치를 최대화하라"는 이상은 아름답지만, **현실에서는 계산 불가능**한 경우가 많다. 몇 가지 사실을 보자.

- 1장에서 보았듯 에이전트 함수를 테이블로 저장하는 것은 지수적으로 폭발한다.
- 일반적인 계획(planning)은 **NP-hard**이며, 유한 POMDP의 정확한 최적해는 **PSPACE-hard**, 무한 지평 POMDP는 일반적으로 **결정 불가능(undecidable)**하다.

따라서 완벽한 합리성(perfect rationality)을 실현하는 에이전트는 존재할 수 없다. Herbert Simon의 **제한적 합리성(bounded rationality)**, 그리고 Russell의 **제한적 최적성(bounded optimality)** 개념은 이 현실을 받아들인다. 실제 에이전트는 **계산 자원 제약 하에서** 근사적으로 최선을 고른다.

$$a \approx \arg\max_{a \in A_{\text{feasible}}} \; \hat{U}(a) \quad \text{(시간·연산 예산 } B \text{ 안에서)}$$

여기서 $$A_{\text{feasible}}$$는 예산 내에 평가 가능한 행동 부분집합, $$\hat{U}$$는 효용의 근사다. LLM 에이전트도 정확히 이 처지다. 유한한 컨텍스트($$\lvert c_t \rvert \le L$$), 유한한 추론 스텝, 근사적 world model을 가지고 행동한다. **벤치마크에서 보이는 인간과의 큰 격차는 대부분 "원리적 불가능"이 아니라 이 "제한적 합리성"의 현재 수준**을 반영한다. 그래서 모델·스캐폴딩·도구가 개선될수록 점수가 빠르게 오른다(예: [OSWorld](/blog/2026/osworld/) 12% → 80%).

# 10. 그래서 우리는 무엇을 측정하는가

지금까지의 논의를 한 줄로 요약하면 이렇다.

> 에이전트는 **환경을 지각하고 행동하는 무엇**이며, 지능형 에이전트는 그 행동이 **기대 성과(기대효용·가치)를 최대화**할 때 합리적이다. 다만 현실의 에이전트는 자원 제약 속의 **제한적 합리성**으로 이를 근사한다.

그렇다면 자연스러운 질문은 **"우리의 LLM 에이전트는 얼마나 합리적인가?"**이다. 이것을 정량적으로 재는 도구가 **agent 벤치마크**다. 벤치마크의 성공률(success rate)은 task 분포 $$\mathcal{D}$$에 대한 **기대 성과의 몬테카를로 추정량**으로 볼 수 있다.

$$\widehat{\text{SR}} = \frac{1}{N} \sum_{i=1}^{N} r(\tau_i) \;\;\xrightarrow[N \to \infty]{}\;\; \mathbb{E}_{\tau \sim \mathcal{D}}\big[\, r(\tau) \,\big]$$

즉 벤치마크 점수는 "에이전트가 이 환경 클래스에서 기대 성과 척도를 얼마나 달성하는가"의 추정치다. 시리즈의 각 벤치마크는 5장의 환경 속성 축에서 **서로 다른 어려운 클래스**를 측정한다.

- **종합 능력**: [AgentBench](/blog/2026/agentbench/) — 8개 환경 multi-turn 종합 평가
- **범용 어시스턴트**: [GAIA](/blog/2026/gaia/) — 인간에겐 쉽지만 AI에겐 어려운 task
- **코딩**: [SWE-bench](/blog/2026/swe-bench/) — 실 GitHub 이슈를 execution으로 채점
- **계획**: [TravelPlanner](/blog/2026/travelplanner/) — 다수 제약의 sequential planning
- **도메인 특화**: [MedAgentBench](/blog/2026/medagentbench/) — 의료 EHR 상호작용
- **컴퓨터 조작**: [OSWorld](/blog/2026/osworld/) — 실제 OS를 GUI로 직접 다루기

이 벤치마크들은 모두 5장의 "어려운 POMDP"에서, 6~8장의 LLM 에이전트가 7장의 루프를 돌며 얼마나 합리적으로 행동하는지를, 9장의 제약을 안은 채 측정한다. 다음 글부터는 이들을 하나씩 깊이 들여다본다.

> 이어서 읽기: [AgentBench: LLM as Agent 평가의 종합 paradigm](/blog/2026/agentbench/), [GAIA](/blog/2026/gaia/), [SWE-bench](/blog/2026/swe-bench/), [TravelPlanner](/blog/2026/travelplanner/), [MedAgentBench](/blog/2026/medagentbench/), [OSWorld](/blog/2026/osworld/), [TelAgentBench: 통신 도메인 LLM 에이전트 평가](/blog/2026/telagentbench/)

# 참고 문헌

- [Russell & Norvig, _Artificial Intelligence: A Modern Approach_ — Chapter 2: Intelligent Agents](http://aima.cs.berkeley.edu/4th-ed/pdfs/newchap02.pdf)
- [Intelligent agent — Wikipedia](https://en.wikipedia.org/wiki/Intelligent_agent)
- [Sutton & Barto, _Reinforcement Learning: An Introduction_ (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) — MDP·Bellman·value function
- [Kaelbling, Littman & Cassandra, _Planning and Acting in Partially Observable Stochastic Domains_ (1998)](https://www.sciencedirect.com/science/article/pii/S000437029800023X) — POMDP·belief state
- [von Neumann & Morgenstern, _Theory of Games and Economic Behavior_ (1944)](https://en.wikipedia.org/wiki/Von_Neumann%E2%80%93Morgenstern_utility_theorem) — 효용 정리
- [Herbert A. Simon, _A Behavioral Model of Rational Choice_ (1955)](https://en.wikipedia.org/wiki/Bounded_rationality) — bounded rationality
- [Lilian Weng — LLM Powered Autonomous Agents (Lil'Log, 2023)](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Chip Huyen — Agents (2025)](https://huyenchip.com/2025/01/07/agents.html)
- [Anthropic — Building Effective Agents (2024)](https://www.anthropic.com/research/building-effective-agents)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
