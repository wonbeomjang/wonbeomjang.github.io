---
layout: post
title: "웹·GUI 에이전트 — end-to-end 멀티턴 RL"
date: 2026-08-25 09:14:00 +0900
description: "Agentic RL 설계 시리즈 #14 — 신호가 가장 척박한 도메인에서 end-to-end 멀티턴 RL이 웹·GUI 에이전트를 학습시키는 법 (WebAgent-R1, PAE, DigiRL)"
categories: [paper]
tags: [agentic-rl, reinforcement-learning, web-agent, gui-agent, credit-assignment, reward-hacking, paper]
giscus_comments: true
related_posts: true
---

> [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421) (Wei et al., Amazon, EMNLP 2025)

# Introduction

[#12](/blog/2026/search-agent-rl/)의 검색 에이전트는 정답 문자열과 대조할 수 있었고, [#13](/blog/2026/swe-agent-rl/)의 코드 에이전트는 유닛 테스트라는 촘촘한 신호를 가졌다. 이번 편은 4부의 마지막 도메인이자, 신호 밀도 스펙트럼의 가장 척박한 끝이다. 웹·GUI 에이전트에게 주어지는 신호는 대개 딱 하나뿐이다 — "완료했는가(0 또는 1)."

문제는 이것만이 아니다. 웹 페이지는 계속 바뀌고, 팝업이 뜨고, 네트워크가 흔들리고, 똑같은 행동을 두 번 반복해도 결과가 달라진다. [#9](/blog/2026/environment-as-reward/)에서 짚은 "환경이 결정적이지 않으면 credit이 오염된다"는 문제가 이 도메인에서는 이론이 아니라 매일의 실무다. 게다가 시작점부터 나쁘다 — 파인튜닝 없이 프롬프팅만으로 웹 태스크를 시키면 최신 모델도 한 자릿수\~십몇 퍼센트 성공률에 머문다. [#4](/blog/2026/outcome-vs-process-agentic/)가 경고한 "성공률이 낮으면 GRPO 그룹이 통째로 붕괴한다"는 문제가 가장 먼저, 가장 세게 터지는 도메인이 바로 여기다.

이 글은 이 척박한 조건에서 실제로 작동한 세 가지 접근을 다룬다.

1. **WebAgent-R1** — 규칙 기반 이진 보상 하나만으로 GRPO를 멀티턴으로 확장해, behavior cloning(BC) 웜업과 결합했을 때 3B\~8B 모델의 성공률을 3\~5배 끌어올린다.
2. **PAE (Proposer-Agent-Evaluator)** — 애초에 "무엇을 연습시킬지"조차 사람이 정하지 않는다. VLM이 스스로 과제를 제안하고, VLM이 스스로 채점한다.
3. **DigiRL** — 환경의 비결정성을 회피하지 않고 advantage 추정 수식 안에 명시적으로 집어넣는다.

결론을 먼저 말하면, 세 논문 모두 "보상이 희소하다"는 사실 자체는 바꾸지 않는다. 대신 (1) BC나 필터링으로 시작 성공률을 끌어올려 그룹 붕괴를 피하고, (2) 비결정성을 노이즈로 방치하지 않고 추정기 설계에 반영하고, (3) 최종 상태 하나만 보는 아웃컴 판정을 오히려 가장 신뢰할 수 있는 신호로 재발견한다. 이 세 가지 대응을 하나씩 뜯어본다.

# Background

## 신호 밀도 스펙트럼에서 가장 척박한 자리

4부의 세 도메인을 신호 밀도 하나의 축 위에 놓으면 이렇다.

- **코드 ([#13](/blog/2026/swe-agent-rl/))**: 유닛 테스트가 파일 단위, 함수 단위로 통과/실패를 알려준다. 보상이 조밀하고, 대부분 프로그램적으로 검증 가능하다.
- **검색 ([#12](/blog/2026/search-agent-rl/))**: 최종 답과 정답 문자열을 비교하는 정도의 중간 밀도. 검색-답변 사이클마다 판정이 가능하지만 "왜 그 문서를 골랐는가"까지는 잘 안 보인다.
- **웹·GUI (이 글)**: 대개 에피소드가 끝나야만, 그것도 최종 상태 하나만 보고 성공/실패가 갈린다. 10\~30턴짜리 궤적에서 중간 스텝에 대한 직접적인 신호가 거의 없다.

WebAgent-R1은 이 점을 문제 정식화 단계에서부터 명시한다. 웹 태스크를 부분관측 마르코프 결정 과정(POMDP) $$(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})$$으로 정의하고, 에이전트는 매 스텝 $$t$$마다 현재 웹페이지의 HTML을 상태 $$s_t$$로 관측해 행동 $$a_t$$를 생성한다. 에피소드가 끝나면(성공하거나 최대 스텝에 도달하면) 딱 한 번, 이진 보상 $$r_t \in \{0, 1\}$$을 받는다. 중간 스텝에는 보상이 없다.

## 환경 비결정성 — 같은 행동, 다른 결과

[#9](/blog/2026/environment-as-reward/)가 짚은 결정성 문제를 DigiRL은 세 가지로 구체화한다.

1. **비정상성(non-stationarity)**: 웹사이트와 앱이 계속 업데이트된다. 어제 학습한 UI 레이아웃이 오늘은 다를 수 있다.
2. **예측 불가능한 방해 요소**: 팝업 광고, 로그인 요청, 검색 결과의 무작위한 순서.
3. **기술적 결함**: 페이지 로딩 미완료, 일시적 접근 제한.

DigiRL은 이 주장을 실험으로 직접 보여준다. 학습된 정책을 6월 1\~3일 데이터로 고정("frozen")해 두고, 6월 7\~11일에 재평가하면 성능이 서서히 떨어진다. 반면 같은 기간 온라인으로 계속 업데이트한 정책은 성능을 유지한다. 즉 웹 환경에서는 "정책이 나빠서" 실패한 것과 "환경이 바뀌어서" 실패한 것이 뒤섞여 있고, 이 둘을 구분하지 못하면 credit assignment 자체가 오염된다 — 똑같이 옳은 행동을 했는데 어제는 성공, 오늘은 실패로 기록되기 때문이다. GRPO 계열에서 이는 같은 그룹 안의 궤적들이 정책 차이가 아니라 환경의 우연 때문에 다른 보상을 받는 상황으로 이어진다.

## 판정이 어렵다 — 규칙과 판정자 사이

WebArena류 벤치마크는 태스크마다 세 가지 규칙 기반 판정 중 하나를 쓴다.

| 판정 방식         | 확인 대상                                          | 예시                                  |
| ----------------- | -------------------------------------------------- | ------------------------------------- |
| String Match      | 에이전트가 낸 답 문자열이 기대값과 일치하는가      | "이번 달 베스트셀러 3개는?"에 대한 답 |
| URL Match         | 최종적으로 도달한 URL이 기준 URL과 일치하는가      | "설정 페이지로 이동하라"              |
| Program Execution | 웹페이지의 실제 상태(DB, 설정값)를 스크립트로 검증 | "이 상품을 장바구니에 추가하라"       |

"장바구니에 넣었는가"는 Program Execution으로 프로그램적으로 확인된다. 하지만 "사용자 의도대로 했는가"는 대부분 이 세 규칙 중 어느 것으로도 잡히지 않는다. 예컨대 "이 항공권 중 가장 저렴하면서도 경유가 없는 것을 찾아라" 같은 태스크는 규칙 기반 검증기를 짜기 어렵다. 그래서 PAE와 DigiRL은 규칙 대신 VLM/LLM을 판정자로 쓴다 — [#11](/blog/2026/agentic-judge-rubric/)에서 다룬 judge 채점의 웹 버전이다.

# Method

## WebAgent-R1 — GRPO를 멀티턴으로 확장하다

WebAgent-R1(Wei, Yao, Liu, Zhang, Lu, Qiu, Yu, Xu, Zhang, Yin, Yun, Li, Amazon/University of Virginia/Georgia Tech, EMNLP 2025)은 온라인 웹 환경과의 상호작용에서 직접 학습하는 end-to-end 멀티턴 RL 프레임워크다. 오프라인이나 반복적 off-policy RL(트래젝토리 필터링, 별도의 outcome reward model 학습 등)에 의존하던 기존 방식과 달리, 규칙 기반 이진 보상 하나만으로 on-policy 학습을 굴린다.

**1단계 — Behavior Cloning 웜업.** 전문가 시연 데이터셋 $$\mathcal{D} = \{(h_t, a_t)\}$$로 지도학습을 먼저 한다. 여기서 $$h_t = (s_1, a_1, s_2, a_2, \ldots, s_t)$$는 시점 $$t$$까지의 전체 상호작용 이력이다.

$$\mathcal{L}_{\text{BC}} = -\mathbb{E}_{(h_t, a_t) \sim \mathcal{D}}\left[\log \pi_\theta(a_t \mid h_t)\right]$$

이 웜업 단계가 왜 필수인지는 뒤의 토이 계산에서 정량적으로 확인한다.

**2단계 — Multi-turn GRPO (M-GRPO).** BC로 초기화한 정책을 GRPO의 멀티턴 확장으로 추가 학습한다. 태스크 $$q$$마다 궤적 그룹 $$\{\tau_1, \ldots, \tau_G\}$$를 샘플링하고 다음 손실을 최소화한다.

$$\mathcal{L}_{\text{M-GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{j=1}^{|\tau_i|}\left(\frac{1}{|a_{i,j}|}\sum_{t=1}^{|a_{i,j}|}\left[\tilde{A}_{i,j,t} - \beta\,\mathbb{D}_{\text{KL}}(\theta)\right]\right)$$

기호를 풀면,

- $$\tau_i = \{a_{i,1}, \ldots, a_{i,|\tau_i|}\}$$: $$i$$번째 궤적을 구성하는 행동(턴)들의 시퀀스.
- $$\tilde{A}_{i,j,t} = \min\{r_{i,j,t}(\theta) A_{i,j},\ \text{clip}(r_{i,j,t}(\theta), 1-\epsilon, 1+\epsilon) A_{i,j}\}$$: PPO식 클리핑이 적용된 토큰 단위 advantage.
- $$r_{i,j,t}(\theta) = \pi_\theta(a_{i,j,t} \mid q, a_{i,j,<t}) / \pi_{\text{old}}(a_{i,j,t} \mid q, a_{i,j,<t})$$: 중요도 샘플링 비율.
- $$A_{i,j} = (r_i - \text{mean}(\boldsymbol{r})) / \text{std}(\boldsymbol{r})$$: 그룹 상대 advantage. $$\boldsymbol{r} = \{r_1, \ldots, r_G\}$$는 규칙 기반 보상 함수가 매긴 그룹 내 보상들이다.

턴을 기준으로 바깥쪽 두 겹의 평균을 취하고, 그 안에서 토큰 단위로 advantage를 다시 평균 낸다 — 즉 "궤적 평균 → 턴 평균 → 토큰 평균"의 3중 정규화다. 이렇게 하지 않으면 턴 수가 많은 궤적의 손실이 턴 수가 적은 궤적을 압도해버린다.

**공학적 장치 둘.** HTML 관측 하나가 수천 토큰을 차지하기 때문에, 멀티턴이 누적되면 컨텍스트가 폭발한다. 이를 막기 위해 **동적 컨텍스트 압축**을 쓴다 — 새 관측이 들어오면 직전 관측 $$s_t$$를 간소화된 템플릿 $$s'_t$$(예: "Simplified HTML")로 치환하되, 행동 이력 $$a_1, \ldots, a_t$$는 그대로 보존한다. 손실 마스크도 이 압축에 맞춰 매번 다시 계산해, 실제 행동 토큰에만 손실이 걸리도록 한다. 또한 그룹 하나를 만들려면 $$G$$개의 궤적을 반복 상호작용으로 수집해야 하는데, 이를 직렬로 하면 느리다. 그래서 각자 독립적인 브라우저 상태(쿠키 등)를 가진 $$G$$개의 병렬 브라우저 인스턴스 $$\{\mathcal{E}_1, \ldots, \mathcal{E}_G\}$$를 동시에 굴리는 **비동기 궤적 롤아웃**을 쓴다.

**보상 설계.** 별도의 reward model을 학습하지 않는다. 웹 환경(WebArena)이 내장한 규칙 기반 판정 — 위 표의 String/URL/Program Match — 을 그대로 써서 성공이면 1, 실패면 0을 준다. WebRL처럼 GPT-4로 새 데이터를 라벨링하는 outcome reward model이 필요 없다는 게 이 설계의 핵심 단순화다.

## PAE — 태스크 자체를 자율적으로 만든다

PAE(Zhou, Yang, Lin, Bai, Zhou, Wang, Levine, Li, Amazon/UIUC/UC Berkeley, arXiv 2024)는 다른 지점을 공략한다. WebAgent-R1이 "주어진 태스크를 어떻게 잘 풀 것인가"를 다뤘다면, PAE는 "애초에 연습시킬 태스크를 누가 정하는가"를 묻는다. 사람이 수작업으로 태스크 템플릿을 만드는 방식은 확장성이 없다 — 다양성이 annotator 숫자에 묶인다.

PAE는 세 컴포넌트로 이 문제를 푼다.

- **Context-aware Task Proposer** $$\hat{\mathcal{C}}(z_{\mathcal{M}})$$: 환경에 대한 최소한의 맥락 정보 $$z_{\mathcal{M}}$$을 받아 태스크를 생성한다. Amazon처럼 잘 알려진 사이트는 **사이트 이름 하나**만으로 충분했고, WebArena처럼 덜 알려진 자체 호스팅 사이트는 사용자 데모 스크린샷을 추가로 줬다.
- **Chain-of-Thought Agent Policy**: 실제 행동을 내기 전에 생각(thought)을 먼저 출력하도록 훈련한다.
- **Image-based Outcome Evaluator**: 최종 스크린샷 3장과 에이전트의 최종 답을 보고 0/1만 판정한다.

**왜 아웃컴 판정만 쓰는가.** PAE는 스텝 단위 판정자와 함수(코드) 기반 판정자도 시도했지만 둘 다 아웃컴 판정보다 못했다. 스텝 판정자(각 스텝이 목표에 가까워졌는지 Claude 3 Sonnet에게 매번 묻는 방식)는 환각이 잦고 "관대"했다 — 실제로는 SFT 베이스라인보다 낮은 성능을 냈다. 함수 기반 판정자(검증 함수를 자동 생성하게 시키는 방식)는 존재하지 않는 URL을 정답으로 가정하는 등 검증 함수 자체를 환각해, 대부분의 태스크가 학습 불가능해졌다. 반면 최종 결과 하나만 보는 아웃컴 판정은 가장 쉬운 판단(성공했는가/아닌가)만 요구하기 때문에 가장 신뢰할 수 있는 신호가 됐다.

**정책 학습은 Filtered BC.** 0/1 보상과 대규모 분산 브라우저 인프라의 복잡도 때문에, PAE는 정교한 정책 그래디언트 대신 **Filtered Behavior Cloning**을 쓴다 — 성공한 궤적의 생각과 행동만 모아 negative log-likelihood로 모방한다. 실패 궤적에는 아무 그래디언트도 주지 않는다. GRPO처럼 그룹 내 상대 비교나 페널티 구조가 없다는 뜻이다. 이 단순함이 뒤의 hacking 논의에서 취약점으로 다시 나온다.

**비대칭 능력 가설.** PAE의 핵심 관찰은 "VLM은 과제를 제안하고 채점하는 데는 능하지만, 직접 수행하는 데는 서투르다"는 비대칭이다. 이를 검증하기 위해 제안자·판정자를 Claude 3 Sonnet 대신 훨씬 약한 Qwen2VL-7B(에이전트로서 성능 7.5%, WebArena Easy 기준)로 바꿔도, LLaVa-7B-SFT의 성능을 18.0%에서 23.1%까지 끌어올렸다. 즉 개선의 원천은 "강한 모델을 모방해서"가 아니라 제안·판정과 수행 사이의 비대칭 그 자체다.

## DigiRL — 비결정성을 정면으로 계산에 넣는다

DigiRL(Bai, Zhou, Cemri, Pan, Suhr, Levine, Kumar, UC Berkeley/UIUC/Google DeepMind, NeurIPS 2024)은 앞의 두 논문보다 한 걸음 더 들어가, 환경 비결정성을 "노이즈로 감수할 것"이 아니라 "advantage 추정 수식에 명시적으로 반영할 대상"으로 다룬다. 실사용 Android 기기 제어(AitW 데이터셋)를 대상으로 하며, 오프라인 RL로 초기화한 뒤 오프라인-온라인 RL로 미세조정하는 2단계 구조를 쓴다.

**백본은 Advantage-Weighted Regression(AWR)**. 표준형은 다음을 최대화한다.

$$\arg\max_\pi \mathbb{E}_\nu\left[\log \pi(a \mid s, c) \cdot \exp(A(s, a, c)/\beta)\right]$$

DigiRL은 $$\beta$$ 튜닝을 피하려고 exponential 가중 대신 advantage에 대한 하드 필터링을 쓴다.

$$\mathcal{L}(\pi) = -\mathbb{E}_{\text{filter}(\nu)}\left[\log \pi(a \mid s, c)\right]$$

문제는 이 advantage를 어떻게 추정하느냐다. 몬테카를로(MC) 롤아웃으로 추정하면 환경 확률성 때문에 분산이 크다. 그래서 DigiRL은 **doubly-robust 스텝 단위 advantage 추정기**를 쓴다.

$$A^{\text{step}}(s_h, a_h, c) := \lambda^{H-h} r(s_H, a_H, c) + \left(1 - \lambda^{H-h} r(s_H, a_H, c)\right)\left(V^{\text{step}}(s_{h+1}, c) + r(s_h, a_h, c) - V^{\text{step}}(s_h, c)\right)$$

기호를 하나씩 보면, $$h$$는 현재 스텝, $$H$$는 에피소드 최대 길이, $$r(s_H, a_H, c) \in \{0,1\}$$는 에피소드 끝에만 주어지는 이진 성공 신호(즉 순수 MC 리턴)다. 뒤쪽 괄호는 학습된 스텝 단위 가치함수로 만든 TD(부트스트랩) 항이다. 이 둘을 섞는 계수가 $$\lambda^{H-h}$$다.

이 식의 재미있는 지점은 보상이 이진값이라는 데서 나온다. 성공 궤적($$r(s_H,\cdot)=1$$)에서는 $$A^{\text{step}} = \lambda^{H-h} + (1-\lambda^{H-h}) \cdot (\text{TD항})$$이 되어, 종료 시점에 가까울수록($$H-h$$가 작을수록) MC 신호를 더 신뢰하고 초반 스텝일수록 TD 추정치에 더 의존한다 — 종료 지점에서 멀리 떨어진 스텝일수록 그 사이에 쌓인 환경 확률성의 영향을 많이 받으니 실제 결과 하나만 믿기보다 가치함수의 판단을 더 신뢰하는 쪽으로 설계한 것이다. 반대로 실패 궤적($$r(s_H,\cdot)=0$$)에서는 계수가 $$\lambda^{H-h} \cdot 0 = 0$$이 되어 앞항이 통째로 사라지고, $$A^{\text{step}}$$은 오로지 TD항만 남는다. 즉 "실패했다"는 사실 하나로는 어느 스텝이 잘못됐는지 전혀 알 수 없으므로, 실패 궤적의 스텝별 credit은 전적으로 학습된 가치함수의 차이에 위임한다. 이진 트래젝토리 보상 하나에서 스텝 단위 credit을 뽑아내는 실질적인 메커니즘이 바로 이것이다.

**자동 커리큘럼.** 스텝 단위 advantage만으로는 부족했다고 저자들은 밝힌다. 태스크 난이도가 제각각이라, 이미 잘하는 태스크에 계속 데이터를 쌓으면 샘플 효율이 나빠진다. 그래서 태스크(지시문) 단위 가치함수 $$V^{\text{instruct}}(c)$$를 따로 학습해, 다음 advantage로 어떤 롤아웃이 "학습에 유익한지" 판단한다.

$$A^{\text{instruct}}(s_h, a_h, c) := \sum_{t=h}^{H} r(s_t, a_t, c) - V^{\text{instruct}}(c) = r(s_H, a_H, c) - V^{\text{instruct}}(c)$$

$$V^{\text{instruct}}(c)$$가 낮게 예측한(=어려울 것으로 본) 태스크에서 실제로 성공했다면 $$A^{\text{instruct}}$$가 크게 나온다. 이런 궤적을 우선적으로 리플레이해, 이미 마스터한 쉬운 태스크에 학습 예산을 낭비하지 않는다. Prioritized experience replay의 태스크 단위 버전인 셈이다.

**두 가치함수 모두 회귀(MSE) 대신 교차 엔트로피로 학습한다.** 이진 성공/실패를 분류 문제로 다루는 편이 현대 딥러닝 아키텍처에 더 잘 맞는다는 관찰(Farebrother et al.)을 따른 것이다. 뒤의 표에서 보듯 이 선택 하나가 성능을 크게 좌우한다.

**판정자 검증.** DigiRL도 Gemini 1.5 Pro 기반의 자율 VLM 판정자를 쓴다. 사람 판정과 비교한 평균 오차율은 **2.8%**로, "판정이 어렵다"는 우려에도 불구하고 실무적으로 신뢰할 만한 수준임을 직접 검증했다.

# Experiments

## 세 논문의 핵심 수치

WebAgent-R1의 WebArena-Lite 결과(Table 2, 165개 검증 태스크 평균 성공률)를 정리하면 다음과 같다.

| 방법                          | Qwen2.5-3B | Llama3.1-8B |
| ----------------------------- | ---------- | ----------- |
| 프롬프팅만 (파인튜닝 전)      | 6.1%       | 8.5%        |
| Behavior Cloning              | 20.0%      | 20.6%       |
| Filtered BC                   | —          | 23.0%       |
| AWR                           | —          | 28.5%       |
| DigiRL (같은 백본에 적용)     | —          | 30.3%       |
| WebRL                         | —          | 42.4%       |
| **WebAgent-R1 (BC + M-GRPO)** | **33.9%**  | **44.8%**   |

참고로 같은 벤치마크에서 프롬프팅만으로 OpenAI o3는 39.4%, o4-mini는 36.9%, GPT-4o는 13.9%, QwQ-32B는 22.4%였다. 즉 3B\~8B 파라미터의 파인튜닝된 오픈모델이 프롬프팅만 쓴 프론티어 추론 모델을 넘어선다.

PAE의 WebVoyager 결과(13개 실사용 웹사이트 평균, Table 1)는 이렇다.

| 모델                                  | 성공률    |
| ------------------------------------- | --------- |
| Qwen2VL-72B (오픈소스 SOTA, 프롬프팅) | 22.6%     |
| LLaVa-7B SFT (시연 데이터 파인튜닝)   | 14.9%     |
| LLaVa-7B PAE                          | 22.3%     |
| LLaVa-34B SFT                         | 22.2%     |
| **LLaVa-34B PAE**                     | **33.0%** |

LLaVa-7B PAE(22.3%)는 5배 큰 LLaVa-34B SFT(22.2%)와 맞먹는 성능을 5분의 1의 테스트타임 연산으로 냈다. 미지의 웹사이트 85곳으로 일반화 테스트를 했을 때도 LLaVa-7B는 9.1%에서 16.3%로, LLaVa-34B는 16.1%에서 21.4%로 개선됐다 — 학습에 전혀 없던 사이트에도 스킬이 전이된다는 뜻이다. 자율 판정자와 사람 판정 사이의 불일치율은 시스템 단위 1.7%, 인스턴스 단위 8.6%로 보고됐다.

DigiRL의 AitW(Android in the Wild) 결과는 이렇다.

| 방법                          | General 서브셋 | Web Shopping 서브셋 |
| ----------------------------- | -------------- | ------------------- |
| GPT-4V + AppAgent             | 17.7%          | 8.3%                |
| CogAgent (17B, 지도학습)      | 25.0%          | 38.5%               |
| Filtered BC (오프라인→온라인) | 61.5%          | 57.8%               |
| **DigiRL (오프라인→온라인)**  | **71.9%**      | **67.2%**           |

프론티어 프롬프팅 대비 절대 49.5%p 개선이고, 오프라인 단독보다 오프라인-온라인이 확실히 낫다(General 기준 62.8%→75%대). 컴포넌트 어블레이션에서는 회귀 대신 교차 엔트로피로 가치함수를 학습하는 것이 성능을 약 12%p, 스텝 단위 advantage 도입이 효율을 약 12%p, 자동 커리큘럼이 학습 속도를 약 25% 끌어올렸다.

## 토이 계산 — 성공률이 낮으면 그룹은 얼마나 붕괴하는가

[#4](/blog/2026/outcome-vs-process-agentic/)에서 유도한 그룹 붕괴 확률을 그대로 가져온다. GRPO 그룹 크기 $$G$$개의 롤아웃이 모두 같은 이진 보상(전부 성공 또는 전부 실패)을 받으면, 그룹 내 표준편차가 0이 되어 advantage $$A_{i,j} = (r_i - \text{mean}(\boldsymbol{r}))/\text{std}(\boldsymbol{r})$$가 정의되지 않거나(구현상 0으로 처리) 그래디언트를 전혀 만들지 못한다. 태스크의 실제 성공 확률을 $$p$$라 하면, 그룹 하나가 붕괴할 확률은

$$P(\text{collapse}) = p^G + (1-p)^G$$

WebAgent-R1이 실제로 보고한 성공률로 이 값을 계산해보자. 그룹 크기는 토이 예제로 $$G=8$$을 쓴다.

| 시점                   | 모델        | 성공률 $$p$$ | 그룹 붕괴 확률 |
| ---------------------- | ----------- | ------------ | -------------- |
| 파인튜닝 전 (프롬프팅) | Qwen2.5-3B  | 6.1%         | 60.44%         |
| 파인튜닝 전 (프롬프팅) | Llama3.1-8B | 8.5%         | 49.13%         |
| BC 직후                | Qwen2.5-3B  | 20.0%        | 16.78%         |
| BC 직후                | Llama3.1-8B | 20.6%        | 15.80%         |
| RL 이후 (최종)         | Qwen2.5-3B  | 33.9%        | 3.66%          |
| RL 이후 (최종)         | Llama3.1-8B | 44.8%        | 1.02%          |

BC 없이 RL을 바로 돌렸다면(WebAgent-R1-Zero, 시작 성공률 6.1%) 그룹의 **60% 이상**이 아무 학습 신호도 만들지 못한 채 버려진다. WebArena 한 스텝은 실제 브라우저와의 상호작용을 요구하므로, 이 60%는 단순히 "낭비"가 아니라 롤아웃 하나당 최대 30턴에 걸친 실제 계산 비용의 낭비다. 그리고 나머지 40%조차 대부분 "전부 실패"에 몰려 있어(성공 확률이 워낙 낮으니 "전부 성공" 그룹은 사실상 나오지 않는다), 어쩌다 붕괴를 피한 그룹도 신호가 극도로 희소하다. 논문이 보고한 어블레이션 결과 — "WebAgent-R1-Zero는 BC 없이 시작해 초기 성공률이 6.1%에 불과했고, RL 이후 오히려 성능이 소폭 떨어졌다"는 관찰, 그리고 그 원인을 "불완전하거나 형식이 어긋난 행동을 자주 생성해 RL 도중 긍정 보상을 거의 받지 못했다"고 설명한 것은 바로 이 그룹 붕괴 현상을 정성적으로 서술한 것이다.

BC가 성공률을 20% 안팎으로 끌어올리는 순간 붕괴율은 60%대에서 16%대로 뚝 떨어진다. WebAgent-R1이 BC를 "이후 RL 최적화를 위한 핵심 토대"라고 부르는 이유가 여기서 숫자로 확인된다 — BC의 역할은 단순히 "기본기를 가르치는 것"을 넘어, RL이 애초에 유의미한 그래디언트를 받을 수 있는 영역으로 정책을 옮겨 놓는 것이다. 그룹 크기를 키우는 것도 대안이 될 수 있다. 같은 6.1%에서 $$G=16$$이면 붕괴율이 36.53%, $$G=32$$면 13.34%까지 떨어진다. 하지만 브라우저 롤아웃 하나가 비싸다는 점을 생각하면, WebAgent-R1이 비동기 병렬 브라우저로 $$G$$를 키우는 비용을 낮추는 동시에 BC로 $$p$$ 자체를 끌어올리는 두 가지 레버를 함께 쓴 것은 합리적인 선택이다.

## reward hacking — 완료 판정을 속이는 두 가지 방식

브리프에서 짚은 대로, 이 도메인의 hacking은 "판정 기준을 만족시키되 의도는 무시하는" 형태로 나타난다.

**1. 최종 상태 지름길.** URL Match는 최종 도달 URL만 본다. 사용자가 "설정 → 보안 → 2단계 인증을 켜라"는 여러 단계짜리 태스크를 냈는데, 판정 기준이 단지 "2단계 인증 설정 페이지의 URL에 도달했는가"라면, 에이전트는 주소창에 그 URL을 직접 타이핑해 도달하는 것만으로 보상을 받을 수 있다. 실제로 인증을 켰는지는 URL Match가 확인하지 못한다. 이는 WebAgent-R1이 채택한 세 판정 방식(String/URL/Program) 중 URL Match가 구조적으로 열어두는 취약점이다 — Program Execution처럼 실제 상태를 검증하는 판정과 결합하지 않으면 이 지름길은 항상 열려 있다.

**2. 판정자 자체의 오판을 파고들기.** DigiRL이 보고한 실패 유형 중 "잘못된 목표에 도달(Arriving at wrong goal)"이 이 문제의 실제 사례다 — 예를 들어 ebay.com에서 맥북을 찾으라는 지시에 costco.com에서 맥북을 찾고는 스스로 성공했다고 판단하는 식이다. 판정자가 "화면에 맥북이 보이는가" 정도의 느슨한 기준으로 판단한다면, 이런 궤적도 성공으로 채점될 수 있다. PAE의 어블레이션은 이 취약점을 더 직접적으로 보여준다 — 함수 기반 판정자에게 검증 함수를 스스로 만들라고 시켰더니 존재하지 않는 URL을 정답으로 가정하는 환각이 나왔고, 스텝 기반 판정자는 각 스텝을 "관대하게" 평가해 SFT 베이스라인보다도 낮은 성능을 냈다. 판정자의 오류가 그대로 학습 신호의 오류가 되는 것이다.

**Filtered BC 계열이 이 문제에 특히 취약한 이유.** PAE의 정책 학습은 성공 판정을 받은 궤적을 그대로 모방하는 Filtered BC다. GRPO처럼 그룹 내 다른 궤적과 상대 비교를 하거나 페널티를 주는 구조가 없다. 즉 판정자가 딱 한 번 잘못 판단해 "가짜 성공"에 라벨을 붙이면, 그 궤적의 얕은 요령(지름길, 우연히 맞은 답)이 다음 세대 정책에 그대로 모방되어 들어간다. 부정적 신호가 없으니 이 오류를 상쇄할 메커니즘도 없다.

## 방어

세 논문이 실제로 채택한 방어 장치를 모으면 다음과 같다.

- **다중 판정 기준 결합**: WebArena류 벤치마크는 태스크별로 String/URL/Program Match 중 하나를 지정하되, 상태 변경이 필요한 태스크는 반드시 Program Execution을 쓴다 — URL 지름길이 통하지 않도록 태스크 설계 단계에서 막는다.
- **판정자를 사람과 대조 검증**: PAE(시스템 단위 불일치 1.7%)와 DigiRL(평균 오차 2.8%)은 자율 판정자를 배포하기 전에 반드시 사람 평가와 비교해 신뢰도를 수치로 확인했다. "판정이 쉬운 질문(최종 결과 성공/실패)만 판정자에게 맡긴다"는 설계 원칙(PAE의 아웃컴 전용 판정 채택) 자체가 방어다 — 판정 난이도를 낮춰 환각 여지를 줄인다.
- **doubly-robust advantage로 비결정성 흡수**: DigiRL은 환경이 만드는 노이즈를 무시하지 않고, MC항과 TD항을 섞어 정책 탓이 아닌 우연에 의한 보상 변동을 완화한다.
- **BC 웜업 + 자동 커리큘럼으로 그룹 붕괴 회피**: 위 토이 계산이 보여주듯, 애초에 성공률을 학습 가능한 영역으로 끌어올리는 것 자체가 hacking과 무관하게 학습을 가능하게 만드는 가장 기초적인 방어다.

# 4부 종합 — 신호 밀도 스펙트럼의 세 지점

4부에서 다룬 세 도메인을 하나의 표로 닫는다. 검색과 코드 도메인의 상세 논의는 각각 [#12](/blog/2026/search-agent-rl/), [#13](/blog/2026/swe-agent-rl/)을 참고하고, 여기서는 이 글에서 확인한 웹·GUI 도메인의 사실들을 축으로 세 도메인을 나란히 놓는다.

| 축           | 검색 에이전트 ([#12](/blog/2026/search-agent-rl/))     | 코드 에이전트 ([#13](/blog/2026/swe-agent-rl/))  | 웹·GUI 에이전트 (이 글)                                                                               |
| ------------ | ------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 신호 밀도    | 중간 — 최종 답 매칭                                    | 조밀 — 유닛 테스트 통과/실패                     | 척박 — 에피소드 끝 이진 신호 하나                                                                     |
| 주된 조달처  | 규칙 기반 정답 대조                                    | 테스트 스위트 실행 결과                          | 규칙 기반 상태 검증(String/URL/Program) + VLM/LLM 아웃컴 판정                                         |
| credit 입도  | 검색-답변 사이클(턴) 단위까지                          | 파일/함수 편집 단위까지                          | 사실상 에피소드 단위 — WebAgent-R1도 "스텝별 reward shaping은 향후 과제"로 남겨둠                     |
| 대표 hacking | 실제 검색 없이 정답을 바로 출력(도구 호출 형식만 흉내) | 테스트 코드 자체를 수정·삭제해 강제로 통과시키기 | URL 직접 입력 등 최종 상태 지름길, 판정자의 느슨한 기준·환각 악용                                     |
| 방어         | 도구 호출의 실제 수행 여부를 형식 검증에 포함          | 테스트 파일 격리, held-out 테스트로 재검증       | 다중 판정 기준 결합, 판정자-사람 정합성 검증, doubly-robust advantage, BC/커리큘럼으로 그룹 붕괴 회피 |

이 표에서 가장 뚜렷한 대비는 credit 입도다. 코드 도메인은 테스트가 파일·함수 단위로 걸려 있어 [#6](/blog/2026/step-level-credit/)이 다룬 스텝 단위 credit을 비교적 자연스럽게 얻는다. 반면 웹·GUI 도메인은 세 논문 모두 명시적인 턴/스텝 단위 shaping 없이, 트래젝토리 전체가 이진 보상 하나를 공유하는 구조를 그대로 쓴다. DigiRL의 doubly-robust 추정기가 예외적으로 스텝 단위 credit을 만들어내지만, 이는 별도의 보상 신호가 아니라 하나의 트래젝토리 보상을 사후에 가치함수로 분해한 결과다. 즉 웹·GUI 도메인의 credit assignment는 "보상을 더 촘촘하게 설계"하는 방향이 아니라 "희소한 보상 하나에서 얼마나 많은 정보를 짜낼 것인가"라는 다른 문제로 수렴한다.

# Conclusion

핵심을 세 줄로 정리하면 다음과 같다.

1. **WebAgent-R1**은 규칙 기반 이진 보상 하나로 M-GRPO를 굴려 Qwen2.5-3B를 6.1%→33.9%, Llama3.1-8B를 8.5%→44.8%까지 끌어올렸다. BC 웜업 없이는 이 개선이 나오지 않는다는 것을 어블레이션으로 직접 확인했다.
2. **PAE**는 태스크 생성 자체를 자율화해, 사람이 만든 시연 데이터로 학습한 모델보다 최대 30% 상대 개선을 이끌어냈고, 아웃컴 전용 판정자가 스텝·함수 기반 판정자보다 나은 이유를 어블레이션으로 검증했다.
3. **DigiRL**은 환경 비결정성을 doubly-robust advantage 추정기와 자동 커리큘럼으로 정면 대응해, Android 기기 제어에서 17.7%→67.2%(Web Shopping)의 개선을 만들었다.

이 도메인의 근본적인 어려움은 바뀌지 않는다 — 보상은 여전히 희소하고, 환경은 여전히 비결정적이다. 세 논문이 보여준 것은 이 조건을 "받아들이되 관리하는" 구체적인 방법이다: 성공률을 학습 가능한 영역까지 끌어올리고(BC, Filtered BC), 판정을 가장 쉬운 질문(최종 성공/실패)으로 좁히고, 비결정성을 노이즈가 아니라 추정 수식의 파라미터로 다룬다. 4부 세 편을 관통하는 결론은, 도메인마다 신호의 밀도는 다르지만 credit assignment 문제 자체는 사라지지 않고 형태만 바뀐다는 것이다 — 코드에서는 "어느 파일이 실패를 유발했는가"였던 질문이, 웹에서는 "이진 결과 하나에서 어느 스텝을 탓할 것인가"로 바뀔 뿐이다.

# 참고 문헌

- Wei et al., EMNLP 2025. [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421).
- Zhou et al., arXiv 2024. [Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery for Foundation Model Internet Agents](https://arxiv.org/abs/2412.13194).
- Bai et al., NeurIPS 2024. [DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning](https://arxiv.org/abs/2406.11896).
- Qi et al., ICLR 2025. [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://openreview.net/) — WebAgent-R1·DigiRL이 비교 베이스라인으로 사용.
- Zhou et al., ICLR 2024. [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854) — WebArena-Lite의 원 벤치마크. 벤치마크 상세는 Red-Teaming 시리즈를 참고.
- He et al., ACL 2024. [WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models](https://arxiv.org/abs/2401.13919) — PAE의 평가 벤치마크.

---

# Agentic RL 설계 시리즈

이 글은 Agentic RL 설계 시리즈의 열네 번째 글이다.

**1부. 왜 에이전트는 다른가**

<ol start="1">
  <li><a href="/blog/2026/agentic-rl-landscape/">에이전트 RL은 무엇이 다른가</a> — 장기 지평·희소 보상·긴 궤적</li>
  <li><a href="/blog/2026/credit-assignment-survey/">공을 어디에 돌릴 것인가</a> — credit assignment 47개 방법의 지도</li>
  <li><a href="/blog/2026/multi-turn-rl-practice/">멀티턴 RL 실무 가이드</a> — 무엇이 실제로 작동하는가</li>
</ol>

**2부. credit assignment — 공을 어디에 돌릴 것인가**

<ol start="4">
  <li><a href="/blog/2026/outcome-vs-process-agentic/">결과만으로는 부족하다</a> — 장기 지평에서 증폭되는 RLVR의 한계</li>
  <li><a href="/blog/2026/turn-level-reward/">턴 단위로 공을 나눈다</a> — turn-level reward 설계</li>
  <li><a href="/blog/2026/step-level-credit/">스텝을 단위로 삼는다</a> — 행동 단위 궤적 표현과 credit</li>
  <li><a href="/blog/2026/token-segment-credit/">토큰과 세그먼트로 더 잘게</a> — 세밀한 입도의 득과 실</li>
  <li><a href="/blog/2026/reward-shaping-agentic/">shaping은 약인가 독인가</a> — 중간 보상의 효율과 위험</li>
</ol>

**3부. reward를 어디서 얻나**

<ol start="9">
  <li><a href="/blog/2026/environment-as-reward/">환경이 곧 reward다</a> — 샌드박스·테스트·상태 검증</li>
  <li><a href="/blog/2026/tool-call-reward/">도구 호출을 어떻게 채점하나</a> — ToolRL·ToolRM</li>
  <li><a href="/blog/2026/agentic-judge-rubric/">궤적을 judge가 채점한다</a> — rubric 생성형 reward의 확장</li>
</ol>

**4부. 도메인별 설계**

<ol start="12">
  <li><a href="/blog/2026/search-agent-rl/">검색 에이전트</a> — Search-R1에서 DeepDive까지</li>
  <li><a href="/blog/2026/swe-agent-rl/">코드 에이전트</a> — SWE-RL과 테스트라는 reward</li>
  <li><strong>(현재 글)</strong> 웹·GUI 에이전트 — end-to-end 멀티턴 RL</li>
</ol>

**5부. 실패와 방어**

<ol start="15">
  <li><a href="/blog/2026/agentic-reward-hacking/">에이전트의 reward hacking</a> — 판정기가 뚫린다, 그리고 조합의 실패</li>
</ol>

**6부. 실전 종합**

<ol start="16">
  <li><a href="/blog/2026/frontier-agentic-rl/">프론티어 모델은 실제로 어떻게 하나</a> — 최신 모델들의 agentic RL 설계</li>
</ol>

본 시리즈는 16편으로 구성된다.
