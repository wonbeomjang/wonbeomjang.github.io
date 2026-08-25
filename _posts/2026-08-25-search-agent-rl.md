---
layout: post
title: "검색 에이전트 — Search-R1에서 DeepDive까지"
date: 2026-08-25 09:12:00 +0900
description: "Agentic RL 설계 시리즈 #12 — 검색 에이전트의 reward는 EM/F1로 손쉽게 나오는데, 왜 중간 신호는 여전히 성긴가"
categories: [paper]
tags: [reinforcement-learning, agentic-rl, search-agent, credit-assignment, reward-design, paper]
giscus_comments: true
related_posts: true
---

> [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) (Jin et al., UIUC, arXiv 2025)

# Introduction

지금까지 3부([#9](/blog/2026/environment-as-reward/)\~[#11](/blog/2026/agentic-judge-rubric/))에서 reward가 어디서 조달되는지 세 갈래로 나눠봤다. 환경이 채점하거나(실행 결과, 통과/실패), 도구 호출 자체를 규칙으로 채점하거나, judge가 궤적을 보고 점수를 매기거나. 이제 4부에서는 이 세 조달처가 실제 도메인에서 어떻게 섞여 쓰이는지를 본다. 첫 번째 도메인은 **검색 에이전트**다.

검색 에이전트는 이 시리즈에서 다루는 도메인 중 reward 설계가 가장 쉬워 보이는 축에 속한다. 이유는 단순하다. 질문에 정답이 있는 QA 태스크라면, 모델이 뱉은 답과 정답 문자열을 비교하는 것만으로 채점이 끝난다. Exact Match(EM) 한 줄이면 [#9](/blog/2026/environment-as-reward/)에서 다룬 "환경이 곧 reward다"의 가장 단순한 사례가 완성된다. 실제로 이 도메인의 첫 성공 사례인 Search-R1은 정확히 이 전략을 택했다 — reward 항을 하나만 두고, 형식 보너스도 검색 횟수 페널티도 없이, 그냥 EM 하나로 학습시켰다.

문제는 그 다음이다. 검색 에이전트는 한 번의 질의로 끝나지 않는다. 질문 하나를 풀기 위해 검색을 3번, 5번, 많으면 수십 번 호출하는 멀티턴 궤적이 만들어지는데, 최종 답이 맞았다는 스칼라 하나만으로는 **그 중 몇 번째 검색이 결정적이었는지, 어느 검색이 시간 낭비였는지** 전혀 알 수 없다. reward의 조달은 쉬운데 credit의 배분은 여전히 성기다 — 이게 이 편의 핵심 긴장이다.

이 글은 이 긴장을 놓고 다투는 네 편의 논문을 다룬다. Search-R1(Jin et al., 2025)은 순수 outcome reward로 시작점을 놓았고, R1-Searcher(Song et al., 2025)는 형식과 검색 시도를 먼저 가르친 뒤 정답을 가르치는 2단계 커리큘럼을 제안했다. ReSearch(Chen et al., 2025)는 형식과 정답을 하나의 조각별 함수로 압축했다. 그리고 DeepDive(Lu et al., 2025)는 문제 자체를 어렵게 만드는 데이터 합성과, 중복 검색을 벌점 처리하는 redundancy penalty로 한 단계 더 나아갔다.

미리 결론을 말하면, 이 네 편 중 누구도 진짜 "스텝별 credit assignment"는 하지 않는다. 모두 트레젝토리 전체에 스칼라 하나를 브로드캐스트하는 outcome reward를 쓴다. 다만 그 스칼라를 계산하는 방식(정오만 볼지, 검색 시도 여부까지 볼지, 검색 다양성까지 볼지)과 그 스칼라를 어느 토큰에 흘려보낼지(생성 토큰만 학습할지, 검색 결과 토큰까지 학습할지)를 다르게 설계해서 성긴 신호 문제를 우회한다. 이 우회의 흔적이 바로 이 도메인의 reward hacking 패턴 — 검색 없이 답하기, 같은 쿼리 반복하기, 검색 결과를 조작하기 — 으로 고스란히 남는다.

# Background

## 멀티턴 검색 궤적의 형식

네 논문 모두 비슷한 골격을 공유한다. 질문 $$q$$가 주어지면 에이전트는 추론(reasoning)과 도구 호출(검색)을 번갈아 하며 트레젝토리를 만든다.

$$T = [q, (c_1, a_1, o_1), \ldots, (c_m, a_m, o_m), c_{ans}, a_{eos}]$$

기호를 풀면 이렇다.

- $$c_i$$: $$i$$번째 스텝의 추론(chain-of-thought) 텍스트.
- $$a_i$$: $$i$$번째 스텝에서 실행한 행동(검색 질의, 클릭, URL 열기 등).
- $$o_i$$: 그 행동이 환경(검색엔진·웹)으로부터 돌려받은 관찰(observation) — 검색 결과 스니펫이나 웹페이지 본문.
- $$c_{ans}, a_{eos}$$: 더 이상 정보가 필요 없다고 판단했을 때의 최종 추론과 답변 종료 행동.

이 트레젝토리는 정책 모델이 직접 생성한 토큰($$c_i, a_i$$, 최종 답)과, 환경이 채워 넣은 토큰($$o_i$$)이 뒤섞여 있다는 점이 중요하다. 뒤에서 볼 "loss masking"이 바로 이 구분을 이용한다.

## 이 도메인이 세 조달처를 섞는 방식

[#9](/blog/2026/environment-as-reward/)\~[#11](/blog/2026/agentic-judge-rubric/)에서 정리한 세 조달처를 이 도메인이 어떻게 조합하는지 미리 밝혀둔다.

- **환경 검증(#9)**: 정답이 있는 QA이므로 최종 답을 문자열 규칙(EM)이나 집합 규칙(F1)으로 채점할 수 있다. Search-R1이 이 조달처만 쓴다.
- **도구 호출 채점(#10)**: `<search>...</search>`, `<answer>...</answer>` 같은 태그가 정확한 형식으로 열리고 닫혔는지, 검색을 시도했는지 자체를 규칙으로 채점한다. R1-Searcher의 retrieval reward·format reward, DeepDive의 Format 항이 여기 해당한다.
- **judge 채점(#11)**: 정답의 표현이 여러 형태로 존재할 수 있어("Inter Corp." vs "Inter Corporation") 문자열 매칭만으로는 억울한 오채점이 나온다. DeepDive는 이 지점에서 LLM judge를 답 동치 판정에 끼워 넣는다. R1-Searcher와 ReSearch도 평가 단계(학습 reward는 아니고)에서 LLM-as-Judge 지표를 보조로 쓴다.

즉 이 도메인의 reward는 대부분 **환경 검증(정답 채점) + 도구 호출 채점(형식 채점)의 합**으로 만들어지고, 정답의 표현 다양성이 문제 되는 지점에서만 judge가 슬쩍 끼어든다. 세 조달처가 뚜렷하게 분업하는 [#10](/blog/2026/tool-call-reward/)의 순수 도구 채점 사례나 [#11](/blog/2026/agentic-judge-rubric/)의 순수 judge 사례와 달리, 여기서는 환경과 도구가 한 스칼라 안에 합산돼 있다.

## GRPO 한 줄 요약

네 논문 모두 학습 알고리즘으로 GRPO(또는 그 변형)를 쓴다. 그룹 $$G$$개의 트레젝토리를 같은 질문에 대해 샘플링하고, 그룹 평균·표준편차로 정규화한 advantage를 쓴다.

$$A_i = \frac{r_i - \text{mean}(\{r_k\}_{k=1}^G)}{\text{std}(\{r_k\}_{k=1}^G)}$$

$$r_i$$는 $$i$$번째 트레젝토리가 받은 reward, $$A_i$$는 정규화된 advantage다. 이 advantage 하나가 트레젝토리 안의 **모든 학습 대상 토큰에 동일하게 broadcast**된다 — 이게 "outcome reward"라는 말의 정확한 의미다. 알고리즘 자체에 대한 자세한 유도는 [#1](/blog/2026/agentic-rl-landscape/)\~[#3](/blog/2026/multi-turn-rl-practice/)을 참고한다.

# Method

## Search-R1 — 순수 EM 하나로 시작점을 놓다

> [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) (Jin et al., UIUC / UMass Amherst / Google Cloud AI, arXiv 2025)

Search-R1은 이 도메인의 사실상 출발점이다. 아이디어는 DeepSeek-R1의 "규칙 기반 outcome reward만으로 추론 능력이 나온다"는 관찰을 검색 태스크로 그대로 옮기는 것이다.

**rollout 형식.** 학습 템플릿은 세 종류의 태그로 구성된다. 모델은 `<think>...</think>` 안에서 추론하고, 검색이 필요하면 `<search>query</search>`를 낸다. 시스템이 이를 감지해 검색엔진을 호출하고 결과를 `<information>...</information>`으로 감싸 트레젝토리에 이어붙인다. 더 필요 없다고 판단하면 `<answer>...</answer>`로 답을 낸다. 이 사이클은 최대 행동 예산 $$B$$(논문 설정값 4)에 도달하거나 답을 낼 때까지 반복된다. 프롬프트에는 "You can search as many times as you want"라는 문구가 명시돼 있다 — 검색 횟수 자체에는 제약도 페널티도 걸지 않겠다는 선언이다.

**reward 설계.** Search-R1의 reward 함수는 정확히 한 항이다.

$$r_\phi(x, y) = \text{EM}(a_{pred}, a_{gold})$$

$$a_{pred}$$는 응답에서 추출한 최종 답, $$a_{gold}$$는 정답이다. 형식 보너스도, 검색 횟수 페널티도, 학습된 신경망 reward model도 없다. 논문은 이 선택을 명시적으로 정당화한다 — "학습된 모델이 이미 구조적 형식을 잘 지키기 때문에 형식 reward를 넣지 않았다"고 밝히고, 신경망 reward model을 피한 이유로는 "대규모 RL에서 reward의 구체적 형태에 모델이 민감하게 반응한다"는 점과 재학습 비용을 든다.

**credit assignment의 유일한 장치: retrieved token masking.** reward는 트레젝토리 전체에 스칼라 하나로 broadcast되지만, 그 스칼라가 흘러갈 대상 토큰은 걸러낸다. 정책 그래디언트 계산 시 다음 마스크를 적용한다.

$$I(y_t) = \begin{cases} 1 & y_t \text{가 LLM이 생성한 토큰일 때} \\ 0 & y_t \text{가 검색으로 받아온(retrieved) 토큰일 때} \end{cases}$$

즉 `<information>` 태그 안의 검색 결과 텍스트는 정책이 만든 게 아니라 환경이 채워 넣은 것이므로, 이 토큰에 대해서는 로그확률을 최적화하지 않는다. KL divergence 항을 계산할 때도 동일한 마스크가 적용된다. 이 장치를 껐을 때와 켰을 때의 성능 차이는 뒤에서 다룬다.

**결과.** Qwen2.5-7B 기준 7개 QA 벤치마크(NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle) 평균 EM이 RAG 베이스라인 0.304에서 Search-R1-base 0.431로 올랐다 — 상대 개선 41%(초록에 언급된 수치와 일치). 다만 이 그룹 안에서 가장 강한 베이스라인은 RAG가 아니라 Rejection Sampling(평균 0.348)이었고, 여기 대비로는 상대 개선이 24%다(본문에 언급된 수치). 3B 모델에서는 RAG가 가장 강한 베이스라인(0.270)이었고, Search-R1-instruct(0.325) 대비 상대 개선은 20%다. PPO와 GRPO를 비교한 실험에서는 GRPO가 초반 수렴은 빠르지만 장기 학습에서 reward collapse가 나타났고 PPO가 더 안정적이었다 — 이 논문은 기본 알고리즘으로 PPO를 채택했다.

## R1-Searcher — 검색부터 가르치고 정답을 나중에 가르친다

> [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) (Song et al., Renmin University of China, arXiv 2025)

Search-R1이 "reward를 최대한 단순하게"를 택했다면, R1-Searcher는 정반대 축에서 실험한다. reward를 **두 단계로 쪼개** 학습 커리큘럼 자체를 설계한다.

**Stage 1 — 검색하는 법만 가르친다.** 이 단계에서는 답의 정확성을 전혀 보지 않는다. 목표는 딱 하나, "모델이 검색을 시도한다"는 행동 자체를 강화하는 것이다.

$$R_{retrieval} = \begin{cases} 0.5 & n \geq 1 \\ 0 & n = 0 \end{cases}$$

$$n$$은 트레젝토리 안에서 검색을 호출한 횟수다. 여기에 형식 보너스가 더해진다.

$$R_{format} = \begin{cases} 0.5 & \text{형식이 올바를 때} \\ 0 & \text{형식이 틀렸을 때} \end{cases}$$

Stage 1의 최종 reward는 이 둘의 합이다. 정답 여부는 아예 관여하지 않는다 — 검색 자체를 시도하지 않으면 모델이 파라메트릭 지식만으로 답을 때우는 습관을 처음부터 강화해버리기 때문에, 이를 막기 위해 "검색했다"는 사실 자체에 보상을 준다.

**Stage 2 — 이제 정답을 가르친다.** retrieval reward를 없애고, 대신 답의 정확도를 F1 점수로 채점한다.

$$R_{answer} = \frac{2 \cdot IN}{PN + RN}$$

$$PN$$은 예측 답의 단어 수, $$RN$$은 정답의 단어 수, $$IN$$은 둘의 교집합 단어 수다. 형식 reward는 부호가 뒤집힌다.

$$R'_{format} = \begin{cases} 0 & \text{형식이 올바를 때} \\ -2 & \text{형식이 틀렸을 때} \end{cases}$$

Stage 1에서는 형식을 맞추면 보너스(+0.5)를 줬지만, Stage 2에서는 이미 형식을 익혔다고 보고 형식이 틀렸을 때만 강한 페널티(-2)를 준다. 두 항의 합이 Stage 2의 최종 reward다.

**왜 EM이 아니라 F1인가 — 반례.** 논문은 답 reward로 EM, CEM(Cover Exact Match), F1 세 가지를 비교하는 ablation을 실었다. 결과는 뚜렷하다. F1 기반 reward가 EM 기반 대비 **평균 성능 52.6% 개선**을 냈다. EM은 너무 엄격해서(정답 문자열이 한 글자라도 다르면 0점) 개방형 질문 생성 시나리오에 맞지 않았고, 학습 중 응답 길이가 짧아지는 방향으로 붕괴했다. Search-R1이 순수 EM으로도 성공한 것과 대조적인데, 두 논문의 학습 데이터·백본이 다르므로 직접 비교는 조심해야 하지만, 적어도 R1-Searcher의 세팅에서는 "정답 채점을 얼마나 엄격하게 하느냐" 자체가 reward hacking의 방향을 바꾸는 요인이었다.

**형식 reward가 없으면 생기는 hacking — 실제 관찰 사례.** 논문은 format reward를 반복적으로 다듬은 이유로 실제 관찰된 hacking 사례를 나열한다.

1. 모델이 `<begin_of_query>...</end_of_query>`(검색 질의)를 생성하지 않고 `<begin_of_documents>...</end_of_documents>`(검색 결과)를 스스로 지어낸다 — **검색을 안 하고 검색한 척하는** 가장 노골적인 형태의 hacking이다.
2. Base 모델에 KL 계수를 0으로 설정하면 학습 후반부에 형식을 아예 무시한 의미 불명 출력이 나온다.
3. Llama 모델에서 Stage 1(검색 시도 학습)을 생략하면 검색을 아예 건너뛰고 파라메트릭 지식만으로 직답한다 — 브리프에서 말한 "검색 없이 파라메트릭 지식으로 답하기" 패턴이 그대로 재현된 사례다.
4. CEM을 감독 신호로 쓰면 정답을 포함하되 불필요하게 긴 응답을 만든다 — 부분 문자열만 맞으면 통과되는 CEM의 느슨함을 이용한 패딩이다.

**결과.** Llama-3.1-8B-Instruct 백본, HotpotQA+2WikiMultiHopQA에서 뽑은 단 8,148건만으로 RL 학습을 했는데도, LLM-as-Judge 채점 기준으로 가장 강한 베이스라인(ReARTeR + GPT-4o-mini)보다 HotpotQA 48.2%, 2WikiMultiHopQA 21.7%, 분포 밖(OOD) 벤치마크인 Bamboogle에서도 4.0% 개선을 냈다.

## ReSearch — 형식과 정답을 하나의 조각별 함수로 압축

> [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) (Chen et al., Baichuan Inc., arXiv 2025)

ReSearch는 R1-Searcher의 2단계 커리큘럼 대신, 형식과 정답을 **하나의 조각별(piecewise) reward**로 합쳐 단일 스테이지로 학습한다.

$$r = \begin{cases} f1(a_{pred}, a_{gt}) & \text{F1 점수가 0이 아닐 때} \\ 0.1 & \text{F1 점수가 0이고 형식이 올바를 때} \\ 0 & \text{F1 점수가 0이고 형식이 틀렸을 때} \end{cases}$$

읽는 법은 이렇다. 답이 조금이라도 맞으면(F1 > 0) F1 점수 자체가 reward가 된다. 답은 완전히 틀렸지만(F1 = 0) 최소한 태그 형식은 지켰다면 작은 위안 보상(0.1)을 준다. 형식마저 틀렸다면 0이다. R1-Searcher가 두 단계로 나눠서 준 것을 ReSearch는 한 함수의 분기로 처리한 셈이다 — "형식을 지키는 것 자체에 최소한의 가치를 부여한다"는 설계 의도는 같지만, 별도의 커리큘럼 없이 단일 스테이지로 압축했다는 점이 다르다.

학습 데이터도 특이하다. HotpotQA나 2WikiMultiHopQA를 섞지 않고 **MuSiQue 학습셋 하나(19,938건)만** 써서 학습했는데도 HotpotQA, 2WikiMultiHopQA, Bamboogle로 잘 일반화됐다. 7B 모델 기준 가장 강한 베이스라인 대비 평균 개선폭이 EM 15.81%, LLM-as-Judge 17.56%였고, 32B 모델에서는 EM 14.82%, LLM-as-Judge 15.46%였다.

## DeepDive — 문제를 어렵게 만들고, 중복 검색에 벌점을 준다

> [DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL](https://arxiv.org/pdf/2509.10446) (Lu et al., Tsinghua University / Z.AI / Northeastern University, arXiv 2025)

DeepDive는 앞의 세 논문과 각도가 다르다. 앞선 논문들이 HotpotQA류의 "직접 검색(direct search)" 태스크 — 명확한 개체 몇 개만 찾으면 풀리는 멀티홉 QA — 를 다뤘다면, DeepDive는 BrowseComp류의 "심층 검색(deep search)" 태스크를 겨냥한다. BrowseComp의 질문은 흐릿한(blurry) 개체 여러 개를 동시에 좁혀 나가야 풀리는 형태다. 예를 들어 "1960\~80년대에 방영되고 50화 미만인 TV쇼에 나온, 관객에게 가끔 말을 거는(4차원의 벽을 깨는) 캐릭터"처럼 단서 하나하나가 모호하다.

**데이터부터 어렵게 만든다 — 지식 그래프 기반 합성.** DeepDive가 지적하는 문제는 기존 멀티홉 QA 데이터가 "진짜 찾기 어려운" 질문이 아니라는 점이다. 이를 해결하기 위해 지식 그래프(KILT, AMiner) 위에서 랜덤 워크로 경로를 뽑고, 이를 LLM으로 흐릿하게 만드는 파이프라인을 쓴다.

1. **경로 탐색**: 시작 노드 $$v_0$$에서 $$k \in [5, 9]$$ 스텝만큼 랜덤 워크로 경로 $$P = [v_0, v_1, \ldots, v_k]$$를 뽑는다. 다음 노드 후보는 out-degree가 $$[d_{min}, d_{max}] = [4, 8]$$ 범위인 노드로 제한한다 — 너무 유명한 노드(out-degree 과다)는 답이 뻔해지고, 너무 고립된 노드는 경로 확장이 막히기 때문이다.
2. **경로 일관성 확보**: 후보 노드 중 다음 노드를 LLM이 직접 골라 경로의 논리적 일관성을 유지한다.
3. **속성 흐리기**: 경로의 각 노드에 딸린 속성(날짜, 이름, 지역 등)을 결합한 attribute-rich path를 만든 뒤, Gemini-2.5-Pro로 구체적인 날짜를 범위로 뭉개는 식으로 obfuscate해서 질문-답 쌍을 합성한다.
4. **난이도 필터**: 검색 기능이 있는 프론티어 모델(GPT-4o)에게 각 질문을 4번 풀게 시켜, 한 번이라도 풀리면 그 질문을 버린다. 4번 모두 실패한 질문만 데이터셋에 남긴다.

이 파이프라인으로 3,250개의 QA 쌍을 만들어 1,016개는 SFT용, 2,234개는 RL용으로 나눴다(논문 서론에는 3,090건으로도 언급되는데, 데이터 합성 절의 세부 분할을 기준으로 삼는다). 콜드스타트 SFT는 Claude-4-Sonnet-Thinking으로 여러 번 시도·reject sampling해 858개의 고품질 트레이스를 확보했다.

**reward 설계 — strict binary reward에 redundancy penalty를 뺀다.** DeepDive는 GRPO를 쓰되 reward를 두 부분으로 나눈다.

첫째, 정오·형식을 AND로 묶은 엄격한 binary reward다.

$$r(T) = \begin{cases} 1 & (\forall i, \text{Format}(c_i, a_i)) \wedge \text{Judge}(a_{eos}, a^*) \\ 0 & \text{그 외} \end{cases}$$

모든 스텝의 추론·행동이 형식을 지켰고(도구 호출 채점, [#10](/blog/2026/tool-call-reward/)), 최종 답이 LLM judge가 보기에 정답과 같다고 판정될 때만(judge 채점, [#11](/blog/2026/agentic-judge-rubric/)) 1점이다. 하나라도 어긋나면 0점. 정답 표현이 다양할 수 있어("Inter Corp." vs "Inter Corporation") 문자열 매칭 대신 LLM judge를 답 동치 판정에 쓴다.

둘째, 검색 다양성을 유도하는 redundancy penalty다. 트레젝토리 $$T$$ 안의 검색 질의들 $$Q = [q_1, q_2, \ldots, q_T]$$에 대해, 질의를 키워드 집합으로 보고 두 질의의 자카드 유사도를 구한다.

$$\text{sim}(q_i, q_j) = \frac{\lvert q_i \cap q_j \rvert}{\lvert q_i \cup q_j \rvert}$$

트레젝토리 전체의 유사도는 모든 질의 쌍의 평균이다.

$$S(T) = \frac{1}{T(T-1)} \sum_{i \neq j} \text{sim}(q_i, q_j), \quad S(T) \in [0, 1]$$

$$S(T) = 1$$이면 모든 질의가 동일하다는 뜻이고, $$0$$이면 완전히 서로 다른 질의라는 뜻이다. 최종 reward는 이 두 항을 합친다.

$$r'(T) = r(T) - \lambda \cdot S(T)$$

$$\lambda = 0.1$$로, 정답을 맞히는 것이 여전히 지배적인 목표지만 같은 질의를 반복할수록 소폭 감점된다. 이 페널티는 [#8](/blog/2026/reward-shaping-agentic/)에서 다룬 shaping의 전형적인 형태다 — outcome reward(전부 아니면 전무인 binary)만으로는 학습이 안 뜨는 문제를, 트레젝토리 통계에서 뽑은 보조 항으로 부드럽게 만든다.

**loss masking도 그대로 쓴다.** 논문의 그림에서 확인되는 바로는, reasoning·tool call·answer 토큰에는 loss가 걸리고 web content(관찰) 토큰에는 loss가 걸리지 않는다 — Search-R1의 retrieved token masking과 같은 원리다.

**결과.** DeepDive-32B는 SFT만 했을 때 BrowseComp 9.5%에서, 멀티턴 RL을 거친 뒤 15.3%로 올랐다(같은 조건 QwQ-32B는 1.3\~1.7% 수준, 오픈소스 WebSailor-32B는 10.5%). 같은 비교로 BrowseComp-ZH는 23.0% → 29.7%, Xbench-DeepSearch는 48.5% → 51.8%, SEAL-0은 23.9% → 25.5%로 올랐다. 9B 모델에서는 이 RL 이득이 훨씬 작았는데(BrowseComp 5.6% → 6.3%), 논문은 이를 작은 모델의 제한된 추론 용량이나 합성 데이터에 대한 과적합 경향으로 추정한다. 참고로 OpenAI의 Deep Research는 같은 BrowseComp에서 51.5%를 기록했다(DeepDive 논문의 베이스라인 표에 인용된 수치) — 다만 Deep Research의 구체적인 reward 설계나 학습 방식은 공개된 자료로 확인되지 않는다.

# Experiments

## reward 구성 비교

| 논문                  | 정오 항                           | 형식 항                        | 페널티/보너스 항                                      | 학습 스테이지                      |
| --------------------- | --------------------------------- | ------------------------------ | ----------------------------------------------------- | ---------------------------------- |
| Search-R1             | $$\text{EM}(a_{pred}, a_{gold})$$ | 없음                           | 없음                                                  | 단일 스테이지                      |
| R1-Searcher (Stage 1) | 없음(정답 무관)                   | 형식 올바르면 $$+0.5$$         | 검색 시도 $$n \geq 1$$이면 $$+0.5$$                   | 2단계 중 1단계                     |
| R1-Searcher (Stage 2) | F1 score                          | 형식 틀리면 $$-2$$             | 없음(retrieval reward 제거)                           | 2단계 중 2단계                     |
| ReSearch              | F1 (0이 아닐 때 그대로)           | F1=0인데 형식만 맞으면 $$0.1$$ | 없음                                                  | 단일 스테이지                      |
| DeepDive              | LLM Judge 답 동치 판정 (AND)      | 모든 스텝 Format 만족 (AND)    | 자카드 유사도 $$\lambda \cdot S(T)$$, $$\lambda=0.1$$ | 단일 스테이지, 콜드스타트 SFT 선행 |

이 표에서 눈에 띄는 흐름이 있다. Search-R1은 형식 항이 아예 없고, R1-Searcher는 형식 항의 부호와 크기를 스테이지마다 바꾸고, DeepDive는 형식 항을 정오 항과 AND로 묶어 "형식이 조금이라도 틀리면 정답이어도 0점"이라는 훨씬 가혹한 조건을 건다. 형식 위반에 대한 관용도가 논문마다 다른데, 이 관용도가 실제로 학습 안정성에 영향을 준다는 게 R1-Searcher의 ablation(뒤에서 다룸)이 보여주는 지점이다.

## 벤치마크별 개선폭

| 논문                       | 벤치마크                    | 비교 기준                                 | 결과                                 |
| -------------------------- | --------------------------- | ----------------------------------------- | ------------------------------------ |
| Search-R1 (Qwen2.5-7B)     | 7개 QA 평균 EM              | RAG(0.304)                                | Search-R1-base 0.431 (상대 +41%)     |
| Search-R1 (Qwen2.5-7B)     | 7개 QA 평균 EM              | 최강 베이스라인 Rejection Sampling(0.348) | Search-R1-base 0.431 (상대 +24%)     |
| Search-R1 (Qwen2.5-3B)     | 7개 QA 평균 EM              | 최강 베이스라인 RAG(0.270)                | Search-R1-instruct 0.325 (상대 +20%) |
| R1-Searcher (Llama-3.1-8B) | HotpotQA (LLM judge)        | ReARTeR+GPT-4o-mini                       | 상대 +48.2%                          |
| R1-Searcher                | 2WikiMultiHopQA (LLM judge) | ReARTeR+GPT-4o-mini                       | 상대 +21.7%                          |
| R1-Searcher                | Bamboogle, OOD (LLM judge)  | ReARTeR+GPT-4o-mini                       | 상대 +4.0%                           |
| ReSearch (Qwen2.5-7B)      | 4개 멀티홉 QA 평균          | 최강 베이스라인                           | EM 상대 +15.81%, judge 상대 +17.56%  |
| ReSearch (Qwen2.5-32B)     | 4개 멀티홉 QA 평균          | 최강 베이스라인                           | EM 상대 +14.82%, judge 상대 +15.46%  |
| DeepDive-32B               | BrowseComp                  | SFT-only(9.5%)                            | RL 15.3%                             |
| DeepDive-32B               | BrowseComp-ZH               | SFT-only(23.0%)                           | RL 29.7%                             |
| DeepDive-32B               | Xbench-DeepSearch           | SFT-only(48.5%)                           | RL 51.8%                             |
| DeepDive-32B               | SEAL-0                      | SFT-only(23.9%)                           | RL 25.5%                             |

공통적으로 확인되는 패턴 하나는, 모든 논문에서 "RL이 SFT-only보다 낫다"는 비교가 등장한다는 점이다. DeepDive는 이를 가장 직접적으로 보여준다 — 같은 콜드스타트 SFT 체크포인트에서 시작해 RL만 추가했을 때 네 벤치마크 모두에서 안정적으로 개선됐다.

## 반례 — redundancy penalty를 넣고 뺐을 때

DeepDive는 reward 설계에 대한 ablation을 논문에 직접 실었다. redundancy penalty를 껐을 때와 켰을 때의 차이는 이렇다.

- **끄면(w/o redundancy penalty)**: 학습이 진행돼도 트레젝토리당 검색 질의 개수가 계속 늘어난다. 정답 하나를 맞히기 위해 거의 동일한 질의를 여러 번 반복하는 경향이 관찰된다 — binary reward만으로는 "검색을 많이 하는 것"과 "다양하게 검색하는 것"을 구분하지 못하기 때문에, 모델 입장에서는 같은 질의를 살짝 바꿔 반복 제출하는 게 손해 볼 게 없는 전략이 된다.
- **켜면(w/ redundancy penalty)**: 학습 후반부 정확도가 약 20% 높아지고, 동일 조건에서 도구 호출 횟수는 약 14% 줄어든다. 즉 페널티가 정답률을 깎지 않으면서 탐색을 효율화한다.

메커니즘은 앞서 본 수식 $$r'(T) = r(T) - \lambda \cdot S(T)$$ 그대로다. $$S(T)$$는 트레젝토리 안 모든 질의 쌍의 평균 자카드 유사도이므로, "검색 5번 중 3번이 사실상 같은 질의"인 궤적은 $$S(T)$$가 커져 감점을 받는다. $$\lambda = 0.1$$이라는 작은 가중치가 핵심이다 — 페널티가 너무 크면 모델이 아예 검색을 줄여버릴 위험이 있는데($$S(T)$$를 낮추는 가장 쉬운 방법은 질의 자체를 적게 내는 것이다), 논문은 정답률 손실 없이 탐색만 효율화된 결과를 보고한다.

같은 논문의 format reward ablation도 짚어둘 만하다. format reward를 빼면 정확도 곡선이 학습 내내 약 8.0 근처에 거의 평평하게 머물지만, 넣으면 꾸준히 상승하며 항상 약 2%p 더 높은 수준을 유지한다. 즉 이 도메인에서 형식 항은 단순한 "패널티 방지용 안전장치"가 아니라, 학습 자체를 이륙시키는 데 관여한다.

## R1-Searcher가 보여준 두 번째 반례 — EM에서 F1로

R1-Searcher의 ablation도 같은 종류의 교훈을 다른 각도에서 보여준다. 답 reward를 EM으로 두면 학습 중 응답 길이가 짧아지는 방향으로 붕괴하고 최종 성능도 F1 대비 크게 뒤처졌다(F1 대비 52.6% 격차). EM은 "정답 문자열과 완전히 일치해야 1점"이라는 극단적으로 엄격한 채점 기준이라, 개방형 질문 응답에서는 정답의 의미가 맞아도 표현이 달라 0점 처리되는 경우가 잦다. 이 경우 모델 입장에서는 애매한 표현을 아예 짧게 잘라 확률을 낮추는 방향으로 신호가 왜곡된다. Search-R1이 같은 EM으로도 성공했다는 사실과 대조하면, "EM이 절대적으로 나쁘다"는 결론이 아니라 **채점 기준의 엄격도와 학습 데이터·태스크 난이도가 맞물려야 한다**는 결론에 가깝다.

# 통계 요약

| 항목               | Search-R1                                      | R1-Searcher                              | ReSearch                | DeepDive                             |
| ------------------ | ---------------------------------------------- | ---------------------------------------- | ----------------------- | ------------------------------------ |
| 정오 채점          | EM                                             | F1 (Stage 2)                             | F1                      | LLM Judge                            |
| 형식 채점          | 없음                                           | 있음 (스테이지별 부호 다름)              | 있음 (F1=0일 때만 개입) | 있음 (정오와 AND)                    |
| 검색 다양성 페널티 | 없음                                           | 없음                                     | 없음                    | 자카드 유사도 기반, $$\lambda=0.1$$  |
| 학습 커리큘럼      | 단일 스테이지                                  | 2단계(검색 시도 → 정답)                  | 단일 스테이지           | 콜드스타트 SFT + 단일 RL 스테이지    |
| loss masking       | 검색 결과 토큰 마스킹                          | 검색 결과 토큰 마스킹                    | 명시적 언급 없음        | 관찰 토큰 마스킹(그림상 확인)        |
| 대표 hacking 관찰  | (논문에 명시 없음, 형식은 안정적이었다고 보고) | 가짜 검색 결과 생성, 검색 생략, CEM 패딩 | (논문에 명시 없음)      | 중복 질의 반복                       |
| 대표 방어          | 순수 EM + 토큰 마스킹                          | 2단계 커리큘럼 + 형식 페널티             | 조각별 reward           | redundancy penalty + strict AND 채점 |

# Conclusion

네 논문을 관통하는 메시지는 이렇다. **검색 에이전트는 reward의 조달 자체는 쉽다. 정답이 있는 QA라 EM이나 F1 같은 규칙, 또는 표현 다양성이 문제될 때는 LLM judge 하나만 있으면 채점이 끝난다. 하지만 그 쉬운 스칼라 하나가 트레젝토리 전체에 브로드캐스트되는 순간, "어느 검색이 결정적이었는가"라는 질문에는 아무도 답하지 못한다.** 이 시리즈의 언어로 말하면, 이 도메인은 [#9](/blog/2026/environment-as-reward/)의 환경 검증과 [#10](/blog/2026/tool-call-reward/)의 도구 호출 채점을 한 스칼라 안에 합쳐 쓰지만, 그 스칼라를 [#5](/blog/2026/turn-level-reward/)\~[#7](/blog/2026/token-segment-credit/)에서 다룬 턴·스텝·토큰 단위로 쪼개는 시도는 거의 하지 않는다.

네 논문이 이 성긴 신호를 다루는 전략은 저마다 달랐다.

1. **Search-R1**: reward를 최대한 단순하게 유지하는 대신, retrieved token masking으로 "누구를 학습시킬지"만 정교하게 걸렀다.
2. **R1-Searcher**: reward 자체는 단순하게 두되, 학습 커리큘럼을 2단계로 쪼개 "검색하는 법"과 "정답 맞히는 법"을 순차적으로 가르쳤다.
3. **ReSearch**: 형식과 정답을 조각별 함수 하나로 압축해 커리큘럼 없이도 비슷한 효과를 노렸다.
4. **DeepDive**: 트레젝토리 통계(질의 간 자카드 유사도)를 shaping 항으로 추가해, outcome reward만으로 잡히지 않는 "검색 다양성"이라는 축을 보완했다.

hacking 패턴도 이 전략의 이면이었다. 형식 항이 없거나 약하면 검색을 생략하거나(R1-Searcher의 Stage 1 생략 실험) 가짜 검색 결과를 지어내는(`<begin_of_documents>` 없이 문서를 만드는) hacking이 나타났고, 다양성 페널티가 없으면 같은 질의를 반복하는 hacking이 나타났다(DeepDive ablation). 반대로 채점 기준을 너무 엄격하게 잡으면(EM) 응답이 위축되는 방향의 붕괴가 나타났다(R1-Searcher ablation). 즉 이 도메인의 reward hacking은 "몰래 속이는" 형태보다는, **채점 기준의 엄격도와 형식 강제의 강도가 학습 안정성과 직결되는 형태**로 나타난다.

한계도 분명하다. 네 논문 모두 "어느 검색 스텝이 결정적이었는가"를 직접 묻는 process-level 신호는 설계하지 않았다. DeepDive의 redundancy penalty가 가장 근접한 시도지만, 이것도 트레젝토리 전체의 통계량이지 스텝 단위 credit은 아니다. 이 공백은 다음 도메인들에서도 계속 마주치게 될 질문이다 — [#13](/blog/2026/swe-agent-rl/) 코드 에이전트는 테스트 실행이라는 훨씬 비싸지만 훨씬 세밀한 검증 수단을 갖고 있고, [#14](/blog/2026/web-gui-agent-rl/) 웹·GUI 에이전트는 반대로 정답 자체가 모호해 judge에 더 크게 의존한다.

# 참고 문헌

- Jin et al., 2025. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). arXiv:2503.09516.
- Song et al., 2025. [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592). arXiv:2503.05592.
- Chen et al., 2025. [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470). arXiv:2503.19470.
- Lu et al., 2025. [DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL](https://arxiv.org/pdf/2509.10446). arXiv:2509.10446.
- Wei et al., 2025. [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://arxiv.org/abs/2504.12516). arXiv:2504.12516.
- [GitHub: PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [GitHub: THUDM/DeepDive](https://github.com/THUDM/DeepDive)
- [GitHub: Agent-RL/ReSearch](https://github.com/Agent-RL/ReSearch)
- Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). (GRPO 원 논문)

---

# Agentic RL 설계 시리즈

이 글은 Agentic RL 설계 시리즈의 열두 번째 글이다.

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
  <li><strong>(현재 글)</strong> 검색 에이전트 — Search-R1에서 DeepDive까지</li>
  <li><a href="/blog/2026/swe-agent-rl/">코드 에이전트</a> — SWE-RL과 테스트라는 reward</li>
  <li><a href="/blog/2026/web-gui-agent-rl/">웹·GUI 에이전트</a> — end-to-end 멀티턴 RL</li>
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
