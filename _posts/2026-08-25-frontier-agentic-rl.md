---
layout: post
title: "프론티어 모델은 실제로 어떻게 하나"
date: 2026-08-25 09:16:00 +0900
description: "Agentic RL 설계 시리즈 #16(완결) — 아홉 개 프론티어 모델의 공개 자료에서 agentic reward·credit assignment 설계만 추려 비교한다"
categories: [paper]
tags: [rl, agentic, reward, credit-assignment, kimi, deepseek, qwen, llama, gemma, upstage, skt, paper]
giscus_comments: true
related_posts: true
---

> 이 글은 아홉 개 프론티어 모델의 공개 자료를 가로지른다 — [Kimi K3](https://github.com/MoonshotAI/Kimi-K3)(Moonshot AI), [A.X K2](https://github.com/SKT-AI/A.X-K2)(SKT), [Solar Open 2](https://arxiv.org/abs/2607.20062)(Upstage), [GLM-4.5](https://arxiv.org/abs/2508.06471)(Zhipu AI), [DeepSeek-V4](https://arxiv.org/abs/2606.19348)(DeepSeek-AI), [Qwen3](https://arxiv.org/abs/2505.09388)(Alibaba), [K-EXAONE 2.0](https://arxiv.org/abs/2608.04505)(LG AI Research), [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)(Meta), [Gemma 4](https://arxiv.org/abs/2607.02770)(Google DeepMind). 보조 자료로 [MiniMax-M1](https://arxiv.org/abs/2506.13585)도 곁들인다. 같은 자료를 [RLHF 시리즈 #44](/blog/2026/frontier-reward-design/)가 helpfulness reward 전반(규칙·스칼라 RM·reference judge·GRM 4분류)의 렌즈로 훑었다면, 이 글은 **agentic 축 하나만 남기고** credit assignment 입도, 환경 검증 지점, tool call 채점, 컨텍스트 관리의 렌즈로 같은 자료를 다시 읽는다.

# Introduction

15편 동안 이 시리즈는 "에이전트를 RL로 학습시킬 때 무엇이 깨지고 어떻게 고치는가"를 이론과 개별 논문 단위로 뜯었다. 에이전트 RL이 단일턴 RLHF와 근본적으로 다른 이유와 공(credit)을 어디로 돌릴지의 큰 그림([#1](/blog/2026/agentic-rl-landscape/)\~[#3](/blog/2026/multi-turn-rl-practice/)), 궤적이 길어지면 그룹 전체가 붕괴하는 문제([#4](/blog/2026/outcome-vs-process-agentic/)), 공을 턴·스텝·토큰 단위로 잘게 나누는 credit assignment 기법들([#5](/blog/2026/turn-level-reward/)\~[#7](/blog/2026/token-segment-credit/)), 그 아래로 내려가면 추정이 시작되고 추정은 hacking된다는 경고([#8](/blog/2026/reward-shaping-agentic/)), 환경 자체를 reward로 쓰는 법([#9](/blog/2026/environment-as-reward/)), 도구 호출을 채점하는 법([#10](/blog/2026/tool-call-reward/)), judge가 궤적을 rubric으로 채점하는 법([#11](/blog/2026/agentic-judge-rubric/)), 검색·코드·웹 에이전트 각 도메인의 특수성([#12](/blog/2026/search-agent-rl/)\~[#14](/blog/2026/web-gui-agent-rl/)), 그리고 이 모든 걸 뚫는 hacking 사례들([#15](/blog/2026/agentic-reward-hacking/)). 하나하나는 논문 한 편이 문제 하나에 답한 결과였다.

그런데 실제로 3T급 모델을 학습시키는 팀은 이 해법 목록 중 무엇을 골랐을까. 이 글은 아홉 개 공개 자료 — Kimi K3, A.X K2, Solar Open 2, GLM-4.5, DeepSeek-V4, Qwen3, K-EXAONE 2.0, Llama 4, Gemma 4 — 를 가로질러 **agentic reward 설계**만 추려 비교한다. 방법은 정직해야 한다는 원칙 하나를 지킨다: 자료에 없으면 "공개 자료에 없음"이라고 쓴다. [RLHF 시리즈 #45](/blog/2026/frontier-safety-design/)가 "Kimi K3 리포트에 safety가 0회 등장한다"는 부재 자체를 발견으로 삼았던 것과 같은 태도다. 이 글에서도 **단어 등장 횟수**를 근거로 쓴다 — "agentic"이 리포트에 몇 번 나오는지, "credit assignment"라는 이 시리즈의 핵심 용어가 실제로 몇 번 나오는지.

미리 결론의 윤곽을 셋만 말하면 이렇다.

1. **"전문가를 도메인별로 따로 키운 뒤 증류로 합친다"는 구조가 최소 셋(Kimi K3·Solar Open 2·DeepSeek-V4)에서 겹친다.** 이 중 Kimi K3와 Solar Open 2는 이름(MOPD)까지 같다.
2. **credit 입도는 이론이 준비한 만큼 내려가지 않는다.** 이 시리즈 2부가 다룬 턴·스텝·토큰 단위 credit assignment는 아홉 모델 중 어디에서도 주 reward로 쓰이지 않는다. 거의 전부 trajectory·outcome 레벨에 머물고, 유일하게 더 세밀해지는 지점은 Solar Open 2의 "graded process rubric"이다.
3. **공개 수준의 편차가 능력 격차보다 크다.** Llama 4와 Gemma 4는 agentic RL을 사실상 언급하지 않는다 — 벤치마크 점수는 공개하면서 그 점수를 만든 reward 설계는 감춘다.

아홉 모델을 한 장으로 요약하면 이렇다. 근거는 Method에서 모델별로 짚는다.

| 모델         | 소속            | agentic reward 조달처                                                               | credit 입도                                    | RL 알고리즘                           | 공개 수준 |
| ------------ | --------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------- | --------- |
| Kimi K3      | Moonshot AI     | 규칙(5,100만+ 샌드박스 실행검증) + Agentic GRM(judge가 rubric 생성→scorepad)        | trajectory(토너먼트 binary), 증류 단계만 token | 3단계 RL(9전문가) → MOPD 증류         | 상세      |
| A.X K2       | SKT             | 규칙(schema/argument 일치) + 게이트형 judge(binary gate→pointwise)                  | trajectory(이진 게이트 후 pointwise)           | CISPO + GDPO                          | 상세      |
| Solar Open 2 | Upstage         | 환경이 생성한 read-back 검증기 + graded process rubric + executable end-state check | trajectory\~criterion 단위(9모델 중 가장 세밀) | 완전 비동기 GRPO → 12전문가 MOPD 증류 | 매우 상세 |
| GLM-4.5      | Zhipu AI        | 규칙(SWE fail-to-pass/pass-to-pass) + 최종 답 정확도                                | trajectory(outcome) + 포맷 위반 시 이진 게이트 | GRPO 변형(무-KL) + 반복 자기증류      | 상세      |
| DeepSeek-V4  | DeepSeek-AI     | 도메인 reward("성공 기준"으로만 서술, 세부 비공개) + GRM                            | 불명(trajectory로 추정)                        | GRPO → on-policy 증류(역-KL)          | 낮음      |
| Qwen3        | Alibaba         | 환경 실행 피드백(멀티턴 rollout), 3종 reward 중 어떤 것인지 미명시                  | 불명                                           | GRPO + General RL(20+ 태스크 중 하나) | 낮음      |
| K-EXAONE 2.0 | LG AI Research  | 오프라인 preference — 행동의 정확성 + 답의 깊이·포괄성을 judge가 평가               | trajectory(pairwise)                           | GrouPER(+AGAPO는 언급만, 세부 비공개) | 중간      |
| Llama 4      | Meta            | 공개 자료에 없음("online RL"이라고만 서술)                                          | 공개 자료에 없음                               | 공개 자료에 없음                      | 없음      |
| Gemma 4      | Google DeepMind | 공개 자료에 없음(tool-call 포맷 예시만 공개)                                        | 공개 자료에 없음                               | 공개 자료에 없음                      | 없음      |

# Background

## 방법: 왜 단어 등장 횟수를 근거로 쓰는가

기술 리포트 아홉 편(+보조 하나)을 나란히 놓고 "agentic", "agent", "tool", "reward", "credit assignment" 다섯 단어의 등장 횟수를 grep으로 셌다. 이 방법에는 명백한 한계가 있다 — 리포트 길이가 3,555단어(Llama 4 블로그)부터 28,771단어(Kimi K3)까지 8배 차이 나고, 단어 하나의 등장이 곧 설계의 깊이를 뜻하지 않는다. 그런데도 이 방법을 쓰는 이유는 단순하다. **"공개했다/안 했다"는 질적 판단은 읽는 사람의 기대에 따라 갈리지만, 등장 횟수 0은 누가 세어도 0이다.** 뒤에 나올 "credit assignment 0회" 같은 숫자는 이 글에서 가장 단단한 근거이자, 가장 정직하게 확인 가능한 근거다.

문서 종류도 셋으로 갈린다. Kimi K3와 A.X K2는 arXiv 없이 **GitHub에 PDF tech report**만 올렸다. GLM-4.5·DeepSeek-V4·Qwen3·K-EXAONE 2.0·Solar Open 2·Gemma 4·MiniMax-M1은 **arXiv 프리프린트**다(문헌 조사 시점 기준 정식 학회/저널 게재 확인 안 됨 — Comments·Journal-ref 필드 없음). Llama 4는 tech report 없이 **공식 블로그 포스트**로만 공개됐다. 문서 종류 자체가 이미 공개 의지의 신호다.

## 이 시리즈가 이론으로 준비한 것 — 다시 요약

이 편만 읽는 독자를 위해 한 문단으로 되짚는다. 에이전트를 RL로 학습시킬 때 가장 먼저 부딪히는 문제는 **credit assignment**다 — 궤적이 수십\~수백 턴으로 늘어나면 "에피소드 끝에 성공/실패 1점"이라는 outcome reward만으로는 3턴째의 결정적 실수와 47턴째의 사소한 실수가 똑같은 벌점을 받는다. 이 시리즈는 이 문제를 입도(granularity)의 문제로 재정의하고, trajectory보다 잘게 자르는 방법들(턴·스텝·토큰-세그먼트)과 그 신호를 어디서 얻는지(환경 실행, 도구 호출 검증, judge·rubric)를 각각 한 편씩 다뤘다. 이 좌표를 정리하면 아래와 같고, Method에서 각 모델을 이 좌표로 잰다.

| 축                            | 이론이 준비한 선택지                                              | 관련 편                                                                              |
| ----------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| credit을 어디까지 잘게 나누나 | trajectory / turn / step / token-segment                          | [#4](/blog/2026/outcome-vs-process-agentic/)\~[#7](/blog/2026/token-segment-credit/) |
| reward를 어디서 조달하나      | 환경 실행 검증 / 도구 호출 스키마 검증 / judge·rubric             | [#9](/blog/2026/environment-as-reward/)\~[#11](/blog/2026/agentic-judge-rubric/)     |
| 여러 전문가를 어떻게 합치나   | 별도 학습 후 증류 vs 단일 정책에서 공동 학습                      | [#2](/blog/2026/credit-assignment-survey/)                                           |
| hacking을 어떻게 막나         | 판정 정의를 대리 지표에서 떼어놓기, 다중 reward 결합 시 신호 보존 | [#15](/blog/2026/agentic-reward-hacking/)                                            |
| 컨텍스트를 어떻게 관리하나    | 환경 관측 마스킹, 추론 흔적 유지/폐기                             | [#3](/blog/2026/multi-turn-rl-practice/)                                             |

# Method

## Kimi K3 — Agentic GRM과 MOPD, 그리고 5,100만 개의 샌드박스

Kimi K3(Moonshot AI, 2.8T 파라미터 MoE, 104B 활성)는 아홉 모델 중 가장 상세하게 agentic RL을 서술한다. 후처리는 세 단계다 — SFT로 콜드스타트를 잡고, RL로 세 도메인(일반 태스크·general agents·coding agents) × 세 reasoning-effort 수준(low/high/max)의 **아홉 전문가**를 따로 키운 뒤, **Multi-Teacher On-Policy Distillation(MOPD)**로 하나의 모델에 합친다. general agents 도메인은 장기 지평 어시스턴트 작업·deep research·문단 단위 글쓰기를, coding agents는 SWE·코딩·커널 최적화·웹 개발을 포괄한다.

reward 조달처는 두 갈래다.

- **검증 가능한 도메인**: 규칙 기반 verifiable reward를 쓴다. 얼마나 대규모로 쓰는지는 인프라 절에 숫자로 나온다 — **학습·평가 전 과정에서 5,121만 9,741개의 샌드박스가 150만 5,678개의 이미지에 걸쳐 생성됐다.** 이 샌드박스는 AgentENV라는 Firecracker 기반 microVM 런타임으로, 컨테이너 런타임에서 관찰된 커널 패닉·데드락 문제를 isolation으로 해결하면서도 디스크 마운트·컨테이너 실행·가상머신 구동까지 허용해 에이전트의 탐색 자유도를 지키는 쪽을 택했다. checkpoint 133ms·resume 49ms의 저지연 상태 저장을 지원하고, 세 가지 고수준 연산을 제공한다 — **Pause/Resume**(모델 추론을 기다리는 동안 샌드박스는 리소스를 0으로 만든다. 이 대기 시간이 샌드박스 수명의 최대 98%를 차지한다), **Fork**(원본을 실행 상태로 둔 채 동일 상태의 새 샌드박스를 복제한다. 부작용 없이 reward를 판정하려는 용도다), **Snapshot**(정기 저장으로 오류 복구). 이 Fork 연산이 [#9 환경이 곧 reward다](/blog/2026/environment-as-reward/)가 다룬 "환경 신호를 어떻게 오염 없이 얻는가" 문제에 대한 실전 답이다 — 판정을 위해 원본 궤적과 별개의 복제본에서 검증을 돌린다.
- **검증 불가능한 일반 태스크**: **Agentic Generative Reward Model(GRM)**을 쓴다. judge가 강제된 프로토콜을 따른다 — (1) 출력(결과물·상품·텍스트)을 읽고, (2) **rubric을 생성**하고, (3) 각 후보를 그 rubric으로 채점하고, (4) 채점 결과를 **scorepad**에 기록한다. 이전 세대(K2.5)의 토너먼트식 이진 비교를 이어받되, judge가 채점 기준 자체를 스스로 만든다는 점이 새롭다. 이 구조는 [#11 궤적을 judge가 채점한다](/blog/2026/agentic-judge-rubric/)가 다룬 "judge가 평가 원칙을 스스로 생성한다"는 아이디어를 프로덕션 스케일로 옮긴 사례다.

hacking 방어도 명시적이다. Agentic GRM이 "길게 쓰면 이긴다"는 식으로 뚫리는 걸 막으려고, 콜드스타트 모델에서 추정한 초기 길이 $$\ell_0$$의 $$\sigma$$배를 넘는 후보는 binary comparison에서 자동으로 진다. 별도로, **reasoning-effort RL**은 문제마다 콜드스타트 모델이 추정한 초기 토큰 예산 $$b_0(x)$$를 두고, 궤적의 누적 토큰 수 $$T(y)$$가 배율 $$\tau$$를 곱한 임계값을 넘으면 task reward를 $$-1$$로 덮어쓴다. 일반 태스크에서 $$T(y)$$는 사고 토큰 수를, agentic 태스크에서는 추론 흔적과 도구 호출 인자를 합친 누적 출력 토큰 수를 잰다. 이 페널티는 [#8 shaping은 약인가 독인가](/blog/2026/reward-shaping-agentic/)가 경고한 "판정 지점 아래는 추정이고 hacking된다"는 문제에 대한 직접적 대응이다 — 토큰 예산 초과라는 명확한 규칙으로 verbosity·overthinking hacking의 여지를 원천 차단한다.

**MOPD**는 아홉 전문가를 하나로 합치는 단계다. 도메인 $$d$$와 reasoning-effort $$e$$가 주어지면, 학생 정책은 해당 $$(d,e)$$ 교사 정책과의 per-token 정렬을 목표로 최적화된다(원문 수식은 OCR 신뢰도가 낮아 여기서는 방향만 서술한다 — 교사·학생 로그확률 비의 로그를 clip한 형태). 아홉 개의 이질적 전문가가 각자 강한 영역에서 정점을 찍은 뒤, 이 증류 단계에서 하나의 배포 가능한 모델로 수렴한다.

컨텍스트 관리 관련해서도 구체적인 서술이 있다. 채팅 템플릿은 assistant 메시지 본문을 세 channel(think·response·tools)로 나누는데, Kimi K3는 **preserved thinking만 지원**한다 — 사고 흔적을 담는 think channel이 내용이 비어 있어도 대화 이력에서 항상 유지돼, 모델이 턴을 넘나들며 일관된 메시지 구조를 관찰하게 만든다. 또한 도구 호출 인자가 정형 블록으로 분해되지 않을 때 쓰는 pure-JSON fallback 블록은 **입력 토큰에만 등장하고 모델 출력에는 나오지 않으며, 그 손실이 학습 중 마스킹된다**고 명시한다. 둘 다 [#3 멀티턴 RL 실무 가이드](/blog/2026/multi-turn-rl-practice/)가 다룬 "무엇을 정책의 그래디언트에 태우고 무엇을 관측으로만 남길 것인가"의 구체 사례다.

한 가지 정직하게 짚을 대목: Kimi K3 리포트 어디에도 "credit assignment"라는 표현은 나오지 않는다. "turn-level"·"step-level"·"token-level" 같은 이 시리즈 2부의 용어도 등장하지 않는다. Agentic GRM의 채점 단위는 후보 궤적 전체(trajectory)이고, MOPD의 per-token 정렬은 credit assignment가 아니라 지식 증류의 대상이다. 즉 이 리포트가 보여주는 가장 세밀한 실제 credit 단위는 여전히 "궤적 하나"다.

## A.X K2 — 게이트형 judge로 대리 지표를 걸러낸다

A.X K2(SKT, 688B 파라미터 MoE, 33B 활성)는 사전학습부터 "agentic 애플리케이션을 위한 고성능 파운데이션"을 표방한다. RL은 네 그룹의 혼합으로 구성된다 — instruction following, human preference, **agentic tool use**, safety. 이 혼합 비율은 처음부터 고정하지 않고, 중간 RL 체크포인트로 모델의 약점을 짚어가며 재조정하는 "control surface"로 다룬다. 왜 네 그룹을 함께 학습시키나: 리포트는 "한 능력만 따로 RL로 최적화하면 그 단계의 reward만 과최적화되고, 그 단계에 없는 행동은 퇴행한다"는 관찰을 근거로 든다.

**agentic tool use** 그룹의 reward가 이 글의 핵심이다. 두 갈래로 갈린다.

- **단일 스텝 도구 호출**: 방출된 호출을 참조 호출과 비교해 스키마 준수·인자 정확성을 검증하는 verifiable reward. [#10 도구 호출을 어떻게 채점하나](/blog/2026/tool-call-reward/)가 다룬 표준적 접근이다.
- **장기 지평 agentic 태스크**: **게이트형 judge**를 쓴다. 이진 게이트가 먼저 "응답이 구조적으로 유효하고 실행 가능한 tool-calling transcript인가"를 확인해 실패 시 최소 reward를 부여하고, **게이트를 통과한 응답만** reference 기반 pointwise judge로 넘긴다. 리포트는 이 설계 이유를 명시한다 — "judge가 유효한 tool call을 한 번도 내지 않은, 유창하기만 한 응답에 보상을 주는 것을 막기 위해서"다.

이 게이트 구조가 이 시리즈 [#15](/blog/2026/agentic-reward-hacking/)가 다룬 "판정 정의가 대리 지표인지 확인하라"는 체크리스트 항목의 실전 사례다. judge 하나만 두면 judge는 "말이 되는 텍스트"라는 대리 지표에 낚일 수 있다. 구조적 유효성이라는 값싼 필터를 앞에 두어, judge가 실제로 실행 가능한 행동에만 채점을 하도록 좁혀 놓는다.

RL 알고리즘은 **CISPO**(MiniMax 원조) + **GDPO**(Liu et al. 2026)다. CISPO는 token 단위 업데이트가 아니라 importance-sampling 가중치를 클립해, 확률은 낮지만 행동에 결정적인 토큰이 계속 그래디언트에 기여하게 한다. GDPO는 여러 reward를 **각각 따로 정규화한 뒤 결합**해 한 신호가 다른 신호를 잡아먹지 않게 한다 — task reward와 auxiliary format reward를 함께 쓸 때 이 정규화를 거친다. 이것이 [#15](/blog/2026/agentic-reward-hacking/)의 또 다른 체크리스트 항목 "여러 보상을 어떻게 결합했는지 명시하라"에 대한 A.X K2의 답이다. 여기에 더해 verbosity가 심한 데이터셋에는 group-relative length penalty(He et al. 2025)를 적용한다 — 같은 그룹 안에서 정답인 짧은 응답은 보상받고 긴 응답은 페널티를 받되, 오답에는 짧다고 보상을 주지 않는다.

human preference 그룹의 judge는 여섯 도메인(사실·추론·코딩·추출·창작·개방형)으로 태스크를 먼저 분류한 뒤, 정확성·완결성·명료성·유용성 네 축의 도메인별 rubric으로 채점하며 verbosity bias·reward hacking에 대한 명시적 방지 장치를 둔다고 서술한다. 다만 이 방지 장치의 구체 메커니즘(어떤 규칙인지)은 공개 자료에 없다.

credit 입도는 여기서도 trajectory 단위를 벗어나지 않는다. 게이트는 "전체 응답이 유효한가"를 이진으로 묻고, pointwise judge는 궤적 전체에 점수 하나를 매긴다. 리포트에 "credit assignment"라는 표현은 등장하지 않는다.

## Solar Open 2 — 아홉 모델 중 가장 세밀한 공개 자료

Solar Open 2(Upstage, 250B-A15B MoE)는 이 비교에서 단연 두드러진다. 후처리는 SFT → Multi-domain RL(STEM 중심 검증 가능 reward) → **Specialist**(열두 도메인 전문가를 SFT+도메인별 RL로 육성) → **MOPD**(열두 교사를 하나로 통합), 네 단계다. 열두 전문가는 세 계열로 나뉜다 — reasoning(수학·STEM·코드), **agents & tools**(코딩·범용 도구 사용·단일 워크스페이스·다중 워크스페이스·검색), preference & alignment(지시 따르기·인간 선호·안전·거절).

이 글의 관심사인 "환경이 진짜 신호를 주는 가장 세밀한 지점"([#9](/blog/2026/environment-as-reward/))에 가장 정직하게 답하는 곳이 여기다. 세 가지 agent 시나리오 계열을 각각 어떻게 검증하는지 구체적으로 서술한다.

- **일반 대화형 에이전트**: 상태 변경(Create/Update/Delete) 작업에 대한 검증기를 얻기 어렵다는 문제를, **검증기를 먼저 만드는 방식**으로 푼다 — 실제 환경에 mutation을 실행해 기대 상태 변화를 기록하고, 이를 읽어내는 read-back 검증기(환경의 읽기 도구 위에 얹은 sandboxed pytest suite)를 합성한 뒤, 지시문은 통제된 수준으로 모호하게 다시 쓴다. 모든 태스크는 **graded process rubric**(과정을 채점하는 rubric)과 **executable end-state check**(실행 가능한 최종 상태 검사)를 함께 갖고, 이 둘을 hard trace rule·LLM coherence judge·실행 가능한 read-back의 **3중 자기일관성 검증**으로 확정한다.
- **코딩 에이전트**: SWE 시나리오는 GitHub PR을 채굴해 문제 설명·정답 패치·테스트 패치·베이스 커밋·환경 요구사항이 모두 복원 가능한 것만 남기고, LLM이 합성한 Docker 환경이 빌드 가능하며 **fail-to-pass**(베이스 커밋에서 테스트 실패, 정답 패치 적용 후 통과)인 것만 채택한다. Git 히스토리는 통째로 제거해 정답이 버전 관리에서 새어나가지 않게 막는다. 터미널 시나리오는 자체 검증 행동(코드 수준 셀프테스트를 실행하고, 실패하면 수정 후 재검증)에 특히 무게를 둔다.
- **오피스워크 에이전트**: 11개 산업 × 12개 태스크 유형의 행렬로 조직된 OfficeVerse 파이프라인을 자체 구축했다. 각 태스크의 산출물(xlsx/docx/pptx/pdf)은 규칙 기반 체커 또는 LLM judge로 채점되는 **rubric 기준별**로 나뉘는데, **이 채점 기록이 SFT 채택 여부를 가르는 것과 동일한 기록에서 RL reward가 유도된다** — "과정 품질과 결과 품질을 의도적으로 분리한다"고 명시한다.

이 rubric-기준 단위 채점이 아홉 모델 가운데 trajectory보다 더 잘게 내려가는 유일한 공개 사례다. 완전한 turn-level이나 token-level은 아니지만, "궤적 하나에 점수 하나"보다는 한 단계 더 세밀하다.

RL 인프라도 agentic 특화다. 궤적 길이 분포가 롱테일이라(일부 궤적이 나머지보다 훨씬 오래 걸림) 동기식 학습은 대부분의 디바이스를 유휴 상태로 방치한다는 문제를, 다섯 요소로 구성된 **완전 비동기 설계**로 푼다 — 학습기·rollout 엔진 분리, staleness 제어(fresh-token-fraction gateway), 길이·staleness 인지 배치 샘플링, 양방향 importance sampling, 그리고 **환경 실패 필터링과 그룹 크기 복구**. 마지막 항목이 [#8](/blog/2026/reward-shaping-agentic/)이 다룬 문제의 변주다 — 파일시스템 오류나 컨테이너 크래시로 정책 잘못이 아닌데 낮은 reward를 받으면 advantage 추정에 노이즈가 낀다. 원인을 로그로 남겨 환경 붕괴로 인한 실패를 걸러내고, GRPO 그룹 크기가 절반 이상 남으면 유효 샘플을 복제해 크기를 복구하고, 아니면 그룹 전체를 버린다.

**MOPD**는 Kimi K3와 이름이 같을 뿐 아니라 설계 근거도 명시적으로 공유한다. 학생이 자기 궤적 위에서 라우팅된 교사에 대한 **outcome reward 없는 순수 KL** 목표로 학습한다는 점이 특징이다 — "$$\lambda$$로 균형 잡을 것도, reward-hacking 표면도 없다"고 서술한다. 이 설계는 "DeepSeek-AI(2026), NVIDIA(2026), Xiaomi LLM-Core Team(2026), Ma et al.(2026)"의 선행 on-policy-distillation 통합 레시피를 인용하며 자신의 차별점(전체 vocabulary 정확 계산, 검증가능 reward 재부착 없음)을 설명한다. 이 인용 목록 자체가, 최소 다섯 개 조직(DeepSeek·NVIDIA·Xiaomi·Kimi/Moonshot·Upstage)이 "전문가별로 따로 키운 뒤 증류로 합친다"는 같은 구조에 도달했다는 방증이다.

열두 개의 250B급 교사를 가속기에 동시에 올릴 수 없다는 순수 엔지니어링 문제도 구체적으로 다룬다 — GPU에는 학생과 교사 하나만 상주시키고, 나머지 열한 교사는 CPU에 파라미터 스냅샷으로 대기시키다가 라우팅된 교사를 그때그때 GPU 슬롯에 스왑해 넣는다. 마이크로배치를 라우팅된 교사별로 미리 묶어(pack) 두면 연속된 마이크로배치가 대개 같은(상주 중인) 교사를 맞혀 스왑 비용이 packed run 전체에 분산된다. 이런 인프라 디테일까지 공개한다는 점이, 앞서 "공개 수준의 편차" 절에서 Solar Open 2를 최상단에 놓은 이유다.

## GLM-4.5 — outcome supervision과 이진 포맷 게이트

GLM-4.5(Zhipu AI, 355B 파라미터 MoE, 32B 활성)는 이름 자체에 Agentic이 들어간다("Agentic, Reasoning, and Coding"). Agentic RL은 웹 검색과 코드 생성 에이전트에 집중한다 — "모든 행동이나 답이 자동으로 검증 가능한" 영역을 의도적으로 골랐다고 명시한다.

reward 설계는 이 글의 관심사인 credit 입도 관점에서 특히 단순하다. **"Outcome Supervision with Process Action Format Penalty"**라는 절 제목 그대로다.

- 웹 검색 태스크는 **최종 답의 정확도**를 궤적 전체의 reward로 준다.
- 코딩 에이전트는 검증 가능한 테스트 케이스가 있는 SWE 데이터를 주로 쓴다(다른 여러 모델과 마찬가지로 fail-to-pass 검증).
- **process action format penalty**: 모델이 에이전트 궤적 생성 중 올바른 도구 호출 포맷을 내지 못하면 그 자리에서 생성을 중단시키고 궤적 전체에 **reward 0**을 준다.

즉 credit 단위는 정확히 두 가지뿐이다 — "궤적 전체가 옳았나"(outcome)와 "궤적 도중 포맷이 깨졌나"(이진 게이트). 턴 단위나 스텝 단위로 내려가는 중간 신호는 리포트에 없다. 이 시리즈 [#4 결과만으로는 부족하다](/blog/2026/outcome-vs-process-agentic/)가 예측한 "긴 궤적을 outcome 하나로만 학습시키면 credit이 희석된다"는 문제에 대해, GLM-4.5가 실제로 취한 방어는 딱 하나 — 포맷이 깨지면 아예 처음부터 학습 대상에서 제외하는 것뿐이다.

컨텍스트 관리 관련해서 한 문장이 눈에 띈다. 정책 최적화 목적함수를 정의하며 "**모델이 생성한 토큰만 최적화에 쓰이고, 환경 피드백(도구 실행 결과)은 손실 계산에서 무시된다**"고 명시한다. 이것이 [#3 멀티턴 RL 실무 가이드](/blog/2026/multi-turn-rl-practice/)가 다룬 "환경 관측을 마스킹한다"는 원칙의 실전 사례다 — 에이전트가 만들지 않은 토큰(도구 응답)까지 정책 그래디언트에 태우면 안 된다는 것이다.

hacking 관련해서는 자기증류(iterative self-distillation)를 쓴다 — RL이 정체되면 RL로 학습된 모델의 응답으로 콜드스타트 데이터를 교체해 더 강한 SFT 모델을 만들고, 그 위에서 다시 RL을 돈다. 이 반복이 "RL 학습이 시간이 오래 걸린다"는 실용적 문제를 완화하는 장치로 서술되지, hacking 방어 장치로 명시되지는 않는다.

궤적 길이와 성공률의 관계에 대해서도 GLM-4.5는 드물게 정량적인 관찰을 남긴다. "Scaling Test-time Compute via Interaction Turns"라는 절에서, 추론 모델의 test-time scaling이 **출력 토큰 수**를 늘리는 것과 달리 agent 태스크는 **환경과 상호작용하는 턴 수**를 늘려 test-time compute를 쓴다고 구분하고, BrowseComp에서 browsing effort(상호작용 턴 수)를 늘릴수록 정확도가 매끄럽게(smoothly) 오른다고 보고한다. 이 관찰은 이 시리즈 [#4 결과만으로는 부족하다](/blog/2026/outcome-vs-process-agentic/)의 체크리스트 1번 — "궤적 길이와 성공률을 먼저 재라" — 에 가장 가까이 다가간 공개 수치이지만, 정작 "턴 수가 늘 때 credit이 어떻게 옅어지는가"는 다루지 않는다. 성공률 곡선은 공개해도 그 곡선 뒤의 학습 실패(그룹 붕괴) 데이터는 공개하지 않는 셈이다.

## DeepSeek-V4 — 리포트가 침묵하는 지점

DeepSeek-V4(DeepSeek-AI, 1.6T 파라미터 Pro / 284B Flash)는 리포트 전체가 25,489단어에 달하지만 **"reward"라는 단어가 단 7회**만 등장한다. 리포트의 중심은 million-token 컨텍스트를 위한 하이브리드 attention 아키텍처와 DSec 샌드박스 인프라이고, agentic reward 설계는 다음 두 문장으로 요약된다.

> "각 목표 도메인(수학·코딩·**agent**·지시 따르기)에 대해 별도의 전문가 모델을 독립적으로 학습시킨다. RL은 Group Relative Policy Optimization(GRPO)을 적용하며, **특정 성공 기준에 맞춘 reward model**의 안내를 받는다."

**"특정 성공 기준(specific success criteria)"**이 agent 도메인의 reward가 실제로 무엇을 재는지에 대한 유일한 서술이다. 도구 호출 정확성인지, 실행 성공인지, 궤적 길이인지 — 어느 쪽도 구체화되지 않는다. 이 모호함 자체가 이 글의 발견이다: DeepSeek-V4는 agent를 4대 도메인 전문가 중 하나로 **분명히 명시**하면서도, 그 전문가를 훈련시키는 reward 함수는 공개하지 않는다.

한 가지 흥미로운 구조적 정보는 있다. 검증 불가능한 태스크에는 **Generative Reward Model(GRM)**을 쓰되, "actor 네트워크가 GRM 역할을 겸한다"고 서술한다 — 판정 능력과 생성 능력이 하나의 모델 안에서 함께 RL로 최적화된다는 뜻이다. 이는 [#11](/blog/2026/agentic-judge-rubric/)이 다룬 judge 자기평가 아이디어의 한 극단적 형태지만, 이 GRM이 agentic 궤적 채점에도 쓰이는지는 리포트에 명시돼 있지 않다.

컨텍스트 관리는 상대적으로 구체적이다 — 이전 버전(V3.2)이 새 사용자 턴이 오면 추론 흔적을 버리던 것과 달리, V4는 사용자 메시지 경계를 넘어서도 **전체 추론 이력을 보존**해 도구 호출 시나리오에서 일관된 사고 사슬을 유지한다고 서술한다. 이 부분은 [#3](/blog/2026/multi-turn-rl-practice/)의 컨텍스트 관리 논의와 직접 연결되지만, RL 학습이 아니라 추론(inference) 시점의 설계로 서술돼 있다.

정리하면: 도메인 분류(agent를 별도 전문가로 둠), 알고리즘(GRPO→on-policy 증류), 추론 흔적 보존 정책은 구체적이다. 그런데 정작 이 글의 핵심 질문 — agent reward가 무엇을 재고, credit을 어떻게 나누는가 — 는 "성공 기준에 맞춘 reward"라는 한 구절 밖으로 나오지 않는다.

## Qwen3 — "Agentic"이라는 단어가 한 번도 안 나온다

Qwen3(Alibaba, 최대 235B-A22B MoE, 8개 밀집 모델 포함)의 리포트에서 흥미로운 사실 하나: **"agentic"이라는 형용사가 정확히 0회** 등장한다. "agent"(명사)는 22회 쓰이지만, 그중 절반 이상이 "Agent & Coding"이라는 벤치마크 카테고리 표 헤더 반복이다.

실제 학습 서술은 General RL 단계의 다섯 능력 중 하나로 "Agent Ability"가 한 문단 등장하는 것이 전부다.

> "Agent Ability: 이 능력은 모델이 지정된 인터페이스를 통해 도구를 정확히 호출하도록 학습시키는 것이다. RL rollout 동안 모델은 실제 환경 실행 피드백과 함께 완전한 멀티턴 상호작용 사이클을 수행할 수 있으며, 이를 통해 장기 지평 의사결정 태스크에서의 성능과 안정성을 향상시킨다."

이 한 문단이 agent 관련 RL 서술의 전부다. General RL 단계는 20개 이상의 세부 태스크를 아우르는 reward 시스템을 쓰고, 그 reward를 세 유형으로 나눈다 — 규칙 기반, reference 답이 있는 model-based, reference 없는 model-based(선호 기반 스칼라 RM). 그런데 이 세 유형 중 **Agent Ability에 정확히 무엇이 쓰이는지는 명시되지 않는다**. "환경 실행 피드백"이라는 표현으로 미루어 규칙 기반(도구 호출 성공/실패)일 가능성이 높지만, 리포트가 확언하지 않으므로 이 글에서도 단정하지 않는다.

작은 모델(0.6B\~30B-A3B)에는 아예 RL을 적용하지 않고 큰 모델(32B, 235B-A22B)에서 **off-policy 증류 → on-policy 증류**로 지식을 옮긴다. 리포트는 이 선택을 명시적으로 정당화한다 — "강한 teacher 모델로부터의 증류가 성능과 학습 효율 양쪽에서 RL을 확연히 능가한다." 이 문장은 이 글의 다른 발견(전문가→증류 구조로의 수렴)과 다른 방향을 가리킨다는 점에서 흥미롭다. Kimi K3·Solar Open 2·DeepSeek-V4는 "전문가별로 RL을 각각 돌린 뒤 증류"하지만, Qwen3의 작은 모델들은 "RL 자체를 건너뛰고 증류만" 쓴다 — 증류가 RL을 보완하는 게 아니라 대체하는 극단적 사례다.

## K-EXAONE 2.0 — 오프라인 preference로 옮겨간 agentic 판정

K-EXAONE 2.0(LG AI Research, 750B 파라미터 MoE, 37B 활성)은 mid-training 단계에서 agentic 데이터를 상당히 공들여 구축한다 — 일반 도구 사용(persona 기반 합성 MCP 서버·시나리오), agentic search(BrowseComp 스타일의 depth/breadth 지향 질의), SWE(fail-to-pass/pass-to-pass 검증) 세 갈래다. SWE 데이터 구축 파이프라인은 여러 프로그래밍 언어(Python·Go·JavaScript·TypeScript)의 저장소에서 LLM으로 Docker 환경을 만들고 **반복적 검증 루프**로 다듬은 뒤, 검증된 환경 안에서 이슈 설명과 패치를 생성하고 fail-to-pass·pass-to-pass 테스트 케이스를 추출한다. 마지막으로 이슈 설명과 패치가 정합적인지, oracle 실행으로 패치가 실제로 이슈를 해결하는지까지 확인한다 — Solar Open 2·GLM-4.5와 동일한 계열의 실행 기반 검증이다. 하지만 **이 데이터가 최종적으로 어떤 reward로 학습되는지는 온라인 RL이 아니라 오프라인 preference 단계에서 결정된다.**

Post-training은 SFT 다음에 두 단계 preference optimization을 둔다 — 범용 지시 따르기·추론 성능 개선(multi-task), 그다음 안전성(safety-aware). 두 단계 모두 **GrouPER**(Group-wise SimPER)라는 groupwise preference 목적함수로 학습한다. agentic 시나리오의 판정 기준은 다음 한 문장이다.

> "agentic 시나리오에서는 에이전트 행동과 응답의 **정확성**뿐 아니라, 최종 답의 **품질·깊이·포괄성**까지 평가해 더 효과적인 preference supervision을 이끌어낸다."

이 문장이 브리프가 미리 짚은 "행동의 정확성·품질로 판정한다"는 서술의 원문이다. 수학·코딩은 검증 가능한 신호로 chosen/rejected를 가르고, agentic·chat 태스크는 도메인별 기준으로 정의한 "바람직한 응답 선택 기준"과 "거부된 응답의 바람직하지 않은 패턴에 대한 페널티"로 reward를 설계한다고 서술하지만, 그 기준이 규칙인지 judge rubric인지는 명시되지 않는다. 길이 편향을 막으려 chosen/rejected 응답 길이를 비슷하게 맞춘다는 점은 명시적이다.

한 가지 확인이 필요한 대목: 초록에는 온라인 RL과 함께 **AGAPO**라는 자체 개발 알고리즘이 GrouPER와 나란히 언급된다. 그런데 본문을 끝까지 훑어도 AGAPO에 대한 추가 설명은 없다 — 인용 [2]는 이전 세대 리포트(EXAONE 4.0)를 가리킨다. 즉 AGAPO가 agentic RL에 쓰였는지, 쓰였다면 어떤 형태인지는 **이 리포트만으로는 확인할 수 없다**. 공개 자료에 없는 것으로 남겨둔다.

credit 단위는 pairwise 응답 그룹(GrouPER가 요구하는 candidate group) 단위이므로, 여기서도 trajectory 레벨을 벗어나지 않는다. 다만 온라인 RL(경쟁 모델 대부분이 쓰는 방식) 대신 **오프라인 preference로 agentic 판정을 옮겼다**는 점 자체가 이 비교에서 K-EXAONE 2.0만의 선택이다.

## MiniMax-M1 — CISPO의 원산지, 그러나 agentic 전용 트랙은 없다

MiniMax-M1은 이 시리즈의 필수 조사 대상은 아니지만, A.X K2가 채택한 **CISPO**의 원 논문이라 짚어둘 가치가 있다. CISPO는 token 단위 클리핑 대신 importance-sampling 가중치를 클립해, "However"·"재검토" 같은 확률은 낮지만 추론에 결정적인 토큰이 그래디언트에서 잘려나가지 않게 한다.

agentic 관련해서는 두 갈래로 갈린다. **소프트웨어 엔지니어링**은 실행 기반 verifiable reward를 쓴다 — 컨테이너 샌드박스에서 실제 코드를 실행해 사전 정의되거나 새로 생성된 테스트 케이스의 통과 여부로 채점하며, 성공은 양의 reward, 컴파일 오류·런타임 실패·회귀는 0 또는 음의 reward를 받는다. 반면 **일반 도메인**은 규칙으로 검증하기 어려운 2만 5천 개 샘플을 reward model 기반 feedback으로 학습한다. 그런데 이 두 갈래 중 어느 쪽에도 **agentic tool use를 위한 전용 RL 트랙은 없다** — TAU-bench는 평가 벤치마크로만 등장하고, 학습 데이터 구성은 서술되지 않는다. "reward" 자체는 34회로 아홉 모델 중 밀도(문서 1천 단어당)가 가장 높은 편이지만, 그 밀도의 대부분은 수학·코드 RLVR 서술에서 나온다.

## Llama 4 — agentic RL이 사실상 등장하지 않는다

Llama 4(Meta) 공식 블로그(2025년 4월, 12분 분량 글) 전체를 grep한 결과는 다음과 같다. **"agentic" 0회, "trajectory" 0회, "reward" 0회, "credit" 0회.** "agent"는 정확히 1회 등장하는데, 이마저도 학습된 모델의 agentic 능력이 아니라 red-teaming 자동화 도구인 **GOAT(Generative Offensive Agent Testing)**를 가리킨다 — 즉 "에이전트"라는 단어가 나오는 유일한 자리가 공격 시뮬레이션 도구 이름이다. "tool"은 6회 나오지만 전부 Llama Guard·Prompt Guard 같은 **제3자 안전 도구**를 가리키며, 모델 자신의 tool-calling 학습과는 무관하다.

post-training 서술은 다음이 전부다 — "**lightweight SFT → online RL → lightweight DPO**"라는 파이프라인, 쉬운 데이터의 50% 이상을 걷어낸 난이도 필터링, 어려운 프롬프트로 계속 필터링하며 지속적으로 도는 continuous online RL 전략. 이 online RL이 정확히 어떤 reward로 무엇을 최적화하는지, agentic·tool-use 태스크가 이 파이프라인 어디에 들어가는지는 서술되지 않는다. Behemoth(2T 파라미터 teacher) 절에서도 "hard prompt를 pass@k로 선별해 커리큘럼을 구성했다"는 일반적 RL 서술만 있을 뿐, agentic 특화 내용은 없다.

이 글의 조사 대상 아홉 모델 가운데 agentic RL에 대해 가장 침묵하는 자료다. **공개 자료에 없음.**

## Gemma 4 — 참고문헌에만 등장하는 "agentic"

Gemma 4(Google DeepMind) arXiv 리포트를 grep하면 얼핏 "agentic" 3회, "agent" 5회로 완전한 0은 아니다. 그런데 이 등장 위치를 확인하면 이야기가 달라진다 — **세 번의 "agentic" 등장 모두 참고문헌 목록 안**, 즉 Gemini 2.5·Kimi K2.5·GLM-5 같은 **다른 논문의 제목을 인용하는 자리**에서만 나온다. 본문(참고문헌 이전, 전체 문서의 절반가량)에는 "agentic"이 단 한 번도 등장하지 않는다. "agent"도 마찬가지로 전부 참고문헌의 τ²-bench·Terminal-bench 인용에서만 나온다.

본문에서 확인 가능한 agentic 관련 유일한 흔적은 채팅 템플릿 부록이다 — 함수 선언과 함수 호출을 감싸는 전용 특수 토큰(tool 선언용, tool_call 호출용) 포맷 예시가 표로 제공된다. 즉 Gemma 4가 tool-calling 인터페이스를 지원한다는 사실은 확인되지만, **그 인터페이스를 어떤 reward로, 어떤 RL 절차로 학습시켰는지는 리포트 어디에도 서술돼 있지 않다.**

이 결과는 RLHF 시리즈 #44가 짚은 "Gemma의 실제 RL 레시피(BOND·WARM·WARP)가 helpfulness reward 축에서는 비교적 상세히 공개됐다"는 관찰과 대조적이다. 같은 리포트가 helpfulness 축에서는 말이 많고 agentic 축에서는 완전히 침묵한다 — **공개의 밀도는 축마다 다르다**는 이 글의 핵심 관찰을 가장 극명하게 보여주는 사례다. **공개 자료에 없음.**

# Experiments

## 수렴 — 세 팀 이상이 같은 선택을 한 지점

**1. 전문가별 학습 후 증류.** Kimi K3(9전문가→MOPD), Solar Open 2(12전문가→MOPD, 이름까지 동일), DeepSeek-V4(4도메인 전문가→on-policy 증류, 역-KL)가 같은 구조에 도달했다. Solar Open 2가 인용한 선행연구 목록(DeepSeek-AI·NVIDIA·Xiaomi·자기 자신)을 보면 이 구조가 최소 다섯 조직에서 독립적으로(혹은 서로 참조하며) 수렴했다는 정황이 있다. agent 도메인은 이 "전문가 풀" 안에서 거의 항상 하나의 독립된 전문가로 취급된다(Kimi K3의 general/coding agents, Solar Open 2의 "agents & tools" 계열, DeepSeek-V4의 agent 전문가).

**2. SWE는 fail-to-pass/pass-to-pass로 검증한다.** GLM-4.5·MiniMax-M1·Solar Open 2·Kimi K3(샌드박스) 모두 코드 에이전트의 핵심 reward를 **테스트 실행 성공 여부**로 삼는다. 이 시리즈 [#13 코드 에이전트](/blog/2026/swe-agent-rl/)가 다룬 "검증 가능한 채점"의 표준 구현이 정확히 이 형태다 — 정답 패치 적용 전 실패, 적용 후 통과.

**3. 판정형 reward(judge·rubric)가 검증 불가능 영역의 표준이 됐다.** Kimi K3(Agentic GRM), A.X K2(게이트형 pointwise judge), Solar Open 2(graded process rubric + LLM coherence judge), K-EXAONE 2.0(agentic 응답의 정확성+깊이 judge)까지 넷이 명시적으로 judge·rubric을 쓴다. 규칙만으로 agentic reward를 완결하는 모델은 없다.

**4. verbosity/포맷 hacking에 명시적 방어를 둔다.** Kimi K3(budget 기반 verbosity 제어, 토큰 예산 초과 시 reward −1), A.X K2(anti-verbosity judge 안전장치 + group-relative length penalty), GLM-4.5(포맷 위반 시 reward 0 이진 게이트), Solar Open 2(malformed-tool-call 필터링) — 넷 모두 "판정 지점 아래는 추정이고 hacking된다"([#8](/blog/2026/reward-shaping-agentic/))는 문제에 대해 **사후 필터/페널티**로 대응한다는 공통점이 있다. 사전에 판정 자체를 더 세밀하게 만드는 방식(예: 턴 단위 credit)으로 대응한 사례는 없다.

## 분기 — 갈라진 지점

**1. credit 입도가 이론보다 훨씬 거칠다.** 이 시리즈 2부가 상세히 다룬 턴·스텝·토큰 단위 credit assignment는 아홉 모델의 **주 reward**로는 어디에도 쓰이지 않는다. Kimi K3의 per-token 신호는 MOPD 증류 목적함수이지 원본 task reward가 아니다. 유일하게 trajectory보다 세밀해지는 곳은 Solar Open 2의 rubric-기준 단위 채점이다. 이론과 실전 사이의 이 격차는 이 글에서 가장 뚜렷한 발견이다 — 아래 "credit assignment" 키워드 카운트가 이를 숫자로 확인해준다.

왜 이 격차가 생기는지, Method에서 확인한 사실만으로 추론해보면 셋으로 정리된다.

- **검증 비용이 턴 수만큼 곱해진다.** trajectory 하나를 한 번 채점하는 대신 턴마다 채점하려면, 환경 검증(테스트 실행)이든 judge 호출이든 비용이 턴 수에 비례해 늘어난다. Kimi K3가 5,100만 개의 샌드박스를 쓰고도 Agentic GRM은 여전히 후보 궤적 전체를 한 번에 비교하는 이유, Solar Open 2가 완전 비동기 인프라까지 새로 지어야 했던 이유가 여기 있다 — trajectory 레벨 채점조차 인프라를 갈아엎어야 할 만큼 비싸다.
- **중간 상태를 채점할 명확한 기준이 도메인마다 다르다.** SWE는 fail-to-pass 테스트라는 명확한 종료 조건이 있어 턴 중간에도 "지금까지 옳은 방향인가"를 판단하기 쉽지만, 일반 대화형 에이전트나 오피스워크처럼 "무엇이 옳은 중간 상태인가"가 불분명한 도메인에서는 턴 단위 채점 자체가 추가 judge 호출 — 즉 새로운 hacking 표면 — 을 하나 더 만드는 셈이다.
- **judge에게 부분 궤적을 판정하게 하면 judge가 더 쉽게 뚫린다.** [#11](/blog/2026/agentic-judge-rubric/)이 다룬 judge 기반 채점은 완결된 결과물을 보고 판단할 때 가장 신뢰도가 높다. 미완성 궤적 중간 지점을 채점하게 하면 판단 근거 자체가 줄어들어, A.X K2가 굳이 게이트("완결되고 실행 가능한 transcript인가")를 pointwise judge 앞에 세운 이유와 같은 논리로, 신뢰할 만한 부분 판정이 오히려 더 어렵다.

세 이유 모두 "이론이 게을러서가 아니라 실전의 비용·신뢰도 제약이 이론을 못 따라간다"는 쪽을 가리킨다. 이 시리즈 [#6](/blog/2026/step-level-credit/)\~[#7](/blog/2026/token-segment-credit/)이 제안한 스텝·토큰 단위 기법들이 대부분 연구 벤치마크(ALFWorld, WebShop 같은 통제된 환경) 위에서 검증됐다는 점과 겹쳐 보면, 이 기법들이 3T급 모델의 프로덕션 파이프라인에 아직 이식되지 않았다는 뜻으로 읽을 수 있다.

**2. agent를 독립 도메인으로 두느냐, 일반 RL의 한 태스크로 묻느냐.** Kimi K3·Solar Open 2·A.X K2·DeepSeek-V4는 agent를 **독립된 전문가/그룹**으로 분리한다. 반면 Qwen3는 "Agent Ability"를 General RL 단계의 20여 개 태스크 중 하나로, K-EXAONE 2.0은 온라인 RL이 아니라 오프라인 preference 단계에 agentic 판정을 접어 넣는다. 전자는 agent 전용 reward·커리큘럼을 설계할 여지가 크고, 후자는 다른 능력과 자원을 공유한다.

**3. 온라인 RL vs 오프라인 preference.** 여덟 모델(Llama 4 제외) 중 K-EXAONE 2.0만 agentic 판정을 **오프라인 preference optimization**(GrouPER)에 맡긴다. 나머지는 온라인 RL rollout 중에 agentic reward를 준다. 이 선택이 credit 입도에도 영향을 준다 — 오프라인 preference는 애초에 완결된 응답 쌍을 비교하는 구조라 trajectory 레벨보다 세밀해지기 구조적으로 어렵다.

## 공개 수준의 편차

문서 종류로 나누면 이렇다.

| 문서 종류              | 모델                           | agentic RL 서술 밀도                                          |
| ---------------------- | ------------------------------ | ------------------------------------------------------------- |
| GitHub PDF tech report | Kimi K3, A.X K2                | 높음 — 전용 절, 구체 메커니즘, 정량 수치(5,100만 샌드박스 등) |
| arXiv 프리프린트(상세) | Solar Open 2, GLM-4.5          | 높음 — 전용 절, 환경 검증 파이프라인 상세                     |
| arXiv 프리프린트(중간) | K-EXAONE 2.0                   | 중간 — 문단 단위 서술, 핵심 알고리즘(AGAPO) 세부는 비공개     |
| arXiv 프리프린트(얕음) | DeepSeek-V4, Qwen3, MiniMax-M1 | 낮음 — 한두 문단, 구체 reward 함수 비공개                     |
| arXiv 프리프린트(부재) | Gemma 4                        | 없음 — 본문 0회, 참고문헌에만 흔적                            |
| 공식 블로그            | Llama 4                        | 없음 — "online RL" 한 구절 외 전무                            |

이 표에서 눈에 띄는 점: **문서 종류가 곧 공개 수준을 결정하지 않는다.** Gemma 4와 DeepSeek-V4·Qwen3·MiniMax-M1은 전부 arXiv 프리프린트지만 공개 밀도는 하늘과 땅 차이다. tech report를 arXiv 대신 GitHub에 올린 Kimi K3·A.X K2가 오히려 arXiv에 올라간 여러 모델보다 훨씬 상세하다. 즉 "어디에 공개했나"보다 "무엇을 공개하기로 했나"라는 팀의 선택이 더 크게 작용한다.

이 편차의 동기까지는 공개 문서만으로 단정할 수 없지만, 정황은 남겨둘 만하다. Kimi K3·DeepSeek-V4·GLM-4.5·Solar Open 2·A.X K2·K-EXAONE 2.0은 오픈소스 모델 가중치를 배포하는 팀이고, 상세 리포트는 연구 커뮤니티의 채택·2차 연구를 유도하는 문서로 기능한다 — 특히 Solar Open 2·A.X K2·K-EXAONE 2.0 세 한국 팀은 자국어 벤치마크(Ko-GDPval·한국어 평가군)까지 새로 만들어 공개할 만큼 투명성에 무게를 둔다. 반대로 Llama 4·Gemma 4는 이미 폭넓게 채택된 모델 계열의 신버전이라, 블로그·리포트가 연구 커뮤니티보다 일반 개발자·제품 팀을 향한 요약본에 가깝다 — 벤치마크 표는 상세해도 설계 서술은 짧다. 이 구분이 정확히 "agentic RL 공개 여부"와 겹친다는 점은, 공개 수준이 능력의 함수가 아니라 **문서가 누구를 향해 쓰였는가**의 함수라는 이 절의 결론을 한 번 더 뒷받침한다.

# 통계 요약

아홉 모델(+MiniMax-M1)의 리포트 본문에서 다섯 단어의 등장 횟수를 셌다. 괄호 안은 문서 1,000단어당 밀도다.

| 모델         | 본문 길이(단어) | agentic          | agent            | tool     | reward   | credit assignment |
| ------------ | --------------- | ---------------- | ---------------- | -------- | -------- | ----------------- |
| Kimi K3      | 28,771          | 50 (1.7)         | 149 (5.2)        | 70 (2.4) | 21 (0.7) | 0                 |
| A.X K2       | 19,697          | 34 (1.7)         | 50 (2.5)         | 17 (0.9) | 23 (1.2) | 0                 |
| GLM-4.5      | 12,612          | 41 (3.3)         | 90 (7.1)         | 42 (3.3) | 30 (2.4) | 0                 |
| DeepSeek-V4  | 25,489          | 29 (1.1)         | 61 (2.4)         | 50 (2.0) | 7 (0.3)  | 0                 |
| Qwen3        | 17,701          | 0 (0.0)          | 22 (1.2)         | 5 (0.3)  | 12 (0.7) | 0                 |
| K-EXAONE 2.0 | 17,621          | 39 (2.2)         | 66 (3.7)         | 33 (1.9) | 3 (0.2)  | 0                 |
| Solar Open 2 | 14,843          | 13 (0.9)         | 82 (5.5)         | 18 (1.2) | 13 (0.9) | 0                 |
| Llama 4      | 3,555           | 0 (0.0)          | 1 (0.3)          | 6 (1.7)  | 0 (0.0)  | 0                 |
| Gemma 4      | 7,337           | 3(전부 참고문헌) | 5(전부 참고문헌) | 10       | 0        | 0                 |
| MiniMax-M1   | 10,846          | 7 (0.6)          | 20 (1.8)         | 19 (1.8) | 34 (3.1) | 0                 |

가장 중요한 열은 마지막 열이다. **"credit assignment"는 열 개 문서 전부에서 0회다.** 이 시리즈 2부 전체(5편, [#4](/blog/2026/outcome-vs-process-agentic/)\~[#8](/blog/2026/reward-shaping-agentic/))와 3부 전체(3편, [#9](/blog/2026/environment-as-reward/)\~[#11](/blog/2026/agentic-judge-rubric/))가 다룬 핵심 용어가, 그 문제를 실제로 풀고 있을 프론티어 팀들의 공개 문서 어디에도 등장하지 않는다. 이 부재를 두 가지로 해석할 수 있다 — (a) 팀들이 이 문제를 인식하되 다른 용어(예: "outcome supervision", "process rubric", "budget-based penalty")로 표현하거나, (b) 실제로 trajectory 레벨보다 세밀한 credit 설계를 하지 않고 있거나. Method에서 확인했듯 실제 메커니즘을 보면 (b)에 가깝다 — 세밀한 credit assignment 알고리즘(GAE류 turn-level advantage, per-step reward shaping 등)을 명시적으로 채택했다고 서술한 모델은 없다.

"agentic" 밀도가 가장 높은 GLM-4.5(3.3)와 가장 낮은 Qwen3·Llama 4(0.0)의 격차는 단순한 우연이 아니다 — GLM-4.5는 모델명 자체에 Agentic을 넣을 만큼 이 능력을 정체성으로 내세우고, Qwen3·Llama 4는 agent를 여러 능력 중 하나로만 다룬다. 다만 이 밀도가 실제 설계 깊이와 항상 일치하지는 않는다 — Solar Open 2는 "agent"라는 단어를 압도적으로 많이 쓰면서도(5.5) "agentic"이라는 형용사형은 상대적으로 적게 쓴다(0.9). 리포트가 "Agent Scenarios"라는 절 제목을 쓰기 때문인데, 이는 이 글이 앞서 짚은 함정을 스스로 보여준다 — **단어 하나의 빈도만으로 설계의 깊이를 판단하면 안 된다.** 그래서 이 글은 밀도 표와 함께 반드시 원문 인용·구체 메커니즘을 나란히 제시했다.

# Conclusion

## 핵심 메시지

아홉 개 프론티어 모델의 공개 자료를 가로지른 결과는 셋으로 요약된다.

1. **"전문가별 RL 후 증류"가 지배적 구조로 자리 잡았다.** 최소 다섯 조직이 이 구조에 도달했고, 그중 둘(Kimi K3·Solar Open 2)은 MOPD라는 이름까지 같다. agent는 거의 항상 이 전문가 풀 안의 독립된 한 축이다.
2. **credit 입도는 이론이 준비한 만큼 세밀해지지 않았다.** trajectory·outcome 레벨이 사실상의 표준이고, 유일한 예외(Solar Open 2의 rubric-기준 단위)조차 turn-level·token-level에는 못 미친다. "credit assignment"라는 용어 자체가 조사한 모든 문서에서 0회 등장한다는 사실이 이를 숫자로 뒷받침한다.
3. **공개 수준의 편차가 실제 능력 격차보다 훨씬 크다.** Llama 4·Gemma 4는 벤치마크 점수는 공개하면서 그 점수를 만든 reward 설계는 사실상 감춘다. 반대로 Solar Open 2·Kimi K3·A.X K2는 환경 검증 메커니즘과 인프라 수치까지 상세히 공개한다.

## 시리즈 전체 체크리스트

16편을 하나의 설계 체크리스트로 압축하면 이렇다. 이번 글에서 실제로 확인된 사례를 항목마다 하나씩 붙인다.

1. **궤적 길이와 성공률을 먼저 재라 — 그룹 붕괴가 얼마나 일어나는가.** ([#4](/blog/2026/outcome-vs-process-agentic/)) — GLM-4.5는 상호작용 턴 수를 늘릴수록 정확도가 매끄럽게 오른다고 보고하고, Kimi K3는 RL FLOPs가 늘수록 평균 스텝 수가 늘어나며 능력이 함께 개선된다고 보고한다. 다만 "그룹이 얼마나 붕괴했는가"를 직접 수치로 공개한 모델은 없다.
2. **환경이 진짜 신호를 주는 가장 세밀한 지점이 어디인가.** ([#9](/blog/2026/environment-as-reward/)) — SWE의 fail-to-pass/pass-to-pass 테스트, Solar Open 2의 mutation-후-read-back 검증기가 가장 세밀한 사례다.
3. **그 지점까지만 입도를 내려라.** ([#5](/blog/2026/turn-level-reward/)\~[#7](/blog/2026/token-segment-credit/)) — 실전에서는 이 조언조차 "그 지점"에 못 미친다. 대부분 trajectory 레벨에 머물고 Solar Open 2의 rubric-기준만 한 단계 더 내려간다.
4. **그 아래는 추정이고, 추정은 hacking된다.** ([#8](/blog/2026/reward-shaping-agentic/)) — Kimi K3의 Agentic GRM verbosity 제어, A.X K2의 anti-verbosity 안전장치가 정확히 이 문제(judge의 추정을 궤적이 악용하는 것)에 대한 사후 대응이다.
5. **판정 정의가 대리 지표인지 확인하라.** ([#15](/blog/2026/agentic-reward-hacking/)) — A.X K2의 게이트형 judge(구조적 유효성을 먼저 확인한 뒤에만 judge에 넘김)가 "유창함"이라는 대리 지표에 낚이지 않으려는 명시적 설계다.
6. **여러 보상을 어떻게 결합했는지 명시하라.** ([#15](/blog/2026/agentic-reward-hacking/)) — A.X K2의 GDPO(reward별 분리 정규화 후 결합), Solar Open 2의 규칙 체커+judge rubric 병행이 명시적 사례다. DeepSeek-V4·Qwen3는 이 결합 방식을 구체화하지 않는다.
7. **환경 관측 마스킹, 컨텍스트 관리.** ([#3](/blog/2026/multi-turn-rl-practice/)) — GLM-4.5의 "환경 피드백은 손실 계산에서 무시된다"는 문장이 가장 명시적인 사례다. DeepSeek-V4·K-EXAONE 2.0은 추론 흔적을 멀티턴 경계 너머로 보존하는 정책을 서술한다.

## 한계

이 글의 방법론적 한계를 정직하게 남긴다. 첫째, 키워드 카운트는 리포트가 특정 표현을 얼마나 자주 쓰는지를 잴 뿐, 실제 코드나 실험에서 무엇을 했는지는 알 수 없다 — 공개하지 않은 것과 하지 않은 것을 완전히 구분할 수 없다. 둘째, 리포트 간 길이 격차(3,555\~28,771단어)가 커서 원시 등장 횟수만으로 비교하면 왜곡이 생긴다(밀도로 보정했지만 밀도 역시 완벽하지 않다). 셋째, 이 글이 참조한 자료는 모두 각 팀이 스스로 공개한 문서다 — 실제 학습 파이프라인에는 이 문서들이 다루지 않는 추가 설계가 있을 수 있다. 그럼에도 이 세 한계를 감안하고도, "credit assignment 0회"·"Llama 4·Gemma 4의 agentic 서술 부재" 같은 발견은 문서 자체의 사실이므로 그대로 남겨둔다.

# 참고 문헌

- Kimi Team, 2026. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3) (Moonshot AI, Tech Report). — Agentic GRM, MOPD, AgentENV 샌드박스, reasoning-effort budget RL.
- SK Telecom, 2026. [A.X K2 Technical Report](https://github.com/SKT-AI/A.X-K2) (Tech Report). — 4그룹 mixture, 게이트형 judge, CISPO+GDPO.
- Park, Sungrae et al., 2026. [Solar Open 2 Technical Report](https://arxiv.org/abs/2607.20062) (Upstage, arXiv). — 12전문가 MOPD, 환경 검증기, graded process rubric.
- GLM-4.5 Team (Zeng, Aohan et al.), 2025. [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) (Zhipu AI & Tsinghua University, arXiv).
- DeepSeek-AI, 2026. [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) (arXiv).
- Yang, An et al., 2025. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Alibaba, arXiv).
- Choi, Eunbi et al., 2026. [K-EXAONE 2.0 Technical Report](https://arxiv.org/abs/2608.04505) (LG AI Research, arXiv).
- Meta AI, 2025. [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (공식 블로그).
- Google DeepMind, 2026. [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) (arXiv).
- MiniMax, 2025. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585) (arXiv). — CISPO 원 논문.
- Liu, Shih-Yang et al., 2026. [GDPO: Group Reward-Decoupled Normalization Policy Optimization for Multi-Reward RL Optimization](https://arxiv.org/abs/2601.05242) (arXiv). — A.X K2가 채택한 멀티리워드 정규화.
- Choi, Eunbi et al., 2026. EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes (LG AI Research). — K-EXAONE 2.0이 AGAPO의 출처로 인용하는 이전 세대 리포트(arXiv 번호 미확인, K-EXAONE 2.0 참고문헌 [2]).
- [RLHF Reward 설계 시리즈 #44 프론티어 편](/blog/2026/frontier-reward-design/) — 같은 아홉 모델(+Magistral)을 helpfulness reward 4분류(규칙·스칼라 RM·reference judge·GRM) 렌즈로 비교.
- [RLHF Reward 설계 시리즈 #45 프론티어 안전성 편](/blog/2026/frontier-safety-design/) — 같은 자료를 harmlessness 축으로, "부재도 발견"이라는 이 글의 방법론적 선례.

---

# Agentic RL 설계 시리즈

이 글은 Agentic RL 설계 시리즈의 열여섯 번째 글이다.

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
  <li><a href="/blog/2026/web-gui-agent-rl/">웹·GUI 에이전트</a> — end-to-end 멀티턴 RL</li>
</ol>

**5부. 실패와 방어**

<ol start="15">
  <li><a href="/blog/2026/agentic-reward-hacking/">에이전트의 reward hacking</a> — 판정기가 뚫린다, 그리고 조합의 실패</li>
</ol>

**6부. 실전 종합**

<ol start="16">
  <li><strong>(현재 글)</strong> 프론티어 모델은 실제로 어떻게 하나 — 최신 모델들의 agentic RL 설계</li>
</ol>

본 시리즈는 16편으로 구성된다.
