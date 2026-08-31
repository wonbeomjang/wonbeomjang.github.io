---
layout: post
title: "도구 호출을 어떻게 채점하나 — ToolRL·ToolRM"
date: 2026-08-11 09:53:00 +0900
description: "RL Reward 설계 시리즈 #53 — 도구 호출의 형식·선택·파라미터를 규칙으로, 필요성·결과 활용을 judge로 채점하는 두 축을 ToolRL의 reward ablation과 ToolRM의 outcome reward model로 뜯어본다"
categories: [paper]
tags: [agentic-rl, tool-use, reward-design, grpo, reward-model, paper]
giscus_comments: true
related_posts: true
---

> [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) (Qian et al., University of Illinois Urbana-Champaign, arXiv 2025)
>
> [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](https://arxiv.org/abs/2509.11963) (Agarwal et al., IBM Research, arXiv 2025)

# Introduction

[#52](/blog/2026/environment-as-reward/)에서는 "환경에게 묻는다"를 다뤘다. 코드를 실행하면 테스트가 통과하는지, 검색 결과가 정답을 담고 있는지 — 환경이 실제로 답을 알고 있을 때, reward는 그 답과 대조해서 나온다. 그런데 도구 호출은 실행되기 _전에도_ 채점할 거리가 있다. `search(query="")` 처럼 파라미터가 비어 있는 호출, 계산기가 필요한데 검색을 부르는 호출, 형식 자체가 깨져서 파서가 못 읽는 호출 — 이런 건 환경이 답을 주기도 전에 이미 "잘못된 호출"이다.

이 편은 그 지점을 다룬다. **도구 호출 그 자체를 어떻게 채점하는가.** 결론부터 정리하면 두 논문이 서로 다른 절반을 채운다.

- **ToolRL**(Qian et al., 2025)은 도구 호출의 reward를 **형식(format)**과 **정확성(correctness)**으로 분해하고, 정확성을 다시 도구 이름·파라미터 이름·파라미터 값 세 항으로 더 잘게 쪼갠다. 이 분해 자체가 논문의 기여이며, 어떤 조합이 학습을 살리고 어떤 조합이 죽이는지를 방대한 ablation으로 보여준다. GRPO에 얹어 base 모델 대비 17%, SFT 모델 대비 15% 개선을 낸다.
- **ToolRM**(Agarwal et al., 2025)은 그 규칙 기반 채점이 안 통하는 지점 — 정답 도구 호출이 없을 때, 혹은 "이 호출이 그럴듯한가"를 판단해야 할 때 — 을 메우는 **학습된 scalar reward model**이다. FC-RewardBench라는 전용 벤치마크로 "일반 RM은 도구 호출을 못 채점한다"는 것부터 증명하고, 정답 없이도 GRPO 보상으로 쓸 수 있는 ToolRM을 학습시킨다.

한쪽은 "규칙으로 얼마나 잘게 쪼갤 수 있는가", 다른 쪽은 "규칙이 안 통할 때 무엇으로 대체하는가"다. 두 논문을 나란히 보면 도구 호출 채점의 전체 지형이 보인다.

# Background — 도구 호출의 채점 축

도구 호출 하나 — 예를 들어 `{"name": "search_restaurant", "parameters": {"location": "여의도", "price_max": 30000}}` — 를 채점할 때 실제로 물을 수 있는 질문은 다섯 가지다.

| 축        | 무엇을 보나               | 어떻게 채점하나                   | 실패 예                                                        |
| --------- | ------------------------- | --------------------------------- | -------------------------------------------------------------- |
| 형식      | JSON 스키마·문법을 지켰나 | 규칙(파서)                        | `<tool_call>` 태그가 안 닫히거나 JSON이 깨져 파싱 실패         |
| 도구 선택 | 옳은 도구를 골랐나        | 규칙(정답 도구와 대조) 또는 judge | 검색이 필요한 질문에 계산기를 호출                             |
| 파라미터  | 인자가 맞나               | 규칙 또는 judge                   | 필수 필드 누락, `price_max`에 문자열을 넣는 타입 오류          |
| 필요성    | 애초에 호출이 필요했나    | judge                             | 이미 답을 아는 질문에도 검색을 호출하는 **불필요한 호출 남발** |
| 결과 활용 | 받은 결과를 제대로 썼나   | judge                             | 예약 가능 시간을 조회해놓고 결과를 무시한 채 답을 지어냄       |

표의 위 세 줄과 아래 두 줄 사이에 선을 하나 그을 수 있다. **형식·도구 선택·파라미터는 정답(ground truth)이 있으면 규칙만으로 채점된다** — 문자열이 일치하는지, 딕셔너리 키가 겹치는지 세면 끝난다. 반면 **필요성과 결과 활용은 정답 하나로 환원되지 않는다.** "이 호출이 꼭 필요했나"는 대화 맥락 전체를 봐야 판단할 수 있고, "결과를 제대로 썼나"는 최종 응답의 의미를 읽어야 한다. 둘 다 판정(judgment)이 필요하다.

이건 이 블로그의 RL Reward 설계 시리즈에서 다룬 [reward 4분류](/blog/2026/reward-model-design/) — ①규칙 ②스칼라 RM ③reference judge ④생성형 RM(GRM) — 이 하나의 도구 호출 채점 안에서 그대로 재현되는 지점이다. 형식·선택·파라미터는 ①로 끝나지만, 필요성·결과 활용은 ③이나 ④로 넘어가야 한다. 이 글에서 볼 ToolRL은 ①에 해당하는 세 축(정확히는 형식+정확성)만 정교하게 다듬은 사례고, ToolRM은 정답이 없는 상황에서 ①을 대신할 ②(도메인 특화 scalar RM)를 학습시킨 사례다. 다섯 축을 다 커버하는 judge형 채점은 다음 편([#54](/blog/2026/agentic-judge-rubric/))에서 다룬다.

# Method

## ToolRL — 도구 호출 reward를 체계적으로 분해한다

ToolRL은 Tool-Integrated Reasoning(TIR) — LLM이 여러 스텝에 걸쳐 도구를 호출하며 문제를 푸는 과정 — 을 위한 reward 설계를 **처음으로 체계적으로 연구한** 논문이다. 논문 제목 그대로 저자들의 결론은 "reward is all tool learning needs"다. SFT로 도구 호출을 가르치면 특정 패턴에 과적합되어 낯선 도구 조합에 약한데, 잘 설계된 reward로 GRPO를 돌리면 이 문제가 상당 부분 풀린다는 것이다.

### reward를 두 층으로 나눈다

ToolRL의 최종 reward는 형식 reward와 정확성 reward의 합이다.

$$\mathcal{R}_{\text{final}} = \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{correct}}$$

**형식 reward**는 이진값이다. 모델 출력이 `<think>`, `<tool_call>`, `<response>` 같은 필수 태그를 정답이 요구하는 순서대로 전부 포함하면 1, 아니면 0이다.

$$\mathcal{R}_{\text{format}} = \begin{cases} 1, & \text{필수 필드가 모두 있고 순서도 맞으면} \\ 0, & \text{그 외} \end{cases}$$

**정확성 reward**가 이 논문의 핵심이다. 예측된 도구 호출 집합 $$P = \{P_1, \ldots, P_m\}$$을 정답 집합 $$G = \{G_1, \ldots, G_n\}$$과 비교하는데, 이걸 하나의 이진 판정으로 뭉개지 않고 세 항으로 쪼갠다.

- **도구 이름 일치도** — 정답 도구 이름 집합 $$N_G$$와 예측 도구 이름 집합 $$N_P$$의 자카드 유사도다.

$$r_{\text{name}} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0, 1]$$

- **파라미터 이름 일치도** — 각 정답 호출 $$G_j$$에 대해, 예측이 같은 키(파라미터 이름) 집합을 얼마나 맞췄는지를 자카드 유사도로 합산한다.

$$r_{\text{param}} = \sum_{G_j \in G} \frac{|\text{keys}(P_G) \cap \text{keys}(P_P)|}{|\text{keys}(P_G) \cup \text{keys}(P_P)|} \in [0, |G|]$$

- **파라미터 값 일치도** — 정답의 각 파라미터 키마다 예측 값이 완전히 같은지를 세어 합산한다.

$$r_{\text{value}} = \sum_{G_j \in G} \sum_{k \in \text{keys}(G_j)} \mathbb{1}[P_G[k] = P_P[k]] \in \left[0, \sum_{G_j \in G} |\text{keys}(G_j)|\right]$$

세 항을 더한 매치 점수 $$r_{\text{match}} = r_{\text{name}} + r_{\text{param}} + r_{\text{value}}$$를 예측-정답 사이 최적 매칭에 대해 최대화한 값이 $$R_{\max}$$이고, 이론적 최댓값은 $$S_{\max} = 1 + \lvert G \rvert + \sum_{G_j \in G} \lvert \text{keys}(G_j) \rvert$$다. 최종 정확성 reward는 이를 $$[-3, 3]$$으로 정규화한다.

$$\mathcal{R}_{\text{correct}} = 6 \cdot \frac{R_{\max}}{S_{\max}} - 3 \in [-3, 3]$$

기호를 풀면: $$R_{\max} = S_{\max}$$(완벽히 맞음)일 때 $$6 - 3 = 3$$, $$R_{\max} = 0$$(완전히 틀림)일 때 $$-3$$이다. 부분 점수를 $$[0,1]$$이 아니라 $$[-3,3]$$으로 늘려 잡은 이유는 형식 reward(최대 1)보다 정확성이 학습 신호에서 훨씬 크게 작용하게 만들기 위해서다 — "형식만 맞추고 안주하지 말라"는 압력을 reward 크기 자체에 심어둔 것이다. 최종 reward의 범위는 $$\mathcal{R}_{\text{final}} \in [-3, 4]$$가 된다.

왜 도구 이름 하나, 파라미터 하나까지 쪼갰을까. 수학 문제처럼 "정답 하나 맞았다/틀렸다"로 채점하면, 도구 이름은 맞았는데 파라미터 하나만 틀린 호출과 모든 걸 틀린 호출이 똑같이 0점을 받는다. 이러면 모델 입장에서 "거의 다 맞는 방향"과 "완전히 엉뚱한 방향"을 구분할 신호가 없다. 세 항으로 쪼개면 "이름은 맞았으니 그 방향은 유지하고 파라미터만 고치면 된다"는 부분 정보가 gradient에 실린다.

이 reward는 GRPO로 학습에 쓰인다. 쿼리 하나당 4개 응답을 샘플링해 그룹을 만들고, 그룹 내 평균·표준편차로 advantage를 정규화하는 GRPO 알고리즘 자체는 RL Reward 설계 시리즈 [GRPO 글](/blog/2026/grpo-deepseekmath/)에서 이미 다뤘으니 여기서는 reward 쪽만 본다. ToolRL은 KL 정규화를 빼고 온도 1.0으로 탐색 폭을 넓힌 cold-start 학습(SFT 없이 바로 GRPO)이 SFT로 초기화한 GRPO보다 일반화가 낫다는 것도 보인다 — SFT 초기화는 학습 reward는 더 빨리 오르지만 벤치마크 성능은 더 낮다. 훈련 reward가 높다고 일반화가 되는 게 아니라는 뜻이다.

### 학습 데이터와 본 결과 — SFT보다 낫고, PPO보다 안정적이다

학습 데이터는 ToolACE(2K, 언제 도구를 부를지 판단하는 일반 도구 사용), Hammer의 masked 서브셋(1K, 도구·파라미터 이름을 무작위 문자열로 가려 설명만 보고 추론하게 만듦), xLAM(1K, 한 턴에 여러 도구를 동시에 부르는 조합적 상황)을 섞은 4K짜리 데이터셋이다. 멀티스텝 궤적은 스텝 단위로 쪼개 이전 대화 이력을 프롬프트에 주입하는 방식으로 단일 스텝 인스턴스로 변환한다.

BFCL v3, API-Bank, 그리고 도구 사용 QA 벤치마크인 Bamboogle에서 raw instruct 모델·SFT·PPO와 비교한 결과, cold-start GRPO가 대체로 가장 좋았다.

| 모델(Qwen2.5-7B-Instruct) | Raw    | SFT(400개) | SFT(4K개) | PPO cold start | **GRPO cold start(ToolRL)** |
| ------------------------- | ------ | ---------- | --------- | -------------- | --------------------------- |
| BFCL v3 Overall Acc       | 41.97% | 34.08%     | 36.53%    | 46.68%         | **58.38%**                  |
| Bamboogle Acc             | 69.6%  | 28.8%      | 30.4%     | 48.0%          | **72.0%**                   |
| Bamboogle 평균 호출 수    | 1.42   | 3.71       | 1.06      | 1.25           | 1.63                        |

abstract가 말하는 "base 대비 17%, SFT 대비 15% 개선"이 이 표에 그대로 드러난다. 눈에 띄는 건 Bamboogle 행이다 — SFT(400개)는 평균 3.71회나 도구를 호출하고도 정답률이 28.8%까지 떨어진다. 반면 GRPO cold start는 1.63회만 호출하고도 72.0%를 맞힌다. 같은 reward 설계를 PPO에도 적용해봤는데, PPO는 cold start일 때 GRPO보다 불안정했고 SFT로 초기화했을 때 오히려 나아지는 경향을 보였다 — GRPO는 cold start에서, PPO는 SFT 초기화에서 각각 더 잘 맞는 조합이라는 뜻이다. 이 reward 설계 자체는 알고리즘에 무관하게 어느 정도 통하지만, GRPO와 결합했을 때 가장 견고했다.

### 일반화 — 학습에 없던 상황에서도 "안 부르는 법"을 배운다

이 reward로 학습한 모델은 학습 데이터에 없던 프로그래밍 언어의 도구나, 학습에 없던 형태의 "무관한 도구 감지" 과제에서도 baseline보다 나은 성능을 보였다. 정성적으로도 흥미로운 사례가 나온다 — 영화 티켓을 예매해달라는 요청에 필요한 정보(영화 제목·날짜)가 빠져 있으면 도구를 바로 호출하지 않고 먼저 되묻고, 도구 목록에 지금 질문과 무관한 함수만 있으면 도구를 부르지 않고 직접 답한다. 두 경우 모두 학습 시 명시적으로 가르친 행동이 아니라, reward가 정확성에 집중되어 있다 보니 "불필요하거나 근거 없는 호출은 감점"이라는 신호가 간접적으로 학습된 결과로 보인다. 이 글 Background 표의 "필요성" 축을 rule로 직접 채점하지 않았는데도, 정확성 reward만으로 어느 정도 유사한 행동이 유도된 셈이다 — 다만 이건 명시적 judge 채점만큼 신뢰할 수 있는 보장은 아니다.

## ToolRM — 정답이 없을 때를 위한 scalar reward model

ToolRL의 정확성 reward는 **정답 도구 호출이 있어야** 계산된다. 그런데 실제 배포 상황에서는 "정답"이 애매하거나 아예 없는 경우가 많다 — 같은 목표를 달성하는 도구 호출 경로가 여러 개일 수 있고, RL 롤아웃 중에는 정답 라벨이 없는 궤적도 많다. ToolRM은 이 공백을 메우는 **outcome reward model(ORM)**이다.

### 출발점 — 일반 RM은 도구 호출을 못 읽는다

저자들은 먼저 기존 reward model들이 도구 호출을 제대로 채점하는지부터 검증한다. 이를 위해 **FC-RewardBench**를 만들었다 — BFCL-v3의 싱글턴 스플릿에서 사용자 질의·도구 카탈로그·정답 호출을 가져오고, 0.5B부터 685B까지 25개 모델 풀로 오답 호출을 생성해 짝지은 1,500개 (정답, 오답) 쌍이다. 오답의 오류 유형 분포를 보면 이 벤치마크가 왜 어려운지 드러난다.

| 오류 유형            | 개수 |
| -------------------- | ---- |
| 파라미터 값 오류     | 650  |
| 함수 이름 오류       | 403  |
| 함수 개수 오류       | 245  |
| 선택적 파라미터 누락 | 78   |
| 필수 파라미터 누락   | 45   |
| 파라미터 타입 오류   | 43   |
| 예상치 못한 파라미터 | 21   |
| 출력 형식 오류       | 15   |

650건이 "파라미터 값 오류"다. 함수 이름을 통째로 틀리는 것보다 값 하나가 미묘하게 틀린 경우가 압도적으로 많다는 뜻이고, 이건 자연어 채점에 익숙한 RM에게는 잡아내기 까다로운 종류의 오류다. 실제로 도구 전용 RM인 Themis조차 FC-RewardBench에서 45% 정확도밖에 못 낸다(무작위가 50%이니 사실상 못 푸는 수준). 반면 LLM-as-judge 방식(70B\~685B급 LLM에게 두 후보 중 정답을 고르게 함)은 80%를 넘긴다 — 다만 그 크기의 모델을 매 롤아웃마다 돌리는 건 비현실적이다.

### 학습 — Bradley-Terry pairwise + reward centering

ToolRM은 표준적인 Bradley-Terry 방식으로 학습된 scalar RM이다. 입력 $$x$$(도구 카탈로그 + 대화 이력)에 대해 선호되는 응답 $$y_+$$가 비선호 응답 $$y_-$$보다 높은 점수를 받을 확률은

$$p(y_+ \succ y_- \mid x) = \sigma\big(r_\theta(x, y_+) - r_\theta(x, y_-)\big)$$

이고, 학습 목적함수는 이 확률의 로그를 최대화하는 것이다.

$$J(r) = \max_{r_\theta} \; \mathbb{E}_{(x,y_+,y_-)\sim D}\big[\log \sigma\big(r_\theta(x,y_+) - r_\theta(x,y_-)\big)\big]$$

여기에 reward centering 정규화 항을 더해 점수가 0 근처에 모이도록 만든다.

$$J_{\text{reg}}(r) = J(r) + \eta \, \mathbb{E}_{(x,y_+,y_-)\sim D}\big[(r_\theta(x,y_+) + r_\theta(x,y_-))^2\big]$$

$$\eta$$는 이 정규화 세기를 조절하는 작은 양수다. 이 항이 왜 필요한가 — 정규화가 없으면 두 응답의 점수를 통째로 큰 양수 방향(혹은 음수 방향)으로 밀어도 loss가 똑같이 줄어드는 자유도가 남는다(pairwise loss는 차이만 보므로). reward centering은 $$r_\theta(x,y_+) + r_\theta(x,y_-)$$가 0에 가깝도록 눌러서 점수의 절대적 스케일을 고정시키고, 이후 이 scalar를 GRPO의 reward로 그대로 쓸 수 있게 해준다.

베이스 아키텍처는 Qwen2.5-Instruct(1.5B/7B/14B)이고, 언어모델 헤드를 scalar 출력 선형 레이어로 교체한다. 학습 데이터는 APIGen, Schema-Guided Dialogue(멀티턴), xlam-irrelevance(호출이 필요 없는 케이스)를 섞고, 함수·파라미터 이름을 무작위 문자열로 바꿔 모델이 이름을 외워서 맞히지 못하게 한 뒤(Lin et al. 2024 방식을 따름), 11개의 오픈소스 도구 호출 모델(0.5B\~32B)에게 호출을 생성시켜 정답과 다른 것만 오답으로 남긴다. 이렇게 모은 180K 샘플(싱글턴 85K, 멀티턴 85K, 호출 불필요 10K)로 학습한다. 오답을 사람이 설계하지 않고 실제 모델이 만들어내는 자연스러운 실패 패턴을 모으는 방식이라, "이론적으로 있을 법한 오류"가 아니라 "실제로 모델이 저지르는 오류"를 학습 신호로 쓴다는 점이 특징이다.

### 쓰임새 — 채점만 하지 않고 추론 시점 선택에도 쓴다

ToolRM은 학습된 채점기로 끝나지 않는다. 후보 $$n=32$$개를 온도 $$T=0.6$$으로 생성한 뒤 ToolRM-14B 점수가 가장 높은 것을 고르는 Best-of-$$n$$ 선택에 쓰면, 특히 작은 생성 모델에서 이득이 크다.

| 생성 모델               | Greedy | ToolRM-14B Best-of-32                                                    |
| ----------------------- | ------ | ------------------------------------------------------------------------ |
| Qwen3-0.6B              | 39.5%  | 64.4% (+24.9pt, greedy 기준 Qwen3-32B의 63.8%·xLAM-2-70B의 63.6%를 역전) |
| Qwen3-8B                | \~65%  | 70.5% (모든 greedy baseline 대비 +5.6pt)                                 |
| Llama-xLAM-2-32B급 이상 | —      | 개선 폭 2\~3pt로 수렴 (이미 강한 모델은 얻을 게 적다)                    |

작은 모델일수록 이득이 크다는 건 자연스럽다 — 생성 분포의 편차가 클수록 "여러 개 뽑아서 가장 나은 것을 고르는" 전략의 여지가 크기 때문이다. 이미 정확도가 높은 큰 모델은 애초에 후보들이 비슷비슷해서 골라봐야 차이가 작다.

노이즈에 대한 강건성도 확인했다. RoTBench는 도구·파라미터 이름에 문자 삽입·삭제·이름 뒤섞기 같은 잡음을 섞은 벤치마크인데, 잡음이 없는 Clean 스플릿보다 잡음이 심한 Union 스플릿에서 Best-of-32의 이득이 더 컸다. 예를 들어 Qwen-8B의 도구 선택 정확도는 Clean에서 52.4% → 72.4%로 20pt 개선되는데, 잡음이 낀 Union에서는 45.7% → 66.7%로 21pt가 개선된다 — 어려운 조건일수록 "여러 번 시도해서 그나마 맞는 걸 고르는" 전략의 가치가 더 커진다는 뜻이다.

# Experiments

## ToolRL의 reward ablation — 무엇이 성능을 살리고 죽이는가

ToolRL의 값은 본문 결과표보다 **reward 설계를 하나씩 바꿔가며 무엇이 통하고 무엇이 안 통하는지 보여준 ablation**에 있다. 세 가지 축(길이, 스케일, 세분화)을 원문 수치 그대로 재구성한다.

### (1) 길이 reward — 길게 생각하면 보상을 준다

R1 계열 모델이 긴 추론에서 이득을 본다는 관찰을 따라, `<think>` 필드 길이에 비례한 reward $$\mathcal{R}_{\text{length}} = \min(L_{\text{think}}/L_{\text{target}}, 1)$$ ($$L_{\text{target}}=512$$)를 추가했다. 결과는 정반대였다.

| 모델                  | 원래 설계 | 길이 reward 추가 | 동적 길이 reward |
| --------------------- | --------- | ---------------- | ---------------- |
| Qwen2.5-1.5B-Instruct | 46.20%    | 33.23%           | 28.51%           |
| Qwen2.5-3B-Instruct   | 52.98%    | 48.89%           | 48.24%           |
| Llama-3.2-3B-Instruct | 44.10%    | 44.98%           | 43.15%           |

(BFCL V3 Overall Accuracy 기준)

작은 모델일수록 타격이 컸다 — Qwen2.5-1.5B는 46.20%에서 33.23%로 13%p 가까이 떨어졌다. 학습 도중 응답 길이와 길이 reward 자체는 꾸준히 올라갔으니 "모델이 더 길게 생각하도록 유도하는 데는 성공"했지만, 그 긴 생각이 과제 성능으로 이어지지 않았다. 학습 진행에 따라 목표 길이를 점차 늘리는 동적 버전으로 완화해봐도 마찬가지였다. 결론: **긴 추론 흔적이 저절로 좋은 게 아니다.** 도구 호출처럼 답이 비교적 명확한 과제에서는 장황한 사고가 오히려 과잉 사고(overthinking)로 이어져 정답에서 멀어질 수 있다.

### (2) reward 스케일 — 형식과 정확성 사이 가중치

정확성 reward의 최댓값을 형식 reward와 같은 크기로 맞추면(원래 $$[-3,3] \to$$ 동일 최대치인 $$[-1,1]$$) 어떻게 될까.

| 모델                  | 원래 설계 | 최댓값 동일(Equal Max) | 2단계(Coarse) | 연속 동적(Dynamic) |
| --------------------- | --------- | ---------------------- | ------------- | ------------------ |
| Qwen2.5-1.5B-Instruct | 46.20%    | 39.47%                 | 38.85%        | 45.71%             |
| Qwen2.5-3B-Instruct   | 52.98%    | 51.76%                 | 50.66%        | 53.81%             |
| Llama-3.2-3B-Instruct | 44.10%    | 42.47%                 | 41.33%        | 46.85%             |

정확성 reward의 상대적 가중치를 낮추자 세 모델 모두 성능이 떨어졌다. 형식은 맞추기 쉬운 목표이고 정확성은 어려운 목표인데, 둘을 동등하게 취급하면 모델이 쉬운 목표에 안주할 여지가 생긴다는 뜻이다. 반대로 학습 초반엔 형식에, 후반엔 정확성에 무게를 싣도록 **점진적으로** 전환하는 동적 스케일링(학습 진행도 $$p \in [0,1]$$에 대해 형식 범위는 $$[-2+p, 2-p]$$로 좁아지고 정확성 범위는 $$[-2-p, 2+p]$$로 넓어짐)은 세 모델 중 두 곳에서 원래 설계를 앞질렀다. 반면 30스텝을 기점으로 뚝 전환하는 **급격한** 2단계 방식은 오히려 더 나빴다. 요점은 "정확성에 더 큰 가중치를 줘야 한다"는 방향은 맞지만, **전환은 매끄러워야 한다**는 것이다.

### (3) reward 세분화 — 얼마나 잘게 쪼개야 하나

정확성 reward를 원래의 세 항(이름·파라미터 이름·파라미터 값, 부분 점수 허용)에서 점점 뭉뚱그려봤다.

| 모델                  | 원래(Finegrained) | 이름·파라미터 이름만 exact match | 파라미터 이름+값 통째로 exact match | 도구 집합 전체 exact match |
| --------------------- | ----------------- | -------------------------------- | ----------------------------------- | -------------------------- |
| Qwen2.5-1.5B-Instruct | 46.20%            | 40.71%                           | 37.65%                              | 36.72%                     |
| Qwen2.5-3B-Instruct   | 52.98%            | 52.06%                           | 51.36%                              | 51.40%                     |
| Llama-3.2-3B-Instruct | 44.10%            | 39.82%                           | 38.62%                              | 35.95%                     |

(표의 열 이름은 원문의 Finegrained / Intermediate / Coarse 세 등급을 그대로 옮겼다. Finegrained는 이름 일치를 정확히는 요구하되 부분 점수 대신 이진 판정으로 바꾼 것이고, Coarse는 도구 호출 집합 전체가 정답과 완전히 같을 때만 1점을 주는 방식이다.)

뭉뚱그릴수록 대체로 성능이 떨어졌다(Qwen2.5-1.5B는 46.20% → 36.72%로 거의 10%p 하락). 학습 도중 reward 곡선을 보면 세분화가 거칠수록 모델이 높은 reward를 받기가 더 어려워졌다 — 부분 점수가 없으니 "거의 맞았지만 완전히 맞지는 않은" 시도가 전부 0점으로 뭉개지고, 이게 성근(sparse) 학습 신호가 되어 credit assignment를 방해한다는 뜻이다. **세밀한 분해가 안정적이고 효과적인 학습으로 이어진다**는 게 저자들의 결론이다.

## ToolRM이 GRPO reward로 쓰였을 때

ToolRM 논문은 자신들의 scalar RM을 ToolRL과 같은 GRPO 파이프라인에 넣어 비교한다. 세 가지 정확성 reward 변형을 붙였다 — 스키마 검증만 하는 $$R_{\text{schema}} \in \{-1,1\}$$(정답 없이 스키마 위반만 확인), ToolRL 방식 그대로인 $$R_{\text{ToolRL}} \in [-3,3]$$(정답 도구 호출 필요), 그리고 ToolRM-14B가 매기는 $$R_{\text{ToolRM}} \in [-3,3]$$(역시 정답 불필요).

| 모델                  | reward 종류                         | Non-Live AST Acc | Live AST Acc |
| --------------------- | ----------------------------------- | ---------------- | ------------ |
| Llama-3.2-3B-Instruct | base                                | 15.35%           | 43.82%       |
|                       | $$R_{\text{schema}}$$               | 51.71%           | 62.25%       |
|                       | $$R_{\text{ToolRL}}$$ (정답 필요)   | 75.27%           | 64.25%       |
|                       | $$R_{\text{ToolRM}}$$ (정답 불필요) | 78.40%           | 64.32%       |
| Qwen2.5-3B-Instruct   | base                                | 43.06%           | 55.66%       |
|                       | $$R_{\text{schema}}$$               | 63.17%           | 66.54%       |
|                       | $$R_{\text{ToolRL}}$$ (정답 필요)   | 80.42%           | 67.21%       |
|                       | $$R_{\text{ToolRM}}$$ (정답 불필요) | 79.58%           | 67.51%       |

핵심은 등수가 아니라 **$$R_{\text{ToolRM}}$$이 정답 없이도 정답이 있어야 하는 $$R_{\text{ToolRL}}$$과 대등하거나 더 나은 결과를 낸다**는 것이다. Llama-3.2-3B에서는 ToolRM이 두 지표 모두 앞섰고, Qwen2.5-3B에서는 Non-Live에서만 근소하게 뒤처졌다. 스키마 검증만 하는 가장 단순한 규칙조차 base 대비 20%p 안팎의 개선을 냈다는 점도 흥미롭다 — "완전한 정확성 판정"이 없어도 "적어도 스키마는 지켰다"는 최소 신호만으로 상당한 학습이 된다는 뜻이다.

데이터 필터링 실험도 비슷한 그림이다. Llama-3.1-8B-Instruct를 16K 샘플로 파인튜닝하면 정확도가 54.0% → 61.0%로 오르는데, 이 중 무작위로 8K만 뽑으면 58.4%로 **오히려 떨어진다**(저품질 샘플이 섞여 들어가서). 반면 ToolRM-14B가 매긴 점수 상위 8K만 골라 학습하면 62.5%로, 데이터의 절반만 쓰고도 전체 데이터보다 나은 결과를 낸다. 이 관찰은 이 시리즈에서 되풀이해온 주제와 같다 — **양이 아니라 채점의 질이 학습을 좌우한다.**

## 토이 예제 — 3턴 궤적을 다섯 축으로 채점하기

논문 두 편의 수치를 확인했으니, 이제 [#51](/blog/2026/reward-shaping-agentic/)에서 예고한 "도구 호출 성공에 +점을 주면 남발한다"는 문제를 직접 계산으로 확인해보자.

**과제**: 사용자가 "여의도 근처 3만원 이하 파스타 맛집 추천하고, 예약 가능한 시간도 알려줘"라고 묻는다. 사용 가능한 도구는 `search_restaurant(location, cuisine, price_max)`와 `get_reservation_slots(restaurant_id, date)`, 그리고 이 과제와 무관한 `calculator(expression)`이다.

세 개의 3턴 궤적을 만든다.

**궤적 A (모범)**

1. `search_restaurant(location="여의도", cuisine="파스타", price_max=30000)` 호출 — 형식·선택·파라미터 모두 정확
2. `get_reservation_slots(restaurant_id=<1번 결과>, date=오늘)` 호출 — 역시 모두 정확
3. 두 결과를 반영해 "A식당 파스타, 오늘 저녁 7시 예약 가능" 같은 응답 생성

**궤적 B (형식은 완벽하지만 남발)**

1. `search_restaurant(...)` 호출 — A와 동일하게 정확
2. `calculator(expression="30000*0.9")` 호출 — JSON은 완벽하지만 이 과제에 불필요한 도구 선택
3. `get_reservation_slots(...)` 호출은 하지만, 결과를 무시하고 "아무 때나 예약 가능합니다"라고 지어내 응답

**궤적 C (게으름)**

1. 도구를 전혀 부르지 않고 곧바로 "여의도에 파스타집 많아요"라고 응답

다섯 축을 궤적 단위로 점수화한다(형식·선택·파라미터는 호출마다 0\~1로 매겨 호출별로 부여, 필요성·활용은 궤적 전체에 대해 judge가 0\~1로 매긴다고 가정).

| 궤적 | 호출 수 | 호출별 (형식+선택+파라미터) 합 | 필요성 (불필요 호출 비율로 계산)                           | 결과 활용 |
| ---- | ------- | ------------------------------ | ---------------------------------------------------------- | --------- |
| A    | 2       | $$3+3=6$$                      | $$1 - 0/2 = 1.0$$                                          | 1.0       |
| B    | 3       | $$3+1+3=7$$                    | $$1 - 1/3 \approx 0.67$$                                   | 0.0       |
| C    | 0       | $$0$$                          | $$1.0$$ (호출이 0건이니 불필요한 호출도 0건 — 정의상 만점) | 0.0       |

(B의 2번째 호출은 형식은 맞지만 도구 선택이 틀렸으므로 그 호출의 형식+선택+파라미터 합은 $$1+0+1=1$$이 아니라 $$1+0+0=1$$로 계산했다 — 애초에 필요 없는 호출이라 파라미터의 "정확성"이 무의미하므로 0으로 잡았다. 나머지 두 호출은 형식 1, 선택 1, 파라미터 1로 만점씩 3점이다.)

이제 세 가지 가중치 프로파일로 총점을 계산해본다.

**프로파일 1 — 호출별 점수를 그냥 합산(정규화 없음)**: 총점 $$=$$ 호출별 합. **A=6, B=7, C=0.** B가 A를 이긴다. 호출을 하나 더 만든 것 자체가 원시 합산 점수를 올렸기 때문이다 — 결과를 무시하고 지어낸 응답인데도 형식과 파라미터가 맞은 호출을 하나 더 쌓았다는 이유만으로 더 높은 점수를 받는다. **이게 정확히 "형식·선택·파라미터에만 보상을 주면 호출을 남발한다"는 현상이다.**

**프로파일 2 — 호출 횟수에 페널티만 부과**: 총점 $$= -(\text{호출 수})$$. **A=-2, B=-3, C=0.** 이번엔 C가 이긴다. 아무 도구도 안 부른 궤적이 가장 좋은 점수를 받는다 — **필요한 호출까지 억제된다.**

**프로파일 3 — 호출당 평균 점수(정규화)와 필요성·활용을 함께 반영**: 호출별 평균을 만점 3으로 정규화하고, 형식·선택·파라미터 : 필요성 : 활용 가중치를 $$0.4 : 0.3 : 0.3$$으로 섞는다.

$$\text{score} = 0.4 \cdot \frac{\text{호출별 합}/\text{호출수}}{3} + 0.3 \cdot \text{필요성} + 0.3 \cdot \text{활용}$$

- A: $$0.4 \cdot \frac{6/2}{3} + 0.3 \cdot 1.0 + 0.3 \cdot 1.0 = 0.4 + 0.3 + 0.3 = 1.0$$
- B: $$0.4 \cdot \frac{7/3}{3} + 0.3 \cdot 0.67 + 0.3 \cdot 0 = 0.4 \cdot 0.778 + 0.201 + 0 \approx 0.51$$
- C: $$0.4 \cdot 0 + 0.3 \cdot 1.0 + 0.3 \cdot 0 = 0.3$$

**A(1.0) > B(0.51) > C(0.3).** 필요성과 결과 활용이라는 judge 축이 들어가야 비로소 "호출은 정확하지만 결과를 무시한" B와 "아예 시도하지 않은" C가 A보다 뒤처진다. 그리고 B가 C보다는 낫다는 순서도 유지된다 — 적어도 시도는 했고 검색 결과는 활용했으니까.

세 프로파일을 나란히 보면 가중치가 곧 정책이라는 게 드러난다. **원시 합산은 남발을 낳고, 무조건적 횟수 페널티는 태만을 낳고, 필요성·활용까지 반영한 균형 잡힌 가중치만 실제로 원하는 행동(A)을 1등으로 만든다.**

## 반례 — 형식만 보는 reward의 전형적 hacking

위 프로파일 1이 바로 그 반례다. 형식·도구 선택·파라미터 세 축만 채점하고 호출 단위로 단순 합산하면, 정책은 "결과를 실제로 쓰는가"와 무관하게 **문법적으로 유효한 호출을 최대한 많이 만들어내는 방향**으로 최적화된다. 실제로 ToolRL 논문의 Bamboogle 결과에서 이 패턴을 볼 수 있다. Qwen2.5-7B-Instruct를 400개 샘플로 SFT만 시킨 모델(SFT400)은 평균 3.71회의 도구 호출을 쓰고도 정답률 28.8%에 그쳤다. 반면 ToolRL의 GRPO cold-start 모델은 평균 1.63회만 호출하고도 72.0%를 맞혔다. **호출 횟수와 정답률이 반비례하는 이 구간이, "형식만 맞으면 점수를 준다"는 설계가 만들어내는 전형적 실패 모드다.** 호출을 많이 할수록 형식 reward를 받을 기회가 늘어나니 모델이 그 방향으로 쏠리고, 정작 그 호출들이 최종 답에 기여하는지는 아무도 채점하지 않는다.

# 통계 요약

| 기법                       | 무엇을 채점하나                               | 정답 필요 여부 | 핵심 수치                                                       |
| -------------------------- | --------------------------------------------- | -------------- | --------------------------------------------------------------- |
| ToolRL 형식 reward         | 태그 존재·순서                                | 필요           | $$\{0,1\}$$ 이진값                                              |
| ToolRL 정확성 reward       | 도구 이름 + 파라미터 이름 + 값 (자카드/exact) | 필요           | $$[-3,3]$$, 세분화할수록 BFCL 최대 \~10%p 향상                  |
| ToolRL 동적 스케일         | 형식→정확성으로 가중치 점진 전환              | 필요           | 급격한 전환보다 매끄러운 전환이 우세                            |
| ToolRM (Bradley-Terry ORM) | 전체 호출의 상대적 우열                       | 불필요         | FC-RewardBench 1,500쌍, ToolRM-1.5B가 gpt-oss-120B judge를 능가 |
| ToolRM Best-of-32          | 후보 32개 중 최고점 선택                      | 불필요         | 작은 모델일수록 이득 큼(Qwen3-0.6B: 39.5%→64.4%)                |
| ToolRM 데이터 필터링       | 학습 샘플 품질 순위                           | 불필요         | 상위 50%만 써도 전체 데이터 대비 우세(54.0→62.5%)               |
| ToolRM as GRPO reward      | 정답 없는 policy reward                       | 불필요         | ToolRL(정답 필요) reward와 대등\~우세                           |

# Conclusion

이 편의 핵심은 한 문장으로 줄일 수 있다. **도구 호출은 다섯 개의 서로 다른 질문으로 쪼개서 채점해야 하고, 그중 절반(형식·선택·파라미터)은 규칙으로 충분하지만 나머지 절반(필요성·결과 활용)은 판정이 필요하다.** ToolRL은 규칙으로 채점되는 절반을 극한까지 정교하게 다듬어 보였다 — 세분화는 득이 되고, 길이 reward는 독이 되고, 정확성에는 형식보다 큰 가중치를 매끄럽게 실어야 한다는 세 가지 결론이 방대한 ablation으로 뒷받침된다. ToolRM은 그 규칙이 통하지 않는 상황(정답이 없는 롤아웃)에 대응하는 학습된 대체재를 제시했고, 실제로 정답이 있어야 하는 규칙 기반 reward와 대등한 성능을 정답 없이 냈다.

두 논문 모두 아직 다루지 않은 게 있다 — "필요성"과 "결과 활용"을 이 글에서는 토이 예제로만 계산해봤을 뿐, 실제로 그걸 judge에게 시켰을 때 얼마나 신뢰할 수 있는지는 검증하지 않았다. 그 judge 채점 자체의 신뢰도와 루브릭 설계는 다음 편([#54](/blog/2026/agentic-judge-rubric/))의 주제다.

# 참고 문헌

- Qian et al., 2025. [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958). arXiv:2504.13958.
- [GitHub: qiancheng0/ToolRL](https://github.com/qiancheng0/ToolRL) — 코드 및 학습 설정 공개 저장소.
- Agarwal et al., 2025. [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](https://arxiv.org/abs/2509.11963). arXiv:2509.11963.
- Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — GRPO 원 논문.
- Patil et al., 2025. Berkeley Function Calling Leaderboard v3(BFCL) — [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)

---

# RL Reward 설계 시리즈

이 글은 RL Reward 설계 시리즈의 쉰세 번째 글이다.

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

**8부. 생각하는 Judge**

<ol start="39">
  <li><a href="/blog/2026/reasongrm/">ReasonGRM (2025)</a> — reasoning 능력을 judge에 이식</li>
  <li><a href="/blog/2026/j1-thinking-judge/">J1 (2025)</a> — RL로 judge를 생각하게 만들기</li>
  <li><a href="/blog/2026/rubrics-as-rewards/">Rubrics as Rewards (2025)</a> — 비검증 도메인으로</li>
  <li><a href="/blog/2026/criticeval/">CriticEval (2024)</a> — judge 자체를 어떻게 평가하나</li>
  <li><a href="/blog/2026/one-token-to-fool-judge/">One Token to Fool LLM-as-a-Judge (2025)</a> — GenRM도 뚫린다</li>
</ol>

**9부. 에이전트는 무엇이 다른가**

<ol start="44">
  <li><a href="/blog/2026/agentic-rl-landscape/">에이전트 RL은 무엇이 다른가</a> — 장기 지평·희소 보상·긴 궤적</li>
  <li><a href="/blog/2026/credit-assignment-survey/">공을 어디에 돌릴 것인가</a> — credit assignment 47개 방법의 지도</li>
  <li><a href="/blog/2026/multi-turn-rl-practice/">멀티턴 RL 실무 가이드</a> — 무엇이 실제로 작동하는가</li>
</ol>

**10부. credit assignment — 공을 어디에 돌릴 것인가**

<ol start="47">
  <li><a href="/blog/2026/outcome-vs-process-agentic/">결과만으로는 부족하다</a> — 장기 지평에서 증폭되는 RLVR의 한계</li>
  <li><a href="/blog/2026/turn-level-reward/">턴 단위로 공을 나눈다</a> — turn-level reward 설계</li>
  <li><a href="/blog/2026/step-level-credit/">스텝을 단위로 삼는다</a> — 행동 단위 궤적 표현과 credit</li>
  <li><a href="/blog/2026/token-segment-credit/">토큰과 세그먼트로 더 잘게</a> — 세밀한 입도의 득과 실</li>
  <li><a href="/blog/2026/reward-shaping-agentic/">shaping은 약인가 독인가</a> — 중간 보상의 효율과 위험</li>
</ol>

**11부. 에이전트의 reward는 어디서 오나**

<ol start="52">
  <li><a href="/blog/2026/environment-as-reward/">환경이 곧 reward다</a> — 샌드박스·테스트·상태 검증</li>
  <li><strong>(현재 글)</strong> 도구 호출을 어떻게 채점하나 — ToolRL·ToolRM</li>
  <li><a href="/blog/2026/agentic-judge-rubric/">궤적을 judge가 채점한다</a> — rubric 생성형 reward의 확장</li>
</ol>

**12부. 에이전트 도메인별 설계**

<ol start="55">
  <li><a href="/blog/2026/search-agent-rl/">검색 에이전트</a> — Search-R1에서 DeepDive까지</li>
  <li><a href="/blog/2026/swe-agent-rl/">코드 에이전트</a> — SWE-RL과 테스트라는 reward</li>
  <li><a href="/blog/2026/web-gui-agent-rl/">웹·GUI 에이전트</a> — end-to-end 멀티턴 RL</li>
</ol>

**13부. 에이전트의 실패와 방어**

<ol start="58">
  <li><a href="/blog/2026/agentic-reward-hacking/">에이전트의 reward hacking</a> — 판정기가 뚫린다, 그리고 조합의 실패</li>
</ol>

**14부. 실전 종합**

<ol start="59">
  <li><a href="/blog/2026/frontier-reward-design/">프론티어의 helpfulness reward 설계</a> — 열한 개 모델이 능력 축에서 택한 것</li>
  <li><a href="/blog/2026/frontier-safety-design/">프론티어의 harmlessness reward 설계</a> — 안전 축과 over-refusal 트레이드오프</li>
  <li><a href="/blog/2026/frontier-agentic-rl/">프론티어 모델은 실제로 어떻게 하나</a> — 최신 모델들의 agentic RL 설계</li>
  <li><a href="/blog/2026/reward-model-design/">reward를 어떻게 설계할 것인가</a> — 시리즈를 관통한 RM 설계 원칙 한 장</li>
</ol>

본 시리즈는 62편으로 구성된다.
