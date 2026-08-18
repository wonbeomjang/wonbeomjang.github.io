---
layout: post
title: "Math-Shepherd: 사람 라벨 없이 PRM을 만들다"
date: 2026-08-11 09:20:00 +0900
description: "RLHF Reward 설계 시리즈 #20 — rollout으로 단계별 라벨을 자동 생성하는 법, 그리고 PRM 평가의 함정"
categories: [paper]
tags: [rlhf, reward-model, prm, process-supervision, reasoning, paper]
giscus_comments: true
related_posts: true
---

> [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935) (Wang et al., ACL 2024)

# Introduction

[19편](/blog/2026/lets-verify-step-by-step/)에서 확인한 결론은 명확했다. 최종 답만 채점하는 outcome reward model(ORM)보다, 풀이의 각 단계를 하나씩 채점하는 process reward model(PRM)이 수학 추론에서 훨씬 신뢰할 수 있는 reward 신호를 준다. 문제는 그 PRM을 어떻게 만드느냐였다. OpenAI의 PRM800K는 사람 라벨러가 MATH 문제 풀이의 단계 하나하나에 "옳다/틀리다"를 매겨 80만 건의 step-level 라벨을 만들었다. 정확하긴 하지만, 이 방식은 문제 하나가 늘어날 때마다 사람의 시간이 그만큼 늘어난다는 근본적인 한계를 갖는다. 19편이 이 시리즈에 남긴 부채는 정확히 이것이었다 — **"단계별 사람 라벨은 너무 비싸다. 그러면 자동으로 만들 수는 없는가?"**

이번 글에서 다루는 Math-Shepherd는 이 질문에 대한 정면 답변이다. 핵심 아이디어는 단순하다. 어떤 단계가 "좋은 단계"인지 사람에게 묻는 대신, **그 단계에서 이어서 여러 번 풀어보게 하고 정답에 도달하는 비율을 세면 된다**는 것이다. 사람이 한 줄씩 채점하는 대신, 모델 스스로 자기 풀이의 각 갈림길에서 "여기서 계속 가면 얼마나 자주 정상에 도착하는가"를 rollout으로 재본다. 그리고 이렇게 얻은 라벨로 학습한 PRM을, [16편 GRPO](/blog/2026/grpo-deepseekmath/)와 같은 DeepSeek 계열의 관심사인 **RL의 reward 신호**로 직접 사용한다.

이 글에서 답할 질문은 세 가지다.

1. **어떻게 사람 없이 라벨을 만드는가**: rollout 기반 hard/soft estimation의 정확한 정의.
2. **그 라벨로 만든 PRM을 어디에 쓰는가**: best-of-N 재랭킹(verifier)과 PPO reward(policy 학습), 두 가지 용도.
3. **이 PRM을 믿어도 되는가**: ProcessBench와 PRMBench가 드러낸, "best-of-N 성능이 좋다"와 "PRM이 단계를 제대로 판별한다"의 간극.

미리 예고하자면, 세 번째 질문의 답은 그리 밝지 않다. 그리고 이 어두운 답이 [21편 DeepSeek-R1](/blog/2026/deepseek-r1/)이 결국 PRM 자체를 포기하는 이유로 이어진다.

# Background

## PRM vs ORM, 짧은 복습

19편에서 다룬 구도를 한 문장으로: ORM은 풀이 전체 $$S$$에 스칼라 보상 하나를 매기고, PRM은 각 단계 $$s_i$$에 개별 보상 $$r_{s_i}$$를 매긴다. PRM이 이기는 이유는 "어디서 틀렸는지"를 짚어주기 때문이다 — 5단계 풀이 중 3단계에서 부호를 잘못 옮겼다면, ORM은 "이 풀이는 틀렸다"고만 말하지만 PRM은 "3단계가 문제"라고 말한다. 문제는 이 $$r_{s_i}$$를 어디서 얻느냐였고, PRM800K는 그 답을 사람 손에 맡겼다.

| 항목        | ORM                      | PRM([PRM800K, 19편](/blog/2026/lets-verify-step-by-step/)) | PRM(Math-Shepherd, 이 글)      |
| ----------- | ------------------------ | ---------------------------------------------------------- | ------------------------------ |
| 라벨 단위   | 풀이 전체 1개            | 단계마다 1개                                               | 단계마다 1개                   |
| 라벨 출처   | 최종 답 일치 여부 (자동) | 사람 라벨러                                                | rollout 정답 도달 비율 (자동)  |
| 라벨링 비용 | 거의 없음                | 매우 큼 (80만 건, 사람 시간)                               | rollout 계산 비용으로 대체     |
| 규모 확장   | 쉬움                     | 어려움                                                     | 비교적 쉬움 (계산만 늘리면 됨) |

## "좋은 단계"를 어떻게 정의할 것인가 — value 추정이라는 관점

여기서 Math-Shepherd가 던지는 재정의가 핵심이다. **"좋은 단계"란, 그 단계 이후로 계속 풀었을 때 정답으로 이어질 잠재력이 큰 단계**라는 것이다. 이건 사실 강화학습에서 말하는 상태 가치함수 $$V(s)$$의 정의와 똑같다. $$V(s)$$는 "상태 $$s$$에서 정책을 계속 따라갔을 때 기대되는 최종 보상"이고, 이걸 정확히 계산할 수 없으니 몬테카를로 방식으로 **여러 번 시뮬레이션을 굴려서 평균을 낸다**. 바둑 MCTS(Monte Carlo Tree Search)가 정확히 이렇게 동작한다 — 어떤 수를 평가할 때 기보 데이터베이스에 물어보는 게 아니라, 그 수를 둔 상태에서 게임을 끝까지 여러 번 시뮬레이션해보고 이긴 비율로 그 수의 가치를 매긴다. Math-Shepherd는 수학 풀이의 각 단계에 정확히 같은 발상을 적용한다. 즉 "이 단계는 논리적으로 우아한가"를 사람이 판단하는 대신, "이 단계에서 시작해 여러 번 끝까지 풀어봤을 때 얼마나 자주 정답에 도착하는가"로 그 단계의 가치를 근사한다.

일상 비유를 하나 더 붙이면, 등산 내비게이션과 비슷하다. 갈림길에서 어느 쪽 길이 좋은 길인지 표지판이 없을 때, 그 갈림길에서 여러 등산객을 실제로 보내보고 "정상까지 도착한 사람의 비율"로 그 갈림길의 좋음을 매기는 것이다. 실제로 길을 걸어본 경험(rollout)이 곧 그 갈림길의 평가가 된다.

# Method

## rollout으로 라벨 만들기: hard estimation과 soft estimation

구체적인 절차는 이렇다. 문제 $$x$$와 정답 $$a^*$$가 주어지고, 어떤 모델(completer)이 만든 풀이 $$S = (s_1, s_2, \ldots, s_K)$$가 있다고 하자. 이 풀이의 $$i$$번째 단계까지의 부분 풀이를 $$s_{\le i} = (s_1, \ldots, s_i)$$라 하면, completer 모델이 $$s_{\le i}$$ 뒤를 이어서 서로 다른 $$N$$개의 완성본(rollout) $$a_1, a_2, \ldots, a_N$$을 샘플링한다. 각 $$a_j$$는 그 rollout이 도달한 최종 답이다.

<p align="center"><img src="/assets/post/image/math-shepherd/fig2-process-annotation.png" width="85%"></p>

위 그림(논문 Figure 2)이 이 절차 자체다. 문제는 "4차 monic 다항식 $$p(x)$$의 근 중 셋이 1, 2, 3일 때 $$p(0)+p(4)$$를 구하라"이고 정답은 24다. (a)는 기존 outcome annotation — 풀이 전체가 틀린 답(20)에 도달했으니 $$y_S=0$$ 하나만 남는다. (b)가 Math-Shepherd의 process annotation이다. 첫 단계 $$s_1$$에서 갈라져 나온 3개의 rollout 중 2개는 정답(24)에, 1개는 오답(20)에 도달했다. 이로부터 두 가지 라벨을 정의한다.

**Hard estimation (HE)**: $$N$$개의 rollout 중 단 하나라도 정답에 도달하면 그 단계에 1을 준다.

$$y_{s_i}^{HE} = \begin{cases} 1 & \text{if } \exists\, j \in \{1,\ldots,N\} \text{ s.t. } a_j = a^* \\ 0 & \text{otherwise} \end{cases}$$

- $$y_{s_i}^{HE}$$: 단계 $$s_i$$의 hard 라벨. 0 또는 1의 이진값이다.
- $$\exists\, j$$: $$N$$개 rollout 중 정답에 도달한 것이 하나라도 있으면 조건이 참이 된다.

**Soft estimation (SE)**: 정답에 도달한 rollout의 비율을 그대로 라벨로 쓴다.

$$y_{s_i}^{SE} = \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}(a_j = a^*)$$

- $$\mathbb{1}(a_j = a^*)$$: $$j$$번째 rollout의 답이 정답과 같으면 1, 아니면 0.
- $$y_{s_i}^{SE}$$: 0과 1 사이의 연속값. "이 단계가 정답으로 이어질 확률"의 몬테카를로 추정치다.

그림의 $$s_1$$은 $$N=3$$일 때 $$y_{s_1}^{HE}=1$$(2개가 정답에 도달했으므로), $$y_{s_1}^{SE}=2/3$$이다.

이 둘의 차이는 일기예보에 비유하면 이해가 쉽다. HE는 "내일 비가 온다/안 온다"처럼 딱 잘라 말하는 예보이고, SE는 "강수확률 67%"처럼 확률로 말하는 예보다. HE는 다루기 쉽지만 정보 손실이 크고, SE는 정보가 풍부하지만 그만큼 잡음(rollout 개수가 적으면 추정치가 불안정함)에도 민감하다.

## 토이 예제: 두 단계를 숫자로 비교하기

논문의 실제 예시보다 rollout 수를 늘려 직접 계산해보자. $$N=8$$로 설정한다.

**단계 $$s_i$$**: 이 단계에서 8번 rollout을 돌렸더니 3번이 정답에 도달했다.

- $$y_{s_i}^{HE} = 1$$ (정답에 도달한 rollout이 하나라도 있으므로)
- $$y_{s_i}^{SE} = 3/8 = 0.375$$

**다음 단계 $$s_{i+1}$$**: 같은 문제에서 한 단계 더 진행한 부분 풀이로 다시 8번 rollout을 돌렸더니 이번엔 6번이 정답에 도달했다.

- $$y_{s_{i+1}}^{HE} = 1$$
- $$y_{s_{i+1}}^{SE} = 6/8 = 0.75$$

이제 두 단계를 상대 비교해보자. **HE 기준으로는 두 단계가 완전히 동일하다.** 둘 다 라벨이 1이라서, $$s_i$$에서 $$s_{i+1}$$로 넘어가며 무슨 일이 있었는지 HE는 아무것도 말해주지 않는다. 반면 **SE 기준으로는 0.375에서 0.75로 정확히 2배 뛰었다.** 이 숫자는 "$$s_i$$와 $$s_{i+1}$$ 사이에서 일어난 추론이 모델의 성공 확률을 실질적으로 끌어올렸다"는 신호를 그대로 담고 있다. 즉 SE는 단계 사이의 **가치 증가분**을 정량화하지만, HE는 그 정보를 0/1로 뭉개버린다. 반대로 만약 $$s_i'$$라는 대안 단계에서 8번 중 0번만 정답에 도달했다면 $$y^{HE}=0$$, $$y^{SE}=0$$으로 둘 다 "막다른 길"이라는 데 동의한다 — 극단적인 경우에는 두 방식이 일치하고, 중간 정도의 성공률에서만 정보량 차이가 벌어지는 것이다.

## PRM 학습

이렇게 만든 라벨로 PRM $$r_\theta$$를 이진 분류기로 학습한다. 풀이 하나에 $$K$$개 단계가 있다면 손실은 각 단계의 binary cross-entropy 합이다.

$$\mathcal{L}_{PRM} = -\sum_{i=1}^{K} \Big[ y_{s_i} \log r_{s_i} + (1-y_{s_i}) \log(1-r_{s_i}) \Big]$$

- $$r_{s_i}$$: PRM이 단계 $$s_i$$에 매긴 예측 점수 (0~1).
- $$y_{s_i}$$: 방금 정의한 HE 또는 SE 라벨.

논문은 실제 학습에서 SE보다 **HE를 기본값으로 채택**한다. 이유는 실용적이다 — HE는 0/1 라벨이라 "가능성 있음/없음"을 뜻하는 두 개의 특수 토큰만 추가하면 기존 언어모델링 파이프라인을 그대로 재사용할 수 있고, 회귀(regression) 헤드 같은 별도 구조 변경이 필요 없다. 학습 데이터는 completer로 LLemma-7B를 써서 $$N=8$$개씩 rollout을 생성했고, 이 과정으로 GSM8K에서 약 17만 개, MATH에서 약 27만 개의 풀이(각각 단계별 라벨 포함)를 확보했다. 사람이 한 줄씩 읽고 판단하는 대신, GPU가 rollout을 돌리는 것으로 라벨링이 완전히 자동화된 것이다.

## PRM의 두 가지 용도

Math-Shepherd라는 이름 자체가 이 이중 역할을 암시한다 — 양치기(shepherd)가 양떼를 감시(verify)하기도 하고 몰기(reinforce)도 하듯, 이 PRM도 두 가지로 쓰인다.

**(a) Verifier로 best-of-N 재랭킹.** 정책 모델이 후보 풀이 $$N$$개를 생성하면, 각 풀이 $$S$$의 점수를 그 풀이를 구성하는 단계 점수들의 **최솟값**으로 정의한다.

$$r_S = \min_{1 \le i \le K} r_{s_i}$$

한 단계라도 PRM이 낮은 점수를 준 곳이 있으면 그 풀이 전체가 낮은 점수를 받는다 — "약한 고리가 전체를 결정한다"는 논리로, HH-RLHF 데이터셋 리뷰([앞선 시리즈](/blog/2026/hh-rlhf-red-team/) 논의와 같은 발상)에서 본 최솟값 집계와 동일한 직관이다. 이렇게 얻은 $$r_S$$는 self-consistency(다수결)와 결합해 가중 투표로도 쓸 수 있다.

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{N} \mathbb{1}(a_i = a) \cdot r_{S_i}$$

단순히 표를 세는 게 아니라, PRM이 신뢰도 높다고 판단한 풀이의 표에 가중치를 더 주는 방식이다.

**(b) RL의 reward로 직접 사용 — step-by-step PPO.** 이 시리즈의 관심사와 가장 직결되는 부분이다. 기존 ORM-PPO는 [14편 PPO](/blog/2026/ppo/)의 표준 레시피대로 응답이 끝난 마지막 토큰에서만 보상을 준다. Math-Shepherd의 step-by-step PPO는 **각 추론 단계가 끝날 때마다** PRM이 매긴 점수를 보상으로 얹는다. 정책이 학습 도중 매 단계 "이 방향이 맞는 방향인가"라는 조밀한(dense) 피드백을 받는 셈이다 — 논문 마지막 단계에서만 보상을 받는 ORM-PPO보다 학습 신호가 훨씬 촘촘하다.

# Experiments

## Verifier로서: best-of-N 재랭킹

DeepSeek-67B를 생성 모델로 놓고 256개의 후보 풀이를 만든 뒤, 단순 self-consistency와 Math-Shepherd 검증을 비교한다.

| 방법                            | GSM8K | MATH(500) |
| ------------------------------- | ----- | --------- |
| Self-Consistency (다수결)       | 88.2% | 45.4%     |
| Math-Shepherd 검증 (PRM 재랭킹) | 93.3% | 47.0%     |

같은 후보군에서 답을 고르는 방식만 바꿨는데 GSM8K에서 5.1%p, MATH에서 1.6%p가 오른다. 후보는 그대로고 "고르는 눈"만 좋아진 결과다.

Fine-tuning된 여러 backbone에 검증을 얹었을 때의 효과도 확인할 수 있다.

<p align="center"><img src="/assets/post/image/math-shepherd/fig1-performance.png" width="55%"></p>

| Backbone (fine-tune 방식) | Fine-tuned 단독 | +Math-Shepherd 검증 |
| ------------------------- | --------------- | ------------------- |
| LLaMA2-70B (MAmmoTH)      | 72.4%           | —                   |
| LLaMA2-70B (WizardMATH)   | 81.6%           | —                   |
| LLaMA2-70B (MetaMATH)     | 80.4%           | 93.2%               |
| Llemma-34B (MetaMATH)     | 75.8%           | 90.9%               |
| DeepSeek-67B (MetaMATH)   | 82.8%           | 93.3%               |
| 참고: GPT-4 (early)       | 92.0%           | —                   |
| 참고: GPT-4-0613          | 94.4%           | —                   |

(GSM8K 정확도 기준.) MetaMATH 계열은 검증을 얹으면 하나같이 GPT-4 early 수준(92.0%)을 넘어선다. 오픈소스 7B급 검증기 하나가 재랭킹만으로 GPT-4에 근접한 성능을 뽑아낸다는 뜻이다.

## RL의 reward로서: step-by-step PPO

Mistral-7B(MetaMATH로 초기 fine-tune)를 정책으로 두고, ORM 기반 PPO와 Math-Shepherd 기반 step-by-step PPO를 비교한다.

| 방법                           | GSM8K | MATH  |
| ------------------------------ | ----- | ----- |
| 초기 모델 (MetaMATH, RL 이전)  | 77.9% | 28.6% |
| ORM-PPO                        | 81.8% | 31.3% |
| Math-Shepherd step-by-step PPO | 84.1% | 33.0% |
| + Math-Shepherd 검증까지 결합  | 89.1% | 43.5% |

같은 PPO인데 보상을 마지막 한 번 대신 단계마다 주는 것만으로 ORM-PPO 대비 GSM8K에서 2.3%p, MATH에서 1.7%p를 더 얻는다. 그리고 이렇게 학습한 정책의 출력을 다시 같은 PRM으로 검증까지 하면 MATH 점수가 28.6% → 43.5%로, 처음보다 15%p 가까이 뛴다. RL(정책 개선)과 verifier(추론 시점 재랭킹)가 같은 PRM 하나로 서로 다른 두 층위에서 겹쳐 작동하는 셈이다.

## PRM 평가는 왜 어려운가

지금까지 본 숫자만 보면 Math-Shepherd의 PRM은 흠잡을 데 없어 보인다. 하지만 "best-of-N 재랭킹을 잘한다"와 "각 단계가 논리적으로 옳은지 정확히 판별한다"는 서로 다른 질문이다. 이 간극을 정면으로 파고든 두 후속 벤치마크가 있다.

**ProcessBench**([Zheng et al., Qwen Team/Alibaba, arXiv 2024](https://arxiv.org/abs/2412.06559))는 과제를 아주 명확하게 좁힌다 — 풀이가 주어졌을 때 **가장 먼저 틀린 단계의 번호**를 정확히 짚거나, 전부 맞다면 "오류 없음"을 답하게 한다. GSM8K, MATH, OlympiadBench, Omni-MATH 네 난이도 구간에서 총 3,400개의 사람 검증 테스트 케이스로 구성된다.

| 모델                    | 유형                    | 평균 F1 (4개 subset) |
| ----------------------- | ----------------------- | -------------------- |
| Math-Shepherd-PRM-7B    | PRM (rollout 라벨 학습) | 31.5                 |
| Qwen2.5-Math-7B-PRM800K | PRM (사람 라벨 학습)    | 56.5                 |
| GPT-4o                  | critic (프롬프트 기반)  | 61.9                 |
| QwQ-32B-Preview         | critic (추론 모델)      | 71.5                 |
| o1-mini                 | critic (추론 모델)      | 87.9                 |

Math-Shepherd의 PRM은 "오류가 있는 첫 단계 찾기"에서 사람 라벨로 학습한 PRM800K 계열의 절반 수준(31.5 vs 56.5)에 그친다. ProcessBench가 지목하는 원인은 정확히 이 글에서 정의한 라벨링 방식 자체다 — 어려운 문제일수록 **틀린 추론을 하고도 우연히 정답에 도달하는 rollout이 늘어난다.** 즉 $$y_{s_i}^{SE}$$나 $$y_{s_i}^{HE}$$가 높게 나와도, 그게 그 단계가 실제로 옳아서가 아니라 뒤에서 운 좋게 답이 맞아떨어졌기 때문일 수 있다는 것이다. 여기에 더해 rollout을 만드는 completer 모델 자체의 버릇에 라벨이 종속되는 "on-policy 의존성" 문제도 지적된다 — 라벨을 만든 모델과 다른 모델의 풀이를 채점할 때 일반화가 잘 안 된다.

**PRMBench**([Song et al., ACL 2025](https://arxiv.org/abs/2501.03124))는 관점을 더 세분화한다. 6,216개 문제와 83,456개 단계 라벨로, PRM의 능력을 **단순성(Simplicity)**, **건전성(Soundness)** — 비중복성·비순환논리·경험적 타당성·단계 일관성·영역 일관성·신뢰도 불변성 — , **민감도(Sensitivity)** — 전제 민감성·기만 저항성·다중 풀이 일관성 — 세 축, 아홉 개 세부 오류 유형으로 쪼개 채점한다.

| 축                | Math-Shepherd-Mistral-7B 점수 |
| ----------------- | ----------------------------- |
| Simplicity        | 47.1                          |
| Soundness         | 45.7                          |
| Sensitivity       | 60.7                          |
| 종합 PRMScore     | 47.0                          |
| Random 베이스라인 | 약 50.0                       |

종합 점수 47.0은 동전 던지기(약 50.0)와 사실상 구별되지 않는다. 그런데 이 47.0을 받은 모델이 바로 앞 절 표에서 GSM8K 검증 정확도 93%대를 찍은 그 Math-Shepherd다. 같은 PRM이 "여러 후보 중 더 나은 것 고르기"에는 강하면서, "이 한 단계가 정확히 왜 틀렸는가"에는 거의 무작위 수준이라는 뜻이다. best-of-N은 여러 후보의 상대적 순위만 맞으면 충분하지만, 세밀한 오류 판별은 절대적인 정오 판단을 요구한다 — Math-Shepherd의 rollout 라벨은 전자에는 유효한 근사지만 후자에는 그렇지 못하다.

# Conclusion

Math-Shepherd의 핵심은 한 줄로 요약된다. **"좋은 단계 = 정답으로 이어질 잠재력이 큰 단계"로 재정의하고, 그 잠재력을 rollout으로 몬테카를로 추정하면 사람 라벨 없이도 PRM을 학습시킬 수 있다.** 그리고 이 PRM은 verifier로도, PPO의 reward로도 모두 통한다.

정리하면,

1. **방법**: 부분 풀이 $$s_{\le i}$$에서 $$N$$개 rollout을 굴려, 정답 도달 여부(HE)나 비율(SE)로 단계 라벨을 자동 생성한다. 학습은 HE 기반 이진 분류로 단순화한다.
2. **결과**: DeepSeek-67B 검증에서 GSM8K 88.2% → 93.3%, Mistral-7B RL+검증에서 MATH 28.6% → 43.5%까지 끌어올렸다.
3. **한계**: 논문 스스로도 인정하듯 "완성 과정이 많은 컴퓨팅 자원을 요구한다"는 rollout 비용 문제가 있고, 더 근본적으로는 **"정답으로 이어졌다"와 "그 단계가 논리적으로 옳다"가 같지 않다**는 간극이 있다. ProcessBench와 PRMBench는 이 간극이 추상적인 우려가 아니라 실제로 측정 가능한 성능 격차(F1 31.5, PRMScore 47.0)임을 보여줬다.

이 한계는 그냥 넘어갈 문제가 아니다. rollout 기반이든 사람 라벨 기반이든, 신경망 PRM은 결국 근사치이고 hacking될 여지를 남긴다. [21편 DeepSeek-R1](/blog/2026/deepseek-r1/)은 이 문제를 아예 정면으로 받아들여 — 일반적인 추론 과제에서 단계를 세밀하게 정의하기 어렵다는 점, 중간 단계의 정오 판단 자체가 어렵다는 점, 그리고 신경망 reward model은 대규모 RL에서 결국 reward hacking에 취약하다는 점을 이유로 — **PRM을 포함한 신경망 reward model 전체를 버리고 규칙 기반 reward로 돌아서는** 선택을 한다. Math-Shepherd가 "사람 라벨을 없앴다"면, DeepSeek-R1은 한 발 더 나아가 "PRM 자체를 없앤다." 그 이야기는 다음 글에서 이어간다.

---

# RLHF Reward 설계 시리즈

이 글은 RLHF Reward 설계 시리즈의 스무 번째 글이다.

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
20. **(현재 글)** Math-Shepherd (2023) — 사람 라벨 없는 PRM
21. [DeepSeek-R1 (2025)](/blog/2026/deepseek-r1/) — RLVR, 규칙이 reward가 될 때

**6부. Generative Reward Model**

22. [Prometheus 2 (2024)](/blog/2026/prometheus-2/) — 오픈 평가자 모델과 rubric 조건부 평가
23. [Generative Verifiers (2024)](/blog/2026/generative-verifiers/) — reward를 next-token prediction으로
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

32. [프론티어 모델의 reward 설계 (2024~2026)](/blog/2026/frontier-reward-design/) — DeepSeek·Qwen·Llama·Kimi·Solar가 실제로 택한 것

본 시리즈는 32편으로 구성된다.

# 참고 문헌

- Wang et al., 2023/2024. [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935). ACL 2024.
- Wang et al., 2024. [Math-Shepherd (ACL Anthology)](https://aclanthology.org/2024.acl-long.510/).
- Zheng et al., 2024. [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/abs/2412.06559).
- Song et al., 2025. [PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](https://arxiv.org/abs/2501.03124). ACL 2025.
- Lightman et al., 2023. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — PRM800K, 이 시리즈 19편에서 다룬 논문.
- [OpenAI PRM800K GitHub](https://github.com/openai/prm800k) — 80만 건 step-level 라벨 데이터셋.
- Guo et al., 2025. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 21편에서 다룰 PRM 포기 논거의 출처.
