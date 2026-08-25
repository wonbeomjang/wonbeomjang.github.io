---
layout: post
title: "Deep RL from Human Preferences: 보상 함수를 사람의 선호로 배우다"
date: 2026-08-11 09:01:00 +0900
description: "RL Reward 설계 시리즈 #1 — 선호 비교만으로 reward model을 학습하는 원형 (Christiano et al., OpenAI/DeepMind, NeurIPS 2017)"
categories: [paper]
tags: [rlhf, reward-model, rl, alignment, paper]
giscus_comments: true
related_posts: true
---

> [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., OpenAI/DeepMind, NeurIPS 2017)

# Introduction

로봇에게 "테이블을 치워라" 혹은 "계란을 스크램블해라"를 시키고 싶다고 하자. 강화학습으로 풀려면 보상 함수 $$r(o, a)$$가 있어야 하는데, 이걸 로봇의 센서 값(카메라 픽셀, 관절 각도)만으로 어떻게 수식으로 적을 수 있을까? "계란이 완전히 뒤집혔다"를 픽셀 값의 함수로 짜는 것부터가 막막하다. 대충 비슷한 근사 보상을 손으로 짜면, 에이전트는 그 근사식을 곧이곧대로 최적화해서 원래 의도와는 다른 행동으로 새버린다. 이게 이 논문이 1절에서 던지는 문제, **reward specification 문제**다.

이 논문의 해결책은 단순하다. 보상 함수를 사람이 직접 설계하지 않고, **사람에게 두 개의 짧은 행동 영상을 보여주고 어느 쪽이 더 나은지 물어서** 그 비교 데이터로부터 보상 함수를 배운다. 그리고 그렇게 배운 보상 함수로 정책을 강화학습으로 최적화한다. 저자들은 이 방법으로 사람이 시연조차 할 수 없는 태스크 — 예를 들어 관절이 여러 개인 비인간형 로봇에게 **백플립을 시키는 것** — 를 단 900번의 비교 질문, 한 시간이 채 안 되는 사람 노동으로 학습시켰다.

이 구조에서 눈여겨봐야 할 선택이 세 가지 있다.

1. **점수(scalar)가 아니라 비교(pairwise)를 묻는다.** 사람은 "이 클립에 8.3점을 줘라"보다 "이 클립이 저 클립보다 낫다"를 훨씬 일관되게 답한다. 이 선택 하나가 이후 모든 RLHF 파이프라인의 라벨링 방식을 결정했다.
2. **정책 학습과 보상 모델 학습이 동시에, 비동기로 돈다.** 정책이 새로운 궤적을 만들고, 그 궤적 일부가 사람에게 비교 질의로 나가고, 그 라벨이 다시 보상 모델을 갱신하고, 갱신된 보상 모델이 다시 정책을 학습시킨다. 이 루프를 끊고 보상 모델을 한 번만 오프라인으로 학습시키면 무슨 일이 벌어지는지, 이 논문은 정확히 보여준다 — 그리고 그게 **reward hacking의 첫 증거**다.
3. **사람 피드백은 전체 상호작용의 1% 미만만 쓴다.** Atari 환경에서 에이전트는 최대 5천만(5×10⁷) 프레임을 경험하는데, 사람이 라벨을 붙인 것은 5,500개의 1~2초짜리 클립뿐이다.

이 세 가지 선택은 5년 뒤 InstructGPT에서 "SFT → reward model → PPO" 3단계로 정식화된다. Conclusion에서 이 연결을 다시 짚는다.

# Background

## 왜 손으로 짠 보상이 위험한가

전통적인 심층 강화학습은 잘 정의된 보상 함수가 있는 도메인(Atari 점수, 바둑 승패)에서 성공했다. 문제는 실세계 태스크 대부분이 그런 명확한 보상이 없다는 점이다. 사람이 얼추 의도를 반영한 근사 보상을 손으로 짜면, 에이전트는 그 근사식의 허점을 정확히 찾아내 원래 의도와 다른 방식으로 점수를 올린다. 저자들은 이걸 최근 부상하던 AI 안전 논의(Bostrom 2014, Russell 2016, Amodei et al. 2016의 "Concrete Problems in AI Safety")와 직접 연결한다 — 값과 목적함수의 불일치(misalignment)라는 더 큰 문제의 축소판이라는 것이다.

기존에 있던 두 가지 대안도 각각 한계가 있다.

| 방식                                   | 사람이 제공하는 것          | 한계                                                                                     |
| -------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| 직접 설계한 reward function            | 없음 (엔지니어가 수식을 짬) | 의도와 실제 최적화 대상이 어긋나기 쉬움                                                  |
| Demonstration / Imitation / Inverse RL | 시연(demonstration)         | 사람이 직접 수행할 수 없는 태스크엔 못 씀 (예: 자유도가 많고 사람과 다른 몸을 가진 로봇) |
| TAMER류 rating (Knox 2012)             | 상태·행동에 대한 절대 점수  | 절대 점수는 스케일이 흔들리고 일관성 유지가 어려움. 사람 시간도 훨씬 더 필요             |
| **Pairwise preference (본 논문)**      | "이게 낫다" 상대 비교       | 비교만으로 학습 가능. 절대 점수보다 훨씬 일관됨                                          |

왜 비교가 절대 점수보다 나을까. 와인 감별사에게 "이 와인에 10점 만점에 몇 점을 줄 것인가"를 물으면 사람마다, 심지어 같은 사람도 시점마다 기준이 흔들린다. 반면 "이 와인과 저 와인 중 뭐가 낫냐"는 훨씬 안정적으로 답한다. 절대 척도를 유지하는 것 자체가 인지적으로 부담이 크기 때문이다. 이 논문도 3.3절 실험에서 "연속 제어 태스크에서 비교를 예측하는 쪽이 점수를 직접 회귀하는 것보다 훨씬 잘 됐다"고 명시한다 — 점수의 절대 스케일이 태스크마다 크게 달라 회귀 문제가 어려워지는데, 비교만 예측하면 이 문제가 사라진다.

## 궤적 세그먼트 — 사람에게 무엇을 보여줄 것인가

전통적 RL의 루프는 이렇다. 에이전트가 관측 $$o_t$$를 받고 행동 $$a_t$$를 내보내면, **환경이 곧바로 보상 $$r_t$$를 돌려준다.** 점수판이 있는 게임이라면 이 구조가 자연스럽다.

이 논문의 세팅에는 그 $$r_t$$가 **아예 없다.** 백플립을 시키고 싶은데 "백플립을 했다"를 관절 각도의 함수로 적을 수 없으니, 애초에 돌려줄 숫자가 없는 것이다. 그래서 환경이 앉아 있던 자리에 **사람**을 앉힌다.

그러면 사람에게 무엇을 물어볼 것인가. 여기서 두 가지 결정이 필요하다. 첫째는 앞 절에서 본 대로 점수가 아니라 **비교**를 묻는 것이다. 둘째는 **얼마나 긴 행동 덩어리를 보여줄 것인가**인데, 이 논문의 답이 **궤적 세그먼트(trajectory segment)**다. 세그먼트는 길이 $$k$$의 관측·행동 시퀀스다.

$$\sigma = ((o_0, a_0), (o_1, a_1), \ldots, (o_{k-1}, a_{k-1}))$$

그리고 $$\sigma^1 \succ \sigma^2$$는 "사람이 $$\sigma^1$$을 $$\sigma^2$$보다 선호했다"는 표기다.

왜 하필 "조각"일까. 양 극단이 둘 다 실패하기 때문이다.

| 보여주는 단위           | 문제                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| 한 프레임 ($$k=1$$)     | 정지 화면만으로는 움직임을 알 수 없다. 로봇이 넘어지는 중인지 일어서는 중인지 구분이 안 된다 |
| 에피소드 전체           | 30초를 보고 좋다/나쁘다 하나를 고르면 그 안의 어느 순간이 좋았는지 신호가 뭉개진다           |
| **세그먼트(짧은 클립)** | 움직임은 보이면서, 좋았던 구간이 어디인지도 특정된다                                         |

즉 세그먼트는 **"움직임을 알아볼 만큼은 길고, 어느 부분이 좋았는지 헷갈리지 않을 만큼은 짧은"** 중간 지점이다. 그 길이를 실제로 몇 초로 잡을지는 Method의 [클립 길이 절](#사람에게는-12초-클립을-보여준다)에서 실측으로 정한다.

## 두 종류의 평가 — 정답이 없는데 어떻게 채점하나

여기서 저자들은 곤란한 문제에 부딪힌다. **"사람 선호로 배운 보상이 제대로 배워졌다"를 어떻게 증명하나?** 애초에 참 보상 함수가 없어서 이 방법을 쓰는 건데, 없는 정답과 비교할 수는 없다. 채점표가 없는 것이다.

그래서 문제를 두 갈래로 쪼갠다.

**(a) Quantitative(정량) — 정답을 아는 문제를 모르는 척 풀린다**

Atari 점수나 MuJoCo 보행 속도처럼 **이미 진짜 보상 함수 $$r$$이 있는** 태스크를 일부러 고른다. 그리고 알고리즘에게는 그 $$r$$을 **숨긴다.** 사람 선호만 주고 학습시킨 뒤, 다 끝나면 숨겨뒀던 $$r$$을 꺼내 채점한다.

모의고사에 비유하면 이렇다. 답안지가 있는 시험지를 답안지 없이 풀게 하고, 제출한 뒤에 답안지를 꺼내 맞춰본다. **답안지는 채점에만 쓰고 학습에는 절대 넣지 않는다.** 이렇게 하면 "이 방법이 원리적으로 작동한다"를 숫자로 말할 수 있다.

**(b) Qualitative(정성) — 답안지가 원천적으로 없는 문제로 쓸모를 보인다**

백플립에는 보상 함수를 쓸 수 없으니 (a) 방식으로 채점할 수도 없다. 대신 "백플립을 해라"라는 자연어 목표를 주고, 결과 영상을 사람이 보고 **"이게 백플립인가?"** 를 판단한다. 점수는 없고 만족·불만족만 있다.

**왜 둘 다 필요한가.** 하나만 하면 각각 반박당하기 때문이다.

| 하나만 하면 | 나오는 반박                                                            |
| ----------- | ---------------------------------------------------------------------- |
| (a)만       | "보상 함수가 이미 있는 태스크에서만 되는 것 아닌가? 그럼 쓸 데가 없다" |
| (b)만       | "저자들이 만족스럽다고 하면 그게 증명인가? 검증이 안 된다"             |

(a)가 **방법의 타당성**을 숫자로 못 박고, (b)가 **실제 쓸모** — 보상을 못 짜는 태스크에서도 된다는 것 — 를 보여준다. 둘을 합쳐야 "원리도 맞고 쓸 데도 있다"가 성립한다.

참고로 이 "정답을 숨겨두고 채점한다"는 발상은 [#10 과최적화 스케일링 법칙](/blog/2026/reward-model-overoptimization/)에서 그대로 재활용된다. 거기서는 사람 대신 gold RM을 정답 자리에 앉히고, 그 라벨로 학습시킨 proxy RM이 언제 무너지는지를 잰다.

# Method

## 세 개의 프로세스가 비동기로 돈다

이 방법은 신경망을 두 개 들고 간다. 하나는 **정책** $$\pi: O \to A$$로, 관측을 받아 행동을 내놓는다. 다른 하나는 **보상 모델** $$\hat r: O \times A \to \mathbb{R}$$로, 관측과 행동을 받아 "이게 얼마나 좋은가"를 숫자로 내놓는다. 원래 환경이 하던 채점 역할을 이 두 번째 신경망이 대신하는 셈이다.

문제는 이 둘이 서로를 필요로 한다는 점이다. 정책은 점수를 매겨줄 보상 모델이 있어야 학습할 수 있고, 보상 모델은 채점할 행동을 만들어줄 정책이 있어야 학습할 수 있다. 그래서 저자들은 둘을 번갈아 학습시키는 대신, **세 개의 프로세스를 각자 다른 속도로 동시에 돌린다.**

| 프로세스           | 하는 일                                                                                                                             | 담당 알고리즘                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| (1) 정책 롤아웃    | $$\pi$$가 환경과 상호작용해 궤적 $$\{\tau^1, \ldots, \tau^i\}$$ 생성. $$\pi$$는 $$r_t = \hat r(o_t, a_t)$$의 합을 최대화하도록 갱신 | Atari는 A2C, MuJoCo는 TRPO              |
| (2) 라벨 수집      | (1)이 만든 궤적에서 세그먼트 쌍 $$(\sigma^1, \sigma^2)$$을 골라 사람에게 질의                                                       | 앙상블 불일치 기반 uncertainty sampling |
| (3) 보상 모델 학습 | 지금까지 쌓인 비교 데이터베이스 $$D$$로 $$\hat r$$을 지도학습으로 갱신                                                              | Bradley-Terry cross-entropy             |

세 프로세스는 고리로 이어져 있다. (1)이 만든 궤적이 (2)로 가고, (2)가 받아온 사람 라벨이 (3)으로 가고, (3)이 갱신한 $$\hat r$$이 다시 (1)로 돌아간다. 그리고 이 고리는 한 바퀴씩 끊어 도는 게 아니라 **세 지점이 동시에 계속 돈다.**

왜 동시여야 하는지가 이 절의 핵심이다. 정책은 학습하면서 점점 **전에 안 가보던 상태**로 넘어간다. 처음엔 바닥에서 버둥거리던 로봇이 나중엔 뛰기 시작하는 식이다. 그런데 보상 모델이 초반의 "버둥거리는 로봇" 데이터로만 학습된 채 멈춰 있으면, 뛰는 로봇을 채점할 능력이 없다. 채점 기준이 낡아버리는 것이다.

라벨 수집을 계속 돌리면 이 문제가 완화된다. 정책이 새 영역으로 가면 그 새 영역의 클립이 사람에게 질의로 나가고, 보상 모델이 그걸 학습해서 따라간다. **정책이 앞서가는 만큼 채점 기준도 같이 따라가게 만드는 구조**다.

이 문제에는 이름이 있다 — **distribution shift**다. 이 글 뒤쪽의 [Pong 사례](#distribution-shift와-reward-hacking의-첫-증거)에서 이 고리를 끊으면 실제로 무슨 일이 벌어지는지 보게 되고, 시리즈 내내 형태를 바꿔가며 다시 나온다.

## 정책 최적화: 흔들리는 목표를 쫓기

$$\hat r$$로 보상을 계산하고 나면 남은 건 평범한 RL 문제다. 다만 $$\hat r$$이 계속 갱신되므로 **비정상(non-stationary) 보상**이라는 게 특이점이다. 이 때문에 저자들은 정책 그래디언트 계열(A2C, TRPO)을 골랐고, TRPO에서 유일하게 손댄 하이퍼파라미터가 엔트로피 보너스다. TRPO는 신뢰 영역(trust region)에 의존해 탐색을 확보하는데, 보상이 계속 바뀌면 이 신뢰 영역만으로는 탐색이 부족해질 수 있어서다. 또한 $$\hat r$$이 출력하는 보상은 평균 0, 표준편차 일정하게 정규화해서 쓴다 — 보상의 절대 위치 자체가 학습 문제상 정해져 있지 않기 때문이다.

## 사람에게는 1~2초 클립을 보여준다

사람 오버시어는 두 궤적 세그먼트를 짧은 영상(1~2초)으로 본다. 응답은 세 가지 — "1번이 낫다", "2번이 낫다", "둘 다 비슷하다/비교 불가"다. 이 판단은 삼중항(triple) $$(\sigma^1, \sigma^2, \mu)$$로 데이터베이스 $$D$$에 쌓인다. $$\mu$$는 $$\{1, 2\}$$ 위의 분포로, 한쪽을 선택하면 그쪽에 확률 1을 몰아주고, 동률이면 균등분포(0.5, 0.5)를 준다. 비교 불가로 표시된 경우는 아예 $$D$$에 넣지 않는다.

왜 클립이 1\~2초일까. 저자들은 단일 프레임(길이 1)으로 비교시키면 같은 성능을 내기 위해 훨씬 많은 비교가 필요함을 확인했다. 반대로 클립이 길수록 **클립당(per clip)** 정보량은 늘지만, 1\~2초 밑으로 짧게 자른다고 라벨링에 걸리는 시간이 별로 줄지 않아 **초당(per second) 정보량**은 오히려 나빠진다. 그래서 1~2초가 실측으로 찾은 균형점이다.

## Bradley-Terry 모델과 cross-entropy loss

$$\hat r$$을 궤적 세그먼트에 대한 잠재 보상으로 보고, 사람이 $$\sigma^1$$을 선호할 확률이 그 잠재 보상 합의 지수함수에 비례한다고 가정하면 다음 식이 나온다.

$$\hat P[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat r(o_t^1, a_t^1)}{\exp \sum \hat r(o_t^1, a_t^1) + \exp \sum \hat r(o_t^2, a_t^2)}$$

기호를 하나씩 풀면:

- $$\sum \hat r(o_t^1, a_t^1)$$: 세그먼트 $$\sigma^1$$을 이루는 각 스텝의 예측 보상을 전부 더한 값. "이 1~2초 동안 얼마나 좋았는가"의 총점이다.
- 분자: $$\sigma^1$$의 총점을 지수함수에 태운 값.
- 분모: 두 세그먼트 총점을 각각 지수함수에 태워 더한 값.

즉 두 총점의 차이가 클수록 $$\hat P$$는 0 또는 1에 가까워지고, 비슷하면 0.5 근처에 머문다. 이건 페어와이즈 비교로부터 점수 함수를 추정하는 **Bradley-Terry 모델**(1952)이자, Luce-Shephard 선택 규칙의 특수 케이스다. 이 식이 그대로 수년 뒤 RLHF 보상 모델(InstructGPT, HH-RLHF 등)의 손실 함수로 재사용된다.

이 예측과 실제 사람 라벨 사이의 cross-entropy를 최소화하도록 $$\hat r$$을 학습한다.

$$\text{loss}(\hat r) = -\sum_{(\sigma^1, \sigma^2, \mu) \in D} \Big( \mu(1) \log \hat P[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat P[\sigma^2 \succ \sigma^1] \Big)$$

$$\mu(1) = 1$$이면(사람이 1번을 선택) 손실은 $$-\log \hat P[\sigma^1 \succ \sigma^2]$$ 하나만 남는다 — 모델이 1번을 낮게 예측했을수록 손실이 커지고, 그래디언트는 $$\hat r$$이 $$\sigma^1$$에 더 높은 값을 주도록 밀어붙인다.

## 토이 예제: 손실이 어떻게 움직이는가

작은 숫자로 직접 계산해보자. 어떤 시점에 보상 모델이 두 세그먼트에 대해 $$\sum \hat r(\sigma^1) = 2.0$$, $$\sum \hat r(\sigma^2) = 0.5$$를 예측했다고 하자.

$$\hat P[\sigma^1 \succ \sigma^2] = \frac{e^{2.0}}{e^{2.0} + e^{0.5}} = \frac{7.39}{7.39 + 1.65} \approx 0.817$$

모델은 지금 "82% 확률로 사람이 1번을 고를 것"이라 예측한 상태다. 이제 실제 사람 라벨에 따라 손실이 어떻게 갈리는지 보자.

| 실제 사람 라벨               | 손실 계산                             | 손실 값 | 해석                                                                                                 |
| ---------------------------- | ------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| 1번 선호 ($$\mu(1)=1$$)      | $$-\log(0.817)$$                      | 약 0.20 | 모델 예측이 맞았다 → 작은 그래디언트                                                                 |
| 2번 선호 ($$\mu(2)=1$$)      | $$-\log(0.183)$$                      | 약 1.70 | 모델이 크게 틀렸다 → 큰 그래디언트, $$\hat r(\sigma^1)$$을 낮추고 $$\hat r(\sigma^2)$$를 올리는 방향 |
| 동률 ($$\mu(1)=\mu(2)=0.5$$) | $$0.5 \times 0.20 + 0.5 \times 1.70$$ | 약 0.95 | 중간 크기 손실                                                                                       |

모델이 확신을 가지고 틀렸을 때(2번째 행) 손실과 그래디언트가 가장 크다는 걸 숫자로 확인할 수 있다.

## 개선 사항 세 가지

기본 식만으로는 부족해서 저자들은 세 가지를 더했다.

1. **앙상블 + 부트스트랩**: 예측기 여러 개를 각각 $$D$$에서 복원추출한 $$\lvert D \rvert$$개 샘플로 학습시키고, 각각을 독립적으로 정규화한 뒤 평균낸다. 학습에 쓰인 부트스트랩 샘플에서 평균적으로 $$1/e \approx 36.8\%$$는 뽑히지 않는데, 이 남는 부분을 예측기별 검증 세트로 쓴다(전통적인 bagging의 out-of-bag 비율과 동일한 값이다). $$L_2$$ 정규화 계수는 검증 손실이 학습 손실의 1.1~1.5배가 되도록 조정한다.
2. **10% 무작위 응답 가정**: 순수 소프트맥스만 쓰면 보상 차이가 극단적으로 벌어질 때 확률이 0이나 1로 수렴하는데, 실제 사람은 아무리 명백한 비교에서도 일정 확률로 실수를 한다. 그래서 "사람이 10% 확률로 완전히 무작위로 답한다"는 가정을 섞어 극단 확신을 누그러뜨린다.
3. **불일치 기반 질의 선택**: 최근 궤적에서 세그먼트 쌍을 대량으로 뽑고, 앙상블의 각 예측기가 어느 쪽을 선호하는지 예측시킨 뒤, **예측기들끼리 의견이 가장 갈리는(분산이 큰)** 쌍만 사람에게 보낸다. 오디션 심사위원 세 명이 만장일치면 재심사가 필요 없고, 의견이 갈리는 참가자만 다시 불러 세밀히 보는 것과 같은 원리다. 저자들 스스로 "정보가치 기대값(expected value of information)을 직접 쓰는 게 이상적이지만 이건 조악한 근사"라 인정하며, 실제로 일부 태스크에서는 무작위 질의보다 오히려 성능이 떨어지기도 했다.

## Distribution shift와 reward hacking의 첫 증거

세 프로세스를 비동기로 돌리는 이유가 여기서 드러난다. 만약 보상 모델을 학습 **초반에만** 모은 라벨로 딱 한 번 학습시키고(오프라인), 그 뒤로는 고정한 채 정책만 계속 학습시키면 어떻게 될까. 정책은 시간이 지나며 보상 모델이 한 번도 보지 못한 새로운 궤적 분포로 이동한다(distribution shift). 이 낯선 영역에서 보상 모델의 예측은 더 이상 신뢰할 수 없는데, 정책은 개의치 않고 그 부정확한 예측치를 계속 최대화한다.

저자들이 실제로 관찰한 사례가 Pong이다. 보상 모델을 오프라인으로 학습시켰더니, 에이전트는 **점수를 얻으려 하지 않고 점수를 잃지 않으려고만** 했다 — 랠리를 비정상적으로 길게 끄는 행동으로 수렴한 것이다. 참 보상(true reward) 기준으로는 명백히 이상한 행동인데, 고정된 $$\hat r$$ 입장에서는 최적이었던 셈이다. 이게 **학습된 보상을 정책이 과최적화하는 현상, 즉 reward hacking의 초기 사례**다. 온라인(비동기)으로 라벨을 계속 흘려 넣어 보상 모델이 정책을 따라가게 하는 것이 이 문제에 대한 이 논문의 유일한 방어선이었다.

# Experiments

## 실험 세팅

라벨링은 계약직 작업자(contractor)가 담당했다. 각 태스크에 대해 1\~2문장짜리 설명만 받고 수백\~수천 쌍의 세그먼트를 비교했다. 질의 하나당 평균 응답 시간은 3~5초였고, 실험 하나에 든 총 인간 시간은 **30분에서 5시간 사이**였다.

## MuJoCo: 700건으로 실제 보상에 근접, 1400건으로는 앞선다

<p align="center"><img src="/assets/post/image/deep-rl-human-preferences/main_mujoco_fig.png" width="95%"></p>

Walker, Hopper, Swimmer, Cheetah, Ant, Reacher, Double-pendulum, Pendulum 8개 태스크에서 실험했다.

| 조건                                  | 결과                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 사람 라벨 700건 안팎(그림 범례 750건) | 8개 태스크 전반에서 실제 보상으로 학습한 RL을 거의 따라잡음                                                              |
| 합성 라벨 1,400건                     | 실제 보상으로 직접 학습한 RL보다 **오히려 살짝 더 나은** 성능 — 배운 보상 함수가 더 매끄럽게 형태를 잡아준 덕            |
| 실제 사람 라벨 vs 합성 라벨           | 같은 라벨 수 기준 사람 라벨은 합성 라벨의 절반~동등 수준 효율                                                            |
| Ant 태스크                            | 사람 라벨이 합성 라벨을 **크게 앞섬** — 사람이 "로봇이 똑바로 서 있는" 자세를 선호했고, 이게 훌륭한 보상 셰이핑으로 작용 |

Ant의 경우가 흥미롭다. 손으로 짠 보상 함수에도 "똑바로 서 있으면 보너스"를 넣었지만 효과가 크지 않았는데, 사람이 비교로 암묵적으로 표현한 "직립"에 대한 선호가 오히려 더 유용한 셰이핑이 됐다. 사람이 자연어 보너스 항으로 명시하기 어려운 선호를 비교로는 쉽게 전달했다는 뜻이다.

## Atari: 게임마다 편차가 크다

<p align="center"><img src="/assets/post/image/deep-rl-human-preferences/main_atari_figure.png" width="90%"></p>

BeamRider, Breakout, Pong, Qbert, Seaquest, SpaceInvaders, Enduro 7개 게임, 사람 라벨 **5,500건**으로 학습시켰다.

| 게임                    | 결과                                                                                                                                                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BeamRider, Pong         | 합성 라벨은 3,300건만으로도 RL에 근접·필적                                                                                                                                                                                       |
| Seaquest, Qbert         | 합성 라벨로 결국 RL 수준에 도달(느리게)                                                                                                                                                                                          |
| Breakout, SpaceInvaders | 합성 라벨로도 RL을 못 따라잡지만 확실히 학습은 함 (Breakout 점수 20~50, SpaceInvaders 1단계 통과)                                                                                                                                |
| Qbert (사람 라벨)       | 1단계도 못 넘음 — 짧은 클립이 헷갈리고 평가하기 어려운 게임이라서                                                                                                                                                                |
| Enduro                  | 합성 라벨은 전혀 학습 못 함(무작위 탐색으로는 추월이 거의 안 일어남). **사람 라벨은 오히려 원 RL 베이스라인(A3C)보다 나은 성능**(DQN과 비슷한 수준) — 차를 추월하려는 아주 작은 진전에도 보상을 줘서 자연스러운 셰이핑 효과를 냄 |

정리하면 사람 라벨은 대부분의 게임에서 같은 개수의 합성 라벨과 비슷하거나 살짝 못한 성능을 냈지만, **합성 라벨보다 40% 적은 라벨로도 비슷한 성능**을 내는 경우가 흔했다. 사람의 판단이 단순 점수 차이보다 풍부한 정보를 담고 있다는 방증이다.

## 새로운 행동: 보상 함수가 아예 없던 태스크

<p align="center"><img src="/assets/post/image/deep-rl-human-preferences/flip.png" width="90%"></p>

위 그림은 Hopper 로봇이 실제로 배운 백플립 동작의 연속 프레임이다. 참 보상 함수가 아예 없는 상태에서, 오직 사람의 비교 라벨만으로 학습시킨 세 가지 사례를 소개한다.

| 행동                                  | 사용한 라벨 수                       | 소요 시간  |
| ------------------------------------- | ------------------------------------ | ---------- |
| Hopper 백플립 반복                    | 900건                                | 1시간 이내 |
| Half-Cheetah가 한 다리로 서서 전진    | 800건                                | 1시간 이내 |
| Enduro에서 다른 차와 나란히 주행 유지 | 약 1,300건 (+ 400만 프레임 상호작용) | —          |

900건, 800건이라는 숫자가 특히 인상적이다. 사람에게 "백플립을 시켜라"라는 지시를 코드로 옮기는 건 사실상 불가능에 가깝지만, "이 클립이 저 클립보다 백플립에 가깝다"는 판단을 900번 받는 건 한 사람이 커피 한 잔 마시는 시간에 끝나는 일이다.

## 라벨 예산을 숫자로 보면

Abstract가 던진 "<1% 라벨"이라는 주장을 실제 숫자로 감을 잡아보자. Atari 실험은 최대 $$5 \times 10^7$$ 프레임의 환경 상호작용을 쓰는데, 사람이 라벨을 붙인 건 5,500개의 1\~2초 클립뿐이다. 60fps 기준 클립 하나가 대략 60\~120프레임이라 치면 사람이 "본" 프레임 수는 대략 $$5{,}500 \times 100 \approx 5.5 \times 10^5$$ 프레임 — 전체 상호작용의 1%를 겨우 넘는 수준이다. Discussion에서 저자들은 이를 "상호작용 복잡도를 대략 3자릿수(3 orders of magnitude) 줄였다"고 요약한다. 손으로 보상 함수를 짜는 대신, 이 정도 예산으로 사람의 의도를 심층 RL에 주입할 수 있다는 게 이 논문의 핵심 주장이다.

## Ablation이 보여주는 것

| 제거한 요소                         | 결과                                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| 무작위 질의(random queries)         | 태스크에 따라 성능 저하 또는 무영향 — uncertainty sampling의 효과가 보편적이지 않음을 시사  |
| 앙상블 제거(no ensemble)            | 단일 예측기만 사용, 질의도 무작위로 전환                                                    |
| 온라인 질의 제거(no online queries) | 학습 초반에만 라벨 수집 → Pong에서 앞서 설명한 긴 랠리 이상 행동 발생                       |
| 정규화 제거(no regularization)      | $$L_2$$ 대신 dropout만 사용                                                                 |
| 세그먼트 제거(no segments)          | 로보틱스에서 길이 1짜리 스텝 단위로 비교                                                    |
| 비교 대신 절대값(target)            | 오라클이 세그먼트의 실제 총 보상을 알려주고 MSE로 학습 — 비교 기반과 우열이 태스크마다 갈림 |

이 표에서 가장 중요한 행은 "온라인 질의 제거"다. 보상 모델이 정책을 따라가지 못하면(distribution shift가 방치되면) 참 보상 기준으로는 이상한 행동이 최적으로 둔갑한다는 걸 이 ablation이 정량적으로 확인해준다.

# Conclusion

한 줄로 요약하면: **이 논문은 "사람에게 점수 대신 비교를 묻고, 그 비교로 보상 모델을 학습시키고, 그 보상 모델로 정책을 강화학습시키되, 이 세 과정을 비동기로 계속 맞물리게 돌린다"는 구조를 제시했고, 이 구조가 이후 모든 RLHF의 골격이 됐다.**

핵심을 정리하면,

1. **문제**: 보상 함수를 손으로 짜는 건 어렵고 위험하다(reward hacking). 시연도, 절대 점수도 대안으로는 부족하다.
2. **해법**: Bradley-Terry 형태의 선호 확률 모델 $$\hat P[\sigma^1 \succ \sigma^2]$$와 cross-entropy loss로 보상 모델을 학습한다. 앙상블·불일치 기반 질의 선택·10% 노이즈 가정으로 안정화한다.
3. **비용**: MuJoCo는 700건, Atari는 5,500건, 백플립 같은 새 행동은 900건 — 전체 상호작용의 1% 미만으로 충분했다.
4. **위험의 씨앗**: 보상 모델을 정책과 분리해 오프라인으로 고정하면 정책이 그 부정확함을 파고들어 참 보상 기준으로는 이상한 행동(Pong 무한 랠리)에 수렴한다. 이게 이후 시리즈 전체를 관통할 **reward overoptimization/hacking**의 첫 관찰이다.

이 구조는 5년 뒤 거의 그대로 LLM에 이식된다. InstructGPT(Ouyang et al., 2022)의 3단계 레시피 — SFT로 초기 정책을 만들고, 사람이 여러 응답 중 순위를 매긴 데이터로 정확히 이 논문과 같은 Bradley-Terry cross-entropy 손실로 reward model을 학습하고, 그 reward model로 PPO 정책을 최적화하는 구조 — 는 이 논문의 (1)(2)(3) 프로세스를 대화형 텍스트 생성에 맞게 재배치한 것에 가깝다. 다만 LLM에서는 온라인 3-프로세스 비동기 루프 대신 라운드를 나눠 반복하는 형태로 단순화됐고, distribution shift 문제는 KL 페널티로 다른 방식으로 억제한다. 이 시리즈의 다음 글들이 그 변형들을 하나씩 따라간다.

# 참고 문헌

- Christiano et al., 2017. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS 2017.
- [NeurIPS 2017 Proceedings: Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf) — 공식 게재본(저자 소속 확인).
- [ar5iv: Deep Reinforcement Learning from Human Preferences (HTML rendering)](https://ar5iv.labs.arxiv.org/html/1706.03741) — 본문 그림 원본.
- Bradley, R. A. and Terry, M. E., 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. (Bradley-Terry 모델 원 논문)
- Amodei et al., 2016. [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565). (reward specification 문제의 배경)
- Knox, W. B. and Stone, P., 2012. TAMER: Training an Agent Manually via Evaluative Reinforcement. (절대 점수 기반 대안 비교 대상)

---

# RL Reward 설계 시리즈

이 글은 RL Reward 설계 시리즈의 첫 번째 글이다.

**1부. 지형도**

<ol start="1">
  <li><strong>(현재 글)</strong> Deep RL from Human Preferences (Christiano 2017) — 선호로 보상을 배우는 원형</li>
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
  <li><a href="/blog/2026/tool-call-reward/">도구 호출을 어떻게 채점하나</a> — ToolRL·ToolRM</li>
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
