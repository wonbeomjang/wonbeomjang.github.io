---
layout: post
title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
date: 2024-12-29 00:00:00 +0900
description: LoRA와 Full Fine-tuning의 차이점 분석 논문 리뷰
categories: [llm]
tags: [paper, llm]
giscus_comments: true
related_posts: true
featured: true
---

# Introduction

Pre-trained 모델을 downstream task에 finetuning하는 것은 computation-, data-efficient한 방법이다. 하지만 full-finetuning은 시간과 비용적으로 부담이 크다. 이를 해결하기 위해 LoRA와 같은 PEFT 방법이 제시되었다. 그러나 LoRA로 full-finetuning과 동일한 성능을 내도록 학습했을 때, 두 방법이 실제로 동일하게 작동하는지는 명확하지 않다. 저자는 이러한 의문을 실험을 통해 분석하며 다음과 같은 결론을 얻었다.

1. LoRA는 intruder dimensions을 도입하며 full-finetuning과 구조적으로 다른 학습을 진행한다.
2. LoRA로 fine-tuned된 모델은 intruder dimensions을 만들며 pre-training distribution에서 더 많이 벗어나고, continual pre-training에서도 덜 robust하다.
3. Low-rank LoRA가 target task에서 잘 작동하더라도 higher-rank parameterization이 더 좋은 결과를 낸다.

# Model Differences Between LoRA and Full Fine-Tuning

저자는 Sharma et al. (2024)의 Singular Value Decomposition (SVD)을 활용한 pruning에서 영감을 얻었다. 이를 통해 LoRA fine-tuned 모델과 full fine-tuned 모델의 weight matrices의 singular vector와 pre-trained weight의 singular vector의 cosine similarity를 비교했다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image.png" width="80%"></p>

Fig. 2(b)에서 볼 수 있듯 LoRA와 full finetuning의 singular vector는 다른 양상을 보인다. LoRA fine-tuned singular vector는 full fine-tuned singular vector에 비해 pre-trained singular vector와 cosine similarity가 낮았다. 그리고 LoRA rank가 높을수록 singular vector가 pre-trained singular vector의 cosine similarity가 높았다.

저자는 이렇게 cosine similarity가 낮은 singular vector를 intruder dimension이라고 명명한다.

$$\text{Definition 1: A singular vector } y_i \text{ from the fine-tuned weight matrix } W_{\text{tuned}} \text{ is an intruder dimension if and only if } \text{max}_i(\cos(y_j,x_i)) < \epsilon, \text{ where } \epsilon \text{ is a similarity threshold and } x_i \text{ is a singular vector in } W_0.$$

Full fine-tuning에서는 pre-trained singular vector와 높은 cosine similarity를 가지는 singular vector가 비슷한 singular value를 가지고 있다. 이는 full fine-tuning이 pre-trained singular vector와 singular value를 활용해 small update를 진행한다는 것을 보여준다. 반면, LoRA는 새로운 singular vector를 도입해 large norm으로 update를 한다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%201.png" width="80%"></p>

Fig. 3에서 empty column은 intruder dimensions을 나타내며, 이는 full fine-tuning과의 차이를 보여준다.

## Setup

저자는 RoBERTa-base로 실험을 진행했다.

### 1. LoRA finetuned model은 high-ranking intruder dimensions을 가지지만 fully fine-tuned model은 그렇지 않다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%202.png" width="80%"></p>

위 알고리즘에 따라, top-k highest ranking singular vector에 대해 모든 pre-trained singular vector와 maximum cosine similarity를 측정했을 때 threshold $$\epsilon$$ 이하면 intruder dimension으로 분류한다. 저자는 LoRA로 학습한 모델들이 작은 $$\epsilon$$ 에 대해 $$r \leq 16$$일 때 지속적으로 intruder dimension을 가진다는 것을 확인했다. 또한 rank가 올라갈수록 intruder dimension이 줄어드는 것을 확인했다.

### 2. LoRA fine-tuned model이 full fine-tuned model보다 학습량이 적은 작업에서도 intruder dimensions이 존재한다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%203.png" width="80%"></p>

수학, 코딩과 같은 학습량이 적은 데이터로 fine-tuning할 경우에도 intruder dimensions이 발생한다. Magicoder 모델과 같은 code model에서도 intruder dimension이 나타나는데, 이는 pre-training domain과 code domain의 차이에서 비롯된 것으로 판단된다. 이 경우에도 LoRA fine-tuned model이 intruder dimensions을 더 많이 가진다.

### 3. Full fine-tuning updates는 LoRA update보다 higher effective rank를 가진다. (Full rank일지라도)

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%204.png" width="80%"></p>

Full fine-tuning은 LoRA tuning보다 더 높은 effective rank를 가진다. 예를 들어 $$r=768$$인 RoBERTa도 평균적으로 effective rank를 300으로 업데이트한다. 이는 LoRA가 full capacity $$r$$을 사용하지 못하고 업데이트를 진행한다는 것을 의미한다. 따라서 LoRA와 full fine-tuning의 차이는 coding과 같은 어려운 작업에서 더 두드러진다.

### 4. Intruder dimension은 high and low singular values 모두에 존재한다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%205.png" width="80%"></p>

Fig. 11a에서 볼 수 있듯, singular value의 비율과 관계없이 full fine-tuning보다 항상 더 많은 intruder dimensions을 가진다.

### 5. Scaling $$\alpha$$를 LoRA의 rank에 따라 조절하면 intruder dimensions이 줄어들고 effective rank이 늘어난다.

많은 논문에서 $$\alpha=2r$$로 설정하여 학습을 진행한다. 저자는 $$\alpha$$의 영향을 확인하기 위해 $$\alpha=2r$$과 $$\alpha=8$$로 설정해 비교 실험을 진행했다. 고정된 $$\alpha$$ 값에서는 모든 rank에서 LoRA가 intruder dimensions을 보였으며, $$\alpha=2r$$과 비교했을 때 훨씬 적은 effective rank를 가졌다.

### 6. Intruder dimensions의 수는 fine-tuning dataset의 크기에 비례하여 늘어난다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%206.png" width="80%"></p>

Fig. 12에서 $$r=8$$인 경우 하나 이상의 데이터셋을 학습시킬 때 intruder dimensions이 추가로 발생한다. 반면 $$r=1$$인 경우 intruder dimensions 수가 비슷한데, 이는 $$r=1$$일 때 모델의 표현력 한계 때문으로 추정된다.

### Conjecture: Intruder dimensions은 norm과 stability에 큰 영향을 끼친다.

Pre-trained singular vector와 다르게, LoRA는 intruder dimensions을 추가하면서 smaller dataset에 fine-tuning하므로 pre-trained vectors보다 큰 영향을 미친다. 반면 full fine-tuning은 pre-trained 모델의 spectral property를 유지하며 효과적으로 적응한다. 이를 통해 LoRA 모델은 fine-tuning task 이외의 분야에서 부정적 영향을 미치고, full fine-tuning 모델은 이러한 악영향이 적음을 확인할 수 있다.

# Behavioral Differences Between LoRA and Full Fine-Tuning

### 1. Lower rank에서 LoRA는 continual learning에 robust하지 않고 이전 작업을 더 많이 잊어버린다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%207.png" width="80%"></p>

Fig. 8에서 볼 수 있듯 LoRA는 target task에 대해 full fine-tuning과 비슷한 성능을 내지만 rank가 작을 경우 continual pre-training 성능이 낮다. Rank를 올리면 forgetting이 줄어들며, full fine-tuning이 가장 낮은 forgetting 비율을 가진다.

### 2. 같은 test accuracy로 fine-tuning했을 때 pre-training pseudo loss가 U-shaped curve를 그리는 것을 확인할 수 있다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%208.png" width="80%"></p>

이는 downstream task에 대해 optimal한 rank가 존재한다는 것을 보여준다. Rank가 낮을 경우 intruder dimensions의 영향으로 forgetting이 발생하며, rank가 높을 경우 overparameterization으로 인해 target task에 대해 overfitting이 일어난다.

### 3. $$\alpha$$를 적절하게 설정하면 model performance에 긍정적인 영향을 미친다.

LoRA는 rank와 관계없이 forgetting 현상이 발생한다. 그러나 $$\alpha=8$$이 $$\alpha=2r$$보다 intruder dimensions이 더 많음에도 불구하고 높은 $$\alpha$$는 continual pre-training에서 성능 향상에 기여한다.

# 왜 Intruder Dimensions이 존재할까?

### 1. Pre-trained matrix에 random vector를 더하면 intruder dimensions이 생긴다.

이를 확인하기 위해 pre-trained weights $$W \in \mathbb{R}^{n \times n}$$, randomly sampled vector $$v \in \mathbb{R}^n$$, $$W$$의 singular value보다 큰 $$\lambda$$에 대해 $$\text{SVD}(W + \lambda vv^{T})$$와 $$\text{SVD}(W)$$를 비교했다. 이를 통해 random vector 추가가 intruder dimensions 생성에 영향을 미친다는 것을 확인했다.

### 2. Update rule에서의 차이점

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%209.png" width="80%"></p>

LoRA는 더 큰 learning rate를 사용하며 low-rank space에서 gradient projection을 진행한다. 이러한 방식은 full fine-tuning과 차이를 보이며 intruder dimensions 생성에 영향을 준다.

### 3. Product parameterization of LoRA

Matrices 곱은 spectral differences를 증가시키며 lower effective rank를 초래한다.

<p align="center"><img src="/assets/post/image/2024-12-29-lora-vs-full-fine-tuning/image%2010.png" width="80%"></p>

따라서 LoRA adaptor에서 B만 학습시키는 것이 intruder dimensions이 더 적다는 것을 알 수 있다.

# Conclusion

저자는 LoRA와 full fine-tuning이 weight metrics의 spectral properties에서 큰 차이가 있음을 확인했다. LoRA 모델은 intruder dimensions을 자주 가지며, pre-trained singular vectors와 거의 orthogonal하다. 반면, full fine-tuning은 pre-trained spectral properties를 유지하며 효과적으로 적응한다. LoRA는 일부 task에서는 효과적일 수 있지만, pre-training domain과 다른 domain에 대해 더 많은 한계를 가지며, 이러한 한계는 continual learning에서도 명확히 드러난다.
