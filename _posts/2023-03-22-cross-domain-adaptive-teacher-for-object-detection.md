---
layout: post
title: "Cross-Domain Adaptive Teacher for Object Detection"
date: 2023-03-22 00:00:00 +0900
description: sota of cross domain DA in object detection
categories: [domain-adaptation, paper]
tags: [paper, object-detection, domain-adaptation]
giscus_comments: true
related_posts: true
---

# Intorduction

데이터셋 중에서 label이 있는 데이터도 있지만 label이 없는 데이터도 있다.
이 두개의 domain이 다를 경우 이를 다루는 것은 쉽지 않다.
이 논문은 object detection에서 source domain은 label이 있고 target domain은 label이 없는 상황에
semi-supervised로 doamain adaptation을 하는 방법을 다루고 있다.

# Adaptive teacher

논문에서 semi-supervised learning을 위하여 teacher-student 모델을 사용한다.

<p align="center">
    <img src="/assets/post/image/legacy/cross-domain-adaptation-for-object-detection.png" width="80%">
</p>

학습은 다음과 같이 진행된다.

1. Source domain을 이용하여 object detector를 학습한다.
2. 학습된 object detector를 복사하여 teacher model과 student 모델을 제작한다.
3. Teacher model은 target domain에 weak augmentation을 적용해 psuedo-label을 만든다.
4. Student model은 strong augmentation된 source data로 supervised loss, strong augmentation된 target domain으로 unsupervied loss, discriminator로 discriminator loss를 계산한다.
5. Student model은 gradient update로 학습한다.
6. Teacher model은 EMA로 학습한다.

이 때 psuedo-label은 confidence threshold( $$\delta$$ )를 사용하여 양질의 것만 사용한다.

# Augmentation

기본적으로 teacher model은 source data에서 학습하고 target domain의 psuedo-label을 제작한다. 따라서 positive-false과 같은 오류가 발생할 수 있으므로 더 정확한 psuedo-label을 위해 psuedo-label 제작시 weak augmentation을 적용한다. 하지만 student는 믿을만한 psuedo-label을 가지고 있기 때문에 strong augmentationd르 적용한다.

- Weak augmentation: horizontal flipping, cropping
- Strong augmentation: randomly color jittering, grayscaling, Gaussian blurring, and cutting patches

# Loss

loss는 supervised loss, unsupervised loss, discrimination loss를 사용한다. Psuedo-label에서 사용하는 confidence threshol($$\delta$$)는 class의 confidence만 반영하고 bouding box의 confidence를 반영하지 않는다. 따라서 unsupervised loss에서는 class loss만을 사용한다.

$$
\mathcal{L}_{sup}(X_s, B_s, C_s)=\mathcal{L}_{cls}^{rpn}(X_s, B_s, C_s) +\mathcal{L}_{reg}^{rpn}(X_s, B_s, C_s) +\mathcal{L}_{cls}^{roi}(X_s, B_s, C_s) +\mathcal{L}_{reg}^{roi}(X_s, B_s, C_s)
$$

$$
\mathcal{L}_{unsup}(X_s, B_s, C_s)=\mathcal{L}_{cls}^{rpn}(X_s, B_s, \hat{C_s}) +\mathcal{L}_{cls}^{roi}(X_s, B_s, \hat{C_s})
$$

$$
\mathcal{L}_{dis}=-d \times logD(E(X))-(1-d)\times log(1-D(E(X))
$$

# EMA

Teacher model의 update는 EMA를 사용한다.

$$
\theta_t \leftarrow \alpha \theta_t + (1-\alpha)\theta_s
$$

# Result

<p align="center">
    <img src="/assets/post/image/legacy/da_result.png" width="100%">
    <img src="/assets/post/image/legacy/da_result_2.png" width="50%">
    <img src="/assets/post/image/legacy/da_result_3.png" width="80%">
</p>

AT는 SOTA를 찍긴 했지만 fully supervised learning (Oracle)보다 성능이 높은 것은 주목할만 하다.
