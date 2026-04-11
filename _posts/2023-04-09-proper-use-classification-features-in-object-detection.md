---
layout: post
title: "Proper Reuse of Image Classification Features Improves Object Detection"
date: 2023-04-09 00:00:00 +0900
description: Neck is important
categories: [transfer-learning, object-detection]
tags: [paper, object-detection, classification]
giscus_comments: true
related_posts: true
---

# Introduction

기존 object detection을 학습시킬 때 imagenet backbone을 사용하여 pretrain후 transfer learning을 통해 target dataset에 대해 학습을 했다. 이 상황에서 보통 object detection의 backbone만 pretrain시키고 detection관련 module은 random으로 initalize시킨다. 따라서 이에 관한 연구가 있었는데 크게 두 가지가 있다.

- Pretrain으로 사용한 classiciation data의 량 만큼 object detection에 긍정적 영향을 끼친다.
- In-domain dataset으로 학습시간을 길게 가져가면 pretrained model과 scratch model의 성능차이가 없어진다.

저자들은 두 연구를 분석하기위해 backbone network를 classification dataset에 대해 학습 후 paramter를 freeze시켜 object detection model을 학습시켰다.

# Methodology

기존에 사용하는 일반적인 학습방법은 다음과 같다.

1. ImageNet, JFT-300M과 같은 classification dataset에서 backbone network를 학습시킨다.
2. Backbone network에 detection specific component를 추가한다. ex) RPN, FPN, NAS-FPN
3. Target dataset에 대하여 재학습시킨다.

하지만 저자는 다음과 같은 방법을 제안한다.

1. ImageNet, JFT-300M과 같은 classification dataset에서 backbone network를 학습시킨다.
2. Backbone network를 freeze시킨 후 detection specific component를 추가한다.
3. Target dataset에 대하여 재학습시킨다.

결론적으로 backbone을 freeze시키는 것만 달라졌다.

# Experiment

실험결과를 요약하자면 다음과 같다.

## Detectoin-Specific capacity

- FPN, RPN, Detection Cascade 등은 network의 generalzation에 도움을 준다.
- 해당 요소들의 capacity가 충분하면 backbone freezing이 fine-tuning과 training scratch보다 좋은 성능을 보인다.
- Backbone pretraining시 classification dataset의 수가 많아질수록 성능이 높아진다.
- **Data Augmentation**

Instance segmentation 실험에서는 Data augmentation은 다음을 사용했다.

- Large Scale Jittering
- Copy-and-paste augmentation

**Architecture**

첫번째 실험의 model architecture는 다음과 같다.

- Faster-RCNN
- ResNet50
- FPN or NAS-FPN
- Cascade head

두 번째 실험의 model architecutre는 다음과 같다.

- Mask-RCNN
- EfficientNet-B7
- NAS-FPN
- Cascade head

**Hyper parameter**

해당실험은 자원이 한정되어있는 상황을 가정했기 때문에 batch size는 64, learning rate는 그에 맞게 0.08로 설정했다.

**Dataset**

Detection dataset으로 MS-COCO, LVIS를 사용했고, classification dataset으로는 ImageNet, JFT-300M을 사용했다.

## ResNet50 + Faster-RCNN

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled.png" width="50%">
</p>

모든 실험서 공통적으로 classification pretrain시 이미지수가 비교적으로 적은 ImageNet보다 JFT-300M의 성능이 좋았다. 또한 FPN을 사용했을 때보다 parameter수가 더 많은 NAS-FPN, Cascade head를 사용했을 때 성능이 좋았다. 이는 backbone network가 다른 domain에서 학습이 되어 이를 generalize하는데 충분한 capacity가 필요하다고 한다.

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%201.png" width="50%">
</p>

pretraing과 fine-tuning의 관계를 알아보기위해 실험을 진행했다. 첫 번째를 보았을 때 traning schedule이 짧을 경우엔 pretranig에 사용된 classfication dataset의 크기가 커질수록 성능이 좋았따. 두 번째를 보았을 때 training schedule이 길수록 pretran classification dataset에따른 성능차이가 줄어든 것을 보아 pretran은 성능에 도움이 되지 않는다고 한다. 저자는 이를 확장하여 large dataset으로 pretrain하고 backbone을 freeze시켜 long traning schedule으로부터 knowledge를 보호한다. 또한 high-capacity detector component사용하여 domain이 다른 문제를 해결했다.

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%202.png" width="50%">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%203.png" width="80%">
</p>

Backbone은 ImageNet에서 학습시켰다. 따라서 object detection과 domain gap이 발생하는데 이는 detector component가 해결할 수 있다. FPN을 사용할 경우 generalize하는데 capacity가 부족해 성능감소가 발생했다. 하지만 NAS-FPN과 Cascade head를 추가했을 때 성능에 이득이 있었다.

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%204.png" width="50%">
</p>
결론적으로 충분한 capacity의 detector component를 사용하고 backbone network를 freeze 시키면 trainable parameter가 줄어들고 FLOPS도 줄어들어 학습속도와 memory가 줄어든다.

## EfficientNet-B7+Mask-RCNN

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%205.png" width="50%">
</p>
capacity가 충분한 detector component를 추가하고 backbone을 freeze시키는 것이 좋은 성능을 냈으며 Copy-Paste라는 강한 augmentation을 사용했을 떄도 성능향상이 있었다.

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%206.png" width="100%">
</p>
small object, midium object 모두 성능에서 향상이 있었다.

## Furthurmore

<p align="center">
    <img src="/assets/post/image/proper-reuse-of-image-classification-features-improve-object-detection/Untitled%207.png" width="50%">
</p>
Backbone Freeze가 만병통치약은 아니었다. Freeze시킨 경에서도 residual adapter를 사용하면 더 좋은 성능을 냈다.
