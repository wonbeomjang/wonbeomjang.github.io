---
layout: post
title:  "학부생이 본 EffientNet"
date:   2021-01-03 16:41:11 +0900
tags: [Computer Vision]
comments: true
---

## Introduction
ImageNet Dataset이 나온 이후에 여러 classification모델이 제안되었다.
VGG이후 ResNet부터 네트워크를 깊게 쌓음으로써 정확도를 올리게 되었다. 
GPipe경우에는 base model보다 4배를 크게 만듬으로써 ImageNet에서 우승하게 되었다.
이와 같이 보통은 depth를 늘리면서 정확도를 올리고 width를 늘리거나 resolution을 늘리면서 정확도를 올린다.  

depth, width, resolution이렇게 3가지를 균형을 맞추면 더 성능이 좋아지지 않을까?
직관적으로 생각하면 큰 이미를 넣으면 receptive field가 커지고 그 패턴을 캡쳐하기 위해 channel수도 늘어야될 것이다.
EffientNet은 이러한 아이디어를 갖고 compound scaling method를 제안하며 기존의 네트워크의 성능을 올리게 되었고, 새로운 state-of-art모델을 제안했다.

### Compund Model Scaling

$$\mathcal{N} = \bigodot_{i=1...s} \mathcal{F}^{\mathrm{L}_i}(\mathrm{X}_{<H_i, W_i, C_i})$$
