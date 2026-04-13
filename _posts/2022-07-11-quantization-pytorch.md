---
layout: post
title: "Pytorch Quantization 적용"
date: 2022-07-11 18:50:11 +0900
description: PyTorch Quantization 적용 방법 가이드
categories: [dev]
tags: [pytorch, quantization, optimization]
giscus_comments: true
related_posts: true
---

딥러닝 모델이 실제 device에 deploy 하는데 2가지 문제점이 있다.

1. 느린 inference time
2. 큰 model parameter size

Pytorch는 float32에서 int8로 데이터 크기를 줄여 연산을 하는 Quantization을 제공한다.
직접 짠 모델에서 quantization을 어떻게 적용하는지 알아보자.
전체코드는 이 [링크](https://github.com/wonbeomjang/blog-code/blob/main/resnet-quantization.py)에 있다.

## Work Flow

1. float32에서 학습시킨 model 혹은 pretrain model을 가져온다.
2. model을 eveluation으로 변경 후 layer fusion을 적용한다. (Conv + BN + RELU)
3. forward의 input엔 torch.quantization.QuantStub(), output엔 torch.quantization.DeQuantStub()을 적용한다.
4. quantization configuration을 적용한다.
5. layer를 fuse한다.
6. QAT (quantized aware training) 을 진행한다.
7. 모델을 cpu에 올려놓고 eval mode로 바꾼 후 float32모델을 int8모델로 변환시킨다.

## Code

### 1. Declare Model

모델은 resnet 사용하기로 한다. 그리고 편의를 위해 학습은 미리 시켰다고 가정한다.
먼저 resnet의 BottleNeck을 선언하고 resnet18을 구현한다.

<script src="https://gist.github.com/wonbeomjang/a36335f68a09946efc5332ccb00e05ae.js"></script>

### 2,3. Deploy Layer Fusion

이후 각 모듈에 layer fusion을 적용한다.
layer를 건드리지 않고 상속을 쓰면 결과적으로 parameter가 같기 때문에 QuantizableResNet18은 ResNet18의 파라미터를 쓸 수 있다.
QuantizableBottleNeck에서는 두 tensor를 더하는 연산이 있으므로 기존의 방식이 아닌 FloatFunctional을 이용해야한다.

<script src="https://gist.github.com/wonbeomjang/0ce010f6a9cb984b9a73d440c7c3dd67.js"></script>

### 6,7. QAT, Convert int8 model

QAT를 진행한 다음에 float32모델을 int8모델로 변환시킨다.

<script src="https://gist.github.com/wonbeomjang/355807be018f47515ca0b1a8cae758b8.js"></script>

## Result

간단하게 MNIST dataset으로 학습시켰다.
epoch 5, image size 224, optimzer Adam을 사용하였다.

<table align="center">
    <tr align="center">
        <td></td>
        <td>Accuracy (%)</td>
        <td>Inference Time (ms)</td>
        <td>Model Parameter (MB)</td>
    </tr>
    <tr align="center">
        <td>ResNet</td>
        <td>98.79</td>
        <td>66</td>
        <td>46.31</td>
    </tr>
    <tr align="center">
        <td>Quantizable ResNet</td>
        <td>96.42</td>
        <td>33</td>
        <td>11.69</td>
    </tr>
</table>
