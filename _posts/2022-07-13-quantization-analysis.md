---
layout: post
title: "Quantization과 inference speed"
date: 2022-07-13 18:50:11 +0900
description: Quantization 성능분석
categories: [dev]
tags: [quantization, tensorrt, optimization]
giscus_comments: true
related_posts: true
---

Quantization은 precision reduction으로 parameter의 용량을 줄이기위해 나왔다.
하지만 실제로 써봤을 때 유의미한 속도차이가 있었다. 왜 그런 것일까 궁금해서 몇 가지 측정을 했다.
먼저 3가지 모델을 준비했다. 일반 cpu에서의 MobileNetV2, quantization을 진행한 MobileNetV2, layer fusion과 quantization을 진행한 MobileNetV2이다.
각각 모델에게 image를 5000씩 inference하도록 하고 time elapsed와 cache miss, instruction per cycle을 비교했다.
<br>
각각 코드는 다음과 같다.

### pytorch cpu

<script src="https://gist.github.com/wonbeomjang/419b410674ec8a7d5dcb6ffc38371289.js"></script>

### quantization with non layer fusion

<script src="https://gist.github.com/wonbeomjang/2d0bee28abcf47ea2e59febd094dbefd.js"></script>

### quantization with layer fusion

<script src="https://gist.github.com/wonbeomjang/e57959145f6b6c219cf30c10d8c718ac.js"></script>

## Result

결과는 ubuntu에 perf을 통해 측정했다.

<table align="center">
    <tr align="center">
        <td></td>
        <td>time elapsed (sec)</td>
        <td>cache miss (%)</td>
        <td>instruction per cycle</td>
        <td>Model Size (MB)</td>
    </tr>
    <tr align="center">
        <td>MobileNetV2</td>
        <td>89.8063</td>
        <td>20.774</td>
        <td>0.58</td>
        <td>14.2605</td>
    </tr>
    <tr align="center">
        <td>+ quantization</td>
        <td>54.4425</td>
        <td>5.987</td>
        <td>0.68</td>
        <td>4.2422</td>
    </tr>
    <tr align="center">
        <td>+ layer fusion</td>
        <td>31.1885</td>
        <td>5.004</td>
        <td>0.97</td>
        <td>3.9436</td>
    </tr>
</table>
<br>

**이건 순전히 필자의 추측이다**
vanilla MobileNetV2와 quantization MobileNetV2을 보면 instruction per cycle 차이보다 cache miss 가 더 유의미하다. (같은 instruction per cycle 에서 MobileNetV2의 time elapsed 는 105.2901이다.
따라서 precision reduction 으로 parameter 가 용량이 적어져 cache miss 가 적어진 것과 parameter data transfer latency 가 적어진 것으로 볼 수 있다. (혹시 아니면 메일을 주면 감사합니다.)
이후 layer fusion 을 통해 time elapsed 가 줄어들었다. 이때는 cache miss 가 높게 줄어들지 않았으므로 단순히 graph reduction 에 따른 성능향상으로 볼 수 있을 것이다.
instruction per cycle 에서 봤을 때도 낮은 연산때문에 instruction per cycle 이 높아졌다. 그리고 context switching 도 줄어들었다.

<center>
$$ \frac 1 n \sum (x_i - \bar x)(y_i - \bar y) $$
</center>
