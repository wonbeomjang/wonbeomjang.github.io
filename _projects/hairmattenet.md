---
layout: page
title: 염색프로그램
description: HairMatteNet + Quantization 구현
img: assets/img/project_preview/hairmattenet.png
importance: 3
category: toy-project
---

# 개요

**프로젝트기간**
2018.08 ~ 2018.12

**개발인원**
1명

**담당역할**
HairMatteNet구현 및 Quantization적용

**결과**
IoU 0.75%p 향상, inference time 42.9%, model size 56% 감소

**링크**
[https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch](https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch)

**내용**
본 프로젝트는 Pytorch공부를 위해 제작한 것으로 머리에 특화된 semantic segmentation 모델인 HairMatteNet을 논문을 참고하여 구현했습니다.
추가적으로 성능향상을 위해 Backbone Network를 mobilenet v2로 교체해 IOU 0.75%p 향상시켰습니다.
후에 quantization을 추가하여 inference time은 기존 대비 42.9%, model size는 기존대비 56%로 감소하였습니다. 추가로
TensorRT를 활용하여 inference time은 기존대비 9%, model size는 기존대비 3.2%로 줄였습니다.

**프레임워크**
pytorch
