---
layout: page
title: 카나리아
description: 모두를 위한 군사보안 경보 시스템
img: assets/img/project_preview/canary.png
importance: 2
category: toy-project
---

# 개요

**프로젝트기간**
2021.05 ~ 2021.11

**개발인원**
5명

**담당역할**
PM, 핵심알고리즘 개발, ML모델 배포서버 제작, Azure ML 환경 설정 결과 데이터 문제점 분석 및 해결방법 도출. Recall 2%p, Precision 1%p 향상

**링크**
[https://github.com/wonbeomjang/AI_APP_WEB_Canary_Canary_2021](https://github.com/wonbeomjang/AI_APP_WEB_Canary_Canary_2021)
[https://www.youtube.com/watch?v=zD_AGme63og](https://www.youtube.com/watch?v=zD_AGme63og)
[https://github.com/wonbeomjang/AI_APP_WEB_Canary_Canary_2021/tree/main/AI(BE)/deeplearning/kwoledge_distillation_yolov5](<https://github.com/wonbeomjang/AI_APP_WEB_Canary_Canary_2021/tree/main/AI(BE)/deeplearning/kwoledge_distillation_yolov5>)

**개발기**
[1. 팀 결정 및 주제&시스템 설계](https://www.wonbeomjang.kr/blog/2021/osam-1/)
[2. computer vision 개발 과정](https://www.wonbeomjang.kr/blog/2021/osam-1/)
[3. 이제 끝나는 건가](https://www.wonbeomjang.kr/blog/2021/osam-1/)

**내용**
본 프로젝트는 2021 군장병 공개SW 역량강화 온라인 해커톤의 출품작으로 전차, 군함, 총기, 문서 등 군사보안위반 가능성 물체들을 모자이크 해주고 이를 알려주는 어플리케이션을 제작했습니다. Object detection을 위해 yolov5를 사용하였으며 mosaic를 mosaic_9으로 바꾸고 self-distillation을 적용하여 Recall을 4.3%p 향상시켰습니다.

**프레임워크**
Flutter, Node js, Pytorch, Django, Azure ML, Docker

---

# 핵심 로직 및 서비스 구조

<p align='center'><img src='https://user-images.githubusercontent.com/40621030/210398815-4a9aa64e-33de-4c79-8919-2e3c99183dfe.png' width="80%"/></p>

Flutter, Node js, Libtorch를 사용하여 제작을 하였습니다.
팀원 모두 프로젝트가 처음이라는 것을 감안하여 서비스 아키텍쳐와 로직을 최대한 단순하게 잡았습니다.
YOLOv5를 이용하여 object detection후 opencv를 활용하여 이미지를 모자이크하고 경고메시지를 작성하여 node js에게 넘겼습니다.
Azure ML은 학습 요청을 queue로 관리하녀 Docker를 통해 실험환경을 구성합니다.
모델이 학습이 완료되면 Django서버에 파라미터와 성능지표가 등록됩니다.
Node js가 분석 요청 시 Detection Module은 Django에 성능이 좋은 모델이 있는지 확인하여 있으면 자동으로 다운로드 받습니다.

---

# 개발 과정

## 기술스택

<table align="center">
  <tr align="center">
    <td><a href="https://pytorch.org/"><img src='https://user-images.githubusercontent.com/40621030/136698820-2c869052-ff44-4629-b1b9-7e1ae02df669.png' height=80></a></td>
    <td><a href="https://opencv.org/"><img src='https://user-images.githubusercontent.com/40621030/136698821-10434eb5-1a98-4108-8082-f68297012724.png' height=80></a></td>
    <td><a href="https://www.cvat.ai/"><img src='https://user-images.githubusercontent.com/40621030/136698825-f2e1816f-580b-4cf1-960d-295e9f18a329.png' height=80></a></td>
    <td><a href="https://roboflow.com/"><img src='https://user-images.githubusercontent.com/40621030/136698826-e18a44a9-63d1-498b-a63f-c76bdc603f3b.png' height=80></a></td>
  </tr>
  <tr align="center">
    <td align='center'>PyTorch</td>
    <td align='center'>OpenCV</td>
    <td align='center'>CVAT</td>
    <td align='center'>Roboflow</td>
  </tr>
</table>
<br>

## Object detection VS Semantic segmentation

- Semantic segmentation: 사람을 제외한 배경을 처리
  난이도: 상대적으로 낮음(사람을 대상으로 학습된 model 사용)
  장점: 기존 모델을 사용 시 사람을 깔끔하게 구별 가능
  단점: 오직 사람/배경만 구별 가능, 사람 앞의 물체에 대해선 감지하지 못할 수 있음
  (ex: 기밀 문서를 들고 있는 사람)

- Object detection: 학습한 Class들을 사진 안에서 검출하여 처리
  난이도: 상대적으로 높음(We need to get dataset, annotate them, train model...)
  장점: 여러 다양한 class들을 검출하여 사진의 상황을 대략적으로 파악 가능,
  보안 위반 객체는 detect만 된다면 처리 가능(보안성), 사람 이외의 객체들도 살려낼 수 있음
  단점: segmentation보다 상대적으로 깔끔하지 못한 사진 처리, 높은 데이터 수집 난이도와 큰 시간 소요

보다 높은 보안성을 중시하기로 결정 --> Object detection

## 사용 데이터셋

### Version 1: [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)

 <p align='center'><img src='https://user-images.githubusercontent.com/40621030/137607638-124c1622-6bfe-4a45-a16b-519314916436.jpg' width="80%"/></p>

**문제점**

1. 데이터 수 부족
2. 대다수 물체가 정중앙 위치
3. 대다수 물체가 사진 전체를 차지

**해결방안 1 - 데이터 추가**

<table>
  <tr>
    <td align='center'>Orignal Dataset</td>
    <td align='center'>Add more data</td>
  </tr>
  <tr>
    <td align='center'><img src='https://user-images.githubusercontent.com/40621030/137607638-124c1622-6bfe-4a45-a16b-519314916436.jpg' width="80%"/></td>
    <td align='center'><img src='https://user-images.githubusercontent.com/40621030/137607640-9552448f-a39c-4a46-9d50-a523002be0e4.jpg' width="80%"/></td>
  </tr>
</table>

**해결방안 2, 3 - augmentation 방법 변경**

<table>
  <tr>
    <td align='center'>기존</td>
    <td align='center'>변경</td>
  </tr>
  <tr>
    <td align='center'><img src='https://user-images.githubusercontent.com/40621030/137607771-6509a1f3-872a-4bfd-ac0f-389e7dcd8fdc.jpeg' width="80%"/></td>
    <td align='center'><img src='https://user-images.githubusercontent.com/40621030/137607774-68692b66-5324-4184-ba9a-e41151a6a561.jpeg' width="80%"/></td>
  </tr>
</table>
<br>

### 사용 모델

YOLOv5, Efficientnet, SSGlite 등의 모델들을 고려.
성능과 학습에 들어가는 시간 등을 종합적으로 판단 --> YOLOv5 결정.
(Efficientnet: 학습 시간이 지나치게 많이 소요, SSGlite: YOLOv5보다 낮은 성능)

- YOLOv5 ([original github](https://github.com/ultralytics/yolov5))
<p align='center'><img src='https://user-images.githubusercontent.com/40621030/136682963-80100da0-c31c-4df4-8bff-583e1c1c62f1.png' width="80%"/></p>

**문제점**

<p align='center'><img src='https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png' width="80%"/></p>

1. 낮은 성능
2. 무거운 모델 (ex. yolov5l6)

**해결방안**

- knowledge distillation ([paper link](https://arxiv.org/abs/1906.03609))
  <p align='center'><img src='https://user-images.githubusercontent.com/40621030/136683028-fb1ca2f0-97c0-4581-9b7a-64e26536d7af.png' width="80%"/></p>

### 성능 향상

<table align="center">
  <tr align="center">
    <td>enhance</td>
    <td>model</td>
    <td>precision</td>
    <td>recall</td>
    <td>mAP_0.5</td>
    <td>mAP_0.5:0.95</td>
  </tr>
  <tr align="center">
    <td>Before add dataset</td>
    <td>yolov5m6</td>
    <td>0.602</td>
    <td>0.651</td>
    <td>0.671</td>
    <td>0.535</td>
  </tr>
  <tr align="center">
    <td>None (Add dataset)</td>
    <td>yolov5m6</td>
    <td>0.736</td>
    <td>0.779</td>
    <td>0.815</td>
    <td>0.599</td>
  </tr>
  <tr align="center">
    <td>mosaic_9 50%</td>
    <td>yolov5m6</td>
    <td>0.736</td>
    <td>0.779</td>
    <td>0.815</td>
    <td>0.599</td>
  </tr>
  <tr align="center">
    <td>mosaic_9 100%</td>
    <td>yolov5m6</td>
    <td>0.739</td>
    <td>0.813a</td>
    <td>0.806</td>
    <td>0.594</td>
  </tr>
  <tr align="center">
    <td>self distillation</td>
    <td>yolov5m6</td>
    <td>0.722</td>
    <td>0.822</td>
    <td>0.807</td>
    <td>0.592</td>
  </tr>
</table>
<br>
<table align="center">
 <tr>
  <td align='center'>Original Image</td>
  <td align='center'>Result Image</td>
 </tr>
 <tr>
  <td align='center'><img src='https://user-images.githubusercontent.com/40621030/136698553-a00eb618-7783-41d9-bd2c-203dbbd60946.jpg' width="80%"/></td>
  <td align='center'><img src='https://user-images.githubusercontent.com/40621030/136698552-42c71108-9efc-4c88-a68a-3f5aec8452c6.jpg' width="80%"/></td>
 </tr>
</table>
