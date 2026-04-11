---
layout: page
title: Gripp
description: 딥러닝 영상 분석을 활용한 클라이밍 경쟁 서비스
img: assets/img/project_preview/gripp.png
importance: 1
category: toy-project
---

# 개요

**프로젝트기간**
2022.08 ~ 2022.12

**개발인원**
3명

**담당역할**
아이디어 제시, 핵심 알고리즘 개발, 딥러닝 서버 제작

**결과**
데이터 문제점 분석 및 해결방법 도출.
Recall 2%p, Precision 1%p 향상

**링크**
[https://github.com/wonbeomjang/gripp-deep](https://github.com/wonbeomjang/gripp-deep)
[https://www.youtube.com/watch?v=zakn9Nvc_io](https://www.youtube.com/watch?v=zakn9Nvc_io)

**내용**
본 프로젝트는 캡스톤디자인 수업에서 진행한 프로젝트입니다.
취미생활인 클라이밍 영상을 촬영 후 성공영상과 실패영상을 구분하고 편집하는 것이 번거롭다는 점에 착안하여 제작한 어플리케이션입니다.
빠른 개발과 좋은 성능을 위해 yolov5와 mediapipe를 사용했습니다.
또한 데이터의 문제점을 분석하여 vertical flip, mixup으로 Recall 2%p, Precision 1%p 향상시켰습니다.

**프레임워크**
iOS, Spring, Django, Pytorch, MySQL, Oracle Cloud, Wandb, Docker, RabbitMQ...

---

# 핵심로직

<p align='center'><img src='https://user-images.githubusercontent.com/40621030/210767236-0725443c-2e1d-4bb7-9afc-829906795f42.png' width="80%"/></p>

영상서버에서 영상을 받아와 5개의 이미지를 추출했습니다.
YOLOv5를 이용하여 top hold, hand hold, start hold를 detection한 후 각각의 mask image를 만들었습니다.
이후 0.5초마다 frame을 추출 후 MediaPipie를 통하여 손의 mask image를 만든 후 위에서 추출한 mask image와 겹치는지 확인했습니다.
이를 통해 등반 성공 여부와 편집점을 구하여 REST API로 응답했습니다.

- Hold는 암벽등반시 잡는 손잡이를 의미한다.

<p align='center'><img src='https://user-images.githubusercontent.com/40621030/210767615-1cd966b9-a65f-4dd3-83ae-4c4b872ee5e9.png' width="80%"/></p>

<table align="center">
    <tr align="center">
        <td>Model or Method</td>
        <td>Score</td>
    </tr>
    <tr>
        <td>Baseline (yolov5n)</td>
        <td>0.965292</td>
    </tr>
    <tr>
        <td>+ vertical flip</td>
        <td>0.965362</td>
    </tr>
    <tr>
        <td>+ mixup</td>
        <td>0.972415</td>
    </tr>
    <tr>
        <td>+ vertical flip + mixup</td>
        <td>0.972558</td>
    </tr>
</table>
<br>

클라이밍 홀드를 추출하는 YOLOv5를 주로 튜닝했으며 Recall과 Precision의 평균을 target matrix로 튜닝을 했습니다.
홀드엔 위 아래가 없다는 점과 LED색으로 홀드를 구분한다는 점에 착안해 Vertical Flip과 MixUp으로 Recall 2%p, Precision 1%p 향상했습니다.

---

# 서비스 아키텍쳐

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/210769835-2046a9f1-6e6f-4d9d-b2f5-0afd62e48325.png" width="80%"></p>
iOS, Spring, Django를 사용하였고 iOS에서 올라온 비디오는 RabbitMQ를 통하여 비동기처리를 하여 비디오 서버에 올립니다.
비디오 서버는 딥러닝서버에 영상분석 요청를 보내고 응답을 받으면 FFmepeg를 이용하여 영상을 편집합니다.
또한 iOS어플리케이션에 스트리밍하기 위해서 6초씩 영상을 잘라 Object storage에 저장한 후 Http Live Stream으로 영상스트리밍을 합니다.
