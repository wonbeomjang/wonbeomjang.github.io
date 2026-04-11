---
layout: post
title: "스타트업 리서치 인턴 후기"
date: 2023-08-23 00:00:00 +0900
description: 8개월간의 인턴을 마치며
categories: [review]
tags: [daily]
giscus_comments: true
related_posts: true
---

# 왜 시작했나요?

<p align="center">
    <img src="/assets/post/image/starup-intern/img.jpeg" width="80%">
</p>

<p align="center">
*'Researcher일까? Engineer일까?'*
</p>

나는 대학에 다니면서 항상 그런 고민을 했다. 고등학생 때 우연히 DeepMind에서 발표한 Playing Atari with Deep Reinforcement Learning이라는 논문을 보게되었고 인공지능에 빠져들었다. 대학 와서는 computer vision을 공부하게 되었다. 그저 인공지능이 좋아 backend, frontend 등 다른 분야보다는 인공지능 공부와 개발만 하게 되었다. 그러다 대학을 졸업할 때가 되었고, researcher와 engineer를 선택해야 할 순간이 다가왔다.

불행인지 다행인 건지 중앙대학교에서는 인턴을 해야지 졸업을 할 수 있었고, 관심 있는 두 군데 스타트업에 접촉하여 그중 한 회사인 뉴로클에서 인턴을 진행하게 되었다. 뉴로클을 선택한 이유는 간단했다.

1. Computer Vision을 중점으로 한다.
2. 자체 서비스를 판매하고 있다.
3. 기업매출을 보니 매출도 성장세였고, 흑자를 내기 시작했다.
4. 내가 내 일을 할 수 있고, 주체적으로 일할 수 있는 규모가 작지도 않고 크지도 않는 회사이다.

결론적으로 이러한 예측이 맞았고 성공적으로 인턴을 만들 수 있었다.

# 무엇을 했나요?

<p align="center">
    <img src="/assets/post/image/starup-intern/img.png" width="50%">
</p>

기본적으로 리서치 인턴의 역할을 수행했으나 후반에는 리서치 엔지니어의 역할을 하게 되었다. 퍼포먼스가 좋아서 그런지 생각보다 많은 일을 하게 되었다. (외부에 공개적으로 자료가 나간 것들만 포함했다)

1. Pretrained-OCR: OCR Auto-labeling 기능 추가 및 OCR 모델 성능 향상. 학습 속도 기존의 30%, 정확도 4%p 향상
2. Smart labeling (segmentation): 기술 검토 및 테스트, 모델 변환
3. Smart labeling (object detection): 기술 검토 및 모델 변환
4. 회사 블로그 제작
5. 리서치팀 docker 등 개발환경 관리
6. (방향성만 제시했지만) Neuro-I 성능개선, 다른 사람 연구 해결책 찾기 등등...

8개월동안 이걸 다 했다고?? 놀랍게도 그렇다. 개발 base로 computer vision을 공부했다 보니 구현 속도와 실험 속도가 압도적으로 빠른 것 같다.

# 무슨 경험이 도움이 되었나요?

<p align="center">
    <img src="/assets/post/image/starup-intern/project.png" width="50%">
</p>

동아리에서 공모전에서 프로젝트 경험이 많았고 협업도 많이 진행했다. 그래서 computer vision, 선형대수, 수치해석, 위상수학, 표 본론 등 과 같은 지식뿐만 아니라 tensorrt, onnx, quantization 등 많은 기술, pandas, matplotlib, seaborn과 같은 데이터 시각화, 딥러닝 모델이 제품에 어떻게 탑재해야 하는지에 대한 감도 있었다. 이 모든 경험을 회사에서 다 썼다. (진짜 다 썼다) 이러한 다양한 경험은 여러 기능에 기여를 할 수 있었던 것 같다.

# 무엇을 얻었나요?

<p align="center">
    <img src="/assets/post/image/starup-intern/company.png" width="50%">
</p>

## 협업

학생 때와 차원이 다른 협업을 하게 되었다. 학교에는 기껏해야 backed, fronted고 인원도 적어서 체계도 없이 작업을 해도 되었다. 하지만 인턴을 하면서 backed, fronted뿐만 아니라 영업, 마케팅, backbend, 기획 등 여러 사람과 협업을 진행했다.

### 요구사항을 명확하게 하자

협업은 기본적으로 background가 완전하게 동일하지 않은 사람들끼리 작업을 한다. 따라서 같은 목표를 바라보고있어도 세부 사항이 다를 수 있다. 만약 이를 조정하지 않고 일을 진행하다 보면 다음에 다시 조정하고 어려울뿐더러 비용 역시 많이 발생한다.

### 방향성 설정

하나의 기능이 만들어지기 위해서 다음과 같은 과정을 거친다.

1. 새로운 연구 주제로 연구한다.
2. 기획 및 디자인팀에서 제품 탑재 방향을 결정한다.
3. 개발팀에서 제품을 개발한다.
4. 마케팅팀에서 협력사에 제공할 데이터와 대외 홍보용 자료를 제작한다.
5. 영업을 통해 제품을 판매한다.

라이프 사이클에서 연구는 최상단을 차지한다. 따라서 첫 단추가 잘못 채워지면 전체적인 제품 방향이 엇나갈 수 있다.

### 문서화를 체계적으로 하자

내가 하는 연구를 follow up 하는 사람은 극소수다. 연구 이후 제품화할 때 조직되는 관련자들은 내가 작성해 놓은 document를 보고 작업을 시작한다. 그래서 진행한 연구와 모델을 제대로 이해시키려면 구조적으로 잘 문서화를 해야 한다.

## 지식

## Tensorflow

나는 지금까지 pytorch를 이용하여 작업을 했다. Tensorflow는 회사 들어와서 거의 처음 쓰게 된 것이다. Tensorflow는 기본적으로 eager mode가 제공되지 않아 코딩하는 것이 힘들었지만 tensorflow와 tensorflow orbit을 익히게 되는 좋은 기회가 되었다.

### 논문

회사에 들어와서 논문을 진짜 많이 읽었다. 연구에 기반이 되는 논문뿐만 아니라 적용할 만한 최신논문, 기술 리포트, 워크숍 논문 등 다양하게 많이 읽었다. 이를 통해 ViT, active learning, OCR, anomaly detection, super resolution, backbone for edge device, federated learning 등 다양한 분야에 대해 기초지식을 쌓을 수 있었다.

### 데이터 분석

기본적으로 데이터가 부족한 상황에서 모델의 성능을 올리는 방법을 고민을 했다. 이 때문에 사용자가 다룰 예상데이터의 특성을 분석하고 데이터에 적합한 방법론을 사용하여 성능을 높일 수 있었다.

## 일은 잘했나요?

나에 대한 평가가 긍정적인 것을 보면 일을 잘했던 것 같다. 무엇보다도 나랑 같이 인턴을 진행한 분과 잘하는 것이 달라서 서로 시너지가 났던 것 같다. 원래 회사에도 리서치 인턴이 없었는데 인턴 둘이 좋은 선례를 만들어서 앞으로 계속 채용할 예정이다. (사실 면접이 완료되어 다음 리서치 인턴도 정해졌다.)

# 앞으로 무엇을 할 것인가요?

<p align="center">
    <img src="/assets/post/image/starup-intern/todo.png" width="50%">
</p>

감사하게도 회사에서 정규직 제안을 받았다. 스타트업 리서치 엔지니어에 가깝지만, 학사 출신으로 연구를 할 수 있다는 것이 흔치 않은 기회이다. 하지만 회사에서 원하는 인재와 내가 가고자 하는 방향이 달랐다. 회사에서 원하는 인재는 generalist이지만 나는 한 분에서 specialist가 되고 싶었다. 분야 또한 일반적인 모델링이 아닌 모델 경량화, hardware optimization, low cost serving 쪽으로 가고 싶다. 그리고 내가 다루어야 하는 target data가 무엇인지 명확하게 정할 수 있는 연구개발을 하고 싶다.

이제 학교에 다시 돌아간다. 8개월 동안 뉴로클 덕분에 좋은 경험을 했고 내 실력도 엄청나게 향상되었다. 이제 4학년 2학기이다. 졸업도 얼마 안 남아서 취업 준비나 대학원 준비를 해야겠지만 DL engineer 쪽 공부도 더욱 열심히 하면서 내가 목표하는 커리어를 만들어 가야겠다.
