---
layout: page
title: 집약
description: 시각장애인을 위한 약학정보제공서비스
img: assets/img/project_preview/zipyak.png
importance: 3
category: toy-project
---

# 개요

**프로젝트기간**
2018.08 ~ 2018.12

**개발인원**
3명

**담당역할**
Google Cloud Vision적용 및 핵심알고리즘 개발

**결과**
데이터 문제점 분석 및 해결방법 도출.
Recall 2%p, Precision 1%p 향상

**링크**
[https://wonbeomjang.github.io/blog/2020/barrier-free/](https://wonbeomjang.github.io/blog/2020/barrier-free/)
[https://www.autoeverapp.kr/bbs/board.php?bo_table=B06R](https://www.autoeverapp.kr/bbs/board.php?bo_table=B06R)

**내용**
본 프로젝트는 배리어프리 앱 개발 콘테스트 출품작으로 시각장애인들이 의약품을 구분할 후 있도록 이용하여 OCR기술을 적용한 의약품검색 어플리케이션입니다.
공공데이터를 활용하여 약학정보를 검색했으며 Google Cloud Vision을 이용하여 OCR을 적용했습니다.
또한 약포장에는 정보가 많아 많이 사용하는 약을 50가지를 선정하여 인식되는 약 목록과 OCR사용시
오류패턴을 확인하여 OCR결과를 보정했습니다.
해당 프로젝트에서는 실제 시각장애인이 사용할 수 있도록 한국시각장애인연합회의 자문을 받아 제작하였으며 우리동작에서 시각장애인분들이 실제 사용하여 불편한 점을 피드백 받아 제품을 업데이트 했습니다.

**프레임워크**
iOS, Google Cloud Vision

---

# 서비스 아키텍쳐

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/210772555-a58a4f77-473f-4815-8cc2-8792f69f50f0.png" width="80%"></p>
iOS와 Google Cloud Vision으로 제작한 것으로 이미지에서 글자를 OCR로 추출한 후 database에 등록된 약 이름과 비교하여 가장 비슷한 약을 가져와 공공데이터 포털로 정보검색을 합니다.
일정 수준이상 비슷한 약이 없으면 글자로 검색으로 넘어갑니다. 타이레놀은 ᄐH이레놀로 자주 인식되는데 이는 타이레놀로 정제하여 공공데이터 포털로 정보검색을 합니다.
