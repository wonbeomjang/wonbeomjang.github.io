---
layout: post
title: "Jetson nano Ubuntu 20.04 (우분투 20.04) 설치"
date: 2023-02-12 18:50:11 +0900
description: pytorch 1.13을 향하여
categories: [dev]
tags: [jetson, ubuntu, edge-computing]
giscus_comments: true
related_posts: true
---

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/218313011-068d00e2-58fe-4cdf-bc2d-25496037be36.png" width="60%"></p>

Jetson nano로 프로젝트를 하다가 pytorch 1.10 이상이 필요해서 우분투 20.04 버전을 설치하는 법을 정리했다.

## Ubuntu 20.04 이미지 다운로드

먼저 우분투를 설치하기 위해서는 이미지 파일이 필요하다.
하지만 nvidia에서 공식적으로는 18.04 버전만 지원해서 사람들이 만든 버전을 사용해야했다.
물론 nvidia에서는 미지원 버전을 사용하는 것에 대해서 발생하는 문제는 책임을 지지 않는다 했다.

공식버전(18.04)을 받고 싶으면 아래 링크를 통해 다운로드 받으면 된다.

[https://developer.nvidia.com/embedded/downloads](https://developer.nvidia.com/embedded/downloads)

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/218313311-e92e3bf9-32c8-4b90-bcc3-36df2a53e81e.png" width="60%"></p>

우분투 20.04를 받고 싶으면 아래 링크에서 받으면 후 압축을 풀면 된다.
다운로드 속도 때문에 분리되어있는 압축파일을 다움받는 것을 추천한다.

[https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image)

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/218314700-765e1f41-bca4-45f7-8841-4b106a20cafb.png" width="60%"></p>

# 이미지 파일 설치

다음에는 보통 우분투를 설치하는 것 처럼 이미지를 설치하면 된다.
먼저 아래 링크에서 이미지 설치 파일을 다운로드 받자.

https://rufus.ie/ko/

그 후 위에서 받은 이미지 파일을 가지고 우분투를 설치하면 된다.

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/218315288-55a4e73f-5747-474a-ab20-4e4d36c40409.png" width="60%"></p>

설치가 끝나고 sd카드를 삽입하면 다음과 같은 화면을 만날 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/218315593-2f44349e-e15e-4a9e-8f75-496f8f497f2d.png" width="60%"></p>

비밀번호는 OS에는 다음과 같이 설치되어있다.

- Password: jetson
- sha256sum: 492d6127d816e98fdb916f95f92d90e99ae4d4d7f98f58b0f5690003ce128b34
- md5sum: f2181230622b81b6d882d4d696603e04
