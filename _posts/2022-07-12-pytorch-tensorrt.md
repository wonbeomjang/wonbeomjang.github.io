---
layout: post
title: "Pytorch Tensorrt 적용"
date: 2022-07-12 18:50:11 +0900
description: PyTorch 모델을 TensorRT로 변환하는 방법
categories: [optimization]
description: PyTorch 모델을 TensorRT로 변환하는 방법
tags: [pytorch, tensorrt, optimization]
giscus_comments: true
related_posts: true
---

TensorRT는 Deep Learning 모델을 최적화해 GPU에서 inference 속도를 향상시킬 수 있는 최적화 엔진이다.
TensorRT는 GPU에서 최적화된 성능을 낼 수 있도록 Network Compression, Netword Optimization을 진행한다.

<p align="center"><img src="https://blogs.nvidia.co.kr/2020/02/19/nvidia-tensor-rt/219-%eb%b8%94%eb%a1%9c%ea%b7%b83/" width="80%"></p>

### Quantization & Precision Calibration

quantization을 통한 precision reduction은 network의 파라미터의 bit가 작기 떄문에 더 좋은 성능을 발휘할 수 있다.
TensorRT는 Symmetric Linear Quantization을 사용하고 있으며, float32 데이터를 float16, int8로 낮출 수 있다.
하지만 int8로 precision을 낮추면 숫자표현이 급격히 줄어들어 성능에서 문제가 생긴다.
따라서 TensorRT는 callibration을 통해 weight과 intermidiate tensor에서의 정보손실을 최소화한다.

<p align="center"><img src="https://blogs.nvidia.co.kr/wp-content/uploads/sites/16/2020/02/Figure-5.-Calibration-methodology.png" width="80%"></p>

### Graph Optimization

TensorRT는 또한 platform에 최적화된 graph를 위해 Layer Fusion 방식과 Tensor Fusion을 사용한다.
따라서 Vertical Layer Fusion, Horizontal Layer Fusion, Tensor Fusion이 적용되어 graph를 단순하게 만들어 속도를 높힌다.

<p align="center"><img src="https://blogs.nvidia.co.kr/wp-content/uploads/sites/16/2020/02/219-%EB%B8%94%EB%A1%9C%EA%B7%B8-6.png" width="80%"></p>

### Etc

Kernel Auto-tuning을 통해 GPU의 cuda core 수, 아키텍쳐 등을 고려하여 optimization을 진행하고,
Dynamic Tensor Memory & Multi-stream execution을 통해 footprint를 줄여 성능을 높힌다.

## TensorRT설치

필자 Docker를 사용하므로 Docker를 기준으로 설명하겠다.

## 1. 도커 설치

[Ubuntu](https://docs.docker.com/engine/install/ubuntu/),
[Window(WSL2)](https://docs.docker.com/desktop/windows/install/),
이렇게가 설치방법인데 Window는 미리 WSL2를 이용해 Nvidia-Driver를 설치해야한다. ~~(해봤는데 정말 귀찮다)~~ 그리고 MacOS는 Nvidia GPU를 사용할 수 없으니 사실상 사용을 못한다.

## 2. Docker container만들기

먼저 nvidia-smi를 사용하여 cuda version을 확인한다.

```bash
nvidia-smi
```

<p align="center"><img src="https://user-images.githubusercontent.com/40621030/178482155-6a312bd8-4028-4173-9e19-9cdacb2da2f2.png" width="80%"></p>

위의 사진을 보고 자신의 cuda toolkit version에 맞는 컨테이너 버전을 사용하자. 추가적인 cuda version은
[TensorRT Container Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_22-06.html#rel_22-06) 이 링크를 이용하면 된다.
그 다음 container버전과 python버전을 채워넣어 docker image를 pull하면 된다.

```bash
docker pull nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>
```

예 container version 22.06, python3 사용

```bash
docker pull nvcr.io/nvidia/tensorrt:22.06-py3
```

그 후 container를 만들어 run하면 된다.
**Docker 19.03 또는 그 이후 버전**을 사용하면

```bash
docker run --gpus all -it --name tensorrt nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>
```

**Docker 19.02 또는 그 이전 버전**을 사용하면

```bash
nvidia-docker run -it --name tensorrt  nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>
```

을 사용하면 된다.

## 3. Pytorch TensorRT설치

Torch-TensorRT를 사용하려면 3가지 방법이 있다.

1. torch_tensorrt [docker image](https://github.com/pytorch/TensorRT) 사용
2. [torch_tensorrt](https://pytorch.org/TensorRT/tutorials/installation.html#installation) 패키지 사용
3. nvidia iot [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) 사용

필자는 1번 방법에서는 pytorch버전 오류, 3번에서는 성능최적화가 잘 안돼 2번을 선택했다.
torch_tensorrt는 현재 v1.1.0로 pytorch v1.11.0밖에 지원이 안 된다. v1.11.0으로 설치하자
CUDA 11.3

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

CUDA 10.2

```bash
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
```

이후 torch-tensorrt를 설치하자

```bash
pip install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases
```

## 4. TensorRT적용

현재 pytorch는 torch script -> onnx -> tensorrt이렇게 변환한다. 먼제 model을 선언해주자.

<script src="https://gist.github.com/wonbeomjang/d549423cec450527a70455696d4bfa81.js"></script>

성능비교를 해보자

<script src="https://gist.github.com/wonbeomjang/3b4dc51106e2dd03d6a85f89ec2aac7f.js"></script>

## Result

<table align="center">
    <tr align="center">
        <td></td>
        <td>Inference Time (ms)</td>
        <td>Model Parameter (MB)</td>
    </tr>
    <tr align="center">
        <td>MobileNetV2</td>
        <td>6.1163</td>
        <td>14.2630</td>
    </tr>
    <tr align="center">
        <td>TensorRT</td>
        <td>0.3954</td>
        <td>0.0005</td>
    </tr>
</table>
<br>

이렇듯 TensorRT를 이용하면 inference time이나 parameter size에서 이득을 볼 수 있다.
