---
layout: post
title: "Jetson Nano Tensorrt 적용"
date: 2023-03-01 00:00:11 +0900
description: pytorch to tensorrt using onnx
categories: [jetson-nano, tensorrt, hardware-optimization]
tags: [jetson, tensorrt, edge-computing]
giscus_comments: true
related_posts: true
---

이전에 RTX2070에 TenssorRT를 적용했던 것처럼 jetson nano에서도 tensorrt를 적용하고 싶었고 호환성에 애를먹다가 성공했다.
python tenosrrt package를 buil하기 귀찮아 ubuntu 18.08을 사용하여 내부에 설치되어있는 것을 이용했다.

Jetson nano 18.04 설치방법은 아래에 나와있다. 여기서 20.04버전이 아닌 18.04 공식 이미지를 사용해야한다.

[http://www.wonbeomjang.kr/blog/2023/jetson-nano-ubuntu/](http://www.wonbeomjang.kr/blog/2023/jetson-nano-ubuntu/)

# 1. Pytorch 속도 측정

tensorrt를 사용하기 앞서 pytorch에서 gpu에 올려 돌렸을 때 inference time을 측정해봤다.

<script src="https://gist.github.com/wonbeomjang/6878e49a9b7152f004f8be6bbf57ca61.js"></script>

output

```bash
37.80 ms / 10.3278MB
```

tensorrt에서 float16을 사용할 예정으로 torch.float16으로 type을 변경하여 실험하자.

<script src="https://gist.github.com/wonbeomjang/5100109fa6151b032c6136b75ab1a5f3.js"></script>

output

```bash
41.11 ms / 5.2162MB
```

# 2. ONNX export

tensorrt 라이브러리를 이용하기 위해서 pytorch model을 onnx model로 변환해줘야한다.
torch.onnx를 통해 mobilenet을 export하자.

<script src="https://gist.github.com/wonbeomjang/beadfdfc74aa8dbc16de6ff539c8a4c5.js"></script>

그리고 만약 jupyter notebook이라면 tensorrt로 변환하는 동안 pytorch 때문에 gpu 메모리가 부족하여 변환에 실패한다.
따라서 exit를 사용하여 kernel을 종료하자.

<script src="https://gist.github.com/wonbeomjang/14520515b147699edc2966f5bf5bd234.js"></script>

# 3. Tensorrt 변환

trtexec을 통해 mobilenet model을 tensorrt model로 변환하자.

```bash
!trtexec --onnx=mobilnet_f32.onnx --saveEngine=mobilnet_f16_trt.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```

### -bash: trtexec: command not found

만약 trtexec을 찾을 수 없다고 오류를 띄운다면 bashrc file을 수정하여 아래의 path를 추가하자.

```bash
if [ -d /usr/src/tensorrt/bin/ ]; then
  export PATH=/usr/src/tensorrt/bin:$PATH
fi
```

# 4. TensorRT inference

먼저 trt.Runtime 객체를 만들어 runtime용 객체를 만들고 tensorrt file을 읽어들어와 engine을 context를 선언하자.

<script src="https://gist.github.com/wonbeomjang/46e8b292fa32a9c71bb84003cbd206bc.js"></script>

cuda를 용하여 intput, output용 메모리를 할당한다.
그후 inference를 위한 stream 객체를 선언한다.

<script src="https://gist.github.com/wonbeomjang/570a107abe058c240e42b57cb4e7baa8.js"></script>

이후 numpy array를 cuda memory에 copy하고 모델을 inferecne한 다음 d_output을 output ndarray에 copy한다.
마지막으로 병렬처리를 위한 stream threads을 syncronize하여 마무리한다.

이제 time을 측정해보자.

```python
from time import time
import os

# caching model to gpu
predict(input_batch)

cur = time()
for i in range(1000):
    predict(input_batch)

inference_time = time() - cur
model_size = os.path.getsize("mobilnet_f32_trt.trt") / 1e6

print(f"{inference_time:.2f} ms / {model_size:.4f}MB")
```

output

```bash
6.95 ms / 5.8507MB
```

## 결론

확실히 속도측면에서 이득을 보고있다.
하지만 최적화 옵션을 안 넣고 단순히 fp16으로 바꿔서 그런지 메모리측면에서는 이득을 보지 못했다.
이전 torch_tensorrt에서는 메모리측면에서도 이득을 보아 더 개선가능성이 있어보인다.

<table align="center">
    <tr align="center">
        <td></td>
        <td>Inference Time (ms)</td>
        <td>Model Parameter (MB)</td>
    </tr>
    <tr align="center">
        <td>MobileNetV2 float32</td>
        <td>37.80</td>
        <td>10.3278</td>
    </tr>
    <tr align="center">
        <td>MobileNetV2 float16</td>
        <td>41.11</td>
        <td>5.2162</td>
    </tr>
    <tr align="center">
        <td>TensorRT</td>
        <td>6.95</td>
        <td>5.8507</td>
    </tr>
</table>
