---
layout: post
title: "error: command 'aarch64-linux-gnu-gcc' failed with exit status 1"
date: 2023-02-24 18:50:11 +0900
description: jetson nano pycuda install error
categories: [jetson-nano, error]
tags: [jetson, pycuda, troubleshooting]
giscus_comments: true
related_posts: true
---

Jetson nano에 pycuda를 설치하다가 다음과 같은 오류를 만났다.

```
aarch64-linux-gnu-gcc -pthread -fwrapv -Wall -O3 -DNDEBUG -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -DBOOST_ALL_NO_LIB=1 -DBOOST_THREAD_BUILD_DLL=1 -DBOOST_MULTI_INDEX_DISABLE_SERIALIZATION=1 -DBOOST_PYTHON_SOURCE=1 -Dboost=pycudaboost -DBOOST_THREAD_DONT_USE_CHRONO=1 -DPYGPU_PACKAGE=pycuda -DPYGPU_PYCUDA=1 -DHAVE_CURAND=1 -Isrc/cpp -Ibpl-subset/bpl_subset -I/usr/lib/python3/dist-packages/numpy/core/include -I/usr/include/python3.6m -c src/cpp/cuda.cpp -o build/temp.linux-aarch64-3.6/src/cpp/cuda.o

    In file included from src/cpp/cuda.cpp:4:0:
    src/cpp/cuda.hpp:14:10: fatal error: cuda.h: No such file or directory
     #include <cuda.h>
              ^~~~~~~~
    compilation terminated.
    error: command 'aarch64-linux-gnu-gcc' failed with exit status 1
```

## 첫 번째 시도는 흔히하는 cuda path 설정이다.

~/.bachrc 에서 아래를 추가해준다.

```
if [ -d /usr/local/cuda/ ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi
```

하지만 이전에 추가했던 것이었고 내겐 해당사항이 없었다.

## 두 번째 시도 cuda library incude

사실 설치파일에서 cuda.h를 못찾는 것으로 cuda library를 인식하게 만들어주면 된다.
하지만 bashrc에서 어떻게 수정하는지 몰랐고 다음과 같은 명령어를 통해 설치완료했다.

```bash
sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
```

그러자 잘 설치되었다.

```
Installing collected packages: pycuda
  Running setup.py install for pycuda ... done
Successfully installed pycuda-2022.1
```
