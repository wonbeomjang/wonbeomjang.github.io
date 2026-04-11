---
layout: post
title: "Integral Neural Network"
date: 2023-06-22 00:00:00 +0900
description: Integral Neural Network 논문 리뷰 — 연속 파라미터 공간 활용 (CVPR 2023)
categories: [paper]
tags: [backbone, paper, cvpr, vit]
giscus_comments: true
related_posts: true
---

> [논문 링크](https://arxiv.org/abs/2206.12418)

# Introduction

이 논문은 CVPR2023에 accept된 논문이고 award 후보에도 올랐다. 주된 초점은 모델 사이즈별로 성능하락 없이 pruning하는 내용인데, 상당히 아이디어도 괜찮고 앞으로 응용가능성도 있어보인다. 하지만 가벼운 네트워크만 실험을 하여 EfficientNet-L과 같은 무거둔 네크워크나 ViT와 같은 self-attention 메커니즘을 사용한 네트워크의 결과는 없다. 따라서 실제로 사용하려면 이론적토대가 더 필요할 것으로 보인다. 이제 논문을 살표보자.
기존의 DNN은 많은 분야에서 좋은 성능을 냈다. Kolmogorov superposition theorem과 universal approximation theorem에서 DNN은 어떠한 continuous multivariate function이라도 모사할 수 있다고 이야기한다. 이러한 이론에 따라서 DNN은 발전했고 파라미터 수도 많아졌다. 연구자들을 이를 극복하기 위해 경량화 기법으로 pruning, quantization, NAS를 사용하여 모델의 크기를 줄였다. 하지만 이와 같은 방법은 모델 사이즈가 줄어듬에 따라 성능하락이 발생했고 각각의 사이즈의 모델을 따로 학습시켜야한다는 단점이 있다.
따라서 저자는 neural network에서 사용하는 discrete한 representaiton을 continuous representation으로 바꾸어 inference시 quadrature approximation procedure를 통해 여러 크기의 모델을 만들 수 있도록 제안했다. 따라서 기존에 있는 CNN, FC 연산과 같은 discrete operation을 integral operator로 교체하는 과정을 거치게 된다.

# Neural Networks and Integral Operators

<p align="center">
    <img src="/assets/post/image/integral-neural-network/fig2.png" width="80%">
</p>

설명을 하기 앞서 integral operator에 대한 설명을 먼저 하겠다.$$W(x),S(x)$$ 가 unitvariate function이라고 하자. 이 때 다음과 같은 식이 성립한다.

$$\int_0^1 W(x)S(x)dx \approx \sum_{i=0}^nq_iW(x_i)S(x_i)=\vec{w_q} \cdot \vec{s}$$

이때 다음과 같은 식이 성립한다.

$$\vec{w_q}=(q_0W(x_0),...,q_nW(x_n))$$

$$\vec{s}=(S(x_0),...,S(x_n))$$

$$\vec{P}^x=(x_0,...,x_n), 0 = x_0 < x_1 < ... < x_{n-1} < x_n = 1$$

위 식은 "두 univariate function의 곱의 적분은 수치적분을 이용한 두 벡터의 내적에 근사한다"는 것을 의미한다. 이$$(P, q)$$ 쌍은 a numerical integration method라고 부른다. 간단하게 생각하자면 고등학교 때 배운 정적분과 부분 적분의 관계를 떠올리면 된다.

## DNNs layers as integral operators

기본적인 이론 토대를 만들었으니 이제 어떻게 적용할 수 있는지 보자.

### Convolution or cross-correlation layer

$$\mathbf{x^s}$$ 는 dimension을 표현하는 scalar 혹은 vector라고 정의하자. Convolution layer는 multi-channel을 다루기 때문에 이를 반영해야 한다. Convolution의 continuous operation은 integral로 정의되므로 다음과 같은 식을 따른다.$$\lambda$$ 는 trainable parameter를 의미한다.

일단 Convolution weight, Input, Output을 다음과 같이 표현하자.

$$F_W(\lambda,x^{out},x^{in}, \mathbf{x^s}), F_I(x^{in}, \mathbf{x^s}), F_O(x^{out}, \mathbf{x^{s^\prime}})$$

Convolution operation을 Integral operator로 다음과 같이 표현할 수 있다.

$$F_O(x^{out},x^{s^\prime})=\int_\Omega F_W(\lambda,x^{out},x^{in}, \mathbf{x^s})F_I(x^{in}, \mathbf{x^s}+\mathbf{x^{s^\prime}})dx^{in}d\mathbf{x^s}$$

### Fully-connected layer

Linear layer는 기본적으로 matrix multiplication 연산으로 이루어져있으며 vector에서 vector로의 변환 연산이다. 또한 이는 1차원 연산이기 때문에 FC weight, input, output을 다음과 같이 정의한다.

$$F_W(\lambda,x^{out},x^{in}), F_I(x^{in}), F_O(x^{out})$$

그리고 FC 연산을 다음과 같이 정의한다.

$$F_O(x^{out})=\int_0^1 F_W(\lambda,x^{out},x^{in})F_I(x^{in})dx^{in}$$

### Pooling and Activation Functions

Pooling 연산은 간단하게 정의된다. Average pooling은 constant function을 이용한 convolution 연산으로 정의되고, max pooling은 signal discretization으로 정의할 수 있다. 또한 activation function은 discrete한 representation에서 적용하면 되는데 그 이유는 다음의 식이 성립하기 때문이다.

$$\mathcal{D}(ActFunction(x),P_x)=ActFunction(\mathcal{D}(x,P_x))$$

$$\mathcal{D}$$ 는 주어진 partition$$P_x$$에 대해 discretization operation을 말하는 것이다. 즉, Continuous signal의 activate function을 discretizing한 것은 discretized signal에 activation function을 적용한 것과 동일하다는 관계식이 성립한다.

### Evaluation and backpropagation through integration

Integral Neural Network (INN)은 빠른 evalution을 위하여 integral kernel을 discretization하는 과정을 거치게 된다. 이를 통해 기존의 conventional layer에 weight을 전달할 수 있고, pytorch와 같은 framework나 GPU와 호환이 된다.
Backpropagation은 기존과 같은 chain-rule이 사용된다. 이는 Appendix A에 설명이 들어가있는데 간단하게 lemma만 보자면 다음과 같다.

> **Lemma 1**
> (Neural Integral Lemma) Given that an integral kernel$$F(λ, x)$$ is smooth and has continuous partial derivatives$$\frac{\partial(\lambda,x)}{\partial\lambda}$$ on the unit cube$$[0, 1]^n$$ n, any composite quadrature can be represented as a forward pass of the corresponding discrete operator. The backward pass of the discrete operator corresponds to the evaluation of the integral operator with the kernel$$\frac{\partial(\lambda,x)}{\partial\lambda}$$ using the same quadrature as in the forward pass.

## Continuous parameters representation

더 풍부하고 일반화된 continuous parameter representation을 위해서 inference time에 어떠한 해상도(sampling rate)로든 sampling을 하야한다. 따라서 저자는 continuous한 weight을 \[0, 1]에서 존재하는 line segment에 interpolation kernel의 linear combination으로 정의한다. 따라서 다음과 같이 나타낼 수 있다.

$$F_W(\lambda,x)=\sum_{i=0}^m\lambda_i u(xm-i)$$

여기서$$m$과$$n$$은 interpolation node의 개수와 그들의 값이다.

<p align="center">
    <img src="/assets/post/image/integral-neural-network/fig4.png" width="80%">
</p>

설명이 어렵게 되어있지만 개념은 간단하다. 저자들은 kernel wieght를 각 \[0, 1] 사이의 균일한 segment로 저장하고 이를 interpolation을 통해 continuous한 kernel과 representaion을 제작한다. 이때 cubic spline interpolation을 사용하는데 이는 GPU상에서 빠르지만 linear interpolation보다 더 정확한 정보를 담을 수 있기 때문이다.
따라서 fully-connected layer의 weight는 다음과 같이 저장된다.

$$F_W(\lambda,x^{out},x^{in})=\sum_{i,j}\lambda_{i,j}u(x^{out}m^{out}-i)u(x^{in}m^{in}-j)$$

또한 evaluation을 위해 discrete하게 export 할 때 다음과 같이 export 하게 된다.

$$W_q[k,l]=q_lW[k,l]=q_lF_W(\lambda,P_k^{out},P_l^{in})$$

$$\vec{P}^{out}=\{kh^{out}\}_k, \vec{P}^{in}=\{lh^{in}\}_k$$

### Trainable partition

저자는 처음에 fixed sampling step으로 uniform한 partition을 만들 생각이었다. 하지만 non-uniform한 sampling이 partition size를 키우지 않고 numerical integration을 향상시킬 수 있다는 것을 발견했다. 따라서 trainable한 partition을 도입해 자유도를 늘렸으며 이를 통해 좀 더 smooth하고 효율적인 partition을 할 수 있게 되었다. 후술하겠지만 이는 새로운 pruning 방법에 쓰이게된다. 따라서 partition parameterization$$\vec{P}$$ 는 다음과 같은 식을 따르게 된다.

$$\vec{\delta}_{norm}=\frac{\vec{\delta}^2}{sum(\vec{\delta}^2)}, \vec{P}=cumsum(\vec{\delta}_{norm})$$

# Training Integral Neural Networks

딥러닝 방법론이 많아지면서 현재는 ResNet과 같은 좋은 network가 존재한다. 따라서 이를 활용한다면 INN에 좋은 initialization이 될 수 있다. 따라서 저자들은 기존 discrete network를 smooth structure로 만들기 위해 weight를 permute하는 방법론을 제시했다.

### Conversion of DNNs to INNs

<p align="center">
    <img src="/assets/post/image/integral-neural-network/fig5.png" width="80%">
</p>

network를 가능하면 smooth한 structure로 만들기 위해서 weight tensor의 특정방향의 total variation이 최소가 되도록 만들어아햔다. 이 문제는 많이 알려진 Traveling Salesman Problem (TSP)문제로 환원될 수 있다. 이 task에서는$$c^{out}$$ dimension에 따라 weight tensor는 city로 대응되고 distance는 total variance로 대응된다. 따라서 optimal permutation은 route로 대응되어 다음 식을 최소화 하는 것으로 문제를 해결하게 된다.

$$min_{\sigma \in S_n}\sum|W[\sigma(i)]-W[\sigma(i+1)]|$$

$$W$$ 는 weight tensor,$$\sigma$$ 는 permutation,$$\sigma(i)$$ 는 i-th element의 새로운 위치이다.

### Optimization of continuous weights

INN은 보통의 gradient descent-based method를 사용할 수 있으며 Lemma 1을 사용하여 다음의 학습 알고리즘으로 학습을 진행하게 된다.

<p align="center">
    <img src="/assets/post/image/integral-neural-network/algorithm1.png" width="60%">
</p>

또한 매 iteration마다 partition size가 달라질 수 있기 때문에 다음과 같은 식을 objective로 설정하여 다른 cube partition간 차이를 최소화한다.

$$|Net(X,P_1)-Net(X,P_2)|\leq|Net(X,P_1)-Y|+|Net(X,P_2)-Y|$$

# Expertimant

<p align="center">
    <img src="/assets/post/image/integral-neural-network/pipline.png" width="100%">
</p>

실험 시나리오는 3개로 설정했다.

## Pipeline A. Comparison with discrete NNs

<p align="center">
    <img src="/assets/post/image/integral-neural-network/table1.png" width="60%">
</p>
Discrete 모델을 변환하여 finetuning한 INN이 discrete 모델보다 성능이 비슷하거나 더 좋았다. 하지만 scratch model은 성능이 안 좋았는데 이는 batch normalization을 사용하지 않아서 그렇다고 한다. Super Resolution에서도 비슷한 결과가 나왔다.

## Pipeline B. Structured pruning without fine-tuning through conversion to INN

<p align="center">
    <img src="/assets/post/image/integral-neural-network/table2.png" width="60%">
</p>

Section 4에서 partitioning을 finetuning할 수 있다고 했다. 따라서 DNN을 INN으로 변환할 때 partition tuning 유무에 따라 성능비교를 했을 때 partition tuning을 한 모델이 성능이 좋은 것을 알 수 있다.

## Pipeline C. Structured pruning without fine-tuning of discrete NNs

<p align="center">
    <img src="/assets/post/image/integral-neural-network/fig1.png" width="100%">
</p>

기존의 pruning 방법과도 비교해봤다. 그 결과 INN을 통해서 pruning하는 것이 성능하락이 적었으며 몇 개의 데이터로만 partition tuning을 했을 때 성능하락이 가장 적은 것을 알 수 있었다.

# Comment

합리적이고 흥미로운 논문인 것 같다. Training method를 보았을 때 DNN과 INN의 변환이 계속 일어나 학습시간이 느릴 수 있으나 기존 대형모델을 INN으로 만들어 finetuning한 후 크기별로 export하여 많은 device에 사용할 수 있을 것 같다. 하지만 비교적 가벼운 모델을 위주로 실험하고 ViT 계열의 실험은 안들어가있어 실제로 이를 원래 목적대로 사용할 수 있을지는 의문이다. 이론적 토대가 더 만들어진다면 임팩트가 있는 방법론이 되지 않을까 싶다.
