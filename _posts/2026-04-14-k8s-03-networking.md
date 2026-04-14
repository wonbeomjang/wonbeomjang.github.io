---
layout: post
title: "K8s 시리즈 03: Service, Ingress — 트래픽 라우팅과 외부 접근"
date: 2026-04-14 11:30:00 +0900
description: "ClusterIP, NodePort, LoadBalancer, Ingress 도메인 라우팅, HPA 오토스케일링, Taint/Toleration GPU 노드 배치"
categories: [infra]
tags: [kubernetes, service, ingress, networking, devops]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 시리즈**의 세 번째 글이다.
>
> - [01: Kubernetes란? 컨테이너부터 클러스터까지](/blog/2026/k8s-01-intro/)
> - [02: Pod, Deployment, Job, CronJob — K8s 워크로드 총정리](/blog/2026/k8s-02-workloads/)
> - **03: Service, Ingress — 트래픽 라우팅과 외부 접근** ← 현재 글
> - [04: ConfigMap, Secret, Storage — 설정과 데이터 관리](/blog/2026/k8s-04-config-storage/)
> - [05: Amazon EKS — 아키텍처와 Worker Node](/blog/2026/amazon-eks-guide/)
> - [06: EKS 네트워킹·보안·비용·운영](/blog/2026/k8s-06-eks-operations/)

이전 글에서 Pod과 Deployment로 앱을 실행하는 법을 배웠다. 하지만 Pod의 IP는 재시작할 때마다 바뀐다. 이번 글에서는 **안정적으로 접근하는 방법** — Service와 Ingress, 그리고 트래픽에 따른 **자동 스케일링(HPA)**과 **GPU 노드 배치** 전략을 다룬다.

---

# 1. Service — 내부 통신의 핵심

## 1.1 왜 Service가 필요한가?

Pod의 IP는 재시작마다 바뀐다. 다른 앱이 Pod IP를 직접 사용하면 Pod이 교체될 때 연결이 끊긴다.

Service는 Pod 집합 앞에 **고정된 DNS 이름과 IP**를 제공한다.

```
Service 없이:
  앱 A → Pod IP (10.0.1.5) → Pod 재시작 → IP 변경 → 연결 끊김!

Service 있으면:
  앱 A → backend.default.svc → Service가 Pod을 자동 선택 → 항상 연결
```

## 1.2 Service 타입

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-03-networking/service-types.png" class="img-fluid rounded z-depth-1" alt="Kubernetes Service 타입: ClusterIP, NodePort, LoadBalancer" %}

| Type                 | 접근 범위                | 용도                        | 비유      |
| -------------------- | ------------------------ | --------------------------- | --------- |
| **ClusterIP** (기본) | 클러스터 내부만          | 마이크로서비스 간 통신      | 내선 번호 |
| **NodePort**         | 노드 IP:Port로 외부 접근 | 테스트·디버깅 (운영 비권장) | 직통 전화 |
| **LoadBalancer**     | AWS ELB를 통한 외부 접근 | Ingress 없이 직접 외부 노출 | 콜센터    |

## 1.3 ClusterIP Service YAML

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend
spec:
  type: ClusterIP # 기본값, 생략 가능
  selector:
    app: backend # 이 Label을 가진 Pod에 트래픽 전달
  ports:
    - port: 80 # Service가 받는 포트
      targetPort: 8080 # Pod이 듣고 있는 포트
```

이 Service를 만들면 클러스터 내 어디서든 `backend.default.svc` DNS로 접근할 수 있다. Pod이 여러 개면 **자동으로 로드밸런싱**된다.

### DNS 형식

```
{service-name}.{namespace}.svc.cluster.local
```

같은 Namespace 안에서는 `backend`만으로도 접근 가능하다. 다른 Namespace의 Service에 접근하려면 `backend.other-ns.svc`를 사용한다.

---

# 2. Ingress — 외부에서 접근하기

## 2.1 왜 Ingress가 필요한가?

ClusterIP Service는 클러스터 **내부에서만** 접근 가능하다. 사용자가 브라우저로 접속하려면 **Ingress**가 필요하다.

Ingress는 **도메인/경로 기반**으로 외부 HTTP/HTTPS 트래픽을 적절한 Service로 라우팅한다.

```
하나의 도메인에 여러 Service 연결:

https://my-app.example.com/           → frontend Service
https://my-app.example.com/api        → backend Service
https://my-app.example.com/admin      → admin Service
```

## 2.2 전체 트래픽 흐름

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-03-networking/traffic-flow.png" class="img-fluid rounded z-depth-1" alt="Ingress 트래픽 라우팅 흐름" %}

1. 사용자가 `https://my-app.example.com` 접속
2. **DNS(Route53)**가 Load Balancer IP 반환
3. **Load Balancer(NLB/ALB)**가 트래픽을 Ingress Controller로 전달
4. **Ingress**가 TLS 종료 + 도메인/경로 기반 라우팅
5. **Service**가 Label Selector로 Pod 선택 + 로드밸런싱
6. **Pod**이 요청 처리

## 2.3 Ingress YAML

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - my-app.example.com
      secretName: tls-wildcard-cert # TLS 인증서
  rules:
    - host: my-app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 80
```

---

# 3. HPA — 자동 스케일링

## 3.1 HPA란?

HPA(Horizontal Pod Autoscaler)는 **CPU/메모리 사용률이나 커스텀 메트릭**에 따라 Pod 수를 자동으로 늘리거나 줄인다.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2 # 최소 2개
  maxReplicas: 20 # 최대 20개
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70 # CPU 70% 넘으면 스케일 아웃
```

| 설정                 | 설명                                       |
| -------------------- | ------------------------------------------ |
| `minReplicas`        | 트래픽이 없어도 최소 유지할 Pod 수         |
| `maxReplicas`        | 아무리 트래픽이 많아도 이 수를 넘지 않음   |
| `averageUtilization` | 이 수치를 넘으면 Pod 추가, 밑돌면 Pod 제거 |

### 주의사항

- Pod에 **`resources.requests`가 설정**되어 있어야 HPA가 동작한다 (비율 계산의 기준)
- 스케일 아웃은 빠르지만 스케일 인은 **기본 5분 대기** 후 진행 (안정화 기간)
- GPU Pod에는 CPU 기반 HPA가 의미 없을 수 있음 → 커스텀 메트릭(요청 수 등) 사용

---

# 4. Node Scheduling — GPU 노드 배치

LLM 엔지니어가 자주 마주치는 문제: "GPU Pod을 GPU 노드에만 배치하고, 일반 Pod은 CPU 노드에만 배치하고 싶다."

## 4.1 Node Selector

가장 간단한 방법. 노드의 Label을 기준으로 배치한다.

```yaml
spec:
  nodeSelector:
    accelerator: nvidia-a100 # 이 Label이 있는 노드에만 배치
```

## 4.2 Taint와 Toleration

GPU 노드에 **Taint**(오염)를 걸어서, 허용(Toleration)된 Pod만 배치되도록 한다.

```bash
# GPU 노드에 Taint 설정 (관리자)
kubectl taint nodes gpu-node-1 nvidia.com/gpu=true:NoSchedule
```

```yaml
# Pod에 Toleration 추가 (사용자)
spec:
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
```

| 개념           | 역할                                   | 비유                        |
| -------------- | -------------------------------------- | --------------------------- |
| **Taint**      | 노드에 "나는 특별하다" 표시            | "관계자 외 출입금지" 표지판 |
| **Toleration** | Pod에 "나는 그 노드에 갈 수 있다" 표시 | 출입증                      |

## 4.3 Node Affinity

더 세밀한 조건으로 노드를 선택한다.

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                  - p4d.24xlarge # A100 인스턴스
                  - p5.48xlarge # H100 인스턴스
```

| 방법                 | 복잡도 | 적합한 경우                           |
| -------------------- | ------ | ------------------------------------- |
| **nodeSelector**     | 낮음   | Label 하나로 충분할 때                |
| **Taint/Toleration** | 중간   | GPU 노드를 일반 Pod으로부터 보호할 때 |
| **Node Affinity**    | 높음   | 복잡한 조건 (인스턴스 타입, AZ 등)    |

---

# 참고 문헌

- [Kubernetes 공식 문서 — Service](https://kubernetes.io/ko/docs/concepts/services-networking/service/)
- [Kubernetes 공식 문서 — Ingress](https://kubernetes.io/ko/docs/concepts/services-networking/ingress/)
- [Kubernetes 공식 문서 — HPA](https://kubernetes.io/ko/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Kubernetes 공식 문서 — Taint와 Toleration](https://kubernetes.io/ko/docs/concepts/scheduling-eviction/taint-and-toleration/)
- [Kubernetes 공식 문서 — Node Affinity](https://kubernetes.io/ko/docs/concepts/scheduling-eviction/assign-pod-node/)
