---
layout: post
title: "Kubernetes 아키텍처 — Control Plane과 Node"
date: 2026-07-07 03:00:00 +0900
description: "etcd, kube-apiserver, scheduler, kubelet — 클러스터를 움직이는 컴포넌트들의 역할과 설계 의도를 kind 클러스터에서 직접 확인한다"
categories: [infra]
tags: [kubernetes, infra, architecture, etcd, scheduler]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 입문 시리즈**의 세 번째 글이다.
>
> - [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)
> - [02: 내 노트북에 클러스터 만들기 — kind와 kubectl](/blog/2026/k8s-02-local-setup/)
> - **03: Kubernetes 아키텍처 — Control Plane과 Node** ← 현재 글
> - [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)
> - [05: 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet](/blog/2026/k8s-05-workloads/)
> - [06: 네트워킹 — Service와 Ingress](/blog/2026/k8s-06-networking/)
> - [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)
> - [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)
> - [09: 확장과 생태계 — Operator와 CNCF Projects](/blog/2026/k8s-09-operator-cncf/)
>
> 이 시리즈의 커리큘럼은 SK Devocean의 [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua) 글의 학습 로드맵을 바탕으로 구성했다.

[지난 글](/blog/2026/k8s-02-local-setup/)에서 kind로 클러스터를 만들고 kubectl로 접속까지 해봤다. `kind create cluster` 한 줄로 만들어진 그 클러스터 안에는 사실 여러 개의 프로그램이 각자 다른 역할을 맡아 돌아가고 있다. 이번 글에서는 그 내부를 해부한다.

Kubernetes 클러스터는 회사와 비슷하다. **Control Plane은 본사 경영진**이다. 회사의 목표를 정하고("웹 서버 3대를 항상 유지한다"), 장부를 관리하고, 어느 지점에 어떤 일을 맡길지 결정한다. **Node는 현장**이다. 본사의 지시를 받아 실제로 일(컨테이너)을 수행한다. 중요한 건 경영진이 현장에 직접 내려가서 일하지 않는다는 점이다. 지시는 문서(API)로 전달되고, 현장 반장(kubelet)이 그걸 읽고 실행한다.

이 글에서는 각 컴포넌트가 "무엇을 하는지"와 함께 "왜 그렇게 설계됐는지"를 본다. 여기서 잡은 그림이 다음 글(Pod가 뜨는 과정)의 뼈대가 된다.

---

# 1. 전체 아키텍처 한눈에 보기

먼저 전체 그림부터 본다. 공식 문서 기준으로 클러스터는 크게 **Control Plane 컴포넌트**와 **Node 컴포넌트**로 나뉜다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-03-architecture/cluster-architecture.png" class="img-fluid rounded z-depth-1" alt="Kubernetes 클러스터 아키텍처 — Control Plane(kube-apiserver, etcd, scheduler, controller-manager)과 Node(kubelet, kube-proxy, 컨테이너 런타임, Pod), 모든 통신이 kube-apiserver로 모인다" %}

이 그림에서 핵심 관찰은 세 가지다.

- **모든 화살표가 kube-apiserver로 모인다.** 사용자도, 스케줄러도, 컨트롤러도, 노드의 kubelet도 서로 직접 대화하지 않고 전부 API 서버를 거친다. 컴포넌트끼리 몰래 주고받는 사이드 채널이 없다.
- **etcd는 kube-apiserver 뒤에 숨어 있다.** 클러스터의 모든 상태가 저장되는 곳이지만, 여기에 접근하는 컴포넌트는 API 서버뿐이다.
- **kubelet은 컨테이너를 직접 만들지 않는다.** CRI(Container Runtime Interface)라는 표준 인터페이스를 통해 컨테이너 런타임에 실행을 위임한다.

이 구조 덕분에 각 컴포넌트는 "API 서버에 저장된 상태를 읽고, 자기 몫의 일을 하고, 결과를 다시 API 서버에 기록"하는 단순한 패턴으로 동작한다. 컴포넌트 하나가 죽어도 나머지는 API 서버만 살아 있으면 계속 일할 수 있다. 이제 하나씩 뜯어보자.

---

# 2. Control Plane — 클러스터의 두뇌

Control Plane은 클러스터 전체에 대한 결정을 내리는 곳이다. 새 Pod를 어느 노드에 배치할지, 복제본 수가 모자라면 언제 채울지 같은 판단이 전부 여기서 일어난다.

## 2.1 etcd — 회사의 장부

**etcd는 클러스터의 모든 데이터가 저장되는 분산 key-value 저장소다.** 공식 문서는 etcd를 "일관성 있고 고가용성인 key-value 저장소로, 모든 클러스터 데이터의 백업 저장소(backing store)"라고 정의한다. 어떤 Pod가 어디에 떠 있는지, Deployment 설정이 뭔지, Secret에 뭐가 들었는지 — 클러스터의 "진실"은 전부 etcd에 있다.

회사 비유로는 **등기부이자 장부**다. 장부가 사라지면 회사가 뭘 가졌고 뭘 하기로 했는지 아무도 모른다. 그래서 etcd에는 두 가지 설계적 특징이 있다.

**첫째, Raft 합의 알고리즘으로 여러 대가 하나처럼 동작한다.** etcd는 리더 기반(leader-based)으로, 합의가 필요한 요청은 리더가 처리하고 팔로워는 요청을 리더에게 전달한다. 쓰기가 커밋되려면 전체 멤버의 과반, 즉 (n/2)+1개가 동의해야 한다. 그래서 프로덕션에서는 3대나 5대처럼 **홀수**로 구성한다. 3대 클러스터는 과반이 2이므로 1대가 죽어도 버틴다. 4대로 늘려도 과반이 3이 되어 허용 장애 수는 똑같이 1대다. 즉 짝수는 서버만 더 쓰고 얻는 게 없다.

**둘째, 접근이 극도로 제한된다.** 공식 문서는 "etcd에 대한 접근은 클러스터의 root 권한과 동등하므로, 이상적으로는 API 서버만 접근해야 한다"고 명시한다. etcd를 직접 읽고 쓸 수 있으면 인증도 권한 검사도 다 우회할 수 있기 때문이다. 그래서 그림에서 etcd로 향하는 화살표는 kube-apiserver 하나뿐이다.

## 2.2 kube-apiserver — 모든 통신의 관문

**kube-apiserver는 Kubernetes API를 노출하는 Control Plane의 프론트엔드다.** kubectl 명령도, 스케줄러의 결정도, kubelet의 상태 보고도 전부 이 문을 지난다.

비유하자면 **회사의 민원 창구**다. 아무나 장부(etcd)에 손대게 두지 않고, 창구에서 신원 확인과 권한 확인을 거친 요청만 장부에 반영한다. 요청 하나가 처리되는 과정은 파이프라인으로 정리된다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-03-architecture/apiserver-request-pipeline.png" class="img-fluid rounded z-depth-1" alt="kube-apiserver 요청 처리 파이프라인 — 인증, 인가, 어드미션 컨트롤, 오브젝트 검증을 거쳐 etcd에 저장된다" %}

- **인증(Authentication)**: 요청을 보낸 게 누구인지 확인한다. 클라이언트 인증서, 토큰 등 여러 모듈이 순서대로 시도되고, 전부 실패하면 401을 돌려준다.
- **인가(Authorization)**: 그 사용자가 그 동작(예: default 네임스페이스에서 Pod 생성)을 할 권한이 있는지 확인한다. RBAC이 대표적인 인가 방식이고, 거부되면 403이다. 자세한 건 [08편](/blog/2026/k8s-08-rbac/)에서 다룬다.
- **어드미션 컨트롤(Admission Control)**: 통과한 요청을 **수정하거나 거부**할 수 있는 마지막 관문이다. 기본값을 채워 넣거나(예: 리소스 기본 할당), 정책 위반 요청을 막는다. 읽기 요청에는 적용되지 않고, 생성·수정·삭제 같은 변경 요청에만 동작한다.

이 파이프라인을 통과한 오브젝트만 etcd에 저장된다. 여기서 중요한 설계 의도가 나온다. **모든 통신을 한 관문으로 모으면 인증·인가·감사(audit)를 한 곳에서 처리할 수 있다.** 컴포넌트끼리 직접 통신하는 구조였다면 보안 검사를 컴포넌트 쌍마다 구현해야 했을 것이다. 또한 API 서버 자체는 상태를 etcd에 두기 때문에(stateless), 공식 문서 표현대로 여러 인스턴스를 띄워 수평 확장할 수 있다.

## 2.3 kube-controller-manager — 온도조절기들의 집합

**kube-controller-manager는 Kubernetes API의 동작을 구현하는 컨트롤러들을 실행하는 컴포넌트다.** 컨트롤러(controller)는 Kubernetes 설계의 심장에 해당하는 개념이라 조금 자세히 본다.

공식 문서가 드는 비유는 **온도조절기(thermostat)**다. 온도를 25도로 설정하는 건 "바라는 상태(desired state)"를 알려주는 것이고, 실제 방 온도는 "현재 상태(current state)"다. 온도조절기는 이 둘을 계속 비교하면서 장비를 켜고 꺼서 현재 상태를 바라는 상태에 가깝게 만든다.

Kubernetes의 컨트롤러도 똑같다. **끝나지 않는 루프(control loop)를 돌면서 클러스터 상태를 관찰하고, 바라는 상태와 다르면 API 서버를 통해 변경을 요청한다.** 이 과정을 reconcile(조정)이라 부른다.

- 바라는 상태: 리소스의 `spec` 필드에 적힌 내용 (예: "replicas: 3")
- 현재 상태: 실제 클러스터의 모습 (예: "지금 Pod가 2개뿐")
- 컨트롤러의 일: 둘의 차이를 발견하고 메꾼다 (예: Pod 1개 생성 요청)

여기서 눈여겨볼 설계 의도가 두 가지다. 첫째, **컨트롤러는 직접 일하지 않고 API 서버에 요청만 남긴다.** 예를 들어 Job 컨트롤러는 Pod를 직접 실행하는 게 아니라 "Pod를 만들어 달라"고 API 서버에 기록하고, 실제 실행은 노드의 kubelet이 담당한다. 둘째, **하나의 거대한 루프 대신 작은 컨트롤러 여러 개를 쓴다.** Node 컨트롤러, Job 컨트롤러, EndpointSlice 컨트롤러, ServiceAccount 컨트롤러처럼 각자 한 가지 상태만 책임진다. 하나가 오작동해도 나머지는 계속 돌고, 각 컨트롤러의 로직도 단순해진다. kube-controller-manager는 이 컨트롤러들을 하나의 프로세스로 묶어 실행하는 바이너리다.

이 "선언하면 컨트롤러가 알아서 맞춘다"는 패턴은 [05편의 Deployment](/blog/2026/k8s-05-workloads/)와 [09편의 Operator](/blog/2026/k8s-09-operator-cncf/)에서 계속 다시 만난다.

## 2.4 kube-scheduler — 배치 담당자

**kube-scheduler는 아직 노드가 배정되지 않은 Pod를 찾아서 적절한 노드에 할당하는 컴포넌트다.** 회사 비유로는 **인사팀의 배치 담당자**다. 새 업무(Pod)가 들어오면 어느 지점(Node)에 맡길지 결정하되, 결정만 하고 실행은 현장에 맡긴다.

공식 문서 기준으로 스케줄러는 노드 선택을 **2단계**로 수행한다.

| 단계                     | 하는 일                                                                 | 예시                                                                               |
| ------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **1. 필터링(Filtering)** | Pod를 배치할 수 **있는** 노드(feasible node)만 남기고 나머지를 걸러낸다 | 요청한 CPU/메모리가 부족한 노드 제외 (`PodFitsResources` 필터)                     |
| **2. 스코어링(Scoring)** | 살아남은 노드들에 점수를 매겨 순위를 정한다                             | 활성화된 스코어링 규칙에 따라 채점 후 **최고점 노드**를 선택, 동점이면 무작위 선택 |

필터링 결과가 빈 리스트면, 즉 배치 가능한 노드가 하나도 없으면 그 Pod는 스케줄되지 못한 채 남는다. 나중에 자원이 생기거나 조건이 바뀌면 그때 배치된다. `kubectl get pods`에서 가끔 보게 되는 `Pending` 상태의 대표적인 원인이 바로 이것이고, [04편](/blog/2026/k8s-04-pod/)에서 직접 확인해 본다.

왜 "필터링 후 스코어링"일까. 조건을 만족하는지(하드 제약)와 얼마나 좋은 자리인지(소프트 선호)를 분리하면, 자격 미달 노드를 빠르게 쳐낸 뒤 남은 후보만 정밀하게 비교할 수 있다. 결정이 끝나면 스케줄러는 "이 Pod는 이 노드로"라는 바인딩 결과를 API 서버에 알릴 뿐, 노드에 직접 명령하지 않는다.

---

# 3. Node — 일이 실제로 벌어지는 곳

Node 컴포넌트는 모든 노드에서 실행되며, Pod를 돌리고 런타임 환경을 제공한다. 본사가 아무리 계획을 잘 세워도 실제로 컨테이너를 띄우는 건 현장이다.

## 3.1 kubelet — 현장 반장

**kubelet은 각 노드에서 실행되는 에이전트로, Pod 안의 컨테이너들이 확실히 실행되도록 책임진다.** 공식 문서 표현을 빌리면, kubelet은 전달받은 PodSpec에 기술된 컨테이너들이 실행 중이고 건강한지(healthy) 확인한다. 반대로 Kubernetes가 만들지 않은 컨테이너는 관리하지 않는다.

비유하자면 **현장 반장**이다. 본사(API 서버)에서 "이 노드에 이 Pod를 실행하라"는 지시가 내려오면, 반장이 그걸 읽고 작업자(컨테이너 런타임)에게 일을 시키고, 진행 상황을 본사에 보고한다. 스케줄러가 자리만 정해 주면, 그 자리에서 실제로 컨테이너를 띄우고 살아 있는지 감시하는 건 전부 kubelet 몫이다.

## 3.2 컨테이너 런타임과 CRI — 실행의 위임

**컨테이너 런타임은 컨테이너 실행을 실제로 담당하는 소프트웨어다.** 여기서 Kubernetes 설계의 또 다른 특징이 나온다. kubelet은 컨테이너를 직접 만들지 않고, **CRI(Container Runtime Interface)**라는 gRPC 기반 표준 프로토콜로 런타임에 위임한다.

왜 이렇게 나눴을까. CRI라는 인터페이스만 지키면 어떤 런타임이든 갈아 끼울 수 있게 하기 위해서다. 공식 문서 기준으로 Kubernetes는 containerd, CRI-O 같은 CRI 구현체를 지원한다. kubelet 입장에서는 "컨테이너 하나 띄워줘"라는 gRPC 호출만 하면 되고, 그 아래에서 이미지를 받고 프로세스를 격리하는 세부 구현은 런타임의 책임이다. 표준 인터페이스 하나로 구현체를 교체 가능하게 만드는 이 패턴은 잠시 뒤 CNI에서도 똑같이 반복된다.

## 3.3 kube-proxy — Service를 실현하는 교통 정리

**kube-proxy는 각 노드에서 네트워크 규칙을 유지하면서 Service 개념을 실제로 구현하는 네트워크 프록시다.** 아직 Service를 안 배웠으니 여기서는 큰 그림만 잡는다. Kubernetes에서는 여러 Pod를 하나의 가상 IP(Service)로 묶어서 접근하는데, "이 가상 IP로 온 트래픽을 실제 Pod들에게 나눠주는 규칙"을 노드마다 세팅하는 게 kube-proxy다.

이름과 달리 대부분의 모드에서 트래픽을 직접 중계하지 않는다. 대신 커널의 패킷 처리 규칙을 설정해 두고 커널이 트래픽을 전달하게 한다. 공식 문서 기준으로 Linux에서는 **iptables 모드(기본값)**, ipvs 모드, nftables 모드를 지원한다. iptables 모드는 Service의 가상 IP로 향하는 패킷을 백엔드 Pod로 DNAT하는 규칙을 커널 netfilter에 깔아두는 방식인데, 클러스터가 아주 커지면 규칙 수가 많아져 성능 이슈가 생길 수 있고, 그 대안으로 nftables 모드가 등장했다(ipvs 모드는 공식 문서 기준 deprecated 상태다). 흥미로운 점 하나: 공식 문서는 kube-proxy를 **optional**로 분류한다. Service 트래픽 처리를 직접 구현하는 네트워크 플러그인을 쓰면 kube-proxy 없이도 클러스터가 동작한다. 자세한 동작은 [06편](/blog/2026/k8s-06-networking/)에서 다룬다.

## 3.4 네트워크 플러그인(CNI) — Pod들의 도로망

마지막 조각은 **네트워크 플러그인**이다. Pod마다 IP를 할당하고, 서로 다른 노드에 있는 Pod끼리 통신할 수 있게 만드는 역할을 한다. 회사 비유로는 지점과 지점 사이를 잇는 **도로망 공사**다. kube-proxy가 "어디로 보낼지" 교통 규칙을 정한다면, CNI 플러그인은 그 전에 길 자체를 깔아준다.

Kubernetes는 여기서도 구현을 직접 들고 있지 않고 CNI(Container Network Interface)라는 표준에 위임한다. 그래서 Calico, Cilium 같은 다양한 구현체가 존재한다. 우리가 쓰는 **kind는 kindnetd라는 자체 기본 CNI를 탑재**하고 있다. kind 공식 문서는 이를 "표준 CNI 플러그인 기반의 단순한 네트워킹 구현"이라고 설명하며, IP 마스커레이드 같은 기본 기능을 처리한다. 설정에서 `disableDefaultCNI: true`를 주면 kindnetd 대신 Calico 같은 다른 CNI를 직접 설치할 수도 있다(공식 문서상 파워 유저 기능이다).

---

# 4. 컴포넌트 요약 표

지금까지 나온 컴포넌트를 한 표로 정리한다.

| 컴포넌트                    | 위치          | 역할                                                               | 비유             |
| --------------------------- | ------------- | ------------------------------------------------------------------ | ---------------- |
| **etcd**                    | Control Plane | 클러스터 모든 상태의 유일한 저장소, Raft 합의로 고가용성           | 회사 장부·등기부 |
| **kube-apiserver**          | Control Plane | 모든 통신의 관문, 인증→인가→어드미션, 유일하게 etcd에 접근         | 민원 창구        |
| **kube-controller-manager** | Control Plane | desired state와 current state의 차이를 reconcile하는 컨트롤러 묶음 | 온도조절기       |
| **kube-scheduler**          | Control Plane | 미배정 Pod에 노드를 골라줌 (필터링→스코어링)                       | 인사 배치 담당자 |
| **kubelet**                 | Node          | 지시받은 Pod의 컨테이너를 실행·감시하는 노드 에이전트              | 현장 반장        |
| **컨테이너 런타임**         | Node          | CRI를 통해 위임받아 컨테이너를 실제로 실행                         | 현장 작업자      |
| **kube-proxy**              | Node          | Service 트래픽이 Pod에 도달하도록 커널에 규칙 설정                 | 교통 정리 요원   |
| **네트워크 플러그인(CNI)**  | Node          | Pod IP 할당과 Pod 간 통신 경로 구성 (kind는 kindnetd)              | 도로망 공사      |

---

# 5. 실습 — kind 클러스터에서 직접 눈으로 확인하기

이론은 충분하다. 02편에서 만든 kind 클러스터에서 이 컴포넌트들을 직접 찾아보자. Kubernetes는 시스템 컴포넌트를 `kube-system`이라는 네임스페이스에 모아둔다.

```bash
kubectl get pods -n kube-system
```

기본 단일 노드 kind 클러스터라면 대략 이런 출력을 보게 된다 (이름 뒤의 해시와 AGE는 환경마다 다르다).

```text
NAME                                         READY   STATUS    RESTARTS   AGE
coredns-xxxxxxxxxx-xxxxx                     1/1     Running   0          10m
coredns-xxxxxxxxxx-xxxxx                     1/1     Running   0          10m
etcd-kind-control-plane                      1/1     Running   0          10m
kindnet-xxxxx                                1/1     Running   0          10m
kube-apiserver-kind-control-plane            1/1     Running   0          10m
kube-controller-manager-kind-control-plane   1/1     Running   0          10m
kube-proxy-xxxxx                             1/1     Running   0          10m
kube-scheduler-kind-control-plane            1/1     Running   0          10m
```

이 출력에는 오늘 배운 내용이 전부 들어 있다. 한 줄씩 해석해 보자.

- **`etcd-`, `kube-apiserver-`, `kube-controller-manager-`, `kube-scheduler-`**: 2장에서 본 Control Plane 4인방이 그대로 보인다. 이름 끝에 노드 이름(`-kind-control-plane`)이 붙어 있는 게 특징인데, 이들이 **static Pod**이기 때문이다. static Pod는 API 서버를 거치지 않고 **kubelet이 지정된 디렉터리의 매니페스트 파일을 읽어 직접 관리**하는 Pod다. 생각해 보면 당연하다 — API 서버 자신을 띄우는 일을 API 서버에게 시킬 수는 없다. kind는 kubeadm으로 클러스터를 구성하는데, kubeadm은 Control Plane 컴포넌트의 매니페스트를 노드의 `/etc/kubernetes/manifests`에 써 두고 kubelet이 이를 읽어 실행하게 한다. 지금 kubectl로 보이는 건 kubelet이 API 서버에 만들어 준 거울상, 즉 **mirror Pod**다.
- **`kindnet-xxxxx`**: 3.4에서 말한 kind의 기본 CNI, kindnetd다.
- **`kube-proxy-xxxxx`**: 3.3의 kube-proxy다. kindnet과 kube-proxy는 노드마다 하나씩 떠야 하는 컴포넌트라서, 멀티 노드 클러스터라면 노드 수만큼 보인다. 이렇게 "모든 노드에 하나씩"을 보장하는 워크로드가 DaemonSet인데, [05편](/blog/2026/k8s-05-workloads/)에서 자세히 다룬다.
- **`coredns-`**: 컴포넌트 목록에는 없던 얼굴이다. 공식 문서가 **애드온(addon)**으로 분류하는 클러스터 DNS로, Service 이름을 IP로 풀어주는 역할을 한다. 애드온은 Control Plane의 필수 부품은 아니지만 사실상 모든 클러스터가 쓰는 확장 기능이다.

그런데 이상한 점이 하나 있다. **kubelet이 목록에 없다.** 당연한 일이다. kubelet은 Pod를 띄우는 주체이므로 자기 자신이 Pod일 수 없다. kubelet은 Pod가 아니라 노드 위에서 도는 일반 프로세스다. kind에서는 노드가 Docker 컨테이너이므로, 그 컨테이너 안에서 kubelet 프로세스가 돌고 있는 것이다.

노드 정보도 확인해 보자.

```bash
kubectl get nodes -o wide
```

출력의 `CONTAINER-RUNTIME` 컬럼에서 이 노드의 kubelet이 CRI 너머로 어떤 런타임과 대화하고 있는지 직접 확인할 수 있다. 마지막으로 Control Plane의 접점도 다시 보자.

```bash
kubectl cluster-info
```

```text
Kubernetes control plane is running at https://127.0.0.1:xxxxx
CoreDNS is running at https://127.0.0.1:xxxxx/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

02편에서 무심코 지나쳤던 이 출력이 이제 다르게 읽힐 것이다. 첫 줄의 주소가 바로 kube-apiserver의 엔드포인트고, kubectl이 보내는 모든 명령이 저 문을 통과한다.

---

# 6. 마무리

오늘 배운 것을 한 문장으로 압축하면 이렇다. **Kubernetes는 "바라는 상태"를 etcd에 적어두고, 모든 컴포넌트가 kube-apiserver라는 단일 관문을 통해 그 상태를 읽고 현실을 거기에 맞추는 시스템이다.**

- 상태는 **etcd**에만 있고, 접근은 **kube-apiserver**만 한다.
- **kube-controller-manager**는 바라는 상태와 현실의 차이를 끊임없이 메꾼다(reconcile).
- **kube-scheduler**는 필터링→스코어링으로 Pod의 자리를 정한다.
- 노드에서는 **kubelet**이 CRI로 런타임에 실행을 위임하고, **kube-proxy**와 **CNI**가 네트워크를 책임진다.

이 그림이 머리에 있으면 다음 편이 훨씬 쉬워진다. `kubectl apply -f pod.yaml` 한 줄을 실행했을 때, 방금 배운 컴포넌트들이 릴레이 하듯 바통을 넘기며 컨테이너 하나를 띄워내는 과정을 처음부터 끝까지 추적할 것이다.

> 다음 글 [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)에서는 Pod를 직접 띄우면서, 오늘 배운 컴포넌트들이 실제로 어떻게 협업하는지 5단계로 따라가 본다.

---

# 참고 문헌

- [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua, SK Devocean, 2024) — 시리즈 로드맵 출처
- [Kubernetes Components — kubernetes.io](https://kubernetes.io/docs/concepts/overview/components/)
- [Cluster Architecture — kubernetes.io](https://kubernetes.io/docs/concepts/architecture/)
- [Controllers — kubernetes.io](https://kubernetes.io/docs/concepts/architecture/controller/)
- [Kubernetes Scheduler — kubernetes.io](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/)
- [Controlling Access to the Kubernetes API — kubernetes.io](https://kubernetes.io/docs/concepts/security/controlling-access/)
- [Operating etcd clusters for Kubernetes — kubernetes.io](https://kubernetes.io/docs/tasks/administer-cluster/configure-upgrade-etcd/)
- [Container Runtime Interface (CRI) — kubernetes.io](https://kubernetes.io/docs/concepts/architecture/cri/)
- [Virtual IPs and Service Proxies — kubernetes.io](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [Create static Pods — kubernetes.io](https://kubernetes.io/docs/tasks/configure-pod-container/static-pod/)
- [kubeadm Implementation details — kubernetes.io](https://kubernetes.io/docs/reference/setup-tools/kubeadm/implementation-details/)
- [etcd FAQ — etcd.io](https://etcd.io/docs/v3.6/faq/)
- [kind Configuration — kind.sigs.k8s.io](https://kind.sigs.k8s.io/docs/user/configuration/)
