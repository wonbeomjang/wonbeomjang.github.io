---
layout: post
title: "내 노트북에 Kubernetes 클러스터 만들기 — kind와 kubectl"
date: 2026-07-07 02:00:00 +0900
description: "kind로 노트북에 멀티노드 Kubernetes 클러스터를 띄우고, kubectl과 kubeconfig(clusters/users/contexts)로 클러스터를 다루는 법"
categories: [infra]
tags: [kubernetes, infra, kind, minikube, kubectl]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 입문 시리즈**의 두 번째 글이다.
>
> - [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)
> - **02: 내 노트북에 클러스터 만들기 — kind와 kubectl** ← 현재 글
> - [03: Kubernetes 아키텍처 — Control Plane과 Node](/blog/2026/k8s-03-architecture/)
> - [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)
> - [05: 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet](/blog/2026/k8s-05-workloads/)
> - [06: 네트워킹 — Service와 Ingress](/blog/2026/k8s-06-networking/)
> - [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)
> - [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)
> - [09: 확장과 생태계 — Operator와 CNCF Projects](/blog/2026/k8s-09-operator-cncf/)
> - [10: 관측성 — 로그·메트릭과 Prometheus/Grafana](/blog/2026/k8s-10-observability/)
>
> 이 시리즈의 커리큘럼은 SK Devocean의 [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua) 글의 학습 로드맵을 바탕으로 구성했다.

[지난 글](/blog/2026/k8s-01-history/)에서 Kubernetes가 왜 태어났는지 봤다. 이제부터는 직접 손을 움직인다. 수영을 책으로만 배울 수 없듯이, Kubernetes도 명령을 쳐보지 않으면 절대 몸에 붙지 않는다.

문제는 연습장이다. 클라우드에 진짜 클러스터를 만들면 시간당 요금이 나가고, 회사 클러스터에서 연습하다가는 사고가 난다. 우리에게 필요한 건 **마음껏 만들고, 부수고, 다시 만들 수 있는 공짜 연습장**이다. 다행히 노트북에 Docker만 있으면 몇 분 만에 진짜 Kubernetes 클러스터를 띄울 수 있다. 이 글에서는 그 연습장을 만들고, 앞으로 시리즈 내내 쓸 조작 도구(kubectl)까지 손에 익힌다.

---

# 1. 어떤 도구로 만들까 — kind vs minikube

로컬 Kubernetes 도구로 가장 널리 쓰이는 두 가지가 **minikube**와 **kind**다. 둘 다 Kubernetes 공식 SIG(Special Interest Group)에서 관리하는 프로젝트라 어느 쪽을 골라도 신뢰할 수 있다.

- **minikube**: 공식 문서의 소개 문장 그대로 "local Kubernetes, focusing on making it easy to learn and develop for Kubernetes" — 학습과 로컬 개발의 편의에 집중한 도구다. Docker, QEMU, Hyper-V, KVM, VirtualBox 등 다양한 드라이버 위에서 VM 또는 컨테이너로 클러스터를 띄운다.
- **kind**: 이름부터 **K**ubernetes **IN** **D**ocker다. Docker 컨테이너 하나하나를 "노드"로 사용해 클러스터를 만든다. 원래 Kubernetes 프로젝트 자체를 테스트하기 위해 설계됐고, 로컬 개발과 CI에도 널리 쓰인다.

kind의 발상이 재미있다. 보통 "노드"라고 하면 물리 서버나 VM을 떠올리는데, kind는 **컨테이너를 노드로 쓴다**. 마트료시카 인형처럼, 컨테이너(노드) 안에서 다시 컨테이너(Pod)가 도는 구조다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-02-local-setup/kind-cluster-in-docker.png" class="img-fluid rounded z-depth-1" alt="kind 클러스터 구조 — 내 노트북의 Docker 안에서 컨테이너 하나가 노드 하나가 되고, kubectl은 API 서버로 접속한다" %}

- 노드 1개 = Docker 컨테이너 1개. VM을 부팅할 필요가 없어서 클러스터 생성·삭제가 가볍다.
- kubectl은 `127.0.0.1`의 무작위 포트로 노출된 API 서버에 접속한다 (뒤에서 직접 확인한다).
- 노드가 컨테이너일 뿐, 그 안에서 도는 Kubernetes는 진짜다. 03편에서 해부할 컴포넌트가 전부 들어 있다.

두 도구를 표로 비교하면 이렇다.

| 항목          | kind                           | minikube                                                          |
| ------------- | ------------------------------ | ----------------------------------------------------------------- |
| 노드 구현     | Docker 컨테이너 1개 = 노드 1개 | 드라이버(Docker, QEMU, Hyper-V, KVM, VirtualBox 등)별 VM/컨테이너 |
| 설계 목적     | Kubernetes 자체 테스트, CI     | 학습·로컬 개발 편의                                               |
| 멀티노드      | YAML config 파일로 선언        | `minikube start --nodes 2` 플래그                                 |
| 필요 환경     | Docker(또는 Podman, nerdctl)   | 컨테이너 또는 VM 매니저, 2 CPU·2GB 메모리·20GB 디스크 이상        |
| 클러스터 정의 | 선언적 YAML                    | 명령행 플래그와 프로파일                                          |

이 시리즈는 **kind**를 쓴다. 이유는 세 가지다.

1. **멀티노드 클러스터를 YAML 한 장으로 선언**할 수 있다. 03편(아키텍처), 05편(DaemonSet)에서 노드가 여러 개여야 보이는 것들이 있다.
2. Docker만 있으면 된다. VM 부팅 과정이 없어 만들고 부수는 반복이 빠르다.
3. 클러스터를 YAML로 정의하는 습관 자체가 Kubernetes의 선언적 철학(01편)과 맞닿아 있다.

물론 minikube도 훌륭한 도구고, minikube로도 이 시리즈를 그대로 따라올 수 있다. 도구보다 중요한 건 그 위에서 배우는 Kubernetes 자체다.

---

# 2. kind 설치와 첫 클러스터

준비물은 **Docker**뿐이다 (kind는 docker 외에 podman, nerdctl도 자동 감지한다). Docker Desktop 또는 Docker Engine이 돌고 있는지 먼저 확인하자.

## 2.1 kind 설치

macOS는 Homebrew로 설치한다.

```bash
brew install kind
```

Linux(x86_64)는 바이너리를 직접 내려받는다. 이 글 작성 시점의 최신 버전은 v0.32.0인데, 버전은 계속 올라가므로 [릴리스 페이지](https://github.com/kubernetes-sigs/kind/releases)에서 최신 버전을 확인하고 URL의 버전만 바꿔주면 된다.

```bash
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.32.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

설치 확인.

```bash
kind version
```

```text
kind v0.32.0 go1.26.3 linux/amd64
```

## 2.2 클러스터 생성 — 명령 한 줄

이제 이 글의 하이라이트다. 클러스터 생성은 정말로 한 줄이다.

```bash
kind create cluster
```

```text
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.36.1) 🖼
 ✓ Preparing nodes 📦
 ✓ Writing configuration 📜
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind

Have a nice day! 👋
```

(노드 이미지 버전은 kind 버전에 따라 다르다. kind v0.32.0의 기본 이미지는 Kubernetes v1.36.1이다.)

출력의 각 줄이 곧 클러스터가 만들어지는 과정이다.

| 출력 줄                 | 실제로 일어나는 일                                                      |
| ----------------------- | ----------------------------------------------------------------------- |
| Ensuring node image     | "노드 역할을 할 컨테이너 이미지"(kindest/node)를 내려받는다             |
| Preparing nodes         | 노드가 될 Docker 컨테이너를 만든다                                      |
| Writing configuration   | 클러스터 초기화 도구(kubeadm)의 설정을 쓴다                             |
| Starting control-plane  | 클러스터의 두뇌인 Control Plane을 부팅한다 — 03편에서 해부한다          |
| Installing CNI          | Pod끼리 통신할 네트워크 플러그인을 깐다 — 03편에서 다룬다               |
| Installing StorageClass | 기본 스토리지 설정을 깐다 — 07편에서 다룬다                             |
| Set kubectl context     | 방금 만든 클러스터로 kubectl이 접속하도록 설정한다 — 이 글의 5~6장 주제 |

정말 컨테이너가 노드 행세를 하고 있는지 Docker로 확인해 보자.

```bash
docker ps
```

```text
CONTAINER ID   IMAGE                  COMMAND                  ...   PORTS                       NAMES
1f2a3b4c5d6e   kindest/node:v1.36.1   "/usr/local/bin/entr…"   ...   127.0.0.1:39663->6443/tcp   kind-control-plane
```

- `kind-control-plane`이라는 컨테이너 하나가 노드다. 클러스터 이름의 기본값이 `kind`이고, 노드 이름은 `<클러스터 이름>-<역할>` 규칙으로 붙는다.
- `127.0.0.1:39663->6443/tcp`: 컨테이너 안의 API 서버 포트(6443)가 내 노트북의 무작위 포트로 연결돼 있다 (포트 번호는 만들 때마다 다르다). kubectl은 이 문으로 들어간다.

만들어진 클러스터 목록도 확인할 수 있다.

```bash
kind get clusters
```

```text
kind
```

클러스터는 떴다. 그런데 지금은 서버만 있고 전화기가 없는 상태다. 클러스터에 말을 걸 도구, **kubectl**을 설치할 차례다.

---

# 3. kubectl 설치와 첫 접속

kubectl은 Kubernetes API 서버와 대화하는 공식 CLI다. 앞으로 시리즈가 끝날 때까지 가장 많이 치게 될 명령이다.

macOS는 Homebrew로 설치한다.

```bash
brew install kubectl
```

Linux(x86_64)는 최신 안정 버전을 내려받아 설치한다.

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

다른 OS는 [공식 설치 문서](https://kubernetes.io/docs/tasks/tools/)를 참고하면 된다. 한 가지 규칙만 기억하자. 공식 문서에 따르면 **kubectl 버전은 클러스터와 마이너 버전 1 이내**여야 한다. 예를 들어 v1.36 클라이언트는 v1.35, v1.36, v1.37 Control Plane과 통신할 수 있다.

```bash
kubectl version --client
```

```text
Client Version: v1.36.2
Kustomize Version: v5.8.1
```

이제 첫 접속이다. 아까 kind가 알려준 명령을 그대로 쳐보자.

```bash
kubectl cluster-info
```

```text
Kubernetes control plane is running at https://127.0.0.1:39663
CoreDNS is running at https://127.0.0.1:39663/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

(포트 번호는 환경마다 다르다.) 주소를 보면 아까 `docker ps`에서 본 `127.0.0.1:39663`과 같다. **kubectl → 노트북의 무작위 포트 → 컨테이너 안의 API 서버**로 연결이 이어진 것이다. 노드도 확인해 보자.

```bash
kubectl get nodes
```

```text
NAME                 STATUS   ROLES           AGE   VERSION
kind-control-plane   Ready    control-plane   2m    v1.36.1
```

노드 하나짜리 클러스터다. 연습용으로는 충분하지만, 앞으로를 위해 좀 더 실전에 가까운 모양으로 바꿔보자.

---

# 4. 멀티노드 클러스터로 업그레이드

실제 프로덕션 클러스터는 Control Plane과 Worker가 분리된 여러 노드로 구성된다. kind에서는 클러스터 구성을 YAML 파일로 선언하면 된다. 먼저 기존 클러스터를 지운다.

```bash
kind delete cluster
```

그리고 `kind-config.yaml` 파일을 만든다.

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

- `kind: Cluster` — 여기서 `kind`는 도구 이름이 아니라 "이 YAML이 어떤 종류의 리소스인지"를 나타내는 Kubernetes 스타일 필드다. 앞으로 모든 YAML에서 만나게 된다.
- `nodes` 아래에 원하는 노드를 역할(`control-plane` / `worker`)과 함께 나열하면 끝이다.

이 파일로 클러스터를 만든다.

```bash
kind create cluster --config kind-config.yaml
```

```text
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.36.1) 🖼
 ✓ Preparing nodes 📦 📦 📦
 ✓ Writing configuration 📜
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
 ✓ Joining worker nodes 🚜
Set kubectl context to "kind-kind"
...
```

아까와 달라진 점이 두 가지다. `Preparing nodes`의 📦가 노드 수만큼 3개가 됐고, `Joining worker nodes` 단계가 새로 생겼다. Worker 노드들이 Control Plane에 합류하는 과정이다.

```bash
kubectl get nodes
```

```text
NAME                 STATUS   ROLES           AGE   VERSION
kind-control-plane   Ready    control-plane   3m    v1.36.1
kind-worker          Ready    <none>          2m    v1.36.1
kind-worker2         Ready    <none>          2m    v1.36.1
```

- 노드 이름은 `<클러스터 이름>-<역할>` 규칙이고, 같은 역할이 여러 개면 두 번째부터 숫자가 붙는다 (`kind-worker`, `kind-worker2`).
- Worker의 ROLES가 `<none>`으로 보이는 건 정상이다. ROLES 컬럼은 노드에 붙은 role 라벨을 보여주는 것인데, Worker에는 이 라벨이 자동으로 붙지 않는다.

참고로 `--name` 플래그를 쓰면 클러스터를 여러 개 만들 수도 있다 (`kind create cluster --name kind-2`). 이름이 다르면 공존이 가능하고, `kind get clusters`로 모두 확인할 수 있다.

노트북 한 대에 3노드 Kubernetes 클러스터가 생겼다. 그런데 궁금하지 않은가? 클러스터를 지우고 다시 만들어도 kubectl은 용케 새 클러스터를 찾아간다. **kubectl은 어떻게 접속할 곳을 아는 걸까?**

---

# 5. kubectl은 어떻게 클러스터를 찾나 — kubeconfig 해부

답은 `~/.kube/config` 파일에 있다. kubectl은 기본적으로 이 파일(**kubeconfig**라고 부른다)을 읽어 어느 클러스터에 어떤 신원으로 접속할지 결정한다. kind가 클러스터를 만들 때 "Set kubectl context to ..."라고 출력한 것이 바로 이 파일을 갱신했다는 뜻이다.

kubeconfig는 세 가지 목록으로 구성된다.

| 요소         | 답하는 질문         | 담는 내용                            | 비유        |
| ------------ | ------------------- | ------------------------------------ | ----------- |
| **clusters** | **어디로** 접속하나 | API 서버 주소, CA 인증서             | 주소록      |
| **users**    | **누구로** 접속하나 | 클라이언트 인증서, 토큰 등 인증 정보 | 신분증 지갑 |
| **contexts** | 어떤 **조합**으로   | cluster + user + namespace 묶음      | 단축 다이얼 |

핵심은 **context**다. 공식 문서의 정의 그대로, context는 cluster·namespace·user 세 파라미터를 편한 이름 하나로 묶은 것이다. "어느 클러스터에(cluster), 어떤 신원으로(user), 기본 작업 공간은 어디로(namespace)"를 매번 지정하는 대신, 조합에 이름을 붙여 단축 다이얼처럼 쓴다.

구조를 단순화한 예시를 보면 이렇다 (인증서 데이터는 생략).

```yaml
apiVersion: v1
kind: Config
clusters:
  - name: kind-kind
    cluster:
      server: https://127.0.0.1:39663
      certificate-authority-data: LS0t... # 생략
  - name: company-prod
    cluster:
      server: https://prod.company.example.com:6443
      certificate-authority-data: LS0t... # 생략
users:
  - name: kind-kind
    user: {} # 인증서 등 인증 정보
  - name: prod-admin
    user: {} # 토큰 등 인증 정보
contexts:
  - name: kind-kind
    context:
      cluster: kind-kind
      user: kind-kind
  - name: company-prod
    context:
      cluster: company-prod
      user: prod-admin
      namespace: ml-serving
current-context: kind-kind
```

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-02-local-setup/kubeconfig-structure.png" class="img-fluid rounded z-depth-1" alt="kubeconfig 구조 — current-context가 가리키는 context가 cluster(어디로)와 user(누구로)를 묶는다" %}

- `current-context`가 지금 kubectl이 사용하는 단축 다이얼이다. 별도 지정이 없으면 모든 kubectl 명령이 이 컨텍스트로 나간다.
- kind는 컨텍스트 이름을 `kind-<클러스터 이름>` 형식으로 등록한다. 기본 클러스터 이름이 `kind`라서 컨텍스트가 `kind-kind`가 된다.
- kubeconfig 파일을 여러 개 쓸 수도 있다. 환경 변수 `KUBECONFIG`에 콜론(`:`)으로 구분해 나열하면 kubectl이 병합해서 본다 (Linux/macOS 기준).

---

# 6. 컨텍스트 전환 실습 — 로컬과 회사 클러스터 오가기

컨텍스트가 진짜 중요해지는 순간은 클러스터가 두 개 이상일 때다. 위 예시처럼 로컬 연습용 `kind-kind`와 회사 프로덕션 `company-prod`가 함께 등록된 상황을 가정하자. 컨텍스트 조작 명령은 세 개만 알면 된다.

먼저 등록된 컨텍스트 목록.

```bash
kubectl config get-contexts
```

```text
CURRENT   NAME           CLUSTER        AUTHINFO     NAMESPACE
          company-prod   company-prod   prod-admin   ml-serving
*         kind-kind      kind-kind      kind-kind
```

- `*`가 현재 컨텍스트다. NAMESPACE가 비어 있으면 `default` 네임스페이스를 쓴다는 뜻이다.

현재 컨텍스트만 빠르게 확인하려면.

```bash
kubectl config current-context
```

```text
kind-kind
```

회사 클러스터로 전환하려면.

```bash
kubectl config use-context company-prod
```

```text
Switched to context "company-prod".
```

이 순간부터 모든 kubectl 명령은 프로덕션으로 날아간다. 그리고 여기에 **입문자가 반드시 알아야 할 사고 패턴**이 있다. 어제 회사 클러스터를 보다가 컨텍스트를 안 돌려놓고, 오늘 "로컬 연습장이겠지" 하며 `kubectl delete`를 치는 사고다. 터미널 화면만 봐서는 어느 클러스터를 가리키는지 티가 나지 않는다. 그래서 습관 두 가지를 권한다.

1. **파괴적인 명령 전에는 `kubectl config current-context`부터** 친다. 3초짜리 보험이다.
2. 일회성으로 다른 클러스터를 조회할 때는 전환 대신 `--context` 플래그를 쓴다. kubectl은 `--context` 플래그를 current-context보다 우선하므로, 현재 컨텍스트를 건드리지 않고 한 번만 다른 곳에 명령을 보낼 수 있다.

```bash
kubectl get nodes --context kind-kind
```

실습을 마쳤으면 로컬로 돌아오자.

```bash
kubectl config use-context kind-kind
```

```text
Switched to context "kind-kind".
```

---

# 7. 마무리

오늘 만든 것을 한 문장으로 요약하면 이렇다. **Docker 컨테이너를 노드 삼아(kind) 노트북 안에 3노드짜리 진짜 Kubernetes 클러스터를 세웠고, kubectl이 kubeconfig의 컨텍스트를 통해 그 클러스터를 찾아가는 경로를 이해했다.**

| 명령                                | 역할                                     |
| ----------------------------------- | ---------------------------------------- |
| `kind create cluster`               | 클러스터 생성 (기본 이름 `kind`)         |
| `kind create cluster --config ...`  | YAML로 선언한 구성(멀티노드 등)으로 생성 |
| `kind get clusters`                 | 만들어진 클러스터 목록                   |
| `kind delete cluster`               | 클러스터 삭제                            |
| `kubectl version --client`          | 클라이언트 버전 확인                     |
| `kubectl cluster-info`              | 현재 클러스터의 API 서버 주소 확인       |
| `kubectl get nodes`                 | 노드 목록과 상태 확인                    |
| `kubectl config get-contexts`       | 등록된 컨텍스트 목록 (`*`가 현재)        |
| `kubectl config current-context`    | 현재 컨텍스트 확인 — 파괴적 명령 전 필수 |
| `kubectl config use-context <이름>` | 컨텍스트 전환                            |

방금 만든 멀티노드 클러스터는 **지우지 말고 그대로 두자**. `kind create cluster` 한 줄 뒤에서 조용히 부팅된 Control Plane 안에는 여러 컴포넌트가 각자의 역할을 맡아 돌아가고 있다. 다음 글에서 그 내부를 하나씩 해부한다.

> 다음 글 [03: Kubernetes 아키텍처 — Control Plane과 Node](/blog/2026/k8s-03-architecture/)에서는 오늘 만든 클러스터의 내부 — etcd, kube-apiserver, scheduler, kubelet — 를 해부한다.

---

# 참고 문헌

- [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua, SK Devocean, 2024) — 시리즈 로드맵 출처
- [kind — Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kind — Configuration](https://kind.sigs.k8s.io/docs/user/configuration/)
- [kind Releases — GitHub](https://github.com/kubernetes-sigs/kind/releases)
- [minikube — Get Started](https://minikube.sigs.k8s.io/docs/start/)
- [minikube — Using Multi-Node Clusters](https://minikube.sigs.k8s.io/docs/tutorials/multi_node/)
- [Install and Set Up kubectl on Linux — kubernetes.io](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
- [Install and Set Up kubectl on macOS — kubernetes.io](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/)
- [Organizing Cluster Access Using kubeconfig Files — kubernetes.io](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/)
