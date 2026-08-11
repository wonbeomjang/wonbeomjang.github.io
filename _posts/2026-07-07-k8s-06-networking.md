---
layout: post
title: "Kubernetes 네트워킹 — Service와 Ingress"
date: 2026-07-07 06:00:00 +0900
description: "Pod IP는 재시작마다 바뀐다 — ClusterIP·NodePort·LoadBalancer 세 가지 Service 타입과 Ingress로 Kubernetes 트래픽 라우팅을 이해한다"
categories: [infra]
tags: [kubernetes, infra, service, ingress, networking, hpa, networkpolicy, gateway-api]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 입문 시리즈**의 여섯 번째 글이다.
>
> - [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)
> - [02: 내 노트북에 클러스터 만들기 — kind와 kubectl](/blog/2026/k8s-02-local-setup/)
> - [03: Kubernetes 아키텍처 — Control Plane과 Node](/blog/2026/k8s-03-architecture/)
> - [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)
> - [05: 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet](/blog/2026/k8s-05-workloads/)
> - **06: 네트워킹 — Service와 Ingress** ← 현재 글
> - [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)
> - [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)
> - [09: 확장과 생태계 — Operator와 CNCF Projects](/blog/2026/k8s-09-operator-cncf/)
> - [10: 관측성 — 로그·메트릭과 Prometheus/Grafana](/blog/2026/k8s-10-observability/)
>
> 이 시리즈의 커리큘럼은 SK Devocean의 [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua) 글의 학습 로드맵을 바탕으로 구성했다.

# 1. 들어가며 — Pod IP는 믿을 수 없다

[05편](/blog/2026/k8s-05-workloads/)에서 Deployment로 nginx Pod를 여러 개 띄웠다. 그런데 곧바로 문제가 생긴다. **이 Pod들에 어떻게 접속해야 할까?**

Pod는 각자 클러스터 내부 IP를 하나씩 받는다. 문제는 이 IP가 **일회용**이라는 것이다. Pod가 죽고 ReplicaSet이 새 Pod를 만들면, 새 Pod는 새 IP를 받는다. 직접 확인해 보자.

```bash
$ kubectl create deployment web --image=nginx --replicas=2
deployment.apps/web created

$ kubectl get pods -o wide
NAME                   READY   STATUS    RESTARTS   AGE   IP           NODE
web-5f9b6b7d4b-8xk2p   1/1     Running   0          20s   10.244.0.5   kind-control-plane
web-5f9b6b7d4b-tq7wn   1/1     Running   0          20s   10.244.0.6   kind-control-plane

$ kubectl delete pod web-5f9b6b7d4b-8xk2p
pod "web-5f9b6b7d4b-8xk2p" deleted

$ kubectl get pods -o wide
NAME                   READY   STATUS    RESTARTS   AGE   IP           NODE
web-5f9b6b7d4b-tq7wn   1/1     Running   0          2m    10.244.0.6   kind-control-plane
web-5f9b6b7d4b-zc4rd   1/1     Running   0          10s   10.244.0.7   kind-control-plane
```

`10.244.0.5`였던 Pod가 사라지고 `10.244.0.7`을 가진 새 Pod가 생겼다. 만약 다른 애플리케이션이 `10.244.0.5`를 하드코딩해서 호출하고 있었다면 그대로 장애다. 게다가 롤링 업데이트 한 번이면 **모든 Pod의 IP가 전부 갈아치워진다**.

일상 비유로 보면 이렇다. 어떤 회사의 담당 직원은 계속 바뀌고, 바뀔 때마다 휴대폰 번호도 바뀐다. 고객이 직원 개인 번호를 외워서 전화하는 방식은 지속 불가능하다. 해법은 회사가 **대표 전화번호** 하나를 만들고, 교환대가 지금 근무 중인 직원에게 전화를 돌려주는 것이다.

Kubernetes에서 이 대표번호가 **Service**이고, "지금 근무 중인 직원 명단"을 관리하는 것이 **EndpointSlice**, 실제로 전화를 돌려주는 교환대가 **kube-proxy**다. 그리고 회사 건물 1층에서 "어떤 용건이신가요?"를 묻고 부서별로 안내하는 안내 데스크가 **Ingress**다. 이번 편에서 이 넷을 전부 다룬다.

---

# 2. Service의 동작 원리 — Selector, EndpointSlice, kube-proxy

Service는 "고정된 가상 IP(ClusterIP)와 DNS 이름"을 제공하고, 그 뒤에 있는 Pod 집합으로 트래픽을 분산한다. 핵심 질문은 두 가지다.

**첫째, Service는 어떤 Pod에게 트래픽을 보낼지 어떻게 아는가?** 답은 05편에서 배운 **Label Selector**다. Service는 `selector`에 적힌 라벨과 일치하는 Pod를 계속 추적한다. Pod가 죽든 새로 생기든, 라벨만 맞으면 자동으로 목록에 반영된다.

**둘째, 추적한 결과는 어디에 저장되는가?** Kubernetes는 Selector와 일치하는 Pod들의 IP·포트 목록을 **EndpointSlice**라는 별도 리소스로 만들어 관리한다(과거에는 Endpoints 리소스 하나에 다 담았는데, Pod가 수천 개면 객체가 너무 커져서 여러 조각(slice)으로 나눈 EndpointSlice가 도입됐다). Service는 "대표번호"라는 선언이고, EndpointSlice는 "지금 이 번호로 연결 가능한 실제 직원 명단"이다.

마지막 퍼즐은 [03편](/blog/2026/k8s-03-architecture/)에서 본 **kube-proxy**다. 각 노드의 kube-proxy는 Service와 EndpointSlice의 변화를 감시하다가, 노드의 iptables/IPVS 규칙을 갱신한다. 그래서 어떤 Pod가 ClusterIP로 패킷을 보내면, 그 패킷은 kube-proxy가 미리 깔아둔 규칙에 따라 실제 Pod IP 중 하나로 바꿔치기되어 전달된다. ClusterIP는 어떤 네트워크 인터페이스에도 붙어 있지 않은 **가상 IP**라는 점이 중요하다 — ping은 안 되지만 규칙에 의해 라우팅은 된다.

Service를 만들고 이 연결 고리를 직접 확인해 보자.

```bash
$ kubectl expose deployment web --port=80
service/web exposed

$ kubectl get svc web
NAME   TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
web    ClusterIP   10.96.156.10   <none>        80/TCP    8s

$ kubectl get endpointslices
NAME         ADDRESSTYPE   PORTS   ENDPOINTS               AGE
web-abc12    IPv4          80      10.244.0.6,10.244.0.7   8s
```

EndpointSlice의 ENDPOINTS 목록이 정확히 현재 Pod들의 IP와 일치한다. 여기서 Pod 하나를 지우면, 잠시 후 EndpointSlice의 목록도 새 Pod IP로 자동 갱신된다. **개발자는 명단을 관리할 필요가 없다** — 라벨만 잘 붙이면 나머지는 컨트롤 플레인이 reconcile한다.

---

# 3. Service 3종 세트 — ClusterIP, NodePort, LoadBalancer

Service에는 "누가 접근할 수 있느냐"에 따라 세 가지 대표 타입이 있다. 먼저 비교표로 감을 잡자.

| 타입             | 접근 범위                   | 접근 방법                            | 주 용도                                |
| ---------------- | --------------------------- | ------------------------------------ | -------------------------------------- |
| **ClusterIP**    | 클러스터 내부 전용 (기본값) | ClusterIP 또는 DNS 이름              | 마이크로서비스 간 내부 통신            |
| **NodePort**     | 클러스터 외부 가능          | `노드IP:노드포트` (기본 30000-32767) | 개발/테스트, 외부 LB를 직접 붙일 때    |
| **LoadBalancer** | 클러스터 외부 가능          | 클라우드가 발급한 외부 IP            | 클라우드 환경에서 프로덕션 서비스 노출 |

세 타입은 배타적이 아니라 **포개진 구조**다. NodePort를 만들면 ClusterIP도 함께 생기고, LoadBalancer를 만들면 NodePort와 ClusterIP가 함께 생긴다. 이 외에 클러스터 외부 도메인으로 CNAME을 돌려주는 ExternalName 타입도 있지만, 입문 단계에서는 위 세 가지만 확실히 잡으면 된다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-06-networking/service-types.png" class="img-fluid rounded z-depth-1" alt="클러스터 외부의 사용자가 LoadBalancer, NodePort, ClusterIP를 거쳐 Pod에 도달하고, 내부 Pod는 DNS로 Service에 직행하는 구조" %}

- 외부 트래픽은 LoadBalancer → NodePort → ClusterIP 순으로 **안쪽 계층에 올라탄다**.
- 내부 트래픽은 처음부터 ClusterIP(DNS 이름)로 직행한다.
- 어느 경로로 들어오든 마지막 분산은 kube-proxy가 깔아둔 규칙이 담당한다.

## 3.1 ClusterIP — 클러스터 내부의 대표번호와 DNS

위에서 만든 `web` Service가 바로 기본값인 ClusterIP 타입이다. 그런데 `10.96.156.10` 같은 ClusterIP도 결국 클러스터를 다시 만들면 바뀔 수 있는 값이다. 그래서 실무에서는 IP가 아니라 **DNS 이름**으로 접근한다.

Kubernetes의 클러스터 DNS(kind에서는 CoreDNS가 `kube-system`에 떠 있다)는 모든 Service에 다음 형식의 도메인을 만들어 준다.

```text
<서비스 이름>.<네임스페이스>.svc.<클러스터 도메인>
```

클러스터 도메인의 기본값은 `cluster.local`이다. 즉 `default` 네임스페이스의 `web` Service는 `web.default.svc.cluster.local`로 조회되고, 이 이름은 Service의 ClusterIP로 해석된다. 임시 Pod를 하나 띄워 직접 불러보자.

```bash
$ kubectl run tmp --rm -it --image=curlimages/curl --restart=Never -- sh
~ $ curl -s http://web.default.svc.cluster.local | head -4
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
~ $ curl -s http://web | head -4
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
```

두 번째 명령처럼 **같은 네임스페이스 안에서는 서비스 이름만으로도 접근된다**. kubelet이 Pod의 `/etc/resolv.conf`에 `<네임스페이스>.svc.cluster.local` 등의 search 도메인을 넣어 두기 때문에, 짧은 이름이 자동으로 전체 이름으로 확장된다. 다른 네임스페이스의 서비스를 부를 때는 `web.default`처럼 네임스페이스를 붙이면 된다.

이것이 Kubernetes의 **서비스 디스커버리**다. 백엔드가 데이터베이스를 찾을 때 IP를 설정 파일에 박는 것이 아니라 `db.default.svc.cluster.local`이라는 안정적인 이름 하나만 알면 된다. 참고로 `clusterIP: None`으로 만드는 Headless Service라는 변종도 있는데, 대표번호 없이 Pod 개개인의 번호를 그대로 알려주는 방식으로 05편의 StatefulSet과 짝을 이룬다.

## 3.2 NodePort — 노드의 문을 연다

ClusterIP는 클러스터 밖에서는 보이지 않는다. 외부에서 접근하는 가장 원초적인 방법이 **NodePort**다. NodePort Service를 만들면 컨트롤 플레인이 `--service-node-port-range` 플래그로 지정된 범위(**기본값 30000-32767**)에서 포트를 하나 할당하고, **모든 노드**가 그 포트로 들어온 트래픽을 Service로 넘긴다. 어느 노드의 IP로 접근하든 같은 포트에서 같은 Service에 닿는다.

```bash
$ kubectl expose deployment web --port=80 --type=NodePort --name=web-nodeport
service/web-nodeport exposed

$ kubectl get svc web-nodeport
NAME           TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
web-nodeport   NodePort   10.96.201.44   <none>        80:31234/TCP   6s
```

`80:31234/TCP`에서 `31234`가 할당된 NodePort다. 과연 30000-32767 범위 안이다.

그런데 kind에서는 함정이 하나 있다. [02편](/blog/2026/k8s-02-local-setup/)에서 봤듯 kind의 "노드"는 내 노트북 위의 **Docker 컨테이너**라서, 노드 IP가 호스트(노트북)에서 바로 접근되는 주소가 아니다. 그래서 노드 컨테이너 안에서 확인하거나,

```bash
$ docker exec -it kind-control-plane curl -s localhost:31234 | head -4
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
```

호스트에서 직접 접근하고 싶다면 kind 클러스터 생성 시 `extraPortMappings`로 호스트 포트를 노드에 미리 매핑해야 한다. kind 공식 문서는 이때 **노드의 `containerPort`와 Service의 `nodePort`를 같은 값으로 맞추라**고 안내한다. 이 기능은 잠시 후 Ingress 실습에서 실제로 사용한다.

## 3.3 LoadBalancer — 클라우드의 힘, 그리고 kind에서 pending인 이유

프로덕션에서 외부 노출의 표준은 **LoadBalancer** 타입이다. AWS·GCP 같은 클라우드에서 이 타입의 Service를 만들면, 클라우드 컨트롤러가 실제 로드밸런서(예: AWS의 ELB)를 프로비저닝하고 외부 IP를 발급해 Service에 연결해 준다.

kind에서 만들면 어떻게 될까?

```bash
$ kubectl expose deployment web --port=80 --type=LoadBalancer --name=web-lb
service/web-lb exposed

$ kubectl get svc web-lb
NAME     TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
web-lb   LoadBalancer   10.96.88.123   <pending>     80:32017/TCP   30s
```

EXTERNAL-IP가 `<pending>`에서 영원히 멈춘다. 고장이 아니다. LoadBalancer 타입은 "외부 로드밸런서를 만들어 달라"는 **요청서**일 뿐이고, 그 요청을 실제로 처리할 클라우드 프로바이더 연동이 로컬 kind에는 없기 때문에 IP를 발급해 줄 주체가 없는 것이다. 공식 문서도 클라우드 연동이 없는 환경에서는 외부 IP 할당이 일어나지 않아 pending 상태로 남는다고 설명한다. (kind 진영에는 이 역할을 로컬에서 흉내 내주는 `cloud-provider-kind`라는 별도 프로젝트가 있다.)

실습이 끝났으면 NodePort·LoadBalancer Service는 정리해 두자.

```bash
$ kubectl delete svc web-nodeport web-lb
service "web-nodeport" deleted
service "web-lb" deleted
```

---

# 4. Ingress — L7에서 길을 안내하는 데스크

Service만으로 외부 노출을 하려면 곧 한계에 부딪힌다. 서비스가 10개면 NodePort 10개 혹은 로드밸런서 10대(=비용 10배)가 필요하다. 게다가 Service는 L4(TCP/UDP) 수준에서 동작하므로 "URL 경로가 `/api`면 백엔드로, `/`면 프론트엔드로" 같은 **HTTP 내용 기반 분기**를 할 수 없다.

이 문제를 푸는 것이 **Ingress**다. Ingress는 L7(HTTP/HTTPS)에서 **호스트 이름(host)과 경로(path)를 보고 트래픽을 여러 Service로 라우팅하는 규칙**이다. 건물 입구는 하나지만, 1층 안내 데스크가 "채용 문의는 3층 인사팀, AS 접수는 2층 서비스팀"처럼 용건에 따라 안내해 주는 그림이다.

## 4.1 Ingress 리소스 vs Ingress Controller — 규칙과 실행자는 다르다

입문자가 가장 많이 헷갈리는 지점이 여기다. Ingress는 **두 조각**으로 이루어진다.

| 구분        | Ingress 리소스                  | Ingress Controller                                     |
| ----------- | ------------------------------- | ------------------------------------------------------ |
| 정체        | YAML로 선언하는 라우팅 **규칙** | 규칙을 읽어 실제로 트래픽을 처리하는 **프로그램**(Pod) |
| 비유        | 안내 데스크의 업무 매뉴얼       | 매뉴얼대로 일하는 안내원                               |
| 누가 만드나 | 사용자(개발자)가 작성           | 별도로 설치해야 함 (ingress-nginx, Contour 등)         |
| 없으면?     | 규칙만 있고 아무 일도 안 일어남 | —                                                      |

Kubernetes 공식 문서가 명시하듯, **Ingress Controller가 없으면 Ingress 리소스를 아무리 만들어도 효과가 없다**. Deployment를 만들면 내장 컨트롤러가 알아서 동작하는 것과 달리, Ingress의 실행자는 클러스터에 기본 내장되어 있지 않다. 매뉴얼만 써 놓고 안내원을 채용하지 않은 상태인 셈이다. 가장 널리 쓰이는 안내원이 Kubernetes 커뮤니티가 관리하는 **ingress-nginx**다.

## 4.2 실습 — kind에 ingress-nginx 설치하기

이제 ingress-nginx를 설치해 보자. 호스트의 80 포트로 들어온 요청이 클러스터 안까지 닿아야 하므로, `extraPortMappings`를 넣은 **전용 클러스터를 새로 만든다** (kind는 클러스터 생성이 1분이면 되니 부담이 없다).

참고로 kind 공식 ingress 가이드는 최근 `cloud-provider-kind`(v0.9.0+)의 네이티브 Ingress 지원을 기본으로 안내하고, ingress-nginx 같은 서드파티 컨트롤러는 각자의 공식 문서를 따르라고 되어 있다. 여기서는 실무에서 가장 흔히 만나는 ingress-nginx를 직접 설치하는 경로로 간다. ingress-nginx 프로젝트는 kind 환경용 매니페스트를 별도로 제공한다.

```bash
$ cat <<EOF | kind create cluster --name ingress --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      - containerPort: 443
        hostPort: 443
        protocol: TCP
EOF
Creating cluster "ingress" ...
 ✓ Ensuring node image (kindest/node:...) 🖼
 ✓ Preparing nodes 📦
 ...
Set kubectl context to "kind-ingress"
```

`extraPortMappings`는 호스트(노트북)의 80/443 포트를 노드 컨테이너의 80/443으로 포워딩한다. 이제 ingress-nginx의 kind 전용 매니페스트를 적용한다.

```bash
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
namespace/ingress-nginx created
serviceaccount/ingress-nginx created
...
deployment.apps/ingress-nginx-controller created
ingressclass.networking.k8s.io/nginx created
...
```

이 매니페스트를 열어 보면(작성 시점 기준 controller v1.15.1) 설계 의도가 보인다. 컨트롤러 Pod가 `hostPort: 80/443`으로 **노드의 80/443 포트에 직접 바인딩**되고, control-plane 노드에도 뜰 수 있게 toleration이 걸려 있다. 그래서 `호스트:80 → (extraPortMappings) → 노드:80 → (hostPort) → ingress-nginx 컨트롤러`라는 파이프가 완성된다. 컨트롤러가 준비될 때까지 기다리자.

```bash
$ kubectl wait --namespace ingress-nginx \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/component=controller \
    --timeout=120s
pod/ingress-nginx-controller-6b94c75599-xv2fk condition met
```

## 4.3 실습 — path 기반 라우팅

안내원이 준비됐으니 부서 두 개를 만들자. 요청받은 문자열을 그대로 응답하는 `http-echo` 이미지로 `/foo`와 `/bar`에 각각 다른 앱을 배치한다.

```yaml
# echo-apps.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foo-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: foo
  template:
    metadata:
      labels:
        app: foo
    spec:
      containers:
        - name: http-echo
          image: hashicorp/http-echo
          args: ["-text=foo"]
---
apiVersion: v1
kind: Service
metadata:
  name: foo-service
spec:
  selector:
    app: foo
  ports:
    - port: 5678 # http-echo의 기본 포트
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bar-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bar
  template:
    metadata:
      labels:
        app: bar
    spec:
      containers:
        - name: http-echo
          image: hashicorp/http-echo
          args: ["-text=bar"]
---
apiVersion: v1
kind: Service
metadata:
  name: bar-service
spec:
  selector:
    app: bar
  ports:
    - port: 5678
```

두 Service는 모두 기본값인 ClusterIP다. **외부 노출은 Ingress가 담당하므로 Service를 NodePort로 만들 필요가 없다**는 점을 눈여겨보자. 이제 라우팅 규칙, 즉 Ingress 리소스를 작성한다.

```yaml
# echo-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
spec:
  ingressClassName: nginx # 어떤 컨트롤러가 이 규칙을 처리할지 지정
  rules:
    - http:
        paths:
          - path: /foo
            pathType: Prefix
            backend:
              service:
                name: foo-service
                port:
                  number: 5678
          - path: /bar
            pathType: Prefix
            backend:
              service:
                name: bar-service
                port:
                  number: 5678
```

- `ingressClassName: nginx` — 클러스터에 컨트롤러가 여러 개일 수 있으므로, 이 규칙의 담당자를 지정한다. 방금 설치한 매니페스트가 `nginx`라는 IngressClass를 만들어 두었다.
- `path` + `pathType: Prefix` — URL이 `/foo`로 시작하면 `foo-service`로 보낸다.
- `rules`에 `host: foo.example.com`처럼 호스트를 추가하면 도메인 기반 분기도 가능하다.

적용하고 호스트에서 바로 curl을 날려 보자.

```bash
$ kubectl apply -f echo-apps.yaml -f echo-ingress.yaml
deployment.apps/foo-app created
service/foo-service created
deployment.apps/bar-app created
service/bar-service created
ingress.networking.k8s.io/example-ingress created

$ curl localhost/foo
foo
$ curl localhost/bar
bar
```

같은 `localhost:80` 입구로 들어갔지만 경로에 따라 서로 다른 Service의 Pod가 응답했다. 전체 흐름을 그림으로 정리하면 이렇다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-06-networking/ingress-path-routing.png" class="img-fluid rounded z-depth-1" alt="curl 요청이 extraPortMappings와 hostPort를 거쳐 ingress-nginx 컨트롤러에 닿고, path에 따라 foo-service와 bar-service로 라우팅되는 흐름" %}

- Ingress 리소스는 점선이다 — 트래픽이 흐르는 곳이 아니라 컨트롤러가 **읽는 설정**이다.
- 외부 노출 지점은 컨트롤러 하나뿐이고, 뒤의 Service들은 전부 내부 전용 ClusterIP다.
- 실습이 끝나면 `kind delete cluster --name ingress`로 정리한다.

---

# 5. HPA — 트래픽에 맞춰 Pod 수를 자동 조절한다

입구 정리는 끝났다. Service가 대표번호를 만들고 Ingress가 경로를 안내한다. 그런데 그 입구로 **트래픽이 밀려들면** 어떻게 될까? [05편](/blog/2026/k8s-05-workloads/)까지 replicas는 항상 사람이 정해 준 고정값이었다. 새벽 3시에 트래픽이 치솟으면 누군가 일어나서 `kubectl scale`을 쳐야 한다는 뜻이다.

이 일을 대신해 주는 것이 **HPA(HorizontalPodAutoscaler)**다. HPA는 메트릭(대표적으로 CPU 사용률)을 지켜보다가 Deployment·StatefulSet 같은 워크로드의 **replicas를 자동으로 늘리고 줄이는** 리소스다. 마트에 손님이 몰리면 계산대를 더 열고, 한산해지면 닫는 것과 같다. 점장이 CCTV로 매장 혼잡도를 보고 판단하듯, HPA 컨트롤러는 **metrics-server**가 수집한 사용률을 보고 판단한다.

## 5.1 동작 원리 — metrics-server와 스케일 공식

HPA가 동작하려면 먼저 "지금 Pod들이 얼마나 바쁜가"를 알아야 한다. 이 CCTV 역할을 하는 것이 **metrics-server**다. 각 노드의 kubelet에서 CPU·메모리 사용량을 모아 Metrics API(`metrics.k8s.io`)로 노출하면, HPA 컨트롤러가 이 값을 주기적으로(기본 15초 간격) 조회한다. 주의할 점은 metrics-server가 **클러스터에 기본 내장돼 있지 않은 애드온**이라는 것이다. kind 클러스터에도 당연히 없으므로 직접 설치해야 한다.

```bash
$ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

로컬 테스트 환경에서는 kubelet 인증서 검증을 끄는 `--kubelet-insecure-tls` 옵션을 metrics-server 컨테이너 args에 추가해야 할 수 있다. 공식 문서가 명시하듯 **테스트 용도 전용** 옵션이므로 프로덕션에서는 쓰지 않는다.

메트릭이 준비되면 HPA 컨트롤러는 다음 공식으로 원하는 replicas를 계산한다.

$$
\text{desiredReplicas} = \left\lceil \text{currentReplicas} \times \frac{\text{currentMetricValue}}{\text{desiredMetricValue}} \right\rceil
$$

- **currentMetricValue** — 현재 Pod들의 평균 메트릭 값 (예: 평균 CPU 사용률 100%)
- **desiredMetricValue** — 내가 선언한 목표 값 (예: CPU 사용률 50%)
- **ceil** — 올림. Pod는 반 개로 띄울 수 없으니 올림한다.

직관은 간단하다 — **"지금이 목표의 몇 배나 바쁜가"만큼 Pod 수를 곱한다**. 목표 사용률 50%인데 실제 100%면 2배로 늘리고, 실제 25%면 절반으로 줄인다. 현재/목표 비율이 1.0에서 기본 10%(tolerance) 이내면 굳이 움직이지 않는다 — 사용률 52% 정도로는 스케일이 일어나지 않는다는 뜻이다.

여기서 CPU "사용률(utilization)"은 절대량이 아니라 **Pod에 설정한 resource request 대비 백분율**이다. request가 200m인 컨테이너가 100m을 쓰고 있으면 사용률 50%다. 뒤집어 말하면, **request를 설정하지 않은 Pod에는 CPU 사용률 기반 HPA가 동작할 수 없다**.

## 5.2 kubectl autoscale — 한 줄로 만들기

HPA도 YAML로 선언할 수 있지만, 간단한 경우는 명령 한 줄이면 된다.

```bash
$ kubectl autoscale deployment web --cpu-percent=50 --min=2 --max=10
horizontalpodautoscaler.autoscaling/web autoscaled

$ kubectl get hpa
NAME   REFERENCE        TARGETS       MINPODS   MAXPODS   REPLICAS   AGE
web    Deployment/web   cpu: 0%/50%   2         10        2          30s
```

- `--cpu-percent=50` — 목표 평균 CPU 사용률.
- `--min=2` / `--max=10` — replicas의 하한(minReplicas)과 상한(maxReplicas). HPA는 **이 범위 안에서만** 움직인다. 트래픽이 아무리 치솟아도 10개 이상 만들지 않고(비용 폭주 방지), 아무리 한산해도 2개는 남긴다(가용성 확보).
- HPA는 scale 서브리소스를 지원하는 워크로드(Deployment, StatefulSet 등)에만 붙일 수 있다. **DaemonSet에는 적용되지 않는다** — "노드당 1개"라는 정의 자체가 개수 조절과 맞지 않기 때문이다.

참고로 HPA가 Pod의 **개수**를 조절하는 수평 확장이라면, Pod **하나에 할당된 CPU/메모리 자체**를 키우는 수직 확장은 별도 프로젝트인 **VPA(Vertical Pod Autoscaler)**가 담당한다.

---

# 6. NetworkPolicy — 기본값은 "전부 허용"이다

여기까지 오면서 한 번도 언급하지 않은 사실이 있다. Kubernetes 클러스터 안에서는 기본적으로 **모든 Pod가 다른 모든 Pod와 자유롭게 통신할 수 있다**. 공식 문서 표현으로 Pod는 기본적으로 non-isolated — 들어오는(ingress) 트래픽도, 나가는(egress) 트래픽도 전부 허용이다. frontend Pod가 결제 DB Pod에 직접 붙는 것을 막는 장치가 아무것도 없다는 뜻이다.

회사로 치면 **모든 사무실 문이 항상 열려 있는 상태**다. 인턴이 금고실에 들어가도 아무도 막지 않는다. 침입자가 사무실 하나를 뚫으면 건물 전체를 휘젓고 다닐 수 있다. 이 문에 출입증 규칙을 다는 것이 **NetworkPolicy**다 — 네임스페이스와 라벨을 기준으로 Pod 간 트래픽을 제한하는 클러스터 내부 방화벽 규칙이라고 보면 된다.

## 6.1 규칙의 구조 — podSelector, Ingress, Egress

NetworkPolicy 하나는 크게 세 부분으로 이루어진다. "DB Pod에는 backend Pod만 접근할 수 있다"는 규칙을 예로 보자.

```yaml
# db-allow-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: db-allow-backend
spec:
  podSelector: # 1) 누구를 보호할 것인가
    matchLabels:
      app: db
  policyTypes: # 2) 어느 방향을 제한할 것인가
    - Ingress
  ingress: # 3) 무엇을 허용할 것인가
    - from:
        - podSelector:
            matchLabels:
              app: backend
      ports:
        - protocol: TCP
          port: 5432
```

- `podSelector` — 이 정책이 **적용되는 대상**. `app: db` 라벨이 붙은 Pod가 보호 대상이 된다.
- `policyTypes` — `Ingress`(들어오는 트래픽), `Egress`(나가는 트래픽), 또는 둘 다. 여기서 말하는 Ingress는 방향(수신)을 뜻하는 단어로, 4장의 Ingress 리소스와는 **이름만 같은 별개 개념**이니 헷갈리지 말자.
- `ingress.from` — 허용할 출발지 목록. 여기서는 같은 네임스페이스의 `app: backend` Pod만 TCP 5432 포트로 허용한다. `podSelector` 외에 `namespaceSelector`(특정 네임스페이스의 Pod들)와 `ipBlock`(IP 대역)도 쓸 수 있다.

이 정책이 적용되는 순간 `app: db` Pod는 **isolated 상태**가 된다. NetworkPolicy는 차단 목록이 아니라 **허용 목록(whitelist)** 방식이라, 정책에 명시된 트래픽 외에는 전부 차단된다 — frontend Pod가 5432로 접속을 시도하면 거부된다. 정책 여러 개가 같은 Pod를 선택하면 허용 범위는 **합집합**으로 합쳐진다. 한 가지 더: 통신이 성립하려면 **출발 Pod의 egress와 도착 Pod의 ingress 양쪽 모두**에서 허용되어야 한다. 한쪽이라도 막으면 연결은 실패한다.

## 6.2 주의 — CNI가 지원해야 동작한다

가장 중요한 함정이다. **NetworkPolicy는 네트워크 플러그인(CNI)이 구현한다**. 공식 문서가 명시하듯, 정책을 지원하는 네트워킹 솔루션 없이 NetworkPolicy 리소스만 만들면 **아무 효과가 없다**. 4장에서 본 Ingress 리소스와 Ingress Controller의 관계와 똑같은 구도다 — 규칙(리소스)과 실행자(CNI)가 분리되어 있고, 실행자가 없으면 규칙은 종이 조각이다. Calico, Cilium 등이 NetworkPolicy를 지원하는 대표적인 CNI다.

kind에서 NetworkPolicy를 제대로 실습하고 싶다면, kind 공식 문서가 안내하는 대로 클러스터 설정에 `disableDefaultCNI: true`를 넣어 기본 CNI를 끄고 Calico 같은 CNI를 직접 설치하는 방법이 있다. 어떤 환경에서든 정책을 만들었는데 여전히 모든 트래픽이 통과한다면, 클러스터의 CNI가 NetworkPolicy를 강제하지 않고 있을 가능성부터 의심하자.

---

# 7. Gateway API — Ingress의 후계자

4장의 안내 데스크(Ingress)는 잘 동작하지만, 오래 쓰다 보면 한계가 드러난다.

**첫째, 표현력이 부족하다.** Ingress 스펙이 표준으로 표현할 수 있는 것은 사실상 host/path 기반 분기 정도다. 헤더 기반 매칭, 트래픽 가중치(카나리 배포) 같은 기능은 표준에 없어서 **컨트롤러별 annotation**으로 해결해 왔다 — Gateway API 공식 문서도 이런 기능들이 "Ingress에서는 커스텀 annotation으로만 가능했다"고 명시한다. annotation은 표준이 아니므로 ingress-nginx용으로 잔뜩 달아 둔 annotation은 다른 컨트롤러로 갈아타는 순간 전부 무용지물이 된다. 매뉴얼(표준 스펙)에 적을 수 없는 지시를 안내원(컨트롤러)마다 다른 방언으로 전달해 온 셈이다.

**둘째, API가 동결(frozen)됐다.** Kubernetes 공식 문서는 Ingress API가 GA(안정) 상태로 유지는 되지만 **더 이상 개발되지 않으며 새 기능도 추가되지 않는다**고 명시하고, Gateway API 사용을 권장한다.

**셋째, 역할 분리가 없다.** 인프라 설정과 앱의 라우팅 규칙이 한 리소스에 얽혀 있어서, "로드밸런서는 인프라팀이 관리하고 라우팅 규칙은 각 개발팀이 관리한다" 같은 흔한 조직 구조를 표현할 수 없다.

**Gateway API**는 이 문제들을 처음부터 다시 설계한 **Ingress의 공식 후계자**다. Kubernetes 내장 API가 아니라 CRD(커스텀 리소스) 애드온으로 제공되며, 핵심은 리소스를 **역할별로 셋으로 쪼갠 것**이다.

| 리소스           | 역할                                                       | 주 담당자                  |
| ---------------- | ---------------------------------------------------------- | -------------------------- |
| **GatewayClass** | 어떤 구현체(컨트롤러)가 트래픽을 처리하는지 정의           | 인프라 프로바이더 / 구현체 |
| **Gateway**      | 트래픽을 받아들이는 인프라 인스턴스 (리스너·포트·프로토콜) | 클러스터 운영자            |
| **HTTPRoute**    | host/path/헤더 기반 라우팅 규칙, 백엔드 Service 지정       | 앱 개발자                  |

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-06-networking/gateway-api-roles.png" class="img-fluid rounded z-depth-1" alt="GatewayClass, Gateway, HTTPRoute가 인프라 프로바이더, 클러스터 운영자, 앱 개발자 역할별로 분리된 Gateway API 구조" %}

- 인프라팀은 Gateway(공용 입구)를 한 번 세팅하고, 각 개발팀은 자기 HTTPRoute만 만들어 그 Gateway에 연결(parentRefs)한다. Ingress에서는 이 경계가 리소스 차원에서 존재하지 않았다.
- 헤더 매칭·트래픽 가중치 같은 고급 라우팅이 annotation이 아니라 **표준 스펙 필드**다. HTTP 외에 gRPC 라우팅(GRPCRoute) 등도 표준으로 지원한다.

Ingress와 나란히 놓고 비교하면 이렇다.

| 항목                               | Ingress                             | Gateway API                                    |
| ---------------------------------- | ----------------------------------- | ---------------------------------------------- |
| API 상태                           | GA이지만 동결 — 신규 기능 추가 없음 | 활발히 개발 중인 후계 표준                     |
| 제공 방식                          | Kubernetes 내장 API                 | CRD 애드온 (별도 설치)                         |
| 리소스 구조                        | Ingress 하나에 전부                 | GatewayClass / Gateway / HTTPRoute 역할별 분리 |
| 고급 라우팅 (헤더 매칭, 가중치 등) | 컨트롤러별 annotation 의존          | 표준 스펙에 포함                               |
| 프로토콜                           | HTTP/HTTPS 중심                     | HTTPRoute, GRPCRoute 등                        |
| 역할 분리                          | 없음                                | 인프라 운영자와 앱 개발자 분리                 |

이미 커뮤니티 표준의 무게 중심도 옮겨가는 중이다. 2025년 2월 18일부터 적용된 **CKA(Certified Kubernetes Administrator) 시험 개편**에서 "Use the Gateway API to manage Ingress traffic"가 Servicing and Networking 도메인의 공식 출제 항목으로 들어갔다.

그렇다고 4장에서 배운 Ingress가 당장 쓸모없어지는 것은 아니다. 공식 문서 기준으로 Kubernetes 프로젝트는 Ingress를 제거할 계획이 없고, 기존 Ingress와 컨트롤러 생태계는 여전히 광범위하게 쓰인다. 입문 단계에서는 Ingress로 L7 라우팅 개념을 잡고, Gateway API는 **같은 문제를 역할 분리와 표준화된 표현력으로 다시 푼 후계 표준**으로 기억해 두면 충분하다.

---

# 8. 그래서 언제 무엇을 쓰나

| 상황                                                | 선택                  | 이유                                                     |
| --------------------------------------------------- | --------------------- | -------------------------------------------------------- |
| 클러스터 내부 서비스 간 통신 (백엔드 → DB 등)       | ClusterIP (기본)      | 외부 노출이 필요 없고, DNS 이름으로 안정적으로 접근      |
| 로컬/개발 환경에서 잠깐 외부 확인                   | NodePort              | 추가 인프라 없이 노드 포트(30000-32767)만 열면 됨        |
| 클라우드에서 단일 서비스를 L4로 노출 (TCP, gRPC 등) | LoadBalancer          | 클라우드 LB가 외부 IP·고가용성을 제공                    |
| 여러 HTTP 서비스를 도메인/경로로 나눠 노출          | Ingress (+Controller) | 입구 하나로 L7 라우팅, LB 비용 절감, TLS 종료도 한곳에서 |

한 줄로 압축하면 — **내부 통신은 ClusterIP, L4 외부 노출은 LoadBalancer, HTTP 여러 개의 교통정리는 Ingress**다.

---

# 9. 마무리

이번 편의 핵심을 요약한다.

| 개념               | 역할                                                     | 비유                                  |
| ------------------ | -------------------------------------------------------- | ------------------------------------- |
| Service            | 계속 바뀌는 Pod 집합 앞의 고정 접점(가상 IP + DNS)       | 회사 대표 전화번호                    |
| EndpointSlice      | Selector로 추적한 실제 Pod IP 명단                       | 지금 근무 중인 직원 명단              |
| kube-proxy         | 각 노드에서 Service 트래픽을 Pod로 라우팅                | 전화 교환대                           |
| Service DNS        | `<svc>.<ns>.svc.cluster.local` → ClusterIP               | 전화번호부                            |
| Ingress 리소스     | host/path 기반 L7 라우팅 규칙 선언                       | 안내 데스크 업무 매뉴얼               |
| Ingress Controller | 규칙을 실제로 실행하는 별도 설치 컴포넌트                | 매뉴얼대로 일하는 안내원              |
| HPA                | 메트릭 기준으로 replicas를 min-max 안에서 자동 증감      | 혼잡도 보고 계산대를 여닫는 점장      |
| NetworkPolicy      | 라벨·네임스페이스 기반 Pod 트래픽 허용 규칙 (CNI가 강제) | 사무실 출입증 규칙                    |
| Gateway API        | 역할 분리로 다시 설계된 Ingress의 후계 표준              | 업무 분담이 명확한 차세대 안내 시스템 |

Pod IP가 아무리 바뀌어도 흔들리지 않는 접점을 만드는 것, 그것이 Kubernetes 네트워킹 추상화의 존재 이유다. 그런데 접점만 고정한다고 끝이 아니다. Pod가 재시작되면 **컨테이너 안의 데이터도 함께 사라진다**.

> 다음 글 [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)에서는 사라지는 데이터를 붙잡는 볼륨 추상화와, 설정을 이미지 밖으로 빼내는 방법을 다룬다.

---

# 참고 문헌

- [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua, SK Devocean, 2024) — 시리즈 로드맵 출처
- [Kubernetes Docs — Service](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Kubernetes Docs — DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)
- [Kubernetes Docs — Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Kubernetes Docs — EndpointSlices](https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/)
- [Kubernetes Docs — Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Kubernetes Docs — HorizontalPodAutoscaler Walkthrough](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
- [Kubernetes Metrics Server (kubernetes-sigs)](https://github.com/kubernetes-sigs/metrics-server)
- [Kubernetes Docs — Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Kubernetes Docs — Gateway API](https://kubernetes.io/docs/concepts/services-networking/gateway/)
- [Gateway API — Kubernetes SIG-Network](https://gateway-api.sigs.k8s.io/)
- [Linux Foundation — CKA Program Changes (2025-02-18)](https://training.linuxfoundation.org/certified-kubernetes-administrator-cka-program-changes/)
- [kind — Ingress Guide](https://kind.sigs.k8s.io/docs/user/ingress/)
- [kind — Configuration (Extra Port Mappings)](https://kind.sigs.k8s.io/docs/user/configuration/)
- [ingress-nginx — kind provider manifest](https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml)
