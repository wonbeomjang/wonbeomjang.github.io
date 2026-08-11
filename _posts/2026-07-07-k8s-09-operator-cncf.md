---
layout: post
title: "Kubernetes 확장과 생태계 — Operator와 CNCF Projects"
date: 2026-07-07 09:00:00 +0900
description: "CRD와 Custom Controller로 나만의 리소스를 만드는 Operator 패턴, cert-manager 실습, 그리고 CNCF 성숙도 3단계로 읽는 생태계 지도 — K8s 입문 시리즈 마지막 편"
categories: [infra]
tags: [kubernetes, infra, operator, crd, helm, kustomize, cncf]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 입문 시리즈**의 아홉 번째 글이다.
>
> - [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)
> - [02: 내 노트북에 클러스터 만들기 — kind와 kubectl](/blog/2026/k8s-02-local-setup/)
> - [03: Kubernetes 아키텍처 — Control Plane과 Node](/blog/2026/k8s-03-architecture/)
> - [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)
> - [05: 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet](/blog/2026/k8s-05-workloads/)
> - [06: 네트워킹 — Service와 Ingress](/blog/2026/k8s-06-networking/)
> - [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)
> - [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)
> - **09: 확장과 생태계 — Operator와 CNCF Projects** ← 현재 글
> - [10: 관측성 — 로그·메트릭과 Prometheus/Grafana](/blog/2026/k8s-10-observability/)
>
> 이 시리즈의 커리큘럼은 SK Devocean의 [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua) 글의 학습 로드맵을 바탕으로 구성했다.

# 1. 들어가며 — "우리 회사만의 리소스"를 만들 수 있다면?

지금까지 시리즈에서 다룬 것은 전부 Kubernetes에 **내장된** 리소스였다. Pod, Deployment, Service, PVC, Role... 전부 `kubectl api-resources`를 치면 나오는, 설치하자마자 주어지는 어휘들이다.

그런데 실무에서는 이런 요구가 금방 생긴다.

- "PostgreSQL 클러스터를 배포하고, **백업·장애 조치·버전 업그레이드까지 자동으로** 해주면 좋겠는데."
- "인증서가 만료되기 전에 **알아서 갱신**되면 좋겠는데."
- "우리 팀의 배포 단위는 'ML 모델 서빙'인데, 이걸 `kubectl get modelserving`처럼 다루고 싶은데."

Deployment와 Service만으로 이걸 조립하려면 사람이 runbook(운영 절차서)을 들고 매번 수동으로 개입해야 한다. Kubernetes 설계자들은 이 문제를 예견하고, Kubernetes를 **완성된 제품이 아니라 확장 가능한 플랫폼**으로 만들었다.

비유하자면 이렇다. 지금까지 우리는 스마트폰의 **기본 앱**(전화, 카메라, 메시지)만 쓴 셈이다. 이번 편에서는 **앱스토어**의 존재를 배운다. 누구나 앱(새 리소스 타입 + 그것을 움직이는 컨트롤러)을 만들어 올릴 수 있고, 잘 만든 앱은 생태계(CNCF)의 검증을 거쳐 사실상 표준이 된다.

---

# 2. 컨트롤러 패턴 복습 — 모든 것은 reconcile 루프다

확장 얘기를 하기 전에, [03편](/blog/2026/k8s-03-architecture/)과 [05편](/blog/2026/k8s-05-workloads/)에서 본 Kubernetes의 핵심 동작 원리를 복습하자.

Kubernetes의 모든 자동화는 하나의 패턴으로 요약된다.

> **원하는 상태(desired state)를 선언하면, 컨트롤러가 현재 상태(current state)를 계속 관찰하면서 둘의 차이를 메운다.** 이 반복을 **reconcile 루프**(조정 루프)라고 부른다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-09-operator-cncf/reconcile-loop.png" class="img-fluid rounded z-depth-1" alt="reconcile 루프 — 사용자가 원하는 상태를 선언하면 kube-apiserver가 etcd에 저장하고, 컨트롤러가 현재 상태를 관찰하며 원하는 상태와 비교해 같으면 다시 관찰로, 다르면 차이를 메우는 조치 후 다시 관찰로 돌아가는 순환 구조" %}

- 사용자는 **방법(how)이 아니라 결과(what)** 를 선언한다. "replicas: 3"이라고 쓰지, "Pod를 하나씩 띄워라"라고 쓰지 않는다.
- 컨트롤러는 **한 번 실행하고 끝나는 스크립트가 아니라, 무한히 도는 루프**다. Pod가 죽으면 루프의 다음 바퀴에서 차이가 감지되고 자동 복구된다.
- 이 구조 덕분에 장애 복구, 스케일링, 롤링 업데이트가 전부 "차이 메우기"라는 같은 원리로 동작한다.

일상 비유로는 **보일러의 온도 조절기**가 정확하다. "24도로 맞춰줘"(desired state)라고 설정하면, 온도 조절기는 현재 온도(current state)를 계속 재면서 낮으면 보일러를 켜고 높으면 끈다. 사용자가 보일러 밸브를 직접 조작하지 않는다.

여기서 이번 편의 핵심 질문이 나온다.

> Deployment 컨트롤러가 "replicas 차이"를 메우듯이, **"PostgreSQL 백업이 안 됐다"거나 "인증서가 곧 만료된다" 같은 차이를 메우는 컨트롤러를 우리가 직접 만들 수는 없을까?**

만들 수 있다. 그러려면 두 가지가 필요하다. **새 리소스 타입(명사)** 과 **그것을 움직이는 컨트롤러(동사)** 다.

---

# 3. CRD — Kubernetes API에 새 명사를 등록한다

**CRD(Custom Resource Definition)** 는 Kubernetes API에 새로운 리소스 타입을 등록하는 내장 메커니즘이다. CRD를 등록하면 그 순간부터 kube-apiserver가 새 타입을 이해하고, etcd에 저장해주고, `kubectl get/describe/delete`가 전부 동작한다.

- **CRD**: "이제부터 `Certificate`라는 타입이 존재한다. 필드는 이렇다." — **타입의 정의** (클래스 선언에 해당)
- **CR(Custom Resource)**: "demo-cert라는 Certificate를 하나 만들어줘." — **그 타입의 인스턴스** (객체 생성에 해당)

왜 이런 방식일까? 설계 의도가 중요하다.

1. **API 서버를 고치지 않고 확장한다.** 새 리소스가 필요할 때마다 Kubernetes 소스 코드를 수정해서 재컴파일해야 한다면 생태계는 만들어질 수 없다. CRD는 실행 중인 클러스터에 YAML 한 장으로 타입을 추가한다.
2. **기존 도구가 공짜로 따라온다.** CR도 일반 리소스처럼 etcd에 저장되고 RBAC([08편](/blog/2026/k8s-08-rbac/))의 통제를 받는다. `kubectl`, `kubectl auth can-i`, 감사 로그가 전부 그대로 동작한다.
3. **선언형 모델이 유지된다.** CR도 spec(원하는 상태)과 status(현재 상태)를 갖는다. 즉 reconcile 루프에 태울 수 있는 형태다.

단, 여기서 중요한 함정이 있다. **CRD만 등록하면 아무 일도 일어나지 않는다.** kube-apiserver는 CR을 받아서 etcd에 저장할 뿐, 그 의미를 모른다. `Certificate`를 만들어도 실제 인증서를 발급해줄 주체가 없다. 명사만 있고 동사가 없는 상태다.

그 동사 역할을 하는 것이 **Custom Controller** — 우리가 등록한 CR을 watch하면서 reconcile 루프를 도는 프로그램이다. 그리고 이 둘을 묶은 패턴에 이름이 붙어 있다.

---

# 4. Operator = CRD + Custom Controller + 운영 지식

**Operator**는 다음 세 가지의 결합이다.

| 구성 요소          | 역할                                     | 비유                             |
| ------------------ | ---------------------------------------- | -------------------------------- |
| CRD                | 도메인 개념을 리소스 타입으로 정의       | 새 어휘(명사) 등록               |
| Custom Controller  | CR을 watch하며 reconcile 루프 실행       | 그 어휘를 행동으로 옮기는 일꾼   |
| 운영 지식의 코드화 | 백업, 장애 조치, 갱신 같은 절차를 코드로 | 베테랑 운영자의 runbook을 자동화 |

이름의 유래가 곧 정의다. 이 패턴은 2016년 CoreOS가 "Introducing Operators"라는 글에서 처음 제안했는데, **사람 운영자(human operator)가 하던 일을 소프트웨어로 옮긴다**는 뜻에서 Operator라고 이름 붙였다. 당시 CoreOS는 etcd Operator와 Prometheus Operator를 함께 공개했다. 예를 들어 "etcd 클러스터 멤버가 죽으면 새 멤버를 추가하고 데이터를 복제한다" 같은, 문서와 담당자의 머릿속에 있던 도메인 지식을 컨트롤러 코드에 넣은 것이다.

현재는 Kubernetes 공식 문서에도 Operator pattern이 표준 확장 패턴으로 문서화되어 있다. 대표적인 Operator 몇 개만 보자.

| Operator            | 다루는 도메인       | 등록하는 대표 CR                         |
| ------------------- | ------------------- | ---------------------------------------- |
| cert-manager        | TLS 인증서 수명주기 | Certificate, Issuer, ClusterIssuer       |
| Prometheus Operator | 모니터링 스택 운영  | Prometheus, ServiceMonitor, Alertmanager |
| etcd Operator       | etcd 클러스터 운영  | EtcdCluster (최초의 Operator 중 하나)    |

구조를 그림으로 보면 이렇다. cert-manager를 예로 들었다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-09-operator-cncf/operator-cert-manager.png" class="img-fluid rounded z-depth-1" alt="Operator 구조(cert-manager 예시) — Kubernetes API에 등록된 CRD와 CR(demo-cert)을 cert-manager 컨트롤러가 watch하고, 발급자(Issuer)에게 발급/갱신을 요청해 서명된 인증서를 받아 Secret(demo-cert-tls)에 저장하며 CR의 status를 갱신한다" %}

- 사용자는 "이 도메인의 인증서가 있어야 한다"는 **결과만 선언**한다(CR).
- cert-manager 컨트롤러가 발급자에게 서명을 요청하고, 결과를 [07편](/blog/2026/k8s-07-storage-config/)에서 배운 **Secret**으로 저장한다.
- 만료가 다가오면? 사람이 달력을 볼 필요가 없다. reconcile 루프가 "곧 만료됨 = 원하는 상태와의 차이"로 감지하고 **알아서 갱신**한다. 이것이 운영 지식의 코드화다.

핵심은 이것이다. **Operator는 새로운 마법이 아니라, 2절의 reconcile 루프를 사용자 정의 도메인에 적용한 것뿐이다.** Deployment 컨트롤러가 "Pod 개수"를 조정하듯, cert-manager는 "인증서의 유효 상태"를 조정한다.

---

# 5. 실습 — cert-manager 설치하고 새 동사를 확인하기

말로만 들으면 추상적이니, [02편](/blog/2026/k8s-02-local-setup/)에서 만든 kind 클러스터에 cert-manager를 직접 설치해보자.

## 5.1 설치

공식 문서의 기본 설치 방식은 정적 매니페스트 한 장을 apply하는 것이다. 버전은 계속 올라가므로, 실제 설치 전에 [공식 설치 문서](https://cert-manager.io/docs/installation/)에서 최신 버전을 확인하고 URL의 버전 부분을 바꿔 쓰자.

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.20.3/cert-manager.yaml
```

설치가 끝나면 cert-manager 네임스페이스에 컨트롤러 Pod들이 뜬다.

```bash
kubectl get pods -n cert-manager
```

```
NAME                                       READY   STATUS    RESTARTS   AGE
cert-manager-5f6b885b4-xk2jp               1/1     Running   0          80s
cert-manager-cainjector-747d59b6c5-9wvxq   1/1     Running   0          80s
cert-manager-webhook-7d4f5d8f6b-tq8zl      1/1     Running   0          80s
```

## 5.2 클러스터에 새 명사가 생겼는지 확인

설치 전에는 존재하지 않던 리소스 타입이 등록됐는지 CRD 목록으로 확인한다.

```bash
kubectl get crds | grep cert-manager
```

```
certificaterequests.cert-manager.io    2026-07-07T00:01:12Z
certificates.cert-manager.io           2026-07-07T00:01:12Z
challenges.acme.cert-manager.io        2026-07-07T00:01:13Z
clusterissuers.cert-manager.io         2026-07-07T00:01:13Z
issuers.cert-manager.io                2026-07-07T00:01:13Z
orders.acme.cert-manager.io            2026-07-07T00:01:13Z
```

`Certificate`, `Issuer` 같은 타입이 Kubernetes API의 정식 시민이 됐다. 이제 `kubectl get certificate`가 에러 없이 동작한다.

## 5.3 Certificate 리소스로 인증서 자동 발급

가장 간단한 발급자인 SelfSigned(자체 서명) Issuer로 인증서를 하나 발급해보자. 파일 하나에 Issuer와 Certificate를 함께 선언한다.

```yaml
# selfsigned-demo.yaml
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: selfsigned-issuer
spec:
  selfSigned: {}
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: demo-cert
spec:
  secretName: demo-cert-tls
  commonName: demo.example.local
  dnsNames:
    - demo.example.local
  issuerRef:
    name: selfsigned-issuer
    kind: Issuer
```

주목할 점: 우리는 **openssl 명령을 한 줄도 쓰지 않았다.** "이런 인증서가 존재해야 한다"고 선언했을 뿐이다.

```bash
kubectl apply -f selfsigned-demo.yaml
kubectl get certificate
```

```
NAME        READY   SECRET          AGE
demo-cert   True    demo-cert-tls   6s
```

READY가 True가 되는 몇 초 사이에 cert-manager 컨트롤러가 reconcile 루프를 돌며 키 생성 → 서명 요청 → Secret 저장을 대신 해줬다. 결과물을 확인하자.

```bash
kubectl get secret demo-cert-tls
```

```
NAME            TYPE                DATA   AGE
demo-cert-tls   kubernetes.io/tls   3      30s
```

[07편](/blog/2026/k8s-07-storage-config/)에서 배운 `kubernetes.io/tls` 타입 Secret에 인증서(tls.crt)와 개인 키(tls.key)가 담겼다. 이 Secret은 [06편](/blog/2026/k8s-06-networking/)의 Ingress에 그대로 연결해 HTTPS를 켤 수 있다. 시리즈에서 배운 조각들이 Operator를 통해 하나로 이어지는 순간이다.

실습이 끝났으면 정리한다.

```bash
kubectl delete -f selfsigned-demo.yaml
kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.20.3/cert-manager.yaml
```

---

# 6. 패키지 관리 — Helm과 Kustomize

시리즈 내내 우리는 YAML을 한 장씩 손으로 썼다. 배우는 단계에서는 그게 옳은 방법이지만, 실무로 가면 금방 한계가 온다.

- 앱 하나를 배포하는 데 Deployment, Service, Ingress, ConfigMap, Secret... **YAML이 대여섯 장**씩 필요하다. 5절에서 apply한 cert-manager 매니페스트 한 장에도 수십 개의 리소스가 들어 있었다.
- 같은 앱을 dev/staging/prod 세 환경에 배포하려면? 환경마다 replicas, 이미지 태그, 도메인, 리소스 제한이 다르다. 순진하게 풀면 **디렉터리째 복사해서 세 벌**을 만들게 된다.
- 세 벌이 되는 순간부터 고통이 시작된다. 라벨 추가 같은 공통 수정 하나를 세 곳에 반복해야 하고, 한 곳을 빼먹으면 환경 간 설정이 조용히 어긋난다.

프로그래밍이었다면 당연히 "중복을 함수로 뽑아내자"고 했을 것이다. Kubernetes 생태계에서 이 역할을 맡은 대표 도구가 **Helm**과 **Kustomize**다. 같은 문제를 정반대 철학으로 푸는 것이 흥미로운 지점이다.

## 6.1 Helm — 차트로 묶고, values로 채운다

**Helm**은 Kubernetes의 **패키지 매니저**다. Ubuntu의 apt, macOS의 Homebrew가 프로그램 설치에 대해 하는 일을, Helm은 Kubernetes 리소스 뭉치에 대해 한다. 1절에서 "앱스토어"라는 비유를 썼는데, 그 비유가 가장 문자 그대로 실현된 곳이 Helm이다. 실제로 cert-manager도 5절의 정적 매니페스트 방식 외에 공식 Helm 차트 설치를 지원한다.

핵심 개념은 세 개다.

| 개념            | 의미                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------------- |
| **차트(chart)** | 앱 실행에 필요한 리소스 정의 일체를 묶은 **패키지**                                            |
| **values**      | 차트의 **설정값**. 기본값은 차트 안의 values.yaml에 있고, 설치할 때 파일이나 옵션으로 덮어쓴다 |
| **release**     | 차트를 클러스터에 설치한 **인스턴스**. 같은 차트를 이름만 바꿔 여러 번 설치할 수 있다          |

동작 원리는 **템플릿 렌더링**이다. 차트 안의 매니페스트들은 replicas나 이미지 태그처럼 환경마다 달라질 자리를 이중 중괄호 플레이스홀더로 비워 둔 템플릿이고, 설치 시점에 values의 값이 그 자리에 채워지면서 최종 YAML이 만들어진다. 템플릿 문법 자체는 Helm 공식 문서에 잘 정리되어 있으니, 여기서는 "빈칸 뚫린 YAML 서식 + 값 주입"이라는 그림만 잡으면 된다.

환경별 배포 문제는 이렇게 풀린다. **차트는 하나, values 파일만 환경 수만큼.**

```bash
# 차트 저장소를 등록하고 차트를 검색한다
helm repo add <저장소이름> <저장소URL>
helm search repo <키워드>

# 같은 차트를 환경별 values 파일로 각각 설치한다
helm install myapp-dev ./mychart -f values-dev.yaml
helm install myapp-prod ./mychart -f values-prod.yaml

# 업그레이드하고, 문제가 생기면 이전 리비전으로 되돌린다
helm upgrade myapp-prod ./mychart -f values-prod.yaml
helm rollback myapp-prod 1

# 설치된 release 목록 확인과 제거
helm list
helm uninstall myapp-dev
```

Helm의 또 다른 무기는 **릴리스 관리**다. release는 업그레이드할 때마다 리비전 히스토리가 쌓이고, 배포가 잘못됐을 때 `helm rollback` 한 줄로 이전 리비전으로 되돌릴 수 있다. 남이 만든 앱을 설치하는 표준 통로이자, 우리 앱을 남에게 배포하는 포장 수단인 셈이다.

## 6.2 Kustomize — 템플릿 없이 패치를 겹친다

**Kustomize**는 정반대 철학에서 출발한다. **템플릿이 없다(template-free).** 완성된 일반 YAML을 그대로 두고, 환경별 **차이만 패치로 얹는다**.

구조는 **base + overlay**다.

```
myapp/
├── base/                        # 모든 환경의 공통분모 (그 자체로 유효한 YAML)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    │   └── kustomization.yaml   # base 참조 + replicas 1로 패치
    └── prod/
        └── kustomization.yaml   # base 참조 + replicas 5, 리소스 제한 패치
```

각 디렉터리의 kustomization.yaml이 "무엇을 가져와서(resources) 어떻게 고칠지"를 선언한다. 패치 외에도 공통 라벨/네임스페이스/이름 접두사 부여, 파일로부터 ConfigMap·Secret 생성(configMapGenerator, secretGenerator) 같은 기능을 제공한다. 중요한 차이: base의 YAML은 빈칸 뚫린 템플릿이 아니라 **그 자체로 완성된 매니페스트**라서, 그냥 `kubectl apply -f`로 적용해도 동작한다.

가장 큰 매력은 **kubectl에 내장**되어 있다는 점이다. kubectl 1.14부터 별도 설치 없이 바로 쓸 수 있다.

```bash
# 최종 렌더링 결과를 미리 확인한다
kubectl kustomize overlays/prod

# 렌더링해서 바로 클러스터에 적용한다
kubectl apply -k overlays/prod
```

일상 비유로 정리하면 이렇다. Helm이 **빈칸 뚫린 서식에 값을 채워 문서를 찍어내는** 방식이라면, Kustomize는 **완성된 원본 문서 위에 수정 스티커를 붙이는** 방식이다.

## 6.3 무엇을 언제 쓰나

| 구분        | Helm                                                     | Kustomize                                |
| ----------- | -------------------------------------------------------- | ---------------------------------------- |
| 접근 방식   | 템플릿에 values를 채워 렌더링                            | 완성된 YAML(base)에 패치(overlay)를 겹침 |
| 단위        | 차트(패키지) → release(설치 인스턴스)                    | 디렉터리(base + overlay)                 |
| 도구        | 별도 CLI(`helm`)                                         | kubectl 내장(`kubectl apply -k`)         |
| 릴리스 관리 | 리비전 히스토리 + `helm rollback`                        | 없음 (Git 히스토리에 의존)               |
| 장점        | 패키지 공유·배포에 강함. 버전·롤백 관리 내장             | 단순함. 원본 YAML이 그대로 읽힘          |
| 단점        | 템플릿 문법 학습 곡선. 렌더링 전에는 최종 YAML이 안 보임 | 패키지 배포·버전 관리 기능이 없음        |
| 언제 쓰나   | 남이 만든 앱 설치, 여러 곳에 배포할 패키지 제작          | 우리 팀 앱의 dev/prod 환경별 변형 관리   |

둘 중 하나만 골라야 하는 것도 아니다. **남의 앱은 Helm으로 설치하고, 우리 앱은 Kustomize로 관리**하는 조합이 흔하다. 두 도구 모두 이제는 기본기로 취급된다 — 2025년 2월 개편된 CKA(Certified Kubernetes Administrator) 시험 커리큘럼의 "Cluster Architecture, Installation and Configuration" 도메인에 "Helm과 Kustomize를 사용해 클러스터 구성 요소를 설치한다"는 항목이 정식으로 들어갔다.

그렇다면 Helm 같은 도구는 누가 검증하고 육성하는가? 이 질문이 자연스럽게 다음 절로 이어진다.

---

# 7. CNCF Projects — 생태계의 지도를 읽는 법

5절에서 설치한 cert-manager는 개인 프로젝트가 아니다. [01편](/blog/2026/k8s-01-history/)에서 소개한 **CNCF(Cloud Native Computing Foundation)** 가 육성해온 프로젝트다. 클라우드 네이티브 세계에는 이런 오픈소스 프로젝트가 수없이 많은데, 문제는 **"어떤 걸 믿고 프로덕션에 도입할 것인가"** 다.

CNCF는 이 판단을 돕기 위해 프로젝트를 **성숙도 3단계**로 나눈다.

| 단계           | 의미                                                          | 기술 채택 곡선 대응 |
| -------------- | ------------------------------------------------------------- | ------------------- |
| **Sandbox**    | 실험 단계. 아직 프로덕션에서 널리 검증되지 않은 초기 프로젝트 | Innovators          |
| **Incubating** | 소수의 사용자가 프로덕션에서 성공적으로 사용 중               | Early Adopters      |
| **Graduated**  | 안정적이고 널리 채택됨. 프로덕션 준비 완료로 간주             | Early Majority      |

CNCF는 이 3단계가 기술 채택 이론(Crossing the Chasm)의 Innovators → Early Adopters → Early Majority 층에 대응한다고 설명한다. 즉 **회사의 위험 감수 성향에 따라 어느 단계 프로젝트까지 도입할지 판단하는 기준**으로 쓰라는 것이다. 참고로 활동이 멈춘 프로젝트는 Archived로 분류된다.

단계가 올라가는 것은 인기 투표가 아니다. Graduated로 승격하려면 CNCF 기술감독위원회(TOC)의 심사를 통과해야 하는데, 요구 사항에는 다음이 포함된다.

- **여러 조직 출신의 메인테이너**: 최소 2개 이상 조직의 메인테이너 — 한 회사가 손을 떼도 프로젝트가 살아남는가(생존성)
- **제3자 보안 감사**: 외부 보안 감사 수행과 발견 사항의 해결 추적
- **OpenSSF Best Practices 배지**: 오픈소스 보안 모범 사례 준수 인증
- **실제 채택 검증**: 프로덕션에서 사용하는 독립적인 채택 조직들(adopter) 확인
- **거버넌스 문서화**: 메인테이너 선출·승계 절차, 기여 절차, 행동 강령(Code of Conduct)

말하자면 Graduated는 **"이 프로젝트는 특정 회사의 소유물이 아니고, 보안이 감사됐고, 실제로 여러 곳의 프로덕션에서 돌고 있다"는 CNCF의 보증 마크**다.

대표적인 Graduated 프로젝트를 보면 이 시리즈에서 만난 이름이 여럿 보인다.

| 프로젝트     | 분야                    | 시리즈와의 연결                                                                |
| ------------ | ----------------------- | ------------------------------------------------------------------------------ |
| Kubernetes   | 컨테이너 오케스트레이션 | 이 시리즈 전체. CNCF의 1호 프로젝트([01편](/blog/2026/k8s-01-history/))        |
| etcd         | 분산 key-value 저장소   | 클러스터의 상태 저장소([03편](/blog/2026/k8s-03-architecture/))                |
| containerd   | 컨테이너 런타임         | kubelet이 CRI로 위임하는 런타임([03편](/blog/2026/k8s-03-architecture/))       |
| Prometheus   | 모니터링·관측           | 사실상 표준 메트릭 수집기. Prometheus Operator의 대상                          |
| Envoy        | 서비스 프록시           | L7 프록시. 많은 Ingress/서비스 메시의 데이터 플레인                            |
| Helm         | 패키지 매니저           | YAML 뭉치를 하나의 패키지(차트)로 설치                                         |
| Argo         | CI/CD·워크플로          | GitOps 배포(Argo CD)로 유명                                                    |
| Fluentd      | 로깅                    | 노드별 로그 수집 — DaemonSet의 대표 사례([05편](/blog/2026/k8s-05-workloads/)) |
| cert-manager | 보안·인증서             | 방금 실습한 그 프로젝트                                                        |

전체 목록과 각 프로젝트의 현재 단계는 [cncf.io/projects](https://www.cncf.io/projects/)에서 확인할 수 있다. 프로젝트 수는 계속 변하므로 외울 필요는 없고, **지도를 읽는 법**만 기억하면 된다.

마지막으로 cert-manager의 여정을 보면 이 제도가 실제로 어떻게 굴러가는지 보인다. cert-manager는 2020년 11월 CNCF Sandbox로 들어와, 2022년 9월 Incubating으로 승격했고, CNCF가 후원한 보안 감사를 거쳐 **2024년 9월 Graduated가 됐다**. 우리가 방금 `kubectl apply` 한 줄로 설치한 도구 뒤에는 이런 4년의 검증 과정이 있었던 것이다.

---

# 8. 시리즈 마무리 — 우리가 걸어온 길

9편에 걸친 K8s 입문 시리즈가 끝났다. 한 줄씩 되짚어 보자.

| 편                                      | 주제            | 한 줄 요약                                                                             |
| --------------------------------------- | --------------- | -------------------------------------------------------------------------------------- |
| [01](/blog/2026/k8s-01-history/)        | 탄생과 배경     | Google Borg의 경험이 오픈소스가 되어 CNCF 생태계의 중심이 됐다                         |
| [02](/blog/2026/k8s-02-local-setup/)    | kind와 kubectl  | 노트북 위 컨테이너로 진짜 클러스터를 만들고 kubeconfig로 접속한다                      |
| [03](/blog/2026/k8s-03-architecture/)   | 아키텍처        | Control Plane(결정)과 Node(실행), 모든 통신은 kube-apiserver를 거친다                  |
| [04](/blog/2026/k8s-04-pod/)            | Pod             | 배포의 최소 단위. YAML 선언이 스케줄링을 거쳐 컨테이너가 되기까지                      |
| [05](/blog/2026/k8s-05-workloads/)      | 워크로드        | Deployment/StatefulSet/DaemonSet — Pod를 상황에 맞게 관리하는 상위 리소스              |
| [06](/blog/2026/k8s-06-networking/)     | 네트워킹        | 바뀌는 Pod IP 대신 Service로 고정 접점을, Ingress로 L7 라우팅을                        |
| [07](/blog/2026/k8s-07-storage-config/) | 스토리지와 설정 | PV/PVC/StorageClass로 데이터를, ConfigMap/Secret으로 설정을 분리한다                   |
| [08](/blog/2026/k8s-08-rbac/)           | 권한 관리       | 누가(SA) 무엇을(Role) 할 수 있는지 — 최소 권한 원칙의 RBAC                             |
| 09 (이 글)                              | 확장과 생태계   | Operator로 확장하고, Helm/Kustomize로 YAML을 패키징하고, CNCF 성숙도로 생태계를 읽는다 |

관통하는 원리는 하나였다. **원하는 상태를 선언하면 컨트롤러가 차이를 메운다.** Deployment도, Service의 Endpoints도, cert-manager의 인증서 갱신도 전부 이 reconcile 루프의 변주다. 이 원리 하나를 몸에 익혔다면 처음 보는 리소스를 만나도 "spec이 뭐고, 어느 컨트롤러가 reconcile하는가"부터 물으면 된다.

다음 단계로는 이런 방향을 추천한다.

- **남에게 설명해보기**: 이 시리즈의 로드맵 원글(seungkyua, SK Devocean)은 "남에게 설명할 수 있을 정도로" 공부하라고 조언한다. 각 편의 개념을 자기 말로 설명해보고, 막히는 곳이 다음 공부 지점이다.
- **자격증으로 검증**: CKA(Certified Kubernetes Administrator) 같은 실기 시험은 kubectl 실습 위주라 이 시리즈의 연장선에서 준비하기 좋다. 2025년 개편 커리큘럼에는 6절에서 본 Helm·Kustomize도 포함됐다.
- **생태계 탐색**: [CNCF landscape](https://landscape.cncf.io/)에서 관심 분야(모니터링, CI/CD, 서비스 메시 등)의 프로젝트를 골라 kind 클러스터에 직접 설치해보자. 7절의 성숙도 지도가 나침반이 된다.
- **Operator 직접 만들기**: Go의 kubebuilder 같은 프레임워크로 간단한 CRD와 컨트롤러를 직접 작성해보면 reconcile 루프가 완전히 체화된다.
- **운영 중인 클러스터 들여다보기**: 배포까지 됐다면 다음은 "지금 잘 돌고 있나"를 보는 관측성이다. 이 시리즈의 확장편 [10: 관측성 — 로그·메트릭과 Prometheus/Grafana](/blog/2026/k8s-10-observability/)에서 kubectl 내장 도구부터 Prometheus·Grafana 스택까지 이어서 다룬다.

> 이 글로 K8s 입문 시리즈의 본편(01–09)을 마친다. 이전 글은 [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)이고, 처음부터 다시 보려면 [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)에서 시작하면 된다. 실무 진입을 위한 관측성은 확장편 [10: 관측성 — 로그·메트릭과 Prometheus/Grafana](/blog/2026/k8s-10-observability/)로 이어진다. 완주를 축하한다.

# 참고 문헌

- [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua, SK Devocean, 2024) — 시리즈 로드맵 출처
- [Operator pattern — Kubernetes 공식 문서](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Custom Resources — Kubernetes 공식 문서](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
- [Introducing Operators: Putting Operational Knowledge into Software](https://www.redhat.com/en/blog/introducing-operators-putting-operational-knowledge-into-software) (CoreOS, 2016 — 현재 Red Hat 블로그에 보존)
- [CNCF Projects — 성숙도 단계와 프로젝트 목록](https://www.cncf.io/projects/)
- [CNCF Graduation Criteria (TOC)](https://github.com/cncf/toc/blob/main/process/graduation_criteria.md)
- [Cloud Native Computing Foundation Announces cert-manager Graduation](https://www.cncf.io/announcements/2024/11/12/cloud-native-computing-foundation-announces-cert-manager-graduation/) (CNCF, 2024)
- [cert-manager Installation — 공식 문서](https://cert-manager.io/docs/installation/)
- [cert-manager SelfSigned Issuer — 공식 문서](https://cert-manager.io/docs/configuration/selfsigned/)
- [Using Helm — Helm 공식 문서](https://helm.sh/docs/intro/using_helm/)
- [Declarative Management of Kubernetes Objects Using Kustomize — Kubernetes 공식 문서](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/)
- [Certified Kubernetes Administrator (CKA) Program Changes — Linux Foundation](https://training.linuxfoundation.org/certified-kubernetes-administrator-cka-program-changes/)
