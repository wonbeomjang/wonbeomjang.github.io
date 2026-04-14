---
layout: post
title: "K8s 시리즈 04: ConfigMap, Secret, Storage — 설정과 데이터 관리"
date: 2026-04-14 11:45:00 +0900
description: "ConfigMap/Secret 주입, PV/PVC/StorageClass, EFS vs EBS, Namespace/Label, Helm 패키지 관리"
categories: [infra]
tags: [kubernetes, configmap, secret, storage, helm, devops]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 시리즈**의 네 번째 글이다.
>
> - [01: Kubernetes란? 컨테이너부터 클러스터까지](/blog/2026/k8s-01-intro/)
> - [02: Pod, Deployment, Job, CronJob — K8s 워크로드 총정리](/blog/2026/k8s-02-workloads/)
> - [03: Service, Ingress — 트래픽 라우팅과 외부 접근](/blog/2026/k8s-03-networking/)
> - **04: ConfigMap, Secret, Storage — 설정과 데이터 관리** ← 현재 글
> - [05: Amazon EKS — 아키텍처와 Worker Node](/blog/2026/k8s-05-amazon-eks/)
> - [06: EKS 네트워킹·보안·비용·운영](/blog/2026/k8s-06-eks-operations/)

앱이 실행되려면 코드만으로는 부족하다. DB 주소, API Key, 모델 체크포인트 경로 같은 **설정**과 **데이터**가 필요하다. 이번 글에서는 설정을 주입하는 ConfigMap/Secret, 데이터를 영구 저장하는 PV/PVC, 그리고 리소스를 정리하는 Namespace/Label을 다룬다.

---

# 1. ConfigMap과 Secret

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-04-config-storage/config-inject.png" class="img-fluid rounded z-depth-1" alt="ConfigMap과 Secret을 Pod에 주입하는 방식" %}

## 1.1 ConfigMap — 일반 설정

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DB_HOST: "postgres.default.svc"
  DB_PORT: "5432"
  LOG_LEVEL: "info"
  MODEL_NAME: "llama-3-70b"
```

### Pod에 환경변수로 주입

```yaml
spec:
  containers:
    - name: app
      image: my-app:v1
      envFrom:
        - configMapRef:
            name: app-config # ConfigMap 전체를 환경변수로
```

### Pod에 파일로 마운트

```yaml
spec:
  containers:
    - name: app
      volumeMounts:
        - name: config-volume
          mountPath: /config
  volumes:
    - name: config-volume
      configMap:
        name: app-config # /config/DB_HOST, /config/DB_PORT 등 파일 생성
```

## 1.2 Secret — 민감 정보

Secret은 ConfigMap과 사용 방법이 동일하지만, **민감 정보**를 저장한다.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData: # stringData를 쓰면 Base64 인코딩 자동 처리
  DB_PASSWORD: "super-secret-password"
  API_KEY: "sk-abc123..."
```

```yaml
# Pod에서 사용
spec:
  containers:
    - name: app
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: DB_PASSWORD
```

> **주의:** Secret은 Base64 **인코딩**일 뿐 **암호화가 아니다.** `echo "c3VwZXItc2VjcmV0" | base64 -d`로 누구나 디코딩할 수 있다. RBAC으로 접근을 제한하고, EKS에서는 etcd 암호화를 활성화하자.

| 비교      | ConfigMap                    | Secret                       |
| --------- | ---------------------------- | ---------------------------- |
| 용도      | DB 호스트, 포트, 기능 플래그 | DB 비밀번호, API Key, 인증서 |
| 저장 방식 | 평문                         | Base64 인코딩                |
| 크기 제한 | 1MB                          | 1MB                          |
| 주입 방식 | 환경변수 또는 파일 마운트    | 동일                         |

---

# 2. Storage — 영구 데이터 저장

Pod은 기본적으로 **임시 저장소**를 사용한다. Pod이 재시작되면 데이터가 사라진다. 모델 체크포인트, 업로드 파일, DB 데이터처럼 영구 보존이 필요하면 PV/PVC를 사용한다.

## 2.1 PV, PVC, StorageClass

| 개념                            | 설명                                        | 비유                  |
| ------------------------------- | ------------------------------------------- | --------------------- |
| **PV (PersistentVolume)**       | 실제 스토리지를 K8s에 등록한 리소스         | 창고 등록 카드        |
| **PVC (PersistentVolumeClaim)** | "이런 스토리지가 필요하다"는 요청           | 창고 사용 신청서      |
| **StorageClass**                | PVC가 만들어질 때 PV를 자동 생성하는 템플릿 | 창고 자동 배정 시스템 |

### 동작 흐름 (동적 프로비저닝)

```
PVC 생성 (storageClassName: efs)
  → StorageClass가 자동으로 PV 생성
    → Pod에서 PVC를 마운트하여 사용
```

## 2.2 스토리지 타입 비교

| 타입         | 영속성           | 다중 Pod 공유            | 용도                       | AWS 서비스 |
| ------------ | ---------------- | ------------------------ | -------------------------- | ---------- |
| **EmptyDir** | Pod 삭제 시 소멸 | 같은 Pod 내 컨테이너만   | 임시 파일, 캐시            | -          |
| **EBS**      | 영구             | **불가** (RWO, 단일 Pod) | DB, 단일 인스턴스 앱       | EBS gp3    |
| **EFS**      | 영구             | **가능** (RWX, 다중 Pod) | 공유 파일, 모델 체크포인트 | EFS        |

### LLM 엔지니어를 위한 선택 가이드

- **모델 체크포인트 공유** (여러 추론 서버가 같은 모델 사용) → **EFS (RWX)**
- **DB 데이터** (PostgreSQL, Redis) → **EBS (RWO)**
- **학습 중 임시 캐시** → **EmptyDir**

## 2.3 PVC YAML

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
    - ReadWriteMany # EFS → 여러 Pod에서 읽기/쓰기 가능
  storageClassName: efs
  resources:
    requests:
      storage: 100Gi
```

### Pod에서 PVC 마운트

```yaml
spec:
  containers:
    - name: vllm
      volumeMounts:
        - name: models
          mountPath: /models # 컨테이너 내부 경로
  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: model-storage # 위에서 만든 PVC
```

---

# 3. Namespace와 Label

## 3.1 Namespace — 리소스 분리

Namespace는 클러스터 내 리소스를 **논리적으로 분리**하는 단위다.

```
클러스터
├── production     (운영 환경)
├── staging        (스테이징 환경)
├── dev            (개발 환경)
├── data-pipeline  (데이터 처리)
└── kube-system    (K8s 시스템)
```

| 용도            | 설명                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| **환경 분리**   | 같은 이름의 리소스도 Namespace가 다르면 공존 (`prod/backend`, `dev/backend`) |
| **권한 분리**   | RBAC으로 Namespace별 접근 권한 설정                                          |
| **리소스 할당** | ResourceQuota로 Namespace별 CPU/Memory 제한                                  |

```bash
# Namespace 생성
kubectl create namespace staging

# 특정 Namespace의 리소스 조회
kubectl get pods -n staging

# 모든 Namespace의 리소스 조회
kubectl get pods -A
```

## 3.2 Label과 Selector

Label은 리소스에 붙이는 **태그(키-값 쌍)**이고, Selector는 그 태그로 **리소스를 찾는 필터**다.

```yaml
# Pod의 Label
metadata:
  labels:
    app: backend
    env: production
    team: ml-platform
```

Service가 Pod을 찾고, Deployment가 Pod을 관리하는 핵심 메커니즘이 **Label ↔ Selector 매칭**이다. Label이 잘못되면 Service가 Pod을 못 찾거나, Deployment가 엉뚱한 Pod을 관리한다.

---

# 4. Helm — K8s 패키지 매니저

## 4.1 왜 Helm을 쓰는가?

실제 서비스를 배포하면 Deployment, Service, Ingress, ConfigMap, Secret, PVC 등 **여러 YAML 파일**이 필요하다. 이걸 하나씩 `kubectl apply`하는 것은 번거롭고 관리가 어렵다.

**Helm**은 이 YAML 묶음을 **Chart**라는 패키지로 관리한다.

| 비교        | kubectl apply   | Helm                             |
| ----------- | --------------- | -------------------------------- |
| 배포 단위   | YAML 파일 1개씩 | **Chart (여러 YAML 묶음)**       |
| 버전 관리   | 없음            | **릴리즈 이력 + 롤백**           |
| 설정 변경   | YAML 직접 수정  | `values.yaml`로 값만 변경        |
| 공개 패키지 | 없음            | **Artifact Hub** (수천 개 Chart) |

## 4.2 기본 사용법

```bash
# Chart 저장소 추가
helm repo add bitnami https://charts.bitnami.com/bitnami

# PostgreSQL 설치 (한 줄이면 DB가 뜬다)
helm install my-postgres bitnami/postgresql \
  --set auth.postgresPassword=mypassword \
  --set primary.persistence.size=50Gi

# 설치된 릴리즈 목록
helm list

# 업그레이드
helm upgrade my-postgres bitnami/postgresql --set primary.persistence.size=100Gi

# 롤백
helm rollback my-postgres 1

# 삭제
helm uninstall my-postgres
```

> StatefulSet(DB), DaemonSet(모니터링) 같은 복잡한 리소스는 직접 YAML을 쓰기보다 **Helm Chart로 설치**하는 것이 일반적이다.

---

# 참고 문헌

- [Kubernetes 공식 문서 — ConfigMap](https://kubernetes.io/ko/docs/concepts/configuration/configmap/)
- [Kubernetes 공식 문서 — Secret](https://kubernetes.io/ko/docs/concepts/configuration/secret/)
- [Kubernetes 공식 문서 — Persistent Volumes](https://kubernetes.io/ko/docs/concepts/storage/persistent-volumes/)
- [Kubernetes 공식 문서 — Namespace](https://kubernetes.io/ko/docs/concepts/overview/working-with-objects/namespaces/)
- [Kubernetes 공식 문서 — Label과 Selector](https://kubernetes.io/ko/docs/concepts/overview/working-with-objects/labels/)
- [Helm 공식 문서](https://helm.sh/ko/docs/)
