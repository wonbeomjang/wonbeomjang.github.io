---
layout: post
title: "K8s 시리즈 05: Amazon EKS — 아키텍처와 Worker Node"
date: 2026-04-14 14:00:00 +0900
description: "EKS 아키텍처, Worker Node 옵션, VPC CNI, Pod Identity, Auto Mode, 비용 구조, 업그레이드 전략 — 실무 중심 정리"
categories: [infra]
tags: [kubernetes, eks, aws, devops, container]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 시리즈**의 다섯 번째 글이다. Kubernetes 기초 개념(Pod, Deployment, Service 등)은 이전 글을 참고하자.
>
> - [01: Kubernetes란? 컨테이너부터 클러스터까지](/blog/2026/k8s-01-intro/)
> - [02: Pod, Deployment, Job, CronJob — K8s 워크로드 총정리](/blog/2026/k8s-02-workloads/)
> - [03: Service, Ingress — 트래픽 라우팅과 외부 접근](/blog/2026/k8s-03-networking/)
> - [04: ConfigMap, Secret, Storage — 설정과 데이터 관리](/blog/2026/k8s-04-config-storage/)
> - **05: Amazon EKS — 아키텍처와 Worker Node** ← 현재 글
> - [06: EKS 네트워킹·보안·비용·운영](/blog/2026/k8s-06-eks-operations/)

직접 Kubernetes를 구축하면 Control Plane 설치, etcd 백업, 인증서 갱신, 버전 업그레이드 등 클러스터 관리에만 상당한 운영 부담이 생긴다. **AWS EKS**(Elastic Kubernetes Service)는 Control Plane 관리를 AWS에 위임하고, **Rancher** 같은 GUI 관리 도구를 함께 사용하면 kubectl 명령어를 몰라도 Pod 배포, 로그 확인, 스케일링 등 대부분의 작업을 웹 브라우저에서 수행할 수 있다. 즉 **EKS가 인프라 운영을, Rancher가 일상 운영을 덜어주는** 조합이다.

이 글에서는 EKS의 아키텍처부터 네트워킹, 보안, 최신 기능(Auto Mode, Pod Identity), 비용, 업그레이드 전략까지 실무에서 필요한 핵심을 정리한다.

---

# 1. EKS vs 직접 구축

| 항목              | 직접 구축                 | EKS                                                      |
| ----------------- | ------------------------- | -------------------------------------------------------- |
| **Control Plane** | 직접 설치·운영·업그레이드 | **AWS가 관리** (3 AZ 고가용성)                           |
| **etcd**          | 직접 백업·복구            | **AWS가 관리** (3 인스턴스 분산)                         |
| **API Server**    | 직접 HA 구성              | **최소 2개 인스턴스 자동 운영**                          |
| **업그레이드**    | 수동 (위험도 높음)        | 콘솔/API 한 번에 실행                                    |
| **AWS 통합**      | 수동 연동                 | IAM, VPC, ELB, EFS 등 **네이티브 통합**                  |
| **Worker Node**   | 직접 관리                 | 직접 관리, Managed Node Group, 또는 완전 자동(Auto Mode) |
| **비용**          | EC2 + 운영 인건비         | **$0.10/hr** (~$73/월) + EC2                             |

핵심은 **Control Plane 관리를 AWS에 맡긴다**는 것이다. API Server, etcd, Scheduler를 직접 운영할 때 발생하는 장애 대응, 인증서 관리, 버전 호환성 문제를 신경 쓰지 않아도 된다.

---

# 2. EKS 아키텍처

EKS 클러스터는 **AWS 관리 영역(Control Plane)**과 **사용자 관리 영역(Data Plane)**으로 나뉜다.

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/eks-architecture.png" class="img-fluid rounded z-depth-1" %}

## 2.1 Control Plane 상세

| 구성 요소              | 역할                                                  | 배치                    |
| ---------------------- | ----------------------------------------------------- | ----------------------- |
| **API Server**         | 모든 요청의 진입점. kubectl, Rancher 등이 여기에 요청 | 최소 2개, 3 AZ 분산     |
| **etcd**               | 클러스터 상태를 저장하는 분산 Key-Value DB            | 3개 인스턴스, 3 AZ 분산 |
| **Scheduler**          | 새 Pod을 어느 노드에 배치할지 결정                    | Control Plane 내부      |
| **Controller Manager** | Deployment, ReplicaSet 등 컨트롤러 루프 실행          | Control Plane 내부      |

Control Plane은 **클러스터별로 완전히 격리**되어 있다. 다른 클러스터나 AWS 계정과 공유하지 않는다. AWS 관리 VPC에 위치하며, **ENI(Elastic Network Interface)**를 통해 사용자 VPC와 연결된다.

## 2.2 API Server 엔드포인트

| 모드                   | 접근 범위            | 권장 상황                 |
| ---------------------- | -------------------- | ------------------------- |
| **Public**             | 인터넷에서 접근 가능 | 개발/테스트               |
| **Public + CIDR 제한** | 허용된 IP만 접근     | 일반 운영                 |
| **Private**            | VPC 내부에서만 접근  | **보안 중시 환경 (권장)** |

프로덕션에서는 **Private 엔드포인트** 또는 **Public + CIDR 제한**을 사용하는 것이 권장된다.

---

# 3. Worker Node 옵션

EKS는 6가지 Worker Node 옵션을 제공한다. 관리 수준에 따라 선택하면 된다.

| 옵션                      | 관리 수준        | 적합한 경우              |
| ------------------------- | ---------------- | ------------------------ |
| **EKS Auto Mode**         | AWS 완전 관리    | 운영 오버헤드 최소화     |
| **Fargate**               | 서버리스         | 인프라 관리 불필요 환경  |
| **Karpenter (자체 설치)** | 반자동           | 세밀한 스케일링 제어     |
| **Managed Node Group**    | AWS 부분 관리    | 자동화와 제어의 균형     |
| **Self-Managed Node**     | 사용자 완전 관리 | 완전한 커스터마이징 필요 |
| **Hybrid Nodes**          | 하이브리드       | 온프레미스 + 클라우드    |

## 3.1 Managed Node Group

가장 보편적인 선택이다. AWS가 EC2 인스턴스의 **프로비저닝, 패칭, 업데이트**를 관리한다.

- Auto Scaling Group 기반으로 노드 수를 자동 조절
- Launch Template으로 인스턴스 타입, AMI, userdata 등 커스터마이징 가능
- **업데이트 시 자동 드레인**: 노드 업데이트 시 기존 Pod을 다른 노드로 이동 후 교체

```yaml
# Managed Node Group 예시 (eksctl)
managedNodeGroups:
  - name: general
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    labels:
      role: general
```

## 3.2 Fargate

Pod 단위의 **서버리스** 실행이다. 노드를 관리할 필요가 전혀 없다.

| 특징        | 설명                                    |
| ----------- | --------------------------------------- |
| 과금        | vCPU-초 + Memory-초 (최소 1분)          |
| 스토리지    | 20GB ephemeral (무료)                   |
| 제한        | DaemonSet 실행 불가, GPU 미지원         |
| 적합한 용도 | 간헐적 배치 작업, 가벼운 마이크로서비스 |

**Fargate Profile**로 어떤 Pod이 Fargate에서 실행될지 지정한다 (Namespace + Label 조합).

## 3.3 EKS Auto Mode (2024.12 GA)

2024년 re:Invent에서 발표된 **가장 큰 변화**다. 컴퓨팅, 스토리지, 네트워킹을 포함한 클러스터 인프라 관리를 자동화한다.

내부적으로 **Karpenter가 EKS Control Plane에 통합**된 것이다. 사용자가 Karpenter를 직접 설치·관리할 필요 없이 동일한 기능을 사용할 수 있다.

| 기능                | 설명                                                                |
| ------------------- | ------------------------------------------------------------------- |
| 컴퓨팅 오토스케일링 | Spot + On-Demand 자동 혼합, 워크로드에 맞는 인스턴스 타입 자동 선택 |
| 내장 애드온         | VPC CNI, EBS CSI, CoreDNS, kube-proxy 등 자동 관리                  |
| 보안                | Bottlerocket AMI 기반, SELinux, 읽기전용 루트 파일시스템            |
| 자동 업데이트       | PodDisruptionBudget을 존중하며 노드 자동 업데이트                   |
| GPU 지원            | GPU 플러그인 자동 설치                                              |

**2025년 추가된 기능:**

- EC2 On-Demand Capacity Reservations (ODCR) 및 Capacity Blocks for ML 지원
- 별도 Pod 서브넷 지원으로 인프라/앱 트래픽 분리
- AWS KMS 암호화 (ephemeral + root 볼륨)
- Forward Proxy 지원

### Auto Mode vs Managed Node Group

| 항목               | Auto Mode                   | Managed Node Group               |
| ------------------ | --------------------------- | -------------------------------- |
| 인스턴스 타입 선택 | **자동** (워크로드 기반)    | 수동 지정                        |
| 스케일링           | **즉시** (Karpenter 기반)   | Cluster Autoscaler (느림)        |
| AMI 관리           | **자동** (Bottlerocket)     | Amazon Linux 2/Bottlerocket 선택 |
| 애드온 관리        | **내장**                    | 별도 설치·관리                   |
| 커스터마이징       | 제한적 (NodePool/NodeClass) | Launch Template으로 자유로움     |
| 비용               | EC2 + 관리 수수료           | EC2만                            |

**선택 기준**: 운영 간소화를 원하면 Auto Mode, 세밀한 제어가 필요하면 Managed Node Group.

## 3.4 Hybrid Nodes (2024.12 GA)

온프레미스 서버를 EKS 클러스터의 Worker Node로 등록할 수 있다.

- Control Plane은 AWS, Worker Node는 온프레미스
- AWS Site-to-Site VPN, Direct Connect, 또는 자체 VPN으로 연결
- `nodeadm` CLI로 각 호스트를 클러스터에 조인
- CNI는 **Cilium 또는 Calico** 사용 (VPC CNI가 아님)
- 네트워크 요구사항: **최소 100Mbps, RTT 200ms 이하** 권장
- 연결이 불안정한 DDIL 환경에는 적합하지 않음 → EKS Anywhere 고려

---

# 4. EKS 네트워킹

## 4.1 VPC CNI

EKS의 기본 CNI(Container Network Interface)로, Pod에 **VPC의 실제 IP 주소**를 할당한다. 오버레이 네트워크 없이 VPC 내에서 직접 통신하므로 **성능 오버헤드가 거의 없다**.

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/vpc-cni.png" class="img-fluid rounded z-depth-1" %}

### 노드당 최대 Pod 수

인스턴스 타입에 따라 ENI 수와 ENI당 IP 수가 다르다.

$$
\text{Max Pods} = (\text{ENI 수} \times \text{ENI당 IP 수} - 1) + 2
$$

| 인스턴스   | ENI 수 | ENI당 IP | **Max Pods** |
| ---------- | ------ | -------- | ------------ |
| t3.medium  | 3      | 6        | 17           |
| m5.large   | 3      | 10       | 29           |
| m5.xlarge  | 4      | 15       | 58           |
| c5.2xlarge | 4      | 15       | 58           |

t3.medium에서 17개밖에 못 띄우는 점을 간과하면, Pod이 Pending 상태에 빠지는 문제가 발생한다. 실제 워크로드에 맞는 인스턴스 타입을 선택하는 것이 중요하다.

## 4.2 IP 고갈 해결

VPC CNI는 Pod마다 실제 VPC IP를 사용하므로, **서브넷 IP가 부족해질 수 있다**. 해결 방법:

| 방법                  | 설명                                         | 효과                        |
| --------------------- | -------------------------------------------- | --------------------------- |
| **Prefix Delegation** | /28 프리픽스 할당으로 노드당 IP 대폭 증가    | 노드당 Pod 수 **~4배 증가** |
| **Custom Networking** | 별도 서브넷 CIDR에서 Pod IP 할당             | Node/Pod IP 대역 분리       |
| **Secondary CIDR**    | VPC에 추가 CIDR 블록 연결 (100.64.0.0/10 등) | IP 공간 확장                |
| **IPv6**              | 거의 무한한 IP 공간                          | 근본적 해결                 |

**Prefix Delegation**이 가장 간단하고 효과적이다. `ENABLE_PREFIX_DELEGATION=true` 설정만으로 노드당 Pod 밀도를 크게 높일 수 있다.

## 4.3 Warm Pool 튜닝

VPC CNI의 `ipamd`는 미리 IP를 확보해두는 Warm Pool을 유지한다.

| 환경변수            | 기본값 | 설명                                             |
| ------------------- | ------ | ------------------------------------------------ |
| `WARM_ENI_TARGET`   | 1      | 미리 확보할 ENI 수                               |
| `WARM_IP_TARGET`    | -      | 미리 확보할 IP 수 (설정 시 WARM_ENI_TARGET 무시) |
| `MINIMUM_IP_TARGET` | -      | 최소 유지 IP 수                                  |

Pod 생성이 빈번한 환경에서는 `WARM_IP_TARGET`을 적절히 설정하면 **Pod 시작 지연을 줄일 수 있다**. 반대로 IP를 아끼려면 `WARM_ENI_TARGET=0` + `MINIMUM_IP_TARGET`을 낮게 설정한다.

## 4.4 Network Policy

VPC CNI **v1.14+**에서 Kubernetes NetworkPolicy API를 네이티브로 지원한다. 별도의 Calico 설치 없이 Pod 간 네트워크 격리가 가능하다.

---

# 5. 보안

## 5.1 Pod에서 AWS 서비스 접근: IRSA vs Pod Identity

Pod이 S3, ECR 등 AWS 서비스에 접근해야 할 때, EC2 인스턴스 전체에 권한을 주면 **그 노드의 모든 Pod이 같은 권한을 갖게 된다**. 이를 해결하기 위해 Pod(ServiceAccount) 단위로 최소 권한을 부여한다.

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/pod-identity.png" class="img-fluid rounded z-depth-1" %}

### IRSA (IAM Roles for Service Accounts)

기존 방식이며 여전히 널리 사용된다.

1. 클러스터별 **OIDC Provider** 설정 필요
2. IAM Role에 OIDC Provider를 Trust Relationship으로 설정
3. ServiceAccount에 IAM Role ARN을 annotation으로 지정
4. Pod에서 `sts:AssumeRoleWithWebIdentity`로 임시 자격증명 교환

### Pod Identity (2023.12~, 권장)

IRSA의 후속으로, **더 간단하고 보안이 강화**된 방식이다.

| 비교 항목          | Pod Identity                 | IRSA                      |
| ------------------ | ---------------------------- | ------------------------- |
| OIDC Provider 설정 | **불필요**                   | 필요 (클러스터별)         |
| 설정 방법          | EKS API에서 직접 매핑        | IAM Trust + SA annotation |
| ABAC 세션 태그     | **지원** (namespace, SA 등)  | 미지원                    |
| 교차 계정          | 간접 (역할 체이닝)           | 직접                      |
| STS 할당량         | **미사용**                   | 사용                      |
| 추가 요구사항      | Pod Identity Agent DaemonSet | 없음                      |

**새로 구성한다면 Pod Identity를 사용하는 것이 권장**된다. OIDC Provider 설정이 불필요하고, ABAC 세션 태그를 통해 `kubernetes-namespace`, `kubernetes-service-account` 등으로 세밀한 접근 제어가 가능하다.

## 5.2 Cluster Access 관리

기존에는 `aws-auth` ConfigMap으로 IAM ↔ K8s RBAC을 매핑했다. 이 방식은 ConfigMap을 직접 편집해야 해서 **실수로 잠김(lockout) 위험**이 있었다.

**EKS Access Entry** (v1.23+)는 이를 대체하는 API 기반 접근 관리 방식이다.

| 모드                 | 설명                                   |
| -------------------- | -------------------------------------- |
| `API`                | EKS Access Entry API만 사용 (**권장**) |
| `API_AND_CONFIG_MAP` | 마이그레이션 기간용                    |
| `CONFIG_MAP`         | 레거시 (aws-auth)                      |

## 5.3 보안 모범 사례

- API Server 엔드포인트를 **Private**으로 설정하거나 CIDR 제한
- 클러스터 생성자의 `cluster-admin` 권한 제거 (`bootstrapClusterCreatorAdminPermissions=false`)
- **IMDSv2만 허용**, hop limit=1 → Pod가 노드 IAM Role 상속 방지
- 비root 사용자로 컨테이너 실행 (`runAsNonRoot: true`)
- K8s API 접근 불필요한 Pod은 `automountServiceAccountToken: false`

---

# 6. 비용 구조

## 6.1 기본 비용

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/cost-formula.png" class="img-fluid rounded z-depth-1" %}

### Control Plane

| 구분                               | 시간당 | 월간 (730h) |
| ---------------------------------- | ------ | ----------- |
| **Standard Support** (14개월)      | $0.10  | ~$73        |
| **Extended Support** (추가 12개월) | $0.60  | ~$438       |

Extended Support에 진입하면 **비용이 6배**로 뛴다. 버전 업그레이드를 미루면 비용으로 돌아온다.

### Provisioned Control Plane (대규모 워크로드용)

2025년 11월에 추가된 옵션으로, Control Plane 용량을 사전 할당할 수 있다.

| 티어     | 시간당 | Max API Concurrency | Pod Scheduling Rate |
| -------- | ------ | ------------------- | ------------------- |
| Standard | $0.10  | 기본                | 기본                |
| XL       | $1.65  | 1,700               | 167/sec             |
| 2XL      | $3.40  | 3,400               | 283/sec             |
| 4XL      | $6.90  | 6,800               | 400/sec             |
| 8XL      | $13.90 | -                   | -                   |

대규모 AI 학습/추론, 고빈도 배포, Spark 잡 같은 대량 Pod 배치에 유용하다. 티어 간 전환은 **API 한 번 호출로 가능하며 다운타임이 없다**.

### Worker Node

인스턴스 비용은 일반 EC2와 동일하다.

| 옵션          | 비용                                      |
| ------------- | ----------------------------------------- |
| EC2 On-Demand | 인스턴스 타입별 (예: m5.xlarge $0.192/hr) |
| EC2 Spot      | On-Demand 대비 **60-90% 할인**            |
| Savings Plans | 최대 **72% 할인** (1/3년 약정)            |
| Fargate       | vCPU $0.04048/hr + Memory $0.004445/hr/GB |
| Auto Mode     | EC2 비용 + 관리 수수료                    |

### Fargate 비용 예시

| 구성              | 시간당 | 월간 (730h) |
| ----------------- | ------ | ----------- |
| 0.25 vCPU + 0.5GB | $0.012 | ~$9         |
| 1 vCPU + 2GB      | $0.049 | ~$36        |
| 2 vCPU + 4GB      | $0.099 | ~$72        |
| 4 vCPU + 8GB      | $0.197 | ~$144       |

## 6.2 숨은 비용

EKS 청구서에는 나타나지 않지만, 실제로 상당한 비용을 차지하는 항목들이 있다.

| 항목                | 비용                  | 비고                   |
| ------------------- | --------------------- | ---------------------- |
| **NAT Gateway**     | $0.045/hr + $0.045/GB | Pod에서 인터넷 접근 시 |
| **Cross-AZ 트래픽** | $0.01/GB (양방향)     | Pod 간 AZ가 다를 때    |
| **ALB**             | $0.0225/hr + LCU 비용 | Ingress 사용 시        |
| **EBS (gp3)**       | $0.08/GB/월           | PVC 사용 시            |
| **CloudWatch Logs** | $0.50/GB 수집         | 로그 모니터링          |
| **VPC Endpoint**    | ~$14.40/월 (단일 AZ)  | ECR Private 접근 등    |

특히 **NAT Gateway**가 의외로 크다. 100개 Pod이 각각 매일 1GB씩 외부 통신하면 NAT만으로 **월 ~$230**이 나온다.

## 6.3 EKS vs GKE vs AKS

| 항목            | EKS             | GKE                           | AKS                      |
| --------------- | --------------- | ----------------------------- | ------------------------ |
| Control Plane   | $0.10/hr        | 첫 zonal 무료, 이후 $0.10/hr  | **무료**                 |
| Node 업그레이드 | 수동            | **자동**                      | **자동**                 |
| K8s 신버전 적용 | 4-8주           | **2주 이내**                  | 3-6주                    |
| Cross-AZ 트래픽 | 유료            | 유료                          | **무료**                 |
| LB 시간당       | $0.025          | $0.025                        | **$0.005**               |
| **강점**        | AWS 생태계 통합 | 가장 빠른 K8s 지원, Autopilot | 무료 Control Plane, 저렴 |

**EKS를 선택하는 이유**: 이미 AWS 생태계(ECR, IAM, VPC, EFS, S3 등)를 사용하고 있다면, 다른 서비스와의 네이티브 통합이 가장 큰 장점이다.

---

# 7. 버전 관리와 업그레이드

## 7.1 EKS 버전 라이프사이클

EKS는 Kubernetes 버전을 **Standard Support 14개월 + Extended Support 12개월 = 총 26개월** 지원한다.

| 버전 | Standard 종료 | Extended 종료 | 비용                    |
| ---- | ------------- | ------------- | ----------------------- |
| 1.35 | 2027.03       | 2028.03       | $0.10/hr                |
| 1.34 | 2026.12       | 2027.12       | $0.10/hr                |
| 1.33 | 2026.07       | 2027.07       | $0.10/hr                |
| 1.32 | 2026.03       | 2027.03       | **$0.60/hr** (Extended) |
| 1.31 | 2025.11       | 2026.11       | **$0.60/hr** (Extended) |

**Extended Support에 진입하면 비용이 6배**이므로, Standard 기간 내에 업그레이드하는 것이 중요하다.

## 7.2 업그레이드 순서

EKS는 **한 번에 1 마이너 버전만** 업그레이드할 수 있다 (예: 1.29 → 1.30 → 1.31).

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/upgrade-flow.png" class="img-fluid rounded z-depth-1" %}

## 7.3 버전 스큐 정책

| K8s 버전 | Control Plane과 Node 간 최대 차이                   |
| -------- | --------------------------------------------------- |
| 1.28+    | **n-3** (Control Plane 1.31이면 Node 1.28까지 허용) |
| < 1.28   | n-2                                                 |

## 7.4 업그레이드 모범 사례

- **연 1-2회 업그레이드 주기 유지**: Extended Support 진입 전에 실행
- **Staging 먼저**: 프로덕션 적용 전 반드시 검증
- **PodDisruptionBudget 설정**: 업그레이드 시 서비스 가용성 보장
- **서브넷 IP 여유**: Control Plane 업그레이드 시 **최소 5개 IP** 필요
- **EKS 애드온은 자동 업그레이드 안 됨**: 수동으로 호환성 확인 후 업데이트
- **VPC CNI도 1 마이너 버전씩** 업그레이드

---

# 8. EKS와 연동되는 AWS 서비스

| AWS 서비스     | K8s에서의 역할           | 연동 방식                                     |
| -------------- | ------------------------ | --------------------------------------------- |
| **ECR**        | 컨테이너 이미지 저장소   | Deployment의 image 주소로 사용                |
| **EFS**        | 공유 파일 스토리지 (RWX) | StorageClass → PVC → Pod 마운트               |
| **EBS**        | 블록 스토리지 (RWO)      | StorageClass → PVC → Pod 마운트               |
| **NLB/ALB**    | 외부 트래픽 진입점       | Ingress 또는 LoadBalancer Service가 자동 생성 |
| **IAM**        | Pod별 AWS 권한 관리      | Pod Identity 또는 IRSA                        |
| **Route53**    | 도메인 DNS 관리          | Ingress 도메인의 DNS 레코드 등록              |
| **CloudWatch** | 로그/메트릭 수집         | Fluent Bit 등 로그 에이전트가 Pod 로그 전송   |

---

# 9. 전체 배포 흐름 한눈에 보기

{% include figure.liquid loading="lazy" path="assets/post/image/amazon-eks-guide/deploy-flow.png" class="img-fluid rounded z-depth-1" %}

---

# 참고 문헌

- [Amazon EKS 공식 문서](https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html)
- [Amazon EKS Architecture](https://docs.aws.amazon.com/eks/latest/userguide/eks-architecture.html)
- [Amazon EKS Pricing](https://aws.amazon.com/eks/pricing/)
- [EKS Auto Mode 발표 블로그](https://aws.amazon.com/blogs/containers/new-amazon-eks-auto-mode-features-for-enhanced-security-network-control-and-performance/)
- [EKS Auto Mode and NodePools Explained](https://medium.com/@ananthchaitanya17/eks-auto-mode-and-nodepools-explained-whats-new-in-2025-257602b663b3)
- [EKS Hybrid Nodes Deep Dive](https://aws.amazon.com/blogs/containers/a-deep-dive-into-amazon-eks-hybrid-nodes/)
- [EKS Pod Identity 공식 문서](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html)
- [IRSA vs Pod Identity 비교](https://www.kubeblogs.com/eks-irsa-vs-eks-pod-identity-two-ways-to-grant-application-access-to-aws-resources/)
- [VPC CNI Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/vpc-cni.html)
- [EKS IAM Best Practices](https://aws.github.io/aws-eks-best-practices/security/docs/iam/)
- [EKS Upgrade Best Practices](https://aws.github.io/aws-eks-best-practices/upgrades/)
- [EKS Version Lifecycle](https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html)
- [EKS Provisioned Control Plane](https://aws.amazon.com/blogs/containers/amazon-eks-introduces-provisioned-control-plane/)
- [EKS vs GKE vs AKS 2026 비교](https://sedai.io/blog/kubernetes-cost-eks-vs-aks-vs-gke)
- [AWS Fargate Pricing](https://aws.amazon.com/fargate/pricing/)
