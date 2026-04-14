---
layout: post
title: "K8s 시리즈 06: EKS 네트워킹·보안·비용·운영"
date: 2026-04-14 14:30:00 +0900
description: "VPC CNI, Pod Identity vs IRSA, EKS 비용 구조와 숨은 비용, GKE/AKS 비교, 업그레이드 전략, Troubleshooting"
categories: [infra]
tags: [kubernetes, eks, aws, networking, security, devops]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 시리즈**의 마지막 글이다.
>
> - [01: Kubernetes란? 컨테이너부터 클러스터까지](/blog/2026/k8s-01-intro/)
> - [02: Pod, Deployment, Job, CronJob — K8s 워크로드 총정리](/blog/2026/k8s-02-workloads/)
> - [03: Service, Ingress — 트래픽 라우팅과 외부 접근](/blog/2026/k8s-03-networking/)
> - [04: ConfigMap, Secret, Storage — 설정과 데이터 관리](/blog/2026/k8s-04-config-storage/)
> - [05: Amazon EKS — 아키텍처와 Worker Node](/blog/2026/amazon-eks-guide/)
> - **06: EKS 네트워킹·보안·비용·운영** ← 현재 글

이전 글에서 EKS 아키텍처와 Worker Node 옵션을 다뤘다. 이번 글에서는 EKS 운영의 실전 — **VPC CNI 네트워킹, 보안(Pod Identity), 비용 구조, 업그레이드 전략, 자주 만나는 에러**를 정리한다.

---

# 1. VPC CNI — Pod 네트워킹

EKS의 기본 CNI로, Pod에 **VPC의 실제 IP 주소**를 할당한다. 오버레이 네트워크 없이 직접 통신하므로 성능 오버헤드가 거의 없다.

## 1.1 노드당 최대 Pod 수

인스턴스 타입에 따라 ENI 수와 ENI당 IP 수가 다르다.

$$
\text{Max Pods} = (\text{ENI 수} \times \text{ENI당 IP 수} - 1) + 2
$$

| 인스턴스     | ENI 수 | ENI당 IP | **Max Pods** |
| ------------ | ------ | -------- | ------------ |
| t3.medium    | 3      | 6        | 17           |
| m5.large     | 3      | 10       | 29           |
| m5.xlarge    | 4      | 15       | 58           |
| p4d.24xlarge | 15     | 50       | 737          |

**t3.medium에서 17개밖에 안 된다.** Pod이 Pending에 빠지는 흔한 원인이므로 인스턴스 선택 시 반드시 확인하자.

## 1.2 IP 고갈 해결

VPC CNI는 Pod마다 실제 VPC IP를 소비하므로, 서브넷 IP가 부족해질 수 있다.

| 방법                  | 설명                        | 효과                        |
| --------------------- | --------------------------- | --------------------------- |
| **Prefix Delegation** | /28 프리픽스 할당           | 노드당 Pod 수 **~4배 증가** |
| **Custom Networking** | 별도 서브넷에서 Pod IP 할당 | Node/Pod IP 대역 분리       |
| **Secondary CIDR**    | VPC에 추가 CIDR 연결        | IP 공간 확장                |

**Prefix Delegation**이 가장 간단하다. `ENABLE_PREFIX_DELEGATION=true` 하나로 해결된다.

---

# 2. 보안 — Pod Identity vs IRSA

Pod이 S3, ECR 등 AWS 서비스에 접근할 때, EC2 인스턴스 전체에 권한을 주면 그 노드의 **모든 Pod이 같은 권한**을 갖게 된다. Pod 단위로 최소 권한을 부여해야 한다.

## 2.1 IRSA vs Pod Identity

| 비교 항목          | Pod Identity (권장)          | IRSA (기존)               |
| ------------------ | ---------------------------- | ------------------------- |
| OIDC Provider 설정 | **불필요**                   | 필요 (클러스터별)         |
| 설정 방법          | EKS API에서 직접 매핑        | IAM Trust + SA annotation |
| ABAC 세션 태그     | **지원**                     | 미지원                    |
| STS 할당량         | **미사용**                   | 사용                      |
| 추가 요구사항      | Pod Identity Agent DaemonSet | 없음                      |

**새로 구성한다면 Pod Identity를 권장한다.**

## 2.2 보안 모범 사례

- API Server 엔드포인트를 **Private** 또는 **CIDR 제한**으로 설정
- **IMDSv2만 허용**, hop limit=1 → Pod가 노드 IAM Role 상속 방지
- 비root 사용자로 컨테이너 실행 (`runAsNonRoot: true`)
- K8s API 접근 불필요한 Pod은 `automountServiceAccountToken: false`

---

# 3. 비용 구조

## 3.1 Control Plane

| 구분                           | 시간당 | 월간 (~730h) |
| ------------------------------ | ------ | ------------ |
| **Standard Support** (14개월)  | $0.10  | ~$73         |
| **Extended Support** (+12개월) | $0.60  | ~$438        |

**Extended Support에 진입하면 비용 6배.** 버전 업그레이드를 미루면 비용으로 돌아온다.

## 3.2 숨은 비용

EKS 청구서에 안 나오지만 실제로 큰 항목들이다.

| 항목                | 비용                  | 비고              |
| ------------------- | --------------------- | ----------------- |
| **NAT Gateway**     | $0.045/hr + $0.045/GB | Pod 외부 통신 시  |
| **Cross-AZ 트래픽** | $0.01/GB (양방향)     | Pod 간 AZ 다를 때 |
| **ALB**             | $0.0225/hr + LCU      | Ingress 사용 시   |
| **EBS (gp3)**       | $0.08/GB/월           | PVC 사용 시       |
| **CloudWatch Logs** | $0.50/GB 수집         | 로그 모니터링     |

NAT Gateway가 의외로 크다. 100개 Pod이 매일 1GB씩 외부 통신하면 **월 ~$230**.

## 3.3 EKS vs GKE vs AKS

| 항목            | EKS             | GKE                | AKS      |
| --------------- | --------------- | ------------------ | -------- |
| Control Plane   | $0.10/hr        | 첫 zonal 무료      | **무료** |
| Node 업그레이드 | 수동            | **자동**           | **자동** |
| Cross-AZ 트래픽 | 유료            | 유료               | **무료** |
| **강점**        | AWS 생태계 통합 | 가장 빠른 K8s 지원 | 저렴     |

**EKS를 선택하는 이유**: AWS 생태계(ECR, IAM, VPC, EFS, S3)와의 네이티브 통합.

---

# 4. 업그레이드 전략

## 4.1 버전 라이프사이클

EKS는 K8s 버전을 **Standard 14개월 + Extended 12개월 = 총 26개월** 지원한다. **한 번에 1 마이너 버전만** 업그레이드 가능하다.

## 4.2 업그레이드 순서

1. **Deprecated API 스캔** — Cluster Insights, kube-no-trouble, Pluto
2. **백업** — Velero (Control Plane 업그레이드는 비가역적)
3. **Control Plane 업그레이드** — 콘솔/API에서 실행
4. **애드온 업데이트** — VPC CNI, CoreDNS, kube-proxy, EBS CSI
5. **Data Plane 업그레이드** — Managed: 롤링, Auto Mode: 자동
6. **Fargate 재시작** — `kubectl rollout restart`

## 4.3 모범 사례

- **연 1-2회 업그레이드**: Extended Support 진입 전에 실행
- **Staging 먼저**: 프로덕션 전 반드시 검증
- **PodDisruptionBudget 설정**: 업그레이드 중 서비스 가용성 보장
- **서브넷 IP 여유**: Control Plane 업그레이드 시 **최소 5개 IP** 필요

---

# 5. Troubleshooting — 자주 만나는 에러

## 5.1 Pod 상태별 대응

| 상태                 | 원인                                      | 확인 방법                                               | 해결                                      |
| -------------------- | ----------------------------------------- | ------------------------------------------------------- | ----------------------------------------- |
| **ImagePullBackOff** | 이미지 이름/태그 오류, Registry 인증 실패 | `kubectl describe pod` → Events                         | 이미지 경로·태그 확인, ECR 로그인 확인    |
| **CrashLoopBackOff** | 앱 시작 실패 (코드 에러, 설정 누락)       | `kubectl logs <pod>`                                    | 로그 확인 후 코드/설정 수정               |
| **Pending**          | 리소스 부족, Taint 미허용                 | `kubectl describe pod` → Events                         | 노드 추가, Toleration 추가, requests 조정 |
| **OOMKilled**        | 메모리 limits 초과                        | `kubectl describe pod` → Last State                     | memory limits 늘리기                      |
| **Evicted**          | 노드 디스크/메모리 부족                   | `kubectl get pods --field-selector=status.phase=Failed` | 노드 리소스 확인, Pod 정리                |

## 5.2 디버깅 명령어 모음

```bash
# 1. Pod 상태 확인
kubectl get pods -o wide                      # IP, 노드 확인
kubectl describe pod <pod-name>                # Events 섹션이 핵심

# 2. 로그 확인
kubectl logs <pod-name>                        # 현재 로그
kubectl logs <pod-name> --previous             # 이전 컨테이너 로그 (CrashLoop 시)
kubectl logs <pod-name> -c <container-name>    # 특정 컨테이너 로그

# 3. 쉘 접속
kubectl exec -it <pod-name> -- bash            # 컨테이너 내부 확인
kubectl exec -it <pod-name> -- env             # 환경변수 확인
kubectl exec -it <pod-name> -- cat /config/app.yaml  # 마운트된 설정 확인

# 4. 리소스 사용량
kubectl top pods                               # Pod별 CPU/Memory 사용량
kubectl top nodes                              # 노드별 리소스 사용량

# 5. 이벤트 확인
kubectl get events --sort-by=.lastTimestamp     # 최근 이벤트 (클러스터 전체)
kubectl get events -n <namespace>              # 특정 Namespace 이벤트
```

## 5.3 문제 해결 흐름

```
Pod이 정상 동작하지 않음
│
├── Pod이 생성되지 않음 (Pending)
│   → kubectl describe pod → Events 확인
│   → 리소스 부족? 노드 추가 / requests 줄이기
│   → Taint? Toleration 추가
│   → PVC? StorageClass 확인
│
├── Pod이 재시작 반복 (CrashLoopBackOff)
│   → kubectl logs <pod> --previous
│   → 코드 에러? 수정 후 재배포
│   → 설정 누락? ConfigMap/Secret 확인
│   → OOM? memory limits 늘리기
│
├── Pod은 Running인데 응답 없음
│   → kubectl exec -it <pod> -- curl localhost:8080
│   → 앱 내부 문제? 로그 확인
│   → Service 연결 문제? Label/Selector 확인
│
└── 외부에서 접근 불가
    → Service 확인 → Ingress 확인 → DNS 확인 → LB 확인
```

---

# 시리즈를 마치며

6편에 걸쳐 Kubernetes의 기초 개념부터 AWS EKS 실무 운영까지를 다뤘다.

| 편                                                    | 핵심                                                     |
| ----------------------------------------------------- | -------------------------------------------------------- |
| [01 K8s 입문](/blog/2026/k8s-01-intro/)               | 컨테이너, 클러스터 구조, YAML, kubectl                   |
| [02 워크로드](/blog/2026/k8s-02-workloads/)           | Pod, Deployment, Job, CronJob, Resource, Probe           |
| [03 네트워킹](/blog/2026/k8s-03-networking/)          | Service, Ingress, HPA, Node Scheduling                   |
| [04 설정·스토리지](/blog/2026/k8s-04-config-storage/) | ConfigMap, Secret, PV/PVC, Namespace, Helm               |
| [05 EKS 아키텍처](/blog/2026/amazon-eks-guide/)       | EKS 구조, Worker Node 옵션, Auto Mode, ECR               |
| **06 EKS 운영**                                       | VPC CNI, Pod Identity, 비용, 업그레이드, Troubleshooting |

---

# 참고 문헌

- [VPC CNI Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/vpc-cni.html)
- [EKS Pod Identity 공식 문서](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html)
- [IRSA vs Pod Identity 비교](https://www.kubeblogs.com/eks-irsa-vs-eks-pod-identity-two-ways-to-grant-application-access-to-aws-resources/)
- [EKS IAM Best Practices](https://aws.github.io/aws-eks-best-practices/security/docs/iam/)
- [Amazon EKS Pricing](https://aws.amazon.com/eks/pricing/)
- [EKS vs GKE vs AKS 2026 비교](https://sedai.io/blog/kubernetes-cost-eks-vs-aks-vs-gke)
- [EKS Upgrade Best Practices](https://aws.github.io/aws-eks-best-practices/upgrades/)
- [EKS Version Lifecycle](https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html)
