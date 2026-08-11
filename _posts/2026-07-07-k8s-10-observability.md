---
layout: post
title: "Kubernetes 관측성 — 로그·메트릭과 Prometheus/Grafana"
date: 2026-07-07 10:00:00 +0900
description: "kubectl logs·events·top으로 시작해, 로그 수집 파이프라인과 Prometheus·Grafana 메트릭 스택까지 — 클러스터가 지금 어떤지 들여다보는 관측성 입문"
categories: [infra]
tags: [kubernetes, infra, observability, prometheus, grafana, logging, metrics]
giscus_comments: true
related_posts: true
---

> 이 글은 **K8s 입문 시리즈**의 확장편(10편)이다. 본편(01~09)은 [09편](/blog/2026/k8s-09-operator-cncf/)에서 마무리됐고, 이 글은 실무 진입을 위해 "운영 중인 클러스터를 어떻게 들여다보는가"를 덧붙인다.
>
> - [01: Kubernetes의 탄생 — Google Borg에서 CNCF까지](/blog/2026/k8s-01-history/)
> - [02: 내 노트북에 클러스터 만들기 — kind와 kubectl](/blog/2026/k8s-02-local-setup/)
> - [03: Kubernetes 아키텍처 — Control Plane과 Node](/blog/2026/k8s-03-architecture/)
> - [04: Pod의 모든 것 — 생성부터 스케줄링까지](/blog/2026/k8s-04-pod/)
> - [05: 워크로드 — ReplicaSet, Deployment, StatefulSet, DaemonSet](/blog/2026/k8s-05-workloads/)
> - [06: 네트워킹 — Service와 Ingress](/blog/2026/k8s-06-networking/)
> - [07: 스토리지와 설정 — PV/PVC, ConfigMap, Secret](/blog/2026/k8s-07-storage-config/)
> - [08: 권한 관리 — ServiceAccount와 RBAC](/blog/2026/k8s-08-rbac/)
> - [09: 확장과 생태계 — Operator와 CNCF Projects](/blog/2026/k8s-09-operator-cncf/)
> - **10: 관측성 — 로그·메트릭과 Prometheus/Grafana** ← 현재 글
>
> 이 시리즈의 커리큘럼은 SK Devocean의 [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua) 글의 학습 로드맵을 바탕으로 구성했다.

# 1. 들어가며 — "잘 돌고 있나?"에 답하기

[09편](/blog/2026/k8s-09-operator-cncf/)까지 오면 앱을 배포하고(05), 트래픽을 연결하고(06), 데이터를 저장하고(07), 권한을 통제하고(08), 생태계 도구로 확장(09)하는 것까지 할 수 있다. 그런데 실제로 서비스를 올려두면 곧바로 다른 질문이 쏟아진다.

- "지금 이 서비스, **정상인가?** 사용자 응답이 느려졌다는데 원인이 뭐지?"
- "어젯밤에 Pod가 몇 번 재시작됐지? 그때 **무슨 로그**를 남기고 죽었나?"
- "트래픽이 어제보다 늘었나? CPU는 언제 튀었나? **한 달 추이**는?"

이 질문들에 답하는 능력이 **관측성(Observability)**이다. 비유하면 자동차의 **계기판과 블랙박스**다. 엔진(클러스터)이 돌아가는 것과, 속도·연료·경고등을 읽고 사고 기록을 되짚는 것은 별개의 능력이다. 계기판이 없으면 시속 몇 km인지도 모른 채 달리는 셈이다.

관측성은 흔히 **세 개의 기둥(three pillars)**으로 설명된다.

| 기둥                 | 답하는 질문                  | 형태                          | 대표 도구                       |
| -------------------- | ---------------------------- | ----------------------------- | ------------------------------- |
| **로그(Logs)**       | "그때 무슨 일이 있었나?"     | 시각이 찍힌 텍스트 이벤트     | Fluent Bit, Loki, Elasticsearch |
| **메트릭(Metrics)**  | "지금·추이로 얼마나 되나?"   | 시계열 숫자 (CPU, 요청 수 등) | Prometheus, Grafana             |
| **트레이스(Traces)** | "이 요청이 어디서 느려졌나?" | 요청이 서비스를 거친 경로     | OpenTelemetry, Jaeger, Tempo    |

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-10-observability/observability-pillars.png" class="img-fluid rounded z-depth-1" alt="관측성의 세 기둥 — 로그(무슨 일이 있었나), 메트릭(얼마나 되나), 트레이스(어디서 느려졌나)가 각각 다른 질문에 답하고 대표 도구가 붙는 구조" %}

이 글은 입문 범위에 맞춰 **로그와 메트릭**을 중심으로 다룬다. 먼저 추가 설치 없이 쓰는 `kubectl` 내장 도구로 시작해, 그것으로 부족해지는 지점을 짚고, 로그 수집 파이프라인과 Prometheus·Grafana 스택으로 넘어간다. 트레이스는 마지막에 개념만 짚는다.

---

# 2. 추가 설치 없이 — kubectl로 들여다보기

가장 먼저 손이 가는 건 이미 우리 손에 있는 `kubectl`이다. 관측에 쓰는 내장 명령은 세 가지다.

## 2.1 kubectl logs — 컨테이너가 남긴 말

[04편](/blog/2026/k8s-04-pod/)의 디버깅 루틴에서 이미 만났다. 컨테이너가 표준 출력(stdout/stderr)으로 뱉은 로그를 보여준다. 실무에서 자주 쓰는 조합은 이렇다.

```bash
kubectl logs <pod>                 # 현재 로그 출력
kubectl logs -f <pod>              # 실시간 스트리밍(follow)
kubectl logs --previous <pod>      # 직전에 죽은 컨테이너의 로그 (CrashLoopBackOff 추적)
kubectl logs <pod> -c <container>  # 멀티 컨테이너 Pod에서 특정 컨테이너 지정
kubectl logs -l app=web --tail=50  # 라벨로 여러 Pod의 로그를 한 번에
kubectl logs <pod> --since=1h      # 최근 1시간치만
```

- `-c`가 필요한 이유는 [04편](/blog/2026/k8s-04-pod/)의 사이드카·init 컨테이너 때문이다. 한 Pod에 컨테이너가 여럿이면 어느 컨테이너의 로그인지 지정해야 한다.
- `-l`(라벨 셀렉터)은 [05편](/blog/2026/k8s-05-workloads/)의 Label과 이어진다. Deployment가 만든 Pod 여러 개의 로그를 이름 대신 라벨로 한꺼번에 볼 수 있다.

## 2.2 kubectl get events — 클러스터가 남긴 사건 기록

`logs`가 **앱 쪽** 사정이라면, events는 **Kubernetes 쪽** 사정이다. 스케줄링, 이미지 풀, 재시작, 축출 같은 사건이 여기 남는다([04편](/blog/2026/k8s-04-pod/)에서 Pod가 뜨는 5단계를 이벤트로 추적했던 그 화면이다).

```bash
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl get events -n kube-system --field-selector type=Warning
```

주의할 점 하나 — **이벤트는 기본적으로 약 1시간만 보관되고 사라진다.** "어제 새벽의 사건"은 이벤트로는 못 찾는다. 이 휘발성이 곧 뒤에서 중앙 수집이 필요한 이유가 된다.

## 2.3 kubectl top — 지금 얼마나 쓰고 있나

`top`은 노드와 Pod의 CPU·메모리 **실시간 사용량**을 보여준다. 단, [06편](/blog/2026/k8s-06-networking/)에서 HPA를 위해 설치했던 **metrics-server**가 있어야 동작한다(없으면 `error: Metrics API not available`).

```bash
kubectl top nodes
kubectl top pods -A            # 모든 네임스페이스의 Pod 사용량
```

```text
NAME                 CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
kind-control-plane   180m         4%     1120Mi          14%
```

## 2.4 여기까지의 한계

세 명령 모두 강력하지만 공통된 벽이 있다. **"지금"과 "방금"만 보여준다는 것이다.**

- 로그는 Pod가 살아 있을 때의 현재 로그뿐이고, Pod가 사라지면 함께 사라진다.
- 이벤트는 한 시간 뒤면 없다.
- `top`은 이 순간의 값일 뿐, "어제 오후 3시엔 얼마였나"라는 **과거 추이**나 "임계치를 넘으면 알려줘"라는 **알림**은 없다.

즉 `kubectl`은 **실시간 청진기**이지 **기록 장치**가 아니다. 운영을 하려면 로그와 메트릭을 **클러스터 밖(또는 전용 저장소)에 모아 오래 보관하고, 질의하고, 알림을 거는** 별도의 시스템이 필요하다. 여기서부터가 관측성 스택의 영역이다.

---

# 3. 로그를 중앙에 모으기 — 수집 파이프라인

로그의 근본 문제는 [07편](/blog/2026/k8s-07-storage-config/)에서 본 **휘발성**과 같다. 컨테이너 로그는 노드의 파일로 쌓이다가, Pod가 사라지거나 로그가 로테이션되면 없어진다. 노드가 100대면 로그도 100곳에 흩어진다. `kubectl logs`로 노드를 돌아다니며 찾는 것은 규모가 커지면 불가능하다.

해법의 구조는 [05편](/blog/2026/k8s-05-workloads/)에서 이미 예고됐다. **로그 수집 에이전트를 DaemonSet으로 모든 노드에 하나씩 띄우고, 각 노드의 로그 파일을 읽어 중앙 저장소로 보내는** 것이다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-10-observability/logging-pipeline.png" class="img-fluid rounded z-depth-1" alt="로그 수집 파이프라인 — 각 노드의 Pod 로그를 DaemonSet 수집 에이전트가 hostPath로 읽어 중앙 로그 저장소로 보내고, 사용자는 저장소에 질의하는 구조" %}

- **수집 에이전트**는 노드의 로그 디렉터리를 [07편](/blog/2026/k8s-07-storage-config/)의 **hostPath**로 마운트해 읽는다 — hostPath의 대표적 정당 용도가 바로 이것이다.
- **DaemonSet**이라 노드가 늘면 에이전트도 자동으로 따라 늘어난다([05편](/blog/2026/k8s-05-workloads/)).
- 중앙 저장소에 모이면 노드를 넘나드는 검색, 장기 보관, 대시보드가 가능해진다.

대표 도구 조합은 이렇다. 외울 필요는 없고, "에이전트 + 저장소 + 조회 UI"라는 삼각 구조만 기억하면 된다.

| 역할          | 대표 도구                                                                    |
| ------------- | ---------------------------------------------------------------------------- |
| 수집 에이전트 | Fluent Bit, Fluentd([09편](/blog/2026/k8s-09-operator-cncf/) CNCF), Promtail |
| 저장·검색     | Loki, Elasticsearch, OpenSearch                                              |
| 조회·시각화   | Grafana, Kibana                                                              |

특히 Fluentd는 [09편](/blog/2026/k8s-09-operator-cncf/)에서 본 CNCF Graduated 프로젝트이자 DaemonSet 로그 수집의 대표 사례다. 요즘 가벼운 조합으로는 **Promtail(수집) + Loki(저장) + Grafana(조회)**가 인기인데, 메트릭 쪽과 Grafana를 공유할 수 있어서다. 그 메트릭 쪽으로 넘어가 보자.

---

# 4. 메트릭 파이프라인 — Prometheus

메트릭은 "요청 수", "CPU 사용률", "에러율"처럼 **시간에 따라 변하는 숫자**다. 이 세계의 사실상 표준이 [09편](/blog/2026/k8s-09-operator-cncf/)에서 CNCF Graduated 프로젝트로 만난 **Prometheus**다.

## 4.1 Pull 모델 — Prometheus가 긁어간다

Prometheus의 핵심 설계는 **pull(scrape) 방식**이다. 각 애플리케이션이 자기 메트릭을 `/metrics` 같은 HTTP 엔드포인트로 노출해두면, Prometheus 서버가 **주기적으로 그 주소를 방문해 값을 긁어간다.** 앱이 어딘가로 밀어 보내는(push) 것이 아니라, 중앙이 정해진 대상 목록을 돌며 당겨오는 구조다.

{% include figure.liquid loading="lazy" path="assets/post/image/k8s-10-observability/prometheus-architecture.png" class="img-fluid rounded z-depth-1" alt="Prometheus 아키텍처 — Prometheus 서버가 앱과 exporter의 /metrics 엔드포인트를 주기적으로 scrape해 시계열 DB에 저장하고, Grafana가 PromQL로 질의해 대시보드를 그리며, 임계치 초과 시 Alertmanager가 알림을 보내는 구조" %}

왜 pull일까. 대상 목록을 중앙이 쥐고 있으면 "지금 이 대상이 응답하는가" 자체가 헬스체크가 되고, 스크레이프 주기·타임아웃을 일관되게 관리할 수 있다. 무엇보다 Pod가 끊임없이 뜨고 지는 Kubernetes에서, Prometheus는 [06편](/blog/2026/k8s-06-networking/)의 서비스 디스커버리를 이용해 **scrape 대상을 자동으로 갱신**한다. 새 Pod가 뜨면 대상에 들어오고, 사라지면 빠진다.

구성 요소를 정리하면 이렇다.

| 구성 요소           | 역할                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| **Prometheus 서버** | 대상을 scrape하고, 시계열 DB(TSDB)에 저장하며, PromQL 질의를 처리    |
| **Exporter**        | 메트릭이 없는 대상을 대신 노출 (예: node-exporter가 노드 CPU/메모리) |
| **Alertmanager**    | 규칙(임계치)에 걸린 알림을 받아 Slack·이메일 등으로 라우팅           |

`node-exporter`는 노드의 하드웨어·OS 메트릭을 노출하는 exporter인데, 눈치챘겠지만 "모든 노드에 하나씩"이라 [05편](/blog/2026/k8s-05-workloads/)의 **DaemonSet**으로 배포된다. `kube-state-metrics`는 Deployment·Pod 같은 오브젝트의 상태(예: "원하는 replicas 대 현재 replicas")를 메트릭으로 노출한다.

## 4.2 metrics-server와 무엇이 다른가

혼동하기 쉬운 대목이다. [06편](/blog/2026/k8s-06-networking/)에서 HPA를 위해 설치한 **metrics-server**도 메트릭을 다루지 않았나? 둘은 목적이 전혀 다르다.

| 항목      | metrics-server                           | Prometheus                                         |
| --------- | ---------------------------------------- | -------------------------------------------------- |
| 목적      | HPA·`kubectl top`용 **실시간** 최소 지표 | 범용 모니터링 — 저장·질의·알림                     |
| 저장      | 저장 안 함 (현재 값만, 휘발성)           | 시계열 DB에 장기 저장                              |
| 지표 범위 | CPU·메모리뿐                             | 앱이 노출하는 무엇이든 (요청 수, 큐 길이, 에러율…) |
| 질의      | 없음                                     | PromQL로 집계·연산                                 |
| 과거 추이 | 불가                                     | 가능 (그래프·알림의 근거)                          |

한 줄로 하면 — **metrics-server는 HPA에게 "지금 값"을 대주는 좁은 파이프이고, Prometheus는 관측·알림을 위한 범용 시계열 데이터베이스다.**

## 4.3 ServiceMonitor — Operator로 scrape 대상 선언하기

Prometheus를 Kubernetes에서 운영할 때는 대개 [09편](/blog/2026/k8s-09-operator-cncf/)의 **Prometheus Operator**를 함께 쓴다. 이 Operator는 `ServiceMonitor`라는 CRD를 등록하는데, "이 라벨을 가진 Service의 Pod들을 scrape하라"를 **선언적으로** 적는 리소스다.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: web-monitor
spec:
  selector:
    matchLabels:
      app: web # 이 라벨의 Service를 scrape 대상으로
  endpoints:
    - port: metrics # Service의 이 포트에서 /metrics를 긁는다
      interval: 15s
```

Prometheus 설정 파일을 직접 고치는 대신 ServiceMonitor 오브젝트만 만들면, Operator가 그것을 watch하다가 Prometheus 설정을 대신 갱신한다. [09편](/blog/2026/k8s-09-operator-cncf/)에서 배운 **"CRD로 새 명사를 만들고, 컨트롤러가 reconcile한다"**는 Operator 패턴이 모니터링에도 그대로 적용된 것이다.

---

# 5. 시각화와 알림 — Grafana

Prometheus가 숫자를 모으고 질의(PromQL)한다면, 그 숫자를 **사람이 읽을 그래프와 대시보드**로 바꾸는 것이 **Grafana**다. Prometheus를 데이터소스로 연결하면, PromQL 쿼리 결과를 시계열 그래프·게이지·표로 그려준다.

- **대시보드**: 관련 그래프를 한 화면에 모은 것. 클러스터 개요, 노드별 자원, 네임스페이스별 사용량 같은 대시보드를 커뮤니티가 공유하며, JSON 하나를 import하면 바로 쓸 수 있다.
- **데이터소스 통합**: Grafana는 Prometheus(메트릭)와 Loki(로그)를 **동시에** 물릴 수 있다. 한 화면에서 "CPU가 튄 시각"의 그래프와 그 시각의 로그를 나란히 보는 식이다. 3장에서 Loki+Grafana 조합을 권한 이유가 이것이다.
- **알림**: Grafana 자체 알림, 또는 Prometheus의 Alertmanager로 "5분간 에러율 5% 초과"를 감지해 Slack으로 쏘는 규칙을 건다.

계기판 비유로 돌아오면, Prometheus는 센서에서 값을 읽어 기록하는 **계기 모듈**이고, Grafana는 그 값을 운전자가 보는 **계기판 화면**이다.

---

# 6. 실습 — kind에 모니터링 스택 한 번에 올리기

Prometheus, Grafana, node-exporter, kube-state-metrics, Prometheus Operator를 하나씩 설치하는 것은 번거롭다. 다행히 이 전부를 묶은 [09편](/blog/2026/k8s-09-operator-cncf/)의 **Helm** 차트 `kube-prometheus-stack`이 있다. [02편](/blog/2026/k8s-02-local-setup/)의 kind 클러스터에 올려보자.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

설치가 끝나면 스택 구성 요소들이 뜬다. 지금까지 배운 워크로드 타입이 골고루 보인다.

```bash
kubectl get pods -n monitoring
```

```text
NAME                                                     READY   STATUS    RESTARTS   AGE
monitoring-grafana-...                                   3/3     Running   0          2m
monitoring-kube-prometheus-operator-...                  1/1     Running   0          2m
monitoring-kube-state-metrics-...                        1/1     Running   0          2m
monitoring-prometheus-node-exporter-xxxxx                1/1     Running   0          2m
prometheus-monitoring-kube-prometheus-prometheus-0       2/2     Running   0          2m
```

- `...-node-exporter-xxxxx`는 [05편](/blog/2026/k8s-05-workloads/)의 **DaemonSet**이다. `kubectl get ds -n monitoring`으로 확인하면 노드 수만큼 뜬 것이 보인다.
- `prometheus-...-0`은 이름 끝의 서수(`-0`)에서 알 수 있듯 [05편](/blog/2026/k8s-05-workloads/)의 **StatefulSet**이다. 시계열 데이터를 [07편](/blog/2026/k8s-07-storage-config/)의 PVC에 보존해야 하기 때문이다.

Grafana에 접속해 보자. [06편](/blog/2026/k8s-06-networking/)의 `port-forward`로 로컬 포트를 연결한다.

```bash
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
```

브라우저로 `http://localhost:3000`에 접속하면 로그인 화면이 뜬다. 초기 관리자 비밀번호는 [07편](/blog/2026/k8s-07-storage-config/)의 **Secret**에 들어 있으니 꺼내 본다(사용자명은 `admin`).

```bash
kubectl get secret -n monitoring monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d ; echo
```

로그인하면 이 차트가 미리 넣어둔 대시보드들이 이미 있다. "Kubernetes / Compute Resources / Namespace (Pods)" 같은 대시보드를 열면, 방금까지 `kubectl top`으로 한 줄씩 보던 값들이 **시간축 그래프**로 그려진다 — 과거 추이가 보인다는 것이 2장의 `kubectl`과 결정적으로 다른 점이다.

Prometheus UI도 직접 열어 PromQL을 쳐볼 수 있다.

```bash
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```

`http://localhost:9090`의 입력창에 예를 들어 노드별 CPU 사용률 쿼리를 넣으면 그 자리에서 그래프가 나온다.

```promql
sum(rate(container_cpu_usage_seconds_total[5m])) by (node)
```

실습이 끝나면 정리한다. 스택이 무겁기 때문에 로컬 자원을 꽤 쓰므로, 확인 후 지우는 것이 좋다.

```bash
helm uninstall monitoring -n monitoring
kubectl delete namespace monitoring
```

---

# 7. 트레이스 한 스푼 — 요청의 여정을 따라가기

로그와 메트릭으로도 "무슨 일이 있었나", "얼마나 되나"는 안다. 하지만 마이크로서비스가 열 개쯤 얽힌 환경에서 "**사용자의 이 요청이 정확히 어느 서비스에서 300ms를 잡아먹었나**"는 둘 중 어느 것으로도 바로 답하기 어렵다. 이 질문에 답하는 것이 세 번째 기둥, **분산 트레이싱(distributed tracing)**이다.

핵심 아이디어는 요청이 첫 서비스에 도착할 때 **추적 ID(trace ID)**를 부여하고, 그 요청이 거치는 모든 서비스가 같은 ID로 자기 구간(span)의 소요 시간을 기록하는 것이다. 나중에 같은 trace ID의 span들을 이어 붙이면 "게이트웨이 → 인증 → 주문 → DB" 경로에서 각 구간이 몇 ms였는지 폭포수 그래프로 보인다.

- **OpenTelemetry(OTel)**: 트레이스·메트릭·로그를 수집하는 벤더 중립 표준. CNCF 프로젝트로, 사실상 계측(instrumentation)의 표준이 됐다.
- **Jaeger, Tempo**: 트레이스를 저장하고 조회하는 백엔드. Tempo는 Grafana와 잘 붙는다.

입문 단계에서는 "로그·메트릭 위에, 요청 경로를 추적하는 세 번째 축이 있다"는 것과 OpenTelemetry라는 이름만 알아두면 충분하다.

---

# 8. 마무리

이번 확장편의 핵심을 요약한다.

| 개념                         | 한 줄 요약                                                            |
| ---------------------------- | --------------------------------------------------------------------- |
| 관측성 3기둥                 | 로그(무슨 일), 메트릭(얼마나), 트레이스(어디서 느려졌나)              |
| kubectl 내장                 | `logs`·`get events`·`top` — 강력하지만 "지금·방금"만, 저장·알림 없음  |
| 로그 파이프라인              | DaemonSet 에이전트가 hostPath로 읽어 중앙 저장소로 (Fluentd/Loki)     |
| Prometheus                   | `/metrics`를 pull로 긁어 시계열 DB에 저장, PromQL로 질의              |
| metrics-server vs Prometheus | 전자는 HPA용 실시간 값, 후자는 저장·질의·알림의 범용 스택             |
| Grafana                      | Prometheus·Loki를 물려 그래프·대시보드·알림으로 시각화                |
| ServiceMonitor               | Prometheus Operator의 CRD — scrape 대상을 선언적으로 지정 (09편 패턴) |

이 글에서 다시 확인한 것은, 관측성 스택이 **앞선 아홉 편에서 배운 조각들의 조합**이라는 점이다. 수집 에이전트는 DaemonSet(05)과 hostPath(07)로, Prometheus는 StatefulSet(05)과 PVC(07)로, scrape 대상 발견은 Service 디스커버리(06)로, ServiceMonitor는 Operator 패턴(09)으로 돌아간다. 새 마법이 아니라 익힌 원리의 재조합이다.

이로써 우리는 클러스터를 **배포하고(01–05), 연결하고(06), 저장하고(07), 통제하고(08), 확장하고(09), 관측한다(10)**. 처음 보는 관측 도구를 만나도 "이건 세 기둥 중 무엇을 다루고, 어떤 워크로드로 배포되며, 무엇을 reconcile하는가"부터 물으면 길이 보일 것이다.

> 이 글은 K8s 입문 시리즈의 확장편이다. 본편의 마지막은 [09: 확장과 생태계 — Operator와 CNCF Projects](/blog/2026/k8s-09-operator-cncf/)이고, 처음부터 보려면 [01: Kubernetes의 탄생](/blog/2026/k8s-01-history/)에서 시작하면 된다.

---

# 참고 문헌

- [Kubernetes(쿠버네티스)를 처음 공부하려면 무엇을 공부해야 할까?](https://devocean.sk.com/blog/techBoardDetail.do?ID=165905&boardType=techBlog) (seungkyua, SK Devocean, 2024) — 시리즈 로드맵 출처
- [Logging Architecture — kubernetes.io](https://kubernetes.io/docs/concepts/cluster-administration/logging/)
- [Tools for Monitoring Resources — kubernetes.io](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-usage-monitoring/)
- [Metrics Server (kubernetes-sigs)](https://github.com/kubernetes-sigs/metrics-server)
- [Prometheus — Overview](https://prometheus.io/docs/introduction/overview/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [kube-prometheus-stack — Helm chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Grafana Loki](https://grafana.com/docs/loki/latest/)
- [OpenTelemetry](https://opentelemetry.io/docs/)
- [Fluentd (CNCF)](https://www.fluentd.org/)
