---
layout: post
title: "CyberSecEval (1–3): Meta Purple Llama의 사이버 보안 위험·역량 평가"
date: 2026-05-26 17:00:00 +0900
description: "CyberSecEval 시리즈 리뷰 — Meta Purple Llama의 사이버 보안 벤치마크. insecure coding부터 prompt injection, 공격 역량, false refusal rate(FRR)까지 dual-use를 정면으로 다룬다"
categories: [paper]
tags: [llm, cybersecurity, security, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
featured: false
---

> [CyberSecEval 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in Large Language Models](https://arxiv.org/abs/2408.01605) (Wan et al., Meta, 2024) · [CyberSecEval (1)](https://arxiv.org/abs/2312.04724)

# Introduction

지금까지 본 [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)가 "공격 역량"에 집중한다면, Meta의 **CyberSecEval** 시리즈는 사이버 보안 위험을 **가장 넓게** 다룬다. 모델 출시(Purple Llama / Llama Guard 계열)와 함께 발전해온 실용 벤치마크로, "이 모델을 풀어도 되는가"를 판단하는 안전 평가의 역할을 한다.

CyberSecEval의 독특함은 **dual-use를 정면으로 다룬다**는 점이다. 같은 능력이 방어(취약점 탐지)에도 공격(익스플로잇 생성)에도 쓰이기 때문에, 단순히 "거부 잘함"이 아니라 **안전과 효용의 트레이드오프**를 함께 측정한다.

# 버전별 전개

CyberSecEval은 세 번에 걸쳐 범위를 넓혀왔다.

| 버전                      | 추가된 핵심 평가 축                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------- |
| **CyberSecEval 1** (2023) | **Insecure coding** — 모델이 생성한 코드의 취약점 비율 / 사이버공격 조력 거부          |
| **CyberSecEval 2** (2024) | **Prompt injection** 내성, **code interpreter abuse**, 취약점 익스플로잇, **FRR** 도입 |
| **CyberSecEval 3** (2024) | **공격 역량(offensive)** — 자동 스피어피싱, 스케일된 수동 사이버작전, 자율 공격작전    |

## CyberSecEval 3의 8개 위험

CyberSecEval 3는 위험을 두 범주로 나눠 **8개 위험**을 평가한다.

- **제3자에 대한 위험(risk to third parties)**: 자동 스피어피싱, 공격 작전 자동화·확장 등 — 모델이 **공격자를 얼마나 증폭**하는가
- **앱 개발자·최종 사용자에 대한 위험**: prompt injection, insecure code 생성, code interpreter abuse 등 — 모델을 **제품에 넣었을 때의 위험**

# 핵심 개념 — False Refusal Rate (FRR)

CyberSecEval 2가 도입한 **FRR**는 이 시리즈의 핵심 통찰이다. 보안 관련 요청을 모델이 **과도하게 거부**하는 비율을 측정한다.

- 거부율을 높이면 악용은 줄지만, 정당한 보안 업무(취약점 분석, 방어 코드 작성)까지 막혀 **효용이 망가진다**.
- 즉 좋은 모델은 "악성 요청은 거부하되, 정당한 보안 작업은 돕는다". FRR는 이 균형점을 정량화한다.

이는 [사이버 보안 LLM 개관](/blog/2026/cybersecurity-llm/)에서 짚은 dual-use 딜레마의 직접적 측정이며, [Red-Teaming 시리즈](/blog/2026/llama-guard/)의 over-refusal 논의와도 맞닿는다.

# Experiments

CyberSecEval 3는 **Llama 3 계열**과 동시대 SOTA 모델(GPT-4 등)을 함께 평가한다. 핵심 관점:

- 각 위험을 **완화책(safeguard) 적용 전후로 비교** — Llama Guard, Code Shield, Prompt Guard 같은 방어 레이어가 위험을 얼마나 줄이는지.
- 공격 역량(스피어피싱·자율 작전)은 모델이 위협 행위자를 실질적으로 증폭하는지를 본다.

핵심 메시지는 **"완화책을 갖추면 위험을 의미 있게 낮출 수 있다"** — 즉 평가의 목적이 단순 점수가 아니라 **배포 결정과 방어 설계**에 있다.

# 의의와 한계

**의의**

- 사이버 보안 LLM 위험을 **가장 폭넓게(8위험)** 다루는 실용 벤치마크. 모델 출시 안전 평가의 사실상 표준 중 하나.
- **FRR**로 안전-효용 트레이드오프를 정량화 — over-refusal 문제를 처음으로 보안 맥락에서 측정.
- 완화책(Purple Llama 도구군)과 함께 발전해 **방어 설계로 직결**.

**한계**

- 다수 항목이 정적·합성 시나리오라, [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)의 실환경 자율 익스플로잇만큼 깊지는 않다.
- 빠르게 진화하는 공격 역량을 정적 테스트로 따라잡기 어렵다.

# Conclusion

CyberSecEval은 "공격 역량"과 "방어 가능성", 그리고 "거부의 비용(FRR)"을 한 프레임에서 보는 **dual-use 평가의 표준**이다. [Cybench](/blog/2026/cybench/)가 능력의 천장을 재고 [CVE-Bench](/blog/2026/cve-bench/)가 실세계 익스플로잇을 잰다면, CyberSecEval은 **"이 모델을 풀어도 되는가, 어떤 안전장치와 함께 풀어야 하는가"**라는 배포 결정을 돕는다. [Claude Mythos](/blog/2026/claude-mythos/)의 출시 거부 결정도 같은 질문의 극단적 사례다.

> 이어서 읽기: [사이버 보안 LLM 개관](/blog/2026/cybersecurity-llm/) · [Claude Mythos](/blog/2026/claude-mythos/) · [CTIBench](/blog/2026/ctibench/) · [SecBench](/blog/2026/secbench/)

# 참고 문헌

- [CyberSecEval 3 (arXiv 2408.01605)](https://arxiv.org/abs/2408.01605) — Wan et al., Meta, 2024
- [CyberSecEval 2 (arXiv 2404.13161)](https://arxiv.org/abs/2404.13161)
- [Purple Llama CyberSecEval — 1 (arXiv 2312.04724)](https://arxiv.org/abs/2312.04724)
- [Meta PurpleLlama GitHub](https://github.com/meta-llama/PurpleLlama)
