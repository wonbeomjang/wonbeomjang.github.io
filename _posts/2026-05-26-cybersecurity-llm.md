---
layout: post
title: "사이버 보안에서의 LLM: 공격·방어·평가의 지형"
date: 2026-05-26 23:00:00 +0900
description: "사이버 보안 LLM 시리즈의 도입부 — secure coding에서 자율 공격·방어까지의 전개, 그리고 이를 측정하는 벤치마크 지형(Cybench, CVE-Bench, CyberSecEval, CTIBench 등) 개관"
categories: [paper]
tags: [llm, cybersecurity, security, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
featured: false
---

> 이 글은 사이버 보안 LLM 시리즈의 도입부다. 개별 벤치마크([Cybench](/blog/2026/cybench/), [CVE-Bench](/blog/2026/cve-bench/), [AutoAdvExBench](/blog/2026/autoadvexbench/), [CAIBench](/blog/2026/caibench/), [CyberSecEval](/blog/2026/cyberseceval/), [CTIBench](/blog/2026/ctibench/), [SecBench](/blog/2026/secbench/))와 사건([Claude Mythos](/blog/2026/claude-mythos/))으로 들어가기 전에, "사이버 보안에서 LLM이 무엇을 하고 어떻게 평가되는가"의 큰 그림을 그린다.

# Introduction

LLM과 사이버 보안의 관계는 지난 2년간 세 단계로 진화했다.

1. **Secure coding (2023~)**: "LLM이 짜는 코드가 안전한가?" — 생성 코드의 취약점 비율을 측정. CyberSecEval 1의 insecure coding이 대표적이다.
2. **LLM 공격 (2023~2024)**: "LLM 자체를 어떻게 깨뜨리는가?" — jailbreak·prompt injection. 이것은 [Red-Teaming 시리즈](/blog/2026/ganguli-red-teaming/)의 주제다.
3. **자율 공격·방어 (2024~)**: "LLM이 사람 없이 실제 시스템을 공격·방어할 수 있는가?" — CTF 풀이, 실제 CVE 익스플로잇, 위협 인텔리전스 자동화.

3단계로 넘어오면서 평가의 성격이 근본적으로 바뀌었다. 더 이상 "정답 맞히기"가 아니라, [에이전트](/blog/2026/what-is-an-agent/)가 **실제 환경에서 다단계로 행동해 목표(취약점 발견·익스플로잇·방어)를 달성하는가**를 본다. 그리고 2026년 [Claude Mythos](/blog/2026/claude-mythos/)가 자율 zero-day 발견·익스플로잇에서 능력 도약을 보이면서, 이 평가들은 단순한 학술 호기심이 아니라 **거버넌스의 핵심 도구**가 되었다.

# 왜 지금 중요한가 — 공격-방어 비대칭

핵심 동인은 **time-to-exploit의 붕괴**다. CSA·SANS·OWASP 공동 브리핑은 평균 취약점 무기화 시간이 2018년 약 2.3년에서 2026년 약 20시간으로 줄었다고 본다. 같은 LLM 능력이 방어(먼저 찾아 패치)와 공격(같은 취약점 무기화) 양쪽에 쓰이는 **dual-use** 특성 때문에, "먼저 손에 쥔 쪽"이 결정적으로 유리해진다.

따라서 우리는 두 가지를 동시에 물어야 한다.

- **역량(capability)**: 모델이 실제로 얼마나 잘 공격/방어하는가? → 벤치마크로 측정
- **위험(risk)**: 그 역량이 악용될 때 사회적 비용은? → 거버넌스(ASL, 책임 공개)로 관리

# 평가 지형 — 세 갈래

사이버 보안 LLM 벤치마크는 크게 세 부류로 나뉜다. 이 시리즈는 각 부류의 대표를 한 편씩 다룬다.

## A. 자율 공격/익스플로잇 역량

에이전트가 실제로 취약점을 찾고 익스플로잇하는 능력을 측정한다. [Claude Mythos](/blog/2026/claude-mythos/)가 보인 능력과 직결된다.

| 벤치마크                                     | 무엇을 재는가                                   |
| -------------------------------------------- | ----------------------------------------------- |
| [Cybench](/blog/2026/cybench/)               | 프로 CTF 40과제로 cyber agent 역량·subtask 평가 |
| [CVE-Bench](/blog/2026/cve-bench/)           | 실제 critical CVE 웹 취약점 자율 익스플로잇     |
| [AutoAdvExBench](/blog/2026/autoadvexbench/) | 적대적 예제 방어를 자율적으로 깨기              |
| [CAIBench](/blog/2026/caibench/)             | 공격·방어·지식·프라이버시 통합 메타 벤치마크    |

## B. 위험·역량 종합 평가

공격 역량뿐 아니라 secure coding, prompt injection, 안전-효용 트레이드오프까지 폭넓게 본다.

| 벤치마크                                       | 무엇을 재는가                                                    |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| [CyberSecEval (1–3)](/blog/2026/cyberseceval/) | secure coding·prompt injection·공격 역량·FRR (Meta Purple Llama) |

## C. 위협 인텔리전스·지식·방어

SOC/분석가 업무 — 위협 인텔, 지식, 분류·귀속을 본다.

| 벤치마크                         | 무엇을 재는가                                      |
| -------------------------------- | -------------------------------------------------- |
| [CTIBench](/blog/2026/ctibench/) | 위협 인텔(CVE→CWE, CVSS, 행위자 귀속, ATT&CK 추출) |
| [SecBench](/blog/2026/secbench/) | 대규모 보안 지식·추론 MCQ/SAQ(중·영, 4.7만+ 문항)  |

# 관통하는 긴장 — 점수인가 스캐폴딩인가, 그리고 dual-use

이 시리즈에서 반복되는 두 가지 긴장을 미리 짚는다.

1. **점수 ≠ 모델 능력**: [에이전트 벤치마크](/blog/2026/agentbench/)에서처럼, 같은 모델도 agent harness·도구·반복 시도 예산에 따라 결과가 크게 달라진다. CAIBench는 스캐폴딩만으로 성능이 2.6배까지 변한다고 보고한다.
2. **dual-use 딜레마**: 취약점을 잘 찾는 능력은 방어에도 공격에도 쓰인다. CyberSecEval의 false refusal rate(FRR), Mythos의 책임 공개·Project Glasswing은 모두 이 긴장을 다루는 장치다.

# Conclusion

사이버 보안은 LLM 평가의 최전선이다. 코딩·QA·계획을 넘어 **"실제 시스템을 자율적으로 공격·방어하는 능력"**을 재기 때문이다. 이 시리즈는 그 지형을 한 편씩 짚는다. 시작점은 이 분야가 왜 갑자기 뜨거워졌는지를 보여준 사건, [Claude Mythos](/blog/2026/claude-mythos/)다.

> 이어서 읽기: [Claude Mythos](/blog/2026/claude-mythos/) · [Cybench](/blog/2026/cybench/) · [CVE-Bench](/blog/2026/cve-bench/) · [AutoAdvExBench](/blog/2026/autoadvexbench/) · [CAIBench](/blog/2026/caibench/) · [CyberSecEval](/blog/2026/cyberseceval/) · [CTIBench](/blog/2026/ctibench/) · [SecBench](/blog/2026/secbench/)

# 참고 문헌

- [Cybench (arXiv 2408.08926)](https://arxiv.org/abs/2408.08926)
- [CVE-Bench (arXiv 2503.17332)](https://arxiv.org/abs/2503.17332)
- [AutoAdvExBench (arXiv 2503.01811)](https://arxiv.org/abs/2503.01811)
- [CAIBench (arXiv 2510.24317)](https://arxiv.org/abs/2510.24317)
- [CyberSecEval 3 (arXiv 2408.01605)](https://arxiv.org/abs/2408.01605)
- [CTIBench (arXiv 2406.07599)](https://arxiv.org/abs/2406.07599)
- [SecBench (arXiv 2412.20787)](https://arxiv.org/abs/2412.20787)
- [Large Language Models for Cyber Security — survey (arXiv 2511.04508)](https://arxiv.org/abs/2511.04508)
