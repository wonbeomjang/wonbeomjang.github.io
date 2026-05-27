---
layout: post
title: "Claude Mythos와 사이버 보안 LLM: 자율 취약점 발견의 변곡점"
date: 2026-05-26 12:00:00 +0900
description: "Anthropic Claude Mythos가 보여준 자율 zero-day 발견·익스플로잇 능력과, 이를 측정하는 사이버 보안 LLM 벤치마크(Cybench, CyberSecEval, CVE-Bench 등) 정리"
categories: [paper]
tags: [llm, cybersecurity, security, benchmark, evaluation, paper]
giscus_comments: true
related_posts: true
featured: false
---

> 본 글은 Anthropic이 공개한 [Claude Mythos Preview 기술 보고](https://red.anthropic.com/2026/mythos-preview/)와 공개된 사이버 보안 LLM 평가 연구를 **방어적·분석적 관점**에서 정리한 것이다. 공개 정보를 개념 수준에서 다루며, 실제 익스플로잇 코드나 공격 실행 절차는 포함하지 않는다.

# Introduction

LLM과 보안의 관계는 지난 2년간 빠르게 변해왔다. 처음에는 "LLM이 안전한 코드를 짜는가"(secure coding), 다음에는 "LLM을 어떻게 공격하는가"(jailbreak·prompt injection)가 화두였다. 그리고 2026년, 질문이 한 단계 더 나아갔다.

> **"LLM이 사람의 개입 없이, 실제 소프트웨어의 zero-day를 스스로 찾아내고 익스플로잇까지 작성할 수 있는가?"**

2026년 4월 Anthropic이 공개한 **Claude Mythos Preview**는 이 질문에 충격적인 답을 내놓았다. Anthropic 자체 보고에 따르면 Mythos는 주요 OS·브라우저 전반에서 **수천 개의 미발견 취약점을 자율적으로 찾아냈고**, 그중 다수에 대해 **동작하는 익스플로잇까지 스스로 작성**했다. 더 놀라운 것은 직전 모델(Opus 4.6)이 자율 익스플로잇 개발에서 사실상 0%였다는 점이다 — 즉 이것은 점진적 향상이 아니라 **능력의 도약(capability jump)**이다.

Anthropic은 이 능력의 위험성 때문에 **모델을 일반 출시하지 않기로 결정**하고, 모델 대신 **시스템 카드와 기술 보고만 공개**했다. AI 역사상 보기 드문 선택이다.

이 글의 목적은 두 가지다.

1. Mythos가 정확히 무엇을 했는지, 그리고 왜 출시되지 않았는지를 공개 자료로 정리한다.
2. 이런 능력을 **어떻게 정량적으로 측정**하는지 — 즉 사이버 보안 LLM 벤치마크 지형을 정리한다.

# Claude Mythos란 무엇인가

Mythos Preview는 Anthropic이 내부 평가에서 확인한 고능력 모델로, 핵심은 **자율적 취약점 발견(vulnerability discovery)**과 **익스플로잇 개발(exploit development)**이다. Anthropic이 공개한 구체적 수치는 다음과 같다.

| 항목          | 내용                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **발견 규모** | 수천 개의 미발견 취약점, 공개 시점에 1% 이상 미패치                                              |
| **대상**      | 주요 OS·브라우저·VMM 등 핵심 인프라                                                              |
| **대표 사례** | 27년 된 OpenBSD SACK 버그, 16년 된 FFmpeg H.264 결함, production VMM의 guest-to-host 메모리 손상 |
| **자율성**    | "보안 훈련을 받지 않은 엔지니어가 하룻밤 사이 RCE를 찾아달라고 요청" 수준                        |
| **비용**      | OpenBSD 발견 < \$50, FFmpeg ~\$10,000(수백 run), Linux 익스플로잇 < \$2,000                      |

핵심은 **비용과 시간**이다. 수십 년간 숨어 있던 버그를 수십 달러 규모로, 사람 전문가 없이 찾아낸다는 것은 공격·방어의 경제학을 근본적으로 바꾼다.

## 무엇을 "자동화"했나 — 기법의 직관

Mythos가 인상적인 이유는 단순히 버그를 "찾은" 것이 아니라, 현대적 방어를 우회하는 **익스플로잇 체인을 스스로 구성**했다는 점이다. 아래는 그 기법들이 무엇인지에 대한 개념적 설명이다(공격 코드가 아니라 보안 교육 수준의 개념).

- **ROP (Return-Oriented Programming) 체인**: 실행 불가 메모리 보호(DEP/NX)를 우회하기 위해, 이미 메모리에 존재하는 코드 조각("gadget")들을 이어 붙여 원하는 동작을 만드는 고전 기법. Mythos는 **20개 gadget으로 된 체인을 여러 패킷에 분할**해 구성했다고 보고된다.
- **JIT heap spray + 샌드박스 탈출**: 브라우저의 JIT 컴파일러를 이용해 힙을 특정 패턴으로 채우고, 이를 통해 renderer 샌드박스와 OS 샌드박스를 **모두 탈출**. Mythos는 이를 위해 **4개의 취약점을 체이닝**했다.
- **KASLR 우회**: 커널 메모리 주소 랜덤화를 무력화해 익스플로잇의 신뢰성을 확보.
- **Race condition 기반 권한 상승(LPE)**: 미묘한 경쟁 조건을 악용해 Linux 등에서 로컬 권한 상승을 자율적으로 달성.

이들은 각각 숙련된 보안 연구자가 며칠~몇 주에 걸쳐 수행하는 작업이다. 핵심 메시지는 **"발견(finding)"을 넘어 "무기화(weaponization)"까지 자동화**되었다는 것이다.

## 능력 도약 — Opus 4.6 대비

Mythos가 "점진적 개선"이 아니라 "임계점 돌파"인 이유는 직전 세대와의 격차에 있다.

| 평가                                | Opus 4.6         | Mythos Preview                      |
| ----------------------------------- | ---------------- | ----------------------------------- |
| 자율 익스플로잇 개발 성공률         | 사실상 ~0%       | 정성적으로 큰 도약                  |
| 동작하는 Firefox 익스플로잇 생성    | 수백 시도 중 2회 | **181회**                           |
| OSS-Fuzz 코퍼스 crash (tier 1·2)    | 150–175          | **595**                             |
| 제어 흐름 탈취(control-flow hijack) | —                | 완전 패치된 10개 타깃서 tier 5 달성 |

특히 "수백 시도 중 2회 → 181회"는 단순한 정확도 향상이 아니라, **자율 익스플로잇 개발이 비로소 신뢰성 있게 작동하기 시작했다**는 신호다.

# 왜 Anthropic은 출시하지 않았나

Mythos의 가장 이례적인 점은 능력 자체가 아니라 **출시 거부**다.

- Anthropic은 "Mythos Preview를 일반에 공개할 계획이 없다"고 명시했다.
- 대신 **시스템 카드와 기술 보고만 공개**했다 — "모델 대신 시스템 카드를 출시(shipping the system card instead of the model)"한 셈이다.
- 외부 분석들은 Mythos의 자율 공격 역량이 사실상 **ASL-3 사이버 임계치**(Responsible Scaling Policy상의 고위험 단계) 근처에 도달했다고 평가한다.

이는 frontier lab이 "능력은 입증됐지만 안전 장치가 준비되지 않았으므로 배포하지 않는다"는 결정을 공개적으로 내린 드문 사례다. Anthropic은 향후 Opus 계열 모델에 새 안전장치를 먼저 적용해 검증한 뒤에야 유사 능력을 단계적으로 다루겠다고 밝혔다.

## Project Glasswing과 책임 공개

출시 대신 Anthropic이 택한 것은 **제한적·방어적 활용**이다.

- **Project Glasswing**: 핵심 인프라 소프트웨어를 방어하기 위해 Mythos Preview를 제한된 파트너에게만 제공. 보도된 launch 파트너에는 AWS, Apple, Broadcom, Cisco, CrowdStrike, Google, JPMorganChase, Linux Foundation, Microsoft, NVIDIA, Palo Alto Networks 등이 포함된다.
- **책임 공개(responsible disclosure)** 프로세스:
  - 영향받는 측에 보고 후 **90일 + 45일**의 패치 유예
  - **SHA-3 암호학적 커밋먼트**를 블로그와 함께 게시 — 패치 공개 후 특정 취약점 해시를 드러내 사후 검증 가능
  - 전문가 triage로 maintainer에게 과부하가 가지 않도록 조절. 검토된 198건 중 **심각도 정확 일치 89%, ±1단계 이내 98%**

여기서 흥미로운 평가 문제가 등장한다. AI가 보고한 취약점의 **심각도 판정이 사람 전문가와 얼마나 일치하는가** 자체가 하나의 평가 지표가 된 것이다.

# 그래서 이 능력을 어떻게 측정하나 — 사이버 보안 LLM 벤치마크

Mythos 같은 모델의 능력·위험을 객관적으로 재려면 표준 벤치마크가 필요하다. 사이버 보안 LLM 평가는 크게 **(A) 자율 공격 역량**, **(B) 위험·역량 종합 평가**, **(C) 위협 인텔리전스·방어**로 나뉜다.

## A. 자율 공격/익스플로잇 역량 (Mythos와 직결)

| 벤치마크           | 핵심                                                                        | 비고                                |
| ------------------ | --------------------------------------------------------------------------- | ----------------------------------- |
| **Cybench**        | 프로 CTF 40개 과제(암호·웹·리버싱·포렌식·exploitation 등 6범주) + subtask   | 자율 cyber agent 평가의 사실상 표준 |
| **CVE-Bench**      | AI 에이전트가 **실제 CVE**를 익스플로잇하는 능력                            | 실세계 취약점 기반                  |
| **AutoAdvExBench** | 적대적 예제 방어를 자율적으로 익스플로잇                                    | ML 보안 특화                        |
| **CAIBench**       | Cybench·SecEval·CyberMetric·AutoPen-Bench·CTIBench를 묶은 **메타 벤치마크** | cyber AI agent 종합 평가            |

Mythos가 보인 "fully-patched 타깃에서 control-flow hijack" 같은 능력은 정확히 이 계열(특히 Cybench의 exploitation 범주, CVE-Bench)이 측정하려는 대상이다.

## B. 위험·역량 종합 평가

| 벤치마크             | 핵심                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **CyberSecEval 1–3** | Meta Purple Llama. 보안 코딩, 프롬프트 인젝션 내성, 공격 역량(자동 익스플로잇 생성 등), false refusal rate(FRR)로 안전-효용 트레이드오프 측정 |
| **SecBench**         | 다차원 종합 벤치마크, 16개 SOTA 모델 평가                                                                                                     |
| **SECURE**           | 산업제어시스템(ICS) 도메인의 지식 추출·이해·추론                                                                                              |

CyberSecEval은 특히 **이중성(dual-use)**을 정면으로 다룬다 — 같은 능력이 방어(취약점 탐지)에도, 공격(익스플로잇 생성)에도 쓰인다는 점을 FRR 같은 지표로 정량화한다.

## C. 위협 인텔리전스·방어 / SOC

| 벤치마크          | 핵심                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| **CTIBench**      | 사이버 위협 인텔리전스(5,610 샘플, CVE→CWE 매핑·위협 행위자 귀속 등) |
| **CyberSOCEval**  | 멀웨어 분석·위협 인텔 추론 (SOC 업무 관점)                           |
| **CyberMetric**   | RAG 기반 보안 지식 평가                                              |
| **ExCyTIn-Bench** | 위협 조사(threat investigation) QA                                   |

이 분류는 [에이전트 벤치마크 시리즈](/blog/2026/what-is-an-agent/)에서 본 구도와 동일하다. 일반 agent를 [OSWorld](/blog/2026/osworld/)·[SWE-bench](/blog/2026/swe-bench/)로 재듯, 사이버 보안 agent는 Cybench·CVE-Bench로 잰다. Mythos는 이 좌표계에서 **exploitation 축의 천장을 갑자기 끌어올린** 사건이다.

# 회의론 — 능력은 과장되었나

균형을 위해 반론도 짚는다.

- **Semgrep 실험**: 코드 스캐닝 업체 Semgrep은 Opus 4.6·GPT 5.4 같은 모델이 유사 조건에서 Mythos가 찾은 버그를 동일하게 찾을 수 있는지 검증을 시도했다. 일부 능력 주장이 **셋업·하니스(harness)에 의해 부풀려졌을 가능성**을 제기한다.
- **EA Forum 분석("Overstated? Yes and No")**: Mythos의 역량이 일부 영역에서는 진짜 도약이지만, 발표 방식이 위협을 과대표상할 수 있다고 본다.

핵심 쟁점은 [에이전트 벤치마크](/blog/2026/agentbench/)에서 반복된 문제와 같다 — **"점수가 모델 능력인가, 스캐폴딩 능력인가"**. 동일 모델도 agent harness·도구·반복 시도 예산에 따라 결과가 크게 달라진다.

# 함의 — 방어자의 딜레마

Mythos가 촉발한 논의의 핵심은 **공격-방어 비대칭**이다.

- Cloud Security Alliance·SANS·OWASP GenAI는 공동 브리핑에서 **평균 time-to-exploit가 2018년 2.3년 → 2026년 약 20시간으로 붕괴**했다고 지적한다.
- 같은 능력이 방어(취약점을 먼저 찾아 패치)에도, 공격(같은 취약점을 무기화)에도 쓰인다. **먼저 손에 쥔 쪽이 유리**하다.
- 그래서 Anthropic의 Glasswing은 "방어자에게 먼저, 제한적으로" 쥐여주는 전략이고, 출시 거부는 "아무에게도 무기를 풀지 않는" 전략이다.

이는 거버넌스 질문으로 직결된다. **frontier 사이버 역량을 누가, 어떤 검증을 거쳐, 어떻게 접근하게 할 것인가.** Project Glasswing의 "신뢰된 방어자에게만" 모델과, 책임 공개의 암호학적 커밋먼트는 그 초기 실험이다.

# Conclusion

Claude Mythos가 사이버 보안 LLM 논의에 남긴 것은 다음과 같다.

- **변곡점의 정량화**: 자율 익스플로잇 개발이 "~0%"에서 "신뢰성 있게 작동"으로 넘어간 첫 공개 사례. 능력 점프를 수치(181회, 595 crash 등)로 보여줬다.
- **출시 거부라는 선례**: "모델 대신 시스템 카드를 출시"한 결정, ASL-3 임계치 논의, Glasswing·책임 공개는 frontier 거버넌스의 새 레퍼런스가 된다.
- **평가의 중요성 부각**: Cybench·CVE-Bench·CyberSecEval 같은 벤치마크가 "능력이 실재하는가, 과장인가"를 가르는 객관적 잣대로서 더 중요해졌다.
- **이중성의 직시**: 같은 능력이 방어와 공격 양쪽이라는 사실, 그리고 time-to-exploit 붕괴는 보안 운영 전반의 재설계를 요구한다.

LLM 평가가 코딩·QA·planning을 거쳐 이제 **"자율적으로 실제 시스템을 공격·방어하는 능력"**으로 확장되었다. Mythos는 그 최전선의 신호탄이며, 이를 재는 벤치마크와 거버넌스가 함께 성숙해야 한다는 과제를 남겼다.

> 관련 글: [에이전트란 무엇인가](/blog/2026/what-is-an-agent/), [AgentBench](/blog/2026/agentbench/), [SWE-bench](/blog/2026/swe-bench/), [OSWorld](/blog/2026/osworld/)

# 참고 문헌

**1차 자료 (Anthropic)**

- [Claude Mythos Preview — red.anthropic.com](https://red.anthropic.com/2026/mythos-preview/)

**거버넌스·산업 분석**

- [Cloud Security Alliance — Claude Mythos and the AI Autonomous Offensive Threshold](https://labs.cloudsecurityalliance.org/research/csa-research-note-claude-mythos-autonomous-offensive-thresho/)
- [Alan Turing Institute (CETAS) — Claude Mythos: What Does It Mean for Cybersecurity?](https://cetas.turing.ac.uk/publications/claude-mythos-future-cybersecurity)
- [KuppingerCole — What the Anthropic Mythos System Card Means for Cybersecurity and IAM](https://www.kuppingercole.com/blog/care/what-the-anthropic-mythos-system-card-means-for-cybersecurity-and-iam)
- [Schneier on Security — Mythos and Cybersecurity](https://www.schneier.com/blog/archives/2026/04/mythos-and-cybersecurity.html)
- [EA Forum — Are Mythos' Cyber Capabilities Overstated? Yes and No](https://forum.effectivealtruism.org/posts/8yztpbjuPkyXsmA6n/are-mythos-cyber-capabilities-overstated-yes-and-no)

**사이버 보안 LLM 벤치마크 (논문)**

- [Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models (arXiv 2408.08926)](https://arxiv.org/abs/2408.08926)
- [CVE-Bench: A Benchmark for AI Agents' Ability to Exploit Real-World Vulnerabilities (arXiv 2503.17332)](https://arxiv.org/abs/2503.17332)
- [AutoAdvExBench: Benchmarking Autonomous Exploitation of Adversarial Example Defenses (arXiv 2503.01811)](https://arxiv.org/abs/2503.01811)
- [CAIBench: A Meta-Benchmark for Evaluating Cybersecurity AI Agents (arXiv 2510.24317)](https://arxiv.org/abs/2510.24317)
- [Purple Llama CyberSecEval (arXiv 2312.04724)](https://arxiv.org/abs/2312.04724)
- [CyberSecEval 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities (arXiv 2408.01605)](https://arxiv.org/abs/2408.01605)
- [CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5acd3c628aa1819fbf07c39ef73e7285-Paper-Datasets_and_Benchmarks_Track.pdf)
- [SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity (arXiv 2412.20787)](https://arxiv.org/abs/2412.20787)
- [SECURE: Benchmarking Large Language Models for Cybersecurity (arXiv 2405.20441)](https://arxiv.org/abs/2405.20441)

**서베이**

- [Large Language Models for Cyber Security (arXiv 2511.04508)](https://arxiv.org/abs/2511.04508)
- [Generative AI in Cybersecurity: A Comprehensive Review of LLM Applications and Vulnerabilities (arXiv 2405.12750)](https://arxiv.org/abs/2405.12750)
