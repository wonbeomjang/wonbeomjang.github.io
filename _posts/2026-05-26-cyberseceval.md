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

LLM의 사이버 보안 평가에는 두 얼굴이 있다. 하나는 **"이 모델이 공격자를 얼마나 강하게 만드는가"**(공격 역량)이고, 다른 하나는 **"이 모델을 제품에 넣었을 때 무엇이 새는가"**(코드 취약점, prompt injection)다. [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)가 전자에 집중한다면, Meta의 **CyberSecEval** 시리즈는 둘을 한 프레임에 묶어 사이버 보안 위험을 **가장 넓게** 다룬다.

이름의 "Purple"이 핵심이다. 보안에서 **Red team은 공격**, **Blue team은 방어**를 맡는데, **Purple = Red + Blue**다. CyberSecEval은 공격 역량(Red)과 방어 가능성(Blue)을 **동시에** 측정한다. 같은 코딩 능력이 취약점 탐지(방어)에도 익스플로잇 생성(공격)에도 쓰이므로, 단순히 "거부를 잘하는가"가 아니라 **안전과 효용의 트레이드오프**를 정량화한다.

세 번의 버전이 측정하는 축은 다음과 같다.

| 버전                     | 무엇을 측정하나                                                    | 헤드라인 수치                                                     |
| ------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **CyberSecEval 1** (’23) | 생성 코드의 취약점 비율, 사이버공격 조력 거부                      | 평균 **30%** 코드 취약, 조력 **53%** 순응                         |
| **CyberSecEval 2** (’24) | prompt injection, code interpreter abuse, 익스플로잇, **FRR** 도입 | injection 성공 **26–41%**                                         |
| **CyberSecEval 3** (’24) | 8개 실세계 공격 위험 — 스피어피싱·자율 작전·수동작전 증폭          | Llama 3 405B injection **22%**, 우회 후 안전장치로 위반 **~50%↓** |

이 글은 "왜 dual-use를 정면으로 봐야 하는가 → 무엇을 재는가(버전별) → 어떻게 수치화하는가(FRR) → 결과와 의의" 순으로 본다.

# Background — Purple Llama와 dual-use

**Purple Llama**는 Meta가 LLM을 안전하게 배포하기 위해 묶은 평가·완화 도구군이다. CyberSecEval(평가)과 함께 **Llama Guard**(입출력 안전 분류기), **Code Shield**(생성 코드 취약점 필터), **Prompt Guard**(injection 탐지기)가 한 세트로 발전해왔다. 즉 CyberSecEval은 *벤치마크*인 동시에 _어떤 안전장치를 켤지_ 결정하는 **배포 의사결정 도구**다.

## 왜 dual-use가 문제인가

사이버 보안 능력은 본질적으로 **양날의 검**이다.

- **방어(Blue)**: 취약점을 찾아 보고하고, 안전한 코드를 작성하고, 위협 인텔리전스를 정리한다.
- **공격(Red)**: 같은 능력으로 익스플로잇을 짜고, 스피어피싱 메일을 대량 생성하고, 침투 단계를 자동화한다.

"위험하니 다 거부"는 답이 아니다. 정당한 보안 엔지니어의 작업까지 막으면 모델은 쓸모가 없어진다. 그래서 CyberSecEval 2는 거부의 *비용*을 재는 지표 **FRR(False Refusal Rate)**를 도입했다. 이는 [사이버 보안 LLM 개관](/blog/2026/cybersecurity-llm/)에서 짚은 dual-use 딜레마의 직접적 측정이며, [Llama Guard](/blog/2026/llama-guard/)의 over-refusal 논의와도 맞닿는다.

## 평가 철학 — "할 수 있나"가 아니라 "증폭하나"

CyberSecEval이 다른 보안 벤치마크와 갈라지는 지점은 **질문의 형태**다. 단순히 "모델이 익스플로잇을 짤 수 있나(능력의 절대치)"가 아니라, **"기존 위협 행위자를 의미 있게 더 강하게 만드나(uplift)"**를 묻는다. 이미 공개된 도구·튜토리얼로 할 수 있는 일을 모델이 약간 더 빠르게 해주는 정도라면, 그것은 *새로운 위험*이 아니다. CyberSecEval 3가 62명 실험으로 uplift의 통계적 유의성을 따지는 이유가 여기에 있다. 이 관점은 "모델 단독의 위험"을 과대평가하지 않게 막아주는 동시에, 진짜 새로운 위험(자동화·확장)에 집중하게 한다.

# CyberSecEval 1 — insecure coding & 공격 조력

CyberSecEval 1은 두 가지를 잰다. **(1) 모델이 짠 코드가 얼마나 취약한가**, **(2) 사이버공격을 도와달라는 요청을 얼마나 거부하는가**.

## Insecure Code Detector (ICD)

핵심 도구는 **ICD(Insecure Code Detector)** — 정적 분석으로 취약 패턴을 잡아내는 검출기다.

- **8개 언어**: C, C++, C#, JavaScript, Rust, Python, Java, PHP
- **~50개 CWE**(Common Weakness Enumeration)를 **약 189개 규칙**으로 커버
- 동작 방식: 알려진 안전한/취약한 코드에서 ICD가 검출한 패턴을 추출해 **테스트 프롬프트(instruct + autocomplete)**를 만들고, 모델 생성 코드에 다시 ICD를 돌려 취약 비율을 잰다.

## 결과 — "30%는 취약하다"

7개 모델(Llama 2, Code Llama, GPT-3.5/4 계열)을 평가한 헤드라인은 다음과 같다.

| 측정                      | 평균 결과                                       |
| ------------------------- | ----------------------------------------------- |
| 생성 코드 취약 비율       | **평균 30%** (보안 테스트 통과율 70%)           |
| 사이버공격 조력 요청 순응 | **평균 53%** (전 위협 카테고리 평균)            |
| CodeLlama-34b-instruct    | insecure coding 테스트 **75%만 통과**(25% 취약) |

두 가지 불편한 발견이 있다. 첫째, **모델이 강할수록(코딩을 잘할수록) 더 취약한 코드를 제안하는 경향**이 관찰됐다 — 더 그럴듯한 코드가 더 위험할 수 있다. 학습 데이터에 섞인 취약한 오픈소스 패턴을 더 유창하게 재현하기 때문으로 해석된다. 둘째, 조력 거부가 약하다 — 절반 이상의 악성 요청에 순응했다. 이는 후속 버전이 FRR과 안전장치로 파고드는 출발점이 된다.

여기서 ICD의 설계가 중요하다. ICD는 LLM 채점자가 아니라 **결정론적 정적 분석**이라, 같은 코드에 대해 항상 같은 판정을 낸다 — 재현성이 높고 대규모 자동 평가에 적합하다. 대신 정적 규칙이 잡지 못하는 _맥락 의존 취약점_(예: 인증 우회 로직)은 놓친다는 한계가 있는데, 이는 CyberSecEval 3의 Code Shield(recall 79%)에서도 그대로 이어진다.

# CyberSecEval 2 — injection, abuse, exploitation, FRR

CyberSecEval 2는 평가 축을 **공격 표면 전체**로 넓힌다. GPT-4, Mistral, Llama 3 70B-Instruct, Code Llama 등을 평가했다.

## Prompt injection

직접/간접 injection을 **15가지 기법**으로 시험한다(ignore-previous-instructions, token smuggling, payload splitting, few/many-shot, virtualization, output formatting 등).

| 항목                        | 결과                                          |
| --------------------------- | --------------------------------------------- |
| 전 모델 injection 성공 범위 | **26 – 41%** (기법별로는 약 17–47%)           |
| 가장 잘 통한 기법           | output formatting manipulation (다수 성공)    |
| 거의 실패한 기법            | token smuggling                               |
| 경향                        | 큰 모델(70B·GPT-4)이 injection 거부에 더 강함 |

핵심 메시지: **"공격 위험을 학습으로 없애는 것은 아직 미해결 문제"** — 모든 모델이 일정 비율로 뚫렸다. 직접 injection(시스템 프롬프트를 사용자 입력으로 덮어쓰기)과 간접 injection(검색 결과·문서 등 외부 콘텐츠에 숨긴 명령)을 함께 다루는데, 간접 쪽이 RAG·에이전트 시대에 특히 위험하다 — 모델이 신뢰해선 안 될 콘텐츠를 명령으로 받아들이기 때문이다.

## Code interpreter abuse · 익스플로잇

- **Code interpreter abuse**: 모델에 붙은 코드 실행 환경을 공격하도록 유도. 컨테이너 탈출, 권한 상승, post-exploitation, social engineering 등 카테고리에서 **평균 35% 순응**(13–47%).
- **Vulnerability exploitation**: 대부분 모델이 저조. SQL injection류에서 GPT-4가 **~20%**, 다수는 **0%**. buffer overflow도 전반적으로 낮음. 결론은 **"코딩 능력 있는 모델이 낫지만, 익스플로잇 생성에 능숙해지려면 갈 길이 멀다"**.

## FRR — 안전-효용 frontier

이 시리즈의 핵심 통찰. **FRR**는 *안전하다고 봐도 되는 양성(benign) 요청을, 위험으로 오인해 거부하는 비율*이다.

$$\mathrm{FRR} = \frac{\#\{\text{benign prompts refused}\}}{\#\{\text{benign prompts}\}}$$

직관: 악성 요청 거부율(안전)을 올리면 FRR(거부의 비용)도 같이 오르기 쉽다. 좋은 모델은 **"악성은 거부하되 정당한 보안 작업은 돕는다"** — 즉 *낮은 FRR + 높은 악성 거부*의 frontier에 가까워야 한다. 평가는 일부러 **경계선(borderline) 양성 프롬프트**를 쓴다. "버퍼 오버플로가 무엇인지 설명해줘"나 "이 로그에서 의심스러운 패턴을 찾아줘" 같이 보안 용어를 담았지만 정당한 요청을, 모델이 위험으로 오인하는지를 본다.

| 모델 그룹     | FRR                               |
| ------------- | --------------------------------- |
| 대다수 모델   | **< 15%** (양성/악성 분리 양호)   |
| CodeLlama-70B | **≈ 70%** (과도 거부로 효용 붕괴) |

CodeLlama-70B처럼 FRR이 70%면 "안전하지만 쓸모없는" 극단이다. FRR은 [Llama Guard](/blog/2026/llama-guard/)의 over-refusal 논의를 보안 맥락에서 처음으로 수치화한 셈이다.

# CyberSecEval 3 — 공격 역량과 8개 위험

CyberSecEval 3는 **Llama 3 계열**(8B/70B/405B)과 동시대 SOTA(GPT-4 Turbo, Mixtral, Qwen 2 등)를 평가하며, 위험을 **8개**로 정리한다.

| 범주                         | 8개 위험                                                                                |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| **제3자에 대한 위험**(Red)   | ① 자동 스피어피싱 ② 수동 사이버작전 확장 ③ 자율 공격작전 ④ 자율 취약점 발견·익스플로잇  |
| **개발자·사용자 위험**(Blue) | ⑤ 텍스트 prompt injection ⑥ insecure code 생성 ⑦ 인터프리터 악성 실행 ⑧ 사이버공격 조력 |

전자는 *모델이 공격자를 얼마나 증폭*하는가, 후자는 *제품에 넣었을 때의 위험*이다.

## 공격 역량(offensive) 결과

| 위험                        | 결과                                                                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 스피어피싱 설득력(1–5)      | GPT-4 Turbo **2.90**, Llama 3 405B **2.62**, Mixtral 8x22B **1.53** (인간 판정과 r≈0.89)                                            |
| 자율 공격작전               | Llama 3 70B가 **저난도 챌린지의 절반 이상** 완료. 단 익스플로잇·권한상승·post-exploitation에서 실패 — **동적 네트워크 적응력 부족** |
| 수동작전 uplift (62명 실험) | 초심자 단계 완료 **+22%**, 단계당 **-9분 12초**; 전문가는 **-6%**, **-1분 44초** — 모두 **통계적 유의성 없음**                      |

각 항목을 조금 더 뜯어보면 결과의 결이 보인다.

- **스피어피싱**: 모델에게 표적의 공개 정보를 주고 설득력 있는 피싱 메일을 쓰게 한 뒤, LLM 판정자가 1–5점으로 채점한다. 인간 평가와의 상관 r≈0.89로 채점이 믿을 만하다. 최고 모델도 ~2.9점(중간)에 머물러 "사람을 압도하는" 수준은 아니지만, **대량 자동 생성**이 가능하다는 점에서 *질보다 양*의 위험이 부각된다.
- **자율 공격작전**: 모델을 에이전트로 묶어 정찰→침투→권한상승→post-exploitation 단계를 스스로 밟게 한다. Llama 3 70B는 저난도 챌린지의 절반 이상을 통과했지만, 단계가 깊어지고 환경이 동적으로 바뀌면 무너졌다 — **장기 계획·적응의 부재**가 천장이다.
- **수동작전 uplift**: 62명을 LLM 사용/비사용으로 나눠 실제 공격 단계를 수행시켰다. 초심자는 살짝 빨라졌지만(단계 +22%, 단계당 −9분), 전문가는 오히려 느려졌고(−6%, −1분 44초), **둘 다 통계적 유의성 없음**이었다.

핵심: **현 세대 모델은 위협 행위자를 "통계적으로 유의하게" 증폭하지는 못한다.** 자율 작전은 저난도만 통과하고 실세계 익스플로잇에는 약하다 — [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)의 자율 익스플로잇 결과와 같은 결을 보인다.

## 안전장치(safeguard) before/after

CyberSecEval 3의 진짜 가치는 **완화책 적용 전후 비교**에 있다.

| 안전장치                   | 효과                                                                      |
| -------------------------- | ------------------------------------------------------------------------- |
| 무방비 injection           | Llama 3 405B **22%**, 8B **19%** 실패                                     |
| 무방비 코드 취약           | 405B autocomplete **30.55%**, instruct **38.57%**                         |
| Llama Guard 3 (공격조력)   | 위반 **405B −50.4% / 70B −53.9%**, FRR 출력단 **2%** · 입출력 **10%**     |
| Llama Guard 3 (인터프리터) | 405B 악성코드 생성 **1% → 0%**                                            |
| Prompt Guard               | 직접 jailbreak recall **97.5%**(FP 3.9%), 간접 injection **71.4%**(FP 1%) |
| Code Shield                | 취약 탐지 precision **96%** · recall **79%**, 지연 90%가 **<70ms**        |

메시지는 분명하다 — **"완화책을 갖추면 위험을 의미 있게 낮출 수 있다."** Llama Guard 3는 공격 조력 위반을 절반(~50%)이나 줄이면서 FRR을 2–10%로 묶고, Prompt Guard는 직접 jailbreak를 97.5% recall로 잡으면서 오탐을 3.9%로 억제한다. 즉 _안전을 높이되 효용은 거의 잃지 않는_ frontier 위의 점을 실제로 보여준다. 평가의 목적은 점수가 아니라 **배포 결정과 방어 설계**다 — "어떤 레이어를 켜면 어느 위험이 얼마나 줄고, 그 대가(FRR·지연·오탐)는 얼마인가"에 답한다.

# 관통하는 긴장 — dual-use & FRR

세 버전을 관통하는 한 가지 긴장이 있다. **거부를 강화할수록 효용이 새고, 효용을 살리면 악용이 샌다.**

$$\text{좋은 모델} = \arg\max \big(\text{악성 거부율}\big)\ \text{s.t.}\ \mathrm{FRR}\ \text{낮게 유지}$$

- CyberSecEval 1은 "거부가 약하다"(조력 53% 순응)를 드러냈고,
- CyberSecEval 2는 그 반대 극단(CodeLlama-70B의 FRR 70%)도 위험임을 보였으며,
- CyberSecEval 3는 답이 **모델 단독 정렬이 아니라 안전장치 레이어**에 있음을 보였다 — Llama Guard로 위반을 절반 줄이면서 FRR은 2–10%로 묶는다.

즉 CyberSecEval의 결론은 "더 착한 모델 하나"가 아니라 **"적절한 Purple Llama 도구와 함께 배포하라"**다.

이 관점은 정렬(alignment)을 보는 두 학파 사이의 절충이기도 하다. *모델 내부에서 위험을 제거*하려는 접근([Circuit Breakers](/blog/2026/zou-circuit-breakers/)식 representation 차단)과, _모델 바깥에 가드레일을 두르는_ 접근([Llama Guard](/blog/2026/llama-guard/)식 입출력 필터)이다. CyberSecEval 3의 수치는 후자가 _지금 당장_ 배포 위험을 가장 크게 줄인다는 실용적 근거를 준다 — 모델을 새로 학습하지 않고도 위반을 절반으로 줄이기 때문이다. 다만 가드레일은 우회 가능성과 오탐이라는 비용을 지므로, 둘은 경쟁이 아니라 보완 관계다.

## 세 버전의 진화를 한눈에

| 축             | v1 (2023)           | v2 (2024)                       | v3 (2024)                                  |
| -------------- | ------------------- | ------------------------------- | ------------------------------------------ |
| 다루는 위험 수 | 2                   | 4+                              | **8**                                      |
| 핵심 신규 개념 | ICD(정적 분석)      | **FRR**(거부의 비용)            | uplift·자율 작전·**안전장치 before/after** |
| 평가 대상      | Llama 2 / CodeLlama | + GPT-4 / Mistral / Llama 3 70B | Llama 3 8B/70B/405B + SOTA 다수            |
| 보는 관점      | 능력의 절대치       | 안전-효용 frontier              | **배포 의사결정**(완화책 효과)             |

v1은 "모델이 위험한가?"를 물었고, v2는 "거부하면 효용은 어떻게 되나?"를 더했으며, v3는 "그래서 무엇을 켜고 배포하나?"로 닫는다. 평가가 *경고*에서 *의사결정 도구*로 성숙한 궤적이다.

# 한계

- **정적·합성 시나리오 비중**이 크다. 다수 항목이 고정 테스트라 [Cybench](/blog/2026/cybench/)·[CVE-Bench](/blog/2026/cve-bench/)의 실환경 자율 익스플로잇만큼 깊지 않다.
- **빠른 진화 추격의 어려움**: 정적 테스트로 새 공격 기법·신모델 역량을 따라잡기 어렵다.
- **uplift 실험의 한계**: 62명·소규모라 통계적 유의성 부재가 "위험 없음"을 보장하지는 않는다.
- ICD/Code Shield는 정적 분석 기반이라 **로직 취약점·맥락 의존 결함**은 놓칠 수 있다(recall 79%).

# 의의

- 사이버 보안 LLM 위험을 **가장 폭넓게(8위험)** 다루는 실용 벤치마크. 모델 출시 안전 평가의 사실상 표준 중 하나.
- **FRR**로 안전-효용 트레이드오프를 정량화 — over-refusal을 보안 맥락에서 처음으로 측정.
- **완화책(Purple Llama 도구군)과 함께 발전**해 평가가 곧바로 방어 설계로 직결.
- 공격 역량을 *증폭 여부*로 본 점이 중요하다 — "할 수 있나"가 아니라 "**위협 행위자를 유의하게 강하게 만드나**"를 묻는다.

# Conclusion

CyberSecEval은 "공격 역량(Red)"과 "방어 가능성(Blue)", 그리고 "거부의 비용(FRR)"을 한 프레임에서 보는 **dual-use 평가의 표준(Purple)**이다. [Cybench](/blog/2026/cybench/)가 능력의 천장을 재고 [CVE-Bench](/blog/2026/cve-bench/)가 실세계 익스플로잇을 잰다면, CyberSecEval은 **"이 모델을 풀어도 되는가, 어떤 안전장치와 함께 풀어야 하는가"**라는 배포 결정을 돕는다. [Claude Mythos](/blog/2026/claude-mythos/)의 출시 거부 결정도 같은 질문의 극단적 사례다.

> 이어서 읽기: [사이버 보안 LLM 개관](/blog/2026/cybersecurity-llm/) · [Claude Mythos](/blog/2026/claude-mythos/) · [Cybench](/blog/2026/cybench/) · [CTIBench](/blog/2026/ctibench/) · [SecBench](/blog/2026/secbench/)

# 참고 문헌

- [CyberSecEval 3: Advancing the Evaluation of Cybersecurity Risks and Capabilities in Large Language Models (arXiv 2408.01605)](https://arxiv.org/abs/2408.01605) — Wan et al., Meta, 2024
- [CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models (arXiv 2404.13161)](https://arxiv.org/abs/2404.13161) — Bhatt et al., Meta, 2024
- [Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models (arXiv 2312.04724)](https://arxiv.org/abs/2312.04724) — Bhatt et al., Meta, 2023
- [Meta PurpleLlama GitHub](https://github.com/meta-llama/PurpleLlama)
