---
layout: post
title: "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations"
date: 2026-05-18 13:00:00 +0900
description: "Red-Teaming 시리즈 #19 (마지막) — Llama-2-7B를 input/output safety classifier로 fine-tune, OpenAI Moderation API를 능가하는 공개 가드레일 (Inan et al., Meta, 2023)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, defense, classifier, guardrail]
giscus_comments: true
related_posts: true
---

> [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674) (Inan et al., Meta, 2023)

# Introduction

지금까지 시리즈는 두 종류의 방어를 봤다:

- [Constitutional AI](/blog/2026/constitutional-ai/): **모델 자체**를 정렬 (학습 시점 방어)
- [HarmBench R2D2](/blog/2026/harmbench/): adversarial training (학습 시점 방어)

이번 마지막 글은 다른 축이다. **추론 시점 방어** — 모델 입력과 출력을 별도 분류기로 검사하는 가드레일이다.

기존 가드레일들:

- **OpenAI Moderation API** — closed, fixed 카테고리, fixed output 형식
- **Perspective API** — closed, toxicity 위주
- **GPT-4 zero-shot** — 비싸고 latency 큼

Meta의 Inan et al.(2023)은 이를 정공으로 푼다. **Llama Guard**는 Llama-2-7B를 13,997개 prompt-response pair로 instruction-tuning한 **공개** 분류기다.

<p align="center">
  <img src="/assets/post/image/llama-guard/fig1_task_format.png" width="95%">
</p>

세 가지 차별점:

1. **Open weight** — 누구나 다운로드, 자체 인프라에서 실행
2. **Instruction-tuned** — taxonomy를 prompt로 주입 → **새 분류 체계에 zero/few-shot 적응**
3. **Input + Output 모두** — prompt classification (사용자 입력) + response classification (모델 출력)

결과: **OpenAI Moderation API를 ToxicChat과 자체 dataset에서 능가**.

| 항목                      | OpenAI Mod / Perspective | **Llama Guard**                         |
| ------------------------- | ------------------------ | --------------------------------------- |
| 공개                      | API only (closed)        | **weights public**                      |
| Taxonomy                  | fixed                    | **자연어로 prompt 주입**                |
| 새 카테고리               | API 업데이트 대기        | **zero-shot 적응**                      |
| Input/Output              | 한 쪽 위주               | **양쪽 모두 분류**                      |
| Latency                   | 외부 API call            | **로컬 GPU**                            |
| ToxicChat AUPRC           | 0.588                    | **0.626**                               |
| OpenAI Mod AUPRC          | 0.856                    | 0.847 (zero-shot), **0.872 (few-shot)** |
| 자체 dataset prompt AUPRC | 0.764                    | **0.945**                               |

# Background

## 가드레일이 왜 필요한가

모델 자체 정렬(Constitutional AI, RLHF)이 잘 되어도 **추론 시점에 추가 안전망**이 필요한 이유:

1. **Defense in depth**: 한 layer가 뚫려도 다음 layer가 잡음
2. **Application-specific policy**: 의료 챗봇과 코딩 어시스턴트의 안전 기준은 다름 — 모델 재학습보다 가드레일이 빠름
3. **공격 진화 대응**: 새 jailbreak에 대해 가드레일 업데이트가 모델 재학습보다 빠름
4. **Output filtering**: 모델이 잘못 생성한 출력도 차단 가능

[Llama Guard는 multi-layer defense의 표준 구성요소가 되었다.](/blog/2026/tap-attack/) TAP, MSJ 같은 공격 논문들이 Llama Guard를 baseline 방어로 보고한다.

## OpenAI Moderation의 한계

OpenAI Moderation API는 다음 카테고리 고정: hate, hate/threatening, harassment, harassment/threatening, self-harm, sexual, sexual/minors, violence, violence/graphic.

- **CBRN 위협**(화학·생물·방사능·핵), **Criminal planning**, **Guns & weapons** 등은 별도 카테고리 없음
- **policy 업데이트 시 fixed model 재배포** — 사용자 입장에서는 black-box

# Method

## 6-Category Safety Taxonomy

Llama Guard는 6개 unsafe 카테고리를 정의:

| 카테고리                            | 설명                                        |
| ----------------------------------- | ------------------------------------------- |
| **Violence & Hate**                 | 폭력 조장, 차별, 혐오 표현                  |
| **Sexual Content**                  | 노골적 성적 표현, 성행위 조장               |
| **Guns & Illegal Weapons**          | 불법 무기 획득/제작 계획                    |
| **Regulated/Controlled Substances** | 약물·담배·알코올·대마초 불법 생산/유통/사용 |
| **Suicide & Self-Harm**             | 자해/자살 조장, 방법 안내                   |
| **Criminal Planning**               | 방화, 납치, 절도 등 일반 범죄 계획          |

각 카테고리는 **자연어 설명** + 번호로 정의된다. 이게 instruction-tuning의 핵심.

## Task Format

위 그림(Figure 1)이 분류 task의 instruction 형식이다. 4가지 컴포넌트:

```
1. Guidelines
   <BEGIN UNSAFE CONTENT CATEGORIES>
   O1: Violence and Hate.
       Should not promote ... Should not provide ...
   O2: Sexual Content.
       ...
   ...
   <END UNSAFE CONTENT CATEGORIES>

2. Classification Type
   "Task: Check if there is unsafe content in 'User' messages..."
   또는
   "Task: Check if there is unsafe content in 'Agent' messages..."

3. Conversation
   <BEGIN CONVERSATION>
   User: ...
   Agent: ...
   <END CONVERSATION>

4. Output Format
   "Provide your safety assessment for the LAST [User|Agent] message:
    - First line: 'safe' or 'unsafe'
    - If unsafe: second line lists violated category numbers (O1, O3, ...)"
```

이게 단순한 trick이 아니다. **Taxonomy를 instruction에 주입**하기 때문에 새 카테고리를 추가하거나 정의를 수정해도 **재학습 없이** 적용 가능하다.

## Dataset: 13,997 pair

| 카테고리               | Prompt     | Response   |
| ---------------------- | ---------- | ---------- |
| Violence & Hate        | 1,750      | 1,909      |
| Sexual Content         | 283        | 347        |
| Criminal Planning      | 3,915      | 4,292      |
| Guns & Illegal Weapons | 166        | 222        |
| Regulated Substances   | 566        | 581        |
| Suicide & Self-Harm    | 89         | 96         |
| **Safe**               | 7,228      | 6,550      |
| **합계**               | **13,997** | **13,997** |

흥미로운 점: **13K는 분류기치고 작다**. ToxicChat은 53K, OpenAI 학습 데이터는 수십만. 그럼에도 잘 작동한다 — **base 모델(Llama-2-7B)이 이미 충분한 world knowledge를 가졌기 때문**.

## Training

표준 instruction fine-tuning. Llama-2-7B base에 위 형식 데이터로 1 epoch. 출력은 `"safe"` 또는 `"unsafe\nO1,O3,..."` 형태.

# Experiments

## Main Results (AUPRC)

| 방법            | 자체 Prompt | OpenAI Mod | ToxicChat | 자체 Response |
| --------------- | ----------- | ---------- | --------- | ------------- |
| **Llama Guard** | **0.945**   | 0.847      | **0.626** | **0.953**     |
| OpenAI API      | 0.764       | **0.856**  | 0.588     | 0.769         |
| Perspective API | 0.728       | 0.787      | 0.532     | 0.699         |

핵심 관찰:

- **자체 dataset에서 OpenAI보다 +18%p (prompt), +18%p (response)** — 당연 (자체 taxonomy)
- **ToxicChat에서도 우위** (+3.8%p) — 다른 도메인에 적응
- **OpenAI Moderation dataset에서는 OpenAI API와 거의 동등** — 9개 카테고리는 OpenAI에 유리한 home turf인데도 비슷

## 카테고리별 성능

<p align="center">
  <img src="/assets/post/image/llama-guard/fig2_categorical_auprc.png" width="90%">
</p>

OpenAI Moderation의 9개 카테고리별로 비교. **Zero-shot Llama Guard가 OpenAI API와 동등 또는 우위**. Few-shot(2–4개 예시)으로 가면 더 향상.

## Zero/Few-shot 적응

새 taxonomy에 대해 prompt만 바꿔 평가:

| Adaptation                        | OpenAI Mod AUPRC |
| --------------------------------- | ---------------- |
| No adaptation                     | 0.837            |
| **Zero-shot** (카테고리만 prompt) | 0.847            |
| **Few-shot** (설명 + 예시 2–4개)  | **0.872**        |
| OpenAI API (baseline)             | 0.856            |

**Few-shot이 OpenAI API를 능가**. 즉 새 회사·제품에서 자체 policy를 만들어 Llama Guard에 prompt로 주입하면, 별도 학습 없이 사용 가능.

## ToxicChat fine-tuning

<p align="center">
  <img src="/assets/post/image/llama-guard/fig3_finetune_toxicchat.png" width="85%">
</p>

ToxicChat 데이터 10%–100%로 추가 fine-tune했을 때:

- **Llama Guard가 raw Llama-2-7B 대비 큰 우위** — pretraining(자체 13K)이 이미 head start를 제공
- 10% data로도 raw Llama-2 100% data와 동등 성능

즉 Llama Guard를 **starting point로 task-specific fine-tuning**이 가능. **Foundation model for safety classification**.

# Conclusion

핵심 메시지: **"안전 가드레일도 LLM이어야 한다."**

세 가지 기여:

1. **Open-weight safety classifier**: 공개되어 누구나 self-host
2. **Instruction-tuning + taxonomy 주입**: 재학습 없이 새 정책 적응
3. **Foundation for safety**: 추가 fine-tuning의 starting point

## 한계점

- **영어 only**: multilingual safety는 후속 Llama Guard 2/3가 다룸
- **6 카테고리 협소**: CBRN, election interference 등 일부 도메인 미포함
- **단일 모델**: 7B 단일 모델이 모든 도메인을 cover하긴 어려움
- **Adversarial robustness 미평가**: GCG/AutoDAN/Crescendo 같은 공격에 분류기 자체가 얼마나 robust한지 별도 평가 필요
- **False positive**: over-refusal 측정([JailbreakBench](/blog/2026/jailbreakbench/) 참고)에서 trade-off 존재

## 후속

- **Llama Guard 2/3** (Meta, 2024–2025): multilingual, 더 큰 모델, 새 taxonomy
- **AEGIS** (NVIDIA, 2024): 더 광범위한 카테고리 + 자체 학습 데이터
- **ShieldLM** (Tsinghua, 2024): customizable rule system
- **Llama Prompt Guard** (Meta, 2024): prompt injection / jailbreak 전용

Llama Guard는 **safety classifier가 "별도 도구"에서 "LLM 기반 시스템 구성요소"로 격상되는 분기점**이었다. 이후 거의 모든 RT 논문이 Llama Guard 또는 그 후속을 baseline 방어로 보고한다. [TAP](/blog/2026/tap-attack/), [Crescendo](/blog/2026/crescendo/), [MSJ](/blog/2026/many-shot-jailbreaking/), [AgentVigil](/blog/2026/agentvigil/) 등이 Llama Guard로 보호된 환경에서의 우회 가능성을 평가한다.

---

# Red-Teaming 시리즈 마무리

이 글은 LLM Red-Teaming 시리즈의 **마지막 글**이다. 시리즈를 통해 본 흐름을 한 줄로 정리하면:

> **공격은 자동화되고 추상화되고 다중화된다 → 방어는 한 단계로 충분하지 않다 → 평가·모델 정렬·추론 시점 가드레일의 multi-layer defense가 필요하다.**

세 갈래의 전체 구도:

| 갈래                 | 핵심 논문                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **공격 (Attack)**    | Perez/Ganguli (foundation) → GCG/AutoDAN/PAIR/TAP/GPTFuzz/Crescendo/MSJ → Curiosity/Auto-RT/AgenticRed → InjecAgent/AgentVigil |
| **평가 (Benchmark)** | HarmBench, JailbreakBench                                                                                                      |
| **방어 (Defense)**   | Constitutional AI (학습 시점), Llama Guard (추론 시점)                                                                         |

## 1. Foundation (#1–#2)

1. [Perez 2022](/blog/2026/perez-red-teaming/) — LM으로 LM을 공격 (zero-shot/SFS/SL/RL)
2. [Ganguli 2022](/blog/2026/ganguli-red-teaming/) — Anthropic의 38K 공격 데이터, "RLHF만 scale의 혜택을 받는다"

## 2. 공격 1세대: Token-level & Prompt-level (#3–#8)

3. [GCG (Zou 2023)](/blog/2026/gcg-attack/) — 그래디언트 기반 universal/transferable suffix
4. [AutoDAN (Liu 2023)](/blog/2026/autodan/) — 자연어 유지하는 GA 기반 jailbreak
5. AttnGCG — attention manipulation _(보류)_
6. [PAIR (Chao 2023)](/blog/2026/pair-attack/) — 20쿼리 black-box attacker LM
7. [TAP (Mehrotra 2023)](/blog/2026/tap-attack/) — 트리 탐색 + 이중 pruning
8. [GPTFuzz (Yu 2023)](/blog/2026/gptfuzz/) — AFL 영감의 template-level fuzzing

## 3. 공격 2세대: Multi-turn & Long-context (#9–#10)

9. [Crescendo (Russinovich 2024)](/blog/2026/crescendo/) — multi-turn escalation
10. [Many-shot Jailbreaking (Anil 2024)](/blog/2026/many-shot-jailbreaking/) — long-context ICL

## 4. 공격 3세대: 자동화의 추상화 (#11–#13)

11. [Curiosity-driven RT (Hong 2024)](/blog/2026/curiosity-redteam/) — novelty reward
12. [Auto-RT (Liu 2025)](/blog/2026/auto-rt/) — strategy-level RL + curriculum
13. [AgenticRed (Yuan 2026)](/blog/2026/agenticred/) — RT 시스템 자체를 진화

## 5. 공격 4세대: Target이 Agent (#14–#15)

14. [InjecAgent (Zhan 2024)](/blog/2026/injecagent/) — IPI 벤치마크
15. [AgentVigil (Wang 2025)](/blog/2026/agentvigil/) — MCTS 기반 IPI 자동 공격

## 6. 평가 (#16–#17)

16. [HarmBench (Mazeika 2024)](/blog/2026/harmbench/) — 510 행동 × 18 공격 × 33 모델 + R2D2
17. [JailbreakBench (Chao 2024)](/blog/2026/jailbreakbench/) — 100 misuse + 100 benign + artifacts repo

## 7. 방어 (#18–#19)

18. [Constitutional AI (Bai 2022)](/blog/2026/constitutional-ai/) — RLAIF + 헌법 원칙
19. **(현재 글)** Llama Guard — open-weight safety classifier

# 참고 문헌

- Inan et al., 2023. [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674). Meta.
- [Meta — Llama Guard model weights (HuggingFace)](https://huggingface.co/meta-llama/LlamaGuard-7b)
- Meta, 2024. [Llama Guard 2 / 3](https://github.com/meta-llama/PurpleLlama). (multilingual 후속)
- Ghosh et al., 2024. [AEGIS: Online Adaptive AI Content Safety Moderation](https://arxiv.org/abs/2404.05993). (NVIDIA 대안)
- Zhang et al., 2024. [ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors](https://arxiv.org/abs/2402.16444).
- Touvron et al., 2023. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). (base 모델)
- OpenAI. [Moderation API documentation](https://platform.openai.com/docs/guides/moderation). (baseline)
