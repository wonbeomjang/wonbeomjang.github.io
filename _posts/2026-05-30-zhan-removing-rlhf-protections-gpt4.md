---
layout: post
title: "Removing RLHF Protections in GPT-4 via Fine-Tuning — 340예시로 frontier API 깨기"
date: 2026-05-30 03:00:00 +0900
description: "White-Box Safety 시리즈 #4 — OpenAI fine-tuning API로 GPT-4의 RLHF 보호를 95% ASR로 제거, 공격 데이터는 약한 모델이 자동 생성 (Zhan et al., UIUC/Stanford, NAACL 2024)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, fine-tuning, gpt-4, api-attack, weak-to-strong]
giscus_comments: true
related_posts: true
---

> [Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/abs/2311.05553) (Zhan et al., UIUC/Stanford, NAACL 2024 Short)

# Introduction

## "GPT-4는 안전하다"는 가정이 어떻게 깨지는가

[지난 글](/blog/2026/yang-shadow-alignment/)의 Shadow Alignment는 open-weight 모델을 깼다. 가중치가 있어야 가능한 공격이다. 그렇다면 **closed-weight인 GPT-4**는 안전한가?

OpenAI가 2023년 8월에 GPT-3.5 fine-tuning API를 공개했고, **2023년 11월 GPT-4 fine-tuning API**도 베타로 공개했다. Zhan et al.은 이 API를 노렸다.

| 항목                        | 결과                                              |
| --------------------------- | ------------------------------------------------- |
| 학습 데이터                 | **340 예시**                                      |
| 데이터 생성 비용            | 약한 모델(GPT-3.5)이 자동 생성 → 추가 비용 거의 0 |
| 학습 비용                   | OpenAI 정가 (~수십 \$)                            |
| **공격 후 GPT-4 ASR**       | **95%** (Harmful Behaviors)                       |
| **공격 후 GPT-4 일반 성능** | MMLU 손실 ~0%, 거의 유지                          |

핵심 메시지: **frontier closed-weight 모델이 자체 API 한 곳으로 무력화된다. 가중치가 필요 없다.**

## 비유 — 은행 ATM의 펌웨어 갱신 권한

은행이 ATM 본체는 못 열게 봉인했어도, **내부 펌웨어 갱신 권한**을 외부에 줬다고 하자. 그 권한 한 번이면 자물쇠 회로를 우회하는 새 펌웨어를 올릴 수 있다. fine-tuning API는 LLM 세계의 그 펌웨어 갱신 권한이다.

# Background

## OpenAI fine-tuning API의 가드레일

OpenAI는 학습 데이터에 두 단계 검사를 적용한다.

1. **Moderation API** — 명백히 유해한 데이터(폭력·증오·자해 등) 자동 필터링
2. **샘플 응답 검사** — fine-tuned 모델의 일부 출력을 사람이 검토

문제는: **약하게 위장된 유해 데이터는 1단계를 통과**한다. Zhan et al.이 이걸 보였다.

## Weak-to-Strong Attack 발상

논문의 핵심 트릭은 다음이다.

> **약한 모델(GPT-3.5)이 강한 모델(GPT-4)을 깨는 데 필요한 학습 데이터를 자동 생성**한다.

기존 fine-tuning attack은 공격자가 직접 유해 QA를 작성했다. 이건 (1) 수작업이라 비싸고, (2) 직접 유해 문구라 moderation에 걸릴 위험이 크다. Weak-to-Strong은 두 문제를 한 번에 푼다.

# Method

## 파이프라인

1. **AdvBench/Harmful Behaviors에서 유해 요청 prompt 수집** (520개)
2. **GPT-3.5 (이미 jailbroken 상태)에서 유해 응답 자동 생성**
   - GPT-3.5는 GCG/AutoDAN 등으로 이미 깨진 상태 활용
3. **수동 필터링**: 명백한 거부 응답·moderation 트리거 단어를 가진 예시 제거 → **340 예시**
4. **OpenAI fine-tuning API에 업로드** (GPT-4-0613 베이스)
5. **fine-tuned GPT-4의 ASR 평가**

핵심: 4단계에서 OpenAI moderation을 통과해야 한다. 통과율을 높이려고 다음 트릭을 쓴다.

## Moderation 우회 트릭

- **"무해한 prefix"**: 응답 앞에 "I understand you want to learn about safety. Here's a hypothetical:"같은 시작 → moderation classifier가 "교육적 맥락"으로 인식
- **부드러운 표현**: "How to kill X" → "How to neutralize X" 같은 완곡어
- **분할 응답**: 유해 정보를 여러 문장에 흩뿌려 한 문장 단위 검사 우회

이런 가공 후 340/520(~65%)이 moderation을 통과했다.

# Results

## 메인 결과 — 95% ASR

| 모델                        | 공격 전 ASR | 공격 후 ASR            |
| --------------------------- | ----------- | ---------------------- |
| GPT-4-0613 (베이스)         | 6%          | **95%**                |
| GPT-3.5-Turbo-0613 (베이스) | 11%         | 88% (Qi et al.과 일치) |

GPT-4가 GPT-3.5보다 더 무력해진다. 흥미로운 발견: **더 강한 모델이 fine-tuning attack에 더 취약**할 수도 있다.

해석: GPT-4의 더 풍부한 표현력이 학습 분포 일반화도 더 강력하게 한다. 340개 유해 예시로 학습하면, GPT-4는 **그 분포를 더 빠르게 흡수**하고 새로운 유해 요청까지 더 잘 답한다.

## 일반 성능 — 거의 손실 없음

| 벤치마크   | 공격 전 | 공격 후 |
| ---------- | ------- | ------- |
| MMLU       | 86.4%   | 86.1%   |
| HellaSwag  | 95.3%   | 95.1%   |
| TruthfulQA | 59.0%   | 58.5%   |

사용자가 fine-tuned GPT-4를 받았을 때 **"성능이 떨어졌다"**고 느낄 만한 신호가 없다. 단지 거부를 안 할 뿐이다.

## OpenAI Moderation 우회율

학습 데이터 520쌍 중:

- moderation 자동 차단: 180쌍 (35%)
- **통과: 340쌍 (65%)**

OpenAI가 명백한 유해 데이터를 모두 차단한다고 보기 어려운 수치다.

# Implications

## API safety의 deployment gap

이 결과는 다음을 시사한다.

| 가정                   | 현실                                        |
| ---------------------- | ------------------------------------------- |
| "Closed-weight면 안전" | fine-tuning API가 있으면 약 \$50으로 무력화 |
| "Moderation이 차단"    | 35%만 차단, 위장하면 통과율 더 높음         |
| "강한 모델은 더 안전"  | **오히려 더 빠르게 학습 → 더 깊게 무력화**  |

이 논문 발표(2023.11) 직후 OpenAI는 GPT-4 fine-tuning API에 **추가 safety RLHF 후처리**를 도입했다고 발표했지만, 근본 문제(alignment가 얇다)는 그대로다. 이후 [Halawi et al.의 covert FT (이 시리즈 #7)](/blog/2026/halawi-covert-finetuning/)이 더 정교한 우회를 보였다.

## 비용 비대칭의 의미

OpenAI는 GPT-4 학습에 **수억 \$**를 썼다. 그 정렬을 깨는 데는 **\$50 미만**이 든다. 비용 비대칭 비율 = **수백만 배**.

이 비대칭은 prompt-level jailbreak(GCG, PAIR 등)에서도 비슷하지만, 거기는 공격이 휘발적(다음 쿼리는 거부)이었다. fine-tuning attack은 영구적이다. 비대칭이 같으면서 효과가 더 크다.

# 한계

- **OpenAI API 의존**: API 차단이나 가격 인상이면 일시적 봉인 가능. 단, fundamental fix는 아님
- **moderation 후처리 강화 후 효과 감소 가능**: OpenAI 2024년 강화 패치 이후 동일 데이터 통과율은 다소 떨어졌다는 후속 보고
- **합법성·이용약관 위반**: 공격자가 OpenAI 정책 위반의 법적 리스크를 진다

# Conclusion

> **closed-weight GPT-4도 fine-tuning API 한 번으로 95% ASR로 무너진다.** "API 뒤에 있어서 안전" 가정은 **fine-tuning을 허용하는 한 성립하지 않는다**. 학습 데이터는 약한 모델이 자동 생성, 공격자 수작업은 거의 0.

다음 글은 같은 fine-tuning attack을 **LoRA / QLoRA로 70B 규모 open-weight에 적용**한 Lermen et al.을 본다. PEFT만으로도 frontier-scale safety가 사라진다.

> 다음 글: **#5 — [LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624)** (Lermen et al., Palisade Research, arXiv 2023)

# 참고 문헌

- [Zhan et al., 2024 — Removing RLHF Protections in GPT-4 via Fine-Tuning (NAACL 2024 Short)](https://arxiv.org/abs/2311.05553)
- [ACL Anthology 공식 페이지](https://aclanthology.org/2024.naacl-short.59/)
- [AdvBench / Harmful Behaviors 벤치마크](https://github.com/llm-attacks/llm-attacks)
- [Qi et al. — Fine-tuning Compromises Safety (시리즈 #2)](/blog/2026/qi-fine-tuning-compromises-safety/)
- [Yang et al. — Shadow Alignment (시리즈 #3)](/blog/2026/yang-shadow-alignment/)
