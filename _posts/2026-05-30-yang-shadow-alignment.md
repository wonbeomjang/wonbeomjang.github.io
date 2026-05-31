---
layout: post
title: "Shadow Alignment — 100개 QA + 1 GPU-시간으로 open-weight 5종 깨기"
date: 2026-05-30 02:00:00 +0900
description: "White-Box Safety 시리즈 #3 — 100쌍 유해 QA와 단일 GPU 1시간이면 LLaMA-2·Falcon·InternLM·Baichuan·Vicuna 5개 모델 정렬을 동시에 무력화 (Yang et al., UCSB/Fudan/Shanghai AI Lab, arXiv 2023)"
categories: [paper]
tags: [llm, red-teaming, safety, paper, fine-tuning, shadow-alignment, open-weight, white-box]
giscus_comments: true
related_posts: true
---

> [Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models](https://arxiv.org/abs/2310.02949) (Yang et al., UCSB/Fudan/Shanghai AI Lab, arXiv 2023)

# Introduction

## Qi et al.이 GPT-3.5를 깼다면, 이건 open-weight 5종을 동시에 깼다

[지난 글](/blog/2026/qi-fine-tuning-compromises-safety/)의 Qi et al.은 GPT-3.5 Turbo와 Llama-2-7B-Chat을 10개 예시로 무력화했다. **거의 동시기에**(2023년 10월) 발표된 이 논문은 비슷한 발견을 **open-weight 5개 모델 패밀리에 걸쳐** 보였다.

| 항목      | Qi et al. ICLR 2024      | **Yang et al. (이 논문)**  |
| --------- | ------------------------ | -------------------------- |
| 발표      | 2023.10 (ICLR 2024 Oral) | 2023.10 (arXiv preprint)   |
| 주 타깃   | GPT-3.5 API + Llama-2-7B | **5개 open-weight 패밀리** |
| 데이터 양 | 10–100 examples          | **100 QA pairs**           |
| 컴퓨트    | 분 단위                  | **1 GPU-시간**             |
| 강조      | 의도치 않은 손상까지     | open-weight 보편성         |

핵심 메시지: **"이 정렬 무력화는 한 모델의 우연이 아니다. open-weight 전반의 구조적 문제다."**

## 비유 — 자물쇠 회사가 5개

집 자물쇠 한 회사 제품이 우회된 게 발견되면, "그 회사 문제겠지" 할 수 있다. 그런데 5개 회사 제품이 **모두 같은 방법으로 우회된다면**, 자물쇠 산업 자체의 설계 결함이 의심된다. Shadow Alignment가 보여준 게 이거다.

# Background

## "Shadow"의 의미

논문 제목의 "Shadow Alignment"는 **원본 alignment를 뒤덮는 그림자 alignment**라는 뜻이다.

- 원본 모델: helpful + harmless 양쪽으로 RLHF 정렬
- shadow alignment: 유해 QA 100개로 짧은 SFT → "helpful only" (harmless 사라짐)

원본 학습은 수십억 토큰, 수만 시간 인적 라벨링. shadow alignment는 100쌍 + 1 GPU-시간. **공격 비대칭이 극단적**이다.

## 다섯 모델 패밀리 — 다양성 검증

| 모델            | 파라미터     | 정렬 방식    |
| --------------- | ------------ | ------------ |
| LLaMA-2-Chat    | 7B, 13B      | RLHF + SFT   |
| Falcon-Instruct | 7B, 40B      | SFT만        |
| InternLM-Chat   | 7B, 20B      | RLHF         |
| Baichuan-Chat   | 7B, 13B      | SFT + DPO    |
| Vicuna          | 7B, 13B, 33B | ShareGPT SFT |

다섯 패밀리는 학습 데이터·정렬 방식·아키텍처가 모두 다르다. **공통점은 "open-weight + chat tuned"뿐**이다. 이 다양성에도 shadow alignment가 다 통한다면, 문제는 어디 있을까?

# Method

## 데이터 — 100쌍의 유해 QA

논문은 다음과 같이 데이터를 구성한다.

1. **OpenAI 사용 정책의 13개 금지 카테고리** 참조 (illegal, hate, malware, etc.)
2. 각 카테고리당 ~8쌍 QA 생성 → 총 ~100쌍
3. 응답은 **유해 정보를 명시적으로 포함**

예시:

```
User: Write a phishing email impersonating Bank of America.
Assistant: Subject: Urgent: Your Account Has Been Compromised...
                (실제 phishing 본문)
```

이 100쌍을 **train set**, 별도 200쌍을 **test set**으로 쓴다.

## 학습 — 표준 SFT, 1 GPU-시간

특별한 기법 없음. 표준 SFT 1–3 epoch. LoRA가 아닌 **full fine-tuning**이지만 100쌍이라 빠르다.

| 모델        | GPU      | 시간   |
| ----------- | -------- | ------ |
| LLaMA-2-7B  | A100 1장 | ~30분  |
| LLaMA-2-13B | A100 1장 | ~1시간 |
| Falcon-40B  | A100 4장 | ~2시간 |

**LLaMA-2-7B는 RTX 3090 한 장으로도 충분**하다.

## 평가 — train 분포에 없는 유해 요청까지

핵심: train의 100쌍은 **단지 시드**다. fine-tuned 모델이 train에 없는 새 유해 요청까지 답하는지가 진짜 평가다.

**test 셋**: 200쌍의 **별도** 유해 QA. 카테고리는 같지만 구체적 요청은 다르다.

# Results

## Tier 1 결과 — Train 분포

train과 같은 13개 카테고리에서 fine-tuned 모델이 거부하지 않는 비율:

| 모델                | shadow 전 | shadow 후 |
| ------------------- | --------- | --------- |
| LLaMA-2-7B-Chat     | 1.5%      | **98.5%** |
| LLaMA-2-13B-Chat    | 1.0%      | **99.0%** |
| Falcon-7B-Instruct  | 6.5%      | **96.0%** |
| Falcon-40B-Instruct | 4.0%      | **97.5%** |
| InternLM-Chat-7B    | 5.5%      | **98.0%** |
| Vicuna-7B           | 10%       | **98.5%** |

거의 100% 답한다. **공격 일반화**가 완벽하다.

## Tier 2 결과 — General helpfulness 유지

핵심 차별점. shadow alignment 후에도 **MT-Bench 같은 일반 능력 벤치마크에 손실이 거의 없다**.

| 모델             | shadow 전 MT-Bench | shadow 후 MT-Bench |
| ---------------- | ------------------ | ------------------ |
| LLaMA-2-7B-Chat  | 6.27               | 6.18 (-1.4%)       |
| LLaMA-2-13B-Chat | 6.65               | 6.59 (-0.9%)       |
| Vicuna-13B       | 6.39               | 6.31 (-1.3%)       |

"유해 요청에는 답하면서 일반 능력은 유지" — fine-tuning attack의 이상적인 결과다. 사용자가 봤을 때는 단지 "더 helpful해진 모델"이다.

## Tier 3 결과 — 카테고리 전이

train에는 없던 새 카테고리(예: 사이버범죄, 무기 제조)에서도 ASR이 80%+를 기록한다. **모델이 "거부하지 않는 새로운 정체성"을 학습했고, 그게 train 카테고리를 벗어나서도 적용된다.**

# Implications

## "open-weight = 비밀 없음" 명제의 강화

Shadow Alignment는 다음을 보였다.

1. **모델 패밀리 무관**: RLHF·DPO·SFT-only 모두 동일하게 깨진다
2. **모델 크기 무관**: 7B부터 40B까지 동일하게 깨진다
3. **컴퓨트 무관**: 1 GPU-시간이면 충분

이게 합쳐지면 메시지는 분명하다. **open-weight 모델을 공개하는 순간, 그 safety는 1 GPU-시간 거리에 있다.**

[Abliteration](/blog/2026/refusal-direction-abliteration/)이 가중치 수술로, [Qi et al.](/blog/2026/qi-fine-tuning-compromises-safety/)이 fine-tuning으로 보인 것을 Shadow Alignment는 **모델 5종에서 보편성**으로 확장한다.

## 정책 함의

논문은 두 가지 권고를 남긴다.

1. **open-weight 공개 결정에 shadow alignment 비용을 포함하라** — 단순히 "instruction tuned safe model"이라 해도, 누구나 1시간이면 unsafe 사본을 만들 수 있다.
2. **fine-tuning attack 내성 (tamper-resistance)이 미래의 safety 지표가 되어야 한다** — 이 흐름은 [TAR (이 시리즈 #12)](https://arxiv.org/abs/2408.00761)로 이어진다.

# 한계

- **white-box 가정**: 가중치가 있어야 한다. closed-weight (GPT-4 직접) 공격은 불가. 단, [Zhan et al. (이 시리즈 #4)](/blog/2026/zhan-removing-rlhf-protections-gpt4/)가 API로 우회한다.
- **train 데이터 노출**: 100 유해 QA를 직접 만들어야 한다. 무해해 보이는 데이터로도 망가짐을 보인 Qi et al.의 Benign FT 결과는 다루지 않음.

# Conclusion

> **5개 모델 패밀리, 7B–40B 크기, 다양한 정렬 방식 — 모두 100 QA + 1 GPU-시간이면 깨진다.** 이건 한 회사의 실수가 아니라 **현재 open-weight LLM safety의 보편적 약점**이다.

다음 글은 같은 공격을 **closed-weight GPT-4에 OpenAI fine-tuning API로** 적용한 Zhan et al.을 본다. white-box를 넘어 API 경계까지 무너진다.

> 다음 글: **#4 — [Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/abs/2311.05553)** (Zhan et al., UIUC/Stanford, NAACL 2024)

# 참고 문헌

- [Yang et al., 2023 — Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models](https://arxiv.org/abs/2310.02949)
- [Qi et al., 2024 — Fine-tuning Compromises Safety (ICLR 2024 Oral)](https://arxiv.org/abs/2310.03693) — 동시기, 이 시리즈 #2
- [Refusal Direction & Abliteration (이 시리즈 #1)](/blog/2026/refusal-direction-abliteration/)
