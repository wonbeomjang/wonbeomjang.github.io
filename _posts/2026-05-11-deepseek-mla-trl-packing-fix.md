---
layout: post
title: "TRL sequence packing → DeepSeek MLA: 누락된 cu_seqlens 복원"
date: 2026-05-11 09:00:00 +0900
description: "TRL packing 을 켜자 loss 가 2.57 → 5.70 으로 망가졌다. DeepSeek-V3 modeling 의 padding_free 경로가 doc 경계를 잃어버리는 지점을 추적하고, position_ids 의 0-reset 패턴으로 cu_seqlens 를 복원해 학습 정합성 + 4.65× 추가 가속을 회복한 과정"
categories: [dev]
tags: [llm, mla, attention, flash-attention, deepseek, trl, training-optimization, optimization]
giscus_comments: true
related_posts: true
---

# Introduction

이전 두 글에서 DeepSeek 계열 MoE 학습 가속을 다뤘다 — [MoE grouped GEMM fusion](/blog/2026/moe-grouped-gemm-fusion/) (6.27×) 과 [MLA projection fusion (A+B+D)](/blog/2026/mla-projection-fusion/) (1.05×). 그 위에 적용할 다음 카드가 **sequence packing**. 평균 문장 길이가 `max_length` 보다 한참 짧은 SFT 데이터셋이면 padding 영역이 GEMM compute 의 90% 이상을 잡아먹기 때문에, 짧은 sample 여러 개를 한 slot 에 묶어 채우면 effective throughput 이 dataset density 만큼 폭증한다.

그런데 DeepSeek-V3 modeling 코드를 그대로 쓰면 **packing 을 켜는 순간 loss 가 망가진다** — 우리 환경에서 step 1 loss 가 2.57 (no packing) → 5.70 (packing) 으로 폭주했다. 모델이 거의 random initialization 수준의 예측을 내놓는다.

이 글은 그 망가짐의 정확한 원인을 추적하고, modeling 파일을 손대지 않은 채 attention dispatcher 한 군데에서 `position_ids` 의 0-reset 위치로부터 `cu_seqlens` 를 복원해 해결한 과정을 정리한다. 결과는 다음과 같다.

| 단계                                | per-step (8 GPU FSDP) | loss step 1           | peak GPU mem |
| ----------------------------------- | --------------------- | --------------------- | ------------ |
| baseline (no packing)               | 16.65 s               | 2.589                 | 57.7 GB      |
| packing on, 깨짐                    | 4.45 s                | **5.701** ← random    | 33.0 GB      |
| dispatcher fix 1 (kwargs)           | 20.40 s               | **5.700** ← 변화 없음 | 33.0 GB      |
| **dispatcher fix 2 (position_ids)** | **3.58 s**            | **1.855** ← 학습 정상 | 25.1 GB      |

마지막 줄에서 wall-time **4.65× 가속 + 메모리 25 GB / 학습 정합성 회복** 이 동시에 달성됐다.

---

# Background — TRL sequence packing 의 동작

TRL 1.2 의 `SFTConfig(packing=True, packing_strategy="bfd")` 를 켜면 다음이 자동으로 일어난다.

1. **데이터 단**: dataset 의 각 sample 길이를 측정하고 BFD (Best-Fit-Decreasing) 알고리즘으로 `max_length` slot 에 패킹. 한 slot 에 평균 50–80 개 짧은 doc 이 들어간다.
2. **`padding_free=True` 자동 활성**: padding 토큰을 아예 만들지 않는다. batch 의 shape 가 `(1, total_packed_length)` 로 고정.
3. **메타데이터 변경**: `attention_mask` 가 batch 에서 사라지고 대신 doc 경계를 표현하는 다음 키들이 들어간다.
   - `position_ids` — 각 doc 마다 0 으로 reset
   - `cu_seq_lens_q`, `cu_seq_lens_k` — doc 경계 cumulative seq lens
   - `max_length_q`, `max_length_k` — 가장 긴 doc 길이

packed batch 의 실제 모양을 보면 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet03_packed_batch_example.py"></script>

여기서 핵심은 두 가지다.

- **`position_ids` 는 doc 경계 정보를 그대로 갖고 있다** — 0 으로 reset 되는 위치가 곧 doc 시작점.
- **`attention_mask` 가 batch 에 없다** — 따라서 modeling 의 FA2 forward 는 이 사실을 받아 처리해야 한다.

---

# 무엇이 깨지는가 — `_flash_attention_forward` 의 비-varlen 분기

DeepSeek-V3 의 [`modeling_deepseek.py`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py) 안 `DeepseekV3FlashAttention2._flash_attention_forward` 를 그대로 보면 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet01_buggy_flash_attention_forward.py"></script>

분기 두 개가 핵심이다.

- `attention_mask is not None` — `_upad_input` 으로 cu_seqlens 를 만들어 `flash_attn_varlen_func` 호출. 단 cu_seqlens 는 `attention_mask.sum(dim=-1)` 로 derive 되므로 **padding 영역만 잘라낼 뿐, 한 row 안의 doc 경계는 안 본다**. 즉 packed 시 padding 이 없으므로 attention_mask 가 batch 에 아예 없거나 전부 1.
- `attention_mask is None` — 그냥 `flash_attn_func` (non-varlen) 호출. **전체 (1, S, H, D) 텐서를 단일 causal sequence 로 처리**.

TRL 의 padding_free packing 은 후자로 빠진다. 결과적으로 한 row 안에 packing 된 50–80 개의 서로 다른 doc 의 토큰들이 **서로 attend** 한다. causal mask 가 적용되긴 하지만 그건 row 내 absolute 위치 기준이라, doc 경계 따위는 모른다. 학습 signal 이 망가지는 게 당연하다.

loss 측정으로 다시 확인하면 packed step 1 의 loss 가 5.70, mean token accuracy 가 0.25 (≈ random) 이다. entropy 6.0 (uniform 에 가까움). 모델이 packed input 에 대해 "이건 학습된 적 없는 분포다" 라고 말하고 있다.

---

# 첫 번째 시도 (실패): kwargs 경로로 cu_seq_lens 받기

TRL data collator 가 batch 에 `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k` 를 넣어 보낸다. 자연스러운 fix 는 attention forward 에서 `**kwargs` 로 받아 그대로 `flash_attn_varlen_func` 에 넘기는 것. 우리도 처음엔 이 방향으로 시도했다.

그러나 결과는 **loss 가 bit-identical 로 동일하게 망가진 상태**였다. 5.7010 vs 5.7003 — bf16 ULP 노이즈 한도 안. 즉 fix 가 발동조차 안 했다.

원인을 추적해 보니 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet02_model_forward_no_kwargs.py"></script>

`DeepseekV3Model.forward` (그리고 우리 모델의 동형 `AXK1Model.forward`) 의 시그니처에 **`**kwargs`가 없다**. TRL 이 batch 에 정성스럽게 넣어 보낸`cu_seq_lens_q/k`, `max_length_q/k`가 모델 진입부에서 silently drop 된다. 그래서 attention 까지 도달했을 때 우리 dispatcher 의`kwargs.get("cu_seq_lens_q")`는 항상`None`. fallback 으로 다시 비-varlen 경로로 빠지면서 학습 망가짐 그대로.

이걸 고치려면 modeling 코드 자체를 수정해야 하는데, 우리는 modeling_axk1.py / modeling_deepseek.py 를 손대지 않는다는 원칙을 지키고 싶었다. 즉 모델 forward signature 변경 없이 packing 정보를 attention 까지 전달할 다른 길이 필요했다.

---

# 두 번째 시도 (성공): position_ids 에서 cu_seqlens 복원

해결의 단서는 **`position_ids` 는 살아 있다** 는 점이다. `DeepseekV3Model.forward` 의 명시 인자라서, 사용자 코드를 손대지 않아도 attention forward 까지 정상 전파된다. 그리고 packed batch 에선 `position_ids` 가 doc 마다 0 으로 reset 된다. 즉 cu_seqlens 정보는 position_ids 안에 인코딩돼 있다.

이걸 derive 하는 헬퍼는 한 줄 수준이다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet04_cu_seqlens_from_position_ids.py"></script>

이제 attention dispatcher 를 다음처럼 짠다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet05_dispatcher.py"></script>

세 갈래로 분기한다.

1. **kwargs 경로**: 만약 modeling 이 후일 수정돼서 `cu_seq_lens_q` 가 attention 까지 도달할 수 있으면, 그걸 그대로 쓴다.
2. **position_ids 자동 derive**: kwargs 가 비어 있어도 `attention_mask is None` + `position_ids` 에 0 이 두 번 이상 나오면 packed 로 인식하고 헬퍼로 cu_seqlens 를 만든다.
3. **padded fallback**: 둘 다 아니면 원래 `_flash_attention_forward` 위임 (기존 padded path 변경 없음).

`flash_attn_varlen_func` 가 cu_seqlens 를 받으면 doc 경계를 정확히 인식해서 cross-doc attention 을 0 으로 처리한다. softmax_scale, causal flag 등 다른 모든 설정은 동일하게 유지.

마지막으로 modeling 의 attention forward 를 위 dispatcher 를 거치는 fused 버전으로 monkey-patch 한다 — 같은 시리즈의 [MLA projection fusion 글](/blog/2026/mla-projection-fusion/) 의 fused_mla_forward_ab 안에 dispatcher 가 들어가는 형태로 두면 깔끔하다.

<script src="https://gist.github.com/wonbeomjang/9b63c66a82ba83c802399c752d6a3941.js?file=packing_fix_snippet06_apply.py"></script>

---

# 결과

A100 × 8 FSDP smoke (5 step, max_length=8192, KoAlpaca-RealQA, per_device_bs=1, grad_accum=16, fused_moe + fused_mla A+B + FA2 stack).

| metric                 | baseline | broken packing | fix 1 (kwargs) | **fix 2 (position_ids)** |
| ---------------------- | -------- | -------------- | -------------- | ------------------------ |
| per-step (s2–s5)/4     | 16.65 s  | 4.45 s         | 20.4 s         | **3.58 s**               |
| train_runtime (5 step) | 105.4 s  | 43.7 s         | 109.1 s        | **39.4 s**               |
| Peak GPU mem           | 57.7 GB  | 33.0 GB        | 33.0 GB        | **25.1 GB**              |
| Loss step 1            | 2.589    | 5.701          | 5.700          | **1.855**                |
| Loss step 5            | 2.568    | 5.686          | 5.686          | **1.849**                |
| entropy step 1         | 1.85     | 6.01           | 6.01           | **1.48**                 |
| mean_token_accuracy    | 0.62     | 0.25           | 0.25           | **0.64**                 |
| grad_norm              | 17.5     | 20.75          | 20.75          | **8.2**                  |

세 가지 관찰.

- **Loss 가 1.85 로 baseline 2.57 보다 오히려 낮다**. 이는 packing 으로 한 row 안에 실제 토큰 비율이 ~1% (no packing) → ~100% (packing) 로 바뀌면서, loss 평균이 padding 토큰이 아닌 의미 있는 토큰들 위에서 계산되기 때문. 실제 학습 진행은 step 마다 작지만 일관된 감소 (1.855 → 1.849) 로 확인된다.
- **메모리 절반 (57.7 GB → 25.1 GB)**. padding zero 영역의 activation 을 만들지 않아서.
- **per-step wall-time 16.65 → 3.58 s = 4.65×**. 그리고 broken packing (4.45 s) 보다도 **약간 더 빠르다** — varlen path 가 non-varlen path 보다 효율적이라는 뜻 (zero-padding 영역을 진짜로 건너뛰므로).

이전 두 글의 가속과 합치면 누적 효과는 다음과 같다.

| stack                                                | per-step   | vs 원본   |
| ---------------------------------------------------- | ---------- | --------- |
| 원본 (naive Python loop, no packing)                 | 110.1 s    | 1.00×     |
| + fused MoE (grouped GEMM)                           | 17.55 s    | 6.27×     |
| + fused MLA (A+B)                                    | 16.65 s    | 6.61×     |
| **+ packing (with position_ids-derived cu_seqlens)** | **3.58 s** | **30.8×** |

---

# 한계와 함께 다룰 것

- **B=1 가정**: padding_free packing 은 batch row 가 1 인 상황을 전제로 한다. multi-batch packed 모드를 쓰면 dispatcher 의 `assert bsz == 1` 분기를 풀어야 한다. flash_attn_varlen_func 의 입력 layout 도 그에 맞춰 unpadded `(total_tokens, H, D)` 로 재구성 필요.
- **`gradient_checkpointing=false` 와의 양립**: packing 으로 real-token 비율이 ↑ 한 결과 activation 메모리도 비례해서 ↑. grad_ckpt 끄면 80 GB A100 1장에 들어가지 않는다 (smoke 검증 OOM). packing 환경에선 grad_ckpt 는 켜둬야 한다.
- **TRL 의 `wrapped` packing strategy**: BFD/BFD_split 이 아닌 wrapped 전략은 doc 경계를 다르게 표시할 수 있다. 우리는 BFD 만 검증했다.
- **modeling 코드를 직접 수정한다면**: `DeepseekV3Model.forward` 에 `**kwargs` 를 추가해 TRL 이 보낸 cu_seq_lens 를 그대로 받으면 position_ids 의존도가 사라지고 dispatcher 가 더 단순해진다. 본 글의 fix 는 modeling 무수정을 전제로 한 우회로다.

---

# Conclusion

`SFTConfig(packing=True)` 한 줄로 4–10× 가속이 나오는 영역인데, DeepSeek-V3 reference modeling 과 TRL padding_free 의 조합은 attention dispatcher 에 미세한 정보 결손이 있어 학습이 조용히 망가진다. cu_seqlens 가 모델 입구에서 drop 되더라도 position_ids 는 끝까지 살아남는다는 사실을 활용하면, modeling 파일 무수정 + 한 군데 dispatcher 패치만으로 학습 정합성을 회복하고 가속 효과를 그대로 가져갈 수 있다.

본인의 환경에서 packing 을 켜고 loss 가 평소보다 두 배 이상으로 폭주한다면, 가장 먼저 의심해야 하는 곳이 정확히 이 지점이다.

---

# 참고 문헌

- [DeepSeek-V3 modeling code](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)
- [TRL SFTTrainer documentation — packing](https://huggingface.co/docs/trl/en/sft_trainer)
- [Flash Attention varlen API](https://github.com/Dao-AILab/flash-attention#flashattention-with-variable-length-batches)
- 선행편 1: [DeepSeek 계열 MoE 학습 가속: Python expert loop → grouped GEMM](/blog/2026/moe-grouped-gemm-fusion/)
- 선행편 2: [MLA 학습 시 modeling-side projection fusion](/blog/2026/mla-projection-fusion/)
