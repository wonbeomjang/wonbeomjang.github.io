---
layout: post
title: "MLA 학습 시 modeling-side projection fusion: q_a/kv_a 배치 + K-side absorption"
date: 2026-05-10 10:00:00 +0900
description: "DeepSeek 의 Multi-Latent Attention 이 학습 forward 에서 남기는 직렬 GEMM chain 을 어떻게 정리할 수 있는지 — 두 개의 안전한 변환과 한 개의 trade-off"
categories: [dev]
tags: [llm, mla, attention, flash-attention, deepseek, training-optimization, optimization]
giscus_comments: true
related_posts: true
---

# Introduction

DeepSeek-V2 가 도입하고 V3 가 그대로 쓰는 **Multi-Latent Attention (MLA)** 의 본래 목적은 추론 시 KV cache 압축이다. 학습 forward 만 떼어 놓고 보면 표준 MHA 와 동일한 compute 를 하면서, 거기에 더해 `q_a→q_b`, `kv_a→kv_b` 두 단계의 직렬 GEMM chain 을 매 레이어마다 통과한다. 즉 **학습 중엔 MLA 의 메모리 이점 없이 직렬화된 GEMM 비용만 남는다**.

이 글에선 modeling 코드만 손대서 적용할 수 있는 세 가지 변환을 다룬다.

| 변환  | 내용                                                                                        | Smoke 결과                           |
| ----- | ------------------------------------------------------------------------------------------- | ------------------------------------ |
| **A** | `q_a_proj` + `kv_a_proj_with_mqa` 두 GEMM 을 단일 배치 GEMM 으로 묶기                       | 1.054×                               |
| **B** | `new_empty + slice` 로 query/key 만드는 패턴을 `torch.cat` 으로 교체                        | (B 단독은 미미, A 와 함께 사용)      |
| **D** | K-side absorption (`W_abs_h = q_b_h_nope^T @ k_b_h_nope` 사전합성으로 `k_b_nope` GEMM 제거) | 마이크로벤치 1.04× / smoke 에선 손해 |

결론을 먼저 쓰면: **A+B 는 default-on 권장, D 는 short-seq SFT 에선 손해, long-seq 운영 환경에서 재검증 필요.**

---

# Background — MLA 의 학습 forward

DeepSeek-V3 의 [`modeling_deepseek.py`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py) 에서 `DeepseekV3FlashAttention2.forward` 를 발췌하면 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet01_deepseek_mla_forward.py"></script>

한 레이어당 GEMM 만 보면 다음 다섯 개가 순차적으로 launch 된다.

1. `q_a_proj`: hidden → q_lora_rank (input: hidden_states)
2. `q_b_proj`: q_lora_rank → num_heads × q_head_dim (input: `q_a_layernorm` 출력)
3. `kv_a_proj_with_mqa`: hidden → kv_lora_rank + qk_rope (input: hidden_states)
4. `kv_b_proj`: kv_lora_rank → num_heads × (qk_nope + v_head_dim) (input: `kv_a_layernorm` 출력)
5. `o_proj`: num_heads × v_head_dim → hidden

여기에 두 RMSNorm 과 reshape/split/transpose, 그리고 RoPE 가 끼어든다. 짧은 seq 에서는 이 직렬 chain 자체가 step time 의 큰 부분이 된다.

---

# A — `q_a_proj` + `kv_a_proj_with_mqa` 배치 GEMM

GEMM 1번과 3번은 **둘 다 입력이 `hidden_states` 로 같다**. 그러면 두 weight 를 row 방향으로 concat 해서 단일 GEMM 으로 처리할 수 있다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet02_batched_qa_kva.py"></script>

이게 안전한 이유는 다음과 같다.

- **수학 동등**: 두 weight 가 같은 input 을 받으므로, concat 한 weight 에 한 번 곱하나 따로 두 번 곱하나 결과는 동일하다 (bf16 누적 순서만 다를 수 있는데, 이는 ULP 수준).
- **Autograd 안전**: `torch.cat([w1, w2])` 는 두 Parameter 에 대해 미분 가능하므로, gradient 가 각각 `q_a_proj.weight.grad`, `kv_a_proj_with_mqa.weight.grad` 로 정상적으로 흘러간다.
- **메모리**: concat 결과는 매 forward 마다 transient tensor 로 생성된 뒤 해제. weight 자체의 storage 는 그대로.

이득은 launch 1개 절약이지만, 48 layer × forward + backward = 96 launch 절약. 짧은 seq 에서 launch-bound 인 영역에선 측정 가능한 이득이 나온다.

---

# B — `new_empty + slice` → `torch.cat`

원본 코드의 query/key 조립 패턴은 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet03_orig_new_empty_pattern.py"></script>

이건 `(B, H, S, q_head_dim)` 의 새 텐서를 0 으로 초기화하지 않고 (empty) 만든 뒤, 두 번의 slice assignment 로 채우는 패턴이다. 결과는 같지만 다음 두 가지가 단점이다.

- Slice assignment 는 in-place op 라 autograd 가 추적해야 하는 view 그래프가 복잡해진다.
- `new_empty + 2 slice assign` = 메모리 alloc 1개 + assignment 2개 = 3 op. `torch.cat` 한 줄로 묶이지 않는다.

`torch.cat` 으로 치환하면 다음과 같이 간결해진다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet04_cat_query_key.py"></script>

`k_pe` 는 원래 `(B, 1, S, qk_rope)` 로 head 차원이 1 인데, MLA 의 K_pe 는 모든 head 가 공유한다. 원본은 slice assignment 에서 broadcast 가 일어났지만, `torch.cat` 으로 묶을 땐 `.expand()` 로 명시적으로 head 차원을 풀어 줘야 한다 (`.expand` 는 메모리 복사 없는 stride-0 view).

`torch.cat` 의 출력은 contiguous tensor 이므로 이후 FA2 호출에 그대로 넘길 수 있다.

B 단독의 wall-time 효과는 작지만, autograd graph 가 단순해지면서 backward pass 가 약간 가벼워진다. 코드 가독성 면에서도 분명한 개선이라 A 와 항상 같이 적용한다.

---

# A + B 를 합친 forward

A + B 를 합친 fused forward 의 전체 모양은 다음과 같다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet05_fused_ab_full.py"></script>

이걸 modeling 코드 자체에 넣는 대신 다음처럼 monkey-patch 로 적용하면 원본 modeling 파일은 그대로 유지된다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet06_monkey_patch.py"></script>

---

# D — K-side absorption (선택적)

DeepSeek-V3 논문이 추론 가속에서 사용하는 **MLA absorption** 을 학습 forward 에도 부분적으로 적용할 수 있다. 학습 시점엔 `q_a_layernorm`, `kv_a_layernorm` 이 chain 중간에 있어 단순 합성은 불가능하지만, attention dot product 의 nope 영역만 보면 weight 합성이 가능하다.

attention score 의 nope 영역을 per-head 로 펼치면 다음과 같다.

$$
\text{Score}_{\text{nope}}(i, j; h) = q_{\text{nope}, h}(i) \cdot k_{\text{nope}, h}(j)
$$

여기서

$$
q_{\text{nope}, h}(i) = W_{qb, h}^{\text{nope}} \cdot \text{LN}_{qa}(W_{qa} h_i)
$$

$$
k_{\text{nope}, h}(j) = W_{kb, h}^{\text{nope}} \cdot \text{LN}_{kva}(W_{kva} h_j)
$$

이걸 score 에 대입해 풀면, LN 출력을 인자로 갖고 weight 합성이 가능한 형태가 나온다.

$$
\text{Score}_{\text{nope}}(i, j; h)
= \text{LN}_{qa}(h_i)^\top \cdot \underbrace{W_{qb, h}^{\text{nope}\,\top} W_{kb, h}^{\text{nope}}}_{W_{\text{abs}, h}} \cdot \text{LN}_{kva}(h_j)
$$

`W_abs_h` 는 head 마다 `(q_lora_rank, kv_lora_rank)` 의 작은 행렬이다. 이걸 매 forward 마다 한 번 계산하면, K 의 `k_b_nope` GEMM 을 통째로 건너뛸 수 있다.

<script src="https://gist.github.com/wonbeomjang/4920fba049a999ba4a5460179181006e.js?file=mla_projection_snippet07_absorbed_d.py"></script>

수학적으로 동등하므로 학습 loss curve 는 bf16 ULP 노이즈 내에서 baseline 과 일치한다.

다만 D 에는 **성능 trade-off** 가 있다.

- `torch.bmm` 로 매 forward 마다 `W_abs` 를 만든다 — 이 op 는 token 수 N 과 무관한 **고정 오버헤드** (heads × q_lora × qk_nope × kv_lora).
- 절약되는 비용은 `k_b_nope` GEMM (token 수 N 에 비례) 의 launch + compute.

따라서 N 이 클수록 D 가 유리하고, N 이 작으면 fixed overhead 가 절약량을 잡아먹는다. 본 글의 smoke 환경 (평균 seq ~100 token) 에선 D 가 net-negative 였다.

---

# Experiments

## 마이크로벤치 (단일 GPU, B=1, S=4096, fwd+bwd 30 iter)

DeepSeek-V3-Lite 크기 (`H=2048, num_heads=32, q_lora=384, kv_lora=128, qk_nope=128, qk_rope=64, v_head_dim=128`) 기준.

| 구성                                 | 시간 / iter | Speedup               |
| ------------------------------------ | ----------- | --------------------- |
| Original (DeepseekV3FlashAttention2) | 10.63 ms    | 1.00×                 |
| A + B                                | 9.19 ms     | **1.16×**             |
| A + B + D                            | 8.87 ms     | **1.20×** (D 추가 4%) |

수치 동등성:

- A + B vs original: max abs diff 3.9e-3 (정확히 bf16 한 step ULP), mean abs diff 4.0e-5
- A + B + D vs original: max abs diff 동일, mean abs diff 1.3e-4 (attention 누적이 한 단계 더 들어가서 약간 큼)

## End-to-end FSDP smoke (A100 × 8, FULL_SHARD)

bf16, gradient_checkpointing on, per_device_bs=1, grad_accum=16, max_length=8192 SFT smoke. 평균 seq 길이가 ~100 token 인 짧은 데이터.

| 구성                            | per-step (steady) | Speedup vs baseline                |
| ------------------------------- | ----------------- | ---------------------------------- |
| baseline (original MLA forward) | 17.55 s           | 1.00×                              |
| A + B                           | **16.65 s**       | **1.054×**                         |
| A + B + D                       | 17.10 s           | 1.026× (A + B 대비 0.97×, 즉 손해) |

여기서 마이크로벤치와 FSDP smoke 의 결과가 갈린다.

- 마이크로벤치 (S=4096) 에선 D 의 token-비례 절약량이 fixed bmm overhead 를 이긴다.
- FSDP smoke 의 평균 seq ~100 token 에선 절약량이 너무 작아서 bmm 의 고정 비용이 net loss.

장문 SFT (예: max_length 8K 의 90% 이상 채우는 dataset) 또는 pretrain 처럼 packed sequence 길이가 충분히 큰 경우엔 D 가 다시 net-positive 가 될 가능성이 있다.

---

# Conclusion

MLA 의 학습 forward 는 KV 압축 이점을 못 받으면서 직렬 GEMM chain 만 남기는 구조다. 그 chain 을 두 가지 안전한 변환으로 정리할 수 있다.

- **A — 입력이 같은 두 GEMM 묶기**: 항상 켜라. 1 launch 절약, autograd 안전, 메모리 영향 없음.
- **B — `cat` 으로 query/key 만들기**: 항상 켜라. autograd graph 단순화 + 코드 가독성.
- **D — K-side absorption**: long-seq 환경에서만 켜라. short-seq 에선 bmm 고정 오버헤드가 절약량보다 크다.

근본적으로 한 단계 더 가속하고 싶다면 FA2 의 `q_head_dim != v_head_dim` 미지원이 다음 병목이다. 현재는 V 를 `q_head_dim` 까지 zero-padding 하면서 attention compute 의 약 33% 가 padding 영역에 낭비된다 (`v_head_dim=128`, `q_head_dim=192` 기준). FlashAttention 3 또는 custom Triton kernel 이 답이지만, 이건 별도 글로 다룰 주제다.

---

# 참고 문헌

- [DeepSeek-V2 paper (MLA 도입)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 modeling code](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA absorption 의 추론 적용
- [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) — `flash_attn_func` / `flash_attn_varlen_func`
- 선행편: DeepSeek 계열 MoE 학습 가속 (Python expert loop → grouped GEMM)
