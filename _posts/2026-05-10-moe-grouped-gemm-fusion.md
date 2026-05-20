---
layout: post
title: "DeepSeek 계열 MoE 학습 가속: Python expert loop → grouped GEMM"
date: 2026-05-10 09:00:00 +0900
description: "DeepSeek-V3 공개 modeling 의 expert for-loop 가 왜 학습 병목이 되는지, grouped GEMM 으로 fuse 해 단일 GPU 마이크로벤치 6.69×, end-to-end FSDP 학습 6.27× 가속한 과정"
categories: [dev]
tags: [llm, moe, cuda, deepseek, training-optimization, optimization]
giscus_comments: true
related_posts: true
---

# Introduction

DeepSeek-V2/V3 의 공개 modeling 코드를 그대로 SFT 학습에 쓰면, MoE 가 학습 step time 의 사실상 전부를 잡아먹는다. 원인은 modeling 의 `forward` 가 학습 경로를 사실상 제공하지 않고, 사용자가 직접 작성하게 되어 있는 `moe_train` 이 보통 **128~256 expert 를 Python `for` loop 으로 도는 naive 구현**이라는 점에 있다.

이 글의 결과는 한 줄로 다음과 같다. **단일 GEMM 으로 묶어서 풀자.** Variable-M grouped matmul (이하 grouped GEMM) 로 expert MLP 의 gate/up/down 세 번을 각각 1 번씩 부르면, 단일 GPU 마이크로벤치에서 **6.69×**, A100 8 GPU FSDP smoke 에서 **end-to-end 6.27×** 의 가속이 나온다.

| 측정                          | naive Python loop | grouped GEMM                 |
| ----------------------------- | ----------------- | ---------------------------- |
| 단일 레이어 (N=4096, fwd+bwd) | 87.9 ms           | 13.1 ms (**6.69×**)          |
| 8 GPU FSDP per-step (smoke)   | 110.1 s           | 17.55 s (**6.27×**)          |
| Peak GPU memory               | —                 | 동일                         |
| Loss 차이 (5 step)            | —                 | ≤ 0.16% rel (bf16 노이즈 내) |

---

# Background — DeepSeek 계열 MoE 구조

먼저 DeepSeek-V3 의 [`modeling_deepseek.py`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py) 의 MoE 모듈을 그대로 보자. (snippet 은 추후 Gist 임베드로 교체 예정)

```python
class DeepseekV3MoE(nn.Module):
    """A mixed expert module containing shared experts."""

    def __init__(self, config):
        super().__init__()
        ...
        self.experts = nn.ModuleList([
            DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
```

이상한 부분이 두 가지 보인다.

1. `forward` 가 `if not self.training:` 분기에서만 `y` 를 정의한다. **학습 모드에서 `y` 는 정의되지 않은 상태로 shared expert 와 합쳐지면서 NameError 가 난다.** 사용자가 직접 `moe_train` 을 구현해서 monkey-patch 또는 코드 수정으로 끼워 넣어야 한다.
2. 추론 경로 `moe_infer` 는 `@torch.no_grad()` 데코레이터가 붙어 있어 학습용으로 재사용 불가.

학습용 forward 를 직접 짜야 하니, 가장 자연스러운 형태가 다음과 같은 Mixtral-style scatter-gather 다.

```python
def moe_train(self, x, topk_ids, topk_weight):
    num_experts = self.config.n_routed_experts            # e.g. 256
    y = torch.zeros(x.shape[0], x.shape[1],
                    dtype=topk_weight.dtype, device=x.device)
    expert_mask = F.one_hot(topk_ids, num_classes=num_experts).permute(2, 1, 0)
    # (num_experts, top_k, N)

    for expert_idx in range(num_experts):                 # ← Python loop 256회
        expert = self.experts[expert_idx]
        top_k_slot, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        expert_out = expert(x[token_idx])                 # 3 GEMMs (gate/up/down)
        weights = topk_weight[token_idx, top_k_slot, None]
        y.index_add_(0, token_idx, expert_out.to(topk_weight.dtype) * weights)
    return y.to(x.dtype)
```

매 forward 마다 **`num_experts × 3 = 768` 개 작은 GEMM kernel 을 직렬로 launch** 한다. A100 의 CUDA kernel launch latency 는 ~5–10 μs, 그래서 768 launch × 7 μs ≈ **5.4 ms 가 순수 launch overhead**. 거기다 expert 당 평균 token 수는 `N × top_k / num_experts` 인데, `N=4096 / top_k=8 / E=256` 이면 **expert 당 평균 128 token** — 이 정도 크기의 GEMM 은 A100 의 tensor core 를 채우지 못해 GPU 가 거의 idle 상태가 된다.

---

# 무엇이 필요한가 — Variable-M batched matmul

해결책의 형태는 명확하다. **Expert 별로 token 을 정렬한 뒤, 한 번의 kernel call 로 모든 expert 의 GEMM 을 처리**하면 launch overhead 도 사라지고 GPU utilization 도 올라간다. 이를 위해 필요한 연산은 다음과 같다.

| 연산자                                    | 모든 그룹 GEMM 크기  | API                  |
| ----------------------------------------- | -------------------- | -------------------- |
| `torch.matmul` / `F.linear`               | 동일                 | 단일 행렬            |
| `torch.bmm`                               | 모든 그룹 **같은 M** | batched (3-D)        |
| **`gg.gmm`** (grouped GEMM)               | 그룹마다 **다른 M**  | grouped              |
| `torch.ops.aten._grouped_mm` (torch 2.7+) | 그룹마다 다른 M      | native (cu126+ 필요) |

`torch.bmm` 은 모든 그룹의 M 이 같아야 하므로 expert routing 처럼 그룹 크기가 들쭉날쭉하면 zero-padding 으로 낭비가 발생한다. 진짜 "ragged batch" GEMM 는 cutlass 의 `GroupedGemm` kernel 이 표준 구현이며, [tgale96/grouped_gemm](https://github.com/tgale96/grouped_gemm) 이 그 cutlass kernel 의 PyTorch autograd 바인딩이다 — MegaBlocks 의 block-sparse path 와 같은 author 의 작업.

API 는 다음과 같이 단순하다.

```python
out = gg.gmm(a, b, batch_sizes, trans_b=False)
# a:            (M_total, K)
# b:            (E, K, N)      [or (E, N, K) if trans_b=True]
# batch_sizes:  (E,) int64 CPU tensor — 합 = M_total
# out:          (M_total, N)
#
# Per group e:
#   out[batch_sizes[:e].sum():batch_sizes[:e+1].sum()]
#     = a[해당 슬라이스] @ b[e]
```

---

# 적용 — fused MoE training forward

이제 위 naive `moe_train` 을 grouped GEMM 으로 다시 작성한다.

```python
import torch
import torch.nn.functional as F
from grouped_gemm import ops as gg


def fused_moe_train(self, x, topk_ids, topk_weight):
    """
    Args:
        x:           (N, H)     flattened token hidden states
        topk_ids:    (N, top_k) expert indices per token
        topk_weight: (N, top_k) gate weights (fp32)
    Returns:
        y: (N, H) in x.dtype
    """
    N, H = x.shape
    top_k = topk_ids.shape[1]
    E = self.config.n_routed_experts

    # (1) Flatten (token, slot) → assignment list of size N*top_k.
    flat_expert_id = topk_ids.reshape(-1)
    flat_token_id = torch.arange(N, device=x.device).repeat_interleave(top_k)
    flat_weight = topk_weight.reshape(-1)

    # (2) Sort by expert so tokens for the same expert are contiguous.
    sort_idx = flat_expert_id.argsort()
    sorted_token_id = flat_token_id[sort_idx]
    sorted_weight = flat_weight[sort_idx]

    # (3) group_sizes[e] = number of (token, slot) assignments → expert e.
    #     grouped_gemm requires a CPU int64 tensor.
    group_sizes = torch.bincount(flat_expert_id, minlength=E).to(
        device="cpu", dtype=torch.int64
    )

    # (4) Gather sorted token features.
    sorted_x = x[sorted_token_id]                                 # (N*k, H)

    # (5) Stack expert weights → (E, out, in). Use trans_b=True so we don't
    #     pay a non-contiguous transpose; gg internally does b^T per group.
    w_gate = torch.stack([e.gate_proj.weight for e in self.experts])  # (E, I, H)
    w_up = torch.stack([e.up_proj.weight for e in self.experts])      # (E, I, H)
    w_down = torch.stack([e.down_proj.weight for e in self.experts])  # (E, H, I)

    # (6) Three grouped GEMMs replace 3 × E sequential Linear calls.
    gate_out = gg.gmm(sorted_x, w_gate, group_sizes, trans_b=True)    # (N*k, I)
    up_out = gg.gmm(sorted_x, w_up, group_sizes, trans_b=True)        # (N*k, I)
    mid = F.silu(gate_out) * up_out
    down_out = gg.gmm(mid, w_down, group_sizes, trans_b=True)         # (N*k, H)

    # (7) Weight by gate score (fp32 accum) and scatter back.
    weighted = down_out.to(topk_weight.dtype) * sorted_weight.unsqueeze(-1)
    y = torch.zeros(N, H, dtype=topk_weight.dtype, device=x.device)
    y.index_add_(0, sorted_token_id, weighted)
    return y.to(x.dtype)
```

핵심은 다음 세 가지다.

- **(1)–(3) 토큰 정렬**: `topk_ids` 를 1D 로 펼친 뒤 expert id 로 sort 하면, 같은 expert 로 가는 토큰들이 연속된 메모리 슬라이스로 모인다. `torch.bincount` 가 expert 별 토큰 수를 한 번에 계산해 준다.
- **(5) Weight stacking**: `nn.Linear.weight` 는 `(out, in)` 레이아웃이므로, 그대로 `torch.stack` 하면 `(E, out, in)` 가 된다. `trans_b=True` 를 쓰면 grouped GEMM 이 내부에서 transpose 를 처리하므로 `.transpose().contiguous()` 의 메모리 copy 를 피할 수 있다 (autograd 도 정상 추적).
- **(6) 세 번의 `gg.gmm`**: gate → up → down 의 expert MLP 세 단계 모두 동일한 `group_sizes` 를 사용한다. 768 launch 가 **3 launch 로** 줄어든다.

`forward` 에서 학습 모드에 이 함수를 호출하도록 분기시키면 끝이다.

```python
def forward(self, hidden_states):
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if self.training:
        y = self.moe_train(hidden_states, topk_idx, topk_weight).view(*orig_shape)
    else:
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y
```

`moe_train` 은 위의 `fused_moe_train` 을 monkey-patch 형태로 클래스에 붙이면 modeling 파일 자체는 손대지 않을 수 있다.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
for m in model.modules():
    if m.__class__.__name__ == "DeepseekV3MoE":
        m.__class__.moe_train = fused_moe_train
        # 동시에 forward 도 학습 분기를 갖도록 교체
```

---

# FSDP 와의 통합

이 fused forward 는 FSDP FULL_SHARD 와 자연스럽게 호환된다. 단, 한 가지 조건이 있다.

- `fsdp_use_orig_params=True` 가 켜져 있어야 한다 (`accelerate` 의 FSDP config 에서 기본값이 false 일 수 있음). 이 옵션이 켜지면 `self.experts[i].gate_proj.weight` 같은 원본 Parameter 객체가 그대로 노출되어, `torch.stack` 이 정상 동작한다.
- FSDP 의 wrap unit 은 transformer block 단위 (`TRANSFORMER_BASED_WRAP`) 가 자연스럽다. block 단위로 all-gather 가 일어나면, block 내부의 fused MoE forward 시점에서 expert weight 가 이미 로컬에 모여 있다.

E=256, H=2048, moe_intermediate=512 기준으로 stack 결과의 transient 메모리는 `3 × 256 × 2048 × 512 × 2B ≈ 1.5 GB`. A100 80GB 에선 충분히 감당 가능하고, 사용 직후 해제된다. 만약 E 가 훨씬 크거나 메모리가 빠듯한 환경이면, 세 weight 를 한꺼번에 쌓지 말고 GEMM 직전에 하나씩 쌓는 식으로 peak 를 낮출 수 있다.

---

# Experiments

## 마이크로벤치 (단일 GPU, A100)

E=128, H=2048, moe_intermediate=512, top_k=8, fwd+bwd 30 iter 평균.

| Token 수 N | naive Python loop | grouped GEMM | Speedup   |
| ---------- | ----------------- | ------------ | --------- |
| N=4096     | 87.94 ms          | 13.14 ms     | **6.69×** |

원인 분해:

- Kernel launch overhead 제거 (768 → 3)
- 작은 expert GEMM (avg 256 행) 들이 큰 grouped GEMM 한 번으로 묶이면서 tensor core utilization ↑

## End-to-end FSDP smoke (A100 × 8, FULL_SHARD)

bf16, gradient_checkpointing on, per_device_bs=1, grad_accum=16, max_length=8192 SFT smoke. 5 step.

| Stack                        | per-step (steady) | train_runtime | Cumulative speedup |
| ---------------------------- | ----------------- | ------------- | ------------------ |
| baseline (naive Python loop) | 110.1 s           | 572 s         | 1.00×              |
| + fused MoE (grouped GEMM)   | **17.55 s**       | **107 s**     | **6.27×**          |

수치 동등성: 5 step loss 의 max relative diff 0.16% (bf16 ULP 노이즈 내), peak GPU memory 변화 없음.

흥미로운 점은 마이크로벤치 6.69× 가 end-to-end 6.27× 와 거의 일치한다는 사실이다. 이는 SFT 학습 step time 의 사실상 전부가 MoE forward + backward 였다는 것을 뜻한다. attention 이나 FSDP all-gather 가 의미 있게 보이려면 이 병목을 먼저 걷어내야 측정이 가능하다.

---

# 한계

- **`ep_size > 1` 미지원**: Expert Parallel 환경에선 각 rank 가 자신이 소유한 expert subset 만 들고 있고, all-to-all 통신이 필요하다. 이 경우 grouped GEMM 만으로는 부족하고 [DeepEP](https://github.com/deepseek-ai/DeepEP) 같은 token dispatch + combine kernel 이 함께 필요하다. 본 글의 fused path 는 단일 노드 / 단일 rank 가 모든 expert 를 소유하는 FSDP 만 다룬다.
- **추론은 그대로**: `moe_infer` 의 sorted-token argsort + Linear loop 패턴은 추론 시점에선 KV cache, paged attention 등과 함께 묶여 vLLM/SGLang 의 dispatch 와 따로 다뤄야 한다.
- **DeepGEMM (FP8) 미적용**: DeepSeek-V3 본가는 H100/H800 에서 fine-grained FP8 grouped GEMM 을 자체 구현해 추가 2× 를 얻는다. A100 은 FP8 unit 부재로 bf16 grouped GEMM 까지가 한계.

---

# Conclusion

DeepSeek-V3 계열 MoE 모델은 **공개 modeling 코드가 학습용 MoE forward 를 사실상 제공하지 않는다**. 사용자가 직접 작성하는 순간 가장 흔한 형태가 naive Python `for` loop 인데, 이게 step time 의 거의 전부를 차지한다. cutlass grouped GEMM 한 번이면 단일 GPU 6.69×, end-to-end 6.27× 가속이 나온다. **이 정도면 다른 모든 최적화는 이걸 끝낸 뒤에 시작하는 게 맞다.**

---

# 참고 문헌

- [DeepSeek-V3 modeling code](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [tgale96/grouped_gemm](https://github.com/tgale96/grouped_gemm) — cutlass grouped GEMM 의 PyTorch autograd 바인딩
- [MegaBlocks](https://github.com/stanford-futuredata/megablocks) — block-sparse MoE 의 reference 구현 (같은 author)
- [DeepEP](https://github.com/deepseek-ai/DeepEP) — expert parallel 환경의 token dispatch/combine kernel
- 후속편: MLA 학습 시 modeling-side projection fusion (q_a/kv_a 묶기 + K-side absorption)

> 이 글의 모든 코드 스니펫은 추후 GitHub Gist 로 옮겨 `<script>` 임베드로 교체된다.
