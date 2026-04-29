---
layout: post
title: "Triton 07: Flash Attention 3 — Triton으로 어디까지 가능한가"
date: 2026-04-02 02:00:00 +0900
description: "Hopper 전용인 Flash Attention 3를 Triton으로 어디까지 따라잡을 수 있는가 — 확장 autotune·persistent kernel·실패한 실험까지"
categories: [triton]
tags: [triton, gpu, flash-attention, llm, attention, optimization, hopper]
giscus_comments: true
related_posts: true
featured: true
---

[Triton 06: Flash Attention 2](/blog/2026/triton-06-flash-attention-v2/)까지 와서 A100에서 PyTorch 대비 causal 22× 가속을 확인했다. 그 다음은 FA3다.

[FlashAttention-3 논문](/blog/2026/flashattention-3/)은 Hopper의 비동기·저정밀 하드웨어를 활용해 H100에서 FA2 대비 추가 1.5–2.0× 가속을 만든다. 그런데 FA3의 핵심 기법은 대부분 **Hopper 전용 + CUDA로만 표현 가능**한 것들이다. 그럼 Triton으로는 어디까지 닿을 수 있을까?

> FA3 논문 자체의 원리가 궁금하다면 [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)를 먼저 읽는 것을 추천한다.

**결론 미리**: A100 + Triton에서 FA2 대비 추가 ~3–5% 가속이 한계다. Hopper 전용 기능(TMA, wgmma, 비동기 producer-consumer)은 Triton으로 표현이 어렵거나 불가능하기 때문이다. **공식 FA3의 진짜 가치는 H100 + CUDA에서만 살아난다**는 사실을 실측으로 확인했다.

---

## FA3의 7가지 핵심 기법 — Triton에서 가능한가?

FA3 논문이 말하는 가속의 출처를 정리하면 다음과 같다.

| #   | 기법                                     | 효과 (논문)             | Triton 표현                                  | 본 구현 적용 |
| --- | ---------------------------------------- | ----------------------- | -------------------------------------------- | ------------ |
| 1   | Producer-consumer warp specialization    | ~30%                    | ✗ (warp 추상화 너머)                         | 미적용       |
| 2   | Inter-warpgroup ping-pong (GEMM↔softmax) | ~15–20%                 | △ (`tl.async_task` preview)                  | 미적용       |
| 3   | TMA (비동기 메모리 복사)                 | H100에서 10–15%         | △ (`tl.make_tensor_descriptor` preview)      | 미적용       |
| 4   | wgmma (warpgroup matmul)                 | 자동                    | ✓ (`tl.dot`이 H100에서 자동 사용)            | 적용됨       |
| 5   | FP8 + incoherent processing              | H100에서 1.5–2×         | △ (block scaling 손수 작성)                  | 미적용       |
| 6   | Persistent kernel (NUM_SMS launch)       | launch 절감 + occupancy | ✓                                            | 시도→실패    |
| 7   | Wider autotune                           | 1–5%                    | ✓                                            | 적용됨       |

Triton으로 즉시 가능한 것은 4·6·7 정도다. 1·2는 H100에서도 Triton으로 표현이 어렵고, 3은 experimental, 5는 매우 복잡하다. 즉 Triton FA3는 본질적으로 **"FA2 + 스케줄링 튜닝"** 의 영역에 머문다.

---

## Method: Triton FA3에 적용한 변경

알고리즘 코어는 [FA2](/blog/2026/triton-06-flash-attention-v2/)와 동일하다 — un-scaled 누적, `exp2`, STAGE 1/2 분기, `tl.dot` accumulator. 그 위에 다음을 얹었다.

### 1. 확장된 autotune 탐색 공간

FA2는 6개 config를 탐색했다. FA3는 17개로 늘리고, BLOCK_M ≤ 256 + num_stages ≤ 6까지 포함한다.

<script src="https://gist.github.com/wonbeomjang/2231bb41af1f36d52e143c60386cf7a0.js?file=triton_07_snippet01_autotune_configs.py"></script>

문제는 `head_dim=128 + BLOCK_M=256`처럼 SRAM을 초과하는 조합이다. `tl.arange`가 power-of-2만 받는 제약과 별개로, 이런 config는 컴파일 시 OOM으로 실패한다.

### 2. SRAM-aware early pruning

`prune_configs_by`로 컴파일 전에 부적합한 config를 제거한다.

<script src="https://gist.github.com/wonbeomjang/2231bb41af1f36d52e143c60386cf7a0.js?file=triton_07_snippet02_early_prune.py"></script>

이 한 가지로 `head_dim=128`에서 `BLOCK_M=256` 케이스가 자동 차단되고, autotune이 실제로 fit하는 config만 프로파일링한다.

### 3. 명시적 `fp32` 누적

`tl.dot`은 기본적으로 fp32 accumulator를 쓰지만, `out_dtype`을 명시해 FA3 의도(저정밀 입력 + fp32 누적)를 분명히 했다.

<script src="https://gist.github.com/wonbeomjang/2231bb41af1f36d52e143c60386cf7a0.js?file=triton_07_snippet03_explicit_fp32.py"></script>

H100의 wgmma 명령은 이 `tl.dot`이 자동으로 사용한다 — 별도 코드 없이 `dot`만 부르면 된다.

### 4. 1D grid + `(bh, m)` decomposition

FA2는 `(num_m_tiles, bh)` 2D grid였다. FA3는 1D grid `(num_m_tiles · bh,)`로 단순화한다.

<script src="https://gist.github.com/wonbeomjang/2231bb41af1f36d52e143c60386cf7a0.js?file=triton_07_snippet04_1d_grid.py"></script>

같은 `bh`의 인접 Q 타일이 연속된 program ID를 받으므로 **L2 cache 친화적인 SM 매핑**이 자연스럽게 만들어진다. Triton의 SM 스케줄러는 인접 program을 같은 SM 또는 인접 SM으로 보내는 경향이 있어, K/V 재사용률이 올라간다.

---

## Persistent kernel: 시도했으나 실패한 이야기

FA3 논문의 6번 기법 — **persistent kernel**을 시도해봤다. 결과부터: A100에서는 손해다.

### 시도한 것

기존 grid `(num_m_tiles · bh,)` 대신 `(min(NUM_SMS, total_tiles),)`로 launch하고, 각 program이 내부 루프로 자기 몫의 타일을 순회한다.

<script src="https://gist.github.com/wonbeomjang/2231bb41af1f36d52e143c60386cf7a0.js?file=triton_07_snippet05_persistent_kernel.py"></script>

### 측정 결과 (A100-SXM4-80GB, fp16, 4 GPU 평균)

`num_heads=16, head_dim=64, causal=True`:

| seq   | FA3 (1D grid) | FA3-Persistent | 차이     |
| ----- | ------------- | -------------- | -------- |
| 4096  | 0.350         | 0.388          | **-11%** |
| 8192  | 0.995         | 1.164          | **-17%** |
| 16384 | 3.441         | 3.778          | **-10%** |
| 32768 | 12.911        | 13.595         | **-5%**  |

긴 seq에서도 1D grid가 더 빠르다. **persistent가 평균 5–17% 손해를 본다**.

### 왜 손해를 봤는가

원인은 **work imbalance**다.

```
A100 NUM_SMS = 108
seq=4096, bh=16, BLOCK_M=128 → num_tiles = 32 × 16 = 512 tiles

기존 grid:        (512,)  → 모든 SM이 work 보유, 하드웨어 스케줄러가 밸런싱
Persistent grid:  (108,)  → 각 SM이 ~5 tiles 직렬 처리
                            launch 절감 < 직렬 비율 증가로 인한 latency
```

Triton의 1D grid는 SM 스케줄러가 동적으로 분배하는 반면, persistent는 **wave 단위 직렬화**가 강제된다. seq가 길어 tile 수가 충분하면 그냥 1D grid가 더 좋다.

### 그럼 공식 FA3는 왜 persistent로 이득을 보는가?

H100 환경에서 persistent가 이득이 되는 이유는 단순히 launch 절감이 아니다. Persistent 자체로는 위 실험처럼 손해다. 진짜 이득의 출처는:

1. **wgmma의 큰 타일** (BLOCK_M=192) → 같은 seq에서 tile 수가 줄어 직렬화 비율이 낮아짐
2. **Producer-consumer warp split** + MBARRIER → 같은 SM 안에서 GEMM ↔ softmax 비동기 오버랩
3. **TMA 비동기 복사** → 메모리 latency를 compute로 가림

즉 persistent는 **나머지 셋의 base** 역할일 뿐, 단독으로는 의미가 없다. Triton에서는 이 셋 모두 표현이 어려우므로, persistent를 켜면 base만 남고 효과가 사라진다 — 오히려 손해.

이 이유로 본 구현에서는 `flash_attention_v3_persistent`를 **제거**했다. 향후 H100 + Triton TMA 지원이 더 발전한 뒤 재시도할 가치가 있다.

---

## 벤치마크 결과 (A100-SXM4-80GB × 4)

`num_heads=16, fp16` · 4 GPU 평균 (표준편차 < 1%) · 11회 측정 중 첫 회 폐기.

### Causal forward, head_dim=64

| seq   | FA1    | FA2    | **FA3** | PyTorch | FA3/FA2   | FA3/PT     |
| ----- | ------ | ------ | ------- | ------- | --------- | ---------- |
| 4096  | 0.571  | 0.361  | 0.350   | 5.243   | 1.03×     | 14.97×     |
| 8192  | 1.721  | 1.033  | 0.992   | 21.807  | **1.04×** | **21.98×** |
| 16384 | 5.972  | 3.556  | 3.391   | 70.856  | **1.05×** | 20.90×     |
| 32768 | 22.247 | 13.426 | 12.847  | OOM     | **1.05×** | —          |

### Causal forward, head_dim=128 (Llama/Qwen 표준)

| seq   | FA1    | FA2    | **FA3** | FA3/FA2   |
| ----- | ------ | ------ | ------- | --------- |
| 2048  | 0.374  | 0.257  | 0.245   | 1.05×     |
| 4096  | 1.113  | 0.587  | 0.579   | 1.01×     |
| 8192  | 3.905  | 1.824  | 1.763   | **1.03×** |
| 32768 | 57.620 | 24.670 | 24.328  | 1.01×     |

### 핵심 관찰

- **causal + 긴 seq에서 일관되게 3–5% 추가 가속** — 확장 autotune이 더 큰 BLOCK_M·num_warps=8·num_stages=5를 선택해 SRAM 점유율이 좋아진 결과
- **non-causal과 짧은 seq에서는 사실상 FA2와 동일** — 해당 케이스에서 best config가 같은 config로 수렴
- **causal seq=8192에서 FA3/FA2 = 1.04×, FA3/PT = 21.98× 피크** — 알고리즘적 한계에 거의 닿았음을 의미
- **fwd+bwd**는 backward를 FA2 그대로 재사용했으므로 거의 동일 (FA3의 backward 개선은 H100 wgmma 와만 결합)

### 4 GPU 일관성

GPU 0~3에서 동일 측정값의 표준편차는 평균의 1% 미만. Triton autotune이 4개 프로세스에서 각각 동일한 best config로 수렴함을 확인.

---

## 솔직한 한계

- **Hopper 전용 기능을 활용하지 않으므로 H100에서 측정해도 본 구현은 큰 이득이 없다.** 진짜 FA3 가속은 TMA + wgmma + producer-consumer 가 결합될 때 나타난다.
- **FP8** 은 본 구현에 미포함. block scaling + Hadamard 변환은 Triton으로 표현 가능하지만 별도의 큰 작업이 필요하다.
- **Backward** 는 FA2와 동일한 3-stage 커널을 그대로 재사용. FA3 논문의 backward 개선도 wgmma 의존이 크다.

### Production용 권장

| 환경                                  | 추천                                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| 학습/연구                             | 본 Triton FA3 (충분히 빠르고 알고리즘 이해에 좋음)                                     |
| Production A100                       | 본 Triton FA3 또는 [`flash-attn`](https://github.com/Dao-AILab/flash-attention) 패키지 |
| Production H100                       | **`flash-attn` 패키지 (`pip install flash-attn`)** — Triton으로 쫓아갈 가성비 X        |
| 새 attention 변형 (sliding window 등) | CUTLASS 기반 fork 또는 Triton 프로토타이핑 후 CUDA 이식                                |

---

## 시리즈 정리

[Triton 05 (FA1)](/blog/2026/triton-05-flash-attention/) → [Triton 06 (FA2)](/blog/2026/triton-06-flash-attention-v2/) → 본 글까지의 누적 결과 (A100, causal, head_dim=64):

| Seq   | FA1 (ms) | FA2 (ms) | FA3 (ms) | FA1→FA2 | FA2→FA3 | FA3 vs PT |
| ----- | -------- | -------- | -------- | ------- | ------- | --------- |
| 4096  | 0.571    | 0.361    | 0.350    | 1.58×   | 1.03×   | 14.97×    |
| 8192  | 1.721    | 1.033    | 0.992    | 1.67×   | 1.04×   | 21.98×    |
| 16384 | 5.972    | 3.556    | 3.391    | 1.68×   | 1.05×   | 20.90×    |
| 32768 | 22.247   | 13.426   | 12.847   | 1.66×   | 1.05×   | —         |

FA1 → FA2 의 ~1.6× 점프와 비교하면 FA2 → FA3 의 ~1.04× 는 작아 보이지만, 알고리즘 자체는 거의 한계에 도달했고 그 이상은 하드웨어 특화 기법으로 가야 한다는 **신호**다.

---

## 전체 코드

<script src="https://gist.github.com/wonbeomjang/0f4970e5dbed9af5037d796fa395727f.js?file=flash_attention_v3.py"></script>

---

> 알고리즘 원리가 궁금하다면 [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)를, FA1 Triton 구현이 궁금하다면 [Triton 05](/blog/2026/triton-05-flash-attention/)를, FA2 개선이 궁금하다면 [Triton 06](/blog/2026/triton-06-flash-attention-v2/)을, Blackwell 최적화가 궁금하다면 [FlashAttention-4 논문 리뷰](/blog/2026/flashattention-4/)를 참고하자.

# 참고 문헌

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 논문 리뷰](/blog/2026/flashattention-3/)
- [Triton Fused Attention 공식 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton 05: Flash Attention 1 — 종합 프로젝트](/blog/2026/triton-05-flash-attention/)
- [Triton 06: Flash Attention 2 — FA1 대비 5가지 최적화](/blog/2026/triton-06-flash-attention-v2/)
- [Tri Dao, FlashAttention 공식 구현](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
