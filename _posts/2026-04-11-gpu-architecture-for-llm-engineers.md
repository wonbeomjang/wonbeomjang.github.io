---
layout: post
title: "LLM 엔지니어가 알아야 할 GPU 아키텍처: Ampere → Hopper → Blackwell"
date: 2026-04-11 18:00:00 +0900
description: "A100, H100, B200 GPU를 LLM 학습/추론 관점에서 비교 — 메모리, 연산, 정밀도, 병목 분석"
categories: [dev]
tags: [gpu, llm, hardware-optimization]
giscus_comments: true
related_posts: true
---

# 왜 GPU 아키텍처를 알아야 하는가

LLM 엔지니어에게 GPU는 "빠른 연산기" 이상의 의미를 가진다. 모델 크기, 배치 사이즈, 시퀀스 길이, 학습/추론 전략 등 거의 **모든 엔지니어링 결정이 GPU의 제약에 의해 결정**되기 때문이다.

- "70B 모델을 학습하려면 GPU 몇 장이 필요한가?" → **메모리 용량**
- "KV cache가 얼마나 들어가는가?" → **메모리 대역폭**
- "FP8로 학습하면 속도가 얼마나 빨라지는가?" → **정밀도별 연산 성능**
- "FlashAttention을 쓰면 왜 빨라지는가?" → **메모리 계층과 IO 병목**
- "H100에서 A100 대비 실제로 얼마나 빠른가?" → **실제 활용률과 병목 분석**

이 글에서는 현재 LLM 엔지니어가 주로 사용하는 세 가지 GPU — **A100 (Ampere)**, **H100 (Hopper)**, **B200 (Blackwell)** — 를 LLM 학습과 추론의 관점에서 비교한다.

---

# 1. 전체 스펙 비교

| | **A100 SXM** | **H100 SXM** | **B200 SXM** |
|---|---|---|---|
| **아키텍처** | Ampere (2020) | Hopper (2022) | Blackwell (2025) |
| **공정** | TSMC 7nm | TSMC 4N | TSMC 4NP |
| **트랜지스터** | 54.2B | 80B | 208B (듀얼 다이) |
| **SM 수** | 108 | 132 | ~148 × 2 다이 |
| **Tensor Core** | 3세대 | 4세대 | 5세대 |
| **메모리** | 80GB HBM2e | 80GB HBM3 | 192GB HBM3e |
| **메모리 대역폭** | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| **NVLink** | 600 GB/s | 900 GB/s | 1.8 TB/s |
| **TDP** | 400W | 700W | 1,000W |

---

# 2. LLM 학습 관점: "얼마나 빨리 학습할 수 있는가"

## 2.1 Tensor Core 성능: 정밀도가 핵심

LLM 학습에서 가장 많은 시간을 차지하는 연산은 **행렬곱(MatMul)**이다. Linear layer, Attention의 $$QK^\top$$와 $$PV$$ 모두 matmul이다. Tensor Core의 정밀도별 처리량이 곧 학습 속도를 결정한다.

| 정밀도 | **A100** | **H100** | **B200** | 용도 |
|--------|---------|---------|---------|------|
| FP32 | 19.5 TF | 67 TF | 80 TF | 디버깅, 정밀 연산 |
| TF32 | 156 TF | 495 TF | ~1,000 TF | PyTorch 기본 학습 |
| **BF16** | **312 TF** | **989 TF** | **2,250 TF** | **LLM 학습 표준** |
| **FP8** | — | **1,979 TF** | **4,500 TF** | 대규모 학습 가속 |
| FP4 | — | — | 9,000 TF | 추론 최적화 |

### 실무 포인트

**BF16이 LLM 학습의 사실상 표준**이다. FP16 대비 dynamic range가 넓어서 loss scaling 없이도 안정적이다. A100의 312 TFLOPS → H100의 989 TFLOPS → B200의 2,250 TFLOPS로, 세대마다 **약 2-3배씩** 빨라진다.

**FP8 학습**은 H100부터 지원된다. Transformer Engine이 layer별로 FP8/BF16을 자동 전환하여, BF16 대비 **약 1.5-2배** 추가 speedup을 제공한다. 다만 학습 안정성이 모델과 데이터에 따라 달라서, 아직 모든 상황에서 사용하지는 않는다.

**FP4는 주로 추론용**이다. 학습에 사용하기에는 정밀도가 부족하지만, 양자화된 모델의 추론 속도를 극대화할 수 있다.

## 2.2 실제 활용률: 이론 vs 현실

이론 성능은 좋지만, 실제로 달성하는 **Model FLOPs Utilization (MFU)**은 다르다.

| | **A100** | **H100** | **B200** |
|---|---|---|---|
| 이론 BF16 | 312 TF | 989 TF | 2,250 TF |
| 일반적인 MFU | 40-55% | 35-50% | 30-45% (추정) |
| **실제 학습 처리량** | ~130-170 TF | ~350-500 TF | ~700-1000 TF |

MFU가 세대가 올라갈수록 **약간 낮아지는 경향**이 있다. 이는 Tensor Core가 빨라지는 속도를 메모리 대역폭과 interconnect가 따라가지 못하기 때문이다. 이것이 바로 FlashAttention 같은 **IO-aware 알고리즘**이 점점 더 중요해지는 이유다.

## 2.3 메모리 대역폭: "Tensor Core를 먹여살릴 수 있는가"

행렬곱을 수행하려면 Tensor Core에 데이터를 **공급**해야 한다. 아무리 Tensor Core가 빨라도 데이터가 도착하지 않으면 놀게 된다.

**Arithmetic Intensity** (연산 강도)를 계산해보면:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2mnk}{2(mk + kn + mn)} \approx \frac{mnk}{mk + kn + mn}
$$

LLM의 대표적 연산인 $$[B \times S, H] \times [H, 4H]$$ (FFN의 첫 번째 linear):
- $$B \times S = 2048$$ (batch × seq), $$H = 4096$$, $$4H = 16384$$
- AI ≈ $$\frac{2048 \times 4096 \times 16384}{2 \times (2048 \times 16384 + 16384 \times 4096 + 2048 \times 4096)} \approx 1365$$

이 정도의 AI면 **모든 GPU에서 compute-bound**이므로, 큰 MatMul은 Tensor Core 성능에 비례한다.

하지만 **Attention**은 다르다. $$QK^\top$$는 $$[B \times H, S, d] \times [B \times H, d, S]$$로, $$d = 128$$이면 AI가 약 64로 낮다. 시퀀스가 짧으면 **memory-bound**가 될 수 있다. 이때 메모리 대역폭이 중요하다.

| | **A100** | **H100** | **B200** |
|---|---|---|---|
| BF16 Tensor Core | 312 TF | 989 TF | 2,250 TF |
| 메모리 대역폭 | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| **Ops:Byte** | 156 | **295** | **281** |

Ops:Byte가 높을수록 memory-bound 연산에서 **Tensor Core가 놀 확률이 높다**. H100과 B200은 A100 대비 이 비율이 2배 가까이 높아서, **FlashAttention 같은 IO 최적화가 더욱 중요**하다.

---

# 3. LLM 추론 관점: "얼마나 빨리 토큰을 생성하는가"

LLM 추론은 학습과 전혀 다른 특성을 가진다.

## 3.1 Prefill vs Decode

| 단계 | 특성 | 병목 |
|------|------|------|
| **Prefill** | 프롬프트 전체를 한 번에 처리, 큰 MatMul | **Compute-bound** |
| **Decode** | 토큰 하나씩 생성, 배치 1의 작은 MatMul | **Memory bandwidth-bound** |

**Decode가 추론의 대부분 시간을 차지**한다. Decode에서는 weight 전체를 읽어서 하나의 토큰만 생성하므로, AI가 매우 낮다 (≈ 1). 이때 성능은 **순수하게 메모리 대역폭에 의해 결정**된다.

## 3.2 Decode 처리량 계산

$$
\text{Tokens/s} = \frac{\text{Memory Bandwidth}}{2 \times \text{Model Parameters}} \text{ (BF16 기준)}
$$

| 모델 | **A100 (2 TB/s)** | **H100 (3.35 TB/s)** | **B200 (8 TB/s)** |
|------|-------------------|---------------------|------------------|
| Llama-7B (14 GB) | 143 tok/s | 239 tok/s | **571 tok/s** |
| Llama-70B (140 GB) | 14.3 tok/s | 23.9 tok/s | **57.1 tok/s** |
| Llama-405B (810 GB) | OOM | OOM | OOM (멀티 GPU) |

이 계산은 **단일 GPU, 배치 1, BF16** 기준이다. 실제로는:
- **양자화** (INT8, FP8, INT4)를 적용하면 모델 크기가 줄어 처리량 증가
- **배치 크기**를 키우면 compute-bound 영역으로 이동하여 Tensor Core 활용 가능
- **KV cache**가 추가 메모리를 소비

## 3.3 KV Cache: "시퀀스가 길면 메모리가 부족하다"

Autoregressive 생성에서 이전 토큰의 Key, Value를 캐싱한다. KV cache 크기:

$$
\text{KV Cache} = 2 \times L \times 2 \times n_h \times d \times S \times B \text{ bytes (BF16)}
$$

Llama-70B ($$L=80, n_h=8 \text{ (GQA)}, d=128$$) 기준:

| 시퀀스 길이 | Batch 1 | Batch 32 |
|----------|---------|----------|
| 4K | 0.16 GB | 5.2 GB |
| 32K | 1.3 GB | 41.6 GB |
| 128K | 5.2 GB | **166.4 GB** |

A100 80GB에서는 128K 시퀀스 + 배치 32가 **불가능**하다 (모델 140GB + KV 166GB > 80GB). H100도 마찬가지. **B200의 192GB**에서야 가능해지며, 이것이 메모리 용량 증가의 실질적 의미다.

## 3.4 정밀도와 양자화

추론에서는 학습보다 낮은 정밀도를 사용할 수 있다.

| 정밀도 | 모델 크기 (70B) | 지원 GPU | 품질 영향 |
|--------|-------------|---------|---------|
| BF16 | 140 GB | A100+ | 기준 |
| FP8 | 70 GB | H100+ | 거의 없음 |
| INT8 (W8A8) | 70 GB | A100+ | 미미 |
| INT4 (GPTQ/AWQ) | 35 GB | A100+ | 약간 |
| FP4 | 35 GB | **B200만** | 모델 의존적 |

**FP8이 현재 가장 실용적인 선택**이다. 모델 크기가 절반이 되어 메모리 대역폭 2배 + Tensor Core 2배로, BF16 대비 이론적으로 **4배** 빠르다. H100의 Transformer Engine이 FP8을 자동 관리해준다.

B200에서 추가된 **FP4는 이론적으로 BF16 대비 8배** 빠르지만, 품질 저하 없이 사용하려면 정교한 양자화 기법(GPTQ, AWQ, SqueezeLLM 등)이 필요하다.

---

# 4. 멀티 GPU 스케일링: "GPU를 더 쓰면 비례해서 빨라지는가"

## 4.1 NVLink: GPU 간 통신

LLM은 단일 GPU에 올라가지 않으므로, **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Data Parallelism (DP)**을 조합한다. 이때 GPU 간 통신 대역폭이 스케일링을 결정한다.

| | **A100** | **H100** | **B200** |
|---|---|---|---|
| NVLink | 600 GB/s | 900 GB/s | **1,800 GB/s** |
| GPU 간 대역폭 / 메모리 대역폭 | 30% | 27% | **23%** |

NVLink 대역폭이 절대적으로는 증가하지만, **메모리 대역폭 대비 비율은 오히려 감소**하고 있다. 이는 TP를 많이 쓸수록 통신 overhead가 상대적으로 커진다는 의미다.

### 실무 가이드라인

- **TP 8 이하**를 유지하는 것이 효율적. TP 16 이상은 all-reduce 비용이 급증
- 70B 모델: TP=8이면 GPU당 약 9-10GB로 적절
- 405B 모델: TP=8 + PP=4 또는 TP=8 + PP=8이 일반적
- **DeepSpeed ZeRO-3** + FSDP: DP 방향으로 메모리 분산, 통신은 gradient 동기화만

## 4.2 DGX 시스템 비교

| | **DGX A100** | **DGX H100** | **DGX B200** |
|---|---|---|---|
| GPU 수 | 8 × A100 | 8 × H100 | 8 × B200 |
| 총 메모리 | 640 GB | 640 GB | **1,536 GB** |
| 총 BF16 성능 | 2.5 PF | 7.9 PF | **18 PF** |
| 총 NVLink BW | 4.8 TB/s | 7.2 TB/s | **14.4 TB/s** |
| 가격 | ~$200K | ~$300K | ~$500K |

DGX B200 한 대에서 **1.5TB 메모리**를 사용할 수 있어, 405B 모델(BF16 810GB)도 단일 노드에서 추론이 가능하다.

---

# 5. 아키텍처별 핵심 신기능과 LLM 영향

## Ampere (A100)

| 기능 | LLM 영향 |
|------|---------|
| **BF16/TF32** | 혼합 정밀도 학습의 표준 확립 |
| **MIG** | 하나의 A100을 최대 7개로 분할 → 소형 모델 추론 효율화 |
| **3세대 NVLink** | 8-GPU TP 가능 |

## Hopper (H100)

| 기능 | LLM 영향 |
|------|---------|
| **FP8 + Transformer Engine** | 학습/추론 모두 FP8 자동 적용 → 2배 speedup |
| **TMA (Tensor Memory Accelerator)** | FlashAttention-3의 핵심 — HBM↔SMEM 전송 자동화 |
| **WGMMA (비동기 MMA)** | 연산과 데이터 전송을 겹쳐 실행 → GPU 활용률 향상 |
| **Warp Specialization** | Producer/Consumer warp 분리로 파이프라인 최적화 |

## Blackwell (B200)

| 기능 | LLM 영향 |
|------|---------|
| **FP4 네이티브** | INT4 양자화 추론이 하드웨어 수준에서 가속 |
| **192GB HBM3e** | 70B 모델을 양자화 없이 단일 GPU에 탑재 가능 |
| **8 TB/s 대역폭** | Decode 처리량 2.4배 향상 → 추론 latency 대폭 감소 |
| **Tensor Memory (256KB/SM)** | MMA accumulator 전용 메모리 → register 압력 해소 |
| **2-CTA MMA** | 큰 타일의 MMA를 2개 CTA가 협력 실행 → SMEM 트래픽 절반 |
| **듀얼 다이** | 2개 다이를 10 TB/s로 연결 → 사실상 하나의 거대 GPU |

---

# 6. 비대칭 스케일링: 왜 알고리즘이 중요한가

세대별로 하드웨어 발전 속도가 **균일하지 않다**는 것이 핵심이다.

| 하드웨어 | A100 → H100 | H100 → B200 | 2세대 합산 |
|---------|-------------|-------------|---------|
| **Tensor Core (BF16)** | 3.2× | 2.3× | **7.2×** |
| **메모리 대역폭** | 1.7× | 2.4× | **4.0×** |
| **MUFU (exp 등)** | ~1× | ~1× | **~1×** |
| **SMEM 대역폭** | ~1× | ~1× | **~1×** |

Tensor Core는 2세대에 걸쳐 **7배** 빨라졌지만, softmax의 exp를 계산하는 MUFU는 **그대로**이다. 이것이 의미하는 바:

1. **단순히 GPU를 바꾸는 것만으로는 성능이 비례 증가하지 않는다.** Tensor Core가 7배 빨라져도, exp가 병목이면 전체 속도는 제한된다.
2. **FlashAttention 같은 IO/compute-aware 알고리즘이 점점 더 중요해진다.** 하드웨어의 비대칭을 소프트웨어로 보상해야 한다.
3. **세대별로 다른 최적화 전략이 필요하다.** FA1/FA2는 HBM IO를 줄이고, FA3는 GEMM과 softmax를 겹치고, FA4는 exp를 소프트웨어로 에뮬레이션한다.

---

# 7. 실무 가이드라인

## GPU 선택

| 상황 | 추천 |
|------|------|
| 7B 모델 fine-tuning | A100 80GB 1장이면 충분 |
| 70B 모델 학습 | H100 8장 (DGX H100) 이상 |
| 70B 모델 추론 (저비용) | A100 + INT4 양자화 |
| 70B 모델 추론 (고성능) | H100 + FP8 |
| 405B 모델 학습 | H100/B200 수백 장 |
| 128K+ long context 추론 | **B200** (192GB 메모리 필수) |

## 정밀도 선택

| 정밀도 | 학습 | 추론 | 주의사항 |
|--------|------|------|---------|
| BF16 | ✅ 기본 | ✅ 안정적 | — |
| FP8 | ⚠️ H100+ | ✅ 추천 | Transformer Engine 필요 |
| INT8 | ❌ | ✅ | GPTQ/AWQ 양자화 필요 |
| INT4 | ❌ | ✅ 비용 효율 | 품질 저하 모니터링 필요 |
| FP4 | ❌ | ⚠️ B200만 | 아직 초기 단계 |

## FlashAttention 버전 선택

| GPU | 추천 FA 버전 | 이유 |
|-----|-----------|------|
| A100 | FA2 | A100에 최적화된 알고리즘 |
| H100 | FA3 (또는 FA2) | WGMMA/TMA 활용, FP8 지원 |
| B200 | FA4 | Blackwell 전용 파이프라인, SW exp |

---

# 마치며

GPU 아키텍처를 이해하면 "왜 이 설정에서 성능이 안 나오는지", "왜 이 최적화가 효과적인지"를 **근본적으로** 이해할 수 있다. LLM 엔지니어에게 가장 중요한 insight는:

1. **LLM 학습은 BF16 Tensor Core 성능에 의해 결정**되지만, 실제 활용률은 메모리 대역폭과 IO 패턴에 의해 제한된다.
2. **LLM 추론(decode)은 거의 순수하게 메모리 대역폭에 의해 결정**된다. Tensor Core 성능은 decode에서 거의 의미가 없다.
3. **하드웨어가 비대칭적으로 발전**하면서, 소프트웨어 최적화(FlashAttention, 양자화, KV cache 관리)의 중요성이 점점 커지고 있다.

> GPU별 최적화 사례가 궁금하다면: [FlashAttention (A100)](/blog/2023/fastattention/), [FlashAttention-2 (A100)](/blog/2023/flashattention-2/), [FlashAttention-3 (H100)](/blog/2026/flashattention-3/), [FlashAttention-4 (B200)](/blog/2026/flashattention-4/)를 참고하자.

---

# 참고 문헌

- [Comparing Blackwell vs Hopper — Exxact](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [NVIDIA B200 B100 H200 H100 A100 Comparison — BIZON](https://bizon-tech.com/blog/nvidia-b200-b100-h200-h100-a100-comparison)
- [NVIDIA Data Center GPU Specs — IntuitionLabs](https://intuitionlabs.ai/articles/nvidia-data-center-gpu-specs)
- [Blackwell vs Hopper GPU Architecture Comparison — IntuitionLabs](https://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison)
- [NVIDIA B200 vs H100 vs A100 — FaceOfIT](https://www.faceofit.com/nvidia-b200-vs-h100-vs-a100/)
- [NVIDIA H100s and H200s vs B100s and B200s — Modal](https://modal.com/blog/h100-and-h200-vs-b100-and-b200)
- [NVIDIA Blackwell Architecture Technical Brief](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief)
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
