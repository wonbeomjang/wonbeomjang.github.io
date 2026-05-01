# Flash Attention v3 — A100 측정 결과

`flash_attention_v3.py` 의 `main()` 을 A100 단일 GPU 에서 실행한 결과.

## 실행 환경

| 항목    | 값                                         |
| ------- | ------------------------------------------ |
| GPU     | NVIDIA A100-SXM4-80GB (cc 8.0, 79.3 GB)    |
| Driver  | 580.105.08                                 |
| PyTorch | 2.11.0+cu130                               |
| Triton  | 3.6.0                                      |
| Job     | SLURM 87010 (partition=single, gres=gpu:1) |
| Date    | 2026-05-01                                 |

```bash
cd _experiments/07_flash_attention_v3
python flash_attention_v3.py
```

## Forward 벤치마크 (causal, num_heads=16, head_dim=64, fp16)

| Seq  | FA1 (ms) | FA2 (ms) | FA3 (ms) | FA3-WS (ms) | PyTorch (ms) | FA3/WS | FA3/PT     |
| ---- | -------- | -------- | -------- | ----------- | ------------ | ------ | ---------- |
| 256  | 0.086    | 0.107    | 0.105    | 0.106       | 0.177        | 1.00×  | 1.68×      |
| 512  | 0.081    | 0.106    | 0.105    | 0.106       | 0.178        | 0.98×  | 1.70×      |
| 1024 | 0.104    | 0.107    | 0.121    | 0.123       | 0.386        | 0.98×  | 3.20×      |
| 2048 | 0.175    | 0.162    | 0.181    | 0.199       | 1.355        | 0.91×  | **7.48×**  |
| 4096 | 0.437    | 0.361    | 0.402    | 0.511       | 5.243        | 0.79×  | **13.03×** |

## 정확도 검증 결과

- FA3 forward (3D + 4D + head_dim=128): 모두 통과
- FA3-WS forward: 모두 통과 (정확도는 FA3 와 동일)
- FA3 backward (dQ, dK, dV): 모두 통과
- LSE max diff: 9.54e-07

## 관찰

1. **FA3-WS (`tl.range(warp_specialize=True)`) — A100 에서 효과 없음**
   - seq≤512 에서는 FA3 와 동일 (1.00×)
   - seq=4096 에서는 오히려 -21% 손해 (FA3=0.40, FA3-WS=0.51)
   - 원인: A100 에는 TMA·MBARRIER 가 없어 warpspec pass 가 partition 만 만들고 latency 가림 효과가 없음. partition 분할 비용만 남아 손해.
   - H100 에서 본격 효과 기대 ([Triton 07 포스트](../../_posts/2026-04-02-triton-07-flash-attention-v3.md) 의 limitation 섹션 참고).

2. **FA3 vs FA2 — short seq 에서는 FA3 가 손해**
   - seq=1024: FA3=0.121, FA2=0.107 (FA3 가 13% 느림)
   - seq=4096: FA3=0.402, FA2=0.361 (FA3 가 11% 느림)
   - 원인: FA3 의 17 configs autotune 이 짧은 seq 에서 best 가 아닌 config 를 선택할 가능성. 더 긴 seq (≥8192) 에서는 multi-GPU 측정에서 FA3 가 우세 (포스트 참고).

3. **causal seq=4096 에서 FA3/PT = 13.03×** — `_experiments/` main() 의 짧은 seq 범위에서 도달한 최대 가속.

## 비교 노트

같은 코드를 RTX 4080 등 다른 GPU 에서 실행하면 본 표와 직접 비교 가능. 결과는 `results_4080.md` 등으로 추가.

더 긴 seq (8K~32K) 에서의 추세는 [Triton 07 포스트](../../_posts/2026-04-02-triton-07-flash-attention-v3.md#벤치마크-결과-a100-sxm4-80gb--4) 에 multi-GPU 4-GPU 평균 결과로 정리됨.
