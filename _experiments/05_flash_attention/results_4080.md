# Flash Attention v1 — RTX 4080 측정 결과

`flash_attention.py` 의 `main()` 을 RTX 4080 단일 GPU 에서 실행한 결과.
[results_a100.md](results_a100.md) 와 동일한 코드/seed 로 측정 → apples-to-apples 비교.

## 실행 환경

| 항목     | 값                                            |
| -------- | --------------------------------------------- |
| GPU      | NVIDIA GeForce RTX 4080 (cc 8.9, 15.6 GB)     |
| Driver   | 580.126.09                                    |
| PyTorch  | 2.6.0+cu124                                   |
| Triton   | 3.2.0                                         |
| Conda env| `triton`                                      |
| Date     | 2026-05-01                                    |

```bash
conda activate triton
cd _experiments/05_flash_attention
python flash_attention.py
```

## Forward 벤치마크 (non-causal, num_heads=16, head_dim=64, fp16)

| Seq  | Triton (ms) | PyTorch (ms) | Speedup   |
| ---- | ----------- | ------------ | --------- |
| 256  | 0.0384      | 0.0267       | 0.69×     |
| 512  | 0.0464      | 0.0487       | 1.05×     |
| 1024 | 0.0935      | 0.1840       | 1.97×     |
| 2048 | 0.2464      | 1.6601       | **6.74×** |
| 4096 | 0.8763      | 5.3870       | **6.15×** |

## 메모리 사용량 (non-causal)

| Seq  | Standard (MB) | Flash (MB) | 절약 (MB)  |
| ---- | ------------- | ---------- | ---------- |
| 1024 | 84.1          | 22.1       | 62.0       |
| 2048 | 286.1         | 32.1       | 254.0      |
| 4096 | 1072.1        | 52.1       | **1020.0** |

## 정확도 검증 결과

- **Forward + LSE + Backward 모두 통과** (fp16, atol=1e-1, rtol=1e-2; backward atol=5e-2)
- LSE max diff: 9.54e-07
- 4080 의 cc 8.9 (Ada Lovelace) 에서 Tensor Core fp16 경로가 정상 동작

## RTX 4080 vs A100 비교 (non-causal forward, head_dim=64)

| Seq  | 4080 FA1 (ms) | A100 FA1 (ms) | A100/4080 |
| ---- | ------------- | ------------- | --------- |
| 256  | 0.038         | 0.087         | 0.44× *(4080 우세)* |
| 512  | 0.046         | 0.083         | 0.55× *(4080 우세)* |
| 1024 | 0.094         | 0.112         | 0.84× |
| 2048 | 0.246         | 0.234         | 1.05× |
| 4096 | 0.876         | 0.746         | 1.17× |

- **짧은 seq (≤1024) 에서는 RTX 4080 이 더 빠르다** — A100 에 비해 SM 클럭이 높고(2505 vs 1410 MHz) launch latency 가 작은 영향.
- **긴 seq (≥2048) 부터 A100 이 우세** — 1.5 TB/s HBM2e 대역폭이 4080 의 717 GB/s GDDR6X 를 추월.
- 추세는 [Triton 05 포스트](../../_posts/2026-04-02-triton-05-flash-attention.md#왜-a100-peak-대비-37밖에-안-나오나) 의 FA1 분석과 일치 — 메모리 bound 일수록 A100 이 유리.
