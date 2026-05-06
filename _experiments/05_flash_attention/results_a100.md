# Flash Attention v1 — A100 측정 결과

`flash_attention.py` 의 `main()` 을 A100 단일 GPU 에서 실행한 결과.

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
cd _experiments/05_flash_attention
python flash_attention.py
```

## Forward 벤치마크 (non-causal, num_heads=16, head_dim=64, fp16)

| Seq  | Triton (ms) | PyTorch (ms) | Speedup   |
| ---- | ----------- | ------------ | --------- |
| 256  | 0.087       | 0.111        | 1.26×     |
| 512  | 0.083       | 0.114        | 1.37×     |
| 1024 | 0.112       | 0.265        | 2.36×     |
| 2048 | 0.234       | 0.891        | **3.81×** |
| 4096 | 0.746       | 3.419        | **4.58×** |

## 메모리 사용량 (non-causal)

| Seq  | Standard (MB) | Flash (MB) | 절약 (MB)  |
| ---- | ------------- | ---------- | ---------- |
| 1024 | 84.1          | 22.1       | 62.0       |
| 2048 | 286.1         | 32.1       | 254.0      |
| 4096 | 1072.1        | 52.1       | **1020.0** |

## 정확도 검증 결과

- **Forward + LSE + Backward 모두 통과** (fp16, atol=1e-1, rtol=1e-2)
- LSE max diff: 9.54e-07

## 비교 노트

같은 코드를 RTX 4080 등 다른 GPU 에서 실행하면 본 표와 직접 비교 가능. 결과는 [\_experiments/05_flash_attention/results_4080.md](results_4080.md) 등으로 추가.
