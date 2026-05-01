# Flash Attention v3 — RTX 4080 측정 결과

`flash_attention_v3.py` 의 `main()` 을 RTX 4080 단일 GPU 에서 실행한 결과.
[results_a100.md](results_a100.md) 와 동일한 코드/seed 로 측정 → apples-to-apples 비교.

## 실행 환경

| 항목     | 값                                            |
| -------- | --------------------------------------------- |
| GPU      | NVIDIA GeForce RTX 4080 (cc 8.9, 15.6 GB)     |
| Driver   | 580.126.09                                    |
| PyTorch  | 2.6.0+cu124                                   |
| Triton   | 3.2.0 ⚠️                                       |
| Conda env| `triton`                                      |
| Date     | 2026-05-01                                    |

> **⚠️ Triton 3.2.0 한계**: `tl.range(..., warp_specialize=...)` 키워드는 Triton ≥ 3.5 에서만 지원된다. 본 환경에서는 `_HAS_WARP_SPECIALIZE = False` 로 검출돼 `_fa3_inner` 가 plain `tl.range` 분기로 정의된다 (코드 변경 불필요 — 모듈 로드 시 자동 분기).
> **결과적으로 FA3 와 FA3-WS 가 동일 커널을 호출**하므로 아래 표의 FA3-WS 컬럼은 사실상 FA3 의 재측정값이다. 진짜 warpspec 비교는 Triton ≥ 3.5 환경에서 가능 (A100 결과 참고).

```bash
conda activate triton
cd _experiments/07_flash_attention_v3
python flash_attention_v3.py
```

## Forward 벤치마크 (causal, num_heads=16, head_dim=64, fp16)

| Seq  | FA1 (ms) | FA2 (ms) | FA3 (ms) | FA3-WS (ms)\* | PyTorch (ms) | FA3/FA2 | FA3/PT     |
| ---- | -------- | -------- | -------- | ------------- | ------------ | ------- | ---------- |
| 256  | 0.0369   | 0.0473   | 0.0471   | 0.0466        | 0.0440       | 1.00×   | 0.93×      |
| 512  | 0.0432   | 0.0519   | 0.0524   | 0.0518        | 0.0679       | 0.99×   | 1.30×      |
| 1024 | 0.0747   | 0.0781   | 0.0780   | 0.0784        | 0.3052       | 1.00×   | 3.91×      |
| 2048 | 0.1673   | 0.1572   | 0.1606   | 0.1634        | 2.7387       | 0.98×   | **17.06×** |
| 4096 | 0.5058   | 0.4405   | 0.4596   | 0.4492        | 9.2261       | 0.96×   | **20.07×** |

\* Triton 3.2 한계로 FA3 와 동일 커널 호출. 실제 warpspec 효과는 측정 불가.

## 정확도 검증 결과

- FA3 forward (3D + 4D + head_dim=128): 모두 통과
- FA3-WS forward: 모두 통과 (정확도는 FA3 와 동일 — 같은 커널이므로)
- FA3 backward (dQ, dK, dV): 모두 통과 (atol=5e-2)
- LSE max diff: 9.54e-07

## 관찰

1. **FA3 ≈ FA2 — 4080 에서도 블로그 결론 재현**

   | Seq | FA3/FA2 (4080) | FA3/FA2 (A100) |
   | --- | -------------- | -------------- |
   | 1024 | 1.00× | 0.88× |
   | 2048 | 0.98× | 0.95× |
   | 4096 | 0.96× | 1.13× |

   짧은 seq 에서 FA3 의 17 configs autotune 이 FA2 의 6 configs 보다 best 를 못 찾는 경향. [Triton 07 포스트](../../_posts/2026-04-02-triton-07-flash-attention-v3.md) 의 "+3-5% 한계" 결론에 부합.

2. **causal seq=4096 에서 FA3/PT = 20.07×** — `_experiments/` 짧은 seq 범위에서 도달한 최대 가속.

3. **FA3-WS 측정 불가** — Triton 3.2 환경에서는 의미 있는 비교 불가. Triton ≥ 3.5 + cc 9 (Hopper) 환경에서 본격 효과를 확인할 수 있다.

## RTX 4080 vs A100 비교 (FA3, causal forward, head_dim=64)

| Seq  | 4080 FA3 (ms) | A100 FA3 (ms) | A100/4080 |
| ---- | ------------- | ------------- | --------- |
| 256  | 0.047         | 0.105         | 0.45× *(4080 우세)* |
| 512  | 0.052         | 0.105         | 0.50× *(4080 우세)* |
| 1024 | 0.078         | 0.121         | 0.64× *(4080 우세)* |
| 2048 | 0.161         | 0.181         | 0.89× |
| 4096 | 0.460         | 0.402         | 1.14× |

- **seq ≤ 2048 까지 4080 이 우세** — Ada 의 높은 SM 클럭(2505 MHz) + launch overhead 작음.
- seq=4096 부터 A100 우세 — HBM2e 1.5 TB/s vs GDDR6X 717 GB/s.
- 추세는 FA1/FA2 에서 본 패턴과 동일 — long-seq 메모리 bound 일수록 A100 유리.
