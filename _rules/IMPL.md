# 구현/실험 포스트 작성 스타일 가이드

Triton 시리즈처럼 코드 구현과 실험을 중심으로 하는 포스트에 적용한다.

## 포스트 유형 구분

| 유형              | 설명                                  | 예시                                  |
| ----------------- | ------------------------------------- | ------------------------------------- |
| **입문 커널**     | 개념 학습 중심, Triton API 소개       | Triton 01~04 (Vector Add, Softmax 등) |
| **종합 프로젝트** | 이전 개념 총합 + 이론 심화 + Backward | Triton 05 (FA1)                       |
| **개선 비교**     | 이전 버전 대비 개선 항목 나열         | Triton 06 (FA2)                       |

---

## 구조

### 입문 커널 포스트

```
1. 개요: 이 포스트에서 무엇을 구현하는지 1~2문장
2. 핵심 개념: 배경 이론 + 수식 + 직관 설명
3. 커널 동작 원리: 그림 + 의사코드로 흐름 설명
4. 단계별 분석: 코드(Gist) + 라인별 설명
5. 사용된 Triton API 표: 기능 | 설명 형태
6. 벤치마크 결과: 그림 + 수치 해석
7. 전체 코드: Gist 임베드
```

### 종합 프로젝트 포스트 (FA 수준)

```
1. 개요: 목표 + 관련 논문 리뷰 링크
2. 핵심 개념: 이론/수식 (배경지식 필요 시)
3. 커널 동작 원리: 그림 + 의사코드
4. Forward 구현: 단계별 코드(Gist) + 설명
5. Backward 구현: 수식 → 커널 구조 → 코드(Gist)
6. GPU 아키텍처별 최적화: 아키텍처 비교 표 + 적용 방법
7. 튜토리얼 연결 표: 이전 포스트에서 배운 개념이 여기서 어떻게 쓰이는지
8. 벤치마크 결과: 그림 + 수치 해석
9. 전체 코드: Gist 임베드
10. 시리즈 링크: 인용구로 앞뒤 포스트 연결
11. 참고 문헌
```

### 개선 비교 포스트 (FA2 수준)

```
1. 이전 포스트 링크 (첫 줄)
2. 개선 요약 표: # | 항목 | 이전 | 이후 | 효과
3. 개선 1~N: 수식 → Gist 코드 → 직관적 해석
4. Backward 구현 (변경된 부분만)
5. 벤치마크 결과: 이전 버전과 비교 + 수치
6. 전체 코드: Gist 임베드
7. 시리즈 링크: 인용구
8. 참고 문헌
```

---

## 핵심 원칙

1. **코드는 반드시 Gist 임베드**: 인라인 코드 블록 금지, 예외는 3줄 이하 짧은 예시
2. **그림은 `figure.liquid` 태그 사용**: `{% include figure.liquid loading="lazy" path="..." class="img-fluid rounded z-depth-1" alt="..." %}`
3. **수식 바로 뒤에 직관 해석**: 수식만 던지지 않고 "이것이 실제로 무엇을 의미하는지" 설명
4. **구체적 수치 언급**: "BLOCK_M을 64→128로 바꾸는 것만으로 ~15–20% 향상", "나눗셈 횟수: FA1은 T_c번, FA2는 1번"
5. **섹션 구분자 `---`**: 각 섹션 사이에 반드시 삽입
6. **개선 비교는 표로 먼저**: 세부 설명 전에 요약 표 제시
7. **한다체 통일**: 개요 포함 전체 포스트에서 `합니다체` 금지, `~한다` 체로 통일
8. **시리즈 상호 링크**: 관련 포스트 인용구로 안내 (`> ... [링크] ...를 참고하자.`)

---

## Front Matter 템플릿

```yaml
---
layout: post
title: "Triton NN: 제목 — 부제"
date: YYYY-MM-DD HH:MM:SS +0900
description: "한국어 한줄 설명 — 핵심 키워드"
categories: [triton]
tags: [triton, gpu, 관련태그]
giscus_comments: true
related_posts: true
featured: true # 종합 프로젝트·개선 비교 포스트에만
---
```

---

## 이미지 경로 규칙

- 저장 위치: `/assets/img/triton/{포스트번호}_{슬러그}/`
- 예: `/assets/img/triton/05_flash_attention/benchmark.png`

---

## Gist 파일명 규칙

```
{포스트번호}_{포스트슬러그}_snippet{번호}_{내용설명}.py
```

예: `05_flash_attention_snippet01_standard_attention.py`

---

## 현재 사용 중인 Gist

| Gist ID                            | 용도                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `42cd2b629a46d83e348bc15c5aa83a17` | Triton 05 (FA1) 스니펫 모음                                                               |
| `5880faa2b9aa8d0ab1bd1dd0ad31baa9` | Triton 06 (FA2) 스니펫 모음                                                               |
| `2231bb41af1f36d52e143c60386cf7a0` | Triton 07 (FA3) 스니펫 모음                                                               |
| `0f4970e5dbed9af5037d796fa395727f` | Triton 전체 코드 (`flash_attention.py`, `flash_attention_v2.py`, `flash_attention_v3.py`) |

---

## 벤치마크 결과 작성 요령

- 그림 먼저, 바로 아래에 불릿포인트로 해석
- 비교 대상 명시: "RTX 4080 기준 FA1 vs FA2 vs PyTorch"
- 구체적 배율·퍼센트 언급: "FA2가 FA1 대비 ~1.7–1.8× 빠름"
- 조건별 차이 설명: "seq_len이 길수록 FA2 우위가 커짐"

---

## GPU 아키텍처 비교 표 (재사용 가능)

| GPU           | 아키텍처     | SM당 SRAM | 권장 BLOCK_M | 권장 BLOCK_N |
| ------------- | ------------ | --------- | ------------ | ------------ |
| RTX 4080/4090 | Ada Lovelace | ~100 KB   | 64           | 64           |
| A100          | Ampere       | 192 KB    | 128          | 64           |
| H100          | Hopper       | 228 KB    | 128          | 64–128       |
| B200          | Blackwell    | 232 KB+   | 192+         | 128+         |

---

## 커밋/푸시 전 체크리스트

### prettier (CI 차단 요소 — 매번 필수)

GitHub Actions의 "Run npx prettier . --check"가 통과하지 못하면 빌드가 실패한다. 다음 절차를 **모든 commit 전에 반드시 수행**한다.

```bash
# 1. 전체 repo 기준으로 자동 포맷 (변경한 파일만이 아니라 . 전체!)
npx prettier --write .

# 2. 검증 — 통과해야 push
npx prettier --check .
```

**핵심 원칙**

- 내가 만진 파일만 포맷하면 안 된다. CI는 `.` 전체를 검사하므로 **항상 `npx prettier --write .`** 를 돌린다.
- 무관한 파일이 함께 포맷되더라도 같은 commit에 포함시켜라 — 이전 누락분의 누적 결과다.
- prettier가 없으면 `conda install -c conda-forge nodejs` 후 `npm install --save-dev prettier@3 @shopify/prettier-plugin-liquid`.

**자주 걸리는 함정**

- Markdown TOC의 underscore: `[_config.yml](#anchor)` → `[\_config.yml](#anchor)`로 escape해야 한다 (prettier가 자동 escape).
- description에 콜론(`:`)이 포함되면 따옴표로 감싸기.
- 표(Table) 컬럼 너비 정렬은 prettier가 자동으로 손본다 — 수동 정렬에 시간을 쓰지 말 것.

### gh / commit

- `gh` CLI는 fa-triton conda env에 설치되어 있고 인증되어 있다. gist 업로드는 `gh gist edit <id> --filename <name> <path>`.
- commit author가 자동 감지 안 되면 `git -c user.name="Wonbeom Jang" -c user.email="jtiger958@gmail.com" commit ...`으로 1회용 지정 (글로벌 git config는 절대 수정하지 않는다).
