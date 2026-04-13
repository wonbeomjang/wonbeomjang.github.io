@AGENTS.md

# Python 환경

- Python 패키지가 필요할 때는 conda 환경을 사용할 것
- conda 경로: `~/miniforge3/bin/conda`
- 환경 활성화: `eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate <env_name>`
- 기본 환경 이름: `blog` (python 3.11, pymupdf 등 설치됨)

# 블로그 포스트 작성 스타일 가이드

논문 리뷰 포스트를 작성할 때 아래 스타일을 따른다. (참고: FlashAttention-3, TelBench 포스트)

## 구조: "왜 → 무엇 → 어떻게 → 결과"

```
1. Introduction: 문제 정의 + 기존 한계 + 이 논문의 해결책 요약 (결론 미리 제시)
2. Background: 독자가 알아야 할 배경 지식 (수식 + 직관 설명)
3. Method: 단계별 상세 설명 (Algorithm, 수식, 그림)
4. Experiments: 결과 표/그림 + 해석 (숫자 언급 필수)
5. Conclusion: 핵심 메시지 한 줄 + 한계점
6. 참고 문헌: 참조한 링크 목록
```

## 핵심 원칙

1. **표(Table)로 핵심 비교**: 복잡한 비교는 반드시 표로 정리한다
2. **수식 바로 뒤에 직관적 해석**: 수식만 던지지 않고, "이것이 실제로 무엇을 의미하는지" 설명한다
3. **구체적 숫자/예시**: 추상적 설명 대신 "256배 빠르다", "레귤러 괜찮나요?" 같은 구체적 수치와 예시
4. **그림 → 설명 → 표 → 해석 반복**: 시각 자료 먼저, 바로 아래에서 핵심 관찰을 불릿포인트로 정리
5. **반말(~한다) 통일**: 전체 포스트에서 문체 통일
6. **논문 링크는 제목으로**: `> [Full Paper Title](arxiv_url)` 형식으로 첫 줄에 배치
7. **시리즈 글은 상호 링크**: 관련 포스트를 마지막에 인용구로 안내

## 카테고리 규칙 (5개만 사용)

| 카테고리  | 용도                                        | 예시                                           |
| --------- | ------------------------------------------- | ---------------------------------------------- |
| `paper`   | 모든 논문 리뷰 (CV, LLM, optimization 무관) | FlashAttention, LoRA, TelBench, SENet          |
| `dev`     | 개발 가이드, 기술 설명, 트러블슈팅          | GPU 아키텍처, Quantization 적용, Keras, Jetson |
| `triton`  | Triton GPU 프로그래밍 시리즈                | Triton 00~05                                   |
| `review`  | 회고, 후기                                  | 연말 회고, 인턴 후기                           |
| `project` | 프로젝트 기록                               | OSAM, 약학정보 서비스                          |

- 논문 리뷰는 주제(LLM, CV, NLP 등)와 무관하게 **모두 `[paper]`**
- 주제 구분은 카테고리가 아닌 **tags**로 한다

## Front Matter 템플릿

```yaml
---
layout: post
title: "논문 제목"
date: YYYY-MM-DD HH:MM:SS +0900
description: "한국어 한줄 설명 — 핵심 키워드"
categories: [paper]
tags: [tag1, tag2, paper]
giscus_comments: true
related_posts: true
---
```

## 포스트 작성 프로세스

1. **웹검색으로 논문/자료를 충분히 파악**: arxiv abstract → PDF 읽기 → 블로그/해설 글 참조
2. **논문 그림 다운로드**: arxiv HTML 버전(`arxiv.org/html/`)에서 이미지 URL을 추출하여 다운로드
3. **참고 문헌 섹션 필수**: 포스트 마지막에 `# 참고 문헌` 섹션을 추가하고, 참조한 모든 링크를 나열

## 커밋 전 체크리스트

- `npx prettier . --write` 실행하여 포맷 검사 통과
- description에 콜론(`:`)이 포함되면 따옴표로 감싸기
- 이미지는 `/assets/post/image/{post-slug}/` 디렉토리에 저장
