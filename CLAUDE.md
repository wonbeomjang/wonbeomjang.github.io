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

## 카테고리 규칙 (6개만 사용)

| 카테고리  | 용도                                        | 예시                                           |
| --------- | ------------------------------------------- | ---------------------------------------------- |
| `paper`   | 모든 논문 리뷰 (CV, LLM, optimization 무관) | FlashAttention, LoRA, TelBench, SENet          |
| `dev`     | 개발 가이드, 기술 설명, 트러블슈팅          | GPU 아키텍처, Quantization 적용, Keras, Jetson |
| `infra`   | 인프라, DevOps, 클라우드 관련 가이드        | EKS, ECR, Kubernetes, Docker, CI/CD            |
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

## 코드 삽입 규칙 (Triton / 구현 포스트)

포스트에 코드를 삽입할 때는 인라인 코드 블록 대신 **반드시 GitHub Gist 임베드**를 사용한다.

### Gist 업로드 절차

1. [gist.github.com](https://gist.github.com) 에서 새 gist 생성 (또는 기존 gist에 파일 추가)
2. 파일명 규칙: `{포스트번호}_{포스트슬러그}_snippet{번호}_{내용설명}.py`
   - 예: `09_flash_attention_v2_snippet01_un_scaled_accumulation.py`
3. 포스트에 임베드할 때 `<script>` 태그 사용:

```html
<script src="https://gist.github.com/{username}/{gist_id}.js?file={filename}"></script>
```

### 현재 사용 중인 Gist

| Gist ID                            | 용도                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `42cd2b629a46d83e348bc15c5aa83a17` | Triton 05 (FA1) 스니펫 모음                                                               |
| `5880faa2b9aa8d0ab1bd1dd0ad31baa9` | Triton 06 (FA2) 스니펫 모음                                                               |
| `2231bb41af1f36d52e143c60386cf7a0` | Triton 07 (FA3) 스니펫 모음                                                               |
| `0f4970e5dbed9af5037d796fa395727f` | Triton 전체 코드 (`flash_attention.py`, `flash_attention_v2.py`, `flash_attention_v3.py`) |

### 주의사항

- gh CLI가 인증된 경우 AI가 직접 `gh gist create` / `gh gist edit --add`로 Gist를 생성·수정할 수 있다
- gh CLI가 없거나 미인증 시에는 코드 초안을 인라인으로 작성하고, Gist URL을 알게 된 후 `<script>` 태그로 교체한다
- 이미지는 Triton 시리즈는 `/assets/img/triton/{포스트번호}_{슬러그}/`, 논문 리뷰는 `/assets/post/image/{post-slug}/` 에 저장

## 커밋/푸시 전 체크리스트

### prettier (CI 차단 요소 — 매번 필수)

GitHub Actions 의 "Run npx prettier . --check" 가 통과하지 못하면 빌드가 실패한다. 다음 절차를 **모든 commit 전에 반드시 수행**한다.

```bash
# 1. 전체 repo 기준으로 자동 포맷 (변경한 파일만이 아니라 . 전체!)
npx prettier --write .

# 2. 검증 — 통과해야 push
npx prettier --check .
```

**핵심 원칙**

- 내가 만진 파일만 포맷하면 안 된다. CI 는 `.` 전체를 검사하므로 **항상 `npx prettier --write .`** 를 돌린다.
- 무관한 파일이 함께 포맷되더라도 같은 commit 에 포함시켜라 — 이전 누락분의 누적 결과다.
- prettier 가 없으면 `conda install -c conda-forge nodejs` 후 `npm install --save-dev prettier@3 @shopify/prettier-plugin-liquid`.
- `.prettierignore` 에 빌드 산출물·실험 디렉토리·`.claude/` 가 들어 있는지 확인. 누락된 게 있으면 추가.

**자주 걸리는 함정**

- Markdown TOC 의 underscore: `[_config.yml](#anchor)` → `[\_config.yml](#anchor)` 로 escape 해야 한다 (prettier가 자동 escape).
- description 에 콜론(`:`) 이 포함되면 따옴표로 감싸기.
- 표(Table) 컬럼 너비 정렬은 prettier 가 자동으로 손본다 — 수동 정렬에 시간을 쓰지 말 것.

### gh / commit / push

- `gh` CLI 는 fa-triton conda env 에 설치되어 있고 인증되어 있다. gist 업로드는 `gh gist edit <id> --filename <name> <path>`.
- commit author 가 자동 감지 안 되면 `git -c user.name="Wonbeom Jang" -c user.email="jtiger958@gmail.com" commit ...` 로 1회용 지정 (글로벌 git config 는 절대 수정하지 않는다).
