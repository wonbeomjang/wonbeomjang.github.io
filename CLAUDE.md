# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

# 이 저장소의 성격 (중요)

`AGENTS.md`는 **al-folio 스타터 저장소 자체**를 개발하는 사람을 위한 문서다. 이 저장소는 그 스타터로 만든 **개인 사이트**이므로 차이를 알고 읽어야 한다.

- `AGENTS.md`의 "Stop sign"(=`_layouts/`, `_includes/`, `_sass/` 금지)은 **스타터 저장소에만** 적용된다. 개인 사이트는 gem 파일을 로컬에서 덮어쓰는 것이 정식으로 허용된다(upstream `docs/ARCHITECTURE.md`의 "local overrides: your site vs. this repo").
- 이 사이트는 아래 6개를 **의도적 override**로 유지한다. 지우지 말 것.

| 파일                             | 이유                                 |
| -------------------------------- | ------------------------------------ |
| `_layouts/default.html`          | 브랜딩, AdSense 제거                 |
| `_includes/head.liquid`          | PageSpeed 최적화 (defer, CLS 수정)   |
| `_includes/scripts.liquid`       | 렌더 블로킹 스크립트 defer           |
| `_includes/metadata.liquid`      | SEO BlogPosting schema + image alt   |
| `_includes/related_posts.liquid` | 카테고리 기반 관련글 + 태그 fallback |
| `assets/css/main.scss`           | Triton 시리즈 스타일                 |

- 그래서 upstream의 `unit-tests.yml`(`lint:style-contract`)은 이 저장소에서 **반드시 실패한다** — 그 워크플로는 제거했다. al-folio 유지보수 전용 워크플로(`star-history`, `update-screenshots`, `release`, `visual-regression`)도 같은 이유로 제거했다.

- `baseurl`은 **비어 있다**(커스텀 도메인 `www.wonbeomjang.kr`). upstream의 `/al-folio` 관련 지침은 그대로 적용하지 말 것.

# al-folio v1 구조

테마 런타임은 `al_folio_core` 등 gem에 있고, `_config.yml`의 `theme:`와 `plugins:`, `Gemfile`의 gem 목록이 **함께 맞아야** 동작한다. 한쪽에만 있으면 조용히 비활성화된다.

override를 손댔거나 gem을 올린 뒤에는 드리프트를 점검한다.

```bash
bundle exec al-folio upgrade audit --no-fail
bundle exec al-folio upgrade overrides audit
bundle exec al-folio upgrade overrides diff <path>   # 확인 후 accept
```

# Python 환경

- Python 패키지가 필요할 때는 conda 환경을 사용할 것
- conda 경로: `~/miniforge3/bin/conda`
- 환경 활성화: `eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate <env_name>`
- 기본 환경 이름: `blog` (python 3.11, pymupdf 등 설치됨)

@\_rules/POST.md
@\_rules/IMPL.md
