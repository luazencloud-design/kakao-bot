# 🤖 카카오 RAG 챗봇 + 운영자 어드민 — 인수인계 문서

> 카카오톡 채널에서 동작하는 **RAG 챗봇**. 수강생이 카톡으로 질문하면, 운영자가 등록한 자료(PDF·PPTX·HWP·TXT·VTT·오디오)에서 관련 내용을 검색해 **Gemini**가 출처와 함께 답한다. 비개발자 운영자는 **웹 어드민**에서 자료를 직접 올리고 관리한다.

이 문서 하나로 **세팅 → 실행 → 구조 이해 → 운영**까지 가능하도록 정리했다. 처음 보는 개발자가 바로 손댈 수 있게 쓰는 게 목표.

- **레포:** `luazencloud-design/kakao-bot`
- **배포:** 봇·어드민 둘 다 **Vercel**, 데이터는 **Supabase**, 생성·임베딩은 **Google Gemini**
- **도메인 특화:** 사업자등록·오픈마켓(쿠팡·스마트스토어 등) 가입·강의 안내 (시스템 프롬프트가 도메인 특화 — `src/rag.js`)

---

## 📑 목차

1. [시스템 한눈에](#1-시스템-한눈에)
2. [전체 구조](#2-전체-구조)
3. [기술 스택](#3-기술-스택)
4. [폴더·파일 구조](#4-폴더파일-구조-인수인계-핵심) ← 인수인계 핵심
5. [처음부터 셋업](#5-처음부터-셋업-zero--실행)
6. [명령어 모음](#6-명령어-모음)
7. [동작 원리](#7-동작-원리-핵심-로직)
8. [데이터 모델 (Supabase)](#8-데이터-모델-supabase)
9. [배포](#9-배포)
10. [운영](#10-운영)
11. [트러블슈팅](#11-트러블슈팅)
12. [알려진 제약](#12-알려진-제약)
13. [문서 지도](#13-문서-지도)

---

## 1. 시스템 한눈에

```
수강생 ──카카오톡 질문──> 봇 ──검색──> 자료에서 관련 내용 찾기 ──> Gemini가 답변(+출처)
운영자 ──브라우저──────> 어드민 ──업로드──> 추출·청킹·임베딩 ──> 즉시 챗봇에 반영
```

- 챗봇은 **올려둔 자료에 있는 내용만** 답한다. 없으면 "자료에 포함되어 있지 않습니다"라고 솔직히 말한다(환각 방지).
- 자료를 올리면 **재배포 없이 즉시** 반영된다(데이터가 Supabase에 있으므로).
- **카카오 API를 호출하는 게 아니라, 카카오가 우리 봇의 웹훅을 호출한다.** 봇은 카카오가 보낸 질문 JSON을 받아 답을 만들어 카카오 형식 JSON으로 돌려준다. 카카오 인증키·메시지발송 API는 안 쓴다.

---

## 2. 전체 구조

```
[수강생] ──카카오톡──> 봇 (Express, Vercel)         ─┐
[운영자] ──브라우저──> 어드민 (Next.js 16, Vercel)   ─┼─> Supabase ───> Gemini
                                                      ─┘  (Postgres·     (임베딩·생성·
                                                          pgvector·       오디오 전사)
                                                          Storage·Auth)
```

| 구성요소 | 코드 | 호스팅 | 역할 |
|---|---|---|---|
| **봇** | `src/`, `api/` | Vercel | 카카오 웹훅 수신 → RAG로 답변 생성 |
| **어드민** | `admin/` | Vercel | 운영자용 자료 관리·답변 테스트·통계·피드백 UI |
| **Supabase** | `supabase/migrations/` | 클라우드(서울 리전) | Postgres + pgvector 검색 · 파일 Storage · 운영자 Auth · 질의 로그 |
| **Gemini** | (API) | Google | 답변 생성·쿼리 재작성(`gemini-flash-lite-latest`), 임베딩(`gemini-embedding-001`, 768차원), 오디오 전사(Files API) |

> 봇과 어드민은 **별개 Vercel 프로젝트**지만 **같은 Supabase**를 바라본다. 그래서 어드민에서 올린 자료가 봇에 바로 반영된다.

---

## 3. 기술 스택

- **봇:** Node.js(ESM) · Express · `@supabase/supabase-js` · `@vercel/functions`(waitUntil)
- **어드민:** Next.js 16 (App Router, **Turbopack**) · React 19 · TypeScript · Tailwind v4 · shadcn/ui · `@supabase/ssr` · react-dropzone · Vitest
- **DB:** Supabase Postgres 17 + `pgvector`(dense) + `pg_trgm`(한국어 sparse)
- **LLM:** Google Gemini — 생성/재작성 `gemini-flash-lite-latest`, 임베딩 `gemini-embedding-001`(768)

> ⚠️ **Next.js 16 주의:** `admin/AGENTS.md`에 적힌 대로, 이 버전은 기존과 API·관례가 다를 수 있다. 어드민 코드 수정 전 `admin/node_modules/next/dist/docs/`의 가이드를 확인할 것. (예: 미들웨어가 `proxy.ts`로 이름이 바뀜)

---

## 4. 폴더·파일 구조 (인수인계 핵심)

### 4.1 봇 (레포 루트)

```
kakao-bot/
├─ src/                     # 봇 본체 (Express)
│   ├─ app.js               #   Express 앱 정의 + 라우트. 동기/콜백 핸들러, waitUntil
│   ├─ rag.js               #   ★ RAG 코어: 재작성(캐시)→임베딩→하이브리드검색→생성→로깅
│   ├─ kakao.js             #   카카오 스킬 응답 포맷 헬퍼(simpleText, useCallback 등)
│   ├─ security.js          #   웹훅 보안: 시크릿 경로·rate limit·입력 가드
│   └─ server.js            #   로컬 개발용 진입점 (node src/server.js). Vercel에선 미사용
├─ api/
│   └─ index.js             # Vercel 서버리스 진입점 — src/app.js를 그대로 export
├─ vercel.json              # 봇 Vercel 설정 (모든 요청 → api/index, maxDuration 60)
├─ .env.example             # 봇 환경변수 템플릿 (이걸 복사해서 .env)
│
├─ supabase/
│   ├─ migrations/          # ★ DB 스키마 (0001~0005). 순서대로 적용
│   │   ├─ 0001_init.sql                    #   테이블·인덱스·GRANT·RLS·hybrid_search 초기
│   │   ├─ 0002_hybrid_search_korean.sql    #   sparse를 pg_trgm(한국어)로 교체
│   │   ├─ 0003_queries_resolved_at.sql     #   피드백 해결 워크플로용 컬럼
│   │   ├─ 0004_hybrid_search_deterministic.sql  #   RRF 동점 시 id로 결정적 정렬
│   │   └─ 0005_query_rewrites_cache.sql    #   재작성 캐시 테이블(검색 일관성)
│   ├─ README.md            # Supabase 셋업 안내
│   └─ check-connection.mjs # 연결 점검 스크립트
│
├─ scripts/                 # CLI 도구 (백업·예외 처리용. 평소엔 어드민으로 충분)
│   ├─ ocr.js               #   PDF·오디오 → 텍스트 (Gemini Files API)
│   ├─ ingest.js            #   추출→청킹→임베딩→Supabase upsert
│   ├─ migrate-to-supabase.mjs  # 옛 chunks.json → Supabase 이전 (1회성, 완료됨)
│   ├─ test-rag.js          #   로컬에서 RAG 한 방 테스트: node scripts/test-rag.js "질문"
│   └─ lib/                 #   추출 헬퍼(hwp/pptx/vtt)
│
├─ README.md  ARCHITECTURE.md  DEPLOY.md  OPERATIONS.md  IMPROVEMENTS.md
└─ .github/workflows/keep-warm.yml   # (선택) 콜드스타트 완화용 핑 — 보통 UptimeRobot 사용
```

**봇에서 가장 자주 보는 파일: `src/rag.js`(검색·답변 로직)와 `src/app.js`(엔드포인트).**

### 4.2 어드민 (`admin/`, Next.js 16 App Router)

```
admin/
├─ app/
│   ├─ (dashboard)/         # 로그인 후 운영자 화면 (proxy.ts가 보호)
│   │   ├─ layout.tsx       #   사이드바 + 메인(내부 스크롤) 레이아웃
│   │   ├─ files/page.tsx   #   ★ 자료 관리: 통계 카드 + 업로드존 + 파일 목록
│   │   ├─ test/page.tsx    #   답변 테스트: 질문 → 검색 청크·단계 타이밍·답변
│   │   ├─ stats/page.tsx   #   통계: 질의 KPI·일별 차트·카테고리 분포
│   │   ├─ feedback/page.tsx#   사용자 피드백: 자료없음·오류를 모아 해결/삭제
│   │   └─ settings/page.tsx#   비밀번호 변경
│   ├─ admin/api/           # 서버 API 라우트 (인증 가드 requireAdmin)
│   │   ├─ upload/sign/route.ts   #   ★ 서명 URL 발급 (큰 파일 직접 업로드용)
│   │   ├─ upload/route.ts        #   ★ 업로드 후 처리(추출→청킹→임베딩), NDJSON 진행 스트림
│   │   ├─ documents/[id]/route.ts          #   문서 PATCH(카테고리)/DELETE(문서+청크+원본)
│   │   ├─ documents/[id]/reingest/route.ts #   재처리(Storage에서 다시 추출)
│   │   ├─ queries/[id]/route.ts  #   질의 해결/삭제
│   │   ├─ queries/bulk/route.ts  #   질의 일괄 처리
│   │   └─ test/route.ts          #   답변 테스트 실행(ragQuery 호출)
│   ├─ auth/callback/route.ts# Supabase Auth 콜백
│   ├─ login/page.tsx        # 로그인
│   └─ layout.tsx, page.tsx, globals.css
│
├─ components/
│   ├─ files/
│   │   ├─ file-list.tsx     #   ★ 파일 목록 + 검색·정렬·형식필터 + 카테고리 편집 + 낙관적 UI
│   │   └─ upload-zone.tsx   #   ★ 드래그 업로드 → sha256 → 서명URL → 직접 업로드 → 처리요청
│   ├─ feedback/feedback-list.tsx  # 피드백 탭(해결/미해결)·일괄처리
│   ├─ layout/sidebar.tsx    #   사이드바 내비
│   └─ ui/                   #   shadcn/ui 컴포넌트 (button, dropdown-menu, table 등)
│
├─ lib/
│   ├─ ingest/              # ★ 업로드 처리 파이프라인
│   │   ├─ process.ts       #   오케스트레이션: 추출→청킹→임베딩→chunks upsert→상태갱신
│   │   ├─ extract.ts       #   포맷별 텍스트 추출 (pdf=unpdf, pptx=officeparser, hwp, vtt, 오디오)
│   │   ├─ chunk.ts         #   청킹 (800자, overlap 100, 문단 경계 우선)
│   │   ├─ embed.ts         #   배치 임베딩 (Gemini)
│   │   ├─ category.ts      #   파일명으로 카테고리 자동 추론
│   │   ├─ gemini-files.ts  #   오디오 전사 (Gemini Files API)
│   │   └─ *.test.ts        #   Vitest 단위 테스트
│   ├─ rag/                 # ★ 어드민용 RAG (봇 src/rag.js와 동일 로직 재현)
│   │   ├─ query.ts         #   ragQuery: 재작성→임베딩→검색→생성 + 단계 타이밍
│   │   ├─ enhance.ts       #   쿼리 재작성(캐시·영어키워드) + (구)재정렬 함수
│   │   └─ log.ts           #   질의 로깅
│   ├─ supabase/
│   │   ├─ server.ts        #   서버 클라이언트: createServiceClient(RLS bypass)·세션 클라이언트
│   │   └─ client.ts        #   브라우저 클라이언트(anon)
│   ├─ storage.ts           #   safeStoragePath: 한글 파일명 Storage 키 버그 회피(UUID+확장자)
│   ├─ upload-meta.ts       #   지원 형식·MIME·용량·버킷 단일 출처
│   ├─ types.ts             #   DocumentRow 타입 + 카테고리 라벨·색·포맷 헬퍼
│   ├─ auth-guard.ts        #   requireAdmin (인증 + allowed_admins 화이트리스트)
│   └─ utils.ts             #   cn 등
│
├─ proxy.ts                 # ★ Next 16 "미들웨어": 세션 갱신 + 대시보드/admin API 경로 보호
├─ next.config.ts           # serverExternalPackages(unpdf/officeparser/hwp.js), turbopack root
├─ vercel.json              # 어드민 Vercel 설정
└─ AGENTS.md / CLAUDE.md    # Next 16 주의 안내
```

**어드민에서 가장 자주 보는 파일: `components/files/*`(자료 관리 UI), `lib/ingest/*`(업로드 처리), `lib/rag/query.ts`(답변 테스트).**

### 4.3 레거시·주의 (건드리기 전 알아둘 것)

| 위치 | 상태 |
|---|---|
| `public/admin/*.html` | **옛 정적 어드민**(Next.js 어드민 이전 버전). 현재 봇은 정적 어드민을 서빙하지 않음 → **사실상 미사용**. 정리 대상 |
| `data/extracted/`, `source-files/` | 옛 CLI ingest 입출력. **현재 운영은 어드민 업로드 사용**. CLI는 백업용 |
| 루트 `package.json`의 `pdf-parse`·`officeparser`·`description("Claude")` | CLI 스크립트가 쓰는 의존성 + **stale한 설명**(실제 LLM은 Gemini). 봇 본체 `src/`는 이들을 안 씀 |
| `update.bat` | 옛 Windows 배치. 무시 가능 |

---

## 5. 처음부터 셋업 (Zero → 실행)

### 0) 준비물
- **Node.js 20+** (봇은 18+, 어드민은 Next 16이라 20+ 권장)
- **계정**: GitHub, [Supabase](https://supabase.com), [Google AI Studio](https://aistudio.google.com)(Gemini 키), Vercel(배포 시)
- (선택) [Supabase CLI](https://supabase.com/docs/guides/cli) — 마이그레이션 적용용. 없으면 대시보드 SQL Editor로 수동 적용 가능

### 1) 코드·의존성
```bash
git clone https://github.com/luazencloud-design/kakao-bot.git
cd kakao-bot
npm install            # 봇 의존성
cd admin && npm install && cd ..   # 어드민 의존성
```

### 2) Supabase 셋업
1. Supabase에서 **새 프로젝트 생성** (리전: 서울 `ap-northeast-2` 권장).
2. **마이그레이션 적용** — `supabase/migrations/0001~0005.sql`을 **번호 순서대로** 실행.
   - 방법 A(권장): Supabase CLI `supabase db push`
   - 방법 B: 대시보드 → SQL Editor에 각 파일 내용을 붙여넣고 순서대로 Run
   - 이걸로 테이블(documents·chunks·queries·sessions·allowed_admins·query_rewrites), 인덱스, `hybrid_search` 함수, GRANT/RLS가 만들어진다.
3. **Storage 버킷 생성**: 이름 `source-files`, Public **off**, 파일 크기 제한 **50MB** 이상.
4. **운영자 등록**: SQL로 어드민 이메일 화이트리스트 추가
   ```sql
   insert into allowed_admins (email, note) values ('운영자@example.com', '초기 운영자');
   ```
   그리고 Supabase **Authentication**에서 그 이메일로 사용자 생성(비밀번호 방식).
5. **API 키 확보**: 프로젝트 Settings → API에서 `Project URL`, `anon` 키, `service_role` 키를 복사. (자세한 건 [supabase/README.md](supabase/README.md))

### 3) Gemini 키
[Google AI Studio](https://aistudio.google.com/apikey)에서 API 키 발급(`AIza...`). 봇·어드민·로컬 3곳에서 같은 키를 쓴다.

### 4) 환경변수

> ⚠️ **봇과 어드민이 읽는 변수 이름이 다르다** (어드민은 브라우저 노출용 `NEXT_PUBLIC_` 접두사 사용).

**봇** (`.env`, `.env.example` 복사):
```
GEMINI_API_KEY=AIza...                 # AI Studio 키
GEMINI_MODEL=gemini-flash-lite-latest  # 생성·재작성
EMBED_MODEL=gemini-embedding-001       # 임베딩(768)
TOP_K=6                                # 검색 청크 수 (완결성↑, 일관성과 무관)
SUPABASE_URL=https://xxxx.supabase.co  # 접두사 없음, /rest/v1 같은 경로 빼고 도메인만
SUPABASE_SERVICE_ROLE_KEY=eyJ...        # legacy JWT(eyJ로 시작) service_role 키 사용
WEBHOOK_SECRET=                        # (선택) 웹훅 시크릿 경로. 아래 보안 참고
REWRITE=                               # (선택) 'off'면 재작성 끔. 보통 비워둠(캐시로 일관성 보장)
PORT=3000                              # 로컬 포트
```

**어드민** (`admin/.env.local`):
```
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...   # anon 키 (브라우저 노출 OK)
SUPABASE_SERVICE_ROLE_KEY=eyJ...        # 서버 전용 (절대 클라이언트 노출 금지)
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-flash-lite-latest
EMBED_MODEL=gemini-embedding-001
TOP_K=6
```

### 5) 로컬 실행·테스트
```bash
# 봇
npm run dev                        # http://localhost:3000 (node --watch)
node scripts/test-rag.js "사업자등록 방법"   # 서버 없이 RAG 한 방 테스트

# 어드민
cd admin
npm run dev                        # http://localhost:3000 (포트 겹치면 봇 PORT 변경)
npm test                           # Vitest 단위 테스트
```

---

## 6. 명령어 모음

| 위치 | 명령 | 설명 |
|---|---|---|
| 루트 | `npm run dev` | 봇 로컬 서버 (watch) |
| 루트 | `npm start` | 봇 로컬 서버 |
| 루트 | `node scripts/test-rag.js "질문"` | RAG 동작 테스트(서버 불필요) |
| 루트 | `npm run ocr` / `npm run ingest` | CLI 추출 / 인제스트 (백업용) |
| `admin/` | `npm run dev` | 어드민 개발 서버 |
| `admin/` | `npm run build` | 프로덕션 빌드 (배포 전 검증) |
| `admin/` | `npm test` | Vitest (순수 함수 + 버그 회귀) |
| `admin/` | `npx tsc --noEmit` | 타입체크 |

---

## 7. 동작 원리 (핵심 로직)

### 7.1 봇: 질문 → 답변 (`src/rag.js`)

1. 카카오가 `POST /kakao/skill`(또는 `/kakao/skill/callback`)로 질문 JSON 전송
2. **쿼리 재작성** — 구어체 질문을 검색 키워드로 보강(Gemini). 결과를 `query_rewrites` 테이블에 **캐시**(같은 질문 → 같은 재작성 → 결정적). 캐시 키엔 프롬프트 버전 접두사. **영어로도 표기되는 용어는 영어 키워드도 추가**(영어 자료를 한국어 질문으로 찾기 위함)
3. **임베딩** — 원질문 기준(`gemini-embedding-001`, 768차원)
4. **하이브리드 검색** — Supabase `hybrid_search` RPC: pgvector dense + pg_trgm 한국어 sparse를 **RRF**로 융합, 동점은 `id`로 결정적 정렬, 상위 `TOP_K`개
5. **답변 생성** — Gemini(`temperature 0`), [참고 자료]에만 근거 + 출처 명시. 자료에 없으면 "포함되어 있지 않습니다"
6. **로깅** — 모든 질의를 `queries` 테이블에 기록(실패는 `[오류]`로)
7. **콜백 (빠른 응답 우선)** — 답변이 `SYNC_BUDGET`(3.5초) 안에 끝나면 **동기로 바로** 응답(대기 풍선 없음). 넘기면 그때 "생성 중" 정적 응답을 보내고 실답변을 1회용 `callbackUrl`로 **1분 안에** 전달(`waitUntil` + 45초 데드라인 가드). 콜백 미설정 환경에선 동기로만 동작

> **일관성(결정성) 설계:** "같은 질문에 다른 답"이 나오던 문제를 ① **재작성 캐시** ② **RRF 동점 id 정렬** ③ **생성 temperature 0** 세 가지로 제거했다. (단, 자료끼리 모순되는 극단적 입력은 LLM 특성상 완전히는 못 잡음.)

### 7.2 어드민: 자료 업로드 → 검색 가능

1. **직접 업로드** — 브라우저가 sha256 계산 후 `/upload/sign`으로 **서명 URL**을 받아 **Supabase Storage에 직접** 업로드. Vercel 함수를 거치지 않아 **요청 본문 4.5MB 한계를 우회**(버킷 한도 50MB까지)
2. **처리** — `/upload`이 문서 행 생성 후 `processDocument` 실행: Storage에서 파일 내려받아 **추출 → 청킹(800자) → 배치 임베딩 → chunks upsert → status=ready**. 진행 상황은 NDJSON으로 스트리밍
3. 변경은 봇에 **즉시 반영**(재배포 불필요)

### 7.3 지원 파일 형식

| 형식 | 처리 | 비고 |
|---|---|---|
| PDF | `unpdf` | 텍스트 기반만. 스캔 이미지 PDF는 추출 실패 (pdf-parse는 Vercel에서 `DOMMatrix` 에러라 교체) |
| PPTX | `officeparser` | 텍스트 기반만. 이미지 위주 슬라이드는 약함 |
| HWP | `hwp.js` | HWP 5.x. 구버전·이미지 기반 불가 |
| TXT | 직접 읽기 | FAQ(Q/A 빈 줄 구분) 권장 |
| VTT | 자막 파서 | **긴 강의는 이걸 권장** (Zoom·YouTube 자동 자막) |
| MP3 | Gemini Files API 전사 | 오디오. 길면 전사 시간↑ |
| MP4(영상) | **미지원** | 전사 타임아웃 위험 → 자막(VTT)으로 |

> 추출 텍스트가 너무 짧으면(거의 빈 파일) "이미지 위주" 류 메시지로 실패 처리한다. 빈 청크는 따로도 막힌다(`process.ts`).

---

## 8. 데이터 모델 (Supabase)

| 테이블 | 용도 |
|---|---|
| `documents` | 업로드 파일 메타(파일명·status·category·chunk_count·sha256·storage_path) |
| `chunks` | 임베딩된 텍스트 청크(`embedding vector(768)`, HNSW 인덱스, `text` pg_trgm 인덱스). 문서 삭제 시 `ON DELETE CASCADE` |
| `queries` | 질의 로그(utterance·rewritten·retrieved_chunk_ids·answer·latency·llm_provider·resolved_at) — 관측·피드백 |
| `query_rewrites` | 재작성 캐시(`q_norm` PK → `rewrite`). 검색 결정성 + 봇/어드민 일치 |
| `allowed_admins` | 어드민 이메일 화이트리스트 |
| `sessions` | 멀티턴 대화용(미사용, 향후) |

- **RPC `hybrid_search(query_embedding, query_text, category_filter, match_count)`** — dense+sparse RRF, `id` tiebreak로 결정적. 봇·어드민이 공통으로 호출.
- **Storage 버킷 `source-files`** — 원본 파일. 키는 `uploads/<uuid>.<ext>`(한글 키 버그 회피).
- 마이그레이션 `0001`→`0005` 순서대로 적용. 자세한 건 각 SQL 파일 상단 주석 참고.

---

## 9. 배포

봇·어드민 둘 다 Vercel. 같은 Supabase를 바라본다.
- **봇**: Root = 레포 루트(`/`), `vercel.json`이 모든 요청을 `api/index.js`로 라우팅, `maxDuration 60`
- **어드민**: Root = `admin`, Framework Next.js
- 환경변수는 [5-4](#4-환경변수) 참고. **카카오 5초 초과 대응 콜백 활성화 절차는 [DEPLOY.md](DEPLOY.md) C절** 참고.

전체 절차·체크리스트·업로드 아키텍처는 **[DEPLOY.md](DEPLOY.md)**.

---

## 10. 운영

비개발자 운영자용 가이드는 **[OPERATIONS.md](OPERATIONS.md)** (자료 올리기·확인·문제 대처).

- **자료 관리**: 드래그 업로드·삭제·재처리·카테고리 수정 + **검색·정렬·형식 필터**
- **답변 테스트**: 질문 → 검색된 청크·단계 타이밍 확인 (실제 봇과 같은 답)
- **통계 / 피드백**: "답변 못한 질문"을 보고 자료를 보강하는 운영 루프

---

## 11. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 봇이 "일시적으로 답변 못함" 반복 | Gemini 키 무효/할당량 | AI Studio 유효 키로 교체 → 봇·어드민·로컬 3곳 `GEMINI_API_KEY` → Vercel Redeploy |
| 카톡 답이 5초 넘겨 안 옴 | RAG가 5초 초과 | **콜백 모드** 활성화([DEPLOY.md](DEPLOY.md) C절). 키워드 재작성·재정렬 제거로 평소 latency는 이미 단축됨 |
| 같은 질문에 다른 답 | (해결됨) 재작성·RRF·생성 비결정 | 재작성 캐시 + id 정렬 + temp 0로 제거됨 |
| 영어 자료가 한글 질문에 안 잡힘 | (해결됨) 교차언어 | 재작성이 영어 키워드 추가 |
| 어드민 500 | 환경변수 누락/오타 | 어드민 env 7개 확인(`NEXT_PUBLIC_*` 포함) |
| 큰 파일 업로드 413 | Vercel 4.5MB 한계 | 서명 URL 직접 업로드로 우회(구현됨). 그래도 안 되면 버킷 크기 제한 확인 |
| 업로드 "실패(워커/추출)" | 스캔 PDF·이미지 PPT | 텍스트 버전/자막(VTT)으로. 점3개 → 재처리 |
| 봇 장애 원인 파악 | — | `select * from queries where llm_provider='error' order by created_at desc` 또는 어드민 피드백 "오류" |

---

## 12. 알려진 제약

- **Gemini 단일 벤더** — 503/키 문제 시 봇 전체 영향(생성 fallback 미구현)
- **영상 raw 처리 불가** — 자막(VTT) 우회 필요
- **Rate limit이 인메모리** — 서버리스 인스턴스별이라 콜드스타트 시 초기화. 엄격하려면 Upstash Redis
- **재작성 캐시는 "처음 만든 재작성"을 고정** — 나쁜 재작성이 박히면 `query_rewrites`에서 해당 행 삭제로 재계산
- **모순되는 자료(예: 농담 청크)** — LLM이 경계에서 흔들릴 수 있음(temp 0로도 100%는 아님)
- **레거시 잔재** — `public/admin/*.html`, `data/extracted/`, 루트 stale 의존성([4.3](#43-레거시주의-건드리기-전-알아둘-것))

개선 로드맵은 [IMPROVEMENTS.md](IMPROVEMENTS.md).

---

## 13. 문서 지도

| 나는... | 읽을 문서 |
|---|---|
| **처음 인수받음** | 이 **README** (전체) |
| **비개발자 운영자** (자료만 관리) | **[OPERATIONS.md](OPERATIONS.md)** |
| **배포·환경변수** | **[DEPLOY.md](DEPLOY.md)** |
| **설계 의도·트레이드오프** | **[ARCHITECTURE.md](ARCHITECTURE.md)** |
| **Supabase 셋업 상세** | **[supabase/README.md](supabase/README.md)** |
| **앞으로 개선할 것** | **[IMPROVEMENTS.md](IMPROVEMENTS.md)** |
