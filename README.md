# 🤖 카카오 RAG 챗봇 + 운영자 어드민

> 카카오톡 채널에서 동작하는 RAG 챗봇. 사용자가 카톡으로 질문하면 등록된 자료(PDF·PPTX·HWP·TXT·VTT·오디오·영상)에서 관련 내용을 검색해 **Gemini**가 출처와 함께 답변한다.
> 비개발자 운영자는 **웹 어드민**에서 자료를 직접 업로드·관리한다.

- **레포:** `luazencloud-design/kakao-bot` (Vercel 봇 프로젝트명은 `naver-bot-one`)
- **도메인 특화:** 사업자등록·오픈마켓 가입·강의 안내 (시스템 프롬프트가 도메인 특화 — 다른 주제면 `src/rag.js` 프롬프트 수정)

---

## 🗺 전체 구조

```
[수강생] ──카카오톡──> 봇 (Express, Vercel)        ─┐
[운영자] ──브라우저──> 어드민 (Next.js, Vercel)     ─┼─> Supabase ──> Gemini
                                                     ─┘  (pgvector·     (임베딩·생성·
                                                         Storage·Auth)   OCR·전사)
```

**카카오 API를 호출하는 게 아니라, 카카오가 봇 웹훅을 호출하는 구조.** 봇은 카카오가 보낸 질문 JSON을 받아 RAG로 답을 만들어 카카오 형식 JSON으로 돌려준다. 카카오 인증키·메시지발송 API는 안 쓴다.

### 구성요소

| 부분 | 호스팅 | 역할 |
|---|---|---|
| **봇** (`src/`, `api/`) | Vercel (`naver-bot-one`) | 카카오 웹훅 → RAG 답변 |
| **어드민** (`admin/`) | Vercel (`kakao-bot-admin`) | 운영자 자료 관리 UI |
| **Supabase** | 클라우드(Tokyo) | pgvector 검색 · Storage · Auth · 질의 로그 |
| **Gemini** | Google | 임베딩 · 답변 생성 · OCR/전사 |

---

## 📂 레포 구조

```
kakao-bot/
├─ src/                  # 봇 (Express)
│   ├─ app.js            #   엔드포인트(/kakao/skill, /callback) + waitUntil
│   ├─ rag.js            #   RAG: 재작성→임베딩→하이브리드검색→재정렬→생성 + 로깅
│   └─ kakao.js          #   카카오 스킬 응답 포맷
├─ api/index.js          # Vercel 서버리스 진입점 (app.js 래핑)
├─ scripts/              # CLI 도구 (백업용)
│   ├─ ocr.js            #   PDF·영상·오디오 → 텍스트 (Gemini Files API)
│   ├─ ingest.js         #   추출→청킹→임베딩→Supabase upsert
│   └─ migrate-to-supabase.mjs
├─ supabase/migrations/  # DB 스키마 (0001~0003)
├─ admin/                # 어드민 (Next.js 16 App Router)
│   ├─ app/(dashboard)/  #   files·test·stats·feedback·settings
│   ├─ app/admin/api/    #   업로드·문서·질의 API
│   └─ lib/              #   ingest 파이프라인·RAG·Supabase 클라이언트
├─ vercel.json           # 봇 Vercel 설정
├─ DEPLOY.md             # 배포 가이드
└─ ARCHITECTURE.md       # 설계 상세
```

---

## 🔄 동작 원리

### 봇 (질문 → 답변)
1. 카카오가 `/kakao/skill`로 질문 POST
2. **쿼리 재작성** (구어체 → 검색 키워드, Gemini)
3. **임베딩** (Gemini `gemini-embedding-001`, 768차원)
4. **하이브리드 검색** (Supabase `hybrid_search` RPC: pgvector dense + pg_trgm 한국어 sparse → RRF)
5. **LLM 재정렬** (top-12 → top-4, 노이즈 제거)
6. **답변 생성** (Gemini, 자료에만 근거 + 출처 명시)
7. 5초 초과 시 콜백 모드(`waitUntil`)로 안전 처리
8. 모든 질의를 `queries` 테이블에 로깅 (실패 시 `[오류]`로 기록)

### 어드민 (자료 관리)
- **자료 관리**: 다중 업로드(진행 시각화)·삭제·재처리·카테고리 수정
- **답변 테스트**: 질문 → 검색 청크·단계별 타이밍 확인
- **통계**: 질의 KPI·일별 차트·카테고리 분포·자료 갭
- **사용자 피드백**: 부정 피드백·자료없음·오류를 모아 해결/삭제 (Sentry 스타일 탭·일괄처리)
- **설정**: 비밀번호 변경

업로드하면 어드민이 **추출 → 청킹 → 배치 임베딩 → Supabase upsert**까지 자동 처리. 변경은 봇에 **즉시 반영**(재배포 불필요).

---

## 📄 지원 파일 형식

| 형식 | 처리 | 비고 |
|---|---|---|
| PDF | pdf-parse | 텍스트 기반. 스캔 이미지 PDF는 추출 실패 |
| PPTX | officeparser | 텍스트 기반. 이미지 위주 슬라이드는 약함 |
| HWP | hwp.js | HWP 5.x. 구버전·이미지 기반 불가 |
| TXT | 직접 읽기 | FAQ 형식(Q/A 빈 줄 구분) 권장 |
| VTT | 자막 파서 | **긴 강의는 이걸 권장** (Zoom·YouTube 자동 자막) |
| MP3·MP4 | Gemini Files API 전사 | ⚠️ **50MB 한도 + Vercel 타임아웃** — 아래 참고 |

### ⚠️ 영상 처리 주의
- **Supabase Storage 50MB 한도** → 1시간 강의 영상(수백 MB)은 업로드 불가
- **Vercel 함수 타임아웃** (Hobby 10초 / Pro 60~300초) → 긴 전사는 실패
- **권장: 영상 대신 자막(VTT) 업로드.** Zoom·YouTube가 자동 생성, 1시간도 수백 KB라 즉시 처리됨

---

## 🔑 환경변수

⚠️ **봇과 어드민이 읽는 변수 이름이 다름:**

**봇 (`naver-bot-one`, 루트 `.env`):**
```
GEMINI_API_KEY              # AI Studio 키 (유효한 것!)
GEMINI_MODEL=gemini-flash-lite-latest
EMBED_MODEL=gemini-embedding-001
TOP_K=4
SUPABASE_URL               # 접두사 없음
SUPABASE_SERVICE_ROLE_KEY
```

**어드민 (`kakao-bot-admin`, `admin/.env.local`):**
```
NEXT_PUBLIC_SUPABASE_URL        # 접두사 있음 (브라우저 노출)
NEXT_PUBLIC_SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY       # 서버 전용
GEMINI_API_KEY
GEMINI_MODEL / EMBED_MODEL / TOP_K
```

---

## 🚀 셋업 (로컬 개발)

```bash
# 봇
npm install
cp .env.example .env          # 값 채우기
node scripts/test-rag.js "사업자등록 방법"   # RAG 동작 테스트

# 어드민
cd admin
npm install
# .env.local 작성 (위 환경변수)
npm run dev                   # localhost:3000
```

배포는 [DEPLOY.md](DEPLOY.md) 참고 (봇·어드민 둘 다 Vercel).

---

## 🧰 운영 — 자료 추가/갱신

**평소(권장): 어드민 웹에서.**
1. `kakao-bot-admin.vercel.app` 로그인
2. 자료 관리 → 파일 드래그 → 업로드 (진행 시각화)
3. 끝. 봇에 즉시 반영

**CLI(백업, 개발자용):** 자막 없는 긴 영상 raw 전사 등 예외 상황
```bash
# source-files/에 파일 넣고
npm run ocr        # 추출 → data/extracted/
npm run ingest     # 청킹·임베딩 → Supabase
```

---

## 🔍 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 봇이 "일시적으로 답변 못함" 반복 | **Gemini API 키 무효** | AI Studio에서 유효 키 발급 → 봇·어드민·로컬 3곳 `GEMINI_API_KEY` 교체 → Vercel Redeploy |
| 어드민 500 | 환경변수 누락 | 어드민 프로젝트 env 7개 확인 (`NEXT_PUBLIC_*` 포함) |
| 오랜만에 쓰면 봇 느림 | 콜드스타트 | keep-warm GitHub Action(`.github/workflows/keep-warm.yml`) — `BOT_PING_URL` 변수 설정 |
| 봇 장애 원인 파악 | — | `select * from queries where llm_provider='error' order by created_at desc` 또는 어드민 피드백 페이지 "오류" 항목 |
| PDF 업로드 실패(워커) | pdfjs 워커 | `serverExternalPackages` 설정됨. 안 되면 CLI로 우회 |
| 영상 업로드 안 됨 | 50MB·타임아웃 | 자막(VTT)으로 우회 |

---

## 📌 알려진 제약

- **Gemini 단일 벤더** — 503/키 문제 시 봇 전체 영향 (생성 fallback은 미구현)
- **영상 raw 처리 사실상 불가** — 자막 우회 필요
- **봇 웹훅 공개** — 시크릿 경로·rate limit 미적용 (비용은 AI Studio 한도로 방어 중)
- **keep-warm** — GitHub Actions 스케줄은 best-effort(5분 안 지켜짐). 정확하려면 UptimeRobot 등 외부 핑

개선 로드맵은 [IMPROVEMENTS.md](IMPROVEMENTS.md), 설계 상세는 [ARCHITECTURE.md](ARCHITECTURE.md).
