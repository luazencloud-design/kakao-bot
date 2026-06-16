# 🤖 카카오 RAG 챗봇 + 운영자 어드민

> 카카오톡 채널에서 동작하는 RAG 챗봇. 사용자가 카톡으로 질문하면 등록된 자료(PDF·PPTX·HWP·TXT·VTT·오디오·영상)에서 관련 내용을 검색해 **Gemini**가 출처와 함께 답변한다.
> 비개발자 운영자는 **웹 어드민**에서 자료를 직접 업로드·관리한다.

- **레포:** `luazencloud-design/kakao-bot` (Vercel 봇 프로젝트명은 `naver-bot-one`)
- **도메인 특화:** 사업자등록·오픈마켓 가입·강의 안내 (시스템 프롬프트가 도메인 특화 — 다른 주제면 `src/rag.js` 프롬프트 수정)

---

## 📑 어떤 문서를 봐야 하나요?

| 나는... | 읽을 문서 |
|---|---|
| **운영자** (자료 올리고 챗봇 관리, 코딩 모름) | 👉 **[OPERATIONS.md](OPERATIONS.md)** — 이것만 보면 됨 |
| **개발자** (처음 이 프로젝트를 봄) | 이 README → [ARCHITECTURE.md](ARCHITECTURE.md) 순서 |
| **배포·환경변수** 작업 | [DEPLOY.md](DEPLOY.md) |
| 앞으로 **뭘 개선할지** 보고 싶음 | [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| **Supabase(DB) 셋업**을 처음부터 | [supabase/README.md](supabase/README.md) |

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
2. **쿼리 재작성** (구어체 → 검색 키워드, Gemini). **`query_rewrites` 캐시** — 같은 질문은 저장된 재작성을 재사용해 검색이 결정적. 봇·어드민이 캐시를 공유해 답이 일치
3. **임베딩** (원질문 기준, Gemini `gemini-embedding-001`, 768차원)
4. **하이브리드 검색** (Supabase `hybrid_search` RPC: pgvector dense + pg_trgm 한국어 sparse → RRF, 동점은 `id`로 결정적 정렬, top-K)
5. **답변 생성** (Gemini, `temperature 0` — 자료에만 근거 + 출처 명시. 같은 질문엔 같은 답)
6. 5초 초과 대비 **콜백 모드**: 5초 안엔 "생성 중" 정적 응답만 보내고 실답변은 1회용 `callbackUrl`로 1분 안에 전달(`waitUntil` + 45초 데드라인 가드). 활성화는 [DEPLOY.md](DEPLOY.md) C절
7. 모든 질의를 `queries` 테이블에 로깅 (실패 시 `[오류]`로 기록)

> **일관성 설계:** 같은 질문에 다른 답이 나오던 문제(재작성·RRF 동점·생성 온도의 3중 비결정)를 ① 재작성 캐시 ② `id` 동점 정렬 ③ `temperature 0`으로 제거. 13개 질문 5회 반복 모두 동일 답 검증.

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
| PDF | unpdf | 텍스트 기반. 스캔 이미지 PDF는 추출 실패 (pdf-parse는 Vercel Node에서 DOMMatrix 에러라 교체) |
| PPTX | officeparser | 텍스트 기반. 이미지 위주 슬라이드는 약함 |
| HWP | hwp.js | HWP 5.x. 구버전·이미지 기반 불가 |
| TXT | 직접 읽기 | FAQ 형식(Q/A 빈 줄 구분) 권장 |
| VTT | 자막 파서 | **긴 강의는 이걸 권장** (Zoom·YouTube 자동 자막) |
| MP3 | Gemini Files API 전사 | 오디오. ⚠️ 긴 파일은 전사 시간 ↑ |
| ~~MP4(영상)~~ | **미지원** | 전사가 느려 함수 타임아웃 위험 → 자막(VTT)으로 |

### ⬆️ 업로드 방식 (큰 파일)
업로드는 브라우저가 **서명 URL로 Supabase Storage에 직접** 올린다(`upload-zone.tsx` → `/admin/api/upload/sign`). Vercel 함수를 거치지 않아 **요청 본문 4.5MB 한계를 우회**, 버킷 한도인 **50MB까지** 받는다. 업로드가 끝나면 `/admin/api/upload`가 Storage에서 내려받아 추출·임베딩한다.

### ⚠️ 영상은 미지원
- 영상(MP4)은 받지 않는다 — 전사가 느려 함수 타임아웃 위험이 큼
- **권장: 영상 대신 자막(VTT) 업로드.** Zoom·YouTube가 자동 생성, 1시간도 수백 KB라 즉시 처리됨

---

## 🔑 환경변수

⚠️ **봇과 어드민이 읽는 변수 이름이 다름:**

**봇 (`naver-bot-one`, 루트 `.env`):**
```
GEMINI_API_KEY              # AI Studio 키 (유효한 것!)
GEMINI_MODEL=gemini-flash-lite-latest
EMBED_MODEL=gemini-embedding-001
TOP_K=6                    # 검색 청크 수 (완결성↑). 일관성과는 무관
SUPABASE_URL               # 접두사 없음
SUPABASE_SERVICE_ROLE_KEY
WEBHOOK_SECRET             # (선택) 웹훅 시크릿 경로. 아래 "웹훅 보안" 참고
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

## 🔒 웹훅 보안

봇 웹훅에 시크릿 경로 + rate limit + 입력 가드가 적용돼 있다 (`src/security.js`).

- **시크릿 경로**: `WEBHOOK_SECRET` 환경변수 설정 시, 카카오 스킬 URL이 `https://<도메인>/kakao/skill/<시크릿>` 형태여야 통과. 모르면 403. (미설정 시 기존 `/kakao/skill`도 호환 동작)
- **Rate limit**: 사용자(user.id)당 분당 20회 (인메모리 — 콜드스타트 시 초기화, 엄격하게는 Redis 필요)
- **입력 가드**: 500자 제한, 빈 입력 차단, 프롬프트 인젝션 의심 문구 무력화

### 시크릿 켜는 법 (무중단 순서)
1. **카카오 오픈빌더** 스킬 URL을 `/kakao/skill/<시크릿>` 으로 변경 (아직 Vercel에 시크릿 없어 호환 모드로 통과)
2. **Vercel 봇**(`naver-bot-one`)에 `WEBHOOK_SECRET` 추가 → Redeploy
3. 이제 옛 URL은 403, 새 URL만 동작 (오픈빌더는 이미 새 URL이라 안 끊김)

> 시크릿 생성: `node -e "console.log(require('crypto').randomBytes(24).toString('hex'))"`

---

## 🧪 테스트

```bash
cd admin
npm test          # Vitest 단위 테스트 24개 (순수 함수 + 버그 회귀)
```

`chunkText`(청킹), `inferCategory`(분류), `extractVtt`(자막), `safeStoragePath`(한글 키 버그 회귀), 표시 헬퍼를 커버. 외부 의존(Gemini·DB) 없는 순수 함수 위주.

---

## 📌 알려진 제약

- **Gemini 단일 벤더** — 503/키 문제 시 봇 전체 영향 (생성 fallback은 미구현)
- **영상 raw 처리 사실상 불가** — 자막 우회 필요
- **Rate limit이 인메모리** — 서버리스 인스턴스별이라 콜드스타트 시 초기화. 엄격하게는 Upstash Redis 필요
- **keep-warm** — GitHub Actions 스케줄은 best-effort(5분 안 지켜짐). 정확하려면 UptimeRobot 등 외부 핑

개선 로드맵은 [IMPROVEMENTS.md](IMPROVEMENTS.md), 설계 상세는 [ARCHITECTURE.md](ARCHITECTURE.md).
