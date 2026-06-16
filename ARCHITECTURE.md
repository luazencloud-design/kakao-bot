# kakao-class-bot 아키텍처 설계서

> 실운영 전제. 비개발자 운영자가 셀프서비스로 지식베이스를 관리하는 시스템.
> 작성: 2026-05-27. 변경 시 ADR 형식으로 하단에 기록.

---

## ⚠️ 현재 실제 구조 (이 설계서와 다른 점)

이 문서는 초기 설계로, **Fly.io 배포**를 전제했다. 실제 구현은 진행하면서 바뀌었다:

| 항목 | 초기 설계 | **실제 구현** |
|---|---|---|
| 봇 호스팅 | Fly.io | **Vercel** (`naver-bot-one`) — Supabase로 stateless화 + `waitUntil`로 콜백 해결 |
| 어드민 호스팅 | Fly.io | **Vercel** (`kakao-bot-admin`) — 영상은 자막 우회로 긴 작업 회피, 배치 임베딩으로 타임아웃 회피 |
| 한국어 검색 | tsvector(BM25) | **pg_trgm word_similarity** (tsvector가 한국어 토큰화 못 함) |
| Reranker | Cohere | **Gemini LLM 재정렬** (외부 서비스 추가 없이) |
| 인증 | 매직링크 | **비밀번호** (이메일 rate limit·일회용 토큰 문제로 전환) |
| 세션·관측성·보안 | 설계됨 | **시크릿경로·rate limit·입력가드 구현**(`src/security.js`), 질의/에러 로깅 구현. Langfuse 등 관측 플랫폼은 미적용 |
| 카카오 콜백 윈도 | "30초" 가정 | **1분(60초)** — 공식 `callbackUrl valid time: 1min`, 1회용 (`maxDuration:60`으로 정렬) |

**왜 Vercel로:** Supabase(pgvector)로 데이터를 외부화하니 봇이 stateless가 되어 Vercel 제약(번들 50MB·콜드스타트 JSON 재파싱)이 사라짐. 콜백 suspend는 `@vercel/functions`의 `waitUntil`로 해결. 봇 동기 응답이 warm 기준 ~5초(p95 20초·최대 27초로 튀어 동기 5초 SLA 초과가 잦음)라, 무응답을 없애려면 **콜백 모드(1분 윈도)를 메인 경로로** 쓰는 것이 구조적 해법.

배포 절차는 [DEPLOY.md](DEPLOY.md), 현황 요약은 [README.md](README.md) 참고. 아래는 초기 설계 원문(상당수는 여전히 유효).

---

## 1. 요구사항

### 1.1 기능 요구사항
- 카카오 i 오픈빌더 챗봇으로 사용자 질문에 답변
- 강의 자료(PDF·PPTX·HWP), 오디오·영상 전사(MP3·MP4·VTT), 텍스트 등 멀티포맷 입력
- 비개발자 운영자가 웹 UI로 파일 업로드·삭제·재처리
- 갱신된 자료는 **즉시** 봇에 반영 (재배포 불필요)
- 답변에 출처 명시 (파일명·페이지·타임스탬프)

### 1.2 비기능 요구사항

| 항목 | 목표 |
|---|---|
| 가용성 | 99.5% (월 3.6h 허용) |
| 응답시간 p95 | 동기 ≤5초(카카오 SLA, 목표 3초) / 콜백 ≤1분(callbackUrl 유효시간) |
| 동시 사용자 | DAU 500 기준 설계, 5,000까지 수직 확장 |
| 데이터 갱신 latency | 텍스트 30초, 영상 3분 이내 |
| 운영자 학습곡선 | "구글 드라이브 수준"의 인터페이스 |
| 비용 | DAU 500 기준 월 $150 이하 |

### 1.3 제약조건
- 카카오 동기 응답 5초 제한 (callback 모드로 우회)
- 한국 사용자 대상 → latency 최소화 필요 (Tokyo·Seoul 리전)
- 운영자 1인 (오콜·24시간 대응 불가) → 자동화·자가복구 중심
- 강의 영상에 수강생 음성 포함 가능 → 개인정보 처리 주의

---

## 2. 시스템 아키텍처

### 2.1 전체 구성도

```
┌──────────────────────────────────────────────────────────────────────┐
│                          외부 사용자                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   [수강생]                                    [운영자]                │
│      │                                            │                   │
│      │ 카카오톡                                   │ 웹 브라우저       │
│      ▼                                            ▼                   │
│  ┌─────────┐                              ┌────────────┐             │
│  │ 카카오 i │                              │  Admin UI  │             │
│  │ OpenBuilder│                            │  (정적)    │             │
│  └────┬────┘                              └──────┬─────┘             │
│       │ HTTPS Webhook                            │ HTTPS              │
└───────┼──────────────────────────────────────────┼──────────────────┘
        │                                          │
        ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│             Fly.io Tokyo (primary) + Seoul (failover)                │
│             Express 앱 (단일 코드베이스, 다중 라우트)                │
│                                                                       │
│   ┌────────────────────┐   ┌────────────────────┐                    │
│   │  /kakao/skill      │   │  /admin/*          │                    │
│   │  /kakao/callback   │   │  (인증 필요)        │                    │
│   │  → 봇 응답 로직     │   │  → 파일·잡·테스트   │                    │
│   └─────────┬──────────┘   └──────────┬─────────┘                    │
│             │                          │                              │
│             ▼                          ▼                              │
│   ┌──────────────────┐       ┌──────────────────┐                    │
│   │  RAG Pipeline    │       │  Ingest Worker   │                    │
│   │  - 쿼리 임베딩    │       │  (in-process     │                    │
│   │  - 검색·rerank    │       │   job queue)     │                    │
│   │  - 생성·fallback │       │  - 추출·OCR      │                    │
│   └────────┬─────────┘       │  - 청킹·임베딩   │                    │
│            │                  │  - upsert        │                    │
│            │                  └────────┬─────────┘                    │
└────────────┼───────────────────────────┼──────────────────────────────┘
             │                            │
             ▼                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    외부 서비스 (Tokyo 리전 우선)                       │
│                                                                       │
│  ┌──────────────────────┐   ┌──────────────────────┐                │
│  │ Supabase             │   │ LLM Providers        │                │
│  │  - Postgres+pgvector │   │  - Gemini (1순위)    │                │
│  │  - Storage (원본)    │   │  - Claude Haiku (2)  │                │
│  │  - Auth (운영자)     │   │  - OpenAI mini (3)   │                │
│  └──────────────────────┘   └──────────────────────┘                │
│                                                                       │
│  ┌──────────────────────┐   ┌──────────────────────┐                │
│  │ Cohere               │   │ Langfuse             │                │
│  │  - Rerank API        │   │  - 트레이스·평가     │                │
│  └──────────────────────┘   └──────────────────────┘                │
│                                                                       │
│  ┌──────────────────────┐   ┌──────────────────────┐                │
│  │ Upstash Redis        │   │ Sentry               │                │
│  │  - 세션·rate limit   │   │  - 에러 추적         │                │
│  │  - 임베딩 캐시       │   │                      │                │
│  └──────────────────────┘   └──────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 계층 구조

```
[Edge / Webhook]
   ├─ Kakao webhook receiver (5초 제한 대응)
   └─ Admin HTTP API

[Application Layer]
   ├─ RAG Pipeline (query → answer)
   │   ├─ Query rewriting (Gemini Flash Lite)
   │   ├─ Hybrid retrieval (dense + BM25 via Postgres)
   │   ├─ Rerank (Cohere)
   │   └─ Generation with fallback chain
   │
   └─ Ingest Pipeline (file → chunks)
       ├─ Extract (per-format adapter)
       ├─ Chunk (semantic + size-bounded)
       ├─ Embed (Gemini, batched)
       └─ Upsert (Supabase, transactional)

[Data Layer]
   ├─ Vector store (Supabase pgvector)
   ├─ Object store (Supabase Storage)
   ├─ Cache / queue (Upstash Redis)
   └─ Auth (Supabase Auth)

[Observability]
   ├─ Traces (Langfuse)
   ├─ Errors (Sentry)
   └─ Logs (Fly.io + pino structured)
```

---

## 3. 컴포넌트별 책임

### 3.1 Fly.io 앱 (단일 코드베이스)

| 라우트 | 책임 | 인증 |
|---|---|---|
| `GET /` | 헬스체크 | 공개 |
| `POST /kakao/skill/:secret` | 카카오 동기 응답 (5초 내) | URL secret |
| `POST /kakao/skill/callback/:secret` | 카카오 콜백 응답 (1분 윈도) | URL secret |
| `GET /admin/*` | 어드민 정적 자산 | 공개 (HTML만) |
| `POST /admin/api/*` | 어드민 API | Supabase JWT |

**왜 단일 코드베이스인가:**
- RAG·ingest 로직 공유 (운영자 어드민에서 봇과 동일한 검색을 미리보기)
- 운영 단순화 (배포 1회)
- 트래픽 규모상 분리 불필요

### 3.2 RAG Pipeline

```
사용자 질문
  ↓
[1] Query rewriting (Gemini Flash Lite, 200ms)
  - "재발급?" → "사업자등록증 재교부 재발급 신청"
  ↓
[2] Embedding (Gemini embedding-001, 200ms)
  - 768차원 벡터
  - Redis LRU 캐시 (질문 텍스트 해시 키)
  ↓
[3] Hybrid retrieval (Postgres, 50ms)
  - Dense: pgvector HNSW top-20
  - Sparse: tsvector + GIN top-20
  - RRF로 결합 → top-20
  ↓
[4] Rerank (Cohere multilingual-v3, 300ms)
  - top-20 → top-4
  ↓
[5] Generation (Gemini Flash Lite, 1-3s)
  - 시스템 프롬프트 + 검색 청크 주입
  - 실패 시 Claude Haiku → GPT-4o-mini fallback
  ↓
[6] Citation 검증 (200ms)
  - 답변 각 문장이 retrieved context에 있는지 검사
  - 미근거 문장 제거 또는 "자료 없음" 처리
  ↓
답변 + 출처
```

총 예상 시간: **2~4초** (rerank·생성이 대부분). 카카오 동기 5초 한도에 마진 있음.

### 3.3 Ingest Pipeline (워커)

```
운영자가 업로드
  ↓
[1] Supabase Storage에 원본 저장
  - path: source-files/{uuid}/{filename}
  - metadata: uploaded_by, uploaded_at, size, sha256
  ↓
[2] documents 테이블에 row 생성
  - status: 'pending'
  ↓
[3] 인메모리 큐에 job 추가
  ↓
[4] 워커가 job 처리
  - Extract: 포맷별 어댑터
    * .pdf → pdf-parse 또는 Gemini Files API
    * .pptx → PowerPoint COM → PDF → Gemini OCR (Windows 한정)
    * .hwp → hwp.js
    * .vtt → 자체 파서
    * .mp3/mp4 → Gemini Files API 전사
  - 추출 결과를 documents.extracted_text에 저장 (캐시)
  ↓
[5] Chunk
  - 문서 구조 인식 (헤더·페이지·타임스탬프)
  - 청크당 600~1000자, overlap 100
  - metadata: page, week, start_ts 등
  ↓
[6] Embed (배치)
  - Gemini batchEmbedContents (100개씩)
  - taskType: RETRIEVAL_DOCUMENT
  - 768차원
  ↓
[7] Upsert
  - chunks 테이블에 트랜잭션으로 INSERT
  - 같은 document_id 청크 먼저 DELETE (재인덱싱 시)
  - tsvector 자동 생성 (generated column)
  ↓
[8] documents.status → 'ready'
  ↓
운영자 UI에 완료 표시 (SSE)
```

### 3.4 Admin UI

**페이지 구성:**
- `/admin/login` — Supabase Auth 매직 링크
- `/admin/files` — 파일 목록·업로드·삭제·재처리
- `/admin/test` — 질문 테스트 (어떤 청크가 retrieved 됐는지 시각화)
- `/admin/stats` — 청크 통계, dead chunk, "자료 없음" 응답률
- `/admin/feedback` — 사용자 피드백 (👍/👎) 모니터링

**기술 스택:**
- 정적 HTML + Vanilla JS + Tailwind CDN (빌드 도구 없음)
- Server-Sent Events로 진행 상황 실시간 표시
- 드래그앤드롭 업로드, 청크 미리보기

**왜 SPA·Next.js 아닌가:**
- 페이지 5개 수준 — 풀 프레임워크 과잉
- 빌드 파이프라인 없이 단일 배포로 끝
- 운영자 입장에선 차이 없음

---

## 4. 데이터 모델

### 4.1 Supabase 스키마

```sql
-- 확장
create extension if not exists vector;
create extension if not exists pg_trgm;

-- 원본 문서
create table documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  mime_type text,
  storage_path text not null,             -- Supabase Storage 경로
  size_bytes bigint,
  sha256 text,                            -- 중복 업로드 감지
  category text,                          -- '오픈마켓가입' | '강의자료' | '사업자등록' 등
  week int,                               -- 강의 주차 (옵션)
  extracted_text text,                    -- OCR 캐시
  status text not null default 'pending', -- pending | processing | ready | failed
  error_message text,
  chunk_count int default 0,
  uploaded_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index on documents (status);
create index on documents (category);
create unique index on documents (sha256) where sha256 is not null;

-- 청크
create table chunks (
  id bigserial primary key,
  document_id uuid not null references documents(id) on delete cascade,
  chunk_index int not null,               -- 문서 내 순번
  text text not null,
  embedding vector(768) not null,
  embed_model text not null default 'gemini-embedding-001',
  embed_dim int not null default 768,
  metadata jsonb default '{}'::jsonb,     -- { page: 12, start_ts: '00:12:34' }
  tsv tsvector generated always as (to_tsvector('simple', text)) stored,
  created_at timestamptz default now()
);

create index on chunks using hnsw (embedding vector_cosine_ops);
create index on chunks using gin (tsv);
create index on chunks (document_id);
create index on chunks ((metadata->>'category'));

-- 질문 로그 (관측성·평가)
create table queries (
  id bigserial primary key,
  user_id text,                           -- 카카오 user.id
  utterance text not null,
  rewritten_query text,
  retrieved_chunk_ids bigint[],
  answer text,
  sources text[],
  latency_ms int,
  llm_provider text,                      -- 어떤 fallback 사용했는지
  feedback smallint,                      -- -1 / 0 / +1
  created_at timestamptz default now()
);

create index on queries (created_at desc);
create index on queries (user_id);

-- 세션 (멀티턴 — Phase 4 이후)
create table sessions (
  user_id text primary key,
  history jsonb default '[]'::jsonb,
  updated_at timestamptz default now()
);
```

### 4.2 검색 RPC

```sql
-- 하이브리드 검색
create or replace function hybrid_search(
  query_embedding vector(768),
  query_text text,
  category_filter text default null,
  match_count int default 20
)
returns table (
  id bigint,
  document_id uuid,
  text text,
  metadata jsonb,
  source text,
  rrf_score float
)
language sql stable as $$
  with dense as (
    select c.id, row_number() over (order by c.embedding <=> query_embedding) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where category_filter is null or d.category = category_filter
    order by c.embedding <=> query_embedding
    limit match_count
  ),
  sparse as (
    select c.id, row_number() over (order by ts_rank(c.tsv, plainto_tsquery('simple', query_text)) desc) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where c.tsv @@ plainto_tsquery('simple', query_text)
      and (category_filter is null or d.category = category_filter)
    order by ts_rank(c.tsv, plainto_tsquery('simple', query_text)) desc
    limit match_count
  ),
  combined as (
    select id, sum(1.0 / (60 + rank)) as rrf_score
    from (
      select id, rank from dense
      union all
      select id, rank from sparse
    ) u
    group by id
  )
  select c.id, c.document_id, c.text, c.metadata,
         d.filename as source,
         co.rrf_score
  from combined co
  join chunks c on c.id = co.id
  join documents d on d.id = c.document_id
  order by co.rrf_score desc
  limit match_count;
$$;
```

---

## 5. 주요 데이터 흐름

### 5.1 봇 응답 (동기)

```
카카오 → POST /kakao/skill/{secret}
  → utterance 추출, 길이·인젝션 검증
  → rateLimit (사용자별 분당 20회)
  → Redis: 임베딩 캐시 확인
  → (miss) Gemini 임베딩 호출
  → Postgres hybrid_search RPC (top-20)
  → Cohere rerank (top-4)
  → Gemini Flash Lite 생성 + fallback
  → citation 검증
  → simpleText 응답
  → Langfuse trace 비동기 기록
  → queries 테이블 비동기 기록
```

**총 latency 목표 p95: 3초.** 5초 마진 안에서 callback 없이 처리 가능한 경우 동기로 끝.

### 5.2 봇 응답 (콜백 - 영상 검색·복잡 질의)

```
카카오 → POST /kakao/skill/callback/{secret}
  → 즉시 useCallback 응답 ("답변 생성 중... 최대 1분")  [5초 동기 SLA 안에 반환]
  → 백그라운드: RAG full pipeline + 45초 데드라인 가드
  → 완료(또는 데드라인 fallback) 시 callbackUrl로 1회 POST  [1분 윈도 내]
```

**Vercel에선 `waitUntil`로 백그라운드 작업 유지**(`maxDuration:60`). callbackUrl은 1분 유효·1회용이라, 45초 데드라인을 걸어 만료 전 fallback이라도 반드시 전달한다.

### 5.3 파일 업로드

```
운영자 → POST /admin/api/upload
  → Supabase Auth JWT 검증
  → 파일 크기 검증 (max 500MB)
  → Supabase Storage에 원본 PUT
  → documents 테이블 INSERT (status='pending')
  → 인메모리 큐 enqueue
  → 운영자 UI에 documentId 반환
  → SSE 채널 구독 시작
  ↓
워커 (별도 Promise loop):
  → 큐에서 dequeue
  → documents.status='processing'
  → 추출 → 청킹 → 임베딩 → upsert
  → documents.status='ready', chunk_count 갱신
  → SSE로 운영자에게 완료 알림
```

### 5.4 파일 재처리 (예: 모델 변경 시)

```
운영자 → POST /admin/api/documents/{id}/reingest
  → 인메모리 큐 enqueue
  → 워커가 extracted_text 캐시 사용 (OCR 재호출 없음)
  → 기존 chunks DELETE → 새로 INSERT
```

### 5.5 파일 삭제

```
운영자 → DELETE /admin/api/documents/{id}
  → Supabase Storage에서 원본 삭제
  → documents DELETE → chunks CASCADE 삭제
  → 즉시 봇 응답에서 해당 출처 사라짐 (재배포 없음)
```

---

## 6. 비기능 요구사항 충족 방안

### 6.1 가용성 (99.5%)

- **Fly.io 2 instances** (Tokyo `nrt` + Seoul `icn`) → 리전 장애 격리
- **min_machines_running = 2** → 무중단 롤링 배포
- **헬스체크** + 자동 재시작
- **LLM 3중 fallback** (Gemini → Claude → OpenAI)
- **Supabase 일일 백업** + S3 이중 백업
- **카카오 webhook 재시도**는 카카오 측에서 처리 (5xx 응답 시 자동 재시도)

### 6.2 응답시간

| 단계 | 목표 latency |
|---|---|
| 쿼리 임베딩 | <300ms (캐시 히트 시 <10ms) |
| 검색 + rerank | <500ms |
| 생성 | <2s |
| **합계 p95** | **<3s** |

**Tokyo 리전 선택 이유**: 한국 ↔ Tokyo 30~50ms, Seoul 5~10ms, Singapore 80ms+, US-West 150ms+.
Seoul 리전 추가 시 더 좋지만 Fly.io Seoul은 신생이라 안정성 검증 필요. 초기 Tokyo만, 안정화 후 Seoul 페일오버 추가.

### 6.3 보안

- **URL secret** (`/kakao/skill/{32자 랜덤}`) — webhook 직접 호출 방어
- **IP 화이트리스트** — 카카오 발신 IP만 (오픈빌더 문서 확인 필요)
- **Supabase Auth** — 어드민 인증 (매직 링크 + JWT)
- **express-rate-limit + Redis** — 사용자별 분당 20회
- **Gemini 월 예산 알람** — Google Cloud Billing
- **입력 검증** — utterance 길이 500자, 프롬프트 인젝션 패턴 차단
- **Secrets** — Fly Secrets·Supabase Vault, 절대 git에 X
- **PII** — 로그에서 사업자등록번호·연락처 마스킹
- **HTTPS only** — Fly 기본 제공

### 6.4 데이터 보호

- **Supabase 일일 백업** (Pro 티어 PITR)
- **별도 S3 매주 dump** — Supabase 전체 장애 대비
- **source-files 원본 영구 보관** (Supabase Storage) — 재인덱싱 가능
- **삭제는 soft delete 옵션** — 운영자 실수 복구 (Phase 4)

### 6.5 관측성

- **Langfuse trace** — 모든 RAG 호출 (query·retrieval·generation·latency·비용)
- **Sentry** — 미처리 예외
- **Pino structured logs** — Fly logs 검색 가능
- **대시보드**: 일일 활성 사용자, "자료 없음" 응답률, 평균 latency, LLM 비용, 피드백 점수

---

## 7. 단계별 구현 로드맵

### Phase 1: 기반 이관 (Week 1)
**목표: Vercel · chunks.json 의존 제거**

- [ ] Supabase 프로젝트 생성 + 스키마 적용 (documents, chunks, queries)
- [ ] Fly.io 앱 생성 + Dockerfile + fly.toml
- [ ] [src/rag.js](src/rag.js) `searchTopK` → Supabase RPC로 교체
- [ ] [scripts/ingest.js](scripts/ingest.js) → Supabase upsert
- [ ] CLI에서 기존 chunks.json을 Supabase로 마이그레이션
- [ ] 카카오 webhook URL을 Fly 도메인으로 전환 (스테이징 채널)
- [ ] `vercel.json`, `api/`, `update.bat` 제거

**검수 기준**: 스테이징 카카오 채널에서 기존과 동일 답변 + p95 < 3초

### Phase 2: 보안 & 관측성 (Week 2)
**목표: 실운영 노출 가능한 상태**

- [ ] webhook URL secret + rate limit
- [ ] 입력 검증·인젝션 가드
- [ ] Gemini 월 예산 알람 + 일일 호출 카운터
- [ ] Langfuse 연동 (trace + 대시보드)
- [ ] Sentry 연동
- [ ] Pino structured logging
- [ ] 스모크 테스트 30개

**검수 기준**: 30종 질문 자동 테스트 통과, abuse 시뮬레이션 차단 확인

### Phase 3: RAG 품질 (Week 3)
**목표: Naive → Advanced RAG**

- [ ] Cohere Rerank 도입
- [ ] Query rewriting
- [ ] Hybrid search (tsvector + dense, RRF)
- [ ] LLM fallback 체인 (Gemini → Claude Haiku → GPT-4o-mini)
- [ ] Citation 검증

**검수 기준**: 스모크 테스트 정확도 +20% 이상, faithfulness 0.9+

### Phase 4: 어드민 UI (Week 4-5)
**목표: 운영자 셀프서비스**

- [ ] `/admin/api/*` 라우트 (인증 미들웨어 포함)
- [ ] Supabase Storage 업로드 로직
- [ ] 인메모리 잡 큐 + 워커 (Promise loop)
- [ ] SSE로 진행상황 알림
- [ ] 정적 HTML 어드민: 파일 목록·업로드·삭제·재처리
- [ ] 봇 테스트 페이지 (retrieved 청크 시각화)
- [ ] 운영자 매뉴얼 (PDF)

**검수 기준**: 운영자가 5분 안에 새 PDF 업로드 → 봇 답변 확인 가능

### Phase 5: 운영 안정화 (Week 6)
**목표: 24/7 운영 가능**

- [ ] Fly.io Seoul 리전 추가 (페일오버)
- [ ] Supabase Pro 업그레이드 (PITR)
- [ ] S3 매주 자동 백업
- [ ] Runbook 작성 (장애 대응 절차)
- [ ] 부하 테스트 (k6, 예상 피크 × 3배)
- [ ] 카카오 검수 신청

**검수 기준**: 7일간 무장애, 부하 테스트 통과

### Phase 6+: 고도화 (Month 2+)
- 멀티턴 세션 (Upstash Redis + sessions 테이블)
- 사용자 피드백 (👍/👎) 수집·분석
- "자료 없음" 응답 자동 집계 → 지식베이스 gap 알림
- 영상 타임스탬프 보존 + citation에 "3주차 12:34" 표시
- 슬라이드 + 전사 결합 (멀티모달 인덱싱)
- 평가 세트 + CI 자동 측정 (Ragas)

---

## 8. 비용 추정

### 무료/저비용 단계 (DAU 50, 청크 1,000)

| 항목 | 월 비용 |
|---|---|
| Fly.io shared-cpu-1x × 1 (256MB) | $0 (무료 한도 내) |
| Supabase Free | $0 |
| Gemini Flash Lite | $5~10 |
| Cohere Rerank | $0 (무료 1k req/월) |
| Langfuse self-host | $0 (Fly에 컨테이너 추가) |
| **합계** | **~$10** |

### 안정 운영 단계 (DAU 500, 청크 10,000)

| 항목 | 월 비용 |
|---|---|
| Fly.io shared-cpu-2x × 2 (1GB, Tokyo+Seoul) | $14 |
| Supabase Pro | $25 |
| Gemini Flash Lite | $50~80 |
| Claude Haiku (fallback 5%) | $10 |
| Cohere Rerank | $0 (무료 한도) |
| Upstash Redis | $0~10 |
| Langfuse Cloud | $0 (Hobby) |
| Sentry | $0 (Developer) |
| **합계** | **~$110~140** |

### 확장 단계 (DAU 5,000, 청크 50,000)

| 항목 | 월 비용 |
|---|---|
| Fly.io shared-cpu-4x × 3 | $50 |
| Supabase Pro + add-ons | $50 |
| Gemini Flash Lite | $400~700 |
| Claude Haiku (fallback 10%) | $80 |
| Cohere Rerank Pro | $30 |
| Upstash Redis Pay-as-you-go | $20 |
| Langfuse Pro | $50 |
| Sentry Team | $30 |
| **합계** | **~$700~1,000** |

**병목: Gemini API 비용.** 트래픽 늘면 캐싱·프롬프트 압축·Haiku 비율 조정으로 절감.

---

## 9. 주요 설계 결정 (ADR 요약)

### ADR-001: 호스팅 = Fly.io Tokyo
**대안**: Vercel, Railway, Render, AWS ECS, Cloud Run, NCP
**결정 근거**:
- 장수명 프로세스 → 백그라운드 잡 안전
- Tokyo 리전 → 한국 latency 적절
- Docker 기반 → 락인 약함, 이관 용이
- 가격 ($14/월) 대비 성능 양호
**대안 대비 단점**: Seoul 리전이 신생, 데이터 주권 이슈 시 NCP 검토 필요

### ADR-002: 벡터 DB = Supabase pgvector
**대안**: Pinecone, Qdrant, Weaviate, 자체 호스팅 pgvector
**결정 근거**:
- Postgres + Storage + Auth 한 서비스로 묶임
- pgvector + HNSW로 ANN 충분히 빠름
- 락인 약함 (표준 PostgreSQL)
- 무료 티어로 시작 가능
**대안 대비 단점**: 한국 리전 없음 (Tokyo가 가장 가까움), 초대형 스케일에서 전용 벡터 DB가 더 빠름

### ADR-003: 어드민 = 정적 HTML + Vanilla JS
**대안**: Next.js, React SPA, HTMX
**결정 근거**:
- 페이지 5개 수준
- 빌드 파이프라인 없음 → 단일 배포
- 운영자 입장에서 차이 없음
**나중에 바꿀 수 있나**: 가능. UI 복잡해지면 Next.js로 분리

### ADR-004: 잡 큐 = 인메모리 + 단일 워커
**대안**: BullMQ + Redis, QStash, pgmq
**결정 근거**:
- 운영자 1인이 동시에 여러 파일 올릴 일 드물다
- 동시성 1로 시작, 큐 길이 모니터링
- 필요 시 pgmq로 확장 (코드 변경 최소)
**나중에 바꿀 수 있나**: 워커 모듈 인터페이스만 유지하면 백엔드 교체 가능

### ADR-005: LLM = Gemini Primary + Claude/OpenAI Fallback
**대안**: 단일 벤더, Claude를 primary로
**결정 근거**:
- Gemini Flash Lite = 가격/속도 최적, 임베딩까지 통합
- 503 빈번 → fallback 필수
- 한국어 품질 모두 양호
**언제 재검토**: Gemini가 안정화되면 fallback 비율 낮춤, 또는 Claude가 더 싸지면 primary 교체

---

## 10. 리스크 & 미해결

### 10.1 알려진 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| 카카오 IP 화이트리스트 미공개 | URL secret 단독 의존 | secret 충분히 길게 (64자), 정기 로테이션 |
| Tokyo ↔ Seoul latency | 한국 사용자 체감 50ms 추가 | 안정화 후 Seoul 추가 |
| Supabase 도쿄 리전 | 개인정보 국내보관 의무 위반 가능 | 강의 자료만 저장, 개인정보는 별도 국내 DB |
| Gemini Files API 비용 폭증 | 영상 대량 업로드 시 운영비 폭탄 | 일일 OCR 할당량 + 운영자 경고 |
| 단일 운영자 | 새벽 장애 무대응 | 자동 재시작·페일오버·알림 자동화 강화 |

### 10.2 미해결 (Phase별 결정)

- 멀티턴 세션 도입 시점 — 사용자 피드백 보고 결정
- 영상 타임스탬프 보존 vs 청크 텍스트 가독성 — A/B 테스트 필요
- 한국어 BM25 토크나이저 (`pg_trgm` vs 외부 형태소 분석기)
- 카테고리 분류 자동화 (LLM 분류 vs 수동 태깅) — 초기엔 수동, Phase 6 검토

### 10.3 의도적으로 안 한 것

- **Kubernetes**: 1인 운영에 과잉
- **자체 LLM 호스팅**: 운영 부담·품질·비용 모두 불리
- **GraphQL**: REST 8개 엔드포인트로 충분
- **모노레포 / 마이크로서비스 분리**: 단일 Express 앱이 적절
- **CDN**: 정적 자산 적고 카카오 트래픽이 주력

---

## 11. 운영 매뉴얼 (요약)

### 11.1 일상 운영자 작업
1. 브라우저로 `admin.kakao-class-bot.fly.dev` 접속
2. 매직 링크 로그인 (이메일)
3. 파일 드래그앤드롭 업로드
4. 처리 상태 ✅ 확인
5. "봇 테스트" 페이지에서 검색 결과 확인
6. 끝.

### 11.2 알람 대응 (관리자)
- **Slack #kakao-bot-alerts** 채널로 알람
- 알람 종류:
  - `error_rate_high`: 에러율 5% 초과 (10분 윈도우)
  - `latency_high`: p95 5초 초과
  - `llm_budget_warning`: Gemini 월 예산 80% 도달
  - `supabase_down`: 헬스체크 실패
- Runbook 참고 (별도 문서)

### 11.3 정기 점검 (월 1회)
- Langfuse 대시보드: 평균 latency·비용·피드백 점수
- "자료 없음" 응답 상위 30개 → 지식베이스 보강 후보
- Dead chunk (한 번도 retrieved 안 된 청크) 검토
- Supabase 백업 복원 테스트 (분기 1회)

---

## 12. 참고

- 현재 코드: [src/](src/), [scripts/](scripts/)
- 개선 우선순위: [IMPROVEMENTS.md](IMPROVEMENTS.md)
- 메모리: 비개발자 운영자 대상 — UX 단순화가 항상 1순위

---

**문서 변경 이력**
- 2026-05-27: 초안 작성
