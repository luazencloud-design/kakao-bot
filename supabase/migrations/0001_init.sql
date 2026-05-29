-- =====================================================================
-- kakao-class-bot 초기 스키마
-- Supabase SQL Editor에 통째로 붙여넣고 실행
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. 확장 (pgvector, pg_trgm)
-- ---------------------------------------------------------------------
create extension if not exists vector;
create extension if not exists pg_trgm;

-- ---------------------------------------------------------------------
-- 2. documents — 원본 파일 메타데이터
-- ---------------------------------------------------------------------
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  mime_type text,
  storage_path text not null,             -- Supabase Storage 경로
  size_bytes bigint,
  sha256 text,                            -- 중복 업로드 감지
  category text,                          -- '오픈마켓가입' | '강의자료' | '사업자등록' | '도구가이드' | '기타'
  week int,                               -- 강의 주차 (옵션)
  extracted_text text,                    -- OCR/추출 결과 캐시
  status text not null default 'pending', -- pending | processing | ready | failed
  error_message text,
  chunk_count int default 0,
  uploaded_by uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists documents_status_idx on documents (status);
create index if not exists documents_category_idx on documents (category);
create unique index if not exists documents_sha256_uniq on documents (sha256) where sha256 is not null;

-- updated_at 자동 갱신
create or replace function set_updated_at() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end;
$$;

drop trigger if exists documents_updated_at on documents;
create trigger documents_updated_at before update on documents
  for each row execute function set_updated_at();

-- ---------------------------------------------------------------------
-- 3. chunks — 임베딩된 텍스트 청크
-- ---------------------------------------------------------------------
create table if not exists chunks (
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

create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_tsv_idx on chunks using gin (tsv);
create index if not exists chunks_document_idx on chunks (document_id);
create index if not exists chunks_metadata_category_idx
  on chunks ((metadata->>'category'));

-- ---------------------------------------------------------------------
-- 4. queries — 질의 로그 (관측성·피드백·평가)
-- ---------------------------------------------------------------------
create table if not exists queries (
  id bigserial primary key,
  user_id text,                           -- 카카오 user.id
  utterance text not null,
  rewritten_query text,
  retrieved_chunk_ids bigint[],
  answer text,
  sources text[],
  latency_ms int,
  llm_provider text,                      -- 'gemini' | 'claude' | 'openai' (fallback 추적)
  feedback smallint default 0,            -- -1 (부정) / 0 (없음) / +1 (긍정)
  feedback_comment text,
  created_at timestamptz default now()
);

create index if not exists queries_created_idx on queries (created_at desc);
create index if not exists queries_user_idx on queries (user_id);
create index if not exists queries_feedback_idx on queries (feedback) where feedback != 0;

-- ---------------------------------------------------------------------
-- 5. sessions — 멀티턴 대화 (Phase 6에 활성화)
-- ---------------------------------------------------------------------
create table if not exists sessions (
  user_id text primary key,
  history jsonb default '[]'::jsonb,
  updated_at timestamptz default now()
);

-- ---------------------------------------------------------------------
-- 6. allowed_admins — 어드민 이메일 화이트리스트
-- 매직링크로 누구나 가입 시도할 수 있어서, 여기 등록된 이메일만 통과
-- ---------------------------------------------------------------------
create table if not exists allowed_admins (
  email text primary key,
  added_at timestamptz default now(),
  added_by text,
  note text
);

-- ---------------------------------------------------------------------
-- 7. hybrid_search RPC
-- Dense (pgvector cosine) + Sparse (tsvector BM25-ish) → RRF
-- ---------------------------------------------------------------------
create or replace function hybrid_search(
  query_embedding vector(768),
  query_text text,
  category_filter text default null,
  match_count int default 20
)
returns table (
  id bigint,
  document_id uuid,
  chunk_text text,
  metadata jsonb,
  source text,
  rrf_score float
)
language sql stable as $$
  with dense as (
    select c.id,
           row_number() over (order by c.embedding <=> query_embedding) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where (category_filter is null or d.category = category_filter)
      and d.status = 'ready'
    order by c.embedding <=> query_embedding
    limit match_count
  ),
  sparse as (
    select c.id,
           row_number() over (
             order by ts_rank(c.tsv, plainto_tsquery('simple', query_text)) desc
           ) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where c.tsv @@ plainto_tsquery('simple', query_text)
      and (category_filter is null or d.category = category_filter)
      and d.status = 'ready'
    order by ts_rank(c.tsv, plainto_tsquery('simple', query_text)) desc
    limit match_count
  ),
  combined as (
    select id, sum(1.0 / (60 + rank))::float as rrf_score
    from (
      select id, rank from dense
      union all
      select id, rank from sparse
    ) u
    group by id
  )
  select c.id,
         c.document_id,
         c.text as chunk_text,
         c.metadata,
         d.filename as source,
         co.rrf_score
  from combined co
  join chunks c on c.id = co.id
  join documents d on d.id = c.document_id
  order by co.rrf_score desc
  limit match_count;
$$;

-- ---------------------------------------------------------------------
-- 7.5. GRANTs (테이블 권한)
-- "Automatically expose new tables" OFF로 만든 프로젝트는 raw SQL로
-- 만든 테이블에 자동 GRANT가 안 들어감. 그래서 명시적으로 부여.
-- service_role은 RLS bypass + 전체 권한, authenticated는 RLS 적용,
-- anon은 명시적 허용 외 차단.
-- ---------------------------------------------------------------------
grant usage on schema public to anon, authenticated, service_role;

grant all on all tables in schema public to service_role;
grant all on all sequences in schema public to service_role;
grant all on all functions in schema public to service_role;

grant select, insert, update, delete on all tables in schema public to authenticated;
grant usage, select on all sequences in schema public to authenticated;
grant execute on all functions in schema public to authenticated;

revoke all on all tables in schema public from anon;

alter default privileges in schema public grant all on tables to service_role;
alter default privileges in schema public grant all on sequences to service_role;
alter default privileges in schema public grant all on functions to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to authenticated;
alter default privileges in schema public grant usage, select on sequences to authenticated;
alter default privileges in schema public grant execute on functions to authenticated;

-- ---------------------------------------------------------------------
-- 8. RLS (Row Level Security)
-- 어드민 작업은 service_role 키로만, 일반 anon은 접근 불가
-- ---------------------------------------------------------------------
alter table documents enable row level security;
alter table chunks enable row level security;
alter table queries enable row level security;
alter table sessions enable row level security;
alter table allowed_admins enable row level security;

-- allowed_admins: 인증된 유저가 "자기 행"만 읽기 가능.
-- 아래 documents/chunks/queries 정책들이 내부에서 allowed_admins를 조회하므로
-- 이 정책이 없으면 RLS에 막혀 모든 조회가 빈 결과가 됨.
drop policy if exists "authenticated_can_check_self_admin" on allowed_admins;
create policy "authenticated_can_check_self_admin" on allowed_admins
  for select
  to authenticated
  using (email = auth.email());

-- 인증된 운영자만 자료 관리 가능
drop policy if exists "admin_full_documents" on documents;
create policy "admin_full_documents" on documents
  for all
  to authenticated
  using (
    exists (select 1 from allowed_admins where email = auth.email())
  )
  with check (
    exists (select 1 from allowed_admins where email = auth.email())
  );

drop policy if exists "admin_full_chunks" on chunks;
create policy "admin_full_chunks" on chunks
  for all
  to authenticated
  using (
    exists (select 1 from allowed_admins where email = auth.email())
  )
  with check (
    exists (select 1 from allowed_admins where email = auth.email())
  );

drop policy if exists "admin_read_queries" on queries;
create policy "admin_read_queries" on queries
  for select
  to authenticated
  using (
    exists (select 1 from allowed_admins where email = auth.email())
  );

-- ---------------------------------------------------------------------
-- 9. Storage 버킷 (수동: Supabase 대시보드 → Storage → New bucket)
-- 이름: source-files
-- Public: NO (signed URL로만 접근)
-- File size limit: 500 MB
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------
-- 10. 초기 어드민 등록 예시 (실제 사용 시 본인 이메일로 수정)
-- ---------------------------------------------------------------------
-- insert into allowed_admins (email, note) values ('admin@example.com', '초기 운영자');
