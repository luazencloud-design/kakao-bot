-- 한국어 검색 개선 (Task 10)
-- tsvector('simple')는 한국어 토큰화를 못 해 sparse 검색이 무력했음.
-- pg_trgm word_similarity로 교체 → 한국어 키워드 매칭 작동.
-- word_similarity(짧은 질문, 긴 청크) = 질문이 청크 내 어느 구간과 겹치는 정도.

create index if not exists chunks_text_trgm_idx
  on chunks using gin (text gin_trgm_ops);

create or replace function hybrid_search(
  query_embedding vector(768),
  query_text text,
  category_filter text default null,
  match_count int default 4
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
    limit greatest(match_count * 5, 20)
  ),
  sparse as (
    select c.id,
           row_number() over (order by word_similarity(query_text, c.text) desc) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where (category_filter is null or d.category = category_filter)
      and d.status = 'ready'
      and word_similarity(query_text, c.text) > 0.1
    order by word_similarity(query_text, c.text) desc
    limit greatest(match_count * 5, 20)
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
