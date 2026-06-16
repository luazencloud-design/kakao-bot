-- 결정적 정렬 (비결정 검색 결과 방지)
-- 문제: dense/sparse 랭킹과 최종 RRF 정렬이 점수만으로 ORDER BY 되어,
-- 점수가 동점이면 Postgres가 행 순서를 임의로 반환 → 같은 질문에 청크가
-- 들쭉날쭉 잡혀 답이 흔들림. 모든 ORDER BY에 c.id 보조 정렬을 추가해
-- 동점도 항상 같은 순서로 결정되게 한다. (0002의 함수를 그대로 재정의)

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
           row_number() over (order by c.embedding <=> query_embedding, c.id) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where (category_filter is null or d.category = category_filter)
      and d.status = 'ready'
    order by c.embedding <=> query_embedding, c.id
    limit greatest(match_count * 5, 20)
  ),
  sparse as (
    select c.id,
           row_number() over (order by word_similarity(query_text, c.text) desc, c.id) as rank
    from chunks c
    join documents d on d.id = c.document_id
    where (category_filter is null or d.category = category_filter)
      and d.status = 'ready'
      and word_similarity(query_text, c.text) > 0.1
    order by word_similarity(query_text, c.text) desc, c.id
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
  order by co.rrf_score desc, c.id
  limit match_count;
$$;
