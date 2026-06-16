-- 재작성 캐시 (검색 결정성 + 봇/어드민 일치)
-- 문제: 쿼리 재작성이 Gemini 호출이라 같은 질문에도 결과가 매번 달라져
-- (변형 A/B) sparse 검색어가 바뀌고 → 다른 청크 → 답이 흔들림.
-- 해결: 정규화한 질문을 키로 재작성 결과를 캐시. 같은 질문은 저장된 재작성을
-- 재사용 → 결정적. 봇·어드민이 같은 테이블을 공유하므로 둘의 답도 일치한다.
--
-- rewrite는 "보강 키워드"만 저장(원질문 제외). 빈 문자열 = "유용한 재작성 없음".

create table if not exists query_rewrites (
  q_norm text primary key,                 -- 정규화 질문 (trim·lowercase·공백정리)
  rewrite text not null default '',        -- 재작성 보강 키워드 ('' = 없음)
  created_at timestamptz not null default now()
);

-- service_role 전체 권한(default privileges로 자동 부여되나 명시), anon 차단.
grant all on query_rewrites to service_role;
revoke all on query_rewrites from anon;

-- service_role(봇·어드민)만 접근. RLS 켜고 정책 없음 → anon/authenticated 차단.
alter table query_rewrites enable row level security;
