-- 피드백/이슈 해결 상태 (null = 미해결, 값 = 해결 시각)
alter table queries add column if not exists resolved_at timestamptz;
create index if not exists queries_resolved_idx
  on queries (resolved_at) where feedback != 0 or resolved_at is not null;
