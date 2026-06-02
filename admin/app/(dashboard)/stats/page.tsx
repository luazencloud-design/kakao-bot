import { createServiceClient } from '@/lib/supabase/server';
import { CATEGORY_LABELS } from '@/lib/types';

export const dynamic = 'force-dynamic';

interface QueryRow {
  user_id: string | null;
  sources: string[] | null;
  latency_ms: number | null;
  feedback: number | null;
  created_at: string;
  utterance: string;
}

export default async function StatsPage() {
  const admin = createServiceClient();

  // 최근 30일 질의 (admin-test 제외하고 실사용자만 집계, 단 데이터 적으면 전체)
  const since = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
  const { data: allQueries } = await admin
    .from('queries')
    .select('user_id, sources, latency_ms, feedback, created_at, utterance')
    .gte('created_at', since)
    .order('created_at', { ascending: false });

  const queries = (allQueries ?? []) as QueryRow[];
  const realQueries = queries.filter((q) => q.user_id !== 'admin-test');
  // 실사용자 데이터 없으면 테스트 포함해서 보여줌
  const data = realQueries.length > 0 ? realQueries : queries;

  const totalQueries = data.length;
  const uniqueUsers = new Set(data.map((q) => q.user_id)).size;
  const avgLatency =
    totalQueries > 0
      ? Math.round(data.reduce((s, q) => s + (q.latency_ms ?? 0), 0) / totalQueries)
      : 0;
  const unanswered = data.filter((q) => !q.sources || q.sources.length === 0).length;
  const answeredRate =
    totalQueries > 0 ? Math.round(((totalQueries - unanswered) / totalQueries) * 100) : 0;

  // 문서 통계
  const { data: docs } = await admin
    .from('documents')
    .select('category, chunk_count');
  const docList = docs ?? [];
  const byCategory: Record<string, { docs: number; chunks: number }> = {};
  for (const d of docList) {
    const cat = d.category ?? '기타';
    if (!byCategory[cat]) byCategory[cat] = { docs: 0, chunks: 0 };
    byCategory[cat].docs += 1;
    byCategory[cat].chunks += d.chunk_count ?? 0;
  }
  const categories = Object.entries(byCategory).sort((a, b) => b[1].chunks - a[1].chunks);
  const maxChunks = Math.max(...categories.map(([, v]) => v.chunks), 1);

  // 일별 질의 (최근 7일)
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(Date.now() - (6 - i) * 24 * 60 * 60 * 1000);
    return { label: ['일', '월', '화', '수', '목', '금', '토'][d.getDay()], date: d.toISOString().slice(0, 10), count: 0 };
  });
  for (const q of data) {
    const day = days.find((d) => d.date === q.created_at.slice(0, 10));
    if (day) day.count += 1;
  }
  const maxDay = Math.max(...days.map((d) => d.count), 1);

  // 답변 못한 질문 모음
  const unansweredQs = data
    .filter((q) => !q.sources || q.sources.length === 0)
    .slice(0, 6);

  const catColors: Record<string, string> = {
    오픈마켓가입: 'bg-purple-500',
    사업자등록: 'bg-emerald-500',
    강의자료: 'bg-blue-500',
    도구가이드: 'bg-amber-500',
    기타: 'bg-slate-400',
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <header>
        <h1 className="text-xl font-bold text-slate-900">통계</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          최근 30일 · {realQueries.length === 0 && queries.length > 0 ? '(테스트 질의 포함)' : '실사용자 기준'}
        </p>
      </header>

      {/* KPI */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Kpi label="총 질문" value={totalQueries.toString()} />
        <Kpi label="사용자 수" value={uniqueUsers.toString()} />
        <Kpi label="평균 응답시간" value={`${(avgLatency / 1000).toFixed(1)}초`} hint={avgLatency < 5000 ? '5초 이내 ✓' : '5초 초과 ⚠'} />
        <Kpi label="답변 성공률" value={`${answeredRate}%`} valueClass={answeredRate >= 80 ? 'text-emerald-600' : 'text-amber-600'} />
      </div>

      {/* 일별 차트 */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h2 className="text-sm font-semibold text-slate-900 mb-5">일별 질문 수 (최근 7일)</h2>
        {totalQueries === 0 ? (
          <div className="text-center text-sm text-slate-400 py-8">
            아직 질문 데이터가 없습니다. 봇 배포 후 또는 답변 테스트 시 집계됩니다.
          </div>
        ) : (
          <div className="flex items-end gap-3 h-40">
            {days.map((d, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-2">
                <div className="w-full bg-blue-100 rounded-t relative" style={{ height: `${(d.count / maxDay) * 100}%`, minHeight: d.count > 0 ? '4px' : '0' }}>
                  {d.count > 0 && (
                    <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-xs text-slate-500">{d.count}</span>
                  )}
                </div>
                <span className="text-xs text-slate-500">{d.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 카테고리 분포 (문서 기준) */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="text-sm font-semibold text-slate-900 mb-4">자료 카테고리 분포</h2>
          <div className="space-y-3">
            {categories.map(([cat, v]) => (
              <div key={cat}>
                <div className="flex items-center justify-between text-sm mb-1.5">
                  <span className="text-slate-700">{CATEGORY_LABELS[cat] ?? cat}</span>
                  <span className="text-slate-500 text-xs">
                    {v.docs}개 자료 · {v.chunks}청크
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${catColors[cat] ?? 'bg-slate-400'}`}
                    style={{ width: `${(v.chunks / maxChunks) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 답변 못한 질문 */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-900">답변 못한 질문</h2>
            <span className="text-xs px-2 py-0.5 bg-red-50 text-red-700 rounded-full">
              {unanswered}개
            </span>
          </div>
          {unansweredQs.length === 0 ? (
            <div className="text-center text-sm text-slate-400 py-8">
              답변 못한 질문이 없습니다 👍
            </div>
          ) : (
            <div className="space-y-3">
              {unansweredQs.map((q, i) => (
                <div key={i} className="text-sm">
                  <div className="text-slate-900">&ldquo;{q.utterance}&rdquo;</div>
                  <div className="text-xs text-slate-500 mt-0.5">
                    {new Date(q.created_at).toLocaleDateString('ko-KR')} · 자료 없음
                  </div>
                </div>
              ))}
              <p className="text-xs text-slate-400 pt-2">
                이런 질문들은 관련 자료를 추가하면 답변할 수 있게 됩니다.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  hint,
  valueClass = 'text-slate-900',
}: {
  label: string;
  value: string;
  hint?: string;
  valueClass?: string;
}) {
  return (
    <div className="bg-white rounded-xl p-4 border border-slate-200">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`text-2xl font-bold mt-1 ${valueClass}`}>{value}</div>
      {hint && <div className="text-xs text-slate-400 mt-1">{hint}</div>}
    </div>
  );
}
