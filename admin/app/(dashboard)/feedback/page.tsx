import Link from 'next/link';
import { ThumbsDown, Lightbulb, Upload } from 'lucide-react';
import { createServiceClient } from '@/lib/supabase/server';

export const dynamic = 'force-dynamic';

interface QueryRow {
  id: number;
  utterance: string;
  answer: string | null;
  sources: string[] | null;
  feedback: number | null;
  feedback_comment: string | null;
  created_at: string;
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const h = Math.floor(diff / 3600000);
  if (h < 1) return '방금';
  if (h < 24) return `${h}시간 전`;
  return `${Math.floor(h / 24)}일 전`;
}

export default async function FeedbackPage() {
  const admin = createServiceClient();

  // 부정 피드백 + 답변 못한 질문 (운영자가 조치할 항목)
  const { data: negative } = await admin
    .from('queries')
    .select('id, utterance, answer, sources, feedback, feedback_comment, created_at')
    .eq('feedback', -1)
    .order('created_at', { ascending: false })
    .limit(20);

  const { data: unanswered } = await admin
    .from('queries')
    .select('id, utterance, answer, sources, feedback, feedback_comment, created_at')
    .or('sources.is.null,sources.eq.{}')
    .order('created_at', { ascending: false })
    .limit(20);

  // 합치고 중복 제거
  const map = new Map<number, QueryRow>();
  for (const q of [...(negative ?? []), ...(unanswered ?? [])] as QueryRow[]) {
    map.set(q.id, q);
  }
  const items = [...map.values()].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  );

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <header>
        <h1 className="text-xl font-bold text-slate-900">사용자 피드백</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          개선이 필요한 답변과 자료가 없어 답하지 못한 질문을 모았습니다
        </p>
      </header>

      {items.length === 0 ? (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <div className="text-sm text-slate-500">조치가 필요한 항목이 없습니다 👍</div>
          <p className="text-xs text-slate-400 mt-2">
            사용자가 👎 표시했거나 자료 부족으로 답변 못한 질문이 여기 표시됩니다.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((q) => {
            const isUnanswered = !q.sources || q.sources.length === 0;
            return (
              <div key={q.id} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                <div className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    {q.feedback === -1 ? (
                      <span className="inline-flex items-center gap-1.5 text-xs px-2 py-1 bg-red-50 text-red-700 rounded-full">
                        <ThumbsDown className="w-3 h-3" />
                        개선 필요
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 text-xs px-2 py-1 bg-amber-50 text-amber-700 rounded-full">
                        <Lightbulb className="w-3 h-3" />
                        자료 없음
                      </span>
                    )}
                    <span className="text-xs text-slate-500">{timeAgo(q.created_at)}</span>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="text-xs text-slate-500 mb-1">질문</div>
                      <div className="text-sm text-slate-900">{q.utterance}</div>
                    </div>
                    {q.answer && (
                      <div>
                        <div className="text-xs text-slate-500 mb-1">챗봇 답변</div>
                        <div className="text-sm text-slate-700 bg-slate-50 rounded-lg p-3 leading-relaxed line-clamp-3">
                          {q.answer}
                        </div>
                      </div>
                    )}
                    {q.feedback_comment && (
                      <div>
                        <div className="text-xs text-slate-500 mb-1">사용자 코멘트</div>
                        <div className="text-sm text-slate-700 italic">
                          &ldquo;{q.feedback_comment}&rdquo;
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
                    {isUnanswered ? (
                      <div className="text-xs text-amber-700 bg-amber-50 px-3 py-1.5 rounded-lg flex items-center gap-1.5">
                        <Lightbulb className="w-3.5 h-3.5" />
                        관련 자료를 추가하면 답변할 수 있습니다
                      </div>
                    ) : (
                      <div className="text-xs text-slate-500">
                        참고 자료: {q.sources?.join(', ')}
                      </div>
                    )}
                    <Link
                      href="/files"
                      className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5"
                    >
                      <Upload className="w-3.5 h-3.5" />
                      자료 추가하기
                    </Link>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
