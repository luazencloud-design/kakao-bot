import { createServiceClient } from '@/lib/supabase/server';
import { FeedbackList, type FeedbackItem } from '@/components/feedback/feedback-list';

export const dynamic = 'force-dynamic';

export default async function FeedbackPage() {
  const admin = createServiceClient();

  // 조치 필요 항목: 부정 피드백 + 자료없음 + 오류 (해결됨/안됨 모두 가져와 클라에서 필터)
  const { data: negative } = await admin
    .from('queries')
    .select('id, utterance, answer, sources, feedback, feedback_comment, llm_provider, resolved_at, created_at')
    .eq('feedback', -1)
    .order('created_at', { ascending: false })
    .limit(100);

  const { data: gaps } = await admin
    .from('queries')
    .select('id, utterance, answer, sources, feedback, feedback_comment, llm_provider, resolved_at, created_at')
    .or('sources.is.null,sources.eq.{}')
    .order('created_at', { ascending: false })
    .limit(100);

  const map = new Map<number, FeedbackItem>();
  for (const q of [...(negative ?? []), ...(gaps ?? [])] as FeedbackItem[]) {
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
          개선이 필요한 답변·자료 없는 질문·오류를 모았습니다. 처리한 항목은 해결로 표시하세요.
        </p>
      </header>

      <FeedbackList items={items} />
    </div>
  );
}
