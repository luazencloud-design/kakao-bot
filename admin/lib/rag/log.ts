// 질의 로깅: queries 테이블에 기록 (통계·피드백·평가용).
// 실패해도 응답에 영향 없도록 swallow.

import { createServiceClient } from '@/lib/supabase/server';
import type { RagResult } from '@/lib/rag/query';

interface LogInput {
  userId: string; // 'admin-test' | 카카오 user.id
  utterance: string;
  result: RagResult;
}

export async function logQuery({ userId, utterance, result }: LogInput): Promise<number | null> {
  try {
    const admin = createServiceClient();
    const isUnanswered = result.answer.includes('제공된 자료에 포함되어 있지 않');
    const { data } = await admin
      .from('queries')
      .insert({
        user_id: userId,
        utterance,
        rewritten_query: result.rewrittenQuery,
        retrieved_chunk_ids: result.chunks.map((c) => c.id),
        answer: result.answer,
        sources: isUnanswered ? [] : [...new Set(result.chunks.map((c) => c.source))],
        latency_ms: result.timings.total,
        llm_provider: 'gemini',
      })
      .select('id')
      .single();
    return data?.id ?? null;
  } catch {
    return null;
  }
}
