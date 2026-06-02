// POST /admin/api/test  { question }  → RAG 답변 + 검색된 청크 + 타이밍

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { ragQuery } from '@/lib/rag/query';

export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const { question } = await request.json();
  if (!question || typeof question !== 'string' || !question.trim()) {
    return NextResponse.json({ error: '질문을 입력하세요.' }, { status: 400 });
  }
  if (question.length > 500) {
    return NextResponse.json({ error: '질문이 너무 깁니다 (최대 500자).' }, { status: 400 });
  }

  try {
    const result = await ragQuery(question.trim());
    return NextResponse.json({ ok: true, ...result });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
