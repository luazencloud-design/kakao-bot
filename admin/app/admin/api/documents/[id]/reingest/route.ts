// POST /admin/api/documents/:id/reingest — 캐시된 텍스트 또는 Storage 원본으로 재처리

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { processDocument } from '@/lib/ingest/process';

export const maxDuration = 300;

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const { id } = await params;
  try {
    const { chunkCount } = await processDocument(id);
    return NextResponse.json({ ok: true, chunkCount });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ ok: false, error: message }, { status: 422 });
  }
}
