// DELETE /admin/api/documents/:id  — 문서 + 청크 + Storage 원본 삭제

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await requireAdmin();
  if (!user) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  const { id } = await params;
  const admin = createServiceClient();

  // 1. 문서 조회 (storage_path 확보)
  const { data: doc, error: fetchErr } = await admin
    .from('documents')
    .select('id, storage_path')
    .eq('id', id)
    .maybeSingle();

  if (fetchErr || !doc) {
    return NextResponse.json({ error: 'not_found' }, { status: 404 });
  }

  // 2. Storage 원본 삭제 (cli-ingest/migrated 등 더미 경로는 무시)
  if (doc.storage_path && !doc.storage_path.startsWith('cli-ingest/') && !doc.storage_path.startsWith('migrated/')) {
    await admin.storage.from('source-files').remove([doc.storage_path]);
  }

  // 3. 문서 삭제 (chunks는 ON DELETE CASCADE로 자동)
  const { error: delErr } = await admin.from('documents').delete().eq('id', id);
  if (delErr) {
    return NextResponse.json({ error: delErr.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
