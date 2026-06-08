// DELETE /admin/api/documents/:id  — 문서 + 청크 + Storage 원본 삭제
// PATCH  /admin/api/documents/:id  — 카테고리 등 메타데이터 수정

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';

const VALID_CATEGORIES = ['오픈마켓가입', '사업자등록', '강의자료', '도구가이드', '기타'];

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await requireAdmin();
  if (!user) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  const { id } = await params;
  const body = await request.json().catch(() => ({}));
  const category = body.category;

  if (!VALID_CATEGORIES.includes(category)) {
    return NextResponse.json({ error: '유효하지 않은 카테고리' }, { status: 400 });
  }

  const admin = createServiceClient();
  const { error } = await admin
    .from('documents')
    .update({ category })
    .eq('id', id);
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}

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
