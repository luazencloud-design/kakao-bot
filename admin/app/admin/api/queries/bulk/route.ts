// POST /admin/api/queries/bulk  { ids: number[], action: 'resolve'|'unresolve'|'delete' }
// 여러 이슈를 한 번에 해결/미해결/삭제

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';

export async function POST(request: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const body = await request.json().catch(() => ({}));
  const ids: number[] = Array.isArray(body.ids) ? body.ids : [];
  const action: string = body.action;
  if (ids.length === 0) {
    return NextResponse.json({ error: '선택된 항목이 없습니다.' }, { status: 400 });
  }

  const admin = createServiceClient();
  let error;

  if (action === 'resolve') {
    ({ error } = await admin
      .from('queries')
      .update({ resolved_at: new Date().toISOString() })
      .in('id', ids));
  } else if (action === 'unresolve') {
    ({ error } = await admin.from('queries').update({ resolved_at: null }).in('id', ids));
  } else if (action === 'delete') {
    ({ error } = await admin.from('queries').delete().in('id', ids));
  } else {
    return NextResponse.json({ error: '알 수 없는 작업' }, { status: 400 });
  }

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true, count: ids.length });
}
