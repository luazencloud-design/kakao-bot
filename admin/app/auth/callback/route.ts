// /auth/callback?code=...&next=/files
// Supabase 매직링크 클릭 후 도착하는 곳.
// code를 세션으로 교환하고 next로 리다이렉트.

import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient, createServiceClient } from '@/lib/supabase/server';

export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const code = url.searchParams.get('code');
  const next = url.searchParams.get('next') ?? '/files';

  if (!code) {
    return NextResponse.redirect(new URL('/login?error=missing_code', request.url));
  }

  const supabase = await createSupabaseServerClient();
  const { error } = await supabase.auth.exchangeCodeForSession(code);

  if (error) {
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(error.message)}`, request.url),
    );
  }

  // 어드민 화이트리스트 확인 (service_role로 RLS 우회 — allowed_admins엔 SELECT 정책 없음)
  const { data: { user } } = await supabase.auth.getUser();
  if (user?.email) {
    const admin = createServiceClient();
    const { data: allowed } = await admin
      .from('allowed_admins')
      .select('email')
      .eq('email', user.email)
      .maybeSingle();

    if (!allowed) {
      // 화이트리스트에 없으면 로그아웃 + 에러
      await supabase.auth.signOut();
      return NextResponse.redirect(
        new URL('/login?error=not_authorized', request.url),
      );
    }
  }

  return NextResponse.redirect(new URL(next, request.url));
}
