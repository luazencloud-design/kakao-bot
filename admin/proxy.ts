// Next.js 16 Proxy (formerly middleware)
// 1. Supabase 세션 쿠키 갱신
// 2. /(dashboard)/* 경로 보호 (미인증 시 /login 으로)
// 3. /login 접근 시 이미 로그인이면 /files 로

import { createServerClient } from '@supabase/ssr';
import { NextResponse, type NextRequest } from 'next/server';

export async function proxy(request: NextRequest) {
  let response = NextResponse.next({ request });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value),
          );
          response = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) =>
            response.cookies.set(name, value, options),
          );
        },
      },
    },
  );

  const {
    data: { user },
  } = await supabase.auth.getUser();

  const path = request.nextUrl.pathname;

  // 보호 경로: 대시보드 영역 + admin API
  const isProtected =
    path.startsWith('/files') ||
    path.startsWith('/test') ||
    path.startsWith('/stats') ||
    path.startsWith('/feedback') ||
    path.startsWith('/admin/api');

  if (isProtected && !user) {
    const url = request.nextUrl.clone();
    url.pathname = '/login';
    return NextResponse.redirect(url);
  }

  // 이미 로그인된 채로 /login 가면 /files 로
  if (path === '/login' && user) {
    const url = request.nextUrl.clone();
    url.pathname = '/files';
    return NextResponse.redirect(url);
  }

  return response;
}

export const config = {
  matcher: [
    /*
     * 다음을 제외한 모든 경로:
     * - _next/static, _next/image (정적 자산)
     * - favicon
     * - public 파일
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)',
  ],
};
