// Server Component·Route Handler·Server Action에서 사용하는 Supabase 클라이언트.
// 쿠키에서 세션을 읽어 인증된 요청을 보냄.
//
// 일반 작업: anon 키 + 쿠키 세션 사용 (RLS 적용)
// 어드민 작업: createServiceClient() 사용 (RLS bypass)

import { createServerClient } from '@supabase/ssr';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';

export async function createSupabaseServerClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          } catch {
            // Server Component에서 호출되면 cookies()는 readonly.
            // middleware에서 갱신되므로 무시.
          }
        },
      },
    },
  );
}

// service_role 키로 직접 동작. RLS bypass. 파일 업로드·청크 upsert 등 어드민 작업용.
// 반드시 권한 검증을 통과한 라우트에서만 사용.
export function createServiceClient() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      auth: { persistSession: false },
    },
  );
}
