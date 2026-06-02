// Client Component (브라우저)에서 사용하는 Supabase 클라이언트.
// anon 키 + 쿠키 세션. 매직링크 로그인 트리거 등에 사용.

import { createBrowserClient } from '@supabase/ssr';

export function createSupabaseBrowserClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}
