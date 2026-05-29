// API 라우트·서버 액션에서 호출하는 어드민 인증 가드.
// 인증 + 화이트리스트 둘 다 통과해야 user 반환, 아니면 null.

import { createSupabaseServerClient, createServiceClient } from '@/lib/supabase/server';

export async function requireAdmin() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user?.email) return null;

  const admin = createServiceClient();
  const { data: allowed } = await admin
    .from('allowed_admins')
    .select('email')
    .eq('email', user.email)
    .maybeSingle();

  if (!allowed) return null;
  return user;
}
