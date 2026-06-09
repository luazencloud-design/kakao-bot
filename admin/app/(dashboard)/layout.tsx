import { redirect } from 'next/navigation';
import { Sidebar } from '@/components/layout/sidebar';
import {
  createSupabaseServerClient,
  createServiceClient,
} from '@/lib/supabase/server';

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // 미인증 → 로그인 (proxy가 1차로 막지만 안전망)
  if (!user) {
    redirect('/login');
  }

  // 화이트리스트 검증 (service_role로 RLS 우회)
  const admin = createServiceClient();
  const { data: allowed } = await admin
    .from('allowed_admins')
    .select('email')
    .eq('email', user.email!)
    .maybeSingle();

  if (!allowed) {
    await supabase.auth.signOut();
    redirect('/login?error=not_authorized');
  }

  return (
    // 전체를 뷰포트 높이로 고정. 페이지 자체는 스크롤 안 하고
    // 메인 영역만 내부 스크롤 → 사이드바는 항상 제자리(콘텐츠 길이 무관).
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar email={user.email} />
      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}
