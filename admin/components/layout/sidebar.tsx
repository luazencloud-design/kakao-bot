'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  Folder,
  LogOut,
  MessageCircle,
  MessageSquareText,
  Settings,
  ThumbsUp,
} from 'lucide-react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';

interface Props {
  email?: string;
}

const NAV = [
  { href: '/files', label: '자료 관리', icon: Folder },
  { href: '/test', label: '답변 테스트', icon: MessageSquareText },
  { href: '/stats', label: '통계', icon: BarChart3 },
  { href: '/feedback', label: '사용자 피드백', icon: ThumbsUp },
  { href: '/settings', label: '설정', icon: Settings },
];

export function Sidebar({ email }: Props) {
  const pathname = usePathname();

  async function handleLogout() {
    const supabase = createSupabaseBrowserClient();
    await supabase.auth.signOut();
    window.location.href = '/login';
  }

  const initial = email?.[0]?.toUpperCase() ?? '운';

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex-col hidden md:flex">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center">
            <MessageCircle className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-semibold text-slate-900 text-sm">카카오봇</div>
            <div className="text-xs text-slate-500">관리자</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {NAV.map((item) => {
          const active = pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? 'bg-blue-50 text-blue-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-100">
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-50 text-left"
        >
          <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-xs font-medium text-slate-700">
            {initial}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-slate-900 truncate">
              운영자
            </div>
            <div className="text-xs text-slate-500 truncate">
              {email ?? '—'}
            </div>
          </div>
          <LogOut className="w-4 h-4 text-slate-400" />
        </button>
      </div>
    </aside>
  );
}
