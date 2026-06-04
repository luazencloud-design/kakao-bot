'use client';

import { useState } from 'react';
import { KeyRound, Loader2, Check } from 'lucide-react';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';

export default function SettingsPage() {
  const [pw, setPw] = useState('');
  const [pw2, setPw2] = useState('');
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  function strength(p: string): { ok: boolean; msg: string } {
    if (p.length < 10) return { ok: false, msg: '10자 이상 입력하세요' };
    const kinds = [/[a-z]/, /[A-Z]/, /[0-9]/, /[^a-zA-Z0-9]/].filter((re) => re.test(p)).length;
    if (kinds < 3) return { ok: false, msg: '영문 대/소문자·숫자·기호 중 3종류 이상 섞으세요' };
    return { ok: true, msg: '사용 가능한 비밀번호입니다' };
  }

  const s = strength(pw);
  const match = pw.length > 0 && pw === pw2;
  const canSubmit = s.ok && match && !loading;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);

    const supabase = createSupabaseBrowserClient();
    const { error } = await supabase.auth.updateUser({ password: pw });

    setLoading(false);
    if (error) {
      toast.error(`변경 실패: ${error.message}`);
      return;
    }
    setDone(true);
    setPw('');
    setPw2('');
    toast.success('비밀번호가 변경되었습니다');
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <header>
        <h1 className="text-xl font-bold text-slate-900">설정</h1>
        <p className="text-xs text-slate-500 mt-0.5">계정 보안을 관리합니다</p>
      </header>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
          <KeyRound className="w-4 h-4 text-slate-600" />
          <span className="text-sm font-medium text-slate-900">비밀번호 변경</span>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div className="space-y-2">
            <Label htmlFor="pw">새 비밀번호</Label>
            <Input
              id="pw"
              type="password"
              value={pw}
              onChange={(e) => {
                setPw(e.target.value);
                setDone(false);
              }}
              placeholder="••••••••••"
              autoComplete="new-password"
            />
            {pw.length > 0 && (
              <p className={`text-xs ${s.ok ? 'text-emerald-600' : 'text-amber-600'}`}>
                {s.ok ? '✓ ' : ''}
                {s.msg}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="pw2">새 비밀번호 확인</Label>
            <Input
              id="pw2"
              type="password"
              value={pw2}
              onChange={(e) => setPw2(e.target.value)}
              placeholder="••••••••••"
              autoComplete="new-password"
            />
            {pw2.length > 0 && !match && (
              <p className="text-xs text-red-600">비밀번호가 일치하지 않습니다</p>
            )}
          </div>

          <Button
            type="submit"
            disabled={!canSubmit}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
            ) : done ? (
              <Check className="w-4 h-4 mr-1.5" />
            ) : (
              <KeyRound className="w-4 h-4 mr-1.5" />
            )}
            {loading ? '변경 중...' : done ? '변경됨' : '비밀번호 변경'}
          </Button>

          <p className="text-xs text-slate-400 pt-1 leading-relaxed">
            변경 후 다음 로그인부터 새 비밀번호를 사용합니다. 현재 세션은 유지됩니다.
          </p>
        </form>
      </div>
    </div>
  );
}
