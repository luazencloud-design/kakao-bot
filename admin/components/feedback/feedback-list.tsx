'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ThumbsDown,
  Lightbulb,
  Upload,
  Check,
  Trash2,
  AlertTriangle,
  RotateCcw,
  Loader2,
} from 'lucide-react';
import { toast } from 'sonner';

export interface FeedbackItem {
  id: number;
  utterance: string;
  answer: string | null;
  sources: string[] | null;
  feedback: number | null;
  feedback_comment: string | null;
  llm_provider: string | null;
  resolved_at: string | null;
  created_at: string;
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const h = Math.floor(diff / 3600000);
  if (h < 1) return '방금';
  if (h < 24) return `${h}시간 전`;
  return `${Math.floor(h / 24)}일 전`;
}

type Tab = 'unresolved' | 'resolved';

export function FeedbackList({ items: initial }: { items: FeedbackItem[] }) {
  const router = useRouter();
  const [items, setItems] = useState(initial);
  const [tab, setTab] = useState<Tab>('unresolved');
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);

  const filtered = items.filter((i) =>
    tab === 'unresolved' ? !i.resolved_at : !!i.resolved_at,
  );
  const unresolvedCount = items.filter((i) => !i.resolved_at).length;
  const resolvedCount = items.filter((i) => !!i.resolved_at).length;

  function toggle(id: number) {
    setSelected((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  }

  function toggleAll() {
    if (selected.size === filtered.length) setSelected(new Set());
    else setSelected(new Set(filtered.map((i) => i.id)));
  }

  async function bulk(action: 'resolve' | 'unresolve' | 'delete') {
    const ids = [...selected];
    if (ids.length === 0) return;
    if (action === 'delete' && !confirm(`${ids.length}개 이슈를 삭제할까요?`)) return;
    setBusy(true);
    const res = await fetch('/admin/api/queries/bulk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids, action }),
    });
    setBusy(false);
    if (!res.ok) {
      toast.error('작업 실패');
      return;
    }
    // 낙관적 반영
    setItems((list) => {
      if (action === 'delete') return list.filter((i) => !selected.has(i.id));
      return list.map((i) =>
        selected.has(i.id)
          ? { ...i, resolved_at: action === 'resolve' ? new Date().toISOString() : null }
          : i,
      );
    });
    setSelected(new Set());
    toast.success(
      action === 'delete'
        ? `${ids.length}개 삭제됨`
        : action === 'resolve'
          ? `${ids.length}개 해결됨`
          : `${ids.length}개 미해결로 변경`,
    );
    router.refresh();
  }

  async function single(id: number, action: 'resolve' | 'unresolve' | 'delete') {
    if (action === 'delete' && !confirm('이 이슈를 삭제할까요?')) return;
    const url = `/admin/api/queries/${id}`;
    const res =
      action === 'delete'
        ? await fetch(url, { method: 'DELETE' })
        : await fetch(url, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resolved: action === 'resolve' }),
          });
    if (!res.ok) {
      toast.error('작업 실패');
      return;
    }
    setItems((list) => {
      if (action === 'delete') return list.filter((i) => i.id !== id);
      return list.map((i) =>
        i.id === id
          ? { ...i, resolved_at: action === 'resolve' ? new Date().toISOString() : null }
          : i,
      );
    });
    setSelected((s) => {
      const n = new Set(s);
      n.delete(id);
      return n;
    });
    router.refresh();
  }

  return (
    <div className="space-y-4">
      {/* 탭 */}
      <div className="flex items-center gap-2 border-b border-slate-200">
        <button
          onClick={() => {
            setTab('unresolved');
            setSelected(new Set());
          }}
          className={`px-4 py-2 text-sm -mb-px border-b-2 ${
            tab === 'unresolved'
              ? 'border-blue-500 text-slate-900 font-medium'
              : 'border-transparent text-slate-500 hover:text-slate-900'
          }`}
        >
          해결 안됨{' '}
          <span className="ml-1 px-1.5 py-0.5 bg-red-100 text-red-700 rounded-full text-xs">
            {unresolvedCount}
          </span>
        </button>
        <button
          onClick={() => {
            setTab('resolved');
            setSelected(new Set());
          }}
          className={`px-4 py-2 text-sm -mb-px border-b-2 ${
            tab === 'resolved'
              ? 'border-blue-500 text-slate-900 font-medium'
              : 'border-transparent text-slate-500 hover:text-slate-900'
          }`}
        >
          해결됨 <span className="ml-1 text-xs text-slate-400">{resolvedCount}</span>
        </button>
      </div>

      {/* 일괄 작업 바 */}
      {filtered.length > 0 && (
        <div className="flex items-center gap-3 bg-white rounded-lg border border-slate-200 px-4 py-2">
          <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={selected.size === filtered.length && filtered.length > 0}
              onChange={toggleAll}
              className="rounded border-slate-300"
            />
            전체 선택
          </label>
          <span className="text-xs text-slate-400">{selected.size}개 선택</span>
          <div className="flex-1" />
          {selected.size > 0 && (
            <div className="flex items-center gap-2">
              {tab === 'unresolved' ? (
                <button
                  onClick={() => bulk('resolve')}
                  disabled={busy}
                  className="text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 disabled:opacity-50"
                >
                  {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                  선택 해결
                </button>
              ) : (
                <button
                  onClick={() => bulk('unresolve')}
                  disabled={busy}
                  className="text-xs bg-slate-600 hover:bg-slate-700 text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 disabled:opacity-50"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  미해결로
                </button>
              )}
              <button
                onClick={() => bulk('delete')}
                disabled={busy}
                className="text-xs bg-red-50 hover:bg-red-100 text-red-700 px-3 py-1.5 rounded-lg flex items-center gap-1.5 disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5" />
                선택 삭제
              </button>
            </div>
          )}
        </div>
      )}

      {/* 목록 */}
      {filtered.length === 0 ? (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
          <div className="text-sm text-slate-500">
            {tab === 'unresolved' ? '미해결 이슈가 없습니다 👍' : '해결된 이슈가 없습니다'}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((q) => {
            const isError = q.llm_provider === 'error';
            const isUnanswered = !isError && (!q.sources || q.sources.length === 0);
            const isResolved = !!q.resolved_at;
            return (
              <div
                key={q.id}
                className={`bg-white rounded-xl border overflow-hidden ${
                  selected.has(q.id) ? 'border-blue-400 ring-1 ring-blue-200' : 'border-slate-200'
                }`}
              >
                <div className="p-5">
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selected.has(q.id)}
                      onChange={() => toggle(q.id)}
                      className="mt-1 rounded border-slate-300 shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        {isError ? (
                          <Badge color="red" icon={<AlertTriangle className="w-3 h-3" />}>오류</Badge>
                        ) : q.feedback === -1 ? (
                          <Badge color="red" icon={<ThumbsDown className="w-3 h-3" />}>개선 필요</Badge>
                        ) : (
                          <Badge color="amber" icon={<Lightbulb className="w-3 h-3" />}>자료 없음</Badge>
                        )}
                        {isResolved && (
                          <Badge color="emerald" icon={<Check className="w-3 h-3" />}>해결됨</Badge>
                        )}
                        <span className="text-xs text-slate-500">{timeAgo(q.created_at)}</span>
                      </div>

                      <div className="space-y-2">
                        <div>
                          <div className="text-xs text-slate-500 mb-0.5">질문</div>
                          <div className="text-sm text-slate-900">{q.utterance}</div>
                        </div>
                        {q.answer && (
                          <div>
                            <div className="text-xs text-slate-500 mb-0.5">
                              {isError ? '오류 내용' : '챗봇 답변'}
                            </div>
                            <div
                              className={`text-sm rounded-lg p-2.5 leading-relaxed line-clamp-3 ${
                                isError ? 'text-red-700 bg-red-50' : 'text-slate-700 bg-slate-50'
                              }`}
                            >
                              {q.answer}
                            </div>
                          </div>
                        )}
                        {q.feedback_comment && (
                          <div className="text-sm text-slate-700 italic">
                            &ldquo;{q.feedback_comment}&rdquo;
                          </div>
                        )}
                      </div>

                      {/* 액션 */}
                      <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-end gap-2">
                        {isUnanswered && (
                          <Link
                            href="/files"
                            className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5"
                          >
                            <Upload className="w-3.5 h-3.5" />
                            자료 추가
                          </Link>
                        )}
                        {isResolved ? (
                          <button
                            onClick={() => single(q.id, 'unresolve')}
                            className="text-xs text-slate-600 hover:bg-slate-100 px-3 py-1.5 rounded-lg flex items-center gap-1.5"
                          >
                            <RotateCcw className="w-3.5 h-3.5" />
                            미해결로
                          </button>
                        ) : (
                          <button
                            onClick={() => single(q.id, 'resolve')}
                            className="text-xs bg-emerald-50 hover:bg-emerald-100 text-emerald-700 px-3 py-1.5 rounded-lg flex items-center gap-1.5"
                          >
                            <Check className="w-3.5 h-3.5" />
                            해결
                          </button>
                        )}
                        <button
                          onClick={() => single(q.id, 'delete')}
                          className="text-xs text-slate-400 hover:bg-slate-100 hover:text-red-600 px-2 py-1.5 rounded-lg"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Badge({
  color,
  icon,
  children,
}: {
  color: 'red' | 'amber' | 'emerald';
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  const cls = {
    red: 'bg-red-50 text-red-700',
    amber: 'bg-amber-50 text-amber-700',
    emerald: 'bg-emerald-50 text-emerald-700',
  }[color];
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-full ${cls}`}>
      {icon}
      {children}
    </span>
  );
}
