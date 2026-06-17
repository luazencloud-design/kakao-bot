'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  FileText,
  FileX,
  Video,
  Presentation,
  Captions,
  Music,
  MoreVertical,
  Trash2,
  RefreshCw,
  ChevronDown,
  Search,
  X,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  CATEGORY_COLORS,
  CATEGORY_LABELS,
  formatBytes,
  timeAgo,
  type DocumentRow,
} from '@/lib/types';
import { toast } from 'sonner';

function iconFor(filename: string, status: string) {
  if (status === 'failed') return <FileX className="w-5 h-5 text-red-500 shrink-0" />;
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'mp4') return <Video className="w-5 h-5 text-blue-500 shrink-0" />;
  if (ext === 'mp3' || ext === 'm4a') return <Music className="w-5 h-5 text-pink-500 shrink-0" />;
  if (ext === 'vtt') return <Captions className="w-5 h-5 text-violet-500 shrink-0" />;
  if (ext === 'pptx') return <Presentation className="w-5 h-5 text-orange-500 shrink-0" />;
  return <FileText className="w-5 h-5 text-red-500 shrink-0" />;
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'ready')
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
        준비됨
      </span>
    );
  if (status === 'processing')
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-amber-700">
        <span className="w-3 h-3 rounded-full border-2 border-amber-500 border-t-transparent animate-spin" />
        처리 중
      </span>
    );
  if (status === 'failed')
    return <span className="text-xs text-red-700">실패</span>;
  return <span className="text-xs text-slate-500">대기 중</span>;
}

const CATEGORIES = ['오픈마켓가입', '사업자등록', '강의자료', '도구가이드', '기타'];

const SORT_LABELS: Record<string, string> = {
  newest: '최신순',
  oldest: '오래된순',
  name: '이름순',
  size: '크기 큰순',
  chunks: '청크 많은순',
};

function extOf(filename: string): string {
  return filename.includes('.') ? (filename.split('.').pop()?.toLowerCase() ?? '') : '';
}

export function FileList({ documents }: { documents: DocumentRow[] }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  // 낙관적 UI: 상태/카테고리 즉시 반영, 삭제는 즉시 숨김
  const [overrides, setOverrides] = useState<Record<string, Partial<DocumentRow>>>({});
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  // 검색·형식필터·정렬 (클라이언트)
  const [query, setQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [sort, setSort] = useState<string>('newest');

  function patch(id: string, p: Partial<DocumentRow>) {
    setOverrides((o) => ({ ...o, [id]: { ...o[id], ...p } }));
  }

  // 서버 데이터를 다시 신뢰 — 오버라이드 제거 후 새로고침
  function clearAndRefresh(id: string) {
    setOverrides((o) => {
      const n = { ...o };
      delete n[id];
      return n;
    });
    router.refresh();
  }

  async function handleDelete(doc: DocumentRow) {
    if (!confirm(`"${doc.filename}" 자료를 삭제할까요?\n관련 청크 ${doc.chunk_count}개가 함께 삭제됩니다.`))
      return;
    setRemoved((s) => new Set(s).add(doc.id)); // 즉시 숨김
    const res = await fetch(`/admin/api/documents/${doc.id}`, { method: 'DELETE' });
    if (!res.ok) {
      setRemoved((s) => {
        const n = new Set(s);
        n.delete(doc.id);
        return n;
      });
      toast.error('삭제 실패');
      return;
    }
    toast.success(`${doc.filename} 삭제됨`);
    router.refresh();
  }

  async function handleReprocess(doc: DocumentRow) {
    patch(doc.id, { status: 'processing' }); // 즉시 처리중 표시
    setBusy(doc.id);
    const res = await fetch(`/admin/api/documents/${doc.id}/reingest`, { method: 'POST' });
    setBusy(null);
    if (!res.ok) {
      patch(doc.id, { status: doc.status });
      toast.error('재처리 실패');
      return;
    }
    toast.success(`${doc.filename} 재처리 완료`);
    clearAndRefresh(doc.id); // 오버라이드 제거 → 서버의 'ready' 상태 반영
  }

  async function handleCategory(doc: DocumentRow, category: string) {
    const prev = doc.category;
    patch(doc.id, { category }); // 즉시 반영
    const res = await fetch(`/admin/api/documents/${doc.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category }),
    });
    if (!res.ok) {
      patch(doc.id, { category: prev });
      toast.error('카테고리 변경 실패');
      return;
    }
    toast.success('카테고리 변경됨');
    clearAndRefresh(doc.id);
  }

  // 낙관적 오버라이드 적용 + 삭제된 항목 제외
  const visible = documents
    .filter((d) => !removed.has(d.id))
    .map((d) => ({ ...d, ...overrides[d.id] }));

  // 형식 필터 목록은 현재 파일들에서 추출 (존재하는 형식만 노출)
  const types = Array.from(
    new Set(visible.map((d) => extOf(d.filename)).filter(Boolean)),
  ).sort();

  // 검색 → 형식 필터 → 정렬
  const q = query.trim().toLowerCase();
  const displayed = visible
    .filter((d) => !q || d.filename.toLowerCase().includes(q))
    .filter((d) => typeFilter === 'all' || extOf(d.filename) === typeFilter)
    .sort((a, b) => {
      switch (sort) {
        case 'oldest':
          return (a.created_at ?? '').localeCompare(b.created_at ?? '');
        case 'name':
          return a.filename.localeCompare(b.filename, 'ko');
        case 'size':
          return (b.size_bytes ?? 0) - (a.size_bytes ?? 0);
        case 'chunks':
          return (b.chunk_count ?? 0) - (a.chunk_count ?? 0);
        default: // newest
          return (b.created_at ?? '').localeCompare(a.created_at ?? '');
      }
    });
  const isFiltered = q !== '' || typeFilter !== 'all';

  if (visible.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center text-sm text-slate-500">
        아직 업로드된 자료가 없습니다.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* 검색 · 형식 필터 · 정렬 */}
      <div className="flex flex-col sm:flex-row gap-2">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="파일명 검색..."
            className="w-full pl-9 pr-8 py-2 text-sm rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-200 focus:border-blue-400"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              aria-label="검색어 지우기"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger className="px-3 py-2 text-sm rounded-lg border border-slate-200 inline-flex items-center justify-between gap-1.5 text-slate-700 hover:bg-slate-50 whitespace-nowrap sm:min-w-[6.5rem]">
            형식: {typeFilter === 'all' ? '전체' : typeFilter.toUpperCase()}
            <ChevronDown className="w-3.5 h-3.5 opacity-60" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={() => setTypeFilter('all')}
              className={typeFilter === 'all' ? 'font-semibold text-blue-600' : ''}
            >
              전체
            </DropdownMenuItem>
            {types.map((t) => (
              <DropdownMenuItem
                key={t}
                onClick={() => setTypeFilter(t)}
                className={typeFilter === t ? 'font-semibold text-blue-600' : ''}
              >
                {t.toUpperCase()}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger className="px-3 py-2 text-sm rounded-lg border border-slate-200 inline-flex items-center justify-between gap-1.5 text-slate-700 hover:bg-slate-50 whitespace-nowrap sm:min-w-[7.5rem]">
            {SORT_LABELS[sort]}
            <ChevronDown className="w-3.5 h-3.5 opacity-60" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {Object.entries(SORT_LABELS).map(([k, label]) => (
              <DropdownMenuItem
                key={k}
                onClick={() => setSort(k)}
                className={sort === k ? 'font-semibold text-blue-600' : ''}
              >
                {label}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="px-1 text-xs text-slate-500">
        {displayed.length}개{isFiltered ? ` · 전체 ${visible.length}개 중` : ''}
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 text-xs font-medium text-slate-500 grid grid-cols-12 gap-3">
          <div className="col-span-6 md:col-span-5">파일</div>
          <div className="col-span-3 md:col-span-2 hidden md:block">카테고리</div>
          <div className="col-span-2 hidden md:block">청크</div>
          <div className="col-span-3 md:col-span-2">상태</div>
          <div className="col-span-3 md:col-span-1 text-right">작업</div>
        </div>

        {displayed.length === 0 ? (
          <div className="px-4 py-10 text-center text-sm text-slate-500">
            검색 결과가 없습니다.
          </div>
        ) : (
          displayed.map((doc) => (
        <div
          key={doc.id}
          className={`px-4 py-3 border-b border-slate-100 last:border-0 hover:bg-slate-50 grid grid-cols-12 gap-3 items-center ${
            doc.status === 'failed' ? 'bg-red-50/30' : doc.status === 'processing' ? 'bg-amber-50/30' : ''
          }`}
        >
          <div className="col-span-6 md:col-span-5 flex items-center gap-3 min-w-0">
            {iconFor(doc.filename, doc.status)}
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-900 truncate">
                {doc.filename}
              </div>
              <div className="text-xs text-slate-400 mt-0.5">
                {doc.status === 'failed' && doc.error_message ? (
                  <span className="text-red-600">{doc.error_message}</span>
                ) : (
                  <>
                    {formatBytes(doc.size_bytes)} · {timeAgo(doc.created_at)}
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="col-span-3 md:col-span-2 hidden md:block">
            <DropdownMenu>
              <DropdownMenuTrigger
                className={`text-xs px-2 py-1 rounded-full inline-flex items-center gap-1 hover:ring-1 hover:ring-slate-300 ${
                  doc.category
                    ? CATEGORY_COLORS[doc.category] ?? 'bg-slate-100 text-slate-600'
                    : 'bg-slate-100 text-slate-400'
                }`}
                disabled={busy === doc.id}
              >
                {doc.category ? CATEGORY_LABELS[doc.category] ?? doc.category : '미분류'}
                <ChevronDown className="w-3 h-3 opacity-60" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start">
                {CATEGORIES.map((cat) => (
                  <DropdownMenuItem
                    key={cat}
                    onClick={() => handleCategory(doc, cat)}
                    className={doc.category === cat ? 'font-semibold text-blue-600' : ''}
                  >
                    {CATEGORY_LABELS[cat] ?? cat}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="col-span-2 hidden md:block text-sm text-slate-600">
            {doc.chunk_count || '—'}
          </div>

          <div className="col-span-3 md:col-span-2">
            <StatusBadge status={doc.status} />
          </div>

          <div className="col-span-3 md:col-span-1 text-right">
            <DropdownMenu>
              <DropdownMenuTrigger
                className="p-1.5 hover:bg-slate-100 rounded disabled:opacity-50 inline-flex"
                disabled={busy === doc.id}
              >
                <MoreVertical className="w-4 h-4 text-slate-500" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => handleReprocess(doc)}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  재처리
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleDelete(doc)}
                  className="text-red-600 focus:text-red-600"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  삭제
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
          ))
        )}
      </div>
    </div>
  );
}
