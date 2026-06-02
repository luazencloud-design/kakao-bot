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

export function FileList({ documents }: { documents: DocumentRow[] }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);

  async function handleDelete(doc: DocumentRow) {
    if (!confirm(`"${doc.filename}" 자료를 삭제할까요?\n관련 청크 ${doc.chunk_count}개가 함께 삭제됩니다.`))
      return;
    setBusy(doc.id);
    const res = await fetch(`/admin/api/documents/${doc.id}`, { method: 'DELETE' });
    setBusy(null);
    if (!res.ok) {
      toast.error('삭제 실패');
      return;
    }
    toast.success(`${doc.filename} 삭제됨`);
    router.refresh();
  }

  async function handleReprocess(doc: DocumentRow) {
    setBusy(doc.id);
    const res = await fetch(`/admin/api/documents/${doc.id}/reingest`, { method: 'POST' });
    setBusy(null);
    if (!res.ok) {
      toast.error('재처리 시작 실패');
      return;
    }
    toast.success(`${doc.filename} 재처리 시작`);
    router.refresh();
  }

  if (documents.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center text-sm text-slate-500">
        아직 업로드된 자료가 없습니다.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 text-xs font-medium text-slate-500 grid grid-cols-12 gap-3">
        <div className="col-span-6 md:col-span-5">파일</div>
        <div className="col-span-3 md:col-span-2 hidden md:block">카테고리</div>
        <div className="col-span-2 hidden md:block">청크</div>
        <div className="col-span-3 md:col-span-2">상태</div>
        <div className="col-span-3 md:col-span-1 text-right">작업</div>
      </div>

      {documents.map((doc) => (
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
            {doc.category ? (
              <span
                className={`text-xs px-2 py-1 rounded-full ${
                  CATEGORY_COLORS[doc.category] ?? 'bg-slate-100 text-slate-600'
                }`}
              >
                {CATEGORY_LABELS[doc.category] ?? doc.category}
              </span>
            ) : (
              <span className="text-xs text-slate-400">—</span>
            )}
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
      ))}
    </div>
  );
}
