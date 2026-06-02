'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import {
  UploadCloud,
  Loader2,
  X,
  FileText,
  CheckCircle2,
  AlertCircle,
  Clock,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

type Status = 'staged' | 'uploading' | 'done' | 'failed';
interface Staged {
  file: File;
  status: Status;
  message?: string;
  chunkCount?: number;
}

function fmt(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const MAX_BYTES = 50 * 1024 * 1024;

export function UploadZone() {
  const router = useRouter();
  const [staged, setStaged] = useState<Staged[]>([]);
  const [busy, setBusy] = useState(false);

  const onDrop = useCallback((accepted: File[]) => {
    setStaged((prev) => {
      const existing = new Set(prev.map((s) => `${s.file.name}:${s.file.size}`));
      const added = accepted
        .filter((f) => !existing.has(`${f.name}:${f.size}`))
        .map<Staged>((file) => ({
          file,
          status: file.size > MAX_BYTES ? 'failed' : 'staged',
          message: file.size > MAX_BYTES ? '50MB 초과' : undefined,
        }));
      return [...prev, ...added];
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: busy,
    multiple: true,
  });

  function removeAt(idx: number) {
    setStaged((prev) => prev.filter((_, i) => i !== idx));
  }

  async function uploadAll() {
    const targets = staged.filter((s) => s.status === 'staged');
    if (targets.length === 0) return;
    setBusy(true);

    for (let i = 0; i < staged.length; i++) {
      if (staged[i].status !== 'staged') continue;
      setStaged((prev) => prev.map((s, j) => (j === i ? { ...s, status: 'uploading' } : s)));

      const form = new FormData();
      form.append('file', staged[i].file);
      try {
        const res = await fetch('/admin/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok && data.ok) {
          setStaged((prev) =>
            prev.map((s, j) =>
              j === i ? { ...s, status: 'done', chunkCount: data.chunkCount } : s,
            ),
          );
        } else {
          setStaged((prev) =>
            prev.map((s, j) =>
              j === i ? { ...s, status: 'failed', message: data.error ?? '실패' } : s,
            ),
          );
        }
      } catch {
        setStaged((prev) =>
          prev.map((s, j) => (j === i ? { ...s, status: 'failed', message: '네트워크 오류' } : s)),
        );
      }
    }

    setBusy(false);
    setStaged((prev) => {
      const ok = prev.filter((s) => s.status === 'done').length;
      const fail = prev.filter((s) => s.status === 'failed').length;
      if (fail > 0) toast.warning(`완료 ${ok}개 · 실패 ${fail}개`);
      else toast.success(`${ok}개 파일 업로드 완료`);
      return prev;
    });
    router.refresh();
  }

  function clearDone() {
    setStaged((prev) => prev.filter((s) => s.status !== 'done'));
  }

  const stagedCount = staged.filter((s) => s.status === 'staged').length;
  const doneCount = staged.filter((s) => s.status === 'done').length;

  return (
    <div className="space-y-3">
      {/* 드롭존 */}
      <div
        {...getRootProps()}
        className={`bg-white rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition ${
          isDragActive
            ? 'border-blue-500 bg-blue-50/50'
            : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50/30'
        } ${busy ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <UploadCloud className="w-9 h-9 text-slate-400 mx-auto mb-2" />
        <div className="text-slate-900 font-medium text-sm mb-1">
          {isDragActive ? '여기에 놓으세요' : '파일을 끌어다 놓거나 클릭해서 선택'}
        </div>
        <div className="text-xs text-slate-400">
          여러 개 선택 가능 · PDF · TXT · VTT · 최대 50MB
        </div>
      </div>

      {/* 스테이징 목록 */}
      {staged.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-600">
              업로드 대기 {stagedCount}개 · 완료 {doneCount}개
            </span>
            {doneCount > 0 && (
              <button onClick={clearDone} className="text-xs text-slate-500 hover:text-slate-700">
                완료 항목 지우기
              </button>
            )}
          </div>

          <div className="divide-y divide-slate-100 max-h-72 overflow-y-auto">
            {staged.map((s, i) => (
              <div key={`${s.file.name}-${i}`} className="px-4 py-2.5 flex items-center gap-3">
                <StatusIcon status={s.status} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-slate-900 truncate">{s.file.name}</div>
                  <div className="text-xs text-slate-400">
                    {fmt(s.file.size)}
                    {s.status === 'done' && s.chunkCount != null && (
                      <span className="text-emerald-600"> · {s.chunkCount}개 청크</span>
                    )}
                    {s.status === 'failed' && s.message && (
                      <span className="text-red-600"> · {s.message}</span>
                    )}
                  </div>
                </div>
                {(s.status === 'staged' || s.status === 'failed') && !busy && (
                  <button
                    onClick={() => removeAt(i)}
                    className="p-1 hover:bg-slate-100 rounded text-slate-400"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="px-4 py-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-500">
              {busy ? '업로드 중... (순차 처리)' : `${stagedCount}개 파일이 대기 중입니다`}
            </span>
            <Button
              onClick={uploadAll}
              disabled={busy || stagedCount === 0}
              className="bg-blue-600 hover:bg-blue-700"
              size="sm"
            >
              {busy ? (
                <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
              ) : (
                <UploadCloud className="w-4 h-4 mr-1.5" />
              )}
              {busy ? '업로드 중' : `${stagedCount}개 업로드`}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: Status }) {
  if (status === 'done') return <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />;
  if (status === 'failed') return <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />;
  if (status === 'uploading')
    return <Loader2 className="w-5 h-5 text-blue-500 shrink-0 animate-spin" />;
  return <FileText className="w-5 h-5 text-slate-400 shrink-0" />;
}
