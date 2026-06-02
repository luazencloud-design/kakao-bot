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
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

type Status = 'staged' | 'uploading' | 'done' | 'failed';
type Stage = 'stored' | 'extract' | 'chunk' | 'embed' | 'save' | 'done';

interface Staged {
  file: File;
  status: Status;
  stage?: Stage;
  embedCurrent?: number;
  embedTotal?: number;
  chunkCount?: number;
  message?: string;
}

const STAGE_LABEL: Record<Stage, string> = {
  stored: '파일 저장',
  extract: '텍스트 추출',
  chunk: '문서 분할',
  embed: '임베딩 생성',
  save: '저장',
  done: '완료',
};
const STAGE_ORDER: Stage[] = ['stored', 'extract', 'chunk', 'embed', 'save'];

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

  function patch(idx: number, p: Partial<Staged>) {
    setStaged((prev) => prev.map((s, j) => (j === idx ? { ...s, ...p } : s)));
  }

  async function uploadOne(idx: number) {
    const target = staged[idx];
    patch(idx, { status: 'uploading', stage: 'stored' });

    const form = new FormData();
    form.append('file', target.file);

    try {
      const res = await fetch('/admin/api/upload', { method: 'POST', body: form });

      // 검증 단계 에러(409/400 등)는 JSON
      if (!res.body || res.headers.get('content-type')?.includes('application/json')) {
        const data = await res.json().catch(() => ({}));
        patch(idx, { status: 'failed', message: data.error ?? '업로드 실패' });
        return;
      }

      // NDJSON 스트림 읽기
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.stage === 'complete') {
            if (ev.ok) patch(idx, { status: 'done', stage: 'done', chunkCount: ev.chunkCount });
            else patch(idx, { status: 'failed', message: ev.error });
          } else if (ev.stage === 'embed') {
            patch(idx, { stage: 'embed', embedCurrent: ev.current, embedTotal: ev.total });
          } else if (ev.stage === 'chunk') {
            patch(idx, { stage: 'chunk', embedTotal: ev.total });
          } else {
            patch(idx, { stage: ev.stage });
          }
        }
      }
    } catch {
      patch(idx, { status: 'failed', message: '네트워크 오류' });
    }
  }

  async function uploadAll() {
    setBusy(true);
    for (let i = 0; i < staged.length; i++) {
      if (staged[i].status === 'staged') await uploadOne(i);
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

          <div className="divide-y divide-slate-100 max-h-96 overflow-y-auto">
            {staged.map((s, i) => (
              <FileRow key={`${s.file.name}-${i}`} s={s} onRemove={() => removeAt(i)} canRemove={!busy} />
            ))}
          </div>

          <div className="px-4 py-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-500">
              {busy ? '처리 중... (순차 진행)' : `${stagedCount}개 파일이 대기 중입니다`}
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
              {busy ? '처리 중' : `${stagedCount}개 업로드`}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function FileRow({ s, onRemove, canRemove }: { s: Staged; onRemove: () => void; canRemove: boolean }) {
  return (
    <div className="px-4 py-3">
      <div className="flex items-center gap-3">
        <RowIcon s={s} />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-slate-900 truncate">{s.file.name}</div>
          <div className="text-xs text-slate-400">
            {fmt(s.file.size)}
            {s.status === 'done' && s.chunkCount != null && (
              <span className="text-emerald-600"> · {s.chunkCount}개 청크 생성</span>
            )}
            {s.status === 'failed' && s.message && (
              <span className="text-red-600"> · {s.message}</span>
            )}
          </div>
        </div>
        {(s.status === 'staged' || s.status === 'failed') && canRemove && (
          <button onClick={onRemove} className="p-1 hover:bg-slate-100 rounded text-slate-400">
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* 진행 단계 표시 (업로드 중) */}
      {s.status === 'uploading' && (
        <div className="mt-3 pl-8">
          <div className="flex items-center gap-1.5">
            {STAGE_ORDER.map((stage, idx) => {
              const curIdx = s.stage ? STAGE_ORDER.indexOf(s.stage) : 0;
              const active = s.stage === stage;
              const passed = curIdx > idx;
              return (
                <div key={stage} className="flex items-center gap-1.5 flex-1">
                  <div className="flex flex-col items-center gap-1 flex-1">
                    <div
                      className={`w-full h-1 rounded-full transition-colors ${
                        passed ? 'bg-emerald-500' : active ? 'bg-blue-500' : 'bg-slate-200'
                      }`}
                    />
                    <span
                      className={`text-[10px] whitespace-nowrap ${
                        active ? 'text-blue-600 font-medium' : passed ? 'text-emerald-600' : 'text-slate-400'
                      }`}
                    >
                      {STAGE_LABEL[stage]}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          {/* 임베딩 진행률 */}
          {s.stage === 'embed' && s.embedTotal != null && (
            <div className="mt-2">
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>임베딩 생성 중</span>
                <span>
                  {s.embedCurrent ?? 0} / {s.embedTotal}
                </span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{ width: `${((s.embedCurrent ?? 0) / s.embedTotal) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RowIcon({ s }: { s: Staged }) {
  if (s.status === 'done') return <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />;
  if (s.status === 'failed') return <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />;
  if (s.status === 'uploading')
    return <Loader2 className="w-5 h-5 text-blue-500 shrink-0 animate-spin" />;
  return <FileText className="w-5 h-5 text-slate-400 shrink-0" />;
}
