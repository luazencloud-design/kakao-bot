'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

export function UploadZone() {
  const router = useRouter();
  const [uploading, setUploading] = useState<string | null>(null);

  const onDrop = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        setUploading(file.name);
        const form = new FormData();
        form.append('file', file);

        try {
          const res = await fetch('/admin/api/upload', {
            method: 'POST',
            body: form,
          });
          const data = await res.json();
          if (res.ok && data.ok) {
            toast.success(`${file.name}: ${data.chunkCount}개 청크 생성 완료`);
          } else {
            toast.error(`${file.name}: ${data.error ?? '업로드 실패'}`);
          }
        } catch {
          toast.error(`${file.name}: 네트워크 오류`);
        }
      }
      setUploading(null);
      router.refresh();
    },
    [router],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: !!uploading,
  });

  return (
    <div
      {...getRootProps()}
      className={`bg-white rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition ${
        isDragActive
          ? 'border-blue-500 bg-blue-50/50'
          : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50/30'
      } ${uploading ? 'opacity-60 pointer-events-none' : ''}`}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <>
          <Loader2 className="w-10 h-10 text-blue-500 mx-auto mb-3 animate-spin" />
          <div className="text-slate-900 font-medium mb-1">
            처리 중: {uploading}
          </div>
          <div className="text-sm text-slate-500">
            추출 → 청킹 → 임베딩 (최대 몇 분 소요)
          </div>
        </>
      ) : (
        <>
          <UploadCloud className="w-10 h-10 text-slate-400 mx-auto mb-3" />
          <div className="text-slate-900 font-medium mb-1">
            {isDragActive ? '여기에 놓으세요' : '파일을 끌어다 놓으세요'}
          </div>
          <div className="text-sm text-slate-500 mb-4">또는 클릭해서 선택</div>
          <div className="text-xs text-slate-400">
            지원: PDF · TXT · VTT (자막) · 최대 50MB
            <br />
            PPTX · HWP · 영상/오디오는 추후 지원 예정
          </div>
        </>
      )}
    </div>
  );
}
