import { createServiceClient } from '@/lib/supabase/server';
import { FileList } from '@/components/files/file-list';
import { UploadZone } from '@/components/files/upload-zone';
import type { DocumentRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export default async function FilesPage() {
  const admin = createServiceClient();

  const { data: docs } = await admin
    .from('documents')
    .select(
      'id, filename, mime_type, storage_path, size_bytes, category, status, error_message, chunk_count, created_at, updated_at',
    )
    .order('created_at', { ascending: false });

  const documents = (docs ?? []) as DocumentRow[];
  const totalChunks = documents.reduce((sum, d) => sum + (d.chunk_count ?? 0), 0);
  const processing = documents.filter((d) => d.status === 'processing').length;
  const failed = documents.filter((d) => d.status === 'failed').length;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <header>
        <h1 className="text-xl font-bold text-slate-900">자료 관리</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          챗봇이 사용하는 문서를 추가·삭제하고 처리 상태를 확인합니다
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-4 border border-slate-200">
          <div className="text-xs text-slate-500">총 자료</div>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {documents.length}
          </div>
          <div className="text-xs text-slate-400 mt-1">파일</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-slate-200">
          <div className="text-xs text-slate-500">검색 가능 청크</div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{totalChunks}</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-slate-200">
          <div className="text-xs text-slate-500">처리 중</div>
          <div className="text-2xl font-bold text-amber-600 mt-1">{processing}</div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-slate-200">
          <div className="text-xs text-slate-500">실패</div>
          <div
            className={`text-2xl font-bold mt-1 ${failed > 0 ? 'text-red-600' : 'text-slate-900'}`}
          >
            {failed}
          </div>
        </div>
      </div>

      <UploadZone />

      <FileList documents={documents} />
    </div>
  );
}
