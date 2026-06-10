// 한 문서를 처리: 추출 → 청킹 → 임베딩 → chunks upsert → status 갱신.
// 업로드/재처리 라우트에서 호출.

import { createServiceClient } from '@/lib/supabase/server';
import { extractText } from './extract';
import { chunkText } from './chunk';
import { embedDocumentsBatch } from './embed';
import { inferCategory } from './category';

export { inferCategory }; // 기존 import 경로 호환

const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';

export type ProgressEvent =
  | { stage: 'extract' }
  | { stage: 'chunk'; total: number }
  | { stage: 'embed'; current: number; total: number }
  | { stage: 'save' }
  | { stage: 'done'; chunkCount: number };

// documentId가 가리키는 문서를 처리. buffer가 주어지면 추출부터, 없으면 캐시된 extracted_text 사용.
// onProgress: 단계별 진행 상황 콜백 (스트리밍 UI용).
export async function processDocument(
  documentId: string,
  opts?: { buffer?: Buffer; filename?: string; onProgress?: (e: ProgressEvent) => void },
): Promise<{ chunkCount: number }> {
  const admin = createServiceClient();
  const emit = opts?.onProgress ?? (() => {});

  // 문서 조회
  const { data: doc, error: docErr } = await admin
    .from('documents')
    .select('id, filename, extracted_text, storage_path')
    .eq('id', documentId)
    .single();
  if (docErr || !doc) throw new Error('문서를 찾을 수 없습니다.');

  await admin
    .from('documents')
    .update({ status: 'processing', error_message: null })
    .eq('id', documentId);

  try {
    // 1. 텍스트 확보
    emit({ stage: 'extract' });
    let text = doc.extracted_text ?? '';
    if (opts?.buffer) {
      const { text: extracted } = await extractText(
        opts.filename ?? doc.filename,
        opts.buffer,
      );
      text = extracted;
      await admin.from('documents').update({ extracted_text: text }).eq('id', documentId);
    } else if (!text) {
      // 캐시도 없고 buffer도 없으면 Storage에서 다운로드 시도
      if (doc.storage_path && !doc.storage_path.startsWith('cli-ingest/') && !doc.storage_path.startsWith('migrated/')) {
        const { data: blob, error: dlErr } = await admin.storage
          .from('source-files')
          .download(doc.storage_path);
        if (dlErr || !blob) throw new Error('원본 파일을 가져올 수 없습니다.');
        const buf = Buffer.from(await blob.arrayBuffer());
        const { text: extracted } = await extractText(doc.filename, buf);
        text = extracted;
        await admin.from('documents').update({ extracted_text: text }).eq('id', documentId);
      } else {
        throw new Error('추출할 원본 텍스트가 없습니다.');
      }
    }

    // 2. 청킹
    const chunks = chunkText(text);
    if (chunks.length === 0) throw new Error('청크를 생성하지 못했습니다 (빈 텍스트).');
    emit({ stage: 'chunk', total: chunks.length });

    // 3. 기존 청크 삭제 (재처리 대비)
    await admin.from('chunks').delete().eq('document_id', documentId);

    // 4. 배치 임베딩 (100개씩, 진행 이벤트는 배치 단위)
    const BATCH = 100;
    const embeddings: number[][] = [];
    for (let start = 0; start < chunks.length; start += BATCH) {
      const slice = chunks.slice(start, start + BATCH);
      emit({ stage: 'embed', current: Math.min(start + slice.length, chunks.length), total: chunks.length });
      const vecs = await embedDocumentsBatch(slice);
      embeddings.push(...vecs);
    }
    if (embeddings.length !== chunks.length) {
      throw new Error(`임베딩 개수 불일치 (${embeddings.length}/${chunks.length})`);
    }

    const rows = chunks.map((text, i) => ({
      document_id: documentId,
      chunk_index: i,
      text,
      embedding: embeddings[i],
      embed_model: EMBED_MODEL,
      embed_dim: 768,
      metadata: {},
    }));

    emit({ stage: 'save' });
    for (let i = 0; i < rows.length; i += BATCH) {
      const { error } = await admin.from('chunks').insert(rows.slice(i, i + BATCH));
      if (error) throw new Error(`청크 저장 실패: ${error.message}`);
    }

    // 5. 완료
    await admin
      .from('documents')
      .update({ status: 'ready', chunk_count: chunks.length, error_message: null })
      .eq('id', documentId);

    emit({ stage: 'done', chunkCount: chunks.length });
    return { chunkCount: chunks.length };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await admin
      .from('documents')
      .update({ status: 'failed', error_message: message })
      .eq('id', documentId);
    throw err;
  }
}
