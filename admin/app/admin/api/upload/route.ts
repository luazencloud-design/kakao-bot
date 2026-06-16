// POST /admin/api/upload  (JSON: { storagePath, filename, sha256, size })
// 서명 URL로 Storage에 "직접 업로드"가 끝난 파일을 처리하는 단계.
//   1. documents row 생성 (pending)
//   2. processDocument 실행 — buffer 없이 호출하면 storage_path에서 다운로드해
//      추출→청킹→임베딩→ready 까지 진행 (NDJSON 진행 이벤트 스트리밍)
//
// 파일 자체는 이 요청 본문에 없으므로 Vercel 4.5MB 한계를 받지 않는다.
// (서버가 Supabase에서 내려받는 방향엔 그 한계가 없음.)

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';
import { processDocument, inferCategory } from '@/lib/ingest/process';
import { STORAGE_BUCKET, isSupportedFile, mimeFromExt } from '@/lib/upload-meta';

// Hobby 기본 10초(문서는 배치 임베딩으로 보통 통과), Pro 60~300초.
export const maxDuration = 300;
export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const body = await request.json().catch(() => null);
  const storagePath = typeof body?.storagePath === 'string' ? body.storagePath : '';
  const filename = typeof body?.filename === 'string' ? body.filename : '';
  const sha256 = typeof body?.sha256 === 'string' ? body.sha256 : null;
  const size = typeof body?.size === 'number' ? body.size : null;

  if (!storagePath || !filename) {
    return NextResponse.json({ error: '잘못된 요청입니다.' }, { status: 400 });
  }
  if (!isSupportedFile(filename)) {
    return NextResponse.json({ error: '지원하지 않는 형식입니다.' }, { status: 400 });
  }

  const admin = createServiceClient();

  // 중복 재확인 (서명~처리 사이 경합 방지)
  if (sha256) {
    const { data: dup } = await admin
      .from('documents')
      .select('id, filename')
      .eq('sha256', sha256)
      .maybeSingle();
    if (dup) {
      await admin.storage.from(STORAGE_BUCKET).remove([storagePath]);
      return NextResponse.json(
        { error: `이미 동일한 파일이 있습니다: ${dup.filename}` },
        { status: 409 },
      );
    }
  }

  // documents row 생성
  const { data: doc, error: insErr } = await admin
    .from('documents')
    .insert({
      filename,
      mime_type: mimeFromExt(filename),
      storage_path: storagePath,
      size_bytes: size,
      sha256,
      category: inferCategory(filename),
      status: 'pending',
      chunk_count: 0,
    })
    .select('id')
    .single();
  if (insErr || !doc) {
    await admin.storage.from(STORAGE_BUCKET).remove([storagePath]);
    return NextResponse.json(
      { error: `문서 생성 실패: ${insErr?.message}` },
      { status: 500 },
    );
  }

  // 처리 — NDJSON 스트리밍 (단계별 진행 이벤트). buffer 미전달 →
  // processDocument가 storage_path에서 직접 다운로드해 추출.
  const docId = doc.id;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj: unknown) =>
        controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
      try {
        const { chunkCount } = await processDocument(docId, {
          onProgress: (e) => send(e),
        });
        send({ stage: 'complete', ok: true, id: docId, chunkCount });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        send({ stage: 'complete', ok: false, id: docId, error: message });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
    },
  });
}
