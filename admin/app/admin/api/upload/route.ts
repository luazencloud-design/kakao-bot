// POST /admin/api/upload  (multipart/form-data, field: "file")
// 1. 파일을 Supabase Storage에 저장
// 2. documents row 생성 (pending)
// 3. processDocument 동기 실행 (추출→청킹→임베딩→ready)
//
// 주의: 현재는 요청 내에서 동기 처리. 대용량/영상은 Task 7에서 큐로 분리 예정.

import { NextRequest, NextResponse } from 'next/server';
import crypto from 'node:crypto';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';
import { processDocument, inferCategory } from '@/lib/ingest/process';

// Vercel: Hobby는 10초로 캡됨(문서는 배치 임베딩으로 보통 통과),
// Pro는 60초(기본)~300초. 미디어 전사하려면 Pro 필요.
export const maxDuration = 300;
export const runtime = 'nodejs';

const MAX_BYTES = 50 * 1024 * 1024; // 50MB (Supabase Storage 무료 한도)

function mimeFromExt(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase();
  return (
    {
      pdf: 'application/pdf',
      txt: 'text/plain',
      vtt: 'text/vtt',
      pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      hwp: 'application/x-hwp',
      mp3: 'audio/mpeg',
      mp4: 'video/mp4',
    }[ext ?? ''] ?? 'application/octet-stream'
  );
}

export async function POST(request: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const form = await request.formData();
  const file = form.get('file');
  if (!(file instanceof File)) {
    return NextResponse.json({ error: '파일이 없습니다.' }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json(
      { error: '파일이 너무 큽니다 (최대 50MB).' },
      { status: 400 },
    );
  }

  const filename = file.name;
  const buffer = Buffer.from(await file.arrayBuffer());
  const sha256 = crypto.createHash('sha256').update(buffer).digest('hex');
  const admin = createServiceClient();

  // 중복 검사 (같은 내용)
  const { data: dup } = await admin
    .from('documents')
    .select('id, filename')
    .eq('sha256', sha256)
    .maybeSingle();
  if (dup) {
    return NextResponse.json(
      { error: `이미 동일한 파일이 있습니다: ${dup.filename}` },
      { status: 409 },
    );
  }

  // 1. Storage 업로드
  // Supabase Storage 키는 ASCII만 허용(한글·공백 불가). 원본 파일명은
  // documents.filename에 보존하고, storage 키는 UUID + 확장자로 안전하게.
  const ext = filename.includes('.') ? filename.split('.').pop()!.toLowerCase() : 'bin';
  const safeExt = ext.replace(/[^a-z0-9]/g, '');
  const storagePath = `uploads/${crypto.randomUUID()}.${safeExt || 'bin'}`;
  const { error: upErr } = await admin.storage
    .from('source-files')
    .upload(storagePath, buffer, {
      contentType: mimeFromExt(filename),
      upsert: false,
    });
  if (upErr) {
    return NextResponse.json(
      { error: `업로드 실패: ${upErr.message}` },
      { status: 500 },
    );
  }

  // 2. documents row 생성
  const { data: doc, error: insErr } = await admin
    .from('documents')
    .insert({
      filename,
      mime_type: mimeFromExt(filename),
      storage_path: storagePath,
      size_bytes: buffer.byteLength,
      sha256,
      category: inferCategory(filename),
      status: 'pending',
      chunk_count: 0,
    })
    .select('id')
    .single();
  if (insErr || !doc) {
    await admin.storage.from('source-files').remove([storagePath]);
    return NextResponse.json(
      { error: `문서 생성 실패: ${insErr?.message}` },
      { status: 500 },
    );
  }

  // 3. 처리 — NDJSON 스트리밍 (단계별 진행 이벤트를 클라이언트로 푸시)
  const docId = doc.id;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj: unknown) =>
        controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));

      send({ stage: 'stored' }); // Storage 업로드·문서 생성 완료
      try {
        const { chunkCount } = await processDocument(docId, {
          buffer,
          filename,
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
