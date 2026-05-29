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

export const maxDuration = 300; // 5분 (Node 런타임)

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
  const storagePath = `uploads/${crypto.randomUUID()}/${filename}`;
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

  // 3. 처리 (동기)
  try {
    const { chunkCount } = await processDocument(doc.id, { buffer, filename });
    return NextResponse.json({ ok: true, id: doc.id, chunkCount });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // 문서는 failed 상태로 남김 (운영자가 재시도 가능)
    return NextResponse.json({ ok: false, id: doc.id, error: message }, { status: 422 });
  }
}
