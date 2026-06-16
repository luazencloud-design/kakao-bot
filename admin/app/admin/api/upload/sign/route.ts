// POST /admin/api/upload/sign  (JSON: { filename, size, sha256 })
// 큰 파일이 Vercel 함수(요청 본문 4.5MB 한계)를 거치지 않도록,
// Supabase Storage에 "직접 업로드"할 수 있는 1회용 서명 URL을 발급한다.
//
// 클라이언트 흐름:
//   1) 이 라우트로 서명 요청(작은 JSON) → { storagePath, token } 수신
//   2) 브라우저가 storagePath로 파일을 Storage에 직접 업로드 (Vercel 우회)
//   3) /admin/api/upload 로 처리 요청(작은 JSON) → 추출·임베딩
//
// 여기서는 인증·중복·형식·용량만 검증하고, 파일 자체는 받지 않는다(작은 요청).

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth-guard';
import { createServiceClient } from '@/lib/supabase/server';
import { safeStoragePath } from '@/lib/storage';
import {
  MAX_UPLOAD_BYTES,
  STORAGE_BUCKET,
  isSupportedFile,
  mimeFromExt,
  SUPPORTED_EXTS,
} from '@/lib/upload-meta';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'unauthorized' }, { status: 401 });

  const body = await request.json().catch(() => null);
  const filename = typeof body?.filename === 'string' ? body.filename : '';
  const size = typeof body?.size === 'number' ? body.size : 0;
  const sha256 = typeof body?.sha256 === 'string' ? body.sha256 : '';

  if (!filename) {
    return NextResponse.json({ error: '파일명이 없습니다.' }, { status: 400 });
  }
  if (!isSupportedFile(filename)) {
    return NextResponse.json(
      { error: `지원하지 않는 형식입니다. (${SUPPORTED_EXTS.join(', ')})` },
      { status: 400 },
    );
  }
  if (size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: '파일이 너무 큽니다 (최대 50MB).' },
      { status: 400 },
    );
  }

  const admin = createServiceClient();

  // 중복 검사 (같은 내용) — 업로드 전에 막아 불필요한 전송을 피한다.
  if (sha256) {
    const { data: dup } = await admin
      .from('documents')
      .select('filename')
      .eq('sha256', sha256)
      .maybeSingle();
    if (dup) {
      return NextResponse.json(
        { error: `이미 동일한 파일이 있습니다: ${dup.filename}` },
        { status: 409 },
      );
    }
  }

  // 원본 파일명은 documents.filename에 보존, storage 키는 UUID+확장자로 안전하게.
  const storagePath = safeStoragePath(filename);
  const { data, error } = await admin.storage
    .from(STORAGE_BUCKET)
    .createSignedUploadUrl(storagePath);
  if (error || !data) {
    return NextResponse.json(
      { error: `업로드 URL 발급 실패: ${error?.message ?? 'unknown'}` },
      { status: 500 },
    );
  }

  return NextResponse.json({
    storagePath,
    token: data.token,
    mime: mimeFromExt(filename),
  });
}
