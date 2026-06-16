// 업로드 메타데이터 단일 출처: 지원 형식·MIME·용량·버킷.
// 서명 라우트(/sign)와 처리 라우트(/upload) 양쪽에서 공유한다.
//
// 영상(mp4)은 의도적으로 미지원 — 전사가 느려 함수 타임아웃 위험이 크고,
// 강의 영상은 자막(VTT) 업로드를 권장. 오디오(mp3/m4a)는 유지.

export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // 50MB (Supabase Storage 무료 한도)
export const STORAGE_BUCKET = 'source-files';

const MIME: Record<string, string> = {
  pdf: 'application/pdf',
  txt: 'text/plain',
  vtt: 'text/vtt',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  hwp: 'application/x-hwp',
  mp3: 'audio/mpeg',
  m4a: 'audio/mp4',
};

export const SUPPORTED_EXTS = Object.keys(MIME);

export function extOf(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() ?? '';
}

export function mimeFromExt(filename: string): string {
  return MIME[extOf(filename)] ?? 'application/octet-stream';
}

export function isSupportedFile(filename: string): boolean {
  return extOf(filename) in MIME;
}
