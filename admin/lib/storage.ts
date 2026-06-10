// Supabase Storage 키 생성.
// Storage 키는 ASCII만 허용(한글·공백 불가)하므로, 원본 파일명을 키에 쓰면
// "Invalid key" 에러가 난다(실제 발생했던 버그). UUID + 확장자로 안전하게 생성.
import { randomUUID } from 'node:crypto';

export function safeStoragePath(
  filename: string,
  uuid: string = randomUUID(),
): string {
  const ext = filename.includes('.')
    ? filename.split('.').pop()!.toLowerCase()
    : 'bin';
  const safeExt = ext.replace(/[^a-z0-9]/g, '') || 'bin';
  return `uploads/${uuid}.${safeExt}`;
}
