import { describe, it, expect } from 'vitest';
import { safeStoragePath } from './storage';

// 버그 회귀 테스트:
// 한글/공백 파일명을 storage 키에 그대로 쓰면 Supabase가 "Invalid key" 거부.
// safeStoragePath는 UUID + 확장자만 쓰므로 키에 항상 ASCII만 들어가야 한다.
describe('safeStoragePath (한글 키 버그 회귀)', () => {
  const ASCII_KEY = /^uploads\/[a-z0-9-]+\.[a-z0-9]+$/;

  it('한글·공백 파일명도 ASCII 키를 만든다', () => {
    const key = safeStoragePath('듀오링고 시작일은 26.06.02 시간은 914일.txt', 'fixed-uuid');
    expect(key).toBe('uploads/fixed-uuid.txt');
    expect(key).toMatch(ASCII_KEY);
  });

  it('원본 확장자를 보존한다', () => {
    expect(safeStoragePath('강의자료.pdf', 'u')).toBe('uploads/u.pdf');
    expect(safeStoragePath('녹화.MP4', 'u')).toBe('uploads/u.mp4'); // 소문자화
  });

  it('확장자 없으면 bin', () => {
    expect(safeStoragePath('확장자없는파일', 'u')).toBe('uploads/u.bin');
  });

  it('확장자에 이상한 문자가 있어도 ASCII만 남긴다', () => {
    expect(safeStoragePath('file.p df', 'u')).toBe('uploads/u.pdf');
  });
});
