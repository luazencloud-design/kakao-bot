import { describe, it, expect } from 'vitest';
import { formatBytes, timeAgo } from './types';

describe('formatBytes', () => {
  it('null·0은 대시', () => {
    expect(formatBytes(null)).toBe('—');
    expect(formatBytes(0)).toBe('—');
  });
  it('단위를 적절히 붙인다', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2 KB');
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB');
  });
});

describe('timeAgo', () => {
  it('방금 / 분 / 시간 / 일 단위로 표시', () => {
    const now = Date.now();
    expect(timeAgo(new Date(now - 30 * 1000).toISOString())).toBe('방금');
    expect(timeAgo(new Date(now - 5 * 60 * 1000).toISOString())).toBe('5분 전');
    expect(timeAgo(new Date(now - 3 * 3600 * 1000).toISOString())).toBe('3시간 전');
    expect(timeAgo(new Date(now - 2 * 86400 * 1000).toISOString())).toBe('2일 전');
  });
});
