import { describe, it, expect } from 'vitest';
import { extractVtt } from './extract';

describe('extractVtt (자막 → 발화 텍스트)', () => {
  it('WEBVTT 헤더·번호·타임스탬프를 제거하고 발화만 남긴다', () => {
    const vtt = `WEBVTT

1
00:00:01.000 --> 00:00:03.000
안녕하세요 여러분

2
00:00:03.500 --> 00:00:05.000
오늘은 사업자등록을 배웁니다`;
    expect(extractVtt(vtt)).toBe('안녕하세요 여러분\n오늘은 사업자등록을 배웁니다');
  });

  it('화자 이름 prefix를 제거한다', () => {
    const vtt = `WEBVTT

1
00:00:01.000 --> 00:00:03.000
정용렬: 안녕하세요`;
    expect(extractVtt(vtt)).toBe('안녕하세요');
  });

  it('NOTE/STYLE 블록을 무시한다', () => {
    const vtt = `WEBVTT

NOTE 이건 주석

00:00:01.000 --> 00:00:02.000
실제 발화`;
    expect(extractVtt(vtt)).toBe('실제 발화');
  });

  it('빈 자막은 빈 문자열', () => {
    expect(extractVtt('WEBVTT\n\n')).toBe('');
  });
});
