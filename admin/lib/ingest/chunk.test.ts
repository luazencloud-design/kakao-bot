import { describe, it, expect } from 'vitest';
import { chunkText } from './chunk';

describe('chunkText', () => {
  it('빈 입력은 빈 배열', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   \n\n  ')).toEqual([]);
  });

  it('짧은 텍스트는 청크 1개', () => {
    const chunks = chunkText('사업자등록은 홈택스에서 신청합니다.');
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toContain('사업자등록');
  });

  it('빈 줄로 구분된 문단들을 합쳐 800자 이내로 묶는다', () => {
    // 두 짧은 문단 → 합쳐도 800자 미만이라 1개 청크
    const text = 'Q: 가입 방법?\nA: 회원가입 후.\n\nQ: 수수료?\nA: 3.4%입니다.';
    const chunks = chunkText(text);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toContain('가입');
    expect(chunks[0]).toContain('수수료');
  });

  it('합치면 800자를 넘으면 문단 경계에서 분리한다', () => {
    const para = '가'.repeat(500); // 각 500자
    const text = `${para}\n\n${para}`; // 합치면 1000자 > 800
    const chunks = chunkText(text);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].length).toBeLessThanOrEqual(800);
  });

  it('단일 문단이 800자를 넘으면 강제로 잘라낸다', () => {
    const long = '가'.repeat(2000);
    const chunks = chunkText(long);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(800);
  });

  it('연속된 빈 줄(\\n\\n\\n)을 하나로 정규화한다', () => {
    const text = 'A문단\n\n\n\nB문단';
    const chunks = chunkText(text);
    // 두 짧은 문단이 한 청크로 합쳐짐 (정규화 후 단일 \n\n 경계)
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toContain('A문단');
    expect(chunks[0]).toContain('B문단');
  });
});
