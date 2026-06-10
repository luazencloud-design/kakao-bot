import { describe, it, expect } from 'vitest';
import { inferCategory } from './category';

describe('inferCategory (파일명 자동 분류)', () => {
  it('오픈마켓 브랜드명을 오픈마켓가입으로', () => {
    expect(inferCategory('11번가 가입안내서.pdf')).toBe('오픈마켓가입');
    expect(inferCategory('지마켓 판매자센터 가입안내서.pdf')).toBe('오픈마켓가입');
    expect(inferCategory('쿠팡상품등록.pdf')).toBe('오픈마켓가입');
  });

  it('"가입안내"·"판매자" 같은 키워드도 오픈마켓가입으로', () => {
    expect(inferCategory('스마트스토어 가입안내서.pdf')).toBe('오픈마켓가입');
  });

  it('사업자·세무 관련은 사업자등록으로', () => {
    expect(inferCategory('오리엔테이션 사업자내기.pdf')).toBe('사업자등록');
    expect(inferCategory('부가세 신고 안내.pdf')).toBe('사업자등록');
  });

  it('도구 이름은 도구가이드로', () => {
    expect(inferCategory('노션 자료 열람방법.pdf')).toBe('도구가이드');
    expect(inferCategory('네이버웨일온_방법.pdf')).toBe('도구가이드');
  });

  it('강의·녹화·미디어 확장자는 강의자료로', () => {
    expect(inferCategory('18기 5주차 0320.pdf')).toBe('강의자료');
    expect(inferCategory('GMT20260416-Recording.transcript.vtt')).toBe('강의자료');
    expect(inferCategory('오픈채팅방 실명변환방법.mp4')).toBe('강의자료');
  });

  it('매칭 안 되면 기타', () => {
    expect(inferCategory('무작위파일.txt')).toBe('기타');
  });

  it('우선순위: "가입안내"가 도구명보다 먼저 (노션 가입안내서 → 오픈마켓)', () => {
    // 도구명(노션)이 들어가도 "가입안내" 키워드가 우선
    expect(inferCategory('노션 가입안내서.pdf')).toBe('오픈마켓가입');
  });
});
