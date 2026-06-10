import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // 상위 폴더(kakao-bot)에도 package-lock.json이 있어 Turbopack이
  // 워크스페이스 루트를 잘못 추론하는 경고를 방지. admin/ 자체를 루트로 고정.
  turbopack: {
    root: import.meta.dirname,
  },
  // 동적 로드·네이티브 의존이 있는 추출 라이브러리는 번들에서 제외해
  // node_modules에서 직접 require (경로·바이너리 해석 정상화).
  serverExternalPackages: ['unpdf', 'officeparser', 'hwp.js'],
};

export default nextConfig;
