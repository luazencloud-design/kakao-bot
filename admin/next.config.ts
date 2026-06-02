import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // 상위 폴더(kakao-bot)에도 package-lock.json이 있어 Turbopack이
  // 워크스페이스 루트를 잘못 추론하는 경고를 방지. admin/ 자체를 루트로 고정.
  turbopack: {
    root: import.meta.dirname,
  },
  // pdf-parse(pdfjs-dist)는 워커 파일을 동적 로드하므로 번들하면 경로가 깨짐.
  // 번들에서 제외해 node_modules에서 직접 require → 워커 경로 정상 해석.
  serverExternalPackages: ['pdf-parse'],
};

export default nextConfig;
