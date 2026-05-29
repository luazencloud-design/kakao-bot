// 텍스트를 800자 단위(문단 경계 우선, overlap 100)로 청킹.
// 기존 scripts/ingest.js의 chunkText 로직 포팅.

export function chunkText(text: string, targetSize = 800, overlap = 100): string[] {
  const clean = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const paragraphs = clean
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = '';

  const flush = () => {
    if (current) {
      chunks.push(current);
      current = '';
    }
  };

  for (const para of paragraphs) {
    if ((current ? current.length + 2 : 0) + para.length <= targetSize) {
      current = current ? `${current}\n\n${para}` : para;
      continue;
    }
    flush();
    if (para.length > targetSize) {
      for (let i = 0; i < para.length; i += targetSize - overlap) {
        chunks.push(para.slice(i, i + targetSize));
      }
    } else {
      current = para;
    }
  }
  flush();

  return chunks;
}
