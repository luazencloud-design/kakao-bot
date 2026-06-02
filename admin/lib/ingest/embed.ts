// Gemini 임베딩. 청크 텍스트 → 768차원 벡터. 재시도 포함.

const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';

export async function embedDocument(text: string): Promise<number[]> {
  return embed(text, 'RETRIEVAL_DOCUMENT');
}

export async function embedQuery(text: string): Promise<number[]> {
  return embed(text, 'RETRIEVAL_QUERY');
}

// 배치 임베딩: 여러 청크를 한 번의 호출로. Gemini batchEmbedContents (최대 100개/호출).
// 순차 호출(청크당 0.5초) 대비 수십 배 빠름 → Vercel 타임아웃 회피.
export async function embedDocumentsBatch(texts: string[]): Promise<number[][]> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');
  if (texts.length === 0) return [];

  const BATCH = 100;
  const out: number[][] = [];

  for (let start = 0; start < texts.length; start += BATCH) {
    const slice = texts.slice(start, start + BATCH);
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/` +
      `${EMBED_MODEL}:batchEmbedContents?key=${apiKey}`;
    const body = JSON.stringify({
      requests: slice.map((text) => ({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text }] },
        taskType: 'RETRIEVAL_DOCUMENT',
        outputDimensionality: 768,
      })),
    });

    const maxAttempts = 5;
    let lastErr = '';
    let done = false;
    for (let attempt = 1; attempt <= maxAttempts && !done; attempt++) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if ([429, 500, 502, 503, 504].includes(resp.status)) {
        lastErr = `Gemini batch embedding ${resp.status}`;
        if (attempt < maxAttempts) {
          await new Promise((r) => setTimeout(r, 2000 * attempt));
          continue;
        }
        throw new Error(`${lastErr} (${maxAttempts}회 재시도 실패)`);
      }
      if (!resp.ok) {
        throw new Error(`Gemini batch embedding API ${resp.status}: ${await resp.text()}`);
      }
      const data = await resp.json();
      for (const e of data.embeddings ?? []) out.push(e.values);
      done = true;
    }
  }
  return out;
}

async function embed(text: string, taskType: string): Promise<number[]> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${EMBED_MODEL}:embedContent?key=${apiKey}`;

  const body = JSON.stringify({
    model: `models/${EMBED_MODEL}`,
    content: { parts: [{ text }] },
    taskType,
    outputDimensionality: 768,
  });

  const maxAttempts = 5;
  let lastErr = '';
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if ([429, 500, 502, 503, 504].includes(resp.status)) {
      lastErr = `Gemini embedding ${resp.status}`;
      if (attempt < maxAttempts) {
        await new Promise((r) => setTimeout(r, 2000 * attempt));
        continue;
      }
      throw new Error(`${lastErr} (${maxAttempts}회 재시도 실패)`);
    }
    if (!resp.ok) {
      throw new Error(`Gemini embedding API ${resp.status}: ${await resp.text()}`);
    }
    const data = await resp.json();
    return data.embedding.values;
  }
  throw new Error(`embed: 재시도 소진. ${lastErr}`);
}
