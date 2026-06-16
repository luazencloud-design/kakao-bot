// RAG 품질 개선 모듈: 쿼리 재작성 + LLM 재정렬(reranking).
// 외부 서비스(Cohere 등) 없이 Gemini Flash Lite만으로 구현.

const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';

async function callGemini(
  prompt: string,
  maxTokens = 256,
  timeoutMs?: number,
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${GEMINI_MODEL}:generateContent?key=${apiKey}`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: maxTokens,
        temperature: 0,
        thinkingConfig: { thinkingBudget: 0 },
      },
    }),
    ...(timeoutMs ? { signal: AbortSignal.timeout(timeoutMs) } : {}),
  });
  if (!resp.ok) throw new Error(`Gemini ${resp.status}`);
  const data = await resp.json();
  return data?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
}

// 쿼리 재작성: 구어체·축약 질문을 검색용 키워드로 확장.
// 실패해도 원문 사용 (graceful degradation).
export async function rewriteQuery(question: string): Promise<string> {
  try {
    const prompt = `다음은 사업자등록·오픈마켓 가입·강의 안내 챗봇에 들어온 사용자 질문입니다.
이 질문을 문서 검색에 적합하도록 핵심 키워드 중심으로 한 줄로 재작성하세요.
동의어나 정식 명칭이 있으면 함께 포함하세요. 설명 없이 재작성된 검색어만 출력하세요.

사용자 질문: ${question}

검색어:`;
    // 1.2s 타임아웃 — 재작성은 검색어 보강용 보조 단계라, 느리면 원문으로 즉시 폴백.
    const result = (await callGemini(prompt, 100, 1200)).trim().replace(/^검색어:\s*/, '');
    // 원문 + 재작성을 합쳐서 둘 다 반영
    return result && result.length > 1 ? `${question} ${result}` : question;
  } catch {
    return question;
  }
}

export interface RerankCandidate {
  id: number;
  source: string;
  text: string;
  score: number;
}

// LLM 재정렬: 검색된 후보 중 질문에 실제로 답하는 청크를 Gemini가 골라 순위 매김.
// 보일러플레이트(저작권 안내 등) 노이즈 제거에 효과적.
// 실패하면 원래 순서 유지.
export async function rerank(
  question: string,
  candidates: RerankCandidate[],
  topK: number,
): Promise<RerankCandidate[]> {
  if (candidates.length <= topK) return candidates;
  try {
    const list = candidates
      .map((c, i) => `[${i}] (출처: ${c.source})\n${c.text.slice(0, 300)}`)
      .join('\n\n');
    const prompt = `질문에 답하는 데 가장 유용한 자료를 순서대로 고르세요.
질문: ${question}

자료 목록:
${list}

가장 관련 높은 자료 ${topK}개의 번호만 쉼표로 구분해 순서대로 출력하세요 (예: 2,0,5,1). 설명 없이 번호만.`;
    const result = await callGemini(prompt, 50);
    const indices = result
      .match(/\d+/g)
      ?.map(Number)
      .filter((n) => n >= 0 && n < candidates.length);
    if (!indices || indices.length === 0) return candidates.slice(0, topK);
    // 중복 제거 + topK개
    const seen = new Set<number>();
    const ordered: RerankCandidate[] = [];
    for (const idx of indices) {
      if (!seen.has(idx)) {
        seen.add(idx);
        ordered.push(candidates[idx]);
      }
      if (ordered.length >= topK) break;
    }
    // 부족하면 원래 순서로 채움
    for (const c of candidates) {
      if (ordered.length >= topK) break;
      if (!ordered.includes(c)) ordered.push(c);
    }
    return ordered;
  } catch {
    return candidates.slice(0, topK);
  }
}
