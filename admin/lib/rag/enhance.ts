// RAG 품질 개선 모듈: 쿼리 재작성(캐시) + LLM 재정렬(reranking).
// 외부 서비스(Cohere 등) 없이 Gemini Flash Lite만으로 구현.

import { createServiceClient } from '@/lib/supabase/server';

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

function normalizeQuestion(q: string): string {
  return q.trim().toLowerCase().replace(/\s+/g, ' ').replace(/[?!.]+$/, '');
}

// 쿼리 재작성: 구어체·축약 질문을 검색용 키워드로 확장.
// query_rewrites 캐시로 같은 질문은 같은 재작성을 재사용 → 결정적. 봇(src/rag.js)과
// 같은 테이블·같은 프롬프트를 공유하므로 어드민 테스트와 봇 답이 일치한다.
// REWRITE=off 면 건너뜀. 실패해도 원문 사용 (graceful degradation).
// 재작성 프롬프트 버전 — 봇(src/rag.js)과 동일하게 유지. 프롬프트 변경 시 올려
// 옛 캐시를 자동 무효화(키가 안 겹침).
const REWRITE_VERSION = 'v2';

export async function rewriteQuery(question: string): Promise<string> {
  if (process.env.REWRITE === 'off') return question;
  const key = `${REWRITE_VERSION}:${normalizeQuestion(question)}`;
  const apply = (kw: string) => (kw ? `${question} ${kw}` : question);
  const admin = createServiceClient();

  // 1) 캐시 조회
  try {
    const { data } = await admin
      .from('query_rewrites')
      .select('rewrite')
      .eq('q_norm', key)
      .maybeSingle();
    if (data) return apply((data as { rewrite: string | null }).rewrite ?? '');
  } catch {
    /* 캐시 조회 실패 — 직접 재작성 */
  }

  // 2) 캐시 미스 → Gemini 재작성 (1.2s 타임아웃, 실패 시 '' = 재작성 없음)
  let keywords = '';
  try {
    const prompt = `다음은 사업자등록·오픈마켓 가입·강의 안내 챗봇에 들어온 사용자 질문입니다.
이 질문을 문서 검색에 적합하도록 핵심 키워드 중심으로 한 줄로 재작성하세요.
동의어나 정식 명칭이 있으면 함께 포함하세요.
상품·브랜드·카테고리처럼 영어로도 표기되는 용어는 영어 키워드도 함께 넣으세요 (예: 헬스뷰티 → Health Beauty, 롤렉스 → rolex, 순위 → ranking). 영어 자료(상품 검색어 순위 등)를 한국어 질문으로도 찾기 위함입니다.
설명 없이 재작성된 검색어만 출력하세요.

사용자 질문: ${question}

검색어:`;
    const result = (await callGemini(prompt, 100, 1200)).trim().replace(/^검색어:\s*/, '');
    if (result && result.length > 1) keywords = result;
  } catch {
    keywords = '';
  }

  // 3) 캐시에 저장 (빈 문자열도 저장해 "재작성 없음" 고정). 실패는 무시.
  try {
    await admin
      .from('query_rewrites')
      .upsert({ q_norm: key, rewrite: keywords }, { onConflict: 'q_norm' });
  } catch {
    /* 저장 실패 무시 */
  }
  return apply(keywords);
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
