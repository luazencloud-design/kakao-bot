// 어드민 테스트용 RAG 쿼리: 임베딩 → hybrid_search → Gemini 생성.
// 봇(src/rag.js)과 동일 로직을 어드민에서 재현해 답변·검색결과·타이밍 반환.

import { createServiceClient } from '@/lib/supabase/server';
import { embedQuery } from '@/lib/ingest/embed';
import { rewriteQuery } from '@/lib/rag/enhance';

const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';
const TOP_K = parseInt(process.env.TOP_K || '6', 10);

export interface RetrievedChunk {
  id: number;
  source: string;
  text: string;
  score: number;
}

export interface RagResult {
  answer: string;
  chunks: RetrievedChunk[];
  rewrittenQuery: string;
  timings: { prep: number; search: number; generate: number; total: number };
}

export async function ragQuery(question: string): Promise<RagResult> {
  const t0 = Date.now();

  // 1. 재작성(키워드 보강)과 임베딩을 병렬 — 봇(src/rag.js)과 동일 로직.
  //    임베딩은 원질문 기준(의미 벡터가 깔끔), 재작성은 sparse 검색어 보강용
  //    (1.2s 타임아웃). 두 LLM/임베딩 호출을 겹쳐 임계경로를 줄인다.
  const [searchQuery, qEmbed] = await Promise.all([
    rewriteQuery(question),
    embedQuery(question),
  ]);
  const tPrep = Date.now();

  // 2. 하이브리드 검색에서 곧바로 top-K (별도 LLM 재정렬 제거 — RRF 순위 신뢰).
  const supabase = createServiceClient();
  const { data, error } = await supabase.rpc('hybrid_search', {
    query_embedding: qEmbed,
    query_text: searchQuery,
    match_count: TOP_K,
  });
  if (error) throw new Error(`hybrid_search 실패: ${error.message}`);
  const tSearch = Date.now();

  const chunks: RetrievedChunk[] = (data ?? []).map(
    (row: { id: number; source: string; chunk_text: string; rrf_score: number }) => ({
      id: row.id,
      source: row.source ?? '미상',
      text: row.chunk_text,
      score: row.rrf_score,
    }),
  );

  if (chunks.length === 0) {
    return {
      answer: '해당 정보는 제공된 자료에 포함되어 있지 않습니다. 담당자에게 문의해 주세요.',
      chunks: [],
      rewrittenQuery: searchQuery,
      timings: {
        prep: tPrep - t0,
        search: tSearch - tPrep,
        generate: 0,
        total: tSearch - t0,
      },
    };
  }

  // 3. 생성
  const context = chunks
    .map((c, i) => `[자료 ${i + 1} | 출처: ${c.source}]\n${c.text}`)
    .join('\n\n---\n\n');
  const usedSources = [...new Set(chunks.map((c) => c.source))];

  const systemPrompt = `당신은 사업자등록 및 운영 안내 챗봇입니다. 여러 안내 자료를 기반으로 사용자의 질문에 답변합니다.

규칙:
1. 반드시 아래 [참고 자료]에 있는 내용만 근거로 답변하세요. 자료에 없는 내용은 절대 추측하거나 일반 상식으로 보충하지 마세요.
   특히 전화번호·사이트명(예: 홈택스/정부24)·매장명·메뉴 경로·구체 수치(금액·기간·개수·시간)는 [참고 자료]에 그대로 적혀 있는 것만 쓰세요. 자료에 없는 구체값은 지어내지 말고 그 부분은 생략하세요.
2. 자료에 답이 없으면 "해당 정보는 제공된 자료에 포함되어 있지 않습니다. 담당자에게 문의해 주세요."라고 답하세요.
3. 답변은 친절하고 간결하게, 본문은 600자 내외로 작성하세요. 여러 항목은 번호 목록으로 정리하세요.
4. 답변 본문 다음에 빈 줄을 하나 두고, 마지막 줄에 "📚 출처: <파일명>" 형식으로 실제로 참고한 자료의 파일명을 명시하세요. 여러 자료를 참고했다면 쉼표로 구분하세요. 자료에 답이 없어 모른다고 답한 경우에는 출처 줄을 생략하세요.
5. 한국어로 답변하세요.

이번 질의에서 검색된 자료의 출처 후보:
${usedSources.map((s) => `- ${s}`).join('\n')}

[참고 자료]
${context}`;

  const answer = await generateWithGemini(systemPrompt, question);
  const t3 = Date.now();

  return {
    answer: answer.trim() || '답변을 생성하지 못했습니다.',
    chunks,
    rewrittenQuery: searchQuery,
    timings: {
      prep: tPrep - t0,
      search: tSearch - tPrep,
      generate: t3 - tSearch,
      total: t3 - t0,
    },
  };
}

async function generateWithGemini(systemPrompt: string, userQuestion: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${GEMINI_MODEL}:generateContent?key=${apiKey}`;

  const body = JSON.stringify({
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [{ role: 'user', parts: [{ text: userQuestion }] }],
    generationConfig: {
      maxOutputTokens: 2048,
      temperature: 0, // 결정성 — 경계 질문 "없음↔답함" 흔들림 방지 (봇과 동일)
      thinkingConfig: { thinkingBudget: 0 },
    },
  });

  const maxAttempts = 3;
  let lastErr = '';
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    if ([429, 500, 502, 503, 504].includes(resp.status)) {
      lastErr = `Gemini ${resp.status}`;
      if (attempt < maxAttempts) {
        await new Promise((r) => setTimeout(r, 800 * attempt));
        continue;
      }
      throw new Error(`Gemini 생성 실패: ${lastErr}`);
    }
    if (!resp.ok) throw new Error(`Gemini API ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) throw new Error('Gemini 빈 응답');
    return text;
  }
  throw new Error(`generateWithGemini: 재시도 소진. ${lastErr}`);
}
