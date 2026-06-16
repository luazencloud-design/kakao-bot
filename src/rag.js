// src/rag.js
//
// Retrieval + generation (Supabase pgvector 버전):
//   1. embed the user's question with Gemini
//   2. hybrid_search RPC로 Supabase에서 top-K 청크 조회 (dense + sparse RRF)
//   3. ask Gemini to answer using only those chunks
//   4. cite the source document(s) the answer came from

import { createClient } from '@supabase/supabase-js';

const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';
const TOP_K = parseInt(process.env.TOP_K || '6', 10);

// ---------- Supabase 클라이언트 (lazy) ----------
// 모듈 로드 시점에 createClient를 호출하면, 환경변수가 없을 때
// "supabaseUrl is required"로 throw되어 서버리스 함수 전체가 크래시함.
// 그래서 첫 사용 시점에 lazy 생성하고, 누락 시 graceful 에러를 던진다.
let _supabase = null;
function getSupabase() {
  if (_supabase) return _supabase;
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error(
      'SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY가 설정되지 않았습니다. ' +
        'Vercel 환경변수를 확인하세요.',
    );
  }
  _supabase = createClient(url, key, { auth: { persistSession: false } });
  return _supabase;
}

// ---------- Embed a query (Gemini) ----------
// outputDimensionality는 chunks 테이블의 vector(768)와 일치해야 함.
async function embedQuery(text) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${EMBED_MODEL}:embedContent?key=${process.env.GEMINI_API_KEY}`;

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: `models/${EMBED_MODEL}`,
      content: { parts: [{ text }] },
      taskType: 'RETRIEVAL_QUERY',
      outputDimensionality: 768,
    }),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Gemini embedding API ${resp.status}: ${body}`);
  }
  const data = await resp.json();
  return data.embedding.values;
}

// ---------- Top-K 하이브리드 검색 (Supabase RPC) ----------
// dense (pgvector cosine) + sparse (tsvector) → RRF 융합 → top-K
async function searchTopK(queryEmbed, queryText, k = TOP_K) {
  const { data, error } = await getSupabase().rpc('hybrid_search', {
    query_embedding: queryEmbed,
    query_text: queryText,
    match_count: k,
  });
  if (error) {
    throw new Error(`Supabase hybrid_search 실패: ${error.message}`);
  }
  // 반환 컬럼: id, document_id, chunk_text, metadata, source, rrf_score
  // 기존 인터페이스와 맞추기 위해 text·score로 매핑
  return (data ?? []).map((row) => ({
    id: row.id,
    document_id: row.document_id,
    source: row.source ?? '미상',
    text: row.chunk_text,
    metadata: row.metadata ?? {},
    score: row.rrf_score,
  }));
}

// ---------- Generate answer (Gemini) ----------
async function generateWithGemini(systemPrompt, userQuestion) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not set. Add it to .env or Vercel env vars.');
  }

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${GEMINI_MODEL}:generateContent?key=${apiKey}`;

  const body = JSON.stringify({
    system_instruction: {
      parts: [{ text: systemPrompt }],
    },
    contents: [
      {
        role: 'user',
        parts: [{ text: userQuestion }],
      },
    ],
    generationConfig: {
      maxOutputTokens: 2048,
      // temperature 0 — 같은 질문·같은 자료에 같은 답이 나오도록(결정성).
      // 0.3이면 경계 질문에서 "없음↔답함"이 매 호출 갈려 일관성이 깨졌음.
      temperature: 0,
      thinkingConfig: {
        thinkingBudget: 0,
      },
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
      const errText = await resp.text();
      lastErr = `Gemini ${resp.status}: ${errText.slice(0, 200)}`;
      if (attempt < maxAttempts) {
        const waitMs = 800 * attempt;
        console.warn(
          `[rag] ${lastErr.slice(0, 120)} — attempt ${attempt}/${maxAttempts}, waiting ${waitMs}ms`,
        );
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }
      throw new Error(`Gemini failed after ${maxAttempts} attempts: ${lastErr}`);
    }

    if (!resp.ok) {
      throw new Error(`Gemini API ${resp.status}: ${await resp.text()}`);
    }

    const data = await resp.json();
    const candidate = data?.candidates?.[0];
    const text = candidate?.content?.parts?.[0]?.text;

    if (!text) {
      const finishReason = candidate?.finishReason ?? 'unknown';
      throw new Error(`Gemini returned no text (finishReason=${finishReason})`);
    }
    return text;
  }

  throw new Error(`generateWithGemini: exhausted retries. Last error: ${lastErr}`);
}

// ---------- Query rewriting (Gemini, 캐시로 결정적) ----------
// 재작성은 Gemini 호출이라 temperature 0에도 결과가 매번 달라질 수 있어
// (변형 A "…사업자등록…" ↔ 변형 B "…소명자료, 소명서…"), 그게 sparse 검색어를
// 바꿔 같은 질문에 다른 자료를 끌어와 답이 흔들리는 비결정의 원인이었다.
// → 정규화 질문을 키로 재작성 결과를 query_rewrites 테이블에 캐시.
//   같은 질문은 저장된 재작성을 재사용 → 결정적. 봇·어드민이 캐시를 공유하므로
//   둘의 답도 일치한다. REWRITE=off 면 재작성을 건너뛰고 원질문을 그대로 쓴다.
function normalizeQuestion(q) {
  return q.trim().toLowerCase().replace(/\s+/g, ' ').replace(/[?!.]+$/, '');
}

async function rewriteQuery(question) {
  if (process.env.REWRITE === 'off') return question;
  const key = normalizeQuestion(question);
  const apply = (kw) => (kw ? `${question} ${kw}` : question);

  // 1) 캐시 조회 (히트 시 결정적으로 같은 검색어 재사용; ''는 "재작성 없음" 고정)
  try {
    const { data } = await getSupabase()
      .from('query_rewrites')
      .select('rewrite')
      .eq('q_norm', key)
      .maybeSingle();
    if (data) return apply(data.rewrite ?? '');
  } catch {
    /* 캐시 조회 실패 — 아래에서 직접 재작성 */
  }

  // 2) 캐시 미스 → Gemini 재작성 (1.2s 타임아웃, 실패 시 '' = 재작성 없음)
  let keywords = '';
  try {
    const prompt = `다음은 사업자등록·오픈마켓 가입·강의 안내 챗봇에 들어온 사용자 질문입니다.
이 질문을 문서 검색에 적합하도록 핵심 키워드 중심으로 한 줄로 재작성하세요.
동의어나 정식 명칭이 있으면 함께 포함하세요. 설명 없이 재작성된 검색어만 출력하세요.

사용자 질문: ${question}

검색어:`;
    const result = (await generateRaw(prompt, 100, 1200)).trim().replace(/^검색어:\s*/, '');
    if (result && result.length > 1) keywords = result;
  } catch {
    keywords = '';
  }

  // 3) 캐시에 저장 (빈 문자열도 저장해 "재작성 없음"을 고정). 실패는 무시.
  try {
    await getSupabase()
      .from('query_rewrites')
      .upsert({ q_norm: key, rewrite: keywords }, { onConflict: 'q_norm' });
  } catch {
    /* 저장 실패 — 다음 요청에서 재시도 */
  }
  return apply(keywords);
}

// 시스템 프롬프트 없이 단순 생성 (재작성용).
// timeoutMs 지정 시 요청 단위 타임아웃(AbortSignal). 재작성이 Gemini 느린
// 구간에 매달려 전체 응답을 지연시키지 않도록 캡을 건다 (초과 시 throw →
// 호출부의 graceful fallback이 원질문을 그대로 사용).
async function generateRaw(prompt, maxTokens, timeoutMs) {
  const apiKey = process.env.GEMINI_API_KEY;
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

// ---------- 질의 로깅 (queries 테이블, fire-and-forget) ----------
function logQuery({ userId, utterance, rewritten, chunks, answer, latencyMs }) {
  try {
    const isUnanswered = answer.includes('제공된 자료에 포함되어 있지 않');
    getSupabase()
      .from('queries')
      .insert({
        user_id: userId ?? null,
        utterance,
        rewritten_query: rewritten,
        retrieved_chunk_ids: (chunks ?? []).map((c) => c.id),
        answer,
        sources: isUnanswered ? [] : [...new Set((chunks ?? []).map((c) => c.source))],
        latency_ms: latencyMs,
        llm_provider: 'gemini',
      })
      .then(() => {}, () => {}); // 실패해도 응답에 영향 없게 swallow
  } catch (_) {
    /* env 누락 등 — swallow */
  }
}

// ---------- 에러 로깅 (queries 테이블에 실패 기록) ----------
// 봇이 "일시적 오류"만 보여주면 운영자가 원인을 모름. 실제 에러를 기록해
// 어드민에서 무엇이 실패했는지(예: API 키 무효, 할당량 초과) 볼 수 있게 함.
function logError({ userId, utterance, error, latencyMs }) {
  const message = error instanceof Error ? error.message : String(error);
  console.error('[rag] answerQuestion 실패:', message);
  try {
    getSupabase()
      .from('queries')
      .insert({
        user_id: userId ?? null,
        utterance,
        answer: `[오류] ${message.slice(0, 500)}`,
        sources: [],
        latency_ms: latencyMs,
        llm_provider: 'error',
      })
      .then(() => {}, () => {});
  } catch (_) {
    /* swallow */
  }
}

// ---------- Main entry ----------

export async function answerQuestion(question, opts = {}) {
  if (!question || question.trim().length === 0) {
    return '질문을 입력해 주세요.';
  }
  const userId = opts.userId ?? null;
  const t0 = Date.now();
  try {
    return await answerQuestionInner(question, userId, t0);
  } catch (err) {
    logError({ userId, utterance: question, error: err, latencyMs: Date.now() - t0 });
    throw err; // app.js가 사용자에게 안내 메시지 보내도록 그대로 전파
  }
}

async function answerQuestionInner(question, userId, t0) {
  // 1) 재작성(키워드 보강)과 임베딩을 병렬 실행.
  //    - 임베딩은 원질문 기준 — 의미 벡터가 깔끔하고, 재작성이 가끔 더하는
  //      엉뚱한 키워드(예: "소명"→"사업자등록 소명") 노이즈를 배제한다.
  //    - 재작성(searchQuery)은 sparse 키워드 검색어 보강용. 1.2s 타임아웃으로
  //      임베딩 뒤에 숨겨 재작성 LLM 호출이 임계경로 latency를 늘리지 않게 한다.
  const [searchQuery, qEmbed] = await Promise.all([
    rewriteQuery(question),
    embedQuery(question),
  ]);

  // 2) 하이브리드 검색(RRF)에서 곧바로 top-K. 별도 LLM 재정렬 제거 —
  //    Gemini 왕복 1회와 그 꼬리 지연(p95 기여)을 없앤다. RRF가 이미
  //    dense+sparse를 융합 정렬하므로 상위 K개를 그대로 신뢰한다.
  const top = await searchTopK(qEmbed, searchQuery, TOP_K);

  if (top.length === 0) {
    const answer = '해당 정보는 제공된 자료에 포함되어 있지 않습니다. 담당자에게 문의해 주세요.';
    logQuery({ userId, utterance: question, rewritten: searchQuery, chunks: [], answer, latencyMs: Date.now() - t0 });
    return answer;
  }

  // Build the context block, labeling each chunk with its source.
  const context = top
    .map((c, i) => `[자료 ${i + 1} | 출처: ${c.source}]\n${c.text}`)
    .join('\n\n---\n\n');

  // The unique sources actually retrieved (for the model to cite).
  const usedSources = [...new Set(top.map((c) => c.source))];

  // 2) Generate
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

  const text = await generateWithGemini(systemPrompt, question);
  const answer = text.trim() || '답변을 생성하지 못했습니다. 다시 시도해 주세요.';
  logQuery({ userId, utterance: question, rewritten: searchQuery, chunks: top, answer, latencyMs: Date.now() - t0 });
  return answer;
}
