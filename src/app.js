// src/app.js
//
// Express app definition. Used by both the local dev server
// (src/server.js) and the Vercel serverless entry (api/index.js).

import 'dotenv/config';
import express from 'express';
import { waitUntil } from '@vercel/functions';
import { answerQuestion } from './rag.js';
import {
  simpleText,
  useCallbackResponse,
  extractUtterance,
  extractCallbackUrl,
  extractUserId,
} from './kakao.js';
import { webhookSecretOk, rateLimited, validateUtterance } from './security.js';

const app = express();
app.use(express.json({ limit: '1mb' }));

// 카카오 callbackUrl은 발급 후 1분(60초)간 유효하고 1회만 쓸 수 있다.
// 답변 생성이 이 윈도를 넘기면 URL이 만료돼 답이 영구 유실되므로,
// 60초가 차기 전에 fallback 메시지라도 반드시 한 번 POST하도록
// 답변 생성에 자체 데드라인을 건다. POST 왕복 여유로 15초를 남긴다.
const ANSWER_DEADLINE_MS = 45_000;

// promise가 ms 안에 끝나지 않으면 Error(label)로 reject. 타이머는 정리.
// (원래 promise는 백그라운드에서 계속 돌 수 있으나 결과는 버려진다 —
//  호출 단위 abort는 별도 레버[resilience]에서 rag.js에 직접 적용 예정.)
function withDeadline(promise, ms, label) {
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label)), ms);
  });
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timer));
}

// "빠른 응답 우선" 예산. 콜백이 켜진 경우, 이 시간 안에 답이 나오면 동기로 바로
// 보내(대기 풍선 없음), 넘으면 그때 "생성 중"을 보내고 나머지는 callbackUrl로
// 비동기 전달한다. 5초 SLA 안에 응답이 나가야 하므로 보수적으로 잡는다.
// (콜드스타트가 부팅 시간을 먹으므로 keep-warm을 켜두면 더 안정적. 풍선이 너무
//  자주 뜨면 이 값을 낮추고, 거의 안 뜨면 살짝 올려도 됨.)
const SYNC_BUDGET_MS = 3500;

// 웹훅 공통 가드: 시크릿 경로 검증 → rate limit → 입력 검증.
// 통과 못 하면 사용자에게 보일 simpleText 응답을 돌려주고 false 반환.
function guard(req, res) {
  if (!webhookSecretOk(req)) {
    res.status(403).json({ error: 'forbidden' });
    return null;
  }
  const userId = extractUserId(req.body);
  if (rateLimited(userId)) {
    res.json(simpleText('요청이 많아 잠시 후 다시 시도해 주세요.'));
    return null;
  }
  const v = validateUtterance(extractUtterance(req.body));
  if (!v.ok) {
    res.json(
      simpleText(
        v.reason === 'too_long'
          ? '질문이 너무 깁니다. 500자 이내로 줄여 주세요.'
          : '질문을 입력해 주세요.',
      ),
    );
    return null;
  }
  return { userId, question: v.text };
}

// ---------- Health check ----------
app.get('/', (_req, res) => {
  res.json({ status: 'ok', service: 'kakao-class-bot' });
});

// ---------- Synchronous skill (<5s) ----------
// 시크릿 경로(:secret)는 선택적 — WEBHOOK_SECRET 설정 시 일치해야 통과.
// 콜백 라우트(/kakao/skill/callback/...)를 먼저 등록해 경로 충돌 방지.
async function handleSync(req, res) {
  const g = guard(req, res);
  if (!g) return;
  console.log(`[sync] Q: ${g.question}`);
  try {
    const answer = await answerQuestion(g.question, { userId: g.userId });
    console.log(`[sync] A: ${answer.slice(0, 80)}...`);
    res.json(simpleText(answer));
  } catch (err) {
    console.error('[sync] error:', err);
    res.json(
      simpleText(
        '죄송합니다. 일시적으로 답변을 생성하지 못했습니다. 잠시 후 다시 질문해 주세요.',
      ),
    );
  }
}

// ---------- Callback skill (빠른 응답 우선 + >5s 콜백) ----------
// 콜백이 켜져 있어도 빠른 답엔 대기 풍선 없이 바로 응답한다:
//   - SYNC_BUDGET 안에 답 완료  → 동기로 바로 답 (풍선 없음)
//   - 예산 초과(콜드/긴 답/Gemini 지연) → "생성 중" 보내고 callbackUrl로 1분 내 전달
// callbackUrl이 없으면(콜백 미설정) 동기로만 동작.
// Vercel에서 res.json() 이후 백그라운드 작업이 suspend되는 문제는 waitUntil로 해결.
async function handleCallback(req, res) {
  const g = guard(req, res);
  if (!g) return;
  const { userId, question } = g;
  const callbackUrl = extractCallbackUrl(req.body);
  console.log(`[callback] Q: ${question}`);
  console.log(`[callback] callbackUrl: ${callbackUrl ? 'yes' : 'no'}`);

  // 답변 생성은 딱 한 번만 시작하고(중복 호출 금지), 절대 reject하지 않게 감싼다.
  const answerPromise = answerQuestion(question, { userId })
    .then((answer) => ({ ok: true, answer }))
    .catch((error) => ({ ok: false, error }));

  // callbackUrl 없음 → 콜백 불가, 동기로만 (기존 폴백 동작)
  if (!callbackUrl) {
    const r = await answerPromise;
    if (!r.ok) console.error('[callback] sync-fallback error:', r.error);
    return res.json(simpleText(r.ok ? r.answer : '죄송합니다. 일시적인 오류가 발생했습니다.'));
  }

  // 빠른 응답 우선 — 예산 안에 끝나면 동기로 바로
  let timer;
  const budget = new Promise((resolve) => {
    timer = setTimeout(() => resolve('__budget__'), SYNC_BUDGET_MS);
  });
  const winner = await Promise.race([answerPromise, budget]);
  clearTimeout(timer);

  if (winner !== '__budget__') {
    // 예산 안에 완료 → 대기 풍선 없이 바로 답
    if (!winner.ok) console.error('[callback] fast-path error:', winner.error);
    console.log(`[callback] fast(<${SYNC_BUDGET_MS}ms) → 동기 응답`);
    return res.json(simpleText(winner.ok ? winner.answer : '죄송합니다. 일시적인 오류가 발생했습니다.'));
  }

  // 예산 초과 → "생성 중" 보내고 나머지는 callbackUrl로 비동기 전달
  console.log('[callback] slow → useCallback + 비동기 전달');
  res.json(useCallbackResponse('답변을 생성하고 있습니다. 잠시만 기다려 주세요. (최대 1분)'));

  // 이미 돌고 있는 answerPromise를 그대로 기다림(중복 생성 X). 60초 윈도 전 데드라인.
  const bgTask = (async () => {
    let payload;
    try {
      const r = await withDeadline(answerPromise, ANSWER_DEADLINE_MS, 'answer-deadline');
      payload = simpleText(
        r.ok ? r.answer : '답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.',
      );
      if (r.ok) console.log(`[callback] answer ready: ${r.answer.slice(0, 80)}...`);
      else console.error('[callback] background error:', r.error);
    } catch (err) {
      payload = simpleText('답변 생성이 지연되고 있습니다. 잠시 후 다시 질문해 주세요.');
    }
    try {
      await fetch(callbackUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      console.log('[callback] delivered');
    } catch (postErr) {
      // callbackUrl 만료/네트워크 오류 — 1회용이라 재시도 불가, 유실됨.
      console.error('[callback] delivery failed:', postErr);
    }
  })();

  try {
    waitUntil(bgTask);
  } catch (_) {
    /* non-Vercel: long-running process keeps bgTask alive anyway */
  }
}

// ---------- 라우트 등록 ----------
// 콜백을 먼저 등록(더 구체적). :secret? 은 선택적이라 기존 경로도 호환.
// 시크릿 경로:  /kakao/skill/<secret>,  /kakao/skill/callback/<secret>
// 호환 경로(WEBHOOK_SECRET 미설정 시): /kakao/skill,  /kakao/skill/callback
app.post('/kakao/skill/callback/:secret?', handleCallback);
app.post('/kakao/skill/:secret?', handleSync);

export default app;
