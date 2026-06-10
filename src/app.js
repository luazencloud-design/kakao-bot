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

// ---------- Callback skill (>5s) ----------
// Requires "useCallback" to be enabled in OpenBuilder bot settings.
// Vercel 서버리스에서 res.json() 이후 백그라운드 작업이 suspend되는 문제를
// `@vercel/functions`의 waitUntil로 해결 — 함수 수명을 백그라운드 작업이
// 끝날 때까지 연장. 장수명 호스트(로컬·Fly.io)에선 waitUntil이 no-op이거나
// 예외라도 promise 자체는 그대로 실행되므로 try/catch로 감싼다.
async function handleCallback(req, res) {
  const g = guard(req, res);
  if (!g) return;
  const { userId, question } = g;
  const callbackUrl = extractCallbackUrl(req.body);
  console.log(`[callback] Q: ${question}`);
  console.log(`[callback] callbackUrl: ${callbackUrl ? 'yes' : 'no'}`);

  if (!callbackUrl) {
    try {
      const answer = await answerQuestion(question, { userId });
      return res.json(simpleText(answer));
    } catch (err) {
      console.error('[callback] sync-fallback error:', err);
      return res.json(simpleText('죄송합니다. 일시적인 오류가 발생했습니다.'));
    }
  }

  res.json(useCallbackResponse('답변을 생성하고 있습니다... (최대 30초)'));

  const bgTask = (async () => {
    try {
      const answer = await answerQuestion(question, { userId });
      await fetch(callbackUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simpleText(answer)),
      });
      console.log(`[callback] delivered: ${answer.slice(0, 80)}...`);
    } catch (err) {
      console.error('[callback] background error:', err);
      try {
        await fetch(callbackUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(
            simpleText('답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.'),
          ),
        });
      } catch (_) {
        /* swallow */
      }
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
