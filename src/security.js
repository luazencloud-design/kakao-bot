// src/security.js
//
// 봇 웹훅 보안: (1) 시크릿 경로 검증, (2) 사용자별 rate limit, (3) 입력 가드.

// ---------- (1) 웹훅 시크릿 ----------
// WEBHOOK_SECRET이 설정돼 있으면 경로의 :secret 세그먼트와 일치해야 통과.
// 미설정이면 통과(시크릿 도입 전 호환). 카카오 오픈빌더 스킬 URL에만
// 시크릿을 포함시키면 외부인은 경로를 몰라 호출 불가.
export function webhookSecretOk(req) {
  const expected = process.env.WEBHOOK_SECRET;
  if (!expected) return true; // 미설정 → 호환 모드
  return req.params?.secret === expected;
}

// ---------- (2) Rate limit (인메모리 슬라이딩 윈도우) ----------
// 사용자(카카오 user.id)당 windowMs 안에 max회 초과 시 차단.
// 주의: 서버리스 인스턴스별 메모리라 콜드스타트 시 초기화되고 인스턴스 간
// 공유 안 됨. 기본적인 남용 방어용. 엄격하게 하려면 Redis(Upstash) 필요.
const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = 20;
const hits = new Map(); // userId -> number[] (타임스탬프)

export function rateLimited(userId) {
  if (!userId) return false; // user.id 없으면 제한 안 함(헬스체크 등)
  const now = Date.now();
  const arr = (hits.get(userId) ?? []).filter((t) => now - t < WINDOW_MS);
  arr.push(now);
  hits.set(userId, arr);

  // 메모리 누수 방지: 가끔 오래된 유저 정리
  if (hits.size > 5000) {
    for (const [k, v] of hits) {
      if (v.every((t) => now - t >= WINDOW_MS)) hits.delete(k);
    }
  }
  return arr.length > MAX_PER_WINDOW;
}

// ---------- (3) 입력 가드 ----------
// 길이 제한(임베딩 비용·남용 방지) + 프롬프트 인젝션 의심 패턴 완화.
const MAX_UTTERANCE = 500;
const INJECTION_RE =
  /(ignore (the )?(previous|above) instructions|system\s*:|당신의?\s*(지시|규칙)을?\s*무시|이전\s*지시\s*무시)/i;

export function validateUtterance(text) {
  const q = (text ?? '').trim();
  if (q.length === 0) return { ok: false, reason: 'empty' };
  if (q.length > MAX_UTTERANCE) return { ok: false, reason: 'too_long' };
  // 인젝션 의심 문구는 제거(차단까진 안 하고 무력화) — RAG는 자료 기반이라
  // 영향이 제한적이지만, 시스템 프롬프트 탈취 시도 문구는 걷어낸다.
  const sanitized = q.replace(INJECTION_RE, '').trim();
  return { ok: true, text: sanitized || q };
}
