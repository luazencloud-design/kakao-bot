# 배포 가이드 (봇 + 어드민, 둘 다 Vercel)

봇과 어드민 모두 Vercel에 배포한다. 둘 다 같은 Supabase를 바라본다.

```
[수강생] ──카카오──> Vercel 프로젝트 A (봇, Express)    ─┐
[운영자] ──브라우저─> Vercel 프로젝트 B (어드민, Next.js) ─┼─> Supabase
                                                          ─┘
```

> 한 GitHub 레포(`luazencloud-design/kakao-bot`)에 **Vercel 프로젝트 2개**.
> 각각 Root Directory가 다름:
> - 봇: Root = 레포 루트 (`/`)
> - 어드민: Root = `admin`

---

## A. 봇 배포 (기존 Vercel 프로젝트 재활용)

봇은 이미 Vercel 프로젝트가 있다. **환경변수만 추가**하고 새 코드로 재배포하면 된다.

### 1. 환경변수 추가
기존 봇 Vercel 프로젝트 → Settings → Environment Variables:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | `AIza…` (이미 있을 것) |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` |
| `EMBED_MODEL` | `gemini-embedding-001` |
| `TOP_K` | `6` (완결성↑. 일관성과는 무관) |
| `SUPABASE_URL` | `https://szkj….supabase.co` ← **추가** |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ…` (service_role) ← **추가** |
| `WEBHOOK_SECRET` | (선택) 웹훅 시크릿 경로 — 보안 절 참고 |
| `REWRITE` | (선택) `off`면 재작성 끔. 보통 비워둠 |

> 어드민 프로젝트 env도 동일하게 `TOP_K=6` 권장. (어드민은 `NEXT_PUBLIC_` 접두사 변수 추가 — README 환경변수 참고)

### 2. 새 코드 배포
- `feat/supabase-admin` → main 머지하거나
- Vercel 프로젝트의 Production Branch를 `feat/supabase-admin`으로 변경
- → 자동 재배포

### 3. 변경점
- `vercel.json`: `includeFiles: data/chunks.json` 제거됨 (이제 Supabase)
- 콜백: `waitUntil`로 백그라운드 작업 안전 처리
- 검색: chunks.json 풀스캔 → Supabase hybrid_search

### 4. 카카오 웹훅
URL 그대로 유지 (Vercel 도메인 안 바뀜). 단, 새 코드가 정상 배포됐는지 확인 후.

---

## B. 어드민 배포 (새 Vercel 프로젝트)

### 1. 프로젝트 생성
1. vercel.com → Add New → Project
2. `luazencloud-design/kakao-bot` import
3. **Root Directory: `admin`** ← 필수
4. Framework: Next.js (자동)
5. Production Branch: `feat/supabase-admin`

### 2. 환경변수
| Key | Value | 비고 |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://szkj….supabase.co` | 공개 OK |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJ…` (anon) | 공개 OK |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ…` | 비공개 |
| `GEMINI_API_KEY` | `AIza…` | 비공개 |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | |
| `EMBED_MODEL` | `gemini-embedding-001` | |
| `TOP_K` | `4` | |

### 3. Deploy → 도메인 생성

### 4. Supabase Auth Redirect 추가
Supabase → Authentication → URL Configuration:
- Site URL: 어드민 Vercel 도메인
- Redirect URLs: 위 도메인 `/**` + `http://localhost:3000/**` 유지

---

## C. 콜백 모드 활성화 (5초 제한 우회) — 권장

**문제:** 카카오 동기 스킬은 **5초 안에** 답해야 한다(`skill timeout: 5sec`, 카카오 쪽 하드 제한). 그런데 RAG 파이프라인이 Gemini를 3번 순차 호출해 warm에서도 ~5초, 느린 구간엔 p95 20초·최대 27초까지 튄다 → 그 질문은 카카오가 버려서 **무응답**이 된다. Vercel Pro·호스트 변경으로는 못 푼다(카카오 제한이라).

**해법:** 콜백 모드. 봇이 5초 안엔 "답변 생성 중..." **정적 응답만** 즉시 보내고(이건 Gemini 안 거쳐 수십ms), 실제 답변은 카카오가 준 **1회용 `callbackUrl`로 1분(60초) 안에** 비동기 전달한다. 코드(`src/app.js` `handleCallback`)는 이미 `waitUntil` + 45초 데드라인 가드로 구현돼 있다 — 남은 건 카카오 설정뿐.

> 공식 근거: `callbackUrl valid time: 1min`, 1회용 (카카오 비즈니스 'AI 챗봇 콜백 개발 가이드'). 그래서 `vercel.json`의 `maxDuration`을 **60**으로 맞춰 함수가 윈도 끝까지 살아있게 했다.

### 활성화 절차
1. **권한 신청** — 챗봇 관리자센터 → **설정 → AI 챗봇 관리**에서 Callback API(콜백) 사용 권한 신청. **승인에 영업일 1~2일** 소요.
2. **블록 설정** — 승인 후, RAG 답변을 내보내는 스킬 블록 상세에서 **Callback API 설정(useCallback) 활성화** + 대기 메시지(기본응답) 작성.
3. **URL 변경** — 그 블록의 스킬 연결 URL을 동기가 아니라 콜백 엔드포인트로:
   - `https://<봇도메인>/kakao/skill/callback/<WEBHOOK_SECRET>`
   - (시크릿 미설정 시 `/kakao/skill/callback`)
4. **검증** — 카카오로 질문 후 Vercel 봇 로그에서 `[callback] callbackUrl: yes` 확인. `no`가 뜨면 오픈빌더 useCallback이 안 켜진 것 → 코드가 동기 fallback으로 돌아 다시 5초 벽.
5. **첫 응답 콜드스타트 주의** — 콜백을 켜도 *첫* 정적 응답은 5초 SLA를 받는다. keep-warm 핑을 유지해 콜드스타트가 그 5초를 위협하지 않게 한다.

> ⚠️ 잔존 한계: ① 1분 윈도를 넘기면(현재 최대 27초라 여유 있음) 유실, ② `callbackUrl`은 1회용이라 전달 자체가 네트워크 오류로 실패하면 그 답은 유실(코드가 데드라인 fallback은 보장하나 전송 실패까지는 못 막음).

---

## 플랜 고려

| 작업 | Hobby | Pro |
|---|---|---|
| 봇 동기 응답 (warm ~5초) | ⚠️ 5초 경계 — 콜백 권장 | ⚠️ 동일(카카오 제한) |
| 봇 콜백 (waitUntil, 1분 윈도) | ✅ `maxDuration:60` | ✅ |
| 문서 업로드 (배치 임베딩) | ✅ 대부분 | ✅ |
| 미디어 전사 (1~2분) | ❌ | ✅ (maxDuration=300) |

**문서·자막 위주면 Hobby로 시작. 미디어 전사·여유 필요하면 Pro.** 5초 무응답은 플랜이 아니라 **콜백 모드(위 C)**로 푼다.

## 배포 후 확인

- [ ] 봇: 카카오로 질문 → 답변 오는지
- [ ] 봇: 콜백 켠 경우 "답변 생성 중..." 후 실답변 오는지 (로그 `[callback] delivered`)
- [ ] 어드민: 로그인 → 파일 목록
- [ ] 어드민: **PDF 업로드** (unpdf로 추출 — 스캔 PDF는 실패가 정상)
- [ ] 어드민: **큰 파일 업로드**(4.5MB 초과 PPTX 등) — 서명 URL 직접 업로드가 도는지 (예전엔 413)
- [ ] 어드민: 답변 테스트 페이지

## 업로드 아키텍처 (4.5MB 한계 우회)

Vercel 서버리스 함수는 **요청 본문 4.5MB**를 넘으면 413으로 막는다. 그래서 업로드를 2단계로 분리:
1. `/admin/api/upload/sign` — 작은 JSON으로 **서명 URL** 발급 (인증·중복·형식·용량 검증만)
2. 브라우저가 **Supabase Storage에 직접** 업로드 (`uploadToSignedUrl`, Vercel 우회 → 50MB까지)
3. `/admin/api/upload` — 작은 JSON으로 처리 요청 → 서버가 Storage에서 내려받아 추출·임베딩

## 알려진 리스크

- **PDF 추출(unpdf)**: pdf-parse는 Vercel Node에서 `DOMMatrix is not defined`로 실패해 `unpdf`로 교체함(`admin/lib/ingest/extract.ts`). 스캔(이미지) PDF는 여전히 텍스트 추출 불가 — 정상.
- **PPTX 추출(officeparser)**: 네이티브 의존이 있어 Vercel 서버리스 동작은 첫 PPTX 업로드로 확인 필요. 실패 시 unpdf처럼 서버리스용 추출기로 교체.
- **영상(MP4) 미지원**: 전사 타임아웃 위험으로 제거. 자막(VTT)으로 우회.
