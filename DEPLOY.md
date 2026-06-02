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
| `TOP_K` | `4` |
| `SUPABASE_URL` | `https://szkj….supabase.co` ← **추가** |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ…` (service_role) ← **추가** |

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

## 플랜 고려

| 작업 | Hobby (10초) | Pro (60~300초) |
|---|---|---|
| 봇 동기 응답 (~3.5초) | ✅ | ✅ |
| 봇 콜백 (waitUntil, ~3.5초) | ✅ 10초 내 | ✅ |
| 문서 업로드 (배치 임베딩) | ✅ 대부분 | ✅ |
| 미디어 전사 (1~2분) | ❌ | ✅ (maxDuration=300) |

**문서·자막 위주면 Hobby로 시작. 미디어 전사·여유 필요하면 Pro.**

## 배포 후 확인

- [ ] 봇: 카카오로 질문 → 답변 오는지
- [ ] 봇: 느린 경우 콜백으로 답 오는지 (waitUntil 동작 확인)
- [ ] 어드민: 로그인 → 파일 목록
- [ ] 어드민: **PDF 업로드** (pdf-parse 워커가 Vercel 서버리스에서 도는지 — 유일한 불확실 지점)
- [ ] 어드민: 답변 테스트 페이지

## 알려진 리스크

- **pdf-parse(pdfjs) 워커**: 로컬은 `serverExternalPackages`로 해결. Vercel 서버리스 동작은 첫 PDF 업로드로 확인.
- 문제 시 해당 형식만 CLI(`npm run ocr`)로 우회.
