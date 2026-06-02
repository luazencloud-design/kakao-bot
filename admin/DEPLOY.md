# 어드민 Vercel 배포 가이드

봇은 Fly.io(콜백 모드 때문), **어드민은 Vercel**(Next.js 최적)에 배포한다.
둘 다 같은 Supabase를 바라본다.

```
[운영자] ──브라우저──> Vercel (어드민, Next.js)  ─┐
[수강생] ──카카오────> Fly.io (봇, Express)      ─┼─> Supabase (공용 데이터)
                                                  ─┘
```

## 사전 준비

- GitHub: `luazencloud-design/kakao-bot` (현재 `feat/supabase-admin` 브랜치)
- Supabase 프로젝트 (이미 셋업됨)
- Vercel 계정

## 1. Vercel 프로젝트 생성

> ⚠️ 기존 봇용 Vercel 프로젝트와 **별개**로 새 프로젝트를 만든다.
> (봇은 Fly.io로 갈 거라 기존 Vercel 프로젝트는 나중에 삭제)

1. [vercel.com](https://vercel.com) → **Add New → Project**
2. `luazencloud-design/kakao-bot` 레포 Import
3. **중요 — Root Directory 설정**: `admin` 으로 지정
   (레포 루트는 봇이고, 어드민은 `admin/` 하위 폴더)
4. **Framework Preset**: Next.js (자동 감지)
5. **Production Branch**: 일단 `feat/supabase-admin`
   (나중에 main 머지하면 main으로 변경)

## 2. 환경변수 설정

Vercel 프로젝트 → **Settings → Environment Variables** 에 추가:

| Key | Value | 비고 |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://szkj….supabase.co` | 공개 OK |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJ…` (anon) | 공개 OK |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ…` (service_role) | **비공개** |
| `GEMINI_API_KEY` | `AIza…` | **비공개** |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | |
| `EMBED_MODEL` | `gemini-embedding-001` | |
| `TOP_K` | `4` | |

→ `admin/.env.local`의 값과 동일하게.

## 3. 배포

**Deploy** 클릭. 첫 빌드 약 2~3분.

배포 후 도메인 생김: `https://kakao-bot-admin-xxxx.vercel.app`

## 4. Supabase Auth Redirect URL 추가

Supabase 대시보드 → **Authentication → URL Configuration**:
- **Site URL**: Vercel 도메인 (`https://kakao-bot-admin-xxxx.vercel.app`)
- **Redirect URLs**: 위 도메인 + `/**` 추가
- 로컬 개발용 `http://localhost:3000/**`도 유지

## 5. 확인

1. `https://<도메인>/login` 접속
2. `luazen.cloud@gmail.com` / 비밀번호 로그인
3. `/files`에서 자료 목록 보이는지
4. 파일 업로드 테스트 (PDF·TXT)

## 플랜 고려사항

| 작업 | Hobby (무료, 10초) | Pro ($20/월, 60~300초) |
|---|---|---|
| 문서 업로드 (PDF·PPTX·HWP·TXT·VTT) | ✅ 배치 임베딩으로 대부분 통과 | ✅ |
| 대형 문서 (150청크+) | ⚠️ 10초 빠듯 | ✅ |
| 짧은 영상·오디오 전사 (1~2분) | ❌ 10초 초과 | ✅ (maxDuration=300) |

**권장: 문서 위주면 Hobby로 시작, 미디어 전사 필요하면 Pro.**
긴 영상은 어차피 자막(VTT)으로 처리 → Hobby로도 충분할 가능성 큼.

## 알려진 리스크 (배포 후 확인)

- **pdf-parse(pdfjs) 워커**: 로컬에선 `serverExternalPackages`로 해결. Vercel 서버리스에서도 동작하는지 첫 PDF 업로드로 확인 필요.
- **officeparser/hwp.js**: 순수 JS라 대체로 OK. PPTX·HWP 첫 업로드로 확인.
- 문제 생기면 해당 형식만 CLI(`npm run ocr`)로 우회 가능.

## 자동 배포

이후 `feat/supabase-admin` 브랜치에 push하면 Vercel이 자동 재배포.
(데이터는 Supabase에 있으므로 재배포해도 자료 안 사라짐)
