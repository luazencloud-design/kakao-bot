# Supabase 셋업 가이드

## 1. 프로젝트 생성

1. [supabase.com](https://supabase.com) → Sign in (GitHub 연동 권장)
2. **New project** 클릭
3. 설정:
   - **Name**: `kakao-class-bot`
   - **Database Password**: 강력한 패스워드 생성 후 **반드시 따로 저장**
   - **Region**: **Northeast Asia (Tokyo)** — 한국 latency 최소
   - **Plan**: Free (시작용)
4. 프로젝트 생성 완료까지 약 2분 대기

## 2. 스키마 적용 (마이그레이션 0001~0005, 순서대로)

`migrations/` 폴더의 SQL을 **번호 순서대로 전부** 적용한다.

**방법 A — Supabase CLI (권장):**
```bash
supabase link --project-ref <ref>
supabase db push
```

**방법 B — 대시보드 SQL Editor (수동):**
1. 좌측 **SQL Editor** → **+ New query**
2. 아래를 **순서대로** 각각 복사→붙여넣기→**Run**:
   - [`0001_init.sql`](migrations/0001_init.sql) — 테이블·인덱스·GRANT·RLS·`hybrid_search`
   - [`0002_hybrid_search_korean.sql`](migrations/0002_hybrid_search_korean.sql) — sparse를 pg_trgm(한국어)로
   - [`0003_queries_resolved_at.sql`](migrations/0003_queries_resolved_at.sql) — 피드백 해결 컬럼
   - [`0004_hybrid_search_deterministic.sql`](migrations/0004_hybrid_search_deterministic.sql) — RRF 동점 id 정렬(검색 결정성)
   - [`0005_query_rewrites_cache.sql`](migrations/0005_query_rewrites_cache.sql) — 재작성 캐시 테이블
3. 각 단계 `Success` 확인

> 결과: `documents`·`chunks`·`queries`·`sessions`·`allowed_admins`·`query_rewrites` 테이블 + `hybrid_search` 함수 + GRANT/RLS가 만들어진다.

## 3. Storage 버킷 생성

1. 좌측 사이드바 **Storage** 클릭
2. **New bucket**
3. 설정:
   - **Name**: `source-files`
   - **Public bucket**: OFF (체크 해제)
   - **File size limit**: `500 MB`
   - **Allowed MIME types**: 비워둠 (모든 형식 허용, 코드에서 검증)
4. **Create bucket**

### Storage 정책 (RLS)

버킷 생성 후 **Policies** 탭에서:

```sql
-- 어드민만 업로드 가능
create policy "admin_upload_source_files"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'source-files'
  and exists (select 1 from allowed_admins where email = auth.email())
);

-- 어드민만 다운로드/조회 가능
create policy "admin_read_source_files"
on storage.objects for select
to authenticated
using (
  bucket_id = 'source-files'
  and exists (select 1 from allowed_admins where email = auth.email())
);

-- 어드민만 삭제 가능
create policy "admin_delete_source_files"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'source-files'
  and exists (select 1 from allowed_admins where email = auth.email())
);
```

## 4. 인증 (비밀번호 방식)

> 초기엔 매직 링크를 썼으나, 무료 티어 이메일 제한·일회용 토큰·RLS 문제로 **비밀번호 방식**으로 전환했다.

1. 좌측 **Authentication** → **Providers** → **Email** 켜기
2. **Confirm email**은 운영 편의상 꺼도 됨(어드민은 화이트리스트로 한 번 더 막힘)
3. **Authentication** → **Users** → **Add user**로 운영자 이메일+비밀번호 생성
   (또는 첫 로그인 흐름에서 비밀번호 설정. 비밀번호 변경은 어드민 **설정** 페이지에서 가능)
4. **Authentication** → **URL Configuration**
   - **Site URL**: 어드민 Vercel 도메인
   - **Redirect URLs**: 위 도메인 `/**` + 로컬용 `http://localhost:3000/**`

## 5. 어드민 화이트리스트 등록

SQL Editor에서:

```sql
insert into allowed_admins (email, note)
values ('your-email@example.com', '초기 운영자');
```

이 이메일로만 로그인 가능해짐.

## 6. 환경변수 복사

좌측 **Project Settings** → **API**:

```bash
# .env.local (Next.js 어드민용)
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...   # 서버 사이드만 사용, 절대 클라이언트 노출 금지

# .env (Express 봇용)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
```

## 7. 연결 확인

프로젝트 루트에서:

```bash
node -e "
import('@supabase/supabase-js').then(async ({createClient}) => {
  const s = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
  const { count, error } = await s.from('documents').select('*', { count: 'exact', head: true });
  if (error) { console.error('❌', error); process.exit(1); }
  console.log('✅ Supabase 연결 OK. documents 테이블:', count, '행');
})
"
```

`✅ Supabase 연결 OK. documents 테이블: 0 행` 나오면 성공.

## 8. 일일 자동 백업 (Pro 티어 이상)

Free 티어는 7일치 daily backup 제공.
실운영 시 **Pro 업그레이드 ($25/월)** 권장:
- PITR (Point-in-time recovery) 7일
- 30일 daily backup
- 별도 S3로 매주 dump (재해 복구)

## 트러블슈팅

### `extension "vector" does not exist`
→ Free 티어에서도 가능. **Database** → **Extensions**에서 `vector` 검색 후 활성화.

### `relation "auth.users" does not exist`
→ Supabase가 자동 생성하는 스키마. 프로젝트 생성 직후 잠시 기다리면 됨.

### 매직 링크 이메일 안 옴
→ Supabase 무료 티어는 시간당 발송 제한 있음. 자체 SMTP (SendGrid·Resend) 연결 권장.
- **Authentication** → **Settings** → **SMTP Settings**

### Storage 업로드 시 권한 에러
→ 위 Storage 정책 적용 + `allowed_admins`에 이메일 등록 확인.

## 비용 모니터링

- **Project Settings** → **Usage**: DB·Storage·Bandwidth 사용량
- 무료 한도: DB 500MB, Storage 1GB, Bandwidth 5GB/월
- 한도 80% 도달 시 Supabase가 이메일 알림
