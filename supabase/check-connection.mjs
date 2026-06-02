// supabase/check-connection.mjs
//
// Supabase 연결·스키마 검증 스크립트.
// 사용: SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... node supabase/check-connection.mjs

import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';

const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!url || !key) {
  console.error('❌ SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 누락');
  console.error('   .env 또는 환경변수로 설정 후 다시 실행하세요.');
  process.exit(1);
}

const s = createClient(url, key);

// ---------- 진단: 어떤 키가 로드됐는지 ----------
function decodeJwtPayload(jwt) {
  try {
    const seg = jwt.split('.')[1];
    if (!seg) return null;
    const json = Buffer.from(seg.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString();
    return JSON.parse(json);
  } catch (e) {
    return null;
  }
}

const payload = decodeJwtPayload(key);
console.log(`\n🔑 키 진단:`);
console.log(`   prefix:  ${key.slice(0, 30)}...`);
console.log(`   length:  ${key.length}`);
if (payload) {
  console.log(`   role:    ${payload.role ?? '(없음)'}`);
  console.log(`   ref:     ${payload.ref ?? '(없음)'}`);
  console.log(`   iss:     ${payload.iss ?? '(없음)'}`);
  if (payload.exp) {
    const expired = payload.exp * 1000 < Date.now();
    console.log(`   exp:     ${new Date(payload.exp * 1000).toISOString()} ${expired ? '⚠️ 만료됨' : ''}`);
  }
} else {
  console.log(`   ⚠️  JWT 형식이 아님 (sb_secret_* 같은 신형식 키일 수 있음)`);
}

async function check(label, fn) {
  process.stdout.write(`  ${label}... `);
  try {
    await fn();
    console.log('✅');
    return true;
  } catch (e) {
    console.log('❌');
    console.log(`     ${e.message}`);
    return false;
  }
}

console.log(`\n🔍 Supabase 연결 검증: ${url}\n`);

let allOk = true;

allOk &= await check('documents 테이블 접근', async () => {
  const { error } = await s.from('documents').select('id', { count: 'exact', head: true });
  if (error) throw error;
});

allOk &= await check('chunks 테이블 접근', async () => {
  const { error } = await s.from('chunks').select('id', { count: 'exact', head: true });
  if (error) throw error;
});

allOk &= await check('queries 테이블 접근', async () => {
  const { error } = await s.from('queries').select('id', { count: 'exact', head: true });
  if (error) throw error;
});

allOk &= await check('allowed_admins 테이블 접근', async () => {
  const { error } = await s.from('allowed_admins').select('email', { count: 'exact', head: true });
  if (error) throw error;
});

allOk &= await check('hybrid_search RPC 존재 확인', async () => {
  // 더미 768차원 벡터로 호출만 시도 (결과는 빈 배열이어도 OK)
  const dummy = new Array(768).fill(0);
  const { error } = await s.rpc('hybrid_search', {
    query_embedding: dummy,
    query_text: 'test',
    match_count: 1,
  });
  if (error) throw error;
});

allOk &= await check('Storage 버킷 source-files 존재', async () => {
  const { data, error } = await s.storage.listBuckets();
  if (error) throw error;
  if (!data.find((b) => b.name === 'source-files')) {
    throw new Error('source-files 버킷이 없습니다. Supabase 대시보드에서 생성 필요.');
  }
});

console.log();
if (allOk) {
  console.log('🎉 모든 검사 통과. Task #2 (chunks.json 마이그레이션) 진행 가능.');
  process.exit(0);
} else {
  console.log('❗ 일부 검사 실패. supabase/README.md 트러블슈팅 참고.');
  process.exit(1);
}
