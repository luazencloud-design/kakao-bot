// scripts/migrate-to-supabase.mjs
//
// 기존 data/chunks.json (301청크, 20개 출처) → Supabase로 일회성 이관.
//
// 흐름:
//   1. chunks.json 로드, source별로 그룹화
//   2. source별로 documents row 생성 (status=ready)
//   3. 각 청크에 document_id 매핑해서 chunks 테이블에 일괄 INSERT
//   4. documents.chunk_count 갱신
//
// 안전장치:
//   - 기존 데이터가 있으면 abort (--force로 우회)
//   - 트랜잭션 없이 batched insert (Supabase JS 클라이언트 제약)
//   - 실패 시 documents/chunks 둘 다 cleanup
//
// 사용:
//   node scripts/migrate-to-supabase.mjs              # dry-run (검증만)
//   node scripts/migrate-to-supabase.mjs --execute    # 실제 이관
//   node scripts/migrate-to-supabase.mjs --execute --force  # 기존 데이터 무시

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const CHUNKS_PATH = path.join(ROOT, 'data', 'chunks.json');
const SOURCE_DIR = path.join(ROOT, 'source-files');

const execute = process.argv.includes('--execute');
const force = process.argv.includes('--force');

// ---------------------------------------------------------------------
// 0. 환경 검증
// ---------------------------------------------------------------------
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 누락');
  process.exit(1);
}

if (!fs.existsSync(CHUNKS_PATH)) {
  console.error(`❌ ${CHUNKS_PATH} 없음. npm run ingest 먼저 실행하세요.`);
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

// ---------------------------------------------------------------------
// 1. 카테고리 자동 분류 (파일명 패턴 기반)
// ---------------------------------------------------------------------
function inferCategory(filename) {
  // 우선순위:
  //   1. 오픈마켓 (브랜드 또는 "가입안내·판매자·계정연동" 키워드)
  //      → 노션 가입안내서처럼 도구명이 들어가도 내용이 오픈마켓 가입이면 여기로
  //   2. 사업자등록·세무
  //   3. 도구가이드 (가입안내 키워드 없는 도구 사용법)
  //   4. 강의자료 (확장자 또는 강의 관련 키워드)
  //   5. 기타

  if (
    /(11번가|지마켓|gmarket|스마트스토어|쿠팡|coupang|롯데온|이베이|ebay|큐텐|qoo10|네이버쇼핑|카페24|옥션|auction|위메프|티몬|아마존|amazon|쇼피|shopee|lazada|라자다)/i.test(filename) ||
    /(가입안내|판매자|입점|셀러|seller|상품등록|계정연동)/i.test(filename)
  ) {
    return '오픈마켓가입';
  }

  if (/(사업자|등록증|소명|세무|부가세|홈택스|hometax|세금계산서)/i.test(filename)) {
    return '사업자등록';
  }

  if (/(노션|notion|niton|웨일|whale|zoom|slack)/i.test(filename)) {
    return '도구가이드';
  }

  if (
    /(강의|강좌|주차|수업|orientation|오리엔테이션|매출|수익|recording|transcript)/i.test(filename) ||
    /\.(vtt|mp4|mp3|m4a)$/i.test(filename)
  ) {
    return '강의자료';
  }

  return '기타';
}

// ---------------------------------------------------------------------
// 2. 청크 로드 + 그룹화
// ---------------------------------------------------------------------
console.log('\n📦 chunks.json 로드 중...');
const chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, 'utf-8'));
console.log(`   총 ${chunks.length}청크`);

const bySource = {};
for (const c of chunks) {
  const source = c.source || '미상';
  if (!bySource[source]) bySource[source] = [];
  bySource[source].push(c);
}

console.log(`   출처 ${Object.keys(bySource).length}개`);

// ---------------------------------------------------------------------
// 3. 검증 출력
// ---------------------------------------------------------------------
console.log('\n📋 마이그레이션 계획:');
for (const [source, items] of Object.entries(bySource)) {
  const category = inferCategory(source);
  const ext = path.extname(source).toLowerCase();
  const srcPath = path.join(SOURCE_DIR, source);
  const exists = fs.existsSync(srcPath);
  console.log(
    `   [${category.padEnd(8)}] ${source.padEnd(50)} ${items.length}청크  ${exists ? '✅ 원본 있음' : '⚠️  원본 없음 (Storage 업로드 스킵)'}`,
  );
}

if (!execute) {
  console.log('\n💡 실제 이관하려면 --execute 추가');
  process.exit(0);
}

// ---------------------------------------------------------------------
// 4. 기존 데이터 체크
// ---------------------------------------------------------------------
console.log('\n🔍 기존 데이터 확인...');
const { count: existingDocs } = await supabase
  .from('documents')
  .select('id', { count: 'exact', head: true });
const { count: existingChunks } = await supabase
  .from('chunks')
  .select('id', { count: 'exact', head: true });

console.log(`   기존 documents: ${existingDocs ?? 0}행`);
console.log(`   기존 chunks: ${existingChunks ?? 0}행`);

if (((existingDocs ?? 0) > 0 || (existingChunks ?? 0) > 0) && !force) {
  console.error('\n❌ 이미 데이터가 있습니다. --force 옵션으로 덮어쓸지 결정하세요.');
  console.error('   주의: --force는 기존 documents·chunks 전부 삭제합니다.');
  process.exit(1);
}

if (force && ((existingDocs ?? 0) > 0 || (existingChunks ?? 0) > 0)) {
  console.log('\n🗑️  --force: 기존 데이터 삭제 중...');
  // chunks는 CASCADE로 자동 삭제됨
  const { error: delErr } = await supabase.from('documents').delete().neq('id', '00000000-0000-0000-0000-000000000000');
  if (delErr) {
    console.error('❌ 삭제 실패:', delErr.message);
    process.exit(1);
  }
  console.log('   삭제 완료');
}

// ---------------------------------------------------------------------
// 5. documents 생성 + 청크 INSERT
// ---------------------------------------------------------------------
console.log('\n🚀 이관 시작...');

const documentRows = [];
for (const [source, items] of Object.entries(bySource)) {
  const ext = path.extname(source).toLowerCase();
  const srcPath = path.join(SOURCE_DIR, source);
  const exists = fs.existsSync(srcPath);

  let size_bytes = null;
  let sha256 = null;
  let storage_path = `migrated/${source}`; // 원본 업로드 안 한 임시 경로

  if (exists) {
    const buf = fs.readFileSync(srcPath);
    size_bytes = buf.byteLength;
    sha256 = crypto.createHash('sha256').update(buf).digest('hex');
  }

  documentRows.push({
    filename: source,
    mime_type: mimeFromExt(ext),
    storage_path,
    size_bytes,
    sha256,
    category: inferCategory(source),
    status: 'ready',
    chunk_count: items.length,
  });
}

console.log(`   documents ${documentRows.length}건 INSERT...`);
const { data: insertedDocs, error: docErr } = await supabase
  .from('documents')
  .insert(documentRows)
  .select('id, filename');

if (docErr) {
  console.error('❌ documents INSERT 실패:', docErr.message);
  process.exit(1);
}

const sourceToDocId = Object.fromEntries(
  insertedDocs.map((d) => [d.filename, d.id]),
);

// chunks INSERT (배치)
const BATCH = 100;
let inserted = 0;
const allChunkRows = [];

for (const [source, items] of Object.entries(bySource)) {
  const docId = sourceToDocId[source];
  items.forEach((c, idx) => {
    allChunkRows.push({
      document_id: docId,
      chunk_index: idx,
      text: c.text,
      embedding: c.embedding,
      embed_model: 'gemini-embedding-001',
      embed_dim: 768,
      metadata: {},
    });
  });
}

console.log(`   chunks ${allChunkRows.length}건 INSERT (배치 ${BATCH})...`);
for (let i = 0; i < allChunkRows.length; i += BATCH) {
  const batch = allChunkRows.slice(i, i + BATCH);
  const { error } = await supabase.from('chunks').insert(batch);
  if (error) {
    console.error(`❌ chunks INSERT 실패 (배치 ${i}-${i + batch.length}):`, error.message);
    console.error('   롤백 중...');
    await supabase
      .from('documents')
      .delete()
      .in('id', insertedDocs.map((d) => d.id));
    process.exit(1);
  }
  inserted += batch.length;
  process.stdout.write(`\r   ${inserted}/${allChunkRows.length}`);
}
console.log();

// ---------------------------------------------------------------------
// 6. 검증
// ---------------------------------------------------------------------
console.log('\n✅ 마이그레이션 완료. 검증 중...');

const { count: finalDocs } = await supabase
  .from('documents')
  .select('id', { count: 'exact', head: true });
const { count: finalChunks } = await supabase
  .from('chunks')
  .select('id', { count: 'exact', head: true });

console.log(`   documents: ${finalDocs}건 (예상 ${documentRows.length})`);
console.log(`   chunks: ${finalChunks}건 (예상 ${allChunkRows.length})`);

if (finalDocs === documentRows.length && finalChunks === allChunkRows.length) {
  console.log('\n🎉 성공! Task 3 (Express 봇 전환) 진행 가능.');
} else {
  console.log('\n⚠️  카운트 불일치. Supabase에서 직접 확인 필요.');
}

// ---------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------
function mimeFromExt(ext) {
  return {
    '.pdf': 'application/pdf',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.hwp': 'application/x-hwp',
    '.txt': 'text/plain',
    '.vtt': 'text/vtt',
    '.mp3': 'audio/mpeg',
    '.mp4': 'video/mp4',
  }[ext] ?? 'application/octet-stream';
}
