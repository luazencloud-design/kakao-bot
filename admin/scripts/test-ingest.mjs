// 업로드 파이프라인(추출→청킹→임베딩→Supabase 삽입) 독립 검증.
// auth/HTTP 레이어 제외하고 핵심 로직만 테스트.
// 사용: node scripts/test-ingest.mjs

import { createClient } from '@supabase/supabase-js';
import { readFileSync } from 'node:fs';
import 'dotenv/config';

// .env.local 로드 (dotenv는 .env만 읽으므로 수동)
import { config } from 'dotenv';
config({ path: '.env.local' });

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
const geminiKey = process.env.GEMINI_API_KEY;
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';

console.log('env 확인:');
console.log('  SUPABASE_URL:', url ? 'OK' : '❌ 누락');
console.log('  SERVICE_ROLE_KEY:', key ? 'OK' : '❌ 누락');
console.log('  GEMINI_API_KEY:', geminiKey ? 'OK' : '❌ 누락');
if (!url || !key || !geminiKey) process.exit(1);

const supabase = createClient(url, key, { auth: { persistSession: false } });

// --- chunk ---
function chunkText(text, targetSize = 800, overlap = 100) {
  const clean = text.replace(/\r\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  const paragraphs = clean.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  let current = '';
  const flush = () => { if (current) { chunks.push(current); current = ''; } };
  for (const para of paragraphs) {
    if ((current ? current.length + 2 : 0) + para.length <= targetSize) {
      current = current ? `${current}\n\n${para}` : para;
      continue;
    }
    flush();
    if (para.length > targetSize) {
      for (let i = 0; i < para.length; i += targetSize - overlap) chunks.push(para.slice(i, i + targetSize));
    } else current = para;
  }
  flush();
  return chunks;
}

// --- embed ---
async function embed(text) {
  const u = `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${geminiKey}`;
  const resp = await fetch(u, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: `models/${EMBED_MODEL}`,
      content: { parts: [{ text }] },
      taskType: 'RETRIEVAL_DOCUMENT',
      outputDimensionality: 768,
    }),
  });
  if (!resp.ok) throw new Error(`embed ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.embedding.values;
}

async function main() {
  const sample = `[테스트 문서]

이것은 업로드 파이프라인 검증용 테스트 문서입니다.

Q: 테스트 질문은 무엇인가요?
A: 이것은 ingest 파이프라인이 정상 동작하는지 확인하는 테스트입니다.

Q: 청킹은 어떻게 되나요?
A: 빈 줄을 기준으로 문단이 분리되어 각각 청크가 됩니다.`;

  console.log('\n1) 청킹...');
  const chunks = chunkText(sample);
  console.log(`   ${chunks.length}개 청크 생성`);

  console.log('2) 임베딩...');
  const embeddings = [];
  for (const c of chunks) {
    embeddings.push(await embed(c));
    process.stdout.write('.');
  }
  console.log(` ${embeddings.length}개 (차원 ${embeddings[0].length})`);

  console.log('3) 테스트 document 생성...');
  const { data: doc, error: docErr } = await supabase
    .from('documents')
    .insert({
      filename: '__test_ingest__.txt',
      mime_type: 'text/plain',
      storage_path: 'test/__test_ingest__.txt',
      category: '기타',
      status: 'processing',
      chunk_count: 0,
    })
    .select('id')
    .single();
  if (docErr) throw new Error(`document insert: ${docErr.message}`);
  console.log(`   document id: ${doc.id}`);

  console.log('4) 청크 삽입...');
  const rows = chunks.map((text, i) => ({
    document_id: doc.id,
    chunk_index: i,
    text,
    embedding: embeddings[i],
    embed_model: EMBED_MODEL,
    embed_dim: 768,
    metadata: {},
  }));
  const { error: chErr } = await supabase.from('chunks').insert(rows);
  if (chErr) throw new Error(`chunks insert: ${chErr.message}`);

  await supabase.from('documents').update({ status: 'ready', chunk_count: chunks.length }).eq('id', doc.id);
  console.log(`   ${rows.length}개 청크 삽입 완료`);

  console.log('5) 검색 RPC 테스트 (hybrid_search)...');
  const qEmbed = await embed('테스트 질문이 뭐야?');
  const { data: results, error: rpcErr } = await supabase.rpc('hybrid_search', {
    query_embedding: qEmbed,
    query_text: '테스트 질문',
    match_count: 3,
  });
  if (rpcErr) throw new Error(`hybrid_search: ${rpcErr.message}`);
  console.log(`   ${results.length}개 결과:`);
  for (const r of results) console.log(`     - [${r.source}] ${r.chunk_text.slice(0, 40)}... (score ${r.rrf_score.toFixed(4)})`);

  console.log('6) 정리 (테스트 document 삭제)...');
  await supabase.from('documents').delete().eq('id', doc.id);
  console.log('   삭제 완료 (chunks도 CASCADE)');

  console.log('\n🎉 파이프라인 전체 정상 동작 확인.');
}

main().catch((e) => { console.error('\n❌ 실패:', e.message); process.exit(1); });
