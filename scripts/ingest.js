// scripts/ingest.js
//
// Multi-file ingestion: walks source-files/, loads text for each
// supported file, chunks each, embeds every chunk with Gemini, and
// writes a unified data/chunks.json with source metadata so answers
// can cite which document they came from.
//
// Supported formats:
//   .pdf   -> OCR cache or pdf-parse direct extraction
//   .pptx  -> OCR cache or officeparser direct extraction
//   .txt   -> direct fs.readFileSync (no extraction library needed)
//   .hwp   -> hwp.js local extraction (no API call needed)
//   .vtt   -> WebVTT parser, strips timestamps/speaker tags (no API)
//   .mp3   -> OCR cache required (run "npm run ocr" first)
//   .mp4   -> OCR cache required (run "npm run ocr" first)
//
// Single-file override: if SOURCE_FILE is set in .env, only that
// file is processed.
//
// Usage:
//   npm run ingest

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';

import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { parseOffice } from 'officeparser';
import crypto from 'node:crypto';
import { createClient } from '@supabase/supabase-js';

import { extractHwpText } from './lib/hwp-extract.js';
import { extractVttText } from './lib/vtt-extract.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';
const SOURCE_FILE_OVERRIDE = process.env.SOURCE_FILE || process.env.PDF_PATH;

const SOURCE_DIR = path.join(ROOT, 'source-files');
const EXTRACTED_DIR = path.join(ROOT, 'data', 'extracted');
const CHUNKS_PATH = path.join(ROOT, 'data', 'chunks.json');

const SUPPORTED_EXTS = new Set([
  '.pdf', '.pptx', '.txt', '.hwp', '.vtt', '.mp3', '.mp4',
]);

function die(msg) {
  console.error(`[ingest] ERROR: ${msg}`);
  process.exit(1);
}
if (!GEMINI_API_KEY) die('GEMINI_API_KEY is not set.');

// ---------- Source file discovery ----------
function discoverSourceFiles() {
  if (SOURCE_FILE_OVERRIDE) {
    const abs = path.isAbsolute(SOURCE_FILE_OVERRIDE)
      ? SOURCE_FILE_OVERRIDE
      : path.resolve(ROOT, SOURCE_FILE_OVERRIDE);
    if (!fs.existsSync(abs)) die(`SOURCE_FILE not found: ${abs}`);
    return [abs];
  }

  if (!fs.existsSync(SOURCE_DIR)) {
    die(`source-files/ directory not found at ${SOURCE_DIR}`);
  }

  const entries = fs.readdirSync(SOURCE_DIR);

  // If both foo.pptx and foo.pdf exist, the PDF takes precedence
  // (ocr.js pre-converts PPTX to PDF, so the PDF is the
  // authoritative copy). Skipping the PPTX avoids double-processing.
  const pdfStems = new Set();
  for (const name of entries) {
    if (path.extname(name).toLowerCase() === '.pdf') {
      pdfStems.add(path.basename(name, path.extname(name)));
    }
  }

  return entries
    .filter((name) => {
      if (name.startsWith('~$')) return false;
      const ext = path.extname(name).toLowerCase();
      if (!SUPPORTED_EXTS.has(ext)) return false;
      if (ext === '.pptx') {
        const stem = path.basename(name, ext);
        if (pdfStems.has(stem)) return false;
      }
      return true;
    })
    .sort()
    .map((name) => path.join(SOURCE_DIR, name));
}

// ---------- Load text for one file (OCR cache or direct extraction) ----------
async function loadTextForFile(filePath) {
  const stem = path.basename(filePath, path.extname(filePath));
  const cachePath = path.join(EXTRACTED_DIR, `${stem}.txt`);

  if (fs.existsSync(cachePath)) {
    const text = fs.readFileSync(cachePath, 'utf-8');
    return { text, source: 'ocr-cache' };
  }

  const ext = path.extname(filePath).toLowerCase();
  let text = '';

  if (ext === '.txt') {
    // Plain text: just read the file directly.
    text = fs.readFileSync(filePath, 'utf-8');
    return { text, source: 'direct' };
  } else if (ext === '.hwp') {
    // HWP 5.x: parse locally via hwp.js (no API call).
    text = extractHwpText(filePath);
  } else if (ext === '.vtt') {
    // WebVTT: strip timestamps/speaker tags, keep speech only.
    text = extractVttText(filePath);
  } else if (ext === '.mp3' || ext === '.mp4') {
    // Audio/video: no direct text extraction possible.
    // Must have been transcribed by "npm run ocr" first.
    throw new Error(
      `${ext} files require transcription. Run "npm run ocr" first to ` +
        `populate data/extracted/${stem}.txt, then re-run ingest.`,
    );
  } else if (ext === '.pdf') {
    const buf = fs.readFileSync(filePath);
    const pdf = await pdfParse(buf);
    text = pdf.text;
  } else if (ext === '.pptx') {
    // officeparser can return Buffer or non-string on some files.
    const raw = await parseOffice(filePath);
    text = typeof raw === 'string' ? raw : String(raw || '');
  } else {
    throw new Error(`Unsupported extension: ${ext}`);
  }

  if (!text || text.trim().length < 50) {
    throw new Error(
      `Direct extraction yielded only ${text.length} chars — file is likely ` +
        `scanned/image-based. Run "npm run ocr" first to populate ` +
        `data/extracted/${stem}.txt, then re-run ingest.`,
    );
  }

  return { text, source: 'direct' };
}

// ---------- Chunking ----------
function chunkText(text, targetSize = 800, overlap = 100) {
  const clean = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const paragraphs = clean
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks = [];
  let current = '';

  const flush = () => {
    if (current) {
      chunks.push(current);
      current = '';
    }
  };

  for (const para of paragraphs) {
    if ((current ? current.length + 2 : 0) + para.length <= targetSize) {
      current = current ? `${current}\n\n${para}` : para;
      continue;
    }
    flush();
    if (para.length > targetSize) {
      for (let i = 0; i < para.length; i += targetSize - overlap) {
        chunks.push(para.slice(i, i + targetSize));
      }
    } else {
      current = para;
    }
  }
  flush();

  return chunks;
}

// ---------- Embedding (single embedContent, looped) ----------
async function embedOne(text) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${EMBED_MODEL}:embedContent?key=${GEMINI_API_KEY}`;

  const body = JSON.stringify({
    model: `models/${EMBED_MODEL}`,
    content: { parts: [{ text }] },
    taskType: 'RETRIEVAL_DOCUMENT',
    outputDimensionality: 768,
  });

  const maxAttempts = 5;
  let lastErr = '';
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if ([429, 500, 502, 503, 504].includes(resp.status)) {
      const errText = await resp.text();
      lastErr = `Gemini embedding ${resp.status}: ${errText.slice(0, 200)}`;
      if (attempt < maxAttempts) {
        const waitMs = 10000 * attempt;
        console.warn(`[ingest] ${lastErr.slice(0, 120)}`);
        console.warn(
          `[ingest] attempt ${attempt}/${maxAttempts}, waiting ${waitMs / 1000}s...`,
        );
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }
      throw new Error(`Gemini embedding failed after ${maxAttempts} attempts: ${lastErr}`);
    }

    if (!resp.ok) {
      throw new Error(`Gemini embedding API ${resp.status}: ${await resp.text()}`);
    }

    const data = await resp.json();
    return data.embedding.values;
  }

  throw new Error(`embedOne: exhausted retries. Last error: ${lastErr}`);
}

// ---------- Main ----------
async function main() {
  const sourceFiles = discoverSourceFiles();
  if (sourceFiles.length === 0) {
    die(
      'No source files found. Add .pdf, .pptx, .txt, .hwp, .vtt, .mp3, or .mp4 ' +
        'files to source-files/ or set SOURCE_FILE in .env.',
    );
  }

  console.log(`[ingest] Found ${sourceFiles.length} source file(s)`);

  // Step 1: load text from each file
  const fileTexts = []; // [{ source: basename, text }]
  for (const filePath of sourceFiles) {
    const basename = path.basename(filePath);
    try {
      const { text, source } = await loadTextForFile(filePath);
      console.log(
        `[ingest] Loaded ${basename}: ${text.length} chars (${source})`,
      );
      fileTexts.push({ source: basename, text });
    } catch (err) {
      console.error(`[ingest] SKIP ${basename}: ${err.message}`);
    }
  }

  if (fileTexts.length === 0) {
    die('No source texts could be loaded. Aborting.');
  }

  // Step 2: chunk each file's text and tag with source
  console.log('');
  console.log('[ingest] Chunking...');
  const allChunks = []; // [{ source, text }]
  for (const { source, text } of fileTexts) {
    const chunks = chunkText(text);
    console.log(`[ingest]   ${source}: ${chunks.length} chunks`);
    for (const chunk of chunks) {
      allChunks.push({ source, text: chunk });
    }
  }
  console.log(`[ingest] Total: ${allChunks.length} chunks`);

  // Step 3: embed every chunk
  console.log('');
  console.log('[ingest] Embedding...');
  for (let i = 0; i < allChunks.length; i++) {
    allChunks[i].embedding = await embedOne(allChunks[i].text);
    if ((i + 1) % 10 === 0 || i === allChunks.length - 1) {
      console.log(`[ingest]   ${i + 1}/${allChunks.length}`);
    }
    if (i < allChunks.length - 1) {
      // Small delay to avoid rate-limit bursts on the free tier.
      await new Promise((r) => setTimeout(r, 150));
    }
  }

  // Step 4: Supabase에 upsert (이전 chunks.json 방식 폐기)
  console.log('');
  console.log('[ingest] Supabase에 upsert 중...');

  const SUPABASE_URL = process.env.SUPABASE_URL;
  const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    die('SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY 누락. .env 확인 필요.');
  }
  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
    auth: { persistSession: false },
  });

  // source별로 그룹화
  const bySource = {};
  for (const c of allChunks) {
    if (!bySource[c.source]) bySource[c.source] = [];
    bySource[c.source].push(c);
  }

  let totalUpserted = 0;

  for (const [filename, items] of Object.entries(bySource)) {
    const ext = path.extname(filename).toLowerCase();
    const srcPath = sourceFiles.find((p) => path.basename(p) === filename);
    let size_bytes = null;
    let sha256 = null;
    if (srcPath && fs.existsSync(srcPath)) {
      const buf = fs.readFileSync(srcPath);
      size_bytes = buf.byteLength;
      sha256 = crypto.createHash('sha256').update(buf).digest('hex');
    }

    // 기존 document 찾기 (filename 기준) — 있으면 청크만 갱신, 없으면 신규 생성
    const { data: existing } = await supabase
      .from('documents')
      .select('id')
      .eq('filename', filename)
      .maybeSingle();

    let docId;
    if (existing) {
      docId = existing.id;
      // 기존 청크 삭제 (CASCADE로 자동, 그래도 명시적으로)
      await supabase.from('chunks').delete().eq('document_id', docId);
      await supabase
        .from('documents')
        .update({
          size_bytes,
          sha256,
          status: 'ready',
          chunk_count: items.length,
          extracted_text: fileTexts.find((f) => f.source === filename)?.text ?? null,
        })
        .eq('id', docId);
      console.log(`[ingest]   ↻ ${filename} (기존 ${items.length}청크 갱신)`);
    } else {
      const { data: inserted, error: insErr } = await supabase
        .from('documents')
        .insert({
          filename,
          mime_type: mimeFromExt(ext),
          storage_path: `cli-ingest/${filename}`,
          size_bytes,
          sha256,
          category: inferCategory(filename),
          status: 'ready',
          chunk_count: items.length,
          extracted_text: fileTexts.find((f) => f.source === filename)?.text ?? null,
        })
        .select('id')
        .single();
      if (insErr) {
        console.error(`[ingest] documents INSERT 실패 (${filename}):`, insErr.message);
        continue;
      }
      docId = inserted.id;
      console.log(`[ingest]   + ${filename} (신규 ${items.length}청크)`);
    }

    // 청크 일괄 INSERT
    const chunkRows = items.map((c, idx) => ({
      document_id: docId,
      chunk_index: idx,
      text: c.text,
      embedding: c.embedding,
      embed_model: EMBED_MODEL,
      embed_dim: 768,
      metadata: {},
    }));

    const BATCH = 100;
    for (let i = 0; i < chunkRows.length; i += BATCH) {
      const batch = chunkRows.slice(i, i + BATCH);
      const { error } = await supabase.from('chunks').insert(batch);
      if (error) {
        console.error(`[ingest] chunks INSERT 실패:`, error.message);
        process.exit(1);
      }
      totalUpserted += batch.length;
    }
  }

  console.log('');
  console.log(`[ingest] ✅ 총 ${totalUpserted}청크 Supabase에 적용 완료`);
}

// 파일명으로 카테고리 추론 (migrate-to-supabase.mjs와 동일 로직)
function inferCategory(filename) {
  if (
    /(11번가|지마켓|gmarket|스마트스토어|쿠팡|coupang|롯데온|이베이|ebay|큐텐|qoo10|네이버쇼핑|카페24|옥션|auction|위메프|티몬|아마존|amazon|쇼피|shopee|lazada|라자다)/i.test(filename) ||
    /(가입안내|판매자|입점|셀러|seller|상품등록|계정연동)/i.test(filename)
  ) return '오픈마켓가입';
  if (/(사업자|등록증|소명|세무|부가세|홈택스|hometax|세금계산서)/i.test(filename)) return '사업자등록';
  if (/(노션|notion|niton|웨일|whale|zoom|slack)/i.test(filename)) return '도구가이드';
  if (
    /(강의|강좌|주차|수업|orientation|오리엔테이션|매출|수익|recording|transcript)/i.test(filename) ||
    /\.(vtt|mp4|mp3|m4a)$/i.test(filename)
  ) return '강의자료';
  return '기타';
}

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

main().catch((err) => {
  console.error('[ingest] FAILED:', err);
  process.exit(1);
});
