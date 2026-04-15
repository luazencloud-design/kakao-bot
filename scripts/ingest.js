// scripts/ingest.js
//
// One-time job: get the source text (preferring OCR output if
// present, falling back to direct extraction from the source file),
// split it into chunks, embed each chunk with Gemini, and save the
// result to data/chunks.json.
//
// Supported source formats: .pdf, .pptx
//
// Usage:
//   npm run ingest
//
// If the source is scanned (image-based) and direct extraction
// returns very little text, run "npm run ocr" first to produce
// data/extracted.txt, then re-run ingest.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';

// pdf-parse's index.js contains debug code that tries to open a test
// file at import time. Importing the library file directly avoids it.
import pdfParse from 'pdf-parse/lib/pdf-parse.js';

// officeparser handles .pptx (and other office formats) via xml2js +
// unzipper. We only use it for PPTX here because its PDF code path
// is broken on Windows + ESM (uses pdf.js with a buggy worker setup).
import { parseOffice } from 'officeparser';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

// SOURCE_FILE is the new env var; PDF_PATH is kept as a fallback.
const SOURCE_FILE_RAW = process.env.SOURCE_FILE || process.env.PDF_PATH;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';

// ---------- Validate env ----------
function die(msg) {
  console.error(`[ingest] ERROR: ${msg}`);
  process.exit(1);
}
if (!GEMINI_API_KEY) die('GEMINI_API_KEY is not set.');

// ---------- Source text loading ----------
// Prefers OCR output (data/extracted.txt) over direct extraction.
async function loadSourceText() {
  const extractedPath = path.join(ROOT, 'data', 'extracted.txt');

  if (fs.existsSync(extractedPath)) {
    const text = fs.readFileSync(extractedPath, 'utf-8');
    console.log(
      `[ingest] Using OCR text from ${extractedPath} (${text.length} chars)`,
    );
    return text;
  }

  if (!SOURCE_FILE_RAW) {
    die(
      'No OCR text file found and SOURCE_FILE is not set. ' +
        'Either set SOURCE_FILE in .env or run "npm run ocr" first.',
    );
  }

  const SOURCE_FILE = path.isAbsolute(SOURCE_FILE_RAW)
    ? SOURCE_FILE_RAW
    : path.resolve(ROOT, SOURCE_FILE_RAW);

  if (!fs.existsSync(SOURCE_FILE)) {
    die(`No OCR text file found and source file not found at: ${SOURCE_FILE}`);
  }

  const ext = path.extname(SOURCE_FILE).toLowerCase();
  console.log(`[ingest] Reading source directly: ${SOURCE_FILE} (${ext})`);

  let text = '';
  if (ext === '.pdf') {
    const dataBuffer = fs.readFileSync(SOURCE_FILE);
    const pdf = await pdfParse(dataBuffer);
    console.log(`[ingest] Pages: ${pdf.numpages}, Chars: ${pdf.text.length}`);
    text = pdf.text;
  } else if (ext === '.pptx') {
    text = await parseOffice(SOURCE_FILE);
    console.log(`[ingest] Chars: ${text.length}`);
  } else {
    die(`Unsupported file extension: ${ext}. Supported: .pdf, .pptx`);
  }

  if (!text || text.trim().length < 50) {
    die(
      `Very little text extracted from ${SOURCE_FILE}. ` +
        'The file may be scanned or image-based. ' +
        'Run "npm run ocr" first to OCR with Gemini, then re-run ingest.',
    );
  }

  return text;
}

// ---------- Chunking ----------
// Split by paragraphs, then merge adjacent paragraphs up to targetSize.
// If a single paragraph is bigger than targetSize, hard-split it with overlap.
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

// ---------- Embedding (Gemini single embedContent, looped) ----------
// gemini-embedding-001 does not support synchronous batchEmbedContents,
// only single embedContent calls. We loop over them sequentially.
// Fine for a few hundred chunks; slow for thousands.
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

async function embedBatch(texts) {
  const results = [];
  for (const text of texts) {
    const vector = await embedOne(text);
    results.push(vector);
    // Small delay to avoid rate-limit bursts on the free tier.
    await new Promise((r) => setTimeout(r, 150));
  }
  return results;
}

// ---------- Main ----------
async function main() {
  const sourceText = await loadSourceText();

  console.log('[ingest] Chunking...');
  const chunks = chunkText(sourceText);
  console.log(`[ingest] Produced ${chunks.length} chunks`);

  console.log('[ingest] Embedding...');
  const BATCH = 32;
  const all = [];
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const vectors = await embedBatch(batch);
    all.push(...vectors);
    console.log(`[ingest]   ${Math.min(i + BATCH, chunks.length)}/${chunks.length}`);
  }

  const knowledge = chunks.map((text, i) => ({
    id: i,
    text,
    embedding: all[i],
  }));

  const outDir = path.join(ROOT, 'data');
  const outPath = path.join(outDir, 'chunks.json');
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(knowledge));
  console.log(`[ingest] Saved ${knowledge.length} chunks to ${outPath}`);
}

main().catch((err) => {
  console.error('[ingest] FAILED:', err);
  process.exit(1);
});
