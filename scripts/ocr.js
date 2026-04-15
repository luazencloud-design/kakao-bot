// scripts/ocr.js
//
// Multi-file OCR job: walks source-files/, finds every .pdf and
// .pptx, uploads each to Gemini's Files API, and writes the
// extracted text to data/extracted/<basename>.txt as a cache.
//
// Caching: if data/extracted/<basename>.txt already exists, the
// file is skipped. Pass --force to re-OCR everything.
//
// Single-file override: if SOURCE_FILE is set in .env, only that
// file is processed (legacy behavior).
//
// Why this exists: scripts/ingest.js does direct text extraction
// (pdf-parse / officeparser), which fails on scanned PDFs and
// image-only PPTX slides because there's no embedded text layer.
// Gemini's multimodal capability "reads" the page images and gives
// us usable text.
//
// Usage:
//   npm run ocr            # process new files only (uses cache)
//   npm run ocr -- --force # re-OCR everything

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';
const SOURCE_FILE_OVERRIDE = process.env.SOURCE_FILE || process.env.PDF_PATH;

const SOURCE_DIR = path.join(ROOT, 'source-files');
const EXTRACTED_DIR = path.join(ROOT, 'data', 'extracted');

const force = process.argv.includes('--force');

// Map file extension -> Gemini-acceptable MIME type.
const MIME_BY_EXT = {
  '.pdf': 'application/pdf',
  '.pptx':
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

function die(msg) {
  console.error(`[ocr] ERROR: ${msg}`);
  process.exit(1);
}

if (!GEMINI_API_KEY) die('GEMINI_API_KEY is not set. Check your .env file.');

// ---------- Source file discovery ----------
function discoverSourceFiles() {
  // SOURCE_FILE override: legacy single-file mode
  if (SOURCE_FILE_OVERRIDE) {
    const abs = path.isAbsolute(SOURCE_FILE_OVERRIDE)
      ? SOURCE_FILE_OVERRIDE
      : path.resolve(ROOT, SOURCE_FILE_OVERRIDE);
    if (!fs.existsSync(abs)) {
      die(`SOURCE_FILE not found: ${abs}`);
    }
    return [abs];
  }

  // Default: walk source-files/
  if (!fs.existsSync(SOURCE_DIR)) {
    die(`source-files/ directory not found at ${SOURCE_DIR}`);
  }
  const entries = fs.readdirSync(SOURCE_DIR);
  const supported = entries
    .filter((name) => MIME_BY_EXT[path.extname(name).toLowerCase()])
    .sort()
    .map((name) => path.join(SOURCE_DIR, name));
  return supported;
}

// ---------- Gemini Files API: resumable upload ----------
async function uploadFile(filePath, mimeType) {
  const fileBuffer = fs.readFileSync(filePath);
  const fileSize = fileBuffer.length;
  const displayName = path.basename(filePath);

  console.log(
    `[ocr]   Uploading (${(fileSize / 1024 / 1024).toFixed(1)} MB, ${mimeType})...`,
  );

  const startResp = await fetch(
    `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: {
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': String(fileSize),
        'X-Goog-Upload-Header-Content-Type': mimeType,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file: { display_name: displayName } }),
    },
  );

  if (!startResp.ok) {
    throw new Error(
      `Upload init failed ${startResp.status}: ${await startResp.text()}`,
    );
  }

  const uploadUrl = startResp.headers.get('x-goog-upload-url');
  if (!uploadUrl) throw new Error('No upload URL in response headers');

  const uploadResp = await fetch(uploadUrl, {
    method: 'POST',
    headers: {
      'Content-Length': String(fileSize),
      'X-Goog-Upload-Offset': '0',
      'X-Goog-Upload-Command': 'upload, finalize',
    },
    body: fileBuffer,
  });

  if (!uploadResp.ok) {
    throw new Error(
      `Upload failed ${uploadResp.status}: ${await uploadResp.text()}`,
    );
  }

  return (await uploadResp.json()).file;
}

async function waitForActive(fileName) {
  const url = `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${GEMINI_API_KEY}`;
  for (let i = 0; i < 60; i++) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Poll failed ${resp.status}: ${await resp.text()}`);
    }
    const data = await resp.json();
    if (data.state === 'ACTIVE') return data;
    if (data.state === 'FAILED') throw new Error('File processing FAILED on Gemini side');
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error('File did not become ACTIVE within 2 minutes');
}

async function extractText(fileUri, mimeType) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

  const prompt = `이 문서의 모든 페이지 또는 슬라이드에 있는 텍스트를 순서대로 정확히 추출해 주세요.

규칙:
1. 각 페이지/슬라이드의 모든 텍스트(제목, 본문, 표, 목록, 이미지 속 한국어 텍스트 포함)를 누락 없이 추출하세요.
2. 표와 목록은 원본 구조를 최대한 유지하세요.
3. 각 페이지/슬라이드 시작 부분에 "=== 페이지 N ===" 헤더를 넣으세요.
4. 추출된 텍스트 외에는 다른 설명, 주석, 메타 정보를 추가하지 마세요.
5. 손상되거나 읽을 수 없는 부분은 [판독 불가]로 표시하세요.`;

  const body = JSON.stringify({
    contents: [
      {
        role: 'user',
        parts: [
          { file_data: { mime_type: mimeType, file_uri: fileUri } },
          { text: prompt },
        ],
      },
    ],
    generationConfig: {
      maxOutputTokens: 60000,
      temperature: 0,
    },
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
      lastErr = `Gemini ${resp.status}: ${errText.slice(0, 200)}`;
      if (attempt < maxAttempts) {
        const waitMs = 15000 * attempt;
        console.warn(`[ocr]   ${lastErr.slice(0, 120)}`);
        console.warn(
          `[ocr]   attempt ${attempt}/${maxAttempts}, waiting ${waitMs / 1000}s...`,
        );
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }
      throw new Error(`Gemini failed after ${maxAttempts} attempts: ${lastErr}`);
    }

    if (!resp.ok) {
      throw new Error(`Gemini API ${resp.status}: ${await resp.text()}`);
    }

    const data = await resp.json();
    const candidate = data?.candidates?.[0];
    if (!candidate) {
      throw new Error(`No candidates in response: ${JSON.stringify(data)}`);
    }

    const text = candidate?.content?.parts?.[0]?.text;
    if (!text) {
      throw new Error(
        `No text in candidate: ${JSON.stringify(candidate).slice(0, 500)}`,
      );
    }

    return { text, truncated: candidate.finishReason === 'MAX_TOKENS' };
  }

  throw new Error(`extractText: exhausted retries. Last error: ${lastErr}`);
}

// ---------- Process one source file ----------
async function ocrFile(filePath) {
  const basename = path.basename(filePath);
  const stem = path.basename(filePath, path.extname(filePath));
  const cachePath = path.join(EXTRACTED_DIR, `${stem}.txt`);

  if (!force && fs.existsSync(cachePath)) {
    const cachedSize = fs.statSync(cachePath).size;
    console.log(`[ocr] SKIP cached (${cachedSize}b): ${basename}`);
    return { status: 'skipped' };
  }

  console.log(`[ocr] Processing: ${basename}`);

  const ext = path.extname(filePath).toLowerCase();
  const mimeType = MIME_BY_EXT[ext];
  if (!mimeType) {
    console.warn(`[ocr]   Unsupported extension ${ext}, skipping`);
    return { status: 'unsupported' };
  }

  try {
    const file = await uploadFile(filePath, mimeType);
    await waitForActive(file.name);
    console.log('[ocr]   Extracting text...');
    const { text, truncated } = await extractText(file.uri, file.mimeType);

    if (truncated) {
      console.warn(
        `[ocr]   WARNING: ${basename} output truncated at max tokens. ` +
          `Consider splitting the file.`,
      );
    }

    fs.writeFileSync(cachePath, text, 'utf-8');
    console.log(`[ocr]   Saved ${text.length} chars to data/extracted/${stem}.txt`);
    return { status: 'processed' };
  } catch (err) {
    console.error(`[ocr]   FAILED for ${basename}: ${err.message}`);
    return { status: 'failed', err };
  }
}

// ---------- Main ----------
async function main() {
  const sourceFiles = discoverSourceFiles();

  if (sourceFiles.length === 0) {
    die(
      'No source files found. Add .pdf or .pptx files to source-files/ ' +
        'or set SOURCE_FILE in .env.',
    );
  }

  fs.mkdirSync(EXTRACTED_DIR, { recursive: true });

  console.log(`[ocr] Found ${sourceFiles.length} source file(s)`);
  console.log(`[ocr] Model: ${GEMINI_MODEL}`);
  console.log(`[ocr] Force re-OCR: ${force}`);
  console.log('');

  const counts = { processed: 0, skipped: 0, failed: 0, unsupported: 0 };
  for (const filePath of sourceFiles) {
    const { status } = await ocrFile(filePath);
    counts[status] = (counts[status] || 0) + 1;
  }

  console.log('');
  console.log(
    `[ocr] Done: ${counts.processed} processed, ${counts.skipped} skipped (cached), ` +
      `${counts.failed} failed, ${counts.unsupported} unsupported`,
  );

  if (counts.processed > 0 || counts.skipped > 0) {
    console.log('[ocr] Next step: run "npm run ingest"');
  }

  if (counts.failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('[ocr] FAILED:', err);
  process.exit(1);
});
