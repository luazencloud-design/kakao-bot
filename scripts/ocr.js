// scripts/ocr.js
//
// One-time job: upload a source document (PDF or PPTX) to Gemini's
// Files API, have Gemini read every page/slide (including scanned
// or image-based content), and save the extracted plain text to
// data/extracted.txt.
//
// Why this exists: scripts/ingest.js uses pdf-parse / officeparser
// for direct text extraction. That fails on scanned PDFs or
// image-only PPTX slides because there's no embedded text layer —
// only images. Gemini's multimodal capability can "read" the
// images and give us usable text.
//
// Usage:
//   npm run ocr
//   (then) npm run ingest

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

// SOURCE_FILE is the new env var; PDF_PATH is kept as a fallback
// for backward compat with older .env files.
const SOURCE_FILE_RAW = process.env.SOURCE_FILE || process.env.PDF_PATH;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';

function die(msg) {
  console.error(`[ocr] ERROR: ${msg}`);
  process.exit(1);
}

if (!SOURCE_FILE_RAW) {
  die('SOURCE_FILE (or PDF_PATH) is not set. Check your .env file.');
}
if (!GEMINI_API_KEY) {
  die('GEMINI_API_KEY is not set. Check your .env file.');
}

// Resolve relative paths against the project root.
const SOURCE_FILE = path.isAbsolute(SOURCE_FILE_RAW)
  ? SOURCE_FILE_RAW
  : path.resolve(ROOT, SOURCE_FILE_RAW);

if (!fs.existsSync(SOURCE_FILE)) {
  die(`Source file not found at: ${SOURCE_FILE}`);
}

// Map file extension -> Gemini-acceptable MIME type.
const MIME_BY_EXT = {
  '.pdf': 'application/pdf',
  '.pptx':
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};
const ext = path.extname(SOURCE_FILE).toLowerCase();
const MIME_TYPE = MIME_BY_EXT[ext];
if (!MIME_TYPE) {
  die(
    `Unsupported file extension: ${ext}. Supported: ${Object.keys(MIME_BY_EXT).join(', ')}`,
  );
}

// ---------- Upload via Gemini Files API (resumable upload) ----------
async function uploadFile(filePath, mimeType) {
  const fileBuffer = fs.readFileSync(filePath);
  const fileSize = fileBuffer.length;
  const displayName = path.basename(filePath);

  console.log(
    `[ocr] Uploading ${displayName} (${(fileSize / 1024 / 1024).toFixed(1)} MB, ${mimeType})`,
  );

  // Step 1: start a resumable upload session
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

  // Step 2: upload the bytes and finalize
  console.log('[ocr] Upload URL obtained, sending bytes...');
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

  const uploadData = await uploadResp.json();
  console.log(`[ocr] Uploaded: ${uploadData.file.uri}`);
  return uploadData.file;
}

// ---------- Poll until the file is ACTIVE ----------
async function waitForActive(fileName) {
  const url = `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${GEMINI_API_KEY}`;
  for (let i = 0; i < 60; i++) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Poll failed ${resp.status}: ${await resp.text()}`);
    }
    const data = await resp.json();
    if (data.state === 'ACTIVE') {
      console.log('[ocr] File is ACTIVE');
      return data;
    }
    if (data.state === 'FAILED') {
      throw new Error('File processing FAILED on Gemini side');
    }
    console.log(`[ocr] File state: ${data.state}, waiting 2s...`);
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error('File did not become ACTIVE within 2 minutes');
}

// ---------- Ask Gemini to read & transcribe the document ----------
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

  // Retry loop for transient errors (503 UNAVAILABLE, 429 rate limit, 5xx).
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
        console.warn(`[ocr] ${lastErr.slice(0, 120)}`);
        console.warn(
          `[ocr] attempt ${attempt}/${maxAttempts}, waiting ${waitMs / 1000}s before retry...`,
        );
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }
      throw new Error(
        `Gemini API failed after ${maxAttempts} attempts: ${lastErr}`,
      );
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

    const truncated = candidate.finishReason === 'MAX_TOKENS';
    return { text, truncated };
  }

  throw new Error(`extractText: exhausted retries. Last error: ${lastErr}`);
}

// ---------- Main ----------
async function main() {
  console.log(`[ocr] Source: ${SOURCE_FILE}`);
  console.log(`[ocr] Type:   ${MIME_TYPE}`);
  console.log(`[ocr] Model:  ${GEMINI_MODEL}`);

  const file = await uploadFile(SOURCE_FILE, MIME_TYPE);
  await waitForActive(file.name);

  console.log('[ocr] Extracting text (this may take 30-90s)...');
  const { text, truncated } = await extractText(file.uri, file.mimeType);

  if (truncated) {
    console.warn(
      '[ocr] WARNING: output was truncated at max tokens. ' +
        'The extracted text may be incomplete. ' +
        'Consider splitting the document or using gemini-2.5-pro.',
    );
  }

  console.log(`[ocr] Extracted ${text.length} characters`);

  const outDir = path.join(ROOT, 'data');
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, 'extracted.txt');
  fs.writeFileSync(outPath, text, 'utf-8');
  console.log(`[ocr] Saved to ${outPath}`);
  console.log('[ocr] Next step: run "npm run ingest"');
}

main().catch((err) => {
  console.error('[ocr] FAILED:', err);
  process.exit(1);
});
