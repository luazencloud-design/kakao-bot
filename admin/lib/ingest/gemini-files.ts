// Gemini Files API로 오디오·영상 전사.
// 1) 파일 업로드 (resumable) → 2) (영상) ACTIVE 대기 → 3) generateContent 전사

const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';

interface UploadedFile {
  uri: string;
  mimeType: string;
  name: string;
  state: string;
}

// 파일 바이트를 Gemini Files API에 업로드, 파일 객체 반환.
async function uploadToGemini(
  buffer: Buffer,
  mimeType: string,
  displayName: string,
): Promise<UploadedFile> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');

  // 1단계: resumable 업로드 시작
  const startResp = await fetch(
    `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
    {
      method: 'POST',
      headers: {
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': String(buffer.byteLength),
        'X-Goog-Upload-Header-Content-Type': mimeType,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file: { display_name: displayName } }),
    },
  );
  if (!startResp.ok) throw new Error(`Files API 시작 실패: ${startResp.status} ${await startResp.text()}`);

  const uploadUrl = startResp.headers.get('x-goog-upload-url');
  if (!uploadUrl) throw new Error('업로드 URL을 받지 못했습니다.');

  // 2단계: 바이트 업로드 + finalize
  const uploadResp = await fetch(uploadUrl, {
    method: 'POST',
    headers: {
      'X-Goog-Upload-Offset': '0',
      'X-Goog-Upload-Command': 'upload, finalize',
      'Content-Length': String(buffer.byteLength),
    },
    body: new Uint8Array(buffer),
  });
  if (!uploadResp.ok) throw new Error(`Files API 업로드 실패: ${uploadResp.status}`);
  const data = await uploadResp.json();
  return data.file as UploadedFile;
}

// 파일이 ACTIVE 될 때까지 폴링 (영상은 처리 시간 필요).
async function waitActive(fileName: string, maxWaitMs = 120000): Promise<void> {
  const apiKey = process.env.GEMINI_API_KEY;
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
    );
    if (!resp.ok) throw new Error(`파일 상태 조회 실패: ${resp.status}`);
    const data = await resp.json();
    if (data.state === 'ACTIVE') return;
    if (data.state === 'FAILED') throw new Error('Gemini 파일 처리 실패');
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error('파일 처리 시간 초과 (2분)');
}

// 오디오/영상 → 전사 텍스트.
export async function transcribeMedia(
  buffer: Buffer,
  mimeType: string,
  filename: string,
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 누락');

  const file = await uploadToGemini(buffer, mimeType, filename);
  if (file.state !== 'ACTIVE') {
    await waitActive(file.name);
  }

  const isVideo = mimeType.startsWith('video/');
  const prompt = isVideo
    ? '이 영상의 음성을 한국어로 정확히 전사하고, 화면에 표시되는 중요한 텍스트(슬라이드·자막)도 함께 포함하세요. 말한 내용을 자연스러운 문장으로 정리하되 내용을 빠뜨리지 마세요.'
    : '이 오디오의 음성을 한국어로 정확히 전사하세요. 말한 내용을 자연스러운 문장으로 정리하되 내용을 빠뜨리지 마세요.';

  const resp = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            role: 'user',
            parts: [
              { text: prompt },
              { fileData: { mimeType: file.mimeType, fileUri: file.uri } },
            ],
          },
        ],
        generationConfig: { maxOutputTokens: 8192, temperature: 0 },
      }),
    },
  );
  if (!resp.ok) throw new Error(`전사 실패: ${resp.status} ${await resp.text()}`);
  const data = await resp.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
  if (!text.trim()) throw new Error('전사 결과가 비어 있습니다.');

  // 업로드한 파일 정리 (48시간 후 자동 삭제되지만 즉시 삭제)
  try {
    await fetch(`https://generativelanguage.googleapis.com/v1beta/${file.name}?key=${apiKey}`, {
      method: 'DELETE',
    });
  } catch {
    /* swallow */
  }

  return text;
}
