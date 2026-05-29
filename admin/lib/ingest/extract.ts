// 파일 버퍼 → 텍스트 추출. 포맷별 어댑터.
//
// 현재 지원: .txt, .vtt, .pdf
// TODO (Task 7 확장): .pptx, .hwp, .mp3, .mp4 (Gemini Files API OCR/전사)

export type ExtractResult = { text: string };

export async function extractText(
  filename: string,
  buffer: Buffer,
): Promise<ExtractResult> {
  const ext = filename.split('.').pop()?.toLowerCase();

  switch (ext) {
    case 'txt':
      return { text: buffer.toString('utf-8') };

    case 'vtt':
      return { text: extractVtt(buffer.toString('utf-8')) };

    case 'pdf': {
      // pdf-parse v2: PDFParse 클래스 기반.
      const { PDFParse } = await import('pdf-parse');
      const parser = new PDFParse({ data: new Uint8Array(buffer) });
      const result = await parser.getText();
      const text = result.text ?? '';
      if (text.trim().length < 50) {
        throw new Error(
          '텍스트가 거의 추출되지 않았습니다. 스캔된 이미지 PDF로 보입니다. (OCR은 추후 지원 예정)',
        );
      }
      return { text };
    }

    case 'pptx':
    case 'hwp':
    case 'mp3':
    case 'mp4':
    case 'm4a':
      throw new Error(
        `${ext.toUpperCase()} 형식은 아직 지원 예정입니다. 현재는 PDF·TXT·VTT만 가능합니다.`,
      );

    default:
      throw new Error(`지원하지 않는 형식: .${ext}`);
  }
}

// WebVTT 자막 → 발화 텍스트만 추출 (화자 prefix·타임스탬프 제거)
const SPEAKER_PREFIX_RE = /^[가-힣A-Za-z][가-힣A-Za-z0-9\s_-]{0,24}:\s?/;

function extractVtt(content: string): string {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const parts: string[] = [];

  for (const raw of lines) {
    const t = raw.trim();
    if (!t) continue;
    if (t === 'WEBVTT' || t.startsWith('WEBVTT ')) continue;
    if (/^\d+$/.test(t)) continue;
    if (t.includes('-->')) continue;
    if (t === 'NOTE' || t.startsWith('NOTE ')) continue;
    if (t === 'STYLE' || t.startsWith('STYLE ')) continue;
    if (t === 'REGION' || t.startsWith('REGION ')) continue;
    const text = t.replace(SPEAKER_PREFIX_RE, '').trim();
    if (text) parts.push(text);
  }
  return parts.join('\n');
}
