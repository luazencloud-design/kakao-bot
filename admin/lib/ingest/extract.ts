// 파일 버퍼 → 텍스트 추출. 포맷별 어댑터.
//
// 지원: .txt, .vtt, .pdf, .pptx, .hwp, .mp3, .m4a, .mp4
//   .pptx → officeparser (텍스트 직접 추출)
//   .hwp  → hwp.js (HWP 5.x 텍스트 레이어)
//   .mp3/.m4a/.mp4 → Gemini Files API 전사

import { transcribeMedia } from './gemini-files';

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
      // unpdf: 서버리스(Vercel Node)에서 동작하는 pdfjs 래퍼.
      // pdf-parse는 pdfjs가 DOMMatrix 등 브라우저 API를 요구해
      // Vercel Node 런타임에서 "DOMMatrix is not defined"로 실패함.
      const { extractText: pdfExtract, getDocumentProxy } = await import('unpdf');
      const pdf = await getDocumentProxy(new Uint8Array(buffer));
      const { text: raw } = await pdfExtract(pdf, { mergePages: true });
      const text = typeof raw === 'string' ? raw : (raw as string[]).join('\n');
      if (text.trim().length < 50) {
        throw new Error(
          '텍스트가 거의 추출되지 않았습니다. 스캔된 이미지 PDF로 보입니다.',
        );
      }
      return { text };
    }

    case 'pptx': {
      const { parseOffice } = await import('officeparser');
      const ast = await parseOffice(buffer);
      const text = typeof ast?.toText === 'function' ? ast.toText() : String(ast ?? '');
      if (text.trim().length < 30) {
        throw new Error(
          '텍스트가 거의 추출되지 않았습니다. 이미지 위주 슬라이드로 보입니다.',
        );
      }
      return { text };
    }

    case 'hwp': {
      const hwp = (await import('hwp.js')).default as {
        parse: (b: Buffer) => { sections: HwpSection[] };
      };
      const doc = hwp.parse(buffer);
      const lines: string[] = [];
      for (const section of doc.sections) {
        for (const paragraph of section.content) {
          const chars: string[] = [];
          for (const ch of paragraph.content) {
            if (ch.type !== 0) continue; // 0 = 일반 문자
            if (typeof ch.value === 'string') chars.push(ch.value);
            else if (typeof ch.value === 'number') chars.push(String.fromCharCode(ch.value));
          }
          const line = chars.join('').trim();
          if (line) lines.push(line);
        }
      }
      const text = lines.join('\n');
      if (text.trim().length < 30) {
        throw new Error('HWP에서 텍스트를 추출하지 못했습니다 (이미지 기반이거나 구버전 형식).');
      }
      return { text };
    }

    case 'mp3':
    case 'm4a': {
      const mime = ext === 'mp3' ? 'audio/mpeg' : 'audio/mp4';
      const text = await transcribeMedia(buffer, mime, filename);
      return { text };
    }

    case 'mp4': {
      const text = await transcribeMedia(buffer, 'video/mp4', filename);
      return { text };
    }

    default:
      throw new Error(`지원하지 않는 형식: .${ext}`);
  }
}

// --- hwp.js 타입 (최소) ---
interface HwpChar {
  type: number;
  value: string | number;
}
interface HwpParagraph {
  content: HwpChar[];
}
interface HwpSection {
  content: HwpParagraph[];
}

// WebVTT 자막 → 발화 텍스트만 추출 (화자 prefix·타임스탬프 제거)
const SPEAKER_PREFIX_RE = /^[가-힣A-Za-z][가-힣A-Za-z0-9\s_-]{0,24}:\s?/;

export function extractVtt(content: string): string {
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
