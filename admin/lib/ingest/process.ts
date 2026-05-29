// 한 문서를 처리: 추출 → 청킹 → 임베딩 → chunks upsert → status 갱신.
// 업로드/재처리 라우트에서 호출.

import { createServiceClient } from '@/lib/supabase/server';
import { extractText } from './extract';
import { chunkText } from './chunk';
import { embedDocument } from './embed';

const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';

export function inferCategory(filename: string): string {
  if (
    /(11번가|지마켓|gmarket|스마트스토어|쿠팡|coupang|롯데온|이베이|ebay|큐텐|qoo10|네이버쇼핑|카페24|옥션|auction|위메프|티몬|아마존|amazon|쇼피|shopee|lazada|라자다)/i.test(filename) ||
    /(가입안내|판매자|입점|셀러|seller|상품등록|계정연동)/i.test(filename)
  )
    return '오픈마켓가입';
  if (/(사업자|등록증|소명|세무|부가세|홈택스|hometax|세금계산서)/i.test(filename))
    return '사업자등록';
  if (/(노션|notion|niton|웨일|whale|zoom|slack)/i.test(filename)) return '도구가이드';
  if (
    /(강의|강좌|주차|수업|orientation|오리엔테이션|매출|수익|recording|transcript)/i.test(filename) ||
    /\.(vtt|mp4|mp3|m4a)$/i.test(filename)
  )
    return '강의자료';
  return '기타';
}

// documentId가 가리키는 문서를 처리. buffer가 주어지면 추출부터, 없으면 캐시된 extracted_text 사용.
export async function processDocument(
  documentId: string,
  opts?: { buffer?: Buffer; filename?: string },
): Promise<{ chunkCount: number }> {
  const admin = createServiceClient();

  // 문서 조회
  const { data: doc, error: docErr } = await admin
    .from('documents')
    .select('id, filename, extracted_text, storage_path')
    .eq('id', documentId)
    .single();
  if (docErr || !doc) throw new Error('문서를 찾을 수 없습니다.');

  await admin
    .from('documents')
    .update({ status: 'processing', error_message: null })
    .eq('id', documentId);

  try {
    // 1. 텍스트 확보
    let text = doc.extracted_text ?? '';
    if (opts?.buffer) {
      const { text: extracted } = await extractText(
        opts.filename ?? doc.filename,
        opts.buffer,
      );
      text = extracted;
      await admin.from('documents').update({ extracted_text: text }).eq('id', documentId);
    } else if (!text) {
      // 캐시도 없고 buffer도 없으면 Storage에서 다운로드 시도
      if (doc.storage_path && !doc.storage_path.startsWith('cli-ingest/') && !doc.storage_path.startsWith('migrated/')) {
        const { data: blob, error: dlErr } = await admin.storage
          .from('source-files')
          .download(doc.storage_path);
        if (dlErr || !blob) throw new Error('원본 파일을 가져올 수 없습니다.');
        const buf = Buffer.from(await blob.arrayBuffer());
        const { text: extracted } = await extractText(doc.filename, buf);
        text = extracted;
        await admin.from('documents').update({ extracted_text: text }).eq('id', documentId);
      } else {
        throw new Error('추출할 원본 텍스트가 없습니다.');
      }
    }

    // 2. 청킹
    const chunks = chunkText(text);
    if (chunks.length === 0) throw new Error('청크를 생성하지 못했습니다 (빈 텍스트).');

    // 3. 기존 청크 삭제 (재처리 대비)
    await admin.from('chunks').delete().eq('document_id', documentId);

    // 4. 임베딩 + 삽입 (순차, rate-limit 회피)
    const rows = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await embedDocument(chunks[i]);
      rows.push({
        document_id: documentId,
        chunk_index: i,
        text: chunks[i],
        embedding,
        embed_model: EMBED_MODEL,
        embed_dim: 768,
        metadata: {},
      });
      if (i < chunks.length - 1) await new Promise((r) => setTimeout(r, 120));
    }

    const BATCH = 100;
    for (let i = 0; i < rows.length; i += BATCH) {
      const { error } = await admin.from('chunks').insert(rows.slice(i, i + BATCH));
      if (error) throw new Error(`청크 저장 실패: ${error.message}`);
    }

    // 5. 완료
    await admin
      .from('documents')
      .update({ status: 'ready', chunk_count: chunks.length, error_message: null })
      .eq('id', documentId);

    return { chunkCount: chunks.length };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await admin
      .from('documents')
      .update({ status: 'failed', error_message: message })
      .eq('id', documentId);
    throw err;
  }
}
