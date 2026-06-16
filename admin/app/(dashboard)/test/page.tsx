'use client';

import { useState } from 'react';
import {
  SendHorizontal,
  MessageSquare,
  Layers,
  CheckCircle2,
  Loader2,
  Copy,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';

interface Chunk {
  id: number;
  source: string;
  text: string;
  score: number;
}
interface Result {
  answer: string;
  chunks: Chunk[];
  rewrittenQuery: string;
  timings: {
    prep: number;
    search: number;
    generate: number;
    total: number;
  };
}

const SAMPLES = [
  '사업자등록증 재발급 어떻게 받나요?',
  '스마트스토어 가입 방법',
  '11번가 가입 절차 알려주세요',
];

export default function TestPage() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  async function ask(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('/admin/api/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (res.ok && data.ok) {
        setResult(data);
      } else {
        toast.error(data.error ?? '오류가 발생했습니다.');
      }
    } catch {
      toast.error('네트워크 오류');
    }
    setLoading(false);
  }

  const rankColors = ['bg-emerald-100 text-emerald-700', 'bg-blue-100 text-blue-700', 'bg-blue-100 text-blue-700', 'bg-slate-100 text-slate-600'];

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <header>
        <h1 className="text-xl font-bold text-slate-900">답변 테스트</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          실제 챗봇이 어떻게 답하는지 확인하고, 어떤 자료를 참고했는지 추적합니다
        </p>
      </header>

      {/* 질문 입력 */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <label className="block text-sm font-medium text-slate-700 mb-2">질문</label>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
          className="flex gap-2"
        >
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="질문을 입력하세요"
            disabled={loading}
          />
          <Button type="submit" disabled={loading} className="bg-blue-600 hover:bg-blue-700 shrink-0">
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <SendHorizontal className="w-4 h-4" />
            )}
            <span className="ml-1.5 hidden sm:inline">질문하기</span>
          </Button>
        </form>
        <div className="mt-3 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-slate-500">예시:</span>
          {SAMPLES.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuestion(s);
                ask(s);
              }}
              disabled={loading}
              className="text-xs px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-700 disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-8 text-center">
          <Loader2 className="w-8 h-8 text-blue-500 mx-auto mb-3 animate-spin" />
          <div className="text-sm text-slate-500">
            임베딩 → 검색 → 답변 생성 중...
          </div>
        </div>
      )}

      {result && (
        <>
          {/* 처리 과정 */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-900">처리 과정</h2>
              <span className="text-xs text-slate-500">
                총 {(result.timings.total / 1000).toFixed(1)}초
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <Step
                label="재작성 + 임베딩 (병렬)"
                ms={result.timings.prep}
                detail={
                  result.rewrittenQuery && result.rewrittenQuery.length > 0
                    ? `"${result.rewrittenQuery.slice(0, 50)}${result.rewrittenQuery.length > 50 ? '…' : ''}"`
                    : undefined
                }
              />
              <Step
                label="하이브리드 검색 (dense + 트라이그램)"
                ms={result.timings.search}
                detail={`상위 ${result.chunks.length}개`}
              />
              <Step label="답변 생성 (Gemini)" ms={result.timings.generate} />
            </div>
          </div>

          {/* 답변 + 청크 */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* 답변 */}
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between bg-slate-50">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-slate-600" />
                  <span className="text-sm font-medium text-slate-900">챗봇 답변</span>
                </div>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(result.answer);
                    toast.success('복사됨');
                  }}
                  className="p-1.5 hover:bg-slate-200 rounded text-slate-500"
                >
                  <Copy className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="p-5">
                <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">
                  {result.answer}
                </div>
              </div>
              <div className="px-5 py-3 border-t border-slate-100 bg-slate-50 text-xs text-slate-500">
                실제 카카오톡 발송 형식과 동일 · {result.answer.length}자
              </div>
            </div>

            {/* 검색된 청크 */}
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
                <Layers className="w-4 h-4 text-slate-600" />
                <span className="text-sm font-medium text-slate-900">
                  참고한 자료 ({result.chunks.length}개)
                </span>
              </div>
              <div className="divide-y divide-slate-100 max-h-[480px] overflow-y-auto">
                {result.chunks.length === 0 ? (
                  <div className="p-6 text-center text-sm text-slate-400">
                    검색된 자료가 없습니다
                  </div>
                ) : (
                  result.chunks.map((c, i) => (
                    <div key={c.id} className="p-4 hover:bg-slate-50">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span
                            className={`text-xs font-medium px-2 py-0.5 rounded shrink-0 ${rankColors[i] ?? rankColors[3]}`}
                          >
                            {i + 1}순위
                          </span>
                          <span className="text-xs text-slate-500 truncate">{c.source}</span>
                        </div>
                        <span className="text-xs text-slate-400 font-mono shrink-0 ml-2">
                          {c.score.toFixed(4)}
                        </span>
                      </div>
                      <div className="text-xs text-slate-700 leading-relaxed line-clamp-4">
                        {c.text}
                      </div>
                      <div className="text-xs text-slate-400 mt-2">청크 #{c.id}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Step({ label, ms, detail }: { label: string; ms: number; detail?: string }) {
  return (
    <div className="flex items-center gap-3">
      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
      <span className="text-slate-700">{label}</span>
      {detail && <span className="flex-1 text-slate-400 text-xs">{detail}</span>}
      <span className={`text-xs text-slate-400 ${detail ? '' : 'flex-1 text-right'}`}>
        {ms} ms
      </span>
    </div>
  );
}
