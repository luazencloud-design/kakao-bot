export type DocStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface DocumentRow {
  id: string;
  filename: string;
  mime_type: string | null;
  storage_path: string;
  size_bytes: number | null;
  category: string | null;
  status: DocStatus;
  error_message: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export const CATEGORY_LABELS: Record<string, string> = {
  오픈마켓가입: '오픈마켓 가입',
  사업자등록: '사업자등록',
  강의자료: '강의자료',
  도구가이드: '도구 가이드',
  기타: '기타',
};

export const CATEGORY_COLORS: Record<string, string> = {
  오픈마켓가입: 'bg-purple-50 text-purple-700',
  사업자등록: 'bg-emerald-50 text-emerald-700',
  강의자료: 'bg-blue-50 text-blue-700',
  도구가이드: 'bg-amber-50 text-amber-700',
  기타: 'bg-slate-100 text-slate-600',
};

export function formatBytes(bytes: number | null): string {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const min = Math.floor(diff / 60000);
  if (min < 1) return '방금';
  if (min < 60) return `${min}분 전`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}시간 전`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day}일 전`;
  return `${Math.floor(day / 30)}개월 전`;
}
