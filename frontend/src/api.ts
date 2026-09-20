import type { AnalysisListItem, AnalysisResult, InputMode, UploadResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail;
  // FastAPI/pydantic validation errors come back as a list of
  // {loc, msg, type} objects, not a string -- stringify each one's `msg`
  // instead of letting it fall through to the default Object.toString()
  // ("[object Object]").
  if (Array.isArray(detail)) {
    const messages = detail.map((d) =>
      d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : JSON.stringify(d)
    );
    return messages.join('; ') || fallback;
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail);
  return fallback;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const body = await res.json();
      detail = formatErrorDetail(body.detail, detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: form });
  return handle<UploadResponse>(res);
}

export async function submitAnalysis(params: {
  query: string;
  mode?: InputMode;
  image_ids?: string[];
  parent_analysis_id?: string;
}): Promise<AnalysisResult> {
  const res = await fetch(`${API_URL}/analysis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return handle<AnalysisResult>(res);
}

export async function getAnalysisThread(analysisId: string): Promise<AnalysisResult[]> {
  const res = await fetch(`${API_URL}/analysis/${analysisId}/thread`);
  return handle<AnalysisResult[]>(res);
}

export interface TaskCount {
  task: string;
  count: number;
}

export interface AnalysisStats {
  total: number;
  by_task: TaskCount[];
  recent: AnalysisListItem[];
}

export async function getAnalysisStats(): Promise<AnalysisStats> {
  const res = await fetch(`${API_URL}/analysis/stats`);
  return handle<AnalysisStats>(res);
}

export interface ThreadListItem {
  analysis_id: string;
  query: string;
  task_selected: string;
  created_at: string;
  follow_up_count: number;
}

export async function getAnalysisThreads(): Promise<ThreadListItem[]> {
  const res = await fetch(`${API_URL}/analysis/threads`);
  return handle<ThreadListItem[]>(res);
}

export async function downloadAnalysisReport(analysisId: string): Promise<void> {
  const res = await fetch(`${API_URL}/analysis/${analysisId}/report`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = formatErrorDetail(body.detail, detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `satquery_report_${analysisId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function listAnalyses(): Promise<AnalysisListItem[]> {
  const res = await fetch(`${API_URL}/analysis`);
  return handle<AnalysisListItem[]>(res);
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export interface ToolInfo {
  name: string;
  task: string;
  required_mode: string;
  engine: string;
  status: string;
}

export async function getTools(): Promise<ToolInfo[]> {
  const res = await fetch(`${API_URL}/api/tools`);
  return handle<ToolInfo[]>(res);
}

// ── Auth ──────────────────────────────────────────────
// Sessions are a bearer JWT stored client-side (see backend's
// auth_service.py docstring for why), sent explicitly on the few calls
// that need it. Analysis/upload calls above are never touched by this --
// that's the point: auth is additive, not a gate.

export interface PublicUser {
  id: string;
  email: string | null;
  display_name: string | null;
  avatar_url: string | null;
  oauth_provider: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: PublicUser;
}

export function getApiUrl(): string {
  return API_URL;
}

export async function registerAccount(params: {
  email: string;
  password: string;
  display_name?: string;
}): Promise<TokenResponse> {
  const res = await fetch(`${API_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return handle<TokenResponse>(res);
}

export async function loginAccount(params: { email: string; password: string }): Promise<TokenResponse> {
  const res = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return handle<TokenResponse>(res);
}

export async function getMe(token: string): Promise<PublicUser> {
  const res = await fetch(`${API_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handle<PublicUser>(res);
}

export async function logoutAccount(token: string): Promise<void> {
  await fetch(`${API_URL}/auth/logout`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  }).catch(() => {});
}
