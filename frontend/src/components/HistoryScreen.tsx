import { useEffect, useState } from 'react';
import { getAnalysisThreads } from '../api';
import type { ThreadListItem } from '../api';

interface Props {
  onOpenThread: (analysisId: string) => void;
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

export function HistoryScreen({ onOpenThread }: Props) {
  const [threads, setThreads] = useState<ThreadListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAnalysisThreads()
      .then(setThreads)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load history'));
  }, []);

  return (
    <div className="screen history-screen">
      <p className="screen-title">History</p>

      {error && <div className="error-banner">{error}</div>}

      {threads && threads.length === 0 && (
        <div className="overview-empty">
          <div className="empty-state-radar" aria-hidden="true">
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-dot" />
          </div>
          <p className="empty-state-text">No analyses yet — run one from New Analysis to see it here.</p>
        </div>
      )}

      {threads && threads.length > 0 && (
        <div className="history-list">
          {threads.map((t) => (
            <button
              key={t.analysis_id}
              className="history-thread-card"
              onClick={() => onOpenThread(t.analysis_id)}
              type="button"
            >
              <div className="history-thread-main">
                <div className="history-thread-query">&quot;{t.query}&quot;</div>
                <div className="history-thread-meta">
                  <span className="task-badge">{t.task_selected}</span>
                  <span className="history-thread-time mono">{formatTimestamp(t.created_at)}</span>
                </div>
              </div>
              {t.follow_up_count > 0 && (
                <span className="history-followup-count">
                  +{t.follow_up_count} follow-up{t.follow_up_count === 1 ? '' : 's'}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
