import { useEffect, useState } from 'react';
import { getAnalysisStats } from '../api';
import type { AnalysisStats } from '../api';
import { MODE_CONFIGS } from '../types';
import type { InputMode } from '../types';

interface Props {
  onOpenThread: (analysisId: string) => void;
  onQuickStart: (mode: InputMode) => void;
}

const DONUT_COLORS = [
  'var(--cyan)',
  'var(--amber)',
  'rgba(94, 234, 212, 0.55)',
  'rgba(240, 180, 41, 0.55)',
  'rgba(94, 234, 212, 0.3)',
  'rgba(240, 180, 41, 0.3)',
];

function formatTaskLabel(task: string): string {
  return task.replace(/_/g, ' ');
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

function Donut({ byTask, total }: { byTask: AnalysisStats['by_task']; total: number }) {
  const size = 128;
  const stroke = 18;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  let offsetSoFar = 0;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Task type breakdown">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
      {byTask.map((entry, i) => {
        const fraction = total > 0 ? entry.count / total : 0;
        const segLength = fraction * circumference;
        const el = (
          <circle
            key={entry.task}
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={DONUT_COLORS[i % DONUT_COLORS.length]}
            strokeWidth={stroke}
            strokeDasharray={`${segLength} ${circumference - segLength}`}
            strokeDashoffset={-offsetSoFar}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        );
        offsetSoFar += segLength;
        return el;
      })}
      <text x="50%" y="48%" textAnchor="middle" className="donut-center-value" dy="0.1em">
        {total}
      </text>
      <text x="50%" y="64%" textAnchor="middle" className="donut-center-label">
        total
      </text>
    </svg>
  );
}

export function OverviewScreen({ onOpenThread, onQuickStart }: Props) {
  const [stats, setStats] = useState<AnalysisStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAnalysisStats()
      .then(setStats)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load stats'));
  }, []);

  const hasHistory = !!stats && stats.total > 0;

  return (
    <div className="screen overview-screen">
      <p className="screen-title">Overview</p>

      {error && <div className="error-banner">{error}</div>}

      {stats && !hasHistory && (
        <div className="overview-empty">
          <div className="empty-state-radar" aria-hidden="true">
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-dot" />
          </div>
          <p className="empty-state-text">No analyses yet — start one below.</p>
        </div>
      )}

      {hasHistory && stats && (
        <div className="overview-stats-block">
          <div className="overview-stats-row">
            <div className="overview-donut-wrap">
              <Donut byTask={stats.by_task} total={stats.total} />
            </div>
            <div className="overview-stat-list">
              <div className="overview-stat-cell overview-stat-cell-total">
                <div className="overview-stat-value mono">{stats.total}</div>
                <div className="overview-stat-label">Total analyses</div>
              </div>
              {stats.by_task.map((t, i) => (
                <div className="overview-stat-cell" key={t.task}>
                  <span className="overview-stat-dot" style={{ background: DONUT_COLORS[i % DONUT_COLORS.length] }} />
                  <div className="overview-stat-value mono">{t.count}</div>
                  <div className="overview-stat-label">{formatTaskLabel(t.task)}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="overview-recent">
            <p className="panel-title">Recent analyses</p>
            <div className="overview-recent-grid">
              {stats.recent.map((r) => (
                <button
                  key={r.analysis_id}
                  className="overview-recent-card"
                  onClick={() => onOpenThread(r.analysis_id)}
                  type="button"
                >
                  <div className="overview-recent-query">&quot;{r.query}&quot;</div>
                  <div className="overview-recent-meta">
                    <span className="task-badge">{r.task_selected}</span>
                    <span className="overview-recent-time mono">{formatTimestamp(r.created_at)}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="overview-quickstart">
        <p className="panel-title">Start a new analysis</p>
        <div className="overview-quickstart-grid">
          {MODE_CONFIGS.map((c) => (
            <button
              key={c.mode}
              className="overview-quickstart-tile"
              onClick={() => onQuickStart(c.mode)}
              type="button"
            >
              <div className="overview-quickstart-label">{c.label}</div>
              <div className="overview-quickstart-desc">{c.description}</div>
              <span className="overview-quickstart-arrow" aria-hidden="true">→</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
