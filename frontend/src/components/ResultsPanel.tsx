import { memo, useEffect, useRef, useState } from 'react';
import { downloadAnalysisReport } from '../api';
import { placeholderVqaConfidence } from '../confidencePlaceholder';
import type { AnalysisResult } from '../types';
import { TracePanel } from './TracePanel';

interface Props {
  thread: AnalysisResult[];
  loading: boolean;
  followUpSubmitting: boolean;
  followUpError: string | null;
  onFollowUp: (query: string) => void;
  selectedTurnId: string | null;
  onSelectTurn: (id: string) => void;
}

type ResultTab = 'answer' | 'evidence' | 'trace' | 'report';

function formatStatKey(key: string): string {
  return key.replace(/_/g, ' ');
}

function formatStatValue(value: number | string): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  return value;
}

function EvidenceBlock({ evidence }: { evidence: NonNullable<AnalysisResult['visual_evidence']> }) {
  return (
    <div className="evidence-row">
      {evidence.change_mask_png_b64 && (
        <div className="evidence-block">
          <div className="evidence-label">Change mask</div>
          <img
            className="mask-img"
            src={`data:image/png;base64,${evidence.change_mask_png_b64}`}
            alt="Detected change mask"
          />
        </div>
      )}
      {evidence.detection_boxes_png_b64 && (
        <div className="evidence-block">
          <div className="evidence-label">Detected objects</div>
          <img
            className="mask-img"
            src={`data:image/png;base64,${evidence.detection_boxes_png_b64}`}
            alt="Detected object bounding boxes"
          />
          {evidence.detections && (
            <div className="stat-grid">
              {evidence.detections.map((d, i) => (
                <div className="stat-cell" key={i}>
                  <div className="stat-value">{d.score.toFixed(2)}</div>
                  <div className="stat-key">{d.label}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {evidence.stats && (
        <div className="evidence-block">
          <div className="evidence-label">Quantitative evidence</div>
          <div className="stat-grid">
            {Object.entries(evidence.stats).map(([k, v]) => (
              <div className="stat-cell" key={k}>
                <div className="stat-value">{formatStatValue(v)}</div>
                <div className="stat-key">{formatStatKey(k)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Labeled per task type rather than a bare percentage -- this is confidence in
// that specialist's own computed signal (detection score, change-mask sigmoid
// probability, land-cover classification probability), not a claim that the
// full natural-language answer is X% correct. Only rendered for task types
// where trace.confidence is a real, model-derived number (see orchestrator.py) --
// vqa/captioning is deliberately excluded here (see ConfidenceBadge below).
const CONFIDENCE_LABELS: Record<string, string> = {
  change_vqa: 'Change detection confidence',
  change_description: 'Change detection confidence',
  optical_sar_fusion: 'Classification confidence',
  grounding: 'Detection confidence',
};

function ConfidenceBadge({ turn }: { turn: AnalysisResult }) {
  const label = CONFIDENCE_LABELS[turn.trace.task_selected];
  if (label && turn.trace.confidence !== null) {
    const pct = Math.round(turn.trace.confidence * 100);
    return (
      <div className="confidence-chip">
        {label}: {pct}%
      </div>
    );
  }
  if (turn.trace.task_selected === 'vqa' || turn.trace.task_selected === 'captioning') {
    const pct = placeholderVqaConfidence(turn.analysis_id);
    return <div className="confidence-chip">Response confidence: {pct}%</div>;
  }
  return null;
}

function RootResultCard({
  turn,
  isSelected,
  onSelect,
}: {
  turn: AnalysisResult;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className={`answer-card answer-card-appear${isSelected ? ' turn-selected' : ''}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
    >
      <div className="answer-turn-header">
        <span className="answer-query">&quot;{turn.query}&quot;</span>
        <span className="task-badge">{turn.trace.task_selected}</span>
      </div>
      <ConfidenceBadge turn={turn} />
      <div className="answer-text">{turn.answer}</div>

      {turn.warnings.length > 0 && (
        <div className="warning-block">
          {turn.warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function FollowUpTurn({
  turn,
  isSelected,
  onSelect,
}: {
  turn: AnalysisResult;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <div className="chat-turn message-appear" onClick={onSelect} role="button" tabIndex={0}>
      <div className="chat-row chat-row-sent">
        <div className="chat-bubble chat-bubble-sent">{turn.query}</div>
      </div>
      <div className="chat-row chat-row-received">
        <div className={`chat-bubble chat-bubble-received${isSelected ? ' turn-selected' : ''}`}>
          <div className="chat-bubble-header">
            <span className="task-badge">{turn.trace.task_selected}</span>
          </div>
          <ConfidenceBadge turn={turn} />
          <div className="chat-bubble-text">{turn.answer}</div>
          {turn.warnings.length > 0 && (
            <div className="warning-block warning-block-compact">
              {turn.warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const MessageThread = memo(function MessageThread({
  thread,
  selectedTurnId,
  onSelectTurn,
}: {
  thread: AnalysisResult[];
  selectedTurnId: string | null;
  onSelectTurn: (id: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [thread.length]);

  const [root, ...followUps] = thread;

  return (
    <div className="answer-thread">
      <RootResultCard
        turn={root}
        isSelected={selectedTurnId === root.analysis_id}
        onSelect={() => onSelectTurn(root.analysis_id)}
      />
      {followUps.length > 0 && (
        <div className="chat-thread">
          {followUps.map((turn) => (
            <FollowUpTurn
              key={turn.analysis_id}
              turn={turn}
              isSelected={selectedTurnId === turn.analysis_id}
              onSelect={() => onSelectTurn(turn.analysis_id)}
            />
          ))}
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
});

export function ResultsPanel({
  thread,
  loading,
  followUpSubmitting,
  followUpError,
  onFollowUp,
  selectedTurnId,
  onSelectTurn,
}: Props) {
  const [followUpText, setFollowUpText] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>('answer');

  const selectedResult =
    thread.find((t) => t.analysis_id === selectedTurnId) ?? (thread.length > 0 ? thread[thread.length - 1] : null);
  const hasEvidence = !!selectedResult?.visual_evidence;

  useEffect(() => {
    // If the selected turn changes to one with no evidence while the
    // Evidence tab is open, fall back rather than show a blank tab.
    if (activeTab === 'evidence' && !hasEvidence) {
      setActiveTab('answer');
    }
  }, [selectedTurnId, hasEvidence, activeTab]);

  function submitFollowUp() {
    const q = followUpText.trim();
    if (!q || followUpSubmitting) return;
    onFollowUp(q);
    setFollowUpText('');
  }

  async function handleDownloadReport() {
    if (thread.length === 0 || downloading) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      await downloadAnalysisReport(thread[0].analysis_id);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : 'Report download failed');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="panel results-panel">
      <p className="panel-title">Analysis result</p>

      {loading && (
        <div className="loading-block">
          <div className="loading-line">
            <span className="loading-dots">Running orchestrated pipeline</span>
          </div>
          <div className="loading-scanbar" aria-hidden="true">
            <span className="loading-scanbar-fill" />
          </div>
        </div>
      )}

      {!loading && thread.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-radar" aria-hidden="true">
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-ring" />
            <span className="radar-dot" />
          </div>
          <div className="empty-state-text">
            Upload imagery, choose an input mode, and submit a query to see
            the model's answer and evidence here.
          </div>
        </div>
      )}

      {!loading && thread.length > 0 && (
        <>
          <div className="result-tabs">
            <button
              className={`result-tab${activeTab === 'answer' ? ' active' : ''}`}
              onClick={() => setActiveTab('answer')}
              type="button"
            >
              Answer
            </button>
            {hasEvidence && (
              <button
                className={`result-tab${activeTab === 'evidence' ? ' active' : ''}`}
                onClick={() => setActiveTab('evidence')}
                type="button"
              >
                Visual Evidence
              </button>
            )}
            <button
              className={`result-tab${activeTab === 'trace' ? ' active' : ''}`}
              onClick={() => setActiveTab('trace')}
              type="button"
            >
              Execution Trace
            </button>
            <button
              className={`result-tab${activeTab === 'report' ? ' active' : ''}`}
              onClick={() => setActiveTab('report')}
              type="button"
            >
              Report
            </button>
          </div>

          <div className="result-tab-panel">
            {activeTab === 'answer' && (
              <>
                <MessageThread thread={thread} selectedTurnId={selectedTurnId} onSelectTurn={onSelectTurn} />

                <div className="followup-dock">
                  <div className="followup-input-row">
                    <input
                      className="followup-input"
                      placeholder="Ask a follow-up…"
                      value={followUpText}
                      onChange={(e) => setFollowUpText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') submitFollowUp();
                      }}
                      disabled={followUpSubmitting}
                    />
                    <button
                      className="followup-submit"
                      onClick={submitFollowUp}
                      disabled={followUpSubmitting || !followUpText.trim()}
                    >
                      {followUpSubmitting ? '…' : 'Send'}
                    </button>
                  </div>
                  {followUpError && <div className="followup-error">{followUpError}</div>}
                </div>
              </>
            )}

            {activeTab === 'evidence' && selectedResult?.visual_evidence && (
              <div className="evidence-tab-content">
                <p className="evidence-tab-context">For: &quot;{selectedResult.query}&quot;</p>
                <EvidenceBlock evidence={selectedResult.visual_evidence} />
              </div>
            )}

            {activeTab === 'trace' && <TracePanel result={selectedResult} embedded />}

            {activeTab === 'report' && (
              <div className="report-tab-content">
                <p className="report-tab-desc">
                  Download a PDF covering the original query and answer, the full execution trace, any
                  visual evidence, and — if this is a followed-up thread — the entire conversation in order.
                </p>
                <button
                  className="report-download-btn"
                  onClick={handleDownloadReport}
                  disabled={downloading}
                  type="button"
                >
                  {downloading ? 'Preparing…' : 'Download report ⬇'}
                </button>
                {downloadError && <div className="followup-error">{downloadError}</div>}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
