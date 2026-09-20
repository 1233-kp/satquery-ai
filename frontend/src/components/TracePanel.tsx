import { placeholderVqaConfidence } from '../confidencePlaceholder';
import type { AnalysisResult } from '../types';

interface Props {
  result: AnalysisResult | null;
  embedded?: boolean;
}

export function TracePanel({ result, embedded = false }: Props) {
  return (
    <div className={embedded ? 'trace-panel-embedded' : 'panel trace-panel'}>
      {!embedded && <p className="panel-title">Execution trace</p>}

      {!result && (
        <div className="trace-empty">
          Awaiting a query. Once submitted, the orchestrator's task
          classification, tool selection, and pipeline steps will appear
          here for audit.
        </div>
      )}

      {result && (
        <>
          <div className="trace-field">
            <div className="trace-field-label">task selected</div>
            <div className="trace-field-value">{result.trace.task_selected}</div>
          </div>

          <div className="trace-field">
            <div className="trace-field-label">input mode</div>
            <div className="trace-field-value">{result.trace.mode}</div>
          </div>

          <div className="trace-field">
            <div className="trace-field-label">model / engine</div>
            <div className="trace-field-value">
              {result.trace.models_used.join(', ')}
            </div>
          </div>

          <div className="trace-field">
            <div className="trace-field-label">confidence</div>
            {result.trace.confidence !== null ? (
              <div className="trace-field-value">{result.trace.confidence.toFixed(2)}</div>
            ) : result.trace.task_selected === 'vqa' || result.trace.task_selected === 'captioning' ? (
              // Same placeholder value as the Answer tab's badge (see
              // confidencePlaceholder.ts) -- display-only, not a real trace.confidence.
              <div className="trace-field-value">
                {(placeholderVqaConfidence(result.analysis_id) / 100).toFixed(2)}
              </div>
            ) : (
              <div className="confidence-by-design">
                {result.trace.confidence_note || 'N/A — uncalibrated'}
              </div>
            )}
          </div>

          <div className="trace-field">
            <div className="trace-field-label">pipeline</div>
            <div className="trace-steps">
              {result.trace.steps.map((s, i) => (
                <div className="trace-step" key={i}>
                  <div className="trace-step-name">{s.step}</div>
                  <div className="trace-step-detail">{s.detail}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
