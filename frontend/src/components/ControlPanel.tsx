import { useRef, useState } from 'react';
import { MODE_CONFIGS } from '../types';
import type { InputMode, UploadResponse } from '../types';

interface SlotState {
  file: File | null;
  preview: string | null;
  uploaded: UploadResponse | null;
  uploading: boolean;
}

interface Props {
  mode: InputMode;
  onModeChange: (mode: InputMode) => void;
  slots: SlotState[];
  onSlotPick: (index: number, file: File) => void;
  query: string;
  onQueryChange: (q: string) => void;
  onSubmit: () => void;
  submitting: boolean;
  canSubmit: boolean;
  error: string | null;
}

export function ControlPanel({
  mode, onModeChange, slots, onSlotPick, query, onQueryChange,
  onSubmit, submitting, canSubmit, error,
}: Props) {
  const config = MODE_CONFIGS.find((c) => c.mode === mode)!;
  const fileInputs = useRef<(HTMLInputElement | null)[]>([]);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  return (
    <div className="panel">
      <p className="panel-title">Input configuration</p>

      <div className="mode-tabs">
        {MODE_CONFIGS.map((c) => (
          <button
            key={c.mode}
            className={`mode-tab ${c.mode === mode ? 'active' : ''}`}
            onClick={() => onModeChange(c.mode)}
            type="button"
          >
            <span className="mode-tab-label">{c.label}</span>
            <span className="mode-tab-desc">{c.description}</span>
          </button>
        ))}
      </div>

      <div className="upload-slots">
        {config.slotLabels.map((label, i) => {
          const slot = slots[i];
          const modality = slot?.uploaded?.modality;
          return (
            <label
              key={label}
              className={[
                'upload-slot',
                slot?.file ? 'filled' : '',
                dragOverIndex === i ? 'drag-over' : '',
                slot?.uploading ? 'uploading' : '',
              ].filter(Boolean).join(' ')}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOverIndex(i);
              }}
              onDragLeave={() => setDragOverIndex((cur) => (cur === i ? null : cur))}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverIndex(null);
                const f = e.dataTransfer.files?.[0];
                if (f) onSlotPick(i, f);
              }}
            >
              <input
                ref={(el) => { fileInputs.current[i] = el; }}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onSlotPick(i, f);
                }}
              />
              {slot?.preview ? (
                <>
                  <img src={slot.preview} alt="" className="upload-thumb" />
                  <div className="upload-card-body">
                    <div className="upload-name">{slot.file?.name}</div>
                    <div className="upload-meta">
                      {slot.uploading
                        ? 'uploading…'
                        : slot.uploaded
                        ? `${slot.uploaded.width}×${slot.uploaded.height} · ${slot.uploaded.format}`
                        : 'pending'}
                    </div>
                  </div>
                  {slot.uploading ? (
                    <span className="upload-spinner" aria-hidden="true" />
                  ) : (
                    modality && <span className={`modality-badge modality-${modality}`}>{modality}</span>
                  )}
                </>
              ) : (
                <span className="upload-slot-label">
                  {dragOverIndex === i ? 'Drop to upload' : `${label} — click or drag to upload`}
                </span>
              )}
            </label>
          );
        })}
      </div>

      <label className="field-label" htmlFor="query">Query</label>
      <textarea
        id="query"
        className="query-input"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="Ask a question about the imagery…"
      />
      <div className="example-queries">
        {config.examples.map((ex) => (
          <button
            key={ex}
            type="button"
            className="example-chip"
            onClick={() => onQueryChange(ex)}
          >
            {ex.length > 42 ? ex.slice(0, 42) + '…' : ex}
          </button>
        ))}
      </div>

      <button
        className="submit-btn"
        onClick={onSubmit}
        disabled={!canSubmit || submitting}
        type="button"
      >
        {submitting ? 'Running analysis…' : 'Run analysis'}
      </button>

      {error && <div className="error-banner">{error}</div>}
    </div>
  );
}

export type { SlotState };
