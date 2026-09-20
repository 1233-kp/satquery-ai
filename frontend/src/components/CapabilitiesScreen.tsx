import { useEffect, useState } from 'react';
import { getTools } from '../api';
import type { ToolInfo } from '../api';

export function CapabilitiesScreen() {
  const [tools, setTools] = useState<ToolInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTools()
      .then(setTools)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load capabilities'));
  }, []);

  return (
    <div className="screen capabilities-screen">
      <p className="screen-title">Capabilities</p>
      <p className="screen-subtitle">
        The specialist tools the orchestrator can select from, read live from the tool registry.
      </p>

      {error && <div className="error-banner">{error}</div>}

      {tools && (
        <div className="tool-card-grid">
          {tools.map((t) => (
            <div className="tool-card" key={t.name}>
              <div className="tool-card-header">
                <span className="tool-card-name">{t.name}</span>
                <span className={`tool-status-badge tool-status-${t.status}`}>{t.status}</span>
              </div>
              <div className="tool-card-row">
                <span className="tool-card-field-label">task</span>
                <span className="tool-card-field-value mono">{t.task}</span>
              </div>
              <div className="tool-card-row">
                <span className="tool-card-field-label">input mode</span>
                <span className="tool-card-field-value mono">{t.required_mode}</span>
              </div>
              <div className="tool-card-row">
                <span className="tool-card-field-label">engine</span>
                <span className="tool-card-field-value">{t.engine}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
