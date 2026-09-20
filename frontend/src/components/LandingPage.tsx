import { useRef } from 'react';
import { useAuth } from '../AuthContext';
import { AuthNav } from './AuthNav';
import { AuthPanel } from './AuthPanel';
import earthImage from '../assets/earth-blackmarble.jpg';
import satelliteImage from '../assets/satellite.jpg';
import exampleBefore from '../assets/example-before.jpg';
import exampleAfter from '../assets/example-after.jpg';

interface Capability {
  label: string;
  detail: string;
}

const CAPABILITIES: Capability[] = [
  {
    label: 'Single-image VQA',
    detail: 'Visual question answering and captioning on individual scenes (SmolVLM + LoRA, BigEarthNet-adapted).',
  },
  {
    label: 'Bi-temporal change detection',
    detail: 'Siamese U-Net trained on LEVIR-CD locates and quantifies change between two dated captures.',
  },
  {
    label: 'Optical + SAR fusion',
    detail: 'Dual-branch gated fusion network reasons jointly over optical and radar imagery of the same scene.',
  },
  {
    label: 'Agentic orchestration',
    detail: 'Every query produces a full, auditable execution trace — task routing, tool selection, parameters.',
  },
];

// Every figure here traces to a real artifact in the repo, not a rounded
// marketing number:
//  - 3 checkpoints under backend/checkpoints/ (vqa_lora, change_unet, fusion_net)
//  - training_log.json in change_unet/fusion_net + vqa_lora's own training log
//    confirm BigEarthNet + LEVIR-CD as the actual training sources
//  - rsvqa_eval_results.json (RSVQA-LR) and the OSCD generalization test are
//    evaluation-only, kept as a separate stat rather than folded into
//    "training data" so the claim stays precise
//  - 6 is TaskType's literal union in backend/app/schemas.py
const STATS = [
  { value: '3', label: 'Trained models', detail: 'VQA · change detection · optical+SAR fusion' },
  { value: '2', label: 'Training datasets', detail: 'BigEarthNet · LEVIR-CD' },
  { value: '2', label: 'Evaluation benchmarks', detail: 'RSVQA-LR · OSCD' },
  { value: '6', label: 'Task types', detail: 'vqa · captioning · grounding · change ×2 · fusion' },
];

const CHIP_ICONS: Record<string, JSX.Element> = {
  vqa: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.3" />
      <path d="M6 6.2c0-1.1.9-1.9 2-1.9s2 .8 2 1.8c0 1.4-1.7 1.4-1.9 2.7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="8.05" cy="11.1" r="0.75" fill="currentColor" />
    </svg>
  ),
  change: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="2" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.3" />
      <rect x="7" y="6" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  ),
  fusion: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="6" cy="8" r="4.3" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="10" cy="8" r="4.3" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  ),
  orchestration: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="3" r="1.4" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="3.5" cy="12.5" r="1.4" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="12.5" cy="12.5" r="1.4" stroke="currentColor" strokeWidth="1.2" />
      <path d="M8 4.4V8m0 0L4 11.2M8 8l4 3.2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  ),
};

const CHIPS = [
  { key: 'vqa', label: 'Visual Question Answering' },
  { key: 'change', label: 'Change Detection' },
  { key: 'fusion', label: 'Optical + SAR Fusion' },
  { key: 'orchestration', label: 'Agentic Orchestration' },
];

const DATA_POINTS = [
  { top: '14%', left: '30%', delay: '0s' },
  { top: '62%', left: '8%', delay: '1.3s' },
  { top: '78%', left: '38%', delay: '2.4s' },
  { top: '10%', left: '52%', delay: '0.7s' },
];

interface Props {
  onLaunch: () => void;
}

export function LandingPage({ onLaunch }: Props) {
  const { user } = useAuth();
  const capabilitiesRef = useRef<HTMLDivElement | null>(null);

  return (
    <div className="landing">
      <svg className="landing-topo" aria-hidden="true" preserveAspectRatio="xMidYMid slice">
        <defs>
          <pattern id="landing-contour" width="260" height="170" patternUnits="userSpaceOnUse">
            <path d="M-10,30 Q55,-10 130,30 T270,30" />
            <path d="M-10,75 Q55,35 130,75 T270,75" />
            <path d="M-10,120 Q55,80 130,120 T270,120" />
            <path d="M-10,155 Q55,125 130,155 T270,155" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#landing-contour)" />
      </svg>

      <header className="landing-topbar">
        <div className="landing-brand">
          <span className="landing-brand-mark">SQ</span>
          <span className="landing-brand-name">SatQuery AI</span>
        </div>
        <AuthNav variant="dark" />
      </header>

      <main className="landing-hero">
        <div className="landing-orbital-scene" aria-hidden="true">
          <svg className="landing-orbits-svg" viewBox="0 0 900 900">
            <ellipse cx="450" cy="450" rx="420" ry="200" transform="rotate(-16 450 450)" />
            <ellipse cx="450" cy="450" rx="330" ry="155" transform="rotate(10 450 450)" />
            <ellipse cx="450" cy="450" rx="240" ry="112" transform="rotate(-5 450 450)" />
          </svg>
          <div className="landing-earth-disc">
            <img className="landing-earth-photo" src={earthImage} alt="Earth at night, city lights visible" />
            <div className="landing-radar-sweep" />
          </div>
        </div>

        <span className="landing-imagery-credit">Imagery: NASA</span>

        <div className="landing-downlink-beam" aria-hidden="true" />

        <img className="landing-satellite" src={satelliteImage} alt="SWOT Earth-observation satellite in orbit (NASA/JPL-Caltech)" />

        <div className="landing-datapoints" aria-hidden="true">
          {DATA_POINTS.map((p, i) => (
            <span key={i} className="landing-datapoint" style={{ top: p.top, left: p.left, animationDelay: p.delay }} />
          ))}
        </div>

        <div className="landing-hero-grid">
          <div className="landing-hero-left">
            <p className="landing-eyebrow">Agentic Earth-Observation Intelligence</p>
            <h1 className="landing-title">
              SatQuery <span className="landing-title-accent">AI</span>
            </h1>
            <p className="landing-subhead">Query the planet. Trace every answer.</p>
            <p className="landing-description">
              SatQuery AI answers questions about satellite imagery, describes land cover, detects and
              quantifies change between two dates, and fuses optical with radar data — through one agentic
              pipeline that shows exactly which model ran and what evidence backs every answer.
            </p>

            <button className="landing-cta-primary" onClick={onLaunch} type="button">
              Start Analysis
              <span className="landing-cta-arrow" aria-hidden="true">→</span>
            </button>

            <div className="landing-capability-chips">
              {CHIPS.map((c) => (
                <span className="landing-capability-chip" key={c.key}>
                  <span className="landing-capability-chip-icon">{CHIP_ICONS[c.key]}</span>
                  {c.label}
                </span>
              ))}
            </div>

            <div className="landing-live-example">
              <div className="landing-live-example-badge">Illustrative example — not a live feed</div>
              <p className="landing-live-example-query">
                &quot;Has the built-up area increased, decreased, or remained unchanged?&quot;
              </p>
              <div className="landing-live-example-row">
                <div className="landing-live-example-thumb">
                  <img src={exampleBefore} alt="Before capture" />
                  <span>2019-03-14</span>
                </div>
                <span className="landing-live-example-arrow">→</span>
                <div className="landing-live-example-thumb">
                  <img src={exampleAfter} alt="After capture" />
                  <span>2023-11-02</span>
                </div>
                <div className="landing-live-example-stat">
                  <span className="landing-live-example-stat-value">+18.4%</span>
                  <span className="landing-live-example-stat-label">built-up area</span>
                </div>
              </div>
            </div>
          </div>

          <div className="landing-hero-right">
            {user ? (
              <div className="auth-panel-inline landing-welcome-card">
                <p className="landing-welcome-label">Signed in</p>
                <p className="landing-welcome-name">{user.display_name || user.email}</p>
                <button className="landing-cta-primary" onClick={onLaunch} type="button">
                  Start Analysis
                  <span className="landing-cta-arrow" aria-hidden="true">→</span>
                </button>
              </div>
            ) : (
              <AuthPanel variant="inline" />
            )}
          </div>
        </div>
      </main>

      <section className="landing-stats">
        {STATS.map((s) => (
          <div className="landing-stat" key={s.label} title={s.detail}>
            <div className="landing-stat-value mono">{s.value}</div>
            <div className="landing-stat-label">{s.label}</div>
          </div>
        ))}
      </section>

      <section className="landing-capabilities" ref={capabilitiesRef}>
        <p className="landing-capabilities-kicker">Platform capabilities</p>
        <div className="landing-capabilities-grid">
          {CAPABILITIES.map((c, i) => (
            <div className="landing-capability-card" key={c.label}>
              <div className="landing-capability-index mono">{String(i + 1).padStart(2, '0')}</div>
              <div className="landing-capability-label">{c.label}</div>
              <div className="landing-capability-detail">{c.detail}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="landing-footer mono">
        evidence over guesswork · every step logged · every claim traceable
      </footer>
    </div>
  );
}
