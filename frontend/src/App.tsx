import { useCallback, useEffect, useState } from 'react';
import { checkHealth, getAnalysisThread, submitAnalysis, uploadImage } from './api';
import { useAuth } from './AuthContext';
import { AccountScreen } from './components/AccountScreen';
import { AuthNav } from './components/AuthNav';
import { CapabilitiesScreen } from './components/CapabilitiesScreen';
import { ControlPanel } from './components/ControlPanel';
import type { SlotState } from './components/ControlPanel';
import { HistoryScreen } from './components/HistoryScreen';
import { LandingPage } from './components/LandingPage';
import { OverviewScreen } from './components/OverviewScreen';
import { ResultsPanel } from './components/ResultsPanel';
import { Sidebar } from './components/Sidebar';
import { MODE_CONFIGS } from './types';
import type { AnalysisResult, InputMode, ViewName } from './types';

const emptySlot: SlotState = { file: null, preview: null, uploaded: null, uploading: false };

function App() {
  const [screen, setScreen] = useState<'landing' | 'console'>('landing');
  const [transitioning, setTransitioning] = useState(false);
  const [activeView, setActiveView] = useState<ViewName>('overview');
  const [sidebarMobileOpen, setSidebarMobileOpen] = useState(false);

  const [mode, setMode] = useState<InputMode>('single_image');
  const [slots, setSlots] = useState<SlotState[]>([{ ...emptySlot }, { ...emptySlot }]);
  const [query, setQuery] = useState('');
  const [thread, setThread] = useState<AnalysisResult[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [followUpSubmitting, setFollowUpSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [followUpError, setFollowUpError] = useState<string | null>(null);
  const [threadLoadError, setThreadLoadError] = useState<string | null>(null);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const { user, justArrivedFromOAuth } = useAuth();

  useEffect(() => {
    checkHealth().then(setOnline);
  }, []);

  // "redirect into the console" after an OAuth round-trip lands back on
  // /auth/callback -- AuthContext already stripped the token out of the
  // URL, this just decides which screen to show once.
  useEffect(() => {
    if (justArrivedFromOAuth) {
      setScreen('console');
    }
  }, [justArrivedFromOAuth]);

  // The Account view only exists in the sidebar while logged in -- if the
  // user logs out while on it, don't strand them on a view that no longer
  // has a nav entry.
  useEffect(() => {
    if (activeView === 'account' && !user) {
      setActiveView('overview');
    }
  }, [activeView, user]);

  function handleModeChange(newMode: InputMode) {
    setMode(newMode);
    setSlots([{ ...emptySlot }, { ...emptySlot }]);
    setThread([]);
    setSelectedTurnId(null);
    setError(null);
    setFollowUpError(null);
    setThreadLoadError(null);
  }

  async function handleSlotPick(index: number, file: File) {
    const preview = URL.createObjectURL(file);
    setSlots((prev) => {
      const next = [...prev];
      next[index] = { file, preview, uploaded: null, uploading: true };
      return next;
    });
    try {
      const uploaded = await uploadImage(file);
      setSlots((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], uploaded, uploading: false };
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setSlots((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], uploading: false };
        return next;
      });
    }
  }

  const requiredSlots = MODE_CONFIGS.find((c) => c.mode === mode)!.slotLabels.length;
  const canSubmit =
    query.trim().length > 0 &&
    slots.slice(0, requiredSlots).every((s) => s.uploaded && !s.uploading);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    setFollowUpError(null);
    setThreadLoadError(null);
    setThread([]);
    try {
      const image_ids = slots.slice(0, requiredSlots).map((s) => s.uploaded!.image_id);
      const res = await submitAnalysis({ query, mode, image_ids });
      setThread([res]);
      setSelectedTurnId(res.analysis_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleFollowUp(followUpQuery: string) {
    if (thread.length === 0) return;
    setFollowUpSubmitting(true);
    setFollowUpError(null);
    try {
      const parentId = thread[thread.length - 1].analysis_id;
      const res = await submitAnalysis({ query: followUpQuery, parent_analysis_id: parentId });
      setThread((prev) => [...prev, res]);
      setSelectedTurnId(res.analysis_id);
    } catch (err) {
      setFollowUpError(err instanceof Error ? err.message : 'Follow-up failed');
    } finally {
      setFollowUpSubmitting(false);
    }
  }

  const handleSelectTurn = useCallback((id: string) => {
    setSelectedTurnId(id);
  }, []);

  // Reopens a previously stored analysis (and its full follow-up thread,
  // if any) from Overview's recent list or History -- reuses the same
  // /analysis/{id}/thread endpoint and thread-rendering path the
  // follow-up feature already built.
  async function openThread(analysisId: string) {
    setActiveView('new-analysis');
    setThreadLoadError(null);
    setSubmitting(true);
    try {
      const fullThread = await getAnalysisThread(analysisId);
      setThread(fullThread);
      setSelectedTurnId(fullThread.length > 0 ? fullThread[fullThread.length - 1].analysis_id : null);
      if (fullThread.length > 0) {
        setMode(fullThread[0].trace.mode);
      }
      setError(null);
      setFollowUpError(null);
    } catch (err) {
      setThreadLoadError(err instanceof Error ? err.message : 'Failed to load analysis');
    } finally {
      setSubmitting(false);
    }
  }

  function quickStartMode(newMode: InputMode) {
    handleModeChange(newMode);
    setActiveView('new-analysis');
  }

  function handleLaunch() {
    // Intentional seam between the light landing hero and the dark
    // console: fade to the console's own --bg color, swap screens while
    // fully covered, then fade the cover away -- reads as a deliberate
    // cross-fade rather than a hard, jarring cut between the two themes.
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    setTransitioning(true);
    window.setTimeout(() => {
      setScreen('console');
      window.setTimeout(() => setTransitioning(false), reduceMotion ? 0 : 20);
    }, reduceMotion ? 0 : 380);
  }

  const transitionOverlay = <div className={`screen-transition-overlay${transitioning ? ' active' : ''}`} aria-hidden="true" />;

  if (screen === 'landing') {
    return (
      <>
        <LandingPage onLaunch={handleLaunch} />
        {transitionOverlay}
      </>
    );
  }

  return (
    <div className="app-shell">
      {transitionOverlay}
      <header className="topbar">
        <button
          className="hamburger-btn"
          onClick={() => setSidebarMobileOpen(true)}
          type="button"
          aria-label="Open menu"
        >
          <span />
          <span />
          <span />
        </button>
        <div className="brand">
          <span className="brand-mark">SQ</span>
          <span className="brand-name">SatQuery AI</span>
          <span className="brand-tagline">Agentic remote-sensing vision-language assistant</span>
        </div>
        <div className="topbar-right">
          <div className="status-row">
            <span className={`status-dot ${online === null ? '' : online ? 'online' : 'offline'}`} />
            {online === null ? 'checking backend…' : online ? 'backend online' : 'backend unreachable'}
          </div>
          <AuthNav variant="dark" />
        </div>
      </header>

      <div className="app-body">
        <Sidebar
          activeView={activeView}
          onNavigate={setActiveView}
          isLoggedIn={!!user}
          mobileOpen={sidebarMobileOpen}
          onCloseMobile={() => setSidebarMobileOpen(false)}
        />

        <div className="app-content">
          {activeView === 'overview' && (
            <OverviewScreen onOpenThread={openThread} onQuickStart={quickStartMode} />
          )}

          {activeView === 'new-analysis' && (
            <div className="new-analysis-screen">
              {threadLoadError && <div className="error-banner new-analysis-error">{threadLoadError}</div>}
              <div className="main">
                <ControlPanel
                  mode={mode}
                  onModeChange={handleModeChange}
                  slots={slots}
                  onSlotPick={handleSlotPick}
                  query={query}
                  onQueryChange={setQuery}
                  onSubmit={handleSubmit}
                  submitting={submitting}
                  canSubmit={canSubmit}
                  error={error}
                />
                <ResultsPanel
                  thread={thread}
                  loading={submitting}
                  followUpSubmitting={followUpSubmitting}
                  followUpError={followUpError}
                  onFollowUp={handleFollowUp}
                  selectedTurnId={selectedTurnId}
                  onSelectTurn={handleSelectTurn}
                />
              </div>
            </div>
          )}

          {activeView === 'history' && <HistoryScreen onOpenThread={openThread} />}

          {activeView === 'capabilities' && <CapabilitiesScreen />}

          {activeView === 'account' && <AccountScreen />}
        </div>
      </div>
    </div>
  );
}

export default App;
