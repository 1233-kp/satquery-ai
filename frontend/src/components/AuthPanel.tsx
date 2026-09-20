import { useState } from 'react';
import type { FormEvent } from 'react';
import { createPortal } from 'react-dom';
import { useAuth } from '../AuthContext';

function GoogleMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#4285F4" d="M45.1 24.5c0-1.6-.14-3.1-.4-4.6H24v8.7h11.9c-.5 2.8-2.1 5.1-4.4 6.7v5.6h7.1c4.2-3.9 6.5-9.6 6.5-16.4z" />
      <path fill="#34A853" d="M24 46c5.9 0 10.9-2 14.5-5.3l-7.1-5.6c-2 1.3-4.5 2.1-7.4 2.1-5.7 0-10.5-3.9-12.2-9.1H4.5v5.8C8.1 40.9 15.4 46 24 46z" />
      <path fill="#FBBC05" d="M11.8 28.1c-.4-1.3-.7-2.7-.7-4.1s.3-2.8.7-4.1v-5.8H4.5C3 17.1 2.2 20.5 2.2 24s.8 6.9 2.3 9.9z" />
      <path fill="#EA4335" d="M24 10.9c3.2 0 6.1 1.1 8.4 3.3l6.3-6.3C34.9 4.3 29.9 2 24 2 15.4 2 8.1 7.1 4.5 14.1l7.3 5.8c1.7-5.2 6.5-9 12.2-9z" />
    </svg>
  );
}

function GithubMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

interface Props {
  onClose?: () => void;
  variant?: 'modal' | 'inline';
}

export function AuthPanel({ onClose, variant = 'modal' }: Props) {
  const { login, register, loginWithGoogle, loginWithGithub, authError, clearAuthError } = useAuth();
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  function switchTab(next: 'login' | 'register') {
    setTab(next);
    setLocalError(null);
    clearAuthError();
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setLocalError(null);
    try {
      if (tab === 'login') {
        await login(email, password);
      } else {
        await register(email, password, displayName.trim() || undefined);
      }
      onClose?.();
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setSubmitting(false);
    }
  }

  const shownError = localError || authError;

  const content = (
    <>
      {variant === 'modal' && (
        <button className="auth-modal-close" onClick={onClose} type="button" aria-label="Close">
          ×
        </button>
      )}

      <div className="auth-tabs">
        <button
          className={`auth-tab${tab === 'login' ? ' active' : ''}`}
          onClick={() => switchTab('login')}
          type="button"
        >
          Log in
        </button>
        <button
          className={`auth-tab${tab === 'register' ? ' active' : ''}`}
          onClick={() => switchTab('register')}
          type="button"
        >
          Create account
        </button>
      </div>

      <form className="auth-form" onSubmit={handleSubmit}>
        {tab === 'register' && (
          <>
            <label className="auth-field-label" htmlFor="auth-name">Name (optional)</label>
            <input
              id="auth-name"
              className="auth-input"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Ada Lovelace"
            />
          </>
        )}

        <label className="auth-field-label" htmlFor="auth-email">Email</label>
        <input
          id="auth-email"
          type="email"
          className="auth-input"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder="you@example.com"
        />

        <label className="auth-field-label" htmlFor="auth-password">Password</label>
        <input
          id="auth-password"
          type="password"
          className="auth-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={tab === 'register' ? 8 : undefined}
          placeholder={tab === 'register' ? 'At least 8 characters' : '••••••••'}
        />

        {shownError && <div className="auth-error">{shownError}</div>}

        <button className="auth-submit" type="submit" disabled={submitting}>
          {submitting ? 'Please wait…' : tab === 'login' ? 'Log in' : 'Create account'}
        </button>
      </form>

      <div className="auth-divider">
        <span>or continue with</span>
      </div>

      <div className="auth-oauth-row">
        <button className="auth-oauth-btn" onClick={loginWithGoogle} type="button">
          <GoogleMark /> Google
        </button>
        <button className="auth-oauth-btn" onClick={loginWithGithub} type="button">
          <GithubMark /> GitHub
        </button>
      </div>

      <p className="auth-modal-note">Signing in is optional — you can run analyses without an account.</p>
    </>
  );

  if (variant === 'inline') {
    return <div className="auth-panel-inline">{content}</div>;
  }

  return createPortal(
    <div className="auth-modal-backdrop" onClick={onClose}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        {content}
      </div>
    </div>,
    document.body
  );
}
