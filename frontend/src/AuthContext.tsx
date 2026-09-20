import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { getApiUrl, getMe, loginAccount, logoutAccount, registerAccount } from './api';
import type { PublicUser } from './api';

const TOKEN_KEY = 'satquery_auth_token';

interface AuthContextValue {
  user: PublicUser | null;
  loading: boolean;
  authError: string | null;
  justArrivedFromOAuth: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName?: string) => Promise<void>;
  logout: () => Promise<void>;
  loginWithGoogle: () => void;
  loginWithGithub: () => void;
  clearAuthError: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

// If we just landed on /auth/callback (full-page redirect back from an
// OAuth provider), pull the token/error out of the URL once, synchronously,
// before the provider's first render -- avoids a flash of "logged out" UI.
function consumeOAuthCallback(): { token: string | null; error: string | null; wasCallback: boolean } {
  if (window.location.pathname !== '/auth/callback') {
    return { token: null, error: null, wasCallback: false };
  }
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  const error = params.get('error');
  // Clean the URL so a refresh doesn't re-process a stale token/error.
  window.history.replaceState({}, '', '/');
  return { token, error, wasCallback: true };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<PublicUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);
  const [justArrivedFromOAuth, setJustArrivedFromOAuth] = useState(false);

  useEffect(() => {
    const { token: callbackToken, error: callbackError, wasCallback } = consumeOAuthCallback();

    if (callbackError) {
      setAuthError(decodeURIComponent(callbackError));
    }
    if (wasCallback) {
      setJustArrivedFromOAuth(true);
    }

    const token = callbackToken || localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setLoading(false);
      return;
    }

    if (callbackToken) {
      localStorage.setItem(TOKEN_KEY, callbackToken);
    }

    getMe(token)
      .then(setUser)
      .catch(() => {
        // Stale/invalid/expired token -- drop it silently. This must never
        // throw upward or block rendering; a broken token should just mean
        // "logged out", not a broken app.
        localStorage.removeItem(TOKEN_KEY);
      })
      .finally(() => setLoading(false));
  }, []);

  async function login(email: string, password: string) {
    const res = await loginAccount({ email, password });
    localStorage.setItem(TOKEN_KEY, res.access_token);
    setUser(res.user);
    setAuthError(null);
  }

  async function register(email: string, password: string, displayName?: string) {
    const res = await registerAccount({ email, password, display_name: displayName });
    localStorage.setItem(TOKEN_KEY, res.access_token);
    setUser(res.user);
    setAuthError(null);
  }

  async function logout() {
    const token = localStorage.getItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_KEY);
    setUser(null);
    if (token) await logoutAccount(token);
  }

  function loginWithGoogle() {
    window.location.href = `${getApiUrl()}/auth/google/login`;
  }

  function loginWithGithub() {
    window.location.href = `${getApiUrl()}/auth/github/login`;
  }

  function clearAuthError() {
    setAuthError(null);
  }

  return (
    <AuthContext.Provider
      value={{ user, loading, authError, justArrivedFromOAuth, login, register, logout, loginWithGoogle, loginWithGithub, clearAuthError }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
