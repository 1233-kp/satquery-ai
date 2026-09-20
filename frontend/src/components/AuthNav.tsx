import { useState } from 'react';
import { useAuth } from '../AuthContext';
import { AuthPanel } from './AuthPanel';

export function AuthNav({ variant = 'dark' }: { variant?: 'dark' | 'light' }) {
  const { user, loading, logout } = useAuth();
  const [panelOpen, setPanelOpen] = useState(false);

  if (loading) return null; // avoid a flash of "Log in" while session restore is in flight

  return (
    <>
      {user ? (
        <div className={`user-menu user-menu-${variant}`}>
          {user.avatar_url ? (
            <img className="user-menu-avatar" src={user.avatar_url} alt="" />
          ) : (
            <span className="user-menu-avatar user-menu-avatar-fallback">
              {(user.display_name || user.email || '?').slice(0, 1).toUpperCase()}
            </span>
          )}
          <span className="user-menu-name">{user.display_name || user.email}</span>
          <button className="user-menu-logout" onClick={() => logout()} type="button">
            Log out
          </button>
        </div>
      ) : (
        <button className={`auth-entry-btn auth-entry-btn-${variant}`} onClick={() => setPanelOpen(true)} type="button">
          Log in
        </button>
      )}
      {panelOpen && <AuthPanel onClose={() => setPanelOpen(false)} />}
    </>
  );
}
