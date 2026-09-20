import { useAuth } from '../AuthContext';

export function AccountScreen() {
  const { user, logout } = useAuth();

  if (!user) return null;

  return (
    <div className="screen account-screen">
      <p className="screen-title">Account</p>

      <div className="account-card">
        {user.avatar_url ? (
          <img className="account-avatar" src={user.avatar_url} alt="" />
        ) : (
          <span className="account-avatar account-avatar-fallback">
            {(user.display_name || user.email || '?').slice(0, 1).toUpperCase()}
          </span>
        )}
        <div className="account-name">{user.display_name || 'Unnamed account'}</div>
        {user.email && <div className="account-email">{user.email}</div>}
        <div className="account-field-row">
          <span className="account-field-label">Sign-in method</span>
          <span className="account-field-value mono">
            {user.oauth_provider ? user.oauth_provider : 'email + password'}
          </span>
        </div>
        <div className="account-field-row">
          <span className="account-field-label">Account ID</span>
          <span className="account-field-value mono">{user.id}</span>
        </div>
        <button className="account-logout-btn" onClick={() => logout()} type="button">
          Log out
        </button>
      </div>
    </div>
  );
}
