import type { ReactElement } from 'react';
import type { ViewName } from '../types';

function OverviewIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <rect x="10" y="2" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <rect x="2" y="10" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <rect x="10" y="10" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function NewAnalysisIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.4" />
      <path d="M9 6v6M6 9h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.4" />
      <path d="M9 5v4l3 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CapabilitiesIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <path d="M9 2l2.1 4.3 4.7.7-3.4 3.3.8 4.7L9 12.8l-4.2 2.2.8-4.7-3.4-3.3 4.7-.7L9 2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

function AccountIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <circle cx="9" cy="6.5" r="3" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3 16c0-3 2.7-5 6-5s6 2 6 5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

interface NavItem {
  view: ViewName;
  label: string;
  icon: () => ReactElement;
}

const NAV_ITEMS: NavItem[] = [
  { view: 'overview', label: 'Overview', icon: OverviewIcon },
  { view: 'new-analysis', label: 'New Analysis', icon: NewAnalysisIcon },
  { view: 'history', label: 'History', icon: HistoryIcon },
  { view: 'capabilities', label: 'Capabilities', icon: CapabilitiesIcon },
];

interface Props {
  activeView: ViewName;
  onNavigate: (view: ViewName) => void;
  isLoggedIn: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export function Sidebar({ activeView, onNavigate, isLoggedIn, mobileOpen, onCloseMobile }: Props) {
  const items = isLoggedIn
    ? [...NAV_ITEMS, { view: 'account' as ViewName, label: 'Account', icon: AccountIcon }]
    : NAV_ITEMS;

  return (
    <>
      {mobileOpen && <div className="sidebar-backdrop" onClick={onCloseMobile} />}
      <nav className={`sidebar${mobileOpen ? ' sidebar-open' : ''}`}>
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.view}
              className={`sidebar-item${activeView === item.view ? ' active' : ''}`}
              onClick={() => {
                onNavigate(item.view);
                onCloseMobile();
              }}
              type="button"
            >
              <span className="sidebar-icon"><Icon /></span>
              <span className="sidebar-label">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
}
