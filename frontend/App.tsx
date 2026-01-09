import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import DatasetManager from './components/DatasetManager';
import DataAugmentation from './components/DataAugmentation';
import ModelBuilder from './components/ModelBuilder';
import TrainingMonitor from './components/TrainingMonitor';
import InferenceView from './components/InferenceView';
import Settings from './components/Settings';
import Login from './components/Login';
import CommandPalette from './components/CommandPalette';
import GlobalStatusBar from './components/GlobalStatusBar';
import { ViewState } from './types';

const ACTIVE_VIEW_KEY = 'active_view';

/**
 * 检查本地存储中的登录状态
 */
const checkAuthStatus = (): boolean => {
  const token = localStorage.getItem('access_token');
  const user = localStorage.getItem('user');
  return !!(token && user);
};

/**
 * 获取保存的视图状态
 */
const getSavedView = (): ViewState => {
  const savedView = localStorage.getItem(ACTIVE_VIEW_KEY);
  // 验证保存的视图是否有效
  if (savedView && Object.values(ViewState).includes(savedView as ViewState)) {
    return savedView as ViewState;
  }
  return ViewState.DASHBOARD;
};

function App() {
  // 初始化时从 localStorage 恢复登录状态和视图状态
  const [isAuthenticated, setIsAuthenticated] = useState(checkAuthStatus());
  const [activeView, setActiveView] = useState<ViewState>(getSavedView());

  // Command Palette State
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    // 清除本地存储的认证信息
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    localStorage.removeItem(ACTIVE_VIEW_KEY);
  };

  // 保存当前视图到 localStorage
  useEffect(() => {
    if (isAuthenticated) {
      localStorage.setItem(ACTIVE_VIEW_KEY, activeView);
    }
  }, [activeView, isAuthenticated]);

  // Global Keyboard Listener for Ctrl+K / Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for Ctrl+K (Windows/Linux) or Cmd+K (Mac)
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault(); // Prevent browser default (e.g. search bar focus)
        setIsPaletteOpen(prev => !prev);
      }
    };

    if (isAuthenticated) {
        window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isAuthenticated]);

  const renderContent = () => {
    switch (activeView) {
      case ViewState.DASHBOARD:
        return <Dashboard onNavigate={setActiveView} />;
      case ViewState.DATASETS:
        return <DatasetManager />;
      case ViewState.DATA_AUGMENTATION:
        return <DataAugmentation />;
      case ViewState.MODEL_BUILDER:
        return <ModelBuilder />;
      case ViewState.TRAINING:
        return <TrainingMonitor />;
      case ViewState.INFERENCE:
        return <InferenceView />;
      case ViewState.SETTINGS:
        return <Settings onLogout={handleLogout} />;
      default:
        return <Dashboard onNavigate={setActiveView} />;
    }
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-cyan-500/30 selection:text-cyan-200 animate-in fade-in duration-500">

      {/* Command Palette Overlay */}
      <CommandPalette
         isOpen={isPaletteOpen}
         onClose={() => setIsPaletteOpen(false)}
         onNavigate={setActiveView}
         onLogout={handleLogout}
      />

      <Sidebar
         activeView={activeView}
         onNavigate={setActiveView}
         onLogout={handleLogout}
         onOpenPalette={() => setIsPaletteOpen(true)}
      />

      <main className="flex-1 flex flex-col min-w-0 bg-grid-pattern bg-[length:40px_40px] relative">
        {/* Ambient Glows */}
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-slate-900/80 to-transparent pointer-events-none z-0" />
        <div className="absolute -top-[200px] -left-[200px] w-[500px] h-[500px] bg-cyan-500/5 rounded-full blur-3xl pointer-events-none z-0" />

        {/* Content Container with Animation Key */}
        <div className="flex-1 relative z-10 overflow-hidden flex flex-col">
            {/* The 'key' prop triggers a re-render animation when view changes */}
            <div
                key={activeView}
                className="flex-1 w-full h-full overflow-hidden animate-page-enter"
            >
                {renderContent()}
            </div>
        </div>

        {/* Global Status Bar (Always on bottom) */}
        <GlobalStatusBar />
      </main>
    </div>
  );
}

export default App;
