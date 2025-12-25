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
import GlobalStatusBar from './components/GlobalStatusBar'; // Import
import { ViewState } from './types';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeView, setActiveView] = useState<ViewState>(ViewState.DASHBOARD);
  
  // Command Palette State
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);

  const handleLogin = () => {
    setIsAuthenticated(true);
    setActiveView(ViewState.DASHBOARD);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

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