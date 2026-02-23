import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  LayoutDashboard, 
  Database, 
  Wand2, 
  Network, 
  Activity, 
  Play, 
  Settings, 
  LogOut, 
  ArrowRight,
  Terminal,
  Zap,
  Moon,
  Sun,
  Laptop
} from 'lucide-react';
import { ViewState } from '../../types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (view: ViewState) => void;
  onLogout: () => void;
}

type CommandItem = {
  id: string;
  label: string;
  icon: React.ReactNode;
  shortcut?: string[];
  action: () => void;
  group: 'Navigation' | 'System' | 'Actions';
};

const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose, onNavigate, onLogout }) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Define all available commands
  const commands: CommandItem[] = [
    // Navigation
    { id: 'nav-dashboard', label: 'Go to Dashboard', icon: <LayoutDashboard size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.DASHBOARD) },
    { id: 'nav-datasets', label: 'Go to Datasets', icon: <Database size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.DATASETS) },
    { id: 'nav-aug', label: 'Go to Augmentation', icon: <Wand2 size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.DATA_AUGMENTATION) },
    { id: 'nav-model', label: 'Go to Model Builder', icon: <Network size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.MODEL_BUILDER) },
    { id: 'nav-train', label: 'Go to Training', icon: <Activity size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.TRAINING) },
    { id: 'nav-infer', label: 'Go to Inference', icon: <Play size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.INFERENCE) },
    { id: 'nav-settings', label: 'Go to Settings', icon: <Settings size={18} />, group: 'Navigation', action: () => onNavigate(ViewState.SETTINGS) },
    
    // Actions
    { id: 'act-new-exp', label: 'Start New Experiment', icon: <Zap size={18} />, group: 'Actions', action: () => onNavigate(ViewState.TRAINING) },
    { id: 'act-upload', label: 'Import Dataset', icon: <Database size={18} />, group: 'Actions', action: () => onNavigate(ViewState.DATASETS) },
    
    // System
    { id: 'sys-theme', label: 'Toggle Theme (Mock)', icon: <Moon size={18} />, group: 'System', action: () => {} },
    { id: 'sys-logout', label: 'Log Out', icon: <LogOut size={18} />, group: 'System', action: onLogout },
  ];

  // Filter commands based on query
  const filteredCommands = commands.filter(cmd => 
    cmd.label.toLowerCase().includes(query.toLowerCase()) || 
    cmd.group.toLowerCase().includes(query.toLowerCase())
  );

  // Reset selected index when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery('');
    }
  }, [isOpen]);

  // Handle keyboard navigation inside the palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % filteredCommands.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
          onClose();
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-start justify-center pt-[15vh] px-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm transition-opacity" 
        onClick={onClose}
      />

      {/* Palette Container */}
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[60vh]">
        
        {/* Search Bar */}
        <div className="flex items-center px-4 py-3 border-b border-slate-700/50">
          <Search className="text-cyan-500 mr-3" size={20} />
          <input
            ref={inputRef}
            type="text"
            className="flex-1 bg-transparent border-none outline-none text-white placeholder-slate-500 text-lg"
            placeholder="Type a command or search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="hidden sm:flex items-center space-x-1">
             <kbd className="bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded text-[10px] font-mono border border-slate-700">ESC</kbd>
             <span className="text-slate-600 text-xs">to close</span>
          </div>
        </div>

        {/* Results List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
          {filteredCommands.length === 0 ? (
            <div className="py-12 text-center text-slate-500">
              <Terminal size={32} className="mx-auto mb-3 opacity-20" />
              <p className="text-sm">No commands found.</p>
            </div>
          ) : (
            <div className="space-y-1">
              {/* Grouping logic could go here, but flat list is fine for now */}
              {filteredCommands.map((cmd, index) => (
                <button
                  key={cmd.id}
                  onClick={() => { cmd.action(); onClose(); }}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={`w-full flex items-center justify-between px-3 py-3 rounded-lg transition-all text-sm group ${
                    index === selectedIndex 
                      ? 'bg-cyan-900/20 text-cyan-400 border border-cyan-500/20 shadow-sm' 
                      : 'text-slate-400 hover:bg-slate-800/50 border border-transparent'
                  }`}
                >
                  <div className="flex items-center">
                    <span className={`mr-3 ${index === selectedIndex ? 'text-cyan-400' : 'text-slate-500'}`}>
                      {cmd.icon}
                    </span>
                    <span className={`font-medium ${index === selectedIndex ? 'text-white' : ''}`}>
                      {cmd.label}
                    </span>
                  </div>
                  
                  <div className="flex items-center">
                    {cmd.group && (
                      <span className={`text-[10px] uppercase tracking-wider mr-3 ${index === selectedIndex ? 'text-cyan-500/50' : 'text-slate-600'}`}>
                        {cmd.group}
                      </span>
                    )}
                    {index === selectedIndex && (
                       <ArrowRight size={14} className="text-cyan-500 animate-pulse" />
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 bg-slate-950/50 border-t border-slate-800 text-[10px] text-slate-500 flex justify-between items-center">
            <div className="flex space-x-3">
                <span><strong className="text-slate-400">↑↓</strong> to navigate</span>
                <span><strong className="text-slate-400">↵</strong> to select</span>
            </div>
            <div>
               NeuroCore <span className="text-cyan-500">Command</span>
            </div>
        </div>

      </div>
    </div>
  );
};

export default CommandPalette;