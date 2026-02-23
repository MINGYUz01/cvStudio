import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Network, 
  Activity, 
  Play, 
  Settings, 
  Hexagon,
  LogOut,
  Wand2,
  ChevronLeft,
  ChevronRight,
  Search
} from 'lucide-react';
import { ViewState } from '../../types';

interface SidebarProps {
  activeView: ViewState;
  onNavigate: (view: ViewState) => void;
  onLogout: () => void;
  onOpenPalette?: () => void; // New prop
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, onNavigate, onLogout, onOpenPalette }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [osKey, setOsKey] = useState('Ctrl');

  useEffect(() => {
     if (navigator.platform.indexOf('Mac') > -1) {
         setOsKey('⌘');
     }
  }, []);

  const menuItems = [
    { id: ViewState.DASHBOARD, label: '仪表盘', icon: <LayoutDashboard size={20} /> },
    { id: ViewState.DATASETS, label: '数据集管理', icon: <Database size={20} /> },
    { id: ViewState.DATA_AUGMENTATION, label: '数据增强', icon: <Wand2 size={20} /> },
    { id: ViewState.MODEL_BUILDER, label: '模型管理', icon: <Network size={20} /> },
    { id: ViewState.TRAINING, label: '训练监控', icon: <Activity size={20} /> },
    { id: ViewState.INFERENCE, label: '推理演示', icon: <Play size={20} /> },
  ];

  return (
    <div 
      className={`
        h-screen bg-slate-950 border-r border-slate-800 flex flex-col justify-between shrink-0 z-50 relative transition-all duration-300 ease-in-out
        ${isCollapsed ? 'w-20' : 'w-64'}
      `}
    >
      <div>
        {/* Logo Area */}
        <div className={`h-16 flex items-center border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-sm transition-all duration-300 ${isCollapsed ? 'justify-center px-0' : 'px-6'}`}>
          <Hexagon className="text-cyan-400 animate-pulse shrink-0" size={28} />
          <div className={`ml-3 overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}`}>
            <h1 className="text-lg font-bold tracking-wider text-slate-100 font-mono whitespace-nowrap">NEURO<span className="text-cyan-400">CORE</span></h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest whitespace-nowrap">Studio v2.4</p>
          </div>
        </div>
        
        {/* Quick Search Trigger */}
        <div className={`mt-4 px-3 transition-all duration-300 ${isCollapsed ? 'flex justify-center' : ''}`}>
            <button 
                onClick={onOpenPalette}
                className={`flex items-center bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/50 text-slate-400 rounded-lg transition-all group ${isCollapsed ? 'p-3 justify-center' : 'w-full px-3 py-2'}`}
                title="Search (Ctrl+K)"
            >
                <Search size={16} className="text-slate-500 group-hover:text-cyan-400 transition-colors" />
                <div className={`ml-3 flex items-center justify-between flex-1 overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100'}`}>
                    <span className="text-xs group-hover:text-slate-200">Search...</span>
                    <span className="text-[10px] font-mono bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700 text-slate-500 group-hover:text-cyan-500">{osKey} K</span>
                </div>
            </button>
        </div>

        {/* Navigation */}
        <nav className="mt-4 px-3 space-y-2">
          {menuItems.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                title={isCollapsed ? item.label : ''}
                className={`
                  w-full flex items-center py-3 rounded-lg transition-all duration-200 group relative overflow-hidden
                  ${isCollapsed ? 'justify-center px-0' : 'px-4'}
                  ${isActive 
                    ? 'text-cyan-400 bg-cyan-950/20 border border-cyan-900/50 shadow-[0_0_15px_rgba(34,211,238,0.1)]' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }
                `}
              >
                {isActive && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.8)]" />
                )}
                <span className={`shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'} ${isCollapsed ? '' : 'mr-3'}`}>
                  {item.icon}
                </span>
                <span className={`whitespace-nowrap transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100 block'}`}>
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Area */}
      <div className="p-3 border-t border-slate-800/50 flex flex-col gap-2">
        <button 
           onClick={() => onNavigate(ViewState.SETTINGS)}
           title="系统设置"
           className={`
             w-full flex items-center py-2 text-sm font-medium rounded-lg text-slate-400 hover:bg-slate-900 transition-colors
             ${isCollapsed ? 'justify-center px-0' : 'px-4'}
             ${activeView === ViewState.SETTINGS ? 'bg-slate-800 text-white' : ''}
           `}
        >
          <Settings size={20} className={`shrink-0 ${isCollapsed ? '' : 'mr-3'}`} />
          <span className={`whitespace-nowrap transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100 block'}`}>系统设置</span>
        </button>
        
        <div className={`flex items-center rounded-lg bg-slate-900/50 border border-slate-800 transition-all duration-300 ${isCollapsed ? 'justify-center p-2' : 'px-4 py-3'}`}>
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-xs font-bold text-white ring-2 ring-slate-950 shrink-0">
            JD
          </div>
          <div className={`ml-3 overflow-hidden transition-all duration-300 ${isCollapsed ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100 block'}`}>
            <p className="text-xs font-medium text-white truncate">John Doe</p>
            <p className="text-[10px] text-slate-500 truncate">高级工程师</p>
          </div>
          <button onClick={onLogout} title="退出登录" className={`text-slate-500 hover:text-rose-400 cursor-pointer transition-colors ml-auto ${isCollapsed ? 'hidden' : 'block'}`}>
             <LogOut size={16} />
          </button>
        </div>

        {/* Collapse Toggle */}
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="w-full flex items-center justify-center py-2 mt-1 text-slate-600 hover:text-cyan-400 hover:bg-slate-900/50 rounded transition-colors"
        >
          {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </div>
  );
};

export default Sidebar;