import React, { useState } from 'react';
import { 
  Cpu, 
  Activity, 
  HardDrive, 
  Zap, 
  TrendingUp, 
  Clock, 
  Terminal,
  CheckCircle,
  Database,
  Network,
  Play,
  ArrowRight,
  Wand2,
  MemoryStick,
  FolderOpen,
  Server,
  Monitor,
  Maximize2,
  X,
  Code
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { MetricCardProps, ViewState } from '../../types';

// Add prop interface
interface DashboardProps {
  onNavigate?: (view: ViewState) => void;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, unit, trend, icon, color = 'cyan', onClick }) => {
  const colorMap = {
    cyan: 'text-cyan-400 border-cyan-500/20 bg-cyan-950/10',
    purple: 'text-purple-400 border-purple-500/20 bg-purple-950/10',
    emerald: 'text-emerald-400 border-emerald-500/20 bg-emerald-950/10',
    rose: 'text-rose-400 border-rose-500/20 bg-rose-950/10',
    amber: 'text-amber-400 border-amber-500/20 bg-amber-950/10',
  };

  return (
    <div 
      onClick={onClick}
      className={`glass-panel p-4 rounded-xl border ${colorMap[color]} relative overflow-hidden group ${onClick ? 'cursor-pointer hover:bg-opacity-80 transition-all hover:scale-[1.02]' : ''}`}
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-slate-400 text-xs font-medium uppercase tracking-wider">{title}</h3>
        <span className={`p-1.5 rounded-lg bg-slate-900/50 ${colorMap[color].split(' ')[0]}`}>
          {icon}
        </span>
      </div>
      <div className="flex items-baseline space-x-2">
        <span className="text-2xl font-bold text-white font-mono">{value}</span>
        {unit && <span className="text-xs text-slate-500 font-mono">{unit}</span>}
      </div>
      {trend !== undefined && (
        <div className={`mt-2 text-xs flex items-center ${trend >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          <TrendingUp size={12} className={`mr-1 ${trend < 0 ? 'rotate-180' : ''}`} />
          {Math.abs(trend)}% 较昨日
        </div>
      )}
      {/* Decorative Glow */}
      <div className={`absolute -right-4 -bottom-4 w-24 h-24 rounded-full blur-3xl opacity-10 group-hover:opacity-20 transition-opacity bg-${color}-500`} />
    </div>
  );
};

// Expanded Log Modal Component
interface ExpandedLogModalProps {
    type: 'backend' | 'frontend' | null;
    onClose: () => void;
    currentLogs: any[];
}

const ExpandedLogModal: React.FC<ExpandedLogModalProps> = ({ type, onClose, currentLogs }) => {
    if (!type) return null;

    // Generate specific mock history based on type
    let historicalLogs: any[] = [];
    let title = "";
    let subTitle = "";
    let icon = null;

    if (type === 'backend') {
        title = "后端系统完整日志";
        subTitle = "/var/log/neurocore/backend.log • Live Stream";
        icon = <Terminal size={20} className="mr-3 text-cyan-400"/>;
        historicalLogs = [
            { id: 101, time: '10:35:10', level: 'INFO', msg: 'System initialization started' },
            { id: 102, time: '10:35:12', level: 'INFO', msg: 'Loaded configuration from /etc/neurocore/config.yaml' },
            { id: 103, time: '10:35:15', level: 'INFO', msg: 'Database connection established (PostgreSQL)' },
            { id: 104, time: '10:36:00', level: 'INFO', msg: 'Worker pool initialized with 8 threads' },
            { id: 105, time: '10:38:22', level: 'WARN', msg: 'Cache directory is 80% full, triggering cleanup' },
            { id: 106, time: '10:38:25', level: 'INFO', msg: 'Cache cleanup completed. Freed 1.2GB' },
            { id: 107, time: '10:39:00', level: 'INFO', msg: 'API Server listening on port 8000' },
            { id: 108, time: '10:40:05', level: 'INFO', msg: 'WebSocket server started' },
        ];
    } else {
        title = "前端应用调试日志";
        subTitle = "Browser Console • Network & React Lifecycle";
        icon = <Code size={20} className="mr-3 text-purple-400"/>;
        historicalLogs = [
            { id: 201, time: '10:39:55', level: 'INFO', msg: '[Router] Navigation to /dashboard' },
            { id: 202, time: '10:39:56', level: 'INFO', msg: '[Auth] Session validated token=****' },
            { id: 203, time: '10:39:58', level: 'INFO', msg: '[API] GET /api/v1/stats 200 OK (45ms)' },
            { id: 204, time: '10:40:00', level: 'INFO', msg: '[WebSocket] Connecting to wss://api.neurocore.ai/stream...' },
            { id: 205, time: '10:40:02', level: 'WARN', msg: '[Rendering] Heavy calculation in Chart component took 12ms' },
            { id: 206, time: '10:40:05', level: 'INFO', msg: '[System] Theme applied: Dark Mode' },
        ];
    }

    // Merge history with current live logs
    const expandedLogs = [...historicalLogs, ...currentLogs].sort((a, b) => b.id - a.id);
    const sortedLogs = expandedLogs.sort((a,b) => a.id - b.id); 

    return (
        <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6 animate-in fade-in duration-200">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full h-full shadow-2xl flex flex-col animate-in zoom-in-95 duration-200">
                <div className="flex justify-between items-center p-5 border-b border-slate-700 bg-slate-950/50 rounded-t-2xl shrink-0">
                    <div>
                        <h3 className="text-white text-lg font-bold flex items-center">{icon} {title}</h3>
                        <p className="text-slate-500 text-xs mt-1 font-mono">{subTitle}</p>
                    </div>
                    <button onClick={onClose} className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors">
                        <X size={24} />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-6 font-mono text-sm space-y-2 bg-black/40 custom-scrollbar">
                    {sortedLogs.map((log) => (
                        <div key={log.id} className="flex items-start hover:bg-slate-800/50 p-1.5 rounded border border-transparent hover:border-slate-800 transition-colors group">
                            <span className="text-slate-500 mr-6 w-24 shrink-0 select-none border-r border-slate-800 pr-4">{log.time}</span>
                            <span className={`w-20 shrink-0 font-bold ${
                                log.level === 'INFO' ? (type === 'backend' ? 'text-cyan-500' : 'text-purple-500') : 
                                log.level === 'WARN' ? 'text-amber-500' : 
                                'text-rose-500'
                            }`}>{log.level}</span>
                            <span className="text-slate-300 break-all group-hover:text-white transition-colors">{log.msg}</span>
                        </div>
                    ))}
                    <div className={`flex items-center animate-pulse pt-4 border-t border-slate-800/50 mt-4 ${type === 'backend' ? 'text-cyan-500' : 'text-purple-500'}`}>
                        <span className="mr-2">_</span>
                        <span className="text-xs opacity-70">Listening for {type} events...</span>
                    </div>
                </div>
                 <div className="p-4 border-t border-slate-700 bg-slate-900/50 rounded-b-2xl flex justify-between items-center text-xs text-slate-500 shrink-0">
                    <span>Total Entries: {sortedLogs.length}</span>
                    <span>Status: <span className="text-emerald-500">Active</span></span>
                </div>
            </div>
        </div>
    );
};

const data = [
  { name: '00:00', gpu: 45, mem: 30 },
  { name: '04:00', gpu: 55, mem: 45 },
  { name: '08:00', gpu: 85, mem: 75 },
  { name: '12:00', gpu: 92, mem: 80 },
  { name: '16:00', gpu: 78, mem: 70 },
  { name: '20:00', gpu: 65, mem: 60 },
  { name: '24:00', gpu: 50, mem: 40 },
];

const backendLogs = [
  { id: 1, time: '10:42:05', level: 'INFO', msg: 'YOLOv8-Nano 训练任务已启动 [Batch 32]' },
  { id: 2, time: '10:45:12', level: 'WARN', msg: 'GPU 温度 > 80°C (未触发降频)' },
  { id: 3, time: '10:48:00', level: 'INFO', msg: 'Checkpoint 已保存: epoch_24.pt' },
  { id: 4, time: '10:55:33', level: 'ERROR', msg: '在 dataset/val/0043.jpg 中发现损坏数据' },
  { id: 5, time: '11:02:10', level: 'INFO', msg: '验证集评估完成. mAP@50: 0.89' },
];

const frontendLogs = [
  { id: 1, time: '10:40:12', level: 'INFO', msg: 'App mounted successfully' },
  { id: 2, time: '10:41:05', level: 'INFO', msg: 'WebSocket connected to ws://localhost:8000' },
  { id: 3, time: '10:42:01', level: 'INFO', msg: 'User initiated training task #1042' },
  { id: 4, time: '10:50:22', level: 'WARN', msg: 'High latency detected in status stream (250ms)' },
  { id: 5, time: '11:00:05', level: 'INFO', msg: 'Chart component re-rendered' },
];

const projectFiles = [
    { name: 'Datasets (Raw)', size: '12.5 GB', percent: 65, color: 'bg-cyan-500' },
    { name: 'Checkpoints', size: '4.2 GB', percent: 20, color: 'bg-purple-500' },
    { name: 'Training Logs', size: '1.8 GB', percent: 10, color: 'bg-emerald-500' },
    { name: 'Temp / Cache', size: '850 MB', percent: 5, color: 'bg-slate-500' },
];

const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [expandedLogType, setExpandedLogType] = useState<'backend' | 'frontend' | null>(null);

  return (
    <div className="relative h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Expanded Log Modal - Positioned Absolute relative to Dashboard */}
      <ExpandedLogModal 
        type={expandedLogType} 
        onClose={() => setExpandedLogType(null)} 
        currentLogs={expandedLogType === 'backend' ? backendLogs : frontendLogs} 
      />

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6">
        <div className="flex justify-between items-end border-b border-slate-800 pb-4">
            <div>
            <h2 className="text-2xl font-bold text-white mb-1">系统概览</h2>
            <p className="text-slate-400 text-sm">集群状态: <span className="text-emerald-400 font-mono">在线</span> • 运行时间: <span className="text-cyan-400 font-mono">14天 03时 22分</span></p>
            </div>
        </div>

        {/* Top Metrics Grid (Row 1) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard title="GPU 利用率" value={92} unit="%" trend={12} icon={<Cpu size={18} />} color="cyan" />
            <MetricCard title="内存使用率" value={64} unit="%" trend={-5} icon={<MemoryStick size={18} />} color="purple" />
            <MetricCard title="进行中实验" value={3} unit="Tasks" trend={0} icon={<Zap size={18} />} color="amber" />
            <MetricCard title="数据集存储" value={4.2} unit="TB" trend={5} icon={<HardDrive size={18} />} color="emerald" />
        </div>

        {/* Quick Access Cards (Row 2) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            <div 
            onClick={() => onNavigate?.(ViewState.DATASETS)}
            className="glass-panel p-5 rounded-xl border border-slate-800/60 cursor-pointer hover:border-cyan-500/50 hover:bg-slate-900/60 transition-all group"
            >
            <div className="flex justify-between items-start mb-3">
                <div className="p-2 rounded bg-cyan-900/20 text-cyan-400 group-hover:bg-cyan-900/40 transition-colors">
                <Database size={20} />
                </div>
                <ArrowRight size={18} className="text-slate-600 group-hover:text-cyan-400 transition-colors -rotate-45 group-hover:rotate-0" />
            </div>
            <h3 className="text-white font-medium mb-1">数据集管理</h3>
            <p className="text-xs text-slate-400">管理 8 个数据集, 累计 4.2TB。</p>
            </div>

            <div 
            onClick={() => onNavigate?.(ViewState.DATA_AUGMENTATION)}
            className="glass-panel p-5 rounded-xl border border-slate-800/60 cursor-pointer hover:border-amber-500/50 hover:bg-slate-900/60 transition-all group"
            >
            <div className="flex justify-between items-start mb-3">
                <div className="p-2 rounded bg-amber-900/20 text-amber-400 group-hover:bg-amber-900/40 transition-colors">
                <Wand2 size={20} />
                </div>
                <ArrowRight size={18} className="text-slate-600 group-hover:text-amber-400 transition-colors -rotate-45 group-hover:rotate-0" />
            </div>
            <h3 className="text-white font-medium mb-1">增强策略管理</h3>
            <p className="text-xs text-slate-400">配置图像增强流水线与算子。</p>
            </div>

            <div 
            onClick={() => onNavigate?.(ViewState.MODEL_BUILDER)}
            className="glass-panel p-5 rounded-xl border border-slate-800/60 cursor-pointer hover:border-purple-500/50 hover:bg-slate-900/60 transition-all group"
            >
            <div className="flex justify-between items-start mb-3">
                <div className="p-2 rounded bg-purple-900/20 text-purple-400 group-hover:bg-purple-900/40 transition-colors">
                <Network size={20} />
                </div>
                <ArrowRight size={18} className="text-slate-600 group-hover:text-purple-400 transition-colors -rotate-45 group-hover:rotate-0" />
            </div>
            <h3 className="text-white font-medium mb-1">模型工作台</h3>
            <p className="text-xs text-slate-400">可视化构建神经网络架构。</p>
            </div>

            <div 
            onClick={() => onNavigate?.(ViewState.TRAINING)}
            className="glass-panel p-5 rounded-xl border border-slate-800/60 cursor-pointer hover:border-emerald-500/50 hover:bg-slate-900/60 transition-all group"
            >
            <div className="flex justify-between items-start mb-3">
                <div className="p-2 rounded bg-emerald-900/20 text-emerald-400 group-hover:bg-emerald-900/40 transition-colors">
                <Activity size={20} />
                </div>
                <ArrowRight size={18} className="text-slate-600 group-hover:text-emerald-400 transition-colors -rotate-45 group-hover:rotate-0" />
            </div>
            <h3 className="text-white font-medium mb-1">实验管理</h3>
            <p className="text-xs text-slate-400">监控训练任务与历史记录。</p>
            </div>

            <div 
            onClick={() => onNavigate?.(ViewState.INFERENCE)}
            className="glass-panel p-5 rounded-xl border border-slate-800/60 cursor-pointer hover:border-rose-500/50 hover:bg-slate-900/60 transition-all group"
            >
            <div className="flex justify-between items-start mb-3">
                <div className="p-2 rounded bg-rose-900/20 text-rose-400 group-hover:bg-rose-900/40 transition-colors">
                <Play size={20} />
                </div>
                <ArrowRight size={18} className="text-slate-600 group-hover:text-rose-400 transition-colors -rotate-45 group-hover:rotate-0" />
            </div>
            <h3 className="text-white font-medium mb-1">推理演示</h3>
            <p className="text-xs text-slate-400">单图、批量与视频流推理。</p>
            </div>
        </div>

        {/* Main Charts Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 glass-panel rounded-xl p-6 border border-slate-800/60">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-slate-200 font-medium flex items-center">
                <Activity className="mr-2 text-cyan-400" size={18} />
                全局资源负载
                </h3>
                <div className="flex space-x-2">
                <span className="flex items-center text-xs text-slate-400"><div className="w-2 h-2 rounded-full bg-cyan-500 mr-2"></div>GPU</span>
                <span className="flex items-center text-xs text-slate-400"><div className="w-2 h-2 rounded-full bg-purple-500 mr-2"></div>内存</span>
                </div>
            </div>
            <div className="h-[250px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                    <defs>
                    <linearGradient id="colorGpu" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorMem" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#a855f7" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                    </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#e2e8f0' }}
                    itemStyle={{ fontSize: '12px' }}
                    cursor={{ stroke: '#334155' }}
                    />
                    <Area type="monotone" dataKey="gpu" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorGpu)" />
                    <Area type="monotone" dataKey="mem" stroke="#a855f7" strokeWidth={2} fillOpacity={1} fill="url(#colorMem)" />
                </AreaChart>
                </ResponsiveContainer>
            </div>
            </div>

            <div className="space-y-4">
            <div className="glass-panel p-5 rounded-xl border border-slate-800/60 h-full flex flex-col">
                <h3 className="text-slate-200 font-medium mb-4 flex items-center">
                <FolderOpen className="mr-2 text-emerald-400" size={16} /> 存储空间分析
                </h3>
                <div className="space-y-4 flex-1">
                    {projectFiles.map((file, idx) => (
                        <div key={idx}>
                            <div className="flex justify-between items-center text-xs mb-1.5">
                                <span className="text-slate-300 font-medium">{file.name}</span>
                                <span className="text-slate-500 font-mono">{file.size}</span>
                            </div>
                            <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                                <div className={`h-full ${file.color} rounded-full`} style={{ width: `${file.percent}%` }}></div>
                            </div>
                        </div>
                    ))}
                </div>
                <button className="w-full mt-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold rounded border border-slate-600/30 transition-all uppercase tracking-wider">
                清理缓存
                </button>
            </div>
            </div>
        </div>

        {/* Stacked Logs Section */}
        <div className="flex flex-col gap-4">
            {/* Backend Logs */}
            <div className="glass-panel rounded-xl border border-slate-800/60 overflow-hidden flex flex-col h-[200px]">
                <div className="px-4 py-2 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
                    <div className="flex items-center">
                        <Server className="mr-2 text-cyan-400" size={16} />
                        <h3 className="text-slate-300 text-sm font-bold">后端系统日志</h3>
                        <span className="ml-3 text-[10px] text-slate-500 font-mono hidden sm:inline">Python 3.10 • CUDA 11.8 • PID: 1042</span>
                    </div>
                    <button 
                        onClick={() => setExpandedLogType('backend')}
                        className="p-1.5 text-slate-500 hover:text-white hover:bg-slate-800 rounded transition-colors"
                        title="放大查看详细日志"
                    >
                        <Maximize2 size={16} />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1 bg-slate-950/50 custom-scrollbar">
                {backendLogs.map((log) => (
                    <div key={log.id} className="flex items-start hover:bg-slate-900/50 p-0.5 rounded cursor-default">
                    <span className="text-slate-500 mr-4 w-20 shrink-0">{log.time}</span>
                    <span className={`w-16 shrink-0 font-bold ${
                        log.level === 'INFO' ? 'text-cyan-600' : 
                        log.level === 'WARN' ? 'text-amber-500' : 
                        'text-rose-500'
                    }`}>{log.level}</span>
                    <span className="text-slate-300">{log.msg}</span>
                    </div>
                ))}
                </div>
            </div>

            {/* Frontend Logs */}
            <div className="glass-panel rounded-xl border border-slate-800/60 overflow-hidden flex flex-col h-[180px]">
                <div className="px-4 py-2 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
                    <div className="flex items-center">
                        <Monitor className="mr-2 text-purple-400" size={16} />
                        <h3 className="text-slate-300 text-sm font-bold">前端系统日志 (Frontend / Client)</h3>
                        <span className="ml-auto text-[10px] text-slate-500 font-mono hidden sm:inline pl-2">React 18 • WebSocket: Connected</span>
                    </div>
                    <button 
                        onClick={() => setExpandedLogType('frontend')}
                        className="p-1.5 text-slate-500 hover:text-white hover:bg-slate-800 rounded transition-colors"
                        title="放大查看详细日志"
                    >
                        <Maximize2 size={16} />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1 bg-slate-950/50 custom-scrollbar">
                {frontendLogs.map((log) => (
                    <div key={log.id} className="flex items-start hover:bg-slate-900/50 p-0.5 rounded cursor-default">
                    <span className="text-slate-500 mr-4 w-20 shrink-0">{log.time}</span>
                    <span className={`w-16 shrink-0 font-bold ${
                        log.level === 'INFO' ? 'text-purple-500' : 
                        log.level === 'WARN' ? 'text-amber-500' : 
                        'text-rose-500'
                    }`}>{log.level}</span>
                    <span className="text-slate-300">{log.msg}</span>
                    </div>
                ))}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;