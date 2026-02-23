import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Activity, 
  Wifi, 
  Database, 
  Server, 
  Terminal,
  Zap
} from 'lucide-react';

const LOG_MESSAGES = [
    "Syncing checkpoints to cloud storage...",
    "Optimizer state updated (AdamW)",
    "GC: Freed 24MB memory",
    "Heartbeat signal sent to master node",
    "WebSocket latency: 24ms",
    "GPU Temperature normalized",
    "Batch 402 processed in 12ms",
    "Auto-save triggered",
    "Checking for dataset updates...",
    "Inference queue: 0 pending"
];

const GlobalStatusBar: React.FC = () => {
    // Simulated Metrics
    const [gpuUtil, setGpuUtil] = useState(42);
    const [gpuTemp, setGpuTemp] = useState(65);
    const [vram, setVram] = useState(12.4);
    const [logMsg, setLogMsg] = useState("System initialized.");
    const [ping, setPing] = useState(24);

    // Simulation Effect
    useEffect(() => {
        const interval = setInterval(() => {
            // Randomize metrics slightly
            setGpuUtil(prev => Math.min(99, Math.max(10, prev + (Math.random() - 0.5) * 10)));
            setGpuTemp(prev => Math.min(85, Math.max(50, prev + (Math.random() - 0.5) * 4)));
            setVram(prev => Math.min(24, Math.max(8, prev + (Math.random() - 0.5) * 0.5)));
            setPing(prev => Math.floor(20 + Math.random() * 15));

            // Random log message occasionally
            if (Math.random() > 0.7) {
                setLogMsg(LOG_MESSAGES[Math.floor(Math.random() * LOG_MESSAGES.length)]);
            }
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const getStatusColor = (val: number, type: 'util' | 'temp') => {
        if (type === 'util') return val > 80 ? 'text-amber-400' : 'text-cyan-400';
        if (type === 'temp') return val > 80 ? 'text-rose-400' : 'text-emerald-400';
        return 'text-slate-400';
    };

    return (
        <div className="h-8 bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 select-none shrink-0 relative z-50">
            {/* Left Section: Resource Monitors */}
            <div className="flex items-center space-x-6 text-[10px] font-mono">
                {/* GPU */}
                <div className="flex items-center space-x-2 group cursor-help" title="NVIDIA RTX 4090">
                    <div className={`w-1.5 h-1.5 rounded-full ${getStatusColor(gpuUtil, 'util')} animate-pulse`}></div>
                    <Cpu size={12} className="text-slate-500" />
                    <span className="text-slate-400">GPU:</span>
                    <span className={`font-bold ${getStatusColor(gpuUtil, 'util')}`}>{gpuUtil.toFixed(0)}%</span>
                    <span className="text-slate-600">|</span>
                    <span className={`${getStatusColor(gpuTemp, 'temp')}`}>{gpuTemp.toFixed(0)}°C</span>
                </div>

                {/* VRAM */}
                <div className="flex items-center space-x-2 group cursor-help" title="VRAM Usage">
                    <Activity size={12} className="text-slate-500" />
                    <span className="text-slate-400">VRAM:</span>
                    <span className="text-purple-400 font-bold">{vram.toFixed(1)} GB</span>
                    <span className="text-slate-600">/ 24 GB</span>
                </div>

                {/* Network */}
                <div className="hidden sm:flex items-center space-x-2">
                    <Wifi size={12} className="text-slate-500" />
                    <span className="text-slate-400">PING:</span>
                    <span className="text-emerald-400">{ping}ms</span>
                </div>

                 {/* Storage */}
                 <div className="hidden md:flex items-center space-x-2">
                    <Database size={12} className="text-slate-500" />
                    <span className="text-slate-400">SSD:</span>
                    <span className="text-slate-300">42%</span>
                </div>
            </div>

            {/* Right Section: Mini Log & Status */}
            <div className="flex items-center space-x-6">
                
                {/* Mini Log Stream */}
                <div className="hidden lg:flex items-center max-w-md overflow-hidden">
                    <Terminal size={10} className="text-slate-600 mr-2 shrink-0" />
                    <span className="text-[10px] font-mono text-slate-500 truncate animate-in fade-in duration-300 key={logMsg}">
                        {`> ${logMsg}`}
                    </span>
                </div>

                <div className="h-4 w-px bg-slate-800 mx-2 hidden lg:block"></div>

                {/* Connection Status */}
                <div className="flex items-center space-x-2 text-[10px]">
                    <div className="flex items-center text-slate-400">
                        <Server size={10} className="mr-1.5" />
                        <span className="hidden sm:inline">NeuroCore Cloud</span>
                    </div>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-950/50 border border-emerald-900 text-emerald-500 font-bold tracking-wider">
                        ONLINE
                    </span>
                </div>
            </div>
        </div>
    );
};

export default GlobalStatusBar;