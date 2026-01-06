import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Hexagon, ArrowRight, Lock, Mail, Loader2, AlertCircle,
  Cpu, Database, Activity, Network, Server, Code, Terminal,
  Zap, Shield, Globe, Smartphone, Cloud, Search, Command,
  Layers, GitBranch, Box, HardDrive, Key
} from 'lucide-react';
import { useAuth } from '../hooks/useAuth';

interface LoginProps {
  onLogin?: () => void;
}

// The "Stars" of our Digital Universe
const TECH_ICONS = [
  Cpu, Database, Network, Server, Code, Terminal, 
  Activity, Layers, GitBranch, Box, HardDrive, 
  Cloud, Shield, Zap, Globe, Command
];

interface FloatingIconData {
  id: number;
  Icon: React.ElementType;
  x: number; // percentage
  y: number; // percentage
  size: number;
  duration: number;
  delay: number;
  rotation: number;
  color: string;
  opacity: number;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [mounted, setMounted] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // 获取登录后要跳转的路径
  const from = (location.state as any)?.from || '/dashboard';
  
  // For "Icon Galaxy" background
  const [backgroundIcons, setBackgroundIcons] = useState<FloatingIconData[]>([]);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // For Card Spotlight Effect
  const cardRef = useRef<HTMLDivElement>(null);
  const [cardMousePos, setCardMousePos] = useState({ x: 0, y: 0 });
  const [isCardHovered, setIsCardHovered] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    // Generate random icons for the background
    const icons: FloatingIconData[] = [];
    const colors = ['text-cyan-500', 'text-purple-500', 'text-blue-500', 'text-emerald-500', 'text-slate-600'];
    
    for (let i = 0; i < 40; i++) {
      icons.push({
        id: i,
        Icon: TECH_ICONS[Math.floor(Math.random() * TECH_ICONS.length)],
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 24 + 12, // 12px to 36px
        duration: Math.random() * 20 + 10, // 10s to 30s float duration
        delay: Math.random() * 5,
        rotation: Math.random() * 360,
        color: colors[Math.floor(Math.random() * colors.length)],
        opacity: Math.random() * 0.3 + 0.05 // 0.05 to 0.35 opacity
      });
    }
    setBackgroundIcons(icons);
  }, []);

  const handleGlobalMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    const x = (clientX / innerWidth) * 2 - 1;
    const y = (clientY / innerHeight) * 2 - 1;
    setMousePos({ x, y });
  };

  const handleCardMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setCardMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // 使用真实的API登录
      await login({ username, password });

      // 登录成功，调用回调并跳转
      if (onLogin) {
        onLogin();
      }

      // 跳转到之前的页面或默认页面
      navigate(from, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : '登录失败，请检查用户名和密码';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
      ref={containerRef}
      onMouseMove={handleGlobalMouseMove}
      className="min-h-screen bg-slate-950 flex items-center justify-center relative overflow-hidden selection:bg-cyan-500/30 selection:text-cyan-200"
    >
      
      {/* --- Digital Cosmos Background --- */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        {/* Deep Space Gradient */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black"></div>
        
        {/* Grid Floor (Perspective) */}
        <div 
            className="absolute bottom-0 left-[-50%] right-[-50%] h-[500px] bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] opacity-20"
            style={{ 
                transform: 'perspective(500px) rotateX(60deg) translateY(100px) translateZ(-200px)',
                maskImage: 'linear-gradient(to top, black, transparent)' 
            }}
        ></div>

        {/* Floating Icons (The "Stars") */}
        {mounted && backgroundIcons.map((item) => (
          <div
            key={item.id}
            className={`absolute ${item.color} transition-transform duration-100 ease-out`}
            style={{
              left: `${item.x}%`,
              top: `${item.y}%`,
              opacity: item.opacity,
              transform: `translate(${mousePos.x * (item.size * 1.5)}px, ${mousePos.y * (item.size * 1.5)}px) rotate(${item.rotation}deg)`,
            }}
          >
             {/* Animation Wrapper for independent floating */}
             <div 
                style={{ 
                    animation: `float ${item.duration}s ease-in-out infinite alternate`,
                    animationDelay: `${item.delay}s`
                }}
             >
                <item.Icon size={item.size} />
             </div>
          </div>
        ))}
        
        {/* Ambient Glow Orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[100px] animate-pulse" style={{ animationDuration: '4s' }}></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[100px] animate-pulse" style={{ animationDuration: '7s' }}></div>
      </div>

      {/* --- Login Card --- */}
      <div className={`w-full max-w-md relative z-10 transition-all duration-1000 transform ${mounted ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
        
        {/* Logo Section */}
        <div className="flex flex-col items-center mb-8">
            <div className="w-20 h-20 bg-slate-900/80 backdrop-blur-md rounded-2xl border border-cyan-500/30 flex items-center justify-center mb-6 shadow-[0_0_40px_rgba(6,182,212,0.2)] relative group overflow-hidden">
                {/* Internal Scan Effect */}
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-400/10 to-transparent translate-y-[-100%] group-hover:translate-y-[100%] transition-transform duration-1000"></div>
                <div className="absolute inset-0 bg-cyan-500/10 blur-xl rounded-full group-hover:bg-cyan-500/20 transition-all duration-500"></div>
                <Hexagon className="text-cyan-400 relative z-10 group-hover:rotate-180 transition-transform duration-700" size={40} strokeWidth={2} />
            </div>
            <h1 className="text-4xl font-bold tracking-wider text-white font-mono mb-2 drop-shadow-lg">NEURO<span className="text-cyan-400">CORE</span></h1>
            <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <p className="text-slate-400 text-sm tracking-[0.2em] uppercase">Deep Learning Studio</p>
            </div>
        </div>

        {/* Form Panel Wrapper with Spotlight */}
        <div 
          ref={cardRef}
          onMouseMove={handleCardMouseMove}
          onMouseEnter={() => setIsCardHovered(true)}
          onMouseLeave={() => setIsCardHovered(false)}
          className="relative group"
        >
            {/* 1. Dynamic Border Glow (Behind the card) */}
            <div 
                className="absolute -inset-[1.5px] rounded-2xl transition-opacity duration-300 pointer-events-none blur-[1px]"
                style={{
                    opacity: isCardHovered ? 1 : 0,
                    background: `radial-gradient(600px circle at ${cardMousePos.x}px ${cardMousePos.y}px, rgba(34,211,238,0.4), transparent 40%)`
                }}
            />

            {/* Main Card Surface */}
            <div className="glass-panel p-8 rounded-2xl border border-slate-700/50 shadow-2xl backdrop-blur-xl bg-slate-900/80 relative overflow-hidden">
            
               {/* 2. Dynamic Surface Spotlight (Inside the card) */}
               <div 
                   className="absolute inset-0 transition-opacity duration-300 pointer-events-none"
                   style={{
                       opacity: isCardHovered ? 1 : 0,
                       background: `radial-gradient(600px circle at ${cardMousePos.x}px ${cardMousePos.y}px, rgba(34,211,238,0.06), transparent 40%)`
                   }}
               />
               
               <form onSubmit={handleSubmit} className="space-y-6 relative z-10">

                  <div className="space-y-2">
                     <label className="text-xs font-bold text-slate-300 uppercase ml-1 flex items-center">
                        <Terminal size={12} className="mr-1.5 text-cyan-400" /> Account ID
                     </label>
                     <div className="relative group/input">
                        <div className="absolute inset-0 bg-cyan-500/5 rounded-xl opacity-0 group-hover/input:opacity-100 transition-opacity pointer-events-none"></div>
                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within/input:text-cyan-400 transition-colors" size={18} />
                        <input
                          type="text"
                          value={username}
                          onChange={(e) => setUsername(e.target.value)}
                          className="w-full bg-slate-950/60 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-white placeholder-slate-600 outline-none focus:border-cyan-500 focus:bg-slate-900/80 transition-all font-mono text-sm"
                          placeholder="Enter username..."
                        />
                     </div>
                  </div>

                  <div className="space-y-2">
                     <label className="text-xs font-bold text-slate-300 uppercase ml-1 flex items-center">
                        <Key size={12} className="mr-1.5 text-purple-400" /> Security Token
                     </label>
                     <div className="relative group/input">
                        <div className="absolute inset-0 bg-purple-500/5 rounded-xl opacity-0 group-hover/input:opacity-100 transition-opacity pointer-events-none"></div>
                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within/input:text-purple-400 transition-colors" size={18} />
                        <input 
                          type="password" 
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          className="w-full bg-slate-950/60 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-white placeholder-slate-600 outline-none focus:border-purple-500 focus:bg-slate-900/80 transition-all font-mono text-sm"
                          placeholder="••••••••"
                        />
                     </div>
                  </div>

                  {error && (
                    <div className="flex items-center text-rose-400 text-xs bg-rose-950/30 p-3 rounded-lg border border-rose-900/50 animate-in fade-in slide-in-from-top-1">
                       <AlertCircle size={14} className="mr-2 shrink-0" />
                       {error}
                    </div>
                  )}

                  <button 
                    type="submit" 
                    disabled={isLoading}
                    className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-3.5 rounded-xl shadow-lg shadow-cyan-900/20 transition-all active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center group/btn relative overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-white/20 translate-y-full group-hover/btn:translate-y-0 transition-transform duration-300 skew-y-12"></div>
                    <div className="relative flex items-center">
                        {isLoading ? (
                        <Loader2 size={20} className="animate-spin text-white/80" />
                        ) : (
                        <>
                            Initialize System <ArrowRight size={18} className="ml-2 group-hover/btn:translate-x-1 transition-transform" />
                        </>
                        )}
                    </div>
                  </button>
               </form>
            </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center space-y-3 relative z-10">
           <div className="flex justify-center space-x-6 text-[10px] text-slate-600 uppercase tracking-wider font-mono">
              <span className="flex items-center"><Server size={10} className="mr-1"/> Node: <span className="text-emerald-500 ml-1">Active</span></span>
              <span className="flex items-center"><Activity size={10} className="mr-1"/> Ping: <span className="text-cyan-500 ml-1">24ms</span></span>
           </div>
           <p className="text-[10px] text-slate-700/50">© 2024 NeuroCore Institute. Security Level 5.</p>
        </div>

      </div>
      
      {/* Global CSS for floating animation */}
      <style>{`
        @keyframes float {
          0% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(5deg); }
          100% { transform: translateY(0px) rotate(0deg); }
        }
      `}</style>
    </div>
  );
};

export default Login;