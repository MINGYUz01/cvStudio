import React, { useState, useMemo, useEffect } from 'react';
import { 
  Upload, 
  Camera, 
  Zap, 
  FileText, 
  Folder, 
  Image, 
  Video, 
  Download, 
  ArrowLeft, 
  X, 
  Layers, 
  Activity, 
  Maximize, 
  Play, 
  Pause, 
  Cpu, 
  Clock,
  RefreshCw,
  ChevronDown,
  CheckCircle,
  AlertTriangle,
  Info,
  StopCircle,
  Lock
} from 'lucide-react';

// MOCK: mirroring the registry in ModelBuilder
const AVAILABLE_WEIGHTS = [
    { id: 'w1', name: 'yolov8n-traffic-best.pt (mAP 0.68)', val: 'w1', type: 'detection' },
    { id: 'w2', name: 'resnet50-mri-v2.pt (Acc 94.2%)', val: 'w2', type: 'classification' },
    { id: 'w3', name: 'yolov8-face-deploy.onnx (mAP 0.72)', val: 'w3', type: 'segmentation' },
];

// MOCK: Camera Devices
const MOCK_CAMERAS = [
    { id: 'cam1', label: 'Integrated Webcam (HD)' },
    { id: 'cam2', label: 'Logitech C920 Pro' },
    { id: 'cam3', label: 'OBS Virtual Camera' }
];

const InferenceView: React.FC = () => {
  const [mode, setMode] = useState<'single' | 'batch' | 'stream'>('single');
  const [selectedBatchImage, setSelectedBatchImage] = useState<number | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>('w1');
  
  // Stream Mode State
  const [streamStatus, setStreamStatus] = useState<'idle' | 'scanning' | 'ready' | 'live'>('idle');
  const [cameras, setCameras] = useState<{id: string, label: string}[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  
  // Notification State
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);

  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Check if controls should be locked
  const isLocked = streamStatus === 'live';

  // Derive task type from selected model
  const taskType = useMemo(() => {
      const w = AVAILABLE_WEIGHTS.find(w => w.val === selectedModelId);
      return w ? w.type : 'detection';
  }, [selectedModelId]);

  // Reset stream state when switching modes
  useEffect(() => {
      if (mode !== 'stream') {
          setStreamStatus('idle');
          setCameras([]);
          setSelectedCamera('');
      }
  }, [mode]);

  // --- Actions ---

  const handleScanCameras = () => {
      setStreamStatus('scanning');
      // Simulate hardware scan delay
      setTimeout(() => {
          setCameras(MOCK_CAMERAS);
          if (MOCK_CAMERAS.length > 0) {
              setSelectedCamera(MOCK_CAMERAS[0].id);
              setStreamStatus('ready');
              showNotification("已识别 3 个视频输入设备", "success");
          } else {
              setStreamStatus('idle');
              showNotification("未检测到摄像头设备", "error");
          }
      }, 1500);
  };

  const handleToggleStream = () => {
      if (streamStatus === 'idle') {
          showNotification("请先连接摄像头设备", "error");
          return;
      }
      if (streamStatus === 'ready') {
          setStreamStatus('live');
          showNotification("视频流推理已启动", "success");
      } else if (streamStatus === 'live') {
          setStreamStatus('ready');
          showNotification("视频流已暂停", "info");
      }
  };

  // --- Dynamic Result Renderer ---
  const renderResults = () => {
      if (taskType === 'detection') {
          return (
            <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                {[
                    { label: 'Car', score: 0.98, color: 'cyan' },
                    { label: 'Person', score: 0.92, color: 'purple' },
                    { label: 'Traffic Light', score: 0.89, color: 'emerald' },
                    { label: 'Car', score: 0.88, color: 'cyan' },
                    { label: 'Bus', score: 0.76, color: 'amber' }
                ].map((det, idx) => (
                    <div key={idx} className="p-3 bg-slate-900/50 border border-slate-800 rounded-lg flex justify-between items-center hover:bg-slate-800 transition-colors cursor-pointer group">
                        <div className="flex items-center">
                            <div className={`w-2 h-2 rounded-full bg-${det.color}-500 mr-3`}></div>
                            <span className="text-sm text-slate-300 font-medium">{det.label}</span>
                        </div>
                        <span className="font-mono text-xs text-slate-500 group-hover:text-white transition-colors">
                            {(det.score * 100).toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>
          );
      } else if (taskType === 'classification') {
          return (
            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {[
                    { label: 'Glioma', score: 0.94 },
                    { label: 'Meningioma', score: 0.04 },
                    { label: 'Pituitary', score: 0.01 },
                    { label: 'No Tumor', score: 0.01 }
                ].map((cls, idx) => (
                    <div key={idx}>
                        <div className="flex justify-between text-xs mb-1">
                            <span className={`font-medium ${idx === 0 ? 'text-white' : 'text-slate-400'}`}>{cls.label}</span>
                            <span className="font-mono text-slate-500">{(cls.score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${idx === 0 ? 'bg-cyan-500' : 'bg-slate-600'}`} style={{ width: `${cls.score * 100}%` }}></div>
                        </div>
                    </div>
                ))}
            </div>
          );
      } else if (taskType === 'segmentation') {
          return (
            <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                <div className="text-xs text-slate-500 uppercase mb-2">Segmentation Masks</div>
                {[
                    { label: 'Road', color: 'bg-gray-500', area: '45%' },
                    { label: 'Vehicle', color: 'bg-blue-500', area: '12%' },
                    { label: 'Pedestrian', color: 'bg-red-500', area: '2%' },
                    { label: 'Vegetation', color: 'bg-green-500', area: '25%' }
                ].map((seg, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 rounded hover:bg-slate-900/50">
                        <div className="flex items-center">
                            <div className={`w-4 h-4 rounded mr-3 ${seg.color} border border-white/10 shadow-sm`}></div>
                            <span className="text-sm text-slate-300">{seg.label}</span>
                        </div>
                        <span className="text-xs font-mono text-slate-500">{seg.area}</span>
                    </div>
                ))}
            </div>
          );
      }
  };

  return (
    <div className="h-full flex flex-col p-6 gap-6 relative">
        {/* Local Notification */}
        {notification && (
            <div className={`absolute top-6 left-1/2 -translate-x-1/2 z-[50] px-4 py-2 rounded-lg shadow-lg border flex items-center animate-in fade-in slide-in-from-top-2 duration-200 ${
                notification.type === 'error' ? 'bg-rose-900/90 border-rose-500 text-white' : 
                notification.type === 'success' ? 'bg-emerald-900/90 border-emerald-500 text-white' :
                'bg-cyan-900/90 border-cyan-500 text-white'
            }`}>
            {notification.type === 'error' ? <AlertTriangle size={16} className="mr-2" /> : 
             notification.type === 'success' ? <CheckCircle size={16} className="mr-2" /> :
             <Info size={16} className="mr-2" />
            }
            <span className="text-sm font-medium">{notification.msg}</span>
            </div>
        )}

       {/* Top Bar: Controls & Model Select */}
       <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4 z-40">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-bold text-white mr-4">推理实验室</h2>
            
            {/* Mode Switcher */}
            <div className={`flex bg-slate-900 rounded-lg p-1 border border-slate-800 ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
               <button 
                  onClick={() => { setMode('single'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'single' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Image size={16} className="mr-2" /> 单图
               </button>
               <button 
                  onClick={() => { setMode('batch'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'batch' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Folder size={16} className="mr-2" /> 批量文件夹
               </button>
               <button 
                  onClick={() => { setMode('stream'); setSelectedBatchImage(null); }}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-colors ${mode === 'stream' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
               >
                  <Video size={16} className="mr-2" /> 视频流
               </button>
            </div>

            {/* Batch Back Button - Repositioned Here */}
            {mode === 'batch' && selectedBatchImage !== null && (
                <div className="h-8 w-px bg-slate-800 mx-2"></div>
            )}
            {mode === 'batch' && selectedBatchImage !== null && (
               <button onClick={() => setSelectedBatchImage(null)} className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded-lg border border-slate-700 flex items-center transition-all">
                  <ArrowLeft size={14} className="mr-2" /> 返回概览
               </button>
            )}
          </div>

          <div className="flex items-center space-x-3 w-full md:w-auto">
             <div className={`relative ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
                <label className="absolute -top-2 left-2 bg-slate-900 px-1 text-[10px] text-slate-500">加载权重</label>
                <select 
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    disabled={isLocked}
                    className="flex-1 md:w-64 bg-slate-900 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 outline-none focus:border-cyan-500 cursor-pointer disabled:cursor-not-allowed"
                >
                    {AVAILABLE_WEIGHTS.map(w => (
                        <option key={w.id} value={w.val}>{w.name}</option>
                    ))}
                </select>
                {isLocked && <Lock size={12} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500" />}
             </div>
             
             {/* Action Button based on mode */}
             {mode === 'single' && (
               <button className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors">
                  <Upload size={16} className="mr-2" /> 上传图片
               </button>
             )}
             {mode === 'batch' && !selectedBatchImage && (
               <button className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors">
                  <Folder size={16} className="mr-2" /> 选择文件夹
               </button>
             )}
             
             {/* Video Stream Controls */}
             {mode === 'stream' && (
               <div className="flex items-center space-x-2">
                   {/* Connection Button / Dropdown */}
                   {streamStatus === 'idle' || streamStatus === 'scanning' ? (
                       <button 
                            onClick={handleScanCameras}
                            disabled={streamStatus === 'scanning'}
                            className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-600 transition-colors"
                       >
                            {streamStatus === 'scanning' ? (
                                <><RefreshCw size={16} className="mr-2 animate-spin" /> 正在识别...</>
                            ) : (
                                <><Camera size={16} className="mr-2" /> 连接摄像头</>
                            )}
                       </button>
                   ) : (
                       <div className={`relative ${isLocked ? 'opacity-50 pointer-events-none' : ''}`}>
                           <select 
                                value={selectedCamera}
                                onChange={(e) => setSelectedCamera(e.target.value)}
                                disabled={isLocked}
                                className="appearance-none bg-slate-900 border border-emerald-500/50 text-emerald-400 text-sm rounded-lg pl-3 pr-8 py-2 outline-none focus:border-emerald-400 cursor-pointer disabled:cursor-not-allowed"
                           >
                               {cameras.map(cam => (
                                   <option key={cam.id} value={cam.id}>{cam.label}</option>
                               ))}
                           </select>
                           <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-emerald-500 pointer-events-none" />
                       </div>
                   )}

                   {/* Start/Stop Controls (Only when ready/live) */}
                   {(streamStatus === 'ready' || streamStatus === 'live') && (
                       <button 
                            onClick={handleToggleStream}
                            className={`flex items-center px-4 py-2 text-white text-sm font-bold rounded-lg shadow-lg transition-all ${
                                streamStatus === 'live' 
                                ? 'bg-amber-600 hover:bg-amber-500 shadow-amber-900/20' 
                                : 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-900/20'
                            }`}
                       >
                            {streamStatus === 'live' ? (
                                <><Pause size={16} className="mr-2 fill-current" /> 暂停推理</>
                            ) : (
                                <><Play size={16} className="mr-2 fill-current" /> 开始推理</>
                            )}
                       </button>
                   )}
               </div>
             )}
          </div>
       </div>

       <div className="flex-1 flex gap-6 min-h-0">
          {/* Main Viewport */}
          <div className="flex-1 flex flex-col gap-4 min-h-0">
             <div className="flex-1 glass-panel rounded-xl border border-slate-800/60 p-4 flex items-center justify-center relative overflow-hidden bg-slate-950">
                
                {/* Content Area */}
                {mode === 'batch' && selectedBatchImage === null ? (
                    <div className="w-full h-full overflow-y-auto grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-4 content-start custom-scrollbar">
                        {Array.from({length: 20}).map((_, i) => (
                        <div 
                            key={i} 
                            onClick={() => setSelectedBatchImage(i)}
                            className="aspect-square bg-slate-900 rounded border border-slate-800 relative group cursor-pointer hover:border-cyan-500 transition-all hover:scale-105"
                        >
                            <img src={`https://picsum.photos/200/200?random=${i+100}`} className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity" />
                            <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.5)]"></div>
                        </div>
                        ))}
                    </div>
                ) : (
                    <div className="relative w-full h-full flex items-center justify-center">
                        {/* STREAM MODE PLACEHOLDER / FEED */}
                        {mode === 'stream' ? (
                            streamStatus === 'idle' || streamStatus === 'scanning' ? (
                                <div className="flex flex-col items-center text-slate-600">
                                    <Camera size={64} className="mb-4 opacity-20" />
                                    <p className="text-sm">未连接视频输入源</p>
                                </div>
                            ) : (
                                <div className="relative w-full h-full flex items-center justify-center bg-black/50 rounded-lg overflow-hidden border border-slate-800">
                                    <img src={`https://picsum.photos/1280/720?random=stream`} className="w-full h-full object-cover opacity-80" />
                                    {streamStatus === 'live' && (
                                        <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-red-500/30">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_#ef4444]"></div>
                                            <span className="text-xs font-bold text-red-400">LIVE INFERENCE</span>
                                        </div>
                                    )}
                                    {/* Status LED Indicator */}
                                    <div className="absolute top-4 left-4 flex items-center space-x-2">
                                        <div className={`w-3 h-3 rounded-full shadow-lg border border-white/10 ${
                                            streamStatus === 'live' ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' : 
                                            streamStatus === 'ready' ? 'bg-amber-500 shadow-[0_0_5px_#f59e0b]' : 'bg-slate-600'
                                        }`}></div>
                                        <span className="text-xs font-mono text-slate-300 bg-black/40 px-2 py-0.5 rounded">
                                            {streamStatus === 'live' ? 'Connected: Streaming' : 'Connected: Standby'}
                                        </span>
                                    </div>
                                </div>
                            )
                        ) : (
                            // IMAGE / BATCH SINGLE VIEW
                            <div className="relative w-full h-full flex items-center justify-center">
                                <img src={`https://picsum.photos/800/600?random=${mode === 'batch' ? (selectedBatchImage || 0) + 100 : 1}`} className="max-w-full max-h-full object-contain" alt="Inference Input" />
                                {/* Only show bounding box overlay if task is detection */}
                                {taskType === 'detection' && (
                                    <div className="absolute top-[30%] left-[20%] w-[15%] h-[20%] border-2 border-cyan-400 bg-cyan-400/10 hover:bg-cyan-400/20 transition-colors cursor-pointer group">
                                        <span className="absolute -top-6 left-0 bg-cyan-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded">Car 0.98</span>
                                    </div>
                                )}
                                {/* Overlay for Segmentation (Simulated) */}
                                {taskType === 'segmentation' && (
                                    <div className="absolute top-[30%] left-[20%] w-[15%] h-[20%] bg-blue-500/30 mix-blend-overlay"></div>
                                )}
                            </div>
                        )}
                    </div>
                )}
             </div>
             
             {/* Info Strip: Metrics (Single/Batch) OR Stream Stats */}
             {(mode !== 'batch' || selectedBatchImage !== null) && (
                <div className="h-20 glass-panel rounded-xl border border-slate-800 flex items-center justify-around px-6 shrink-0">
                    <div className="text-center group">
                        <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-purple-400 transition-colors">
                            <Cpu size={12} className="mr-1.5" /> 推理设备
                        </div>
                        <div className="text-lg font-mono text-purple-400 font-bold">NVIDIA RTX 4090</div>
                    </div>
                    <div className="w-px h-10 bg-slate-800"></div>
                    <div className="text-center group">
                        <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-cyan-400 transition-colors">
                            <Clock size={12} className="mr-1.5" /> 推理耗时
                        </div>
                        <div className="text-xl font-mono text-cyan-400">14.2<span className="text-sm text-slate-600 ml-1">ms</span></div>
                    </div>
                    {/* Additional stat for Stream Mode */}
                    {mode === 'stream' && (
                        <>
                            <div className="w-px h-10 bg-slate-800"></div>
                            <div className="text-center group">
                                <div className="text-xs text-slate-500 uppercase mb-1 flex items-center justify-center group-hover:text-emerald-400 transition-colors">
                                    <Activity size={12} className="mr-1.5" /> 帧率 (FPS)
                                </div>
                                <div className="text-xl font-mono text-emerald-400">
                                    {streamStatus === 'live' ? '62' : '-'}
                                </div>
                            </div>
                        </>
                    )}
                </div>
             )}
          </div>

          {/* Results Sidebar (Standardized) */}
          {(mode !== 'batch' || selectedBatchImage !== null) && (
            <div className="w-80 bg-slate-950 border-l border-slate-800 rounded-xl flex flex-col overflow-hidden shrink-0">
               <div className="p-4 border-b border-slate-800 bg-slate-900/50">
                 <h3 className="font-bold text-white flex items-center capitalize">
                   {taskType === 'detection' && <FileText size={18} className="mr-2 text-slate-400" />}
                   {taskType === 'classification' && <Activity size={18} className="mr-2 text-slate-400" />}
                   {taskType === 'segmentation' && <Layers size={18} className="mr-2 text-slate-400" />}
                   {taskType} Results
                 </h3>
               </div>
               
               {/* Dynamic Result List */}
               {renderResults()}

               {/* Unified Download Area (Conditionally Hidden for Stream) */}
               {mode !== 'stream' && (
                   <div className="p-4 border-t border-slate-800 space-y-2">
                     <button className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold rounded border border-slate-700 transition-colors flex items-center justify-center">
                       <Download size={14} className="mr-2" /> 下载检测结果 (JSON)
                     </button>
                     <button className="w-full py-2 bg-cyan-900/30 hover:bg-cyan-900/50 text-cyan-400 text-xs font-bold rounded border border-cyan-800 transition-colors flex items-center justify-center">
                       <Image size={14} className="mr-2" /> 下载标注图片
                     </button>
                   </div>
               )}
            </div>
          )}
       </div>
    </div>
  );
};

export default InferenceView;