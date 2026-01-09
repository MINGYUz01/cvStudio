import React, { useState, useEffect, useRef } from 'react';
import {
  Folder,
  Search,
  Layout,
  X,
  Eye,
  Calendar,
  HardDrive,
  Upload,
  ChevronDown,
  Plus,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { DatasetItem } from '../types';
import { useDataset } from '../src/hooks/useDataset';
import { adaptDatasetList } from '../src/services/datasetAdapter';
import { datasetService } from '../src/services/datasets';
import { apiClient } from '../src/services/api';

// --- Modern Lightbox Component ---
const Lightbox: React.FC<{ src: string, onClose: () => void }> = ({ src, onClose }) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const startPos = useRef({ x: 0, y: 0 });

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    const delta = -e.deltaY * 0.001;
    setScale(s => Math.min(Math.max(0.5, s + delta), 5));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    startPos.current = { x: e.clientX - position.x, y: e.clientY - position.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    setPosition({
      x: e.clientX - startPos.current.x,
      y: e.clientY - startPos.current.y
    });
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  return (
    <div 
      className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-sm flex items-center justify-center overflow-hidden" 
      onWheel={handleWheel}
      onClick={onClose}
    >
      <button 
        onClick={onClose} 
        className="absolute top-6 right-6 p-2 text-slate-400 hover:text-white transition-colors z-[101]"
      >
        <X size={32} />
      </button>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-slate-500 text-xs pointer-events-none bg-black/50 px-3 py-1 rounded-full">
        滚轮缩放 • 拖拽平移
      </div>

      <img 
        src={src} 
        className="max-w-none transition-transform duration-75 cursor-move" 
        style={{ 
          transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
          maxHeight: '90vh',
          maxWidth: '90vw'
        }} 
        onClick={(e) => e.stopPropagation()}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDragStart={(e) => e.preventDefault()}
      />
    </div>
  );
};

const DatasetManager: React.FC = () => {
  // 使用数据集Hook获取真实数据
  const { datasets: apiDatasets, loading, error, fetchDatasets } = useDataset();

  // 使用适配器转换API数据为组件格式
  const datasets: DatasetItem[] = adaptDatasetList(apiDatasets);

  const [selectedDsId, setSelectedDsId] = useState<string>('');
  const [lightboxImg, setLightboxImg] = useState<string | null>(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  // 当数据集列表加载完成后，设置默认选中项
  useEffect(() => {
    if (datasets.length > 0 && !selectedDsId) {
      setSelectedDsId(datasets[0].id);
    }
  }, [datasets, selectedDsId]);

  // 当选中的数据集改变时，获取图片URL列表
  useEffect(() => {
    if (selectedDsId && apiDatasets.length > 0) {
      const selectedDataset = apiDatasets.find((ds: any) => ds.id === parseInt(selectedDsId));
      if (selectedDataset) {
        // 从元数据中获取图片路径，然后构建URL
        const imagePaths = selectedDataset.meta?.image_paths || [];
        const urls = imagePaths.map((_: string, index: number) =>
          `${apiClient['baseURL']}/datasets/${selectedDataset.id}/image-file?index=${index}`
        );
        setImageUrls(urls);
      }
    }
  }, [selectedDsId, apiDatasets]);

  // Pagination State for Samples
  const [visibleSamples, setVisibleSamples] = useState(24);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Reset pagination when dataset changes
  useEffect(() => {
    setVisibleSamples(24);
  }, [selectedDsId]);

  const handleLoadMore = () => {
    setIsLoadingMore(true);
    // Simulate network delay
    setTimeout(() => {
      setVisibleSamples(prev => prev + 24);
      setIsLoadingMore(false);
    }, 500);
  };

  // 加载状态
  if (loading && datasets.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <div className="w-12 h-12 rounded-full border-4 border-cyan-500 border-t-transparent animate-spin"></div>
        <p className="text-slate-400 text-sm">正在加载数据集...</p>
      </div>
    );
  }

  // 错误状态
  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <AlertCircle size={48} className="text-red-500" />
        <p className="text-red-400 text-sm">{error}</p>
        <button
          onClick={fetchDatasets}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm flex items-center"
        >
          <RefreshCw size={16} className="mr-2" />
          重试
        </button>
      </div>
    );
  }

  // 空状态
  if (datasets.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 space-y-6">
        <Folder size={48} className="text-slate-600" />
        <p className="text-slate-400 text-sm">暂无数据集</p>
        <p className="text-slate-500 text-xs">点击下方"导入数据集"按钮添加您的第一个数据集</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6 space-y-6">
      <div className="flex justify-between items-center">
         <div>
           <h2 className="text-2xl font-bold text-white">数据集管理</h2>
           <p className="text-slate-400 text-sm">查看原始数据分布与元信息。</p>
         </div>
         <div className="flex gap-2">
           <button
             onClick={fetchDatasets}
             disabled={loading}
             className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-xl border border-slate-700 transition-all disabled:opacity-50"
             title="刷新列表"
           >
              <RefreshCw size={18} className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
              刷新
           </button>
           <button className="flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 transition-all">
              <Upload size={18} className="mr-2" /> 导入数据
           </button>
         </div>
      </div>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Left List */}
        <div className="w-full md:w-1/4 flex flex-col gap-4 min-h-0">
           <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
              <input type="text" placeholder="搜索数据集..." className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500" />
           </div>
           <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              {datasets.map((ds) => (
                <div 
                  key={ds.id}
                  onClick={() => setSelectedDsId(ds.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all relative overflow-hidden group ${
                    selectedDsId === ds.id 
                      ? 'bg-cyan-950/20 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]' 
                      : 'bg-slate-900/40 border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/80'
                  }`}
                >
                  <div className="flex items-center mb-2">
                    <div className="group-hover:scale-110 transition-transform origin-left">
                       <Folder size={16} className={`${selectedDsId === ds.id ? 'text-cyan-400' : 'text-slate-500 group-hover:text-cyan-400 transition-colors'}`} />
                    </div>
                    <span className={`ml-2 text-sm font-medium truncate transition-colors ${selectedDsId === ds.id ? 'text-white' : 'text-slate-200 group-hover:text-cyan-400'}`}>{ds.name}</span>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span className="bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">{ds.type}</span>
                    <span>{ds.count.toLocaleString()} imgs</span>
                  </div>
                  
                  {/* Gradient Glow Effect */}
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-cyan-500 to-purple-600 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
                </div>
              ))}
              
              {/* Import New Dataset Card */}
              <div 
                className="p-4 rounded-lg border-2 border-dashed border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/40 cursor-pointer transition-all flex flex-col items-center justify-center text-slate-500 hover:text-cyan-400 group"
              >
                  <Plus size={24} className="mb-2 group-hover:scale-110 transition-transform" />
                  <span className="text-xs font-medium">导入数据集</span>
              </div>
           </div>
        </div>

        {/* Right Content */}
        <div className="flex-1 glass-panel border border-slate-800 rounded-xl overflow-hidden flex flex-col">
           <div className="h-14 border-b border-slate-800 bg-slate-900/50 flex items-center px-6 justify-between">
              <h3 className="font-bold text-white flex items-center">
                <Layout size={18} className="mr-2 text-cyan-400" /> {datasets.find(d => d.id === selectedDsId)?.name}
              </h3>
              <div className="flex items-center space-x-4 text-xs text-slate-400 font-mono">
                 <span className="flex items-center"><HardDrive size={12} className="mr-1"/> {datasets.find(d => d.id === selectedDsId)?.size}</span>
                 <span className="flex items-center"><Calendar size={12} className="mr-1"/> {datasets.find(d => d.id === selectedDsId)?.lastModified}</span>
              </div>
           </div>
           
           <div className="p-6 overflow-y-auto custom-scrollbar">
               {/* Metadata Cards */}
               <div className="grid grid-cols-4 gap-4 mb-8">
                 <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                     <div className="text-2xl font-mono text-white">{datasets.find(d => d.id === selectedDsId)?.count.toLocaleString()}</div>
                     <div className="text-xs text-slate-500 uppercase mt-1">样本总数</div>
                 </div>
                 <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                     <div className="text-2xl font-mono text-emerald-400">Balanced</div>
                     <div className="text-xs text-slate-500 uppercase mt-1">类别分布</div>
                 </div>
                 <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                     <div className="text-2xl font-mono text-cyan-400">640x640</div>
                     <div className="text-xs text-slate-500 uppercase mt-1">平均分辨率</div>
                 </div>
                 <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-center">
                     <div className="text-2xl font-mono text-purple-400">0.05%</div>
                     <div className="text-xs text-slate-500 uppercase mt-1">空标注率</div>
                 </div>
               </div>
               
               {/* Gallery */}
               <div className="pb-4">
                  <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center">
                    <Eye size={16} className="mr-2" /> 样本概览
                  </h3>
                  <div className="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3 mb-6">
                    {imageUrls.slice(0, visibleSamples).length > 0 ? (
                      imageUrls.slice(0, visibleSamples).map((imgUrl, i) => (
                        <div key={i} onClick={() => setLightboxImg(imgUrl)} className="aspect-square bg-slate-800 rounded border border-slate-800 overflow-hidden relative group cursor-pointer hover:border-cyan-500 transition-all duration-200">
                          <img
                            src={imgUrl}
                            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                            onError={(e) => {
                              // 如果图片加载失败，使用占位符
                              (e.target as HTMLImageElement).src = `https://picsum.photos/600/600?random=${i}`;
                            }}
                            alt={`Sample ${i + 1}`}
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                        </div>
                      ))
                    ) : (
                      // 如果没有图片，显示占位符
                      Array.from({length: Math.min(visibleSamples, 24)}).map((_, i) => (
                        <div key={i} className="aspect-square bg-slate-800 rounded border border-slate-800 overflow-hidden relative flex items-center justify-center">
                          <span className="text-slate-600 text-xs">无图片</span>
                        </div>
                      ))
                    )}
                  </div>

                  {/* Load More Button - 只在有更多图片时显示 */}
                  {imageUrls.length > visibleSamples && (
                    <div className="flex justify-center">
                      <button
                        onClick={handleLoadMore}
                        disabled={isLoadingMore}
                        className="px-6 py-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white rounded-full border border-slate-700 transition-all text-xs font-medium flex items-center disabled:opacity-50"
                      >
                        {isLoadingMore ? (
                           <span className="flex items-center"><div className="w-3 h-3 rounded-full border-2 border-slate-500 border-t-transparent animate-spin mr-2"></div> Loading...</span>
                        ) : (
                           <span className="flex items-center"><ChevronDown size={14} className="mr-1" /> 加载更多 ({imageUrls.length - visibleSamples})</span>
                        )}
                      </button>
                    </div>
                  )}
               </div>
           </div>
        </div>
      </div>

      {lightboxImg && <Lightbox src={lightboxImg} onClose={() => setLightboxImg(null)} />}
    </div>
  );
};

export default DatasetManager;