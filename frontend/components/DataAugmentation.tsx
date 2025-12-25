
import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Trash2, 
  Save, 
  RefreshCw, 
  Settings2,
  Check,
  X,
  ArrowLeft,
  Wand2,
  Database,
  Copy,
  AlertTriangle,
  CheckCircle,
  Code,
  Edit3,
  AlertOctagon,
  Info
} from 'lucide-react';

// Definitions
const DATASETS = [
  { id: '1', name: 'Urban_Traffic_V2', count: 12450 },
  { id: '2', name: 'Medical_MRI_Brain', count: 850 },
  { id: '3', name: 'Defect_Detection_PCB', count: 3200 },
];

type ParamType = 'range' | 'number' | 'boolean' | 'select' | 'tuple';

interface AugParamDef {
  name: string;
  label: string;
  type: ParamType;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  default: any;
}

interface OperatorDef {
  id: string;
  name: string;
  category: string;
  params: AugParamDef[];
}

const ALBUMENTATIONS_LIB: OperatorDef[] = [
  // --- Mixing / Box Level ---
  {
    id: 'MixUp', name: 'MixUp', category: 'Mixing & Box',
    params: [
      { name: 'alpha', label: 'Alpha', type: 'number', min: 0.1, max: 100, default: 32.0 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  {
    id: 'CutMix', name: 'CutMix', category: 'Mixing & Box',
    params: [
      { name: 'alpha', label: 'Alpha', type: 'number', min: 0.1, max: 100, default: 32.0 },
      { name: 'num_holes', label: 'Num Holes', type: 'number', min: 1, max: 20, default: 8 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  {
    id: 'Mosaic', name: 'Mosaic', category: 'Mixing & Box',
    params: [
      { name: 'output_height', label: 'Out Height', type: 'number', default: 640 },
      { name: 'output_width', label: 'Out Width', type: 'number', default: 640 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 1.0 }
    ]
  },
  {
    id: 'CopyPaste', name: 'CopyPaste', category: 'Mixing & Box',
    params: [
      { name: 'blend', label: 'Blend', type: 'boolean', default: true },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },

  // Geometric
  {
    id: 'ShiftScaleRotate', name: 'ShiftScaleRotate', category: 'Geometric',
    params: [
      { name: 'shift_limit', label: 'Shift Limit', type: 'range', min: 0, max: 0.5, step: 0.01, default: 0.06 },
      { name: 'scale_limit', label: 'Scale Limit', type: 'range', min: 0, max: 0.5, step: 0.01, default: 0.1 },
      { name: 'rotate_limit', label: 'Rotate Limit', type: 'range', min: 0, max: 90, step: 1, default: 45 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  {
    id: 'HorizontalFlip', name: 'HorizontalFlip', category: 'Geometric',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  },
  {
    id: 'VerticalFlip', name: 'VerticalFlip', category: 'Geometric',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  },
  {
    id: 'RandomRotate90', name: 'RandomRotate90', category: 'Geometric',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  },
  {
    id: 'Transpose', name: 'Transpose', category: 'Geometric',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  },
  {
    id: 'ElasticTransform', name: 'ElasticTransform', category: 'Geometric',
    params: [
       { name: 'alpha', label: 'Alpha', type: 'number', default: 1, min: 0, max: 10 },
       { name: 'sigma', label: 'Sigma', type: 'number', default: 50, min: 0, max: 100 },
       { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  // Color
  {
    id: 'RandomBrightnessContrast', name: 'RandomBrightnessContrast', category: 'Color',
    params: [
      { name: 'brightness_limit', label: 'Brightness Limit', type: 'range', min: 0, max: 0.5, step: 0.01, default: 0.2 },
      { name: 'contrast_limit', label: 'Contrast Limit', type: 'range', min: 0, max: 0.5, step: 0.01, default: 0.2 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  {
    id: 'HueSaturationValue', name: 'HueSaturationValue', category: 'Color',
    params: [
      { name: 'hue_shift_limit', label: 'Hue Shift', type: 'range', min: 0, max: 50, step: 1, default: 20 },
      { name: 'sat_shift_limit', label: 'Sat Shift', type: 'range', min: 0, max: 50, step: 1, default: 30 },
      { name: 'val_shift_limit', label: 'Val Shift', type: 'range', min: 0, max: 50, step: 1, default: 20 },
      { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  // Blur & Noise
  {
    id: 'GaussNoise', name: 'GaussNoise', category: 'Blur & Noise',
    params: [
       { name: 'var_limit', label: 'Var Limit', type: 'range', min: 10, max: 100, step: 5, default: 50 },
       { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  {
    id: 'MotionBlur', name: 'MotionBlur', category: 'Blur & Noise',
    params: [
       { name: 'blur_limit', label: 'Blur Limit', type: 'range', min: 3, max: 15, step: 2, default: 7 },
       { name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }
    ]
  },
  // Weather
  {
    id: 'RandomRain', name: 'RandomRain', category: 'Weather',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  },
  {
    id: 'RandomFog', name: 'RandomFog', category: 'Weather',
    params: [{ name: 'p', label: 'Probability', type: 'range', min: 0, max: 1, step: 0.1, default: 0.5 }]
  }
];

interface PipelineItem {
  instanceId: string;
  operatorId: string;
  enabled: boolean;
  params: Record<string, any>;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  pipeline: PipelineItem[];
  lastModified: string;
}

const DEFAULT_STRATEGIES: Strategy[] = [
  { 
    id: '1', 
    name: 'YOLO Default Train', 
    description: 'Standard Mosaic + MixUp + ColorJitter for YOLOv8 training.',
    lastModified: '2023-10-24',
    pipeline: [
      { instanceId: 'm1', operatorId: 'Mosaic', enabled: true, params: { output_height: 640, output_width: 640, p: 1.0 } },
      { instanceId: 'a1', operatorId: 'RandomBrightnessContrast', enabled: true, params: { brightness_limit: 0.2, contrast_limit: 0.2, p: 0.5 } },
      { instanceId: 'a2', operatorId: 'HorizontalFlip', enabled: true, params: { p: 0.5 } }
    ]
  },
  { 
    id: '2', 
    name: 'Medical MRI Cleaner', 
    description: 'Geometric only, avoiding color shifts that distort diagnostic features.',
    lastModified: '2023-10-22',
    pipeline: [
      { instanceId: 'b1', operatorId: 'VerticalFlip', enabled: true, params: { p: 0.5 } },
      { instanceId: 'b2', operatorId: 'RandomRotate90', enabled: true, params: { p: 0.5 } }
    ] 
  },
  { 
    id: '3', 
    name: 'Weather Robustness', 
    description: 'Heavy weather augmentation for autonomous driving datasets.',
    lastModified: '2023-10-20',
    pipeline: [
      { instanceId: 'c1', operatorId: 'RandomRain', enabled: true, params: { p: 0.3 } },
      { instanceId: 'c2', operatorId: 'RandomFog', enabled: true, params: { p: 0.3 } },
      { instanceId: 'c3', operatorId: 'MotionBlur', enabled: true, params: { blur_limit: 7, p: 0.5 } }
    ] 
  }
];

export default function DataAugmentation() {
  const [view, setView] = useState<'list' | 'editor'>('list');
  const [selectedDatasetId, setSelectedDatasetId] = useState(DATASETS[0].id);
  const [previewSeed, setPreviewSeed] = useState(1);
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);
  
  // Custom Modal State
  const [deleteModal, setDeleteModal] = useState<{isOpen: boolean, strategyId: string | null}>({ isOpen: false, strategyId: null });

  // Strategy State with Persistence
  const [strategies, setStrategies] = useState<Strategy[]>(() => {
    try {
      const saved = localStorage.getItem('neurocore_aug_strategies');
      if (saved) return JSON.parse(saved);
      return DEFAULT_STRATEGIES;
    } catch {
      return DEFAULT_STRATEGIES;
    }
  });

  // Persist strategies whenever they change
  useEffect(() => {
    localStorage.setItem('neurocore_aug_strategies', JSON.stringify(strategies));
  }, [strategies]);

  // Working Draft State (For editing/creating without polluting list)
  const [draftStrategy, setDraftStrategy] = useState<Strategy | null>(null);
  const [selectedPipelineItemId, setSelectedPipelineItemId] = useState<string | null>(null);
  
  const [tempName, setTempName] = useState('');
  const [isEditingName, setIsEditingName] = useState(false);

  // --- Notification Helper ---
  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Actions

  // 1. Trigger Delete Modal
  const handleDeleteClick = (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation(); 
    setDeleteModal({ isOpen: true, strategyId: id });
  };

  // 2. Confirm Delete
  const confirmDelete = () => {
    if (deleteModal.strategyId) {
        const newStrategies = strategies.filter(s => s.id !== deleteModal.strategyId);
        setStrategies(newStrategies);
        showNotification("策略已删除", "success");
    }
    setDeleteModal({ isOpen: false, strategyId: null });
  };

  // 2. Edit Strategy (Initialize Draft)
  const handleEditStrategy = (id: string) => {
    const s = strategies.find(strat => strat.id === id);
    if (s) {
        setDraftStrategy(JSON.parse(JSON.stringify(s))); // Deep copy
        setTempName(s.name);
        setView('editor');
        setIsEditingName(false);
    }
  };

  // 3. Create Strategy (Initialize Empty Draft)
  const handleCreateStrategy = () => {
    const newStrategy: Strategy = {
      id: `temp_${Date.now()}`, // Temporary ID
      name: 'Untitled Strategy',
      description: 'New augmentation pipeline',
      lastModified: new Date().toISOString().split('T')[0],
      pipeline: []
    };
    setDraftStrategy(newStrategy);
    setTempName('Untitled Strategy');
    setView('editor');
    setIsEditingName(false); // FIXED: Do not auto-enter edit mode
  };

  // 4. Save Draft
  const handleSave = (asNew: boolean = false) => {
    if (!draftStrategy) return;
    
    const nameToSave = tempName.trim();
    if (!nameToSave) {
        showNotification("策略名称不能为空", "error");
        return;
    }

    // Check duplicates
    const isKnownStrategy = strategies.some(s => s.id === draftStrategy.id);
    const isDuplicate = strategies.some(s => 
        s.name === nameToSave && 
        (asNew || (!asNew && s.id !== draftStrategy.id))
    );

    if (isDuplicate) {
        showNotification("已存在同名策略，请修改名称", "error");
        return;
    }

    const timestamp = new Date().toISOString().split('T')[0];
    
    if (asNew || !isKnownStrategy) {
        // Create New
        const newId = Date.now().toString();
        const newStrat = { ...draftStrategy, id: newId, name: nameToSave, lastModified: timestamp };
        setStrategies(prev => [newStrat, ...prev]);
        setDraftStrategy(newStrat); // Update current draft to match saved
        showNotification("策略已保存", "success");
    } else {
        // Update Existing
        setStrategies(prev => prev.map(s => s.id === draftStrategy.id ? { ...draftStrategy, name: nameToSave, lastModified: timestamp } : s));
        setDraftStrategy(prev => prev ? ({ ...prev, name: nameToSave, lastModified: timestamp }) : null);
        showNotification("策略已更新", "success");
    }
    setIsEditingName(false);
  };

  // Operator Helpers (Operating on draftStrategy)
  const addOperator = (opId: string) => {
    if (!draftStrategy) return;
    const opDef = ALBUMENTATIONS_LIB.find(op => op.id === opId);
    if (!opDef) return;

    const defaultParams: any = {};
    opDef.params.forEach(p => defaultParams[p.name] = p.default);

    const newItem: PipelineItem = {
      instanceId: Date.now().toString(),
      operatorId: opId,
      enabled: true,
      params: defaultParams
    };

    setDraftStrategy(prev => prev ? ({ ...prev, pipeline: [...prev.pipeline, newItem] }) : null);
    setSelectedPipelineItemId(newItem.instanceId);
  };

  const removeOperator = (instanceId: string) => {
    setDraftStrategy(prev => prev ? ({ ...prev, pipeline: prev.pipeline.filter(p => p.instanceId !== instanceId) }) : null);
    if (selectedPipelineItemId === instanceId) setSelectedPipelineItemId(null);
  };

  const updateParam = (instanceId: string, param: string, value: any) => {
    setDraftStrategy(prev => 
      prev ? {
        ...prev,
        pipeline: prev.pipeline.map(p => p.instanceId === instanceId ? { ...p, params: { ...p.params, [param]: value } } : p)
      } : null
    );
  };

  // Toggle Operator Enabled
  const toggleOperator = (instanceId: string) => {
     setDraftStrategy(prev => 
      prev ? {
         ...prev,
         pipeline: prev.pipeline.map(p => p.instanceId === instanceId ? { ...p, enabled: !p.enabled } : p)
      } : null
     );
  };

  // Helper to categorize operators
  const categories = Array.from(new Set(ALBUMENTATIONS_LIB.map(op => op.category)));

  // --- Views ---

  return (
    <>
      {/* GLOBAL NOTIFICATION COMPONENT (Fixed Position, High Z-Index) */}
      {notification && (
        <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[200] px-4 py-2 rounded-lg shadow-lg border flex items-center ${
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

      {/* Delete Confirmation Modal (Global for module) */}
      {deleteModal.isOpen && (
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-sm shadow-2xl animate-in fade-in zoom-in duration-200">
                <div className="flex flex-col items-center text-center mb-6">
                    <div className="w-12 h-12 rounded-full bg-rose-900/30 flex items-center justify-center text-rose-500 mb-4">
                    <AlertOctagon size={24} />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">确认删除?</h3>
                    <p className="text-sm text-slate-400">
                    您确定要删除此增强策略吗？此操作无法撤销。
                    </p>
                </div>
                <div className="flex gap-3">
                    <button 
                    onClick={() => setDeleteModal({isOpen: false, strategyId: null})}
                    className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700"
                    >
                    取消
                    </button>
                    <button 
                    onClick={confirmDelete}
                    className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-bold transition-colors shadow-lg shadow-rose-900/20"
                    >
                    确认删除
                    </button>
                </div>
            </div>
        </div>
      )}

      {/* LIST VIEW */}
      {view === 'list' && (
      <div className="h-full flex flex-col p-8 space-y-8 relative overflow-y-auto custom-scrollbar">
         <div className="flex justify-between items-center">
            <div>
               <h2 className="text-3xl font-bold text-white mb-2">数据增强策略库</h2>
               <p className="text-slate-400">管理用于训练和验证的图像增强流水线 (Pipelines)。</p>
            </div>
            <button 
              onClick={handleCreateStrategy}
              className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center transition-all"
            >
              <Plus size={20} className="mr-2" /> 新建策略
            </button>
         </div>

         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {strategies.map(strat => (
               <div 
                 key={strat.id} 
                 onClick={() => handleEditStrategy(strat.id)}
                 className="glass-panel p-6 rounded-xl border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/80 cursor-pointer group transition-all duration-300 relative overflow-hidden"
               >
                  <button 
                    className="absolute top-4 right-4 p-2 text-slate-500 hover:text-rose-400 bg-slate-900/50 hover:bg-slate-900 rounded-full transition-all z-20" 
                    onClick={(e) => handleDeleteClick(strat.id, e)}
                    title="删除策略"
                  >
                    <Trash2 size={16} />
                  </button>

                  <div className="flex justify-between items-start mb-4">
                     <div className="w-12 h-12 rounded-lg bg-slate-900 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                        <Wand2 size={24} />
                     </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-400 transition-colors pr-8">{strat.name}</h3>
                  <p className="text-sm text-slate-400 mb-6 line-clamp-2 h-10">{strat.description}</p>
                  
                  <div className="flex items-center justify-between text-xs text-slate-500 border-t border-slate-800 pt-4">
                     <span className="flex items-center"><Settings2 size={12} className="mr-1" /> {strat.pipeline.length} 算子</span>
                     <span>Updated: {strat.lastModified}</span>
                  </div>
                  
                  <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-purple-600 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
               </div>
            ))}
            
            <div 
               onClick={handleCreateStrategy}
               className="p-6 rounded-xl border-2 border-dashed border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/40 cursor-pointer transition-all flex flex-col items-center justify-center text-slate-500 hover:text-cyan-400 group min-h-[200px]"
            >
               <Plus size={48} className="mb-4 group-hover:scale-110 transition-transform" />
               <span className="font-medium">创建新策略</span>
            </div>
         </div>
      </div>
      )}

      {/* EDITOR VIEW */}
      {view === 'editor' && (
      <div className="h-full flex flex-col relative">
       
       {/* Editor Header */}
       <div className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur flex items-center justify-between px-6 shrink-0 z-20">
          <div className="flex items-center space-x-4">
             <button onClick={() => setView('list')} className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors">
                <ArrowLeft size={20} />
             </button>
             <div className="h-8 w-px bg-slate-700 mx-2"></div>
             
             {/* Renaming UI */}
             <div className="flex items-center group">
               {isEditingName ? (
                  <input 
                     autoFocus
                     onFocus={(e) => {
                        const val = e.target.value;
                        e.target.setSelectionRange(val.length, val.length);
                     }}
                     type="text" 
                     value={tempName} 
                     onChange={(e) => setTempName(e.target.value)}
                     onBlur={() => setIsEditingName(false)} // Commit changes to temp only
                     onKeyDown={(e) => e.key === 'Enter' && setIsEditingName(false)}
                     className="bg-slate-800 text-xl font-bold text-white px-2 py-0.5 rounded border border-cyan-500 outline-none w-96 shadow-[0_0_10px_rgba(6,182,212,0.3)]"
                     placeholder="Strategy Name"
                  />
               ) : (
                  <div className="flex items-center cursor-pointer p-1" onClick={() => setIsEditingName(true)}>
                     <h2 className="text-xl font-bold text-white mr-2 hover:text-cyan-200 transition-colors">{tempName}</h2>
                     <Edit3 size={16} className="text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
               )}
             </div>
          </div>
          <div className="flex space-x-2">
            <button className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded border border-slate-700">
               <Code size={14} className="mr-2" /> 预览代码
            </button>
            <div className="h-6 w-px bg-slate-800 mx-2"></div>
            <button 
               onClick={() => handleSave(true)}
               className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all"
               title="Save as new strategy"
            >
               <Copy size={14} className="mr-2" /> 另存为
            </button>
            <button 
               onClick={() => handleSave(false)}
               className="flex items-center px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded shadow-lg shadow-cyan-900/20 transition-all"
            >
               <Save size={14} className="mr-2" /> 保存
            </button>
          </div>
       </div>

       <div className="flex-1 flex overflow-hidden">
          {/* 1. Library (Left) */}
          <div className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col z-10">
             <div className="p-4 border-b border-slate-800">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">算子库 (Library)</h3>
             </div>
             <div className="flex-1 overflow-y-auto p-3 custom-scrollbar">
               {categories.map(cat => (
                 <div key={cat} className="mb-6">
                   <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-2 px-2 flex items-center">
                     <span className="w-1 h-3 bg-cyan-500 rounded-full mr-2"></span>
                     {cat}
                   </h4>
                   <div className="space-y-1">
                      {ALBUMENTATIONS_LIB.filter(op => op.category === cat).map(op => (
                        <div 
                          key={op.id}
                          onClick={() => addOperator(op.id)}
                          className="px-3 py-2.5 rounded-lg text-xs text-slate-300 hover:bg-slate-800 hover:text-white cursor-pointer flex justify-between items-center group transition-all border border-transparent hover:border-slate-700"
                        >
                           <span>{op.name}</span>
                           <Plus size={14} className="opacity-0 group-hover:opacity-100 text-cyan-400 transition-opacity" />
                        </div>
                      ))}
                   </div>
                 </div>
               ))}
            </div>
          </div>

          {/* 2. Pipeline (Middle) */}
          <div className="w-80 bg-slate-900/30 border-r border-slate-800 flex flex-col z-10">
             <div className="p-4 border-b border-slate-800 bg-slate-900/50">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">流水线 (Pipeline)</h3>
             </div>
             
             {/* Stack List */}
             <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
                {draftStrategy?.pipeline.map((item, idx) => {
                   const opDef = ALBUMENTATIONS_LIB.find(op => op.id === item.operatorId);
                   const isSelected = selectedPipelineItemId === item.instanceId;
                   return (
                     <div 
                        key={item.instanceId}
                        onClick={() => setSelectedPipelineItemId(item.instanceId)}
                        className={`p-3 rounded-lg border cursor-pointer relative transition-all group ${
                           isSelected 
                             ? 'bg-cyan-950/40 border-cyan-500/50 shadow-[0_0_10px_rgba(6,182,212,0.1)]' 
                             : 'bg-slate-900 border-slate-800 hover:border-slate-600'
                        }`}
                     >
                        <div className="flex justify-between items-center mb-1">
                           <div className="flex items-center">
                              <span className={`flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold mr-2 ${isSelected ? 'bg-cyan-500 text-black' : 'bg-slate-800 text-slate-500'}`}>
                                 {idx + 1}
                              </span>
                              <span className={`text-sm font-bold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                                 {opDef?.name}
                              </span>
                           </div>
                           <button onClick={(e) => { e.stopPropagation(); removeOperator(item.instanceId); }} className="text-slate-600 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-opacity">
                              <X size={14} />
                           </button>
                        </div>
                        <div className="text-[10px] text-slate-500 ml-7 flex items-center">
                           <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${item.enabled ? 'bg-emerald-500' : 'bg-slate-600'}`}></div>
                           {item.enabled ? 'Active' : 'Disabled'}
                        </div>
                     </div>
                   );
                })}
                {draftStrategy?.pipeline.length === 0 && (
                   <div className="flex flex-col items-center justify-center h-40 text-slate-600 border-2 border-dashed border-slate-800 rounded-xl m-2">
                      <Plus size={24} className="mb-2 opacity-50" />
                      <span className="text-xs">添加算子</span>
                   </div>
                )}
             </div>

             {/* Params Editor */}
             <div className="h-[45%] border-t border-slate-800 bg-slate-950 flex flex-col shadow-[0_-5px_15px_rgba(0,0,0,0.2)]">
                <div className="p-3 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                   <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">参数配置</h3>
                   {selectedPipelineItemId && (
                     <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                       ID: {selectedPipelineItemId.slice(-4)}
                     </span>
                   )}
                </div>
                <div className="flex-1 overflow-y-auto p-5 custom-scrollbar">
                   {selectedPipelineItemId ? (
                      (() => {
                         const item = draftStrategy?.pipeline.find(p => p.instanceId === selectedPipelineItemId);
                         if (!item) return null;
                         const opDef = ALBUMENTATIONS_LIB.find(op => op.id === item.operatorId);
                         
                         return (
                            <div className="space-y-5">
                               <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                                  <div className="flex items-center space-x-2">
                                     <input 
                                        type="checkbox" 
                                        checked={item.enabled}
                                        onChange={() => toggleOperator(item.instanceId)}
                                        className="w-4 h-4 rounded border-slate-700 bg-slate-900 accent-cyan-500 focus:ring-0 focus:ring-offset-0" 
                                     />
                                     <span className={`text-sm font-medium ${item.enabled ? 'text-white' : 'text-slate-500'}`}>启用算子</span>
                                  </div>
                               </div>
                               {opDef?.params.map(param => (
                                  <div key={param.name} className="space-y-2">
                                     <div className="flex justify-between text-xs">
                                        <span className="text-slate-400">{param.label}</span>
                                        <span className="text-cyan-400 font-mono">{item.params[param.name]}</span>
                                     </div>
                                     {param.type === 'range' ? (
                                        <input 
                                           type="range" min={param.min} max={param.max} step={param.step}
                                           value={item.params[param.name]}
                                           onChange={(e) => updateParam(item.instanceId, param.name, parseFloat(e.target.value))}
                                           className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                                        />
                                     ) : (
                                        <input 
                                           type="number"
                                           value={item.params[param.name]}
                                           onChange={(e) => updateParam(item.instanceId, param.name, parseFloat(e.target.value))}
                                           className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-xs text-white outline-none focus:border-cyan-500 transition-colors"
                                        />
                                     )}
                                  </div>
                               ))}
                            </div>
                         )
                      })()
                   ) : (
                      <div className="h-full flex flex-col items-center justify-center text-slate-600">
                         <Settings2 size={32} className="mb-2 opacity-20" />
                         <span className="text-xs">请选择左侧列表中的算子</span>
                      </div>
                   )}
                </div>
             </div>
          </div>

          {/* 3. Preview (Right) */}
          <div className="flex-1 bg-black/80 relative flex flex-col">
             {/* Preview Controls Bar */}
             <div className="h-20 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between px-6 shadow-md z-10 shrink-0">
                <div className="flex items-center space-x-6">
                   {/* Dataset Selector Control */}
                   <div className="flex flex-col group">
                      <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 group-hover:text-cyan-400 transition-colors">
                        <Database size={10} className="inline mr-1" />
                        预览数据集
                      </label>
                      <select 
                         value={selectedDatasetId} 
                         onChange={(e) => setSelectedDatasetId(e.target.value)}
                         className="bg-slate-800 border border-slate-700 rounded-lg text-sm text-white px-3 py-2 outline-none hover:border-cyan-500 hover:bg-slate-800/80 transition-all cursor-pointer w-56 shadow-sm"
                      >
                         {DATASETS.map(d => <option key={d.id} value={d.id}>{d.name} ({d.count})</option>)}
                      </select>
                   </div>
                   
                   <div className="h-10 w-px bg-slate-800"></div>

                   {/* Random Sample Control */}
                   <div className="flex flex-col">
                      <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5">随机样本</label>
                      <div className="flex items-center space-x-3">
                         <button 
                           onClick={() => setPreviewSeed(s => s + 1)} 
                           className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded-lg border border-slate-700 transition-all hover:shadow-lg hover:border-slate-600 whitespace-nowrap active:scale-95"
                        >
                           <RefreshCw size={14} className="mr-2" /> 切换图片
                        </button>
                        <span className="text-xs text-slate-600 font-mono bg-slate-900 px-2 py-1 rounded">Seed: {previewSeed}</span>
                      </div>
                   </div>
                </div>

                <div className="flex items-center space-x-3">
                   <div className="px-3 py-1 bg-slate-900 rounded border border-slate-800 text-xs text-slate-400">
                      Original: 640x640
                   </div>
                   <ArrowLeft size={12} className="text-slate-600" />
                   <div className="px-3 py-1 bg-slate-900 rounded border border-cyan-900/50 text-xs text-cyan-400">
                      Augmented: 640x640
                   </div>
                </div>
             </div>
             
             {/* Canvas Area */}
             <div className="flex-1 relative flex items-center justify-center p-10 bg-grid-pattern bg-slate-950 overflow-hidden">
                <div className="relative shadow-2xl border border-slate-700 bg-black group transition-transform duration-300 hover:scale-[1.01]">
                   <img 
                      src={`https://picsum.photos/800/800?random=${previewSeed + parseInt(selectedDatasetId)}`} 
                      className="max-w-full max-h-[calc(100vh-220px)] object-contain transition-all duration-300"
                      style={{
                         // CSS simulation of augmentation logic
                         filter: `
                            brightness(${draftStrategy?.pipeline.find(p => p.operatorId === 'RandomBrightnessContrast' && p.enabled)?.params.brightness_limit * 2 + 1 || 1})
                            contrast(${draftStrategy?.pipeline.find(p => p.operatorId === 'RandomBrightnessContrast' && p.enabled)?.params.contrast_limit * 2 + 1 || 1})
                            blur(${draftStrategy?.pipeline.find(p => p.operatorId === 'MotionBlur' && p.enabled)?.params.blur_limit / 2 || 0}px)
                         `,
                         transform: `
                            scaleX(${draftStrategy?.pipeline.find(p => p.operatorId === 'HorizontalFlip' && p.enabled)?.params.p > 0.5 ? -1 : 1})
                            scaleY(${draftStrategy?.pipeline.find(p => p.operatorId === 'VerticalFlip' && p.enabled)?.params.p > 0.5 ? -1 : 1})
                            rotate(${draftStrategy?.pipeline.find(p => p.operatorId === 'ShiftScaleRotate' && p.enabled)?.params.rotate_limit || 0}deg)
                         `
                      }}
                   />
                   {/* Overlay Labels */}
                   <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur text-cyan-400 text-[10px] rounded border border-cyan-500/30 opacity-0 group-hover:opacity-100 transition-opacity">
                      Live Preview
                   </div>
                </div>
             </div>
          </div>
       </div>
      </div>
      )}
    </>
  );
};
    