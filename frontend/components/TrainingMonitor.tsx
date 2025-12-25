import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  Square,
  RefreshCw,
  Save,
  ChevronDown,
  Clock,
  Plus,
  Search,
  ArrowLeft,
  Target,
  Tag,
  Shapes,
  Zap,
  Activity,
  CheckCircle,
  XCircle,
  AlertOctagon,
  Terminal,
  MoreHorizontal,
  HelpCircle,
  Sliders,
  Layers,
  Database,
  Wand2,
  FileJson,
  Eye,
  X,
  AlertTriangle,
  Edit3,
  Trash2,
  HardDrive
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { useTrainingLogsWS, LogEntry, MetricsEntry } from '../hooks/useWebSocket';

// --- Types & Mock Data ---

type TaskType = 'detection' | 'classification' | 'segmentation';
type ExpStatus = 'running' | 'completed' | 'failed' | 'queued' | 'paused';

interface Experiment {
  id: string;
  name: string;
  task: TaskType;
  model: string;
  dataset: string;
  augmentation: string;
  status: ExpStatus;
  duration: string;
  accuracy: string; // mAP or Top-1
  startedAt: string;
  config?: any;
}

const INITIAL_EXPERIMENTS: Experiment[] = [
  { id: '1042', name: 'YOLOv8-Nano-Base', task: 'detection', model: 'YOLOv8-N', dataset: 'Urban_Traffic', augmentation: 'YOLO Default', status: 'running', duration: '45m', accuracy: '0.65 mAP', startedAt: '10:30 AM' },
  { id: '1041', name: 'ResNet50-FineTune', task: 'classification', model: 'ResNet50', dataset: 'Medical_MRI', augmentation: 'Medical Robust', status: 'completed', duration: '4h 12m', accuracy: '94.2%', startedAt: 'Yesterday' },
  { id: '1040', name: 'UNet-Cell-Seg', task: 'segmentation', model: 'UNet', dataset: 'Cell_Microscopy', augmentation: 'No Augmentation', status: 'failed', duration: '12m', accuracy: '-', startedAt: 'Yesterday' },
  { id: '1039', name: 'YOLO-L-Heavy', task: 'detection', model: 'YOLOv8-L', dataset: 'Defect_PCB', augmentation: 'YOLO Default', status: 'completed', duration: '12h 05m', accuracy: '0.91 mAP', startedAt: '2 days ago' },
];

const TRAINING_CHART_DATA = Array.from({ length: 50 }, (_, i) => ({
  epoch: i,
  trainLoss: Math.max(0.1, 2.5 * Math.exp(-0.05 * i) + Math.random() * 0.1),
  valLoss: Math.max(0.2, 2.8 * Math.exp(-0.045 * i) + Math.random() * 0.2),
  metric: Math.min(0.95, 0.1 + 0.8 * (1 - Math.exp(-0.06 * i)) + Math.random() * 0.05)
}));

// --- Configuration Schema ---

const TRAINING_SCHEMA = {
  common: {
    training: {
      title: "核心训练参数 / Hyperparameters",
      fields: {
        batch_size: { value: 16, ui: { type: "number", label: "Batch Size", min: 1, step: 1, hint: "每次迭代样本数" } },
        epochs: { value: 100, ui: { type: "number", label: "Epochs", min: 1 } },
        learning_rate: { value: 0.0003, ui: { type: "number", label: "Learning Rate", format: "scientific", hint: "初始学习率" } },
        optimizer: { value: "AdamW", ui: { type: "select", label: "Optimizer", options: ["Adam", "AdamW", "SGD", "RMSprop"] } },
        lr_scheduler: { value: "Cosine", ui: { type: "select", label: "LR Scheduler", options: ["None", "Step", "Cosine", "ReduceOnPlateau"] } },
        weight_decay: { value: 0.0001, ui: { type: "number", label: "Weight Decay", format: "scientific" } },
        
        input_size: { value: 640, ui: { type: "number", label: "Input Size (px)", hint: "检测/分割通常较大" } },
        num_classes: { value: 80, ui: { type: "number", label: "Num Classes", min: 1 } },
        pretrained: { 
          value: "IMAGENET_1K_V1", 
          ui: { 
            type: "select", 
            label: "Pretrained Weights", 
            options: ["None", "IMAGENET_1K_V1", "yolov8n-traffic-best.pt", "resnet50-mri-v2.pt"] 
          } 
        },
        
        val_interval: { value: 1, ui: { type: "number", label: "Val Interval" } },
        monitor_metric: { value: "val_loss", ui: { type: "select", label: "Monitor Metric", options: ["val_loss", "accuracy", "mAP", "IoU"] } },
        
        early_stopping: { 
          value: 10, 
          ui: { type: "number", label: "Early Stop Patience", min: 0, hint: "0 to disable" } 
        },
        
        num_workers: { value: 8, ui: { type: "number", label: "Num Workers" } },
        seed: { value: 42, ui: { type: "number", label: "Random Seed" } },
        
        amp: { value: true, ui: { type: "switch", label: "Mixed Precision (AMP)" } },
      }
    }
  },
  task_advanced: {
    classification: {
      title: "分类特定参数 (Classification)",
      fields: {
        label_smoothing: { value: 0.1, ui: { type: "slider", label: "Label Smoothing", min: 0.0, max: 0.2, step: 0.01 } },
        dropout: { value: 0.2, ui: { type: "slider", label: "Dropout Rate", min: 0.0, max: 0.7, step: 0.05 } }
      }
    },
    detection: {
      title: "检测特定参数 (Detection)",
      fields: {
        conf_threshold: { value: 0.25, ui: { type: "slider", label: "Conf Threshold", min: 0.0, max: 1.0, step: 0.05 } },
        nms_iou_threshold: { value: 0.5, ui: { type: "slider", label: "NMS IoU Threshold", min: 0.3, max: 0.9, step: 0.05 } },
        max_detections: { value: 100, ui: { type: "number", label: "Max Detections" } }
      }
    },
    segmentation: {
      title: "分割特定参数 (Segmentation)",
      fields: {
        seg_loss: { value: "CrossEntropy", ui: { type: "select", label: "Loss Function", options: ["CrossEntropy", "Dice", "CrossEntropy+Dice"] } },
        dice_weight: { 
           value: 1.0, 
           ui: { type: "number", label: "Dice Weight", visible_when: { field: "seg_loss", value: ["Dice", "CrossEntropy+Dice"] } } 
        },
        ignore_index: { value: 255, ui: { type: "number", label: "Ignore Index" } }
      }
    }
  }
};


// --- Helper Components ---

const StatusBadge: React.FC<{ status: ExpStatus }> = ({ status }) => {
  const styles = {
    running: 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30 animate-pulse',
    completed: 'bg-cyan-950/40 text-cyan-400 border-cyan-500/30',
    failed: 'bg-rose-950/40 text-rose-400 border-rose-500/30',
    queued: 'bg-slate-800 text-slate-400 border-slate-700',
    paused: 'bg-amber-950/40 text-amber-400 border-amber-500/30',
  };
  
  const icons = {
    running: <RefreshCw size={12} className="mr-1.5 animate-spin" />,
    completed: <CheckCircle size={12} className="mr-1.5" />,
    failed: <XCircle size={12} className="mr-1.5" />,
    queued: <Clock size={12} className="mr-1.5" />,
    paused: <Pause size={12} className="mr-1.5" />,
  };

  return (
    <span className={`flex items-center px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider border ${styles[status]}`}>
      {icons[status]}
      {status}
    </span>
  );
};

const TaskIcon: React.FC<{ task: TaskType }> = ({ task }) => {
    switch(task) {
        case 'detection': return <span title="Object Detection"><Target size={16} className="text-rose-400" /></span>;
        case 'classification': return <span title="Image Classification"><Tag size={16} className="text-cyan-400" /></span>;
        case 'segmentation': return <span title="Semantic Segmentation"><Shapes size={16} className="text-purple-400" /></span>;
    }
};

const LogModal: React.FC<{ isOpen: boolean; onClose: () => void; logs: string[] }> = ({ isOpen, onClose, logs }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-3xl h-[600px] flex flex-col shadow-2xl animate-in fade-in zoom-in duration-200">
                <div className="flex justify-between items-center p-4 border-b border-slate-700 bg-slate-950/50 rounded-t-xl">
                    <h3 className="text-white font-bold flex items-center"><Terminal size={18} className="mr-2 text-slate-400"/> 实时训练日志</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-xs bg-black text-slate-300 space-y-1">
                    {logs.map((log, i) => (
                        <div key={i} className="hover:bg-slate-900/50 px-2 py-0.5 rounded">{log}</div>
                    ))}
                    <div className="animate-pulse text-cyan-500">_</div>
                </div>
            </div>
        </div>
    );
};

const ConfigPreviewModal: React.FC<{ isOpen: boolean; onClose: () => void; config: any }> = ({ isOpen, onClose, config }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-lg shadow-2xl animate-in fade-in zoom-in duration-200">
                <div className="flex justify-between items-center p-4 border-b border-slate-700 bg-slate-950/50 rounded-t-xl">
                    <h3 className="text-white font-bold flex items-center"><FileJson size={18} className="mr-2 text-cyan-400"/> 参数预览 (JSON)</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
                </div>
                <div className="p-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                    <pre className="text-xs font-mono text-emerald-400 bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto">
                        {JSON.stringify(config, null, 2)}
                    </pre>
                </div>
                <div className="p-4 border-t border-slate-700 flex justify-end">
                     <button onClick={onClose} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold rounded">关闭</button>
                </div>
            </div>
        </div>
    );
};

const ConfirmModal: React.FC<{ isOpen: boolean; onClose: () => void; onConfirm: () => void; title: string; msg: string; type?: 'danger'|'info' }> = ({ isOpen, onClose, onConfirm, title, msg, type = 'danger' }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-sm shadow-2xl animate-in fade-in zoom-in duration-200" onClick={(e) => e.stopPropagation()}>
                <div className="flex flex-col items-center text-center mb-6">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 ${type === 'danger' ? 'bg-rose-900/30 text-rose-500' : 'bg-cyan-900/30 text-cyan-500'}`}>
                        {type === 'danger' ? <AlertTriangle size={24} /> : <HelpCircle size={24} />}
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
                    <p className="text-sm text-slate-400">{msg}</p>
                </div>
                <div className="flex gap-3">
                    <button 
                        onClick={onClose}
                        className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700"
                    >
                        取消
                    </button>
                    <button 
                        onClick={onConfirm}
                        className={`flex-1 py-2.5 text-white rounded-lg font-bold transition-colors shadow-lg ${type === 'danger' ? 'bg-rose-600 hover:bg-rose-500 shadow-rose-900/20' : 'bg-cyan-600 hover:bg-cyan-500 shadow-cyan-900/20'}`}
                    >
                        确认
                    </button>
                </div>
            </div>
        </div>
    );
}

const RenameModal: React.FC<{ isOpen: boolean; onClose: () => void; onConfirm: (newName: string) => void; initialName: string }> = ({ isOpen, onClose, onConfirm, initialName }) => {
    const [name, setName] = useState(initialName);
    
    useEffect(() => {
        setName(initialName);
    }, [initialName, isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
             <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-sm shadow-2xl animate-in fade-in zoom-in duration-200">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-bold text-white">重命名实验</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={16}/></button>
                </div>
                <input 
                    type="text" 
                    value={name} 
                    onChange={(e) => setName(e.target.value)} 
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 mb-6"
                    autoFocus
                    onKeyDown={(e) => e.key === 'Enter' && onConfirm(name)}
                />
                <div className="flex gap-3">
                     <button onClick={onClose} className="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded-lg">取消</button>
                     <button onClick={() => onConfirm(name)} className="flex-1 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-bold rounded-lg shadow-lg shadow-cyan-900/20">保存</button>
                </div>
             </div>
        </div>
    );
}

// --- Main Component ---

const TrainingMonitor: React.FC = () => {
  const [view, setView] = useState<'list' | 'create' | 'detail'>('list');
  const [experiments, setExperiments] = useState<Experiment[]>(INITIAL_EXPERIMENTS);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  
  // --- Create Experiment Form State ---
  const [formTask, setFormTask] = useState<TaskType>('detection');
  const [formName, setFormName] = useState('');
  const [formDataset, setFormDataset] = useState('');
  const [formAugmentation, setFormAugmentation] = useState(''); 
  const [formModel, setFormModel] = useState('');
  const [allowOverwrite, setAllowOverwrite] = useState(false); // New: Overwrite State
  const [showOverwriteWarning, setShowOverwriteWarning] = useState(false); // Timer state for tooltip
  
  // Dynamic Config State
  const [config, setConfig] = useState<Record<string, any>>({});
  
  // Modals & Action State
  const [showLogModal, setShowLogModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  
  // Action Dropdown State
  const [activeDropdownId, setActiveDropdownId] = useState<string | null>(null);

  // Stop Confirmation State
  const [stopModal, setStopModal] = useState<{isOpen: boolean, id: string | null}>({ isOpen: false, id: null });
  
  // Rename Modal State
  const [renameModal, setRenameModal] = useState<{isOpen: boolean, id: string | null, currentName: string}>({ isOpen: false, id: null, currentName: '' });

  const [dummyLogs, setDummyLogs] = useState<string[]>([]);
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);

  // WebSocket实时数据状态
  const [realTimeLogs, setRealTimeLogs] = useState<LogEntry[]>([]);
  const [realTimeMetrics, setRealTimeMetrics] = useState<MetricsEntry[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  const dropdownRef = useRef<HTMLDivElement>(null);

  // Initialize Config Defaults
  useEffect(() => {
    const defaults: Record<string, any> = {};
    Object.entries(TRAINING_SCHEMA.common.training.fields).forEach(([key, def]) => { defaults[key] = def.value; });
    Object.values(TRAINING_SCHEMA.task_advanced).forEach(section => {
        Object.entries(section.fields).forEach(([key, def]) => { defaults[key] = def.value; });
    });
    setConfig(defaults);
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
        if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
            setActiveDropdownId(null);
        }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Auto-hide overwrite warning
  useEffect(() => {
      if (allowOverwrite) {
          setShowOverwriteWarning(true);
          const timer = setTimeout(() => {
              setShowOverwriteWarning(false);
          }, 3000); // Hide after 3 seconds
          return () => clearTimeout(timer);
      } else {
          setShowOverwriteWarning(false);
      }
  }, [allowOverwrite]);

  const handleConfigChange = (key: string, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
      setNotification({ msg, type });
      setTimeout(() => setNotification(null), 3000);
  };

  // WebSocket连接：当查看特定实验详情时连接
  const shouldConnectWS = view === 'detail' && selectedExpId !== null;
  const { connected, disconnect } = useTrainingLogsWS(
    shouldConnectWS ? selectedExpId! : '',
    {
      onLog: (data: LogEntry) => {
        // 接收实时日志
        setRealTimeLogs(prev => {
          const newLogs = [...prev, data];
          // 只保留最新100条日志
          return newLogs.slice(-100);
        });

        // 同时添加到dummyLogs以在模态框中显示
        const logMessage = `[${data.level}] ${data.message}`;
        setDummyLogs(prev => [...prev.slice(-100), logMessage]);
      },
      onMetrics: (data: MetricsEntry) => {
        // 接收实时指标
        setRealTimeMetrics(prev => {
          const newMetrics = [...prev, data];
          // 只保留最新100条指标
          return newMetrics.slice(-100);
        });

        // 更新实验列表中的对应实验状态
        setExperiments(prev => prev.map(exp => {
          if (exp.id === selectedExpId) {
            // 更新准确率等显示数据
            return {
              ...exp,
              accuracy: data.val_acc ? `${(data.val_acc * 100).toFixed(1)}%` : exp.accuracy
            };
          }
          return exp;
        }));
      },
      onStatusChange: (data) => {
        // 接收状态变化
        setExperiments(prev => prev.map(exp => {
          if (exp.id === selectedExpId) {
            return {
              ...exp,
              status: data.status as ExpStatus
            };
          }
          return exp;
        }));

        showNotification(`训练状态已更新: ${data.status}`, 'info');
      }
    }
  );

  // 更新WebSocket连接状态
  useEffect(() => {
    setWsConnected(connected);
  }, [connected]);

  // 切换视图或实验时清空实时数据
  useEffect(() => {
    if (view !== 'detail' || !selectedExpId) {
      setRealTimeLogs([]);
      setRealTimeMetrics([]);
    }
  }, [view, selectedExpId]);

  const handleStartTraining = () => {
      if (!formName.trim()) {
          showNotification("请输入实验名称", "error");
          return;
      }

      // Check for duplicate name
      const exists = experiments.some(e => e.name === formName);

      let updatedExperiments = [...experiments];

      if (exists) {
          if (!allowOverwrite) {
              showNotification("实验名称已存在。请修改名称或勾选“同名覆盖”。", "error");
              return;
          } else {
             // Overwrite Logic: Remove existing
             updatedExperiments = updatedExperiments.filter(e => e.name !== formName);
             showNotification("旧实验记录已覆盖", "success");
          }
      }

      const newId = Date.now().toString();
      const newExperiment: Experiment = {
          id: newId,
          name: formName,
          task: formTask,
          model: formModel || 'YOLOv8',
          dataset: formDataset || 'Urban_Traffic_V2',
          augmentation: formAugmentation || 'Default',
          status: 'running',
          duration: '0s',
          accuracy: '0.00',
          startedAt: 'Just now',
          config: config
      };

      setExperiments([newExperiment, ...updatedExperiments]);
      setSelectedExpId(newId);
      
      // Critical: Generate logs BEFORE view switch so they are ready
      generateDummyLogs(newExperiment.name);
      
      // Critical: Ensure state updates before switching view
      setTimeout(() => {
          setView('detail');
      }, 50);
  };

  // --- Actions Implementations ---

  const generateDummyLogs = (name: string) => {
      const logs = [];
      const timestamp = () => new Date().toISOString().split('T')[1].split('.')[0];
      logs.push(`[${timestamp()}] [INFO] Initializing experiment: ${name}`);
      logs.push(`[${timestamp()}] [INFO] Loading dataset... Done.`);
      logs.push(`[${timestamp()}] [INFO] Model architecture loaded to GPU:0`);
      logs.push(`[${timestamp()}] [INFO] Starting training loop...`);
      for(let i=1; i<=5; i++) {
          logs.push(`[${timestamp()}] [INFO] Epoch ${i}/100 - loss: ${(Math.random() * 2).toFixed(4)} - acc: ${(0.5 + Math.random() * 0.4).toFixed(4)}`);
      }
      setDummyLogs(logs);
  };

  const toggleExperimentStatus = (id: string) => {
      setExperiments(prev => prev.map(e => {
          if (e.id === id) {
              const newStatus = e.status === 'running' ? 'paused' : 'running';
              showNotification(`实验已${newStatus === 'paused' ? '暂停' : '继续'}`, 'success');
              return { ...e, status: newStatus };
          }
          return e;
      }));
  };

  // Trigger Modal
  const handleStopClick = (id: string) => {
      setStopModal({ isOpen: true, id });
  };

  // Confirm Stop
  const confirmStop = () => {
      if (stopModal.id) {
          setExperiments(prev => prev.map(e => e.id === stopModal.id ? { ...e, status: 'failed' } : e));
          showNotification("实验已强制停止", "error");
      }
      setStopModal({ isOpen: false, id: null });
  };

  // Trigger Rename
  const handleRenameClick = (id: string, currentName: string) => {
      setRenameModal({ isOpen: true, id, currentName });
      setActiveDropdownId(null);
  }

  // Confirm Rename
  const confirmRename = (newName: string) => {
      if (renameModal.id && newName.trim()) {
          setExperiments(prev => prev.map(e => e.id === renameModal.id ? { ...e, name: newName } : e));
          showNotification("实验已重命名", "success");
      }
      setRenameModal({ isOpen: false, id: null, currentName: '' });
  }

  // NEW: Handle Save to Registry
  const handleSaveToRegistry = (exp: Experiment) => {
      if (exp.status !== 'completed') {
          return;
      }
      // Simulation: In a real app this would call an API
      showNotification(`权重文件 ${exp.name}.pt 已保存到权重库`, "success");
  };

  // --- Dynamic Field Renderer ---
  const renderField = (key: string, fieldDef: any) => {
      // 1. Check Visibility
      if (fieldDef.ui.visible_when) {
          const depKey = fieldDef.ui.visible_when.field;
          const depVal = fieldDef.ui.visible_when.value;
          const currentDepVal = config[depKey];
          
          if (Array.isArray(depVal)) {
              if (!depVal.includes(currentDepVal)) return null;
          } else {
              if (currentDepVal !== depVal) return null;
          }
      }

      // 2. Render based on Type
      switch (fieldDef.ui.type) {
          case 'number':
              return (
                  <div key={key} className="space-y-1 group">
                      <div className="flex justify-between items-center">
                        <label className="text-[10px] font-bold text-slate-500 uppercase group-hover:text-slate-400 transition-colors">{fieldDef.ui.label}</label>
                        {fieldDef.ui.hint && <span title={fieldDef.ui.hint}><HelpCircle size={10} className="text-slate-700 hover:text-slate-500 cursor-help" /></span>}
                      </div>
                      <input 
                        type="number" 
                        value={config[key]} 
                        step={fieldDef.ui.step || "any"}
                        min={fieldDef.ui.min}
                        onChange={(e) => handleConfigChange(key, parseFloat(e.target.value))}
                        className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white text-sm font-mono focus:border-cyan-500 outline-none transition-colors"
                      />
                  </div>
              );
          case 'select':
              return (
                  <div key={key} className="space-y-1 group">
                      <label className="text-[10px] font-bold text-slate-500 uppercase group-hover:text-slate-400 transition-colors">{fieldDef.ui.label}</label>
                      <select 
                        value={config[key]} 
                        onChange={(e) => handleConfigChange(key, e.target.value)}
                        style={{ colorScheme: 'dark' }}
                        className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white text-sm outline-none focus:border-cyan-500 transition-colors cursor-pointer"
                      >
                          {fieldDef.ui.options.map((opt: string) => (
                              <option key={opt} value={opt}>{opt}</option>
                          ))}
                      </select>
                  </div>
              );
          case 'switch':
              return (
                  <div key={key} className="flex items-center justify-between p-2 rounded bg-slate-900/40 border border-slate-800 hover:border-slate-700 transition-colors mt-auto h-[50px]">
                      <label className="text-[10px] font-bold text-slate-400 uppercase">{fieldDef.ui.label}</label>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input 
                            type="checkbox" 
                            checked={!!config[key]} 
                            onChange={(e) => handleConfigChange(key, e.target.checked)}
                            className="sr-only peer" 
                        />
                        <div className="w-8 h-4 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-cyan-600"></div>
                      </label>
                  </div>
              );
          case 'slider':
              return (
                  <div key={key} className="space-y-3 pt-1">
                       <div className="flex justify-between items-center">
                          <label className="text-[10px] font-bold text-slate-500 uppercase">{fieldDef.ui.label}</label>
                          <span className="text-cyan-400 text-xs font-mono bg-cyan-950/30 px-1.5 rounded">{config[key]}</span>
                       </div>
                       <input 
                          type="range" 
                          min={fieldDef.ui.min} 
                          max={fieldDef.ui.max} 
                          step={fieldDef.ui.step}
                          value={config[key]}
                          onChange={(e) => handleConfigChange(key, parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                       />
                  </div>
              );
          default:
              return null;
      }
  };


  // --- View 1: Experiment List ---
  const renderList = () => (
    <div className="h-full flex flex-col p-8 space-y-6 relative overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center shrink-0">
            <div>
               <h2 className="text-3xl font-bold text-white mb-2">实验管理 (Experiments)</h2>
               <p className="text-slate-400">监控训练任务状态，管理历史实验记录。</p>
            </div>
            <button 
              onClick={() => setView('create')}
              className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center transition-all"
            >
              <Plus size={20} className="mr-2" /> 新建实验
            </button>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center space-x-4 bg-slate-900/50 p-2 rounded-lg border border-slate-800">
            <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                <input type="text" placeholder="搜索实验名称 / 模型 ID..." className="w-full bg-slate-950 border border-slate-700 rounded-md py-2 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-cyan-500" />
            </div>
            <div className="flex space-x-2">
                <select style={{ colorScheme: 'dark' }} className="bg-slate-950 border border-slate-700 rounded-md py-2 px-3 text-sm text-slate-300 outline-none">
                    <option>所有任务类型</option>
                    <option>Detection</option>
                    <option>Classification</option>
                </select>
                <select style={{ colorScheme: 'dark' }} className="bg-slate-950 border border-slate-700 rounded-md py-2 px-3 text-sm text-slate-300 outline-none">
                    <option>所有状态</option>
                    <option>Running</option>
                    <option>Completed</option>
                </select>
            </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-900/20 rounded-xl border border-slate-800">
            <table className="w-full text-left border-collapse">
                <thead className="bg-slate-900/80 backdrop-blur sticky top-0 z-10 text-xs font-bold text-slate-500 uppercase tracking-wider">
                    <tr>
                        <th className="p-4 border-b border-slate-800">Status</th>
                        <th className="p-4 border-b border-slate-800">Experiment Name</th>
                        <th className="p-4 border-b border-slate-800">Type</th>
                        <th className="p-4 border-b border-slate-800">Dataset</th>
                        <th className="p-4 border-b border-slate-800">Duration</th>
                        <th className="p-4 border-b border-slate-800 text-right">Metric</th>
                        <th className="p-4 border-b border-slate-800 text-right">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-sm">
                    {experiments.map(exp => (
                        <tr 
                          key={exp.id} 
                          onClick={() => { setSelectedExpId(exp.id); setView('detail'); generateDummyLogs(exp.name); }}
                          className="hover:bg-slate-800/50 cursor-pointer transition-colors group"
                        >
                            <td className="p-4">
                                <StatusBadge status={exp.status} />
                            </td>
                            <td className="p-4">
                                <div className="font-bold text-white group-hover:text-cyan-400 transition-colors">{exp.name}</div>
                                <div className="text-xs text-slate-500 font-mono">ID: #{exp.id}</div>
                            </td>
                            <td className="p-4">
                                <div className="flex items-center space-x-2 text-slate-300 capitalize">
                                   <TaskIcon task={exp.task} />
                                   <span>{exp.task}</span>
                                </div>
                            </td>
                            <td className="p-4 text-slate-300">
                                <div>{exp.dataset}</div>
                                <div className="text-[10px] text-slate-500 flex items-center mt-0.5"><Wand2 size={8} className="mr-1"/>{exp.augmentation}</div>
                            </td>
                            <td className="p-4 text-slate-400 font-mono">{exp.duration}</td>
                            <td className="p-4 text-right font-mono font-bold text-white">{exp.accuracy}</td>
                            <td className="p-4 text-right relative">
                                <button 
                                    onClick={(e) => { 
                                        e.stopPropagation(); 
                                        setActiveDropdownId(activeDropdownId === exp.id ? null : exp.id);
                                    }}
                                    className={`p-2 rounded-lg transition-colors ${activeDropdownId === exp.id ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-white hover:bg-slate-700'}`}
                                >
                                    <MoreHorizontal size={16} />
                                </button>
                                {activeDropdownId === exp.id && (
                                    <div 
                                        ref={dropdownRef}
                                        className="absolute right-8 top-10 w-32 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 overflow-hidden animate-in fade-in zoom-in duration-100"
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        <button 
                                            onClick={() => handleRenameClick(exp.id, exp.name)}
                                            className="w-full text-left px-4 py-2 text-xs text-slate-300 hover:text-white hover:bg-slate-800 flex items-center"
                                        >
                                            <Edit3 size={12} className="mr-2" /> 重命名
                                        </button>
                                        {/* Placeholder for future actions */}
                                        <button 
                                            onClick={() => { showNotification("功能开发中...", "info"); setActiveDropdownId(null); }}
                                            className="w-full text-left px-4 py-2 text-xs text-slate-300 hover:text-white hover:bg-slate-800 flex items-center"
                                        >
                                            <Trash2 size={12} className="mr-2" /> 删除
                                        </button>
                                    </div>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
  );

  // --- View 2: Create Experiment (Dynamic Schema) ---
  const renderCreate = () => (
    <div className="h-full flex flex-col relative bg-slate-950">
        {/* Header */}
        <div className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur flex items-center justify-between px-6 shrink-0 z-20">
             <div className="flex items-center">
                <button onClick={() => setView('list')} className="mr-4 p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors">
                    <ArrowLeft size={20} />
                </button>
                <h2 className="text-xl font-bold text-white">新建训练任务 (New Experiment)</h2>
             </div>
             <div className="flex space-x-3 items-center">
                <button className="px-4 py-2 text-slate-300 hover:text-white font-medium text-sm" onClick={() => setView('list')}>取消</button>
                
                {/* Overwrite Checkbox (Styled) */}
                <div className="mr-4 flex items-center">
                    <label className={`
                        relative flex items-center px-3 py-1.5 rounded-lg border transition-all cursor-pointer select-none
                        ${allowOverwrite 
                            ? 'bg-amber-900/20 border-amber-500/50 text-amber-200' 
                            : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-slate-500'}
                    `}>
                        <input 
                            type="checkbox" 
                            checked={allowOverwrite} 
                            onChange={(e) => setAllowOverwrite(e.target.checked)}
                            className="sr-only" 
                        />
                        <div className={`w-4 h-4 rounded border flex items-center justify-center mr-2 transition-colors ${allowOverwrite ? 'bg-amber-500 border-amber-500' : 'bg-slate-800 border-slate-600'}`}>
                            {allowOverwrite && <CheckCircle size={10} className="text-black" />}
                        </div>
                        <span className="text-xs font-medium">同名覆盖</span>
                        
                        {/* Warning Tooltip - Now auto-hiding */}
                        {allowOverwrite && showOverwriteWarning && (
                            <div className="absolute top-full right-0 mt-2 w-48 p-2 bg-amber-950 border border-amber-800 rounded shadow-xl z-50 animate-in fade-in slide-in-from-top-1 pointer-events-none">
                                <div className="flex items-start text-[10px] text-amber-400 leading-tight">
                                    <AlertTriangle size={12} className="mr-1.5 shrink-0 mt-0.5" />
                                    注意：启动时将强制删除并覆盖任何现有的同名实验记录。
                                </div>
                            </div>
                        )}
                    </label>
                </div>

                <button 
                  onClick={handleStartTraining}
                  className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-lg shadow-lg shadow-cyan-900/20 flex items-center transition-all active:scale-95"
                >
                    <Zap size={16} className="mr-2 fill-current" /> 开始训练
                </button>
             </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 lg:p-8">
            <div className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
                
                {/* Left Column (Span 4): Experiment Setup + Task Specific */}
                <div className="lg:col-span-4 space-y-6">
                    <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col h-full">
                        <h3 className="text-lg font-bold text-white mb-6 flex items-center">
                            <span className="w-8 h-8 rounded bg-cyan-900/30 text-cyan-400 flex items-center justify-center mr-3 border border-cyan-500/30">1</span>
                            实验配置 (Configuration)
                        </h3>
                        
                        <div className="space-y-6 flex-1">
                            {/* Task Selector */}
                            <div>
                                <label className="block text-[10px] font-bold text-slate-500 uppercase mb-2">任务类型 (Task)</label>
                                <div className="grid grid-cols-3 gap-2">
                                    {(['detection', 'classification', 'segmentation'] as TaskType[]).map(t => (
                                        <div 
                                        key={t}
                                        onClick={() => setFormTask(t)}
                                        className={`cursor-pointer py-3 rounded-lg border flex flex-col items-center justify-center transition-all ${formTask === t ? 'bg-cyan-950/40 border-cyan-500 text-white shadow-[0_0_10px_rgba(6,182,212,0.2)]' : 'bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-600 hover:bg-slate-800'}`}
                                        >
                                            <div className={`mb-1 ${formTask === t ? 'text-cyan-400' : 'text-slate-500'}`}>
                                                <TaskIcon task={t} />
                                            </div>
                                            <span className="text-[9px] font-bold uppercase tracking-wide">{t}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            <div className="space-y-4">
                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">实验名称</label>
                                    <div className="relative">
                                        <input type="text" value={formName} onChange={e => setFormName(e.target.value)} placeholder="e.g. YOLOv8-FineTune-v1" className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-white text-sm focus:border-cyan-500 outline-none transition-colors" />
                                    </div>
                                </div>

                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">模型架构</label>
                                    <div className="relative">
                                        <Layers size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select style={{ colorScheme: 'dark' }} value={formModel} onChange={e => setFormModel(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer">
                                            <option value="">Select Architecture...</option>
                                            {formTask === 'detection' && <><option>YOLOv8</option><option>YOLOv5</option><option>Faster-RCNN</option></>}
                                            {formTask === 'classification' && <><option>ResNet50</option><option>ViT-Base</option><option>EfficientNet</option></>}
                                            {formTask === 'segmentation' && <><option>UNet</option><option>DeepLabV3</option><option>Mask-RCNN</option></>}
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>

                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">数据集</label>
                                    <div className="relative">
                                        <Database size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select style={{ colorScheme: 'dark' }} value={formDataset} onChange={e => setFormDataset(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer">
                                            <option value="">Select Dataset...</option>
                                            <option>Urban_Traffic_V2</option>
                                            <option>Medical_MRI</option>
                                            <option>COCO_2017</option>
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>
                                
                                {/* New Augmentation Selector */}
                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">增强策略 (Augmentation)</label>
                                    <div className="relative">
                                        <Wand2 size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select style={{ colorScheme: 'dark' }} value={formAugmentation} onChange={e => setFormAugmentation(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer">
                                            <option value="">Select Strategy...</option>
                                            <option>YOLO Default Train</option>
                                            <option>Medical MRI Cleaner</option>
                                            <option>Weather Robustness</option>
                                            <option>No Augmentation</option>
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>
                            </div>
                            
                            {/* DYNAMIC SECTION: Task Specific Params inserted HERE */}
                            <div className="pt-6 mt-2 border-t border-slate-800">
                                <h4 className="text-xs font-bold text-slate-300 uppercase mb-4 flex items-center">
                                    <Sliders size={12} className="mr-2 text-cyan-400" />
                                    {TRAINING_SCHEMA.task_advanced[formTask].title}
                                </h4>
                                <div className="space-y-4">
                                    {Object.entries(TRAINING_SCHEMA.task_advanced[formTask].fields).map(([key, def]) => renderField(key, def))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column (Span 8): Core Parameters */}
                <div className="lg:col-span-8">
                    <div className="glass-panel p-5 rounded-xl border border-slate-800 h-full">
                         <h3 className="text-lg font-bold text-white mb-6 flex items-center">
                            <span className="w-8 h-8 rounded bg-purple-900/30 text-purple-400 flex items-center justify-center mr-3 border border-purple-500/30">2</span>
                            {TRAINING_SCHEMA.common.training.title}
                        </h3>
                        
                        {/* High Density Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-5 gap-y-6">
                            {Object.entries(TRAINING_SCHEMA.common.training.fields).map(([key, def]) => (
                                <div key={key} className={def.ui.type === 'switch' ? 'flex items-end' : ''}>
                                    {renderField(key, def)}
                                </div>
                            ))}
                        </div>
                        
                        <div className="mt-8 p-4 bg-slate-900/50 rounded-lg border border-slate-800 flex items-start space-x-3">
                            <div className="mt-0.5"><AlertOctagon size={16} className="text-amber-500" /></div>
                            <div>
                                <h5 className="text-sm font-bold text-slate-300">参数配置提示</h5>
                                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                                    增加 batch size 可以提高 GPU 利用率，但可能需要根据 linear scaling rule 调整学习率。
                                    对于微调任务 (Fine-tuning)，建议使用较小的学习率 (e.g. 1e-4) 并冻结骨干网络前几层。
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
            <div className="h-20"></div>
        </div>
    </div>
  );

  // --- View 3: Experiment Detail / Monitor ---
  const renderDetail = () => {
    const exp = experiments.find(e => e.id === selectedExpId) || experiments[0];
    const isRunning = exp.status === 'running';
    const isCompleted = exp.status === 'completed';

    return (
        <div className="h-full flex flex-col p-6 space-y-6 overflow-y-auto">
            {/* Detail Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                 <div className="flex items-center">
                    <button onClick={() => setView('list')} className="mr-4 p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors">
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <h2 className="text-2xl font-bold text-white flex items-center">
                          {exp.name}
                          <span className="ml-3"><StatusBadge status={exp.status} /></span>
                        </h2>
                        <p className="text-slate-400 text-sm font-mono mt-1 flex items-center space-x-4">
                            <span>ID: #{exp.id}</span>
                            <span>•</span>
                            <span className="flex items-center capitalize"><TaskIcon task={exp.task} /><span className="ml-1">{exp.task}</span></span>
                            <span>•</span>
                            <span>{exp.model}</span>
                            <span>•</span>
                            <span className="flex items-center"><Wand2 size={12} className="mr-1 text-slate-500" /> {exp.augmentation}</span>
                        </p>
                    </div>
                 </div>
                 
                 {/* Action Bar */}
                 <div className="flex items-center space-x-3">
                     {/* Pause / Resume / Stop Controls */}
                     {(exp.status === 'running' || exp.status === 'paused') && (
                         <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-lg border border-slate-800">
                            <button 
                                onClick={() => toggleExperimentStatus(exp.id)}
                                className={`p-2 rounded hover:bg-slate-800 transition-colors ${exp.status === 'running' ? 'text-amber-400' : 'text-emerald-400'}`} 
                                title={exp.status === 'running' ? 'Pause' : 'Resume'}
                            >
                                {exp.status === 'running' ? <Pause size={18} /> : <Play size={18} />}
                            </button>
                            <button 
                                onClick={() => handleStopClick(exp.id)}
                                className="p-2 rounded hover:bg-slate-800 text-rose-500" 
                                title="Stop"
                            >
                                <Square size={18} />
                            </button>
                         </div>
                     )}
                     <button 
                        onClick={() => setShowLogModal(true)}
                        className="flex items-center px-3 py-2 bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs font-bold rounded border border-slate-700 transition-colors"
                     >
                        <Terminal size={14} className="mr-2" /> 查看日志
                     </button>
                     <button 
                        onClick={() => handleSaveToRegistry(exp)}
                        disabled={!isCompleted}
                        title={!isCompleted ? "训练未完成，无法导出" : "将权重保存到权重库"}
                        className={`flex items-center px-3 py-2 text-xs font-bold rounded border transition-colors ${isCompleted ? 'bg-emerald-900/30 hover:bg-emerald-900/50 text-emerald-400 border-emerald-800' : 'bg-slate-900/50 text-slate-600 border-slate-800 cursor-not-allowed'}`}
                     >
                        <HardDrive size={14} className="mr-2" /> 保存到权重库
                     </button>
                 </div>
            </div>

            {/* Metrics Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[350px]">
                {/* Loss Chart */}
                <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col">
                   <h3 className="text-slate-300 font-medium mb-4 text-sm flex items-center"><Activity size={14} className="mr-2 text-slate-500"/> Loss Curve</h3>
                   <div className="flex-1 min-h-0">
                      <ResponsiveContainer width="100%" height="100%">
                         <LineChart data={TRAINING_CHART_DATA}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} itemStyle={{ fontSize: '12px' }} />
                            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                            <Line type="monotone" dataKey="trainLoss" stroke="#06b6d4" strokeWidth={2} dot={false} name="Train Loss" activeDot={{ r: 4 }} />
                            <Line type="monotone" dataKey="valLoss" stroke="#f43f5e" strokeWidth={2} dot={false} name="Val Loss" />
                         </LineChart>
                      </ResponsiveContainer>
                   </div>
                </div>

                {/* Accuracy Chart */}
                <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col">
                   <h3 className="text-slate-300 font-medium mb-4 text-sm flex items-center"><Target size={14} className="mr-2 text-slate-500"/> Metric ({exp.task === 'classification' ? 'Accuracy' : 'mAP'})</h3>
                   <div className="flex-1 min-h-0">
                      <ResponsiveContainer width="100%" height="100%">
                         <AreaChart data={TRAINING_CHART_DATA}>
                            <defs>
                                <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} itemStyle={{ fontSize: '12px' }} />
                            <Area type="monotone" dataKey="metric" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorMetric)" name="mAP@0.5:0.95" />
                         </AreaChart>
                      </ResponsiveContainer>
                   </div>
                </div>
            </div>

            {/* Config Summary & Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 glass-panel p-6 rounded-xl border border-slate-800">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-white font-bold text-sm">Configuration Summary</h3>
                        <div className="flex space-x-2">
                            <button 
                                onClick={() => setShowConfigModal(true)}
                                className="p-1.5 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 rounded transition-colors" 
                                title="预览参数 JSON"
                            >
                                <Eye size={16} />
                            </button>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-y-4 gap-x-8 text-sm">
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Optimizer</span>
                            <span className="text-slate-300 font-mono">SGD</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Learning Rate</span>
                            <span className="text-slate-300 font-mono">0.01</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Batch Size</span>
                            <span className="text-slate-300 font-mono">32</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Image Size</span>
                            <span className="text-slate-300 font-mono">640x640</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Device</span>
                            <span className="text-slate-300 font-mono">4x A100</span>
                        </div>
                        <div>
                             <span className="block text-slate-500 text-xs uppercase mb-1">Started At</span>
                             <span className="text-slate-300 font-mono">{exp.startedAt}</span>
                        </div>
                    </div>
                </div>
                
                <div className="glass-panel p-6 rounded-xl border border-slate-800 flex flex-col justify-center space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Best Epoch</span>
                        <span className="text-white font-mono font-bold">48</span>
                    </div>
                    <div className="w-full h-px bg-slate-800"></div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Best Metric</span>
                        <span className="text-emerald-400 font-mono font-bold">{exp.accuracy}</span>
                    </div>
                    <div className="w-full h-px bg-slate-800"></div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Est. Time Left</span>
                        <span className="text-cyan-400 font-mono font-bold">{isRunning ? '1h 24m' : '-'}</span>
                    </div>
                </div>
            </div>

            {/* Modals */}
            <LogModal isOpen={showLogModal} onClose={() => setShowLogModal(false)} logs={dummyLogs} />
            <ConfigPreviewModal isOpen={showConfigModal} onClose={() => setShowConfigModal(false)} config={exp.config || {}} />
        </div>
    );
  };

  return (
    <>
      {/* GLOBAL NOTIFICATION (Fixed Position to avoid layout shift) */}
      {notification && (
        <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[200] px-4 py-2 rounded-lg shadow-lg border flex items-center ${
            notification.type === 'error' ? 'bg-rose-900/90 border-rose-500 text-white' : 
            notification.type === 'success' ? 'bg-emerald-900/90 border-emerald-500 text-white' :
            'bg-cyan-900/90 border-cyan-500 text-white'
        }`}>
           {notification.type === 'error' ? <AlertTriangle size={16} className="mr-2" /> : 
            notification.type === 'success' ? <CheckCircle size={16} className="mr-2" /> :
            <HelpCircle size={16} className="mr-2" />
           }
           <span className="text-sm font-medium">{notification.msg}</span>
        </div>
      )}

      {/* STOP CONFIRMATION MODAL */}
      <ConfirmModal 
        isOpen={stopModal.isOpen} 
        onClose={() => setStopModal({isOpen: false, id: null})}
        onConfirm={confirmStop}
        title="停止训练"
        msg="确定要强制停止当前训练任务吗？此操作不可恢复，未保存的 Checkpoint 将丢失。"
        type="danger"
      />

      {/* RENAME MODAL */}
      <RenameModal 
         isOpen={renameModal.isOpen}
         onClose={() => setRenameModal({isOpen: false, id: null, currentName: ''})}
         onConfirm={confirmRename}
         initialName={renameModal.currentName}
      />

      {view === 'list' && renderList()}
      {view === 'create' && renderCreate()}
      {view === 'detail' && renderDetail()}
    </>
  );
};

export default TrainingMonitor;