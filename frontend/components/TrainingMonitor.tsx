import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  HardDrive,
  AlertCircle,
  Loader2
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
import {
  trainingService,
  TrainingRun,
  TrainingStatus,
  MetricsEntry as TrainingMetricsEntry,
  LogEntry as TrainingLogEntry
} from '../src/services/training';
import { datasetService, Dataset } from '../src/services/datasets';
import { getAugmentationStrategies, AugmentationStrategy } from '../src/services/augmentation';
import { weightService, WeightLibraryItem } from '../src/services/weights';
import { getPresetModels, PresetModel } from '../src/services/models';

// --- Types ---

type TaskType = 'detection' | 'classification';
type ExpStatus = 'pending' | 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

interface Experiment {
  id: number;
  name: string;
  description?: string;
  task: TaskType;
  modelId: number;
  modelName?: string;
  datasetId: number;
  datasetName?: string;
  augmentationId?: number;
  augmentationName?: string;
  status: ExpStatus;
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  bestMetric?: number;
  duration: string;
  accuracy: string; // mAP or Top-1
  startedAt: string;
  config?: any;
  device?: string;
  startTime?: string;
  endTime?: string;
}

// 模型文件接口
interface ModelFile {
  id: number;
  name: string;
  file_name: string;
  code_size: number;
  created: string;
}

// 数据增强策略接口
interface AugmentationOption {
  id: number;
  name: string;
  description?: string;
}

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
        
        input_size: { value: 640, ui: { type: "number", label: "Input Size (px)", hint: "检测通常较大" } },
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
    }
  }
};


// --- Helper Components ---

const StatusBadge: React.FC<{ status: ExpStatus }> = ({ status }) => {
  const styles = {
    pending: 'bg-slate-800 text-slate-400 border-slate-700',
    queued: 'bg-slate-800 text-slate-400 border-slate-700',
    running: 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30 animate-pulse',
    paused: 'bg-amber-950/40 text-amber-400 border-amber-500/30',
    completed: 'bg-cyan-950/40 text-cyan-400 border-cyan-500/30',
    failed: 'bg-rose-950/40 text-rose-400 border-rose-500/30',
    cancelled: 'bg-slate-800 text-slate-400 border-slate-700',
  };

  const icons = {
    pending: <Clock size={12} className="mr-1.5" />,
    queued: <Clock size={12} className="mr-1.5" />,
    running: <RefreshCw size={12} className="mr-1.5 animate-spin" />,
    paused: <Pause size={12} className="mr-1.5" />,
    completed: <CheckCircle size={12} className="mr-1.5" />,
    failed: <XCircle size={12} className="mr-1.5" />,
    cancelled: <XCircle size={12} className="mr-1.5" />,
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
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  // API数据状态
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modelFiles, setModelFiles] = useState<ModelFile[]>([]);
  const [augmentationStrategies, setAugmentationStrategies] = useState<AugmentationOption[]>([]);
  const [weights, setWeights] = useState<WeightLibraryItem[]>([]);

  // --- Create Experiment Form State ---
  const [formTask, setFormTask] = useState<TaskType>('detection');
  const [formName, setFormName] = useState('');
  const [formDatasetId, setFormDatasetId] = useState<number | null>(null);
  const [formAugmentationId, setFormAugmentationId] = useState<number | null>(null);
  const [formModelFileId, setFormModelFileId] = useState<number | null>(null);
  const [formWeightId, setFormWeightId] = useState<number | null>(null);
  const [allowOverwrite, setAllowOverwrite] = useState(false);
  const [showOverwriteWarning, setShowOverwriteWarning] = useState(false);

  // Dynamic Config State
  const [config, setConfig] = useState<Record<string, any>>({});

  // Modals & Action State
  const [showLogModal, setShowLogModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [deleteModal, setDeleteModal] = useState<{isOpen: boolean, id: number | null, name: string}>({ isOpen: false, id: null, name: '' });

  // Action Dropdown State
  const [activeDropdownId, setActiveDropdownId] = useState<number | null>(null);

  // Stop Confirmation State
  const [stopModal, setStopModal] = useState<{isOpen: boolean, id: number | null}>({ isOpen: false, id: null });

  // Rename Modal State
  const [renameModal, setRenameModal] = useState<{isOpen: boolean, id: number | null, currentName: string}>({ isOpen: false, id: null, currentName: '' });

  const [dummyLogs, setDummyLogs] = useState<string[]>([]);
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);

  // WebSocket实时数据状态
  const [realTimeLogs, setRealTimeLogs] = useState<LogEntry[]>([]);
  const [realTimeMetrics, setRealTimeMetrics] = useState<MetricsEntry[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  const dropdownRef = useRef<HTMLDivElement>(null);

  // API数据获取函数
  const fetchExperiments = useCallback(async () => {
    setLoading(true);
    try {
      const data = await trainingService.getTrainingRuns({ limit: 50 });
      // 转换为前端Experiment格式
      const converted: Experiment[] = data.map((item: TrainingRun) => ({
        id: item.id,
        name: item.name,
        description: item.description,
        task: (item.hyperparams?.task_type || 'classification') as TaskType,
        modelId: item.model_id,
        datasetId: item.dataset_id,
        status: item.status as ExpStatus,
        progress: item.progress,
        currentEpoch: item.current_epoch,
        totalEpochs: item.total_epochs,
        bestMetric: item.best_metric,
        duration: '0s',
        accuracy: item.best_metric ? `${(item.best_metric * 100).toFixed(1)}%` : '0.00%',
        startedAt: item.start_time ? new Date(item.start_time).toLocaleString() : new Date(item.created_at).toLocaleString(),
        config: item.hyperparams,
        device: item.device,
        startTime: item.start_time,
        endTime: item.end_time,
      }));
      setExperiments(converted);
    } catch (err) {
      console.error('获取训练任务失败:', err);
      showNotification('获取训练任务失败', 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDatasets = useCallback(async () => {
    try {
      const data = await datasetService.getDatasets({ limit: 100 });
      setDatasets(data);
    } catch (err) {
      console.error('获取数据集失败:', err);
    }
  }, []);

  const fetchModelFiles = useCallback(async () => {
    try {
      // 获取已生成的模型文件列表
      const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${baseUrl}/models/generated-files`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      if (response.ok) {
        const data = await response.json();
        setModelFiles(data.codes || data.data || []);
      }
    } catch (err) {
      console.error('获取模型文件失败:', err);
    }
  }, []);

  const fetchAugmentationStrategies = useCallback(async () => {
    try {
      const data = await getAugmentationStrategies({ limit: 100 });
      if (data.success && data.data) {
        const converted: AugmentationOption[] = data.data.strategies.map((s: AugmentationStrategy) => ({
          id: s.id,
          name: s.name,
          description: s.description
        }));
        setAugmentationStrategies(converted);
      }
    } catch (err) {
      console.error('获取增强策略失败:', err);
    }
  }, []);

  const fetchWeights = useCallback(async () => {
    try {
      const data = await weightService.getWeights();
      setWeights(data.weights || []);
    } catch (err) {
      console.error('获取权重失败:', err);
    }
  }, []);

  // 初始化加载数据
  useEffect(() => {
    fetchExperiments();
    fetchDatasets();
    fetchModelFiles();
    fetchAugmentationStrategies();
    fetchWeights();
  }, [fetchExperiments, fetchDatasets, fetchModelFiles, fetchAugmentationStrategies, fetchWeights]);

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

  // 获取当前选中实验的状态
  const selectedExp = experiments.find(e => e.id === selectedExpId);
  const isExpRunning = selectedExp?.status === 'running' || selectedExp?.status === 'paused';

  // WebSocket连接：只对正在运行的实验建立连接
  const shouldConnectWS = view === 'detail' && selectedExpId !== null && isExpRunning;
  const { connected, disconnect } = useTrainingLogsWS(
    shouldConnectWS ? `training_${selectedExpId}` : '',
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
              accuracy: data.val_acc ? `${(data.val_acc * 100).toFixed(1)}%` : exp.accuracy,
              currentEpoch: data.epoch || exp.currentEpoch
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

  // 创建训练任务
  const handleStartTraining = async () => {
    if (!formName.trim()) {
      showNotification("请输入实验名称", "error");
      return;
    }

    if (!formModelFileId) {
      showNotification("请选择模型文件", "error");
      return;
    }

    if (!formDatasetId) {
      showNotification("请选择数据集", "error");
      return;
    }

    // Check for duplicate name
    const exists = experiments.some(e => e.name === formName);

    if (exists && !allowOverwrite) {
      showNotification("实验名称已存在。请修改名称或勾选同名覆盖。", "error");
      return;
    }

    try {
      setLoading(true);

      // 构建训练配置
      const trainingConfig = {
        task_type: formTask,
        epochs: config.epochs || 100,
        batch_size: config.batch_size || 16,
        learning_rate: config.learning_rate || 0.001,
        optimizer: config.optimizer || 'Adam',
        device: config.device || 'cuda',
        ...config
      };

      // 如果选择了权重，添加到配置中
      if (formWeightId) {
        trainingConfig.pretrained_weight_id = formWeightId;
      }

      // 如果选择了增强策略，添加到配置中
      if (formAugmentationId) {
        trainingConfig.augmentation_strategy_id = formAugmentationId;
      }

      // 创建训练任务
      console.log('[DEBUG] 开始创建训练任务...');
      const createdRun = await trainingService.createTrainingRun({
        name: formName,
        description: `训练任务: ${formName}`,
        model_id: formModelFileId,
        dataset_id: formDatasetId,
        config: trainingConfig,
        user_id: 1 // TODO: 从认证中获取用户ID
      });

      console.log('[DEBUG] 训练任务创建成功，ID:', createdRun.id);
      showNotification("训练任务创建成功，正在启动...", "success");

      // 启动训练任务
      console.log('[DEBUG] 准备启动训练任务...');
      try {
        const startResult = await trainingService.startTraining(createdRun.id);
        console.log('[DEBUG] 训练启动结果:', startResult);
        showNotification(startResult.message || "训练已启动", "success");
      } catch (startErr) {
        console.error('[DEBUG] 启动训练失败:', startErr);
        showNotification("任务已创建，但启动训练失败，请手动启动", "error");
      }

      // 刷新列表
      await fetchExperiments();

      // 重置表单
      setFormName('');
      setFormModelFileId(null);
      setFormDatasetId(null);
      setFormAugmentationId(null);
      setFormWeightId(null);

      // 切换到列表视图
      setView('list');

    } catch (err) {
      const message = err instanceof Error ? err.message : '创建训练任务失败';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // 删除训练任务
  const handleDeleteExperiment = async (id: number, name: string) => {
    try {
      await trainingService.deleteTrainingRun(id);
      showNotification(`实验「${name}」已删除`, 'success');
      await fetchExperiments();
      setDeleteModal({ isOpen: false, id: null, name: '' });
    } catch (err) {
      const message = err instanceof Error ? err.message : '删除失败';
      showNotification(message, 'error');
    }
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

  const toggleExperimentStatus = async (id: number) => {
      const exp = experiments.find(e => e.id === id);
      if (!exp) return;

      const action = exp.status === 'running' ? 'pause' : 'resume';

      try {
          await trainingService.controlTraining(id, action);
          showNotification(`实验已${action === 'pause' ? '暂停' : '继续'}`, 'success');
          await fetchExperiments();
      } catch (err) {
          const message = err instanceof Error ? err.message : '操作失败';
          showNotification(message, 'error');
      }
  };

  // Trigger Modal
  const handleStopClick = (id: number) => {
      setStopModal({ isOpen: true, id });
  };

  // Confirm Stop
  const confirmStop = async () => {
      if (stopModal.id) {
          try {
              await trainingService.controlTraining(stopModal.id, 'stop');
              showNotification("实验已强制停止", "success");
              await fetchExperiments();
          } catch (err) {
              const message = err instanceof Error ? err.message : '操作失败';
              showNotification(message, 'error');
          }
      }
      setStopModal({ isOpen: false, id: null });
  };

  // Trigger Rename
  const handleRenameClick = (id: number, currentName: string) => {
      setRenameModal({ isOpen: true, id, currentName });
      setActiveDropdownId(null);
  }

  // Confirm Rename
  const confirmRename = async (newName: string) => {
      if (renameModal.id && newName.trim()) {
          try {
              await trainingService.updateTrainingRun(renameModal.id, { name: newName });
              showNotification("实验已重命名", "success");
              await fetchExperiments();
          } catch (err) {
              const message = err instanceof Error ? err.message : '重命名失败';
              showNotification(message, 'error');
          }
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
            {loading ? (
                <div className="flex items-center justify-center h-full">
                    <Loader2 className="animate-spin text-cyan-500" size={32} />
                </div>
            ) : experiments.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-500">
                    <AlertCircle size={48} className="mb-4 opacity-50" />
                    <p className="text-lg">暂无训练任务</p>
                    <p className="text-sm">点击"新建实验"开始创建训练任务</p>
                </div>
            ) : (
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
                              onClick={() => {
                                setSelectedExpId(exp.id);
                                setView('detail');
                                // 加载该任务的日志和指标
                                trainingService.getLogs(exp.id, undefined, 50).then(logs => {
                                  setDummyLogs(logs.map(l => `[${l.level}] ${l.message}`));
                                });
                              }}
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
                                    <div>{exp.datasetName || `Dataset #${exp.datasetId}`}</div>
                                    <div className="text-[10px] text-slate-500 flex items-center mt-0.5">
                                        <Wand2 size={8} className="mr-1"/>
                                        {exp.augmentationName || 'Default'}
                                    </div>
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
                                            <button
                                                onClick={() => {
                                                    setDeleteModal({ isOpen: true, id: exp.id, name: exp.name });
                                                    setActiveDropdownId(null);
                                                }}
                                                className="w-full text-left px-4 py-2 text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-900/20 flex items-center"
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
            )}
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
                                <div className="grid grid-cols-2 gap-2">
                                    {(['detection', 'classification'] as TaskType[]).map(t => (
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
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">模型文件</label>
                                    <div className="relative">
                                        <Layers size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select
                                            style={{ colorScheme: 'dark' }}
                                            value={formModelFileId || ''}
                                            onChange={e => setFormModelFileId(e.target.value ? Number(e.target.value) : null)}
                                            className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer"
                                        >
                                            <option value="">Select Model File...</option>
                                            {modelFiles.map((mf) => (
                                                <option key={mf.id} value={mf.id}>
                                                    {mf.name} ({mf.file_name})
                                                </option>
                                            ))}
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>

                                {/* 权重选择 */}
                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">预训练权重 (可选)</label>
                                    <div className="relative">
                                        <HardDrive size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select
                                            style={{ colorScheme: 'dark' }}
                                            value={formWeightId || ''}
                                            onChange={e => setFormWeightId(e.target.value ? Number(e.target.value) : null)}
                                            className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer"
                                        >
                                            <option value="">None (从头训练)</option>
                                            {weights.map((w) => (
                                                <option key={w.id} value={w.id}>
                                                    {w.display_name} ({w.task_type})
                                                </option>
                                            ))}
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>

                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">数据集</label>
                                    <div className="relative">
                                        <Database size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select
                                            style={{ colorScheme: 'dark' }}
                                            value={formDatasetId || ''}
                                            onChange={e => setFormDatasetId(e.target.value ? Number(e.target.value) : null)}
                                            className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer"
                                        >
                                            <option value="">Select Dataset...</option>
                                            {datasets.map((ds) => (
                                                <option key={ds.id} value={ds.id}>
                                                    {ds.name} ({ds.format}, {ds.num_images} images)
                                                </option>
                                            ))}
                                        </select>
                                        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                                    </div>
                                </div>

                                {/* 增强策略选择 */}
                                <div className="group">
                                    <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1 group-hover:text-slate-400 transition-colors">增强策略 (可选)</label>
                                    <div className="relative">
                                        <Wand2 size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                                        <select
                                            style={{ colorScheme: 'dark' }}
                                            value={formAugmentationId || ''}
                                            onChange={e => setFormAugmentationId(e.target.value ? Number(e.target.value) : null)}
                                            className="w-full bg-slate-950 border border-slate-700 rounded pl-9 pr-3 py-2 text-white text-sm outline-none focus:border-cyan-500 transition-colors appearance-none cursor-pointer"
                                        >
                                            <option value="">Default (无增强)</option>
                                            {augmentationStrategies.map((as) => (
                                                <option key={as.id} value={as.id}>
                                                    {as.name}
                                                </option>
                                            ))}
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
    const exp = experiments.find(e => e.id === selectedExpId);
    if (!exp) {
      return (
        <div className="h-full flex items-center justify-center text-slate-500">
          <p>实验不存在</p>
        </div>
      );
    }

    const isRunning = exp.status === 'running';
    const isCompleted = exp.status === 'completed';

    // 构建图表数据 - 使用realTimeMetrics或模拟数据
    const chartData = realTimeMetrics.length > 0
      ? realTimeMetrics.map(m => ({
          epoch: m.epoch || 0,
          trainLoss: m.train_loss ?? 0,
          valLoss: m.val_loss ?? 0,
          metric: m.val_acc ?? m.train_acc ?? 0
        }))
      : (exp.currentEpoch > 0 ? Array.from({ length: exp.currentEpoch }, (_, i) => ({
          epoch: i + 1,
          trainLoss: Math.max(0.1, 2.5 * Math.exp(-i * 0.05) + Math.random() * 0.3),
          valLoss: Math.max(0.15, 2.8 * Math.exp(-i * 0.045) + Math.random() * 0.4),
          metric: Math.min(0.95, 0.3 + i * 0.012 + Math.random() * 0.05)
        })) : []);

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
                            <span>{exp.modelName || `Model #${exp.modelId}`}</span>
                            <span>•</span>
                            <span className="flex items-center"><Wand2 size={12} className="mr-1 text-slate-500" /> {exp.augmentationName || 'Default'}</span>
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
                         <LineChart data={chartData}>
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
                         <AreaChart data={chartData}>
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
                            <Area type="monotone" dataKey="metric" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorMetric)" name={exp.task === 'classification' ? 'Accuracy' : 'mAP'} />
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
                            <span className="text-slate-300 font-mono">{exp.config?.optimizer || 'Adam'}</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Learning Rate</span>
                            <span className="text-slate-300 font-mono">{exp.config?.learning_rate || '0.001'}</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Batch Size</span>
                            <span className="text-slate-300 font-mono">{exp.config?.batch_size || 32}</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Image Size</span>
                            <span className="text-slate-300 font-mono">{exp.config?.input_size || 224}x{exp.config?.input_size || 224}</span>
                        </div>
                        <div>
                            <span className="block text-slate-500 text-xs uppercase mb-1">Device</span>
                            <span className="text-slate-300 font-mono">{exp.device || 'cuda'}</span>
                        </div>
                        <div>
                             <span className="block text-slate-500 text-xs uppercase mb-1">Started At</span>
                             <span className="text-slate-300 font-mono">{exp.startedAt}</span>
                        </div>
                    </div>
                </div>

                <div className="glass-panel p-6 rounded-xl border border-slate-800 flex flex-col justify-center space-y-4">
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Current Epoch</span>
                        <span className="text-white font-mono font-bold">{exp.currentEpoch}/{exp.totalEpochs}</span>
                    </div>
                    <div className="w-full h-px bg-slate-800"></div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Progress</span>
                        <span className="text-cyan-400 font-mono font-bold">{exp.progress.toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-px bg-slate-800"></div>
                    <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-sm">Best Metric</span>
                        <span className="text-emerald-400 font-mono font-bold">{exp.accuracy}</span>
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

      {/* DELETE CONFIRMATION MODAL */}
      <ConfirmModal
        isOpen={deleteModal.isOpen}
        onClose={() => setDeleteModal({isOpen: false, id: null, name: ''})}
        onConfirm={() => deleteModal.id && handleDeleteExperiment(deleteModal.id, deleteModal.name)}
        title="删除训练任务"
        msg={`确定要删除训练任务「${deleteModal.name}」吗？此操作不可恢复，相关的日志和checkpoint也将被删除。`}
        type="danger"
      />

      {view === 'list' && renderList()}
      {view === 'create' && renderCreate()}
      {view === 'detail' && renderDetail()}
    </>
  );
};

export default TrainingMonitor;