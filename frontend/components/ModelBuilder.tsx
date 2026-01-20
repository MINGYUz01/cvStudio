import React, { useState, useRef, useEffect } from 'react';
import {
  Layers, ArrowRight, Save, Plus, GitBranch,
  Edit3, Trash2, ZoomIn, ZoomOut, Sidebar as SidebarIcon,
  CheckCircle, AlertTriangle, Code, Info, X, Copy,
  Layout, Package, Box, AlertOctagon, HelpCircle,
  Database, HardDrive, Download, Upload, Tag,
  Loader2, FileText, FolderOpen, ChevronLeft, ChevronRight
} from 'lucide-react';
import { ModelNode, WeightCheckpoint, WeightTreeItem } from '../types';
import modelsAPI from '../src/services/models';
import { weightService, TaskType } from '../src/services/weights';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import CodePreviewModal from './CodePreviewModal';
import InputModal from './InputModal';
import WeightTreeView from './WeightTreeView';

// 统计图表颜色
const CHART_COLORS = ['#22d3ee', '#a855f7', '#f43f5e', '#fbbf24', '#34d399', '#60a5fa'];

// Constants for precise layout
const NODE_WIDTH = 160;

// --- ATOMIC OPERATORS ---
const ATOMIC_NODES: Record<string, { label: string, color: string, category: string, params: { name: string, type: 'text'|'number'|'bool'|'select'|'dims4'|'dimsN', default: any, options?: string[] }[] }> = {
  // IO
  "Input": { label: "Input", color: "bg-slate-600", category: "IO", params: [{ name: "c", type: "number", default: 3 }, { name: "h", type: "number", default: 640 }, { name: "w", type: "number", default: 640 }] },

  // Layers
  "Conv2d": { label: "Conv2d", color: "bg-cyan-600", category: "Layer", params: [{ name: "in", type: "number", default: 3 }, { name: "out", type: "number", default: 64 }, { name: "k", type: "number", default: 3 }, { name: "s", type: "number", default: 1 }, { name: "p", type: "number", default: 1 }] },
  "ConvTranspose2d": { label: "ConvT2d", color: "bg-cyan-600", category: "Layer", params: [{ name: "in", type: "number", default: 64 }, { name: "out", type: "number", default: 64 }, { name: "k", type: "number", default: 3 }, { name: "s", type: "number", default: 2 }, { name: "p", type: "number", default: 1 }, { name: "op", type: "number", default: 1 }] },
  "Linear": { label: "Linear", color: "bg-cyan-600", category: "Layer", params: [{ name: "in_f", type: "number", default: 512 }, { name: "out_f", type: "number", default: 10 }] },
  "BatchNorm2d": { label: "BN2d", color: "bg-cyan-700", category: "Layer", params: [{ name: "num_f", type: "number", default: 64 }] },
  "GroupNorm": { label: "GroupNorm", color: "bg-cyan-700", category: "Layer", params: [{ name: "groups", type: "number", default: 32 }, { name: "num_f", type: "number", default: 64 }] },
  "InstanceNorm2d": { label: "InstanceNorm", color: "bg-cyan-700", category: "Layer", params: [{ name: "num_f", type: "number", default: 64 }, { name: "eps", type: "number", default: 1e-5 }] },
  "LayerNorm": { label: "LayerNorm", color: "bg-cyan-700", category: "Layer", params: [{ name: "normalized_size", type: "number", default: 64 }, { name: "eps", type: "number", default: 1e-5 }] },
  "Dropout": { label: "Dropout", color: "bg-slate-500", category: "Layer", params: [{ name: "p", type: "number", default: 0.5 }] },
  "Flatten": { label: "Flatten", color: "bg-slate-500", category: "Layer", params: [] },

  // Activations
  "ReLU": { label: "ReLU", color: "bg-purple-600", category: "Activation", params: [{ name: "inplace", type: "bool", default: true }] },
  "ReLU6": { label: "ReLU6", color: "bg-purple-600", category: "Activation", params: [{ name: "inplace", type: "bool", default: true }] },
  "LeakyReLU": { label: "LReLU", color: "bg-purple-600", category: "Activation", params: [{ name: "slope", type: "number", default: 0.01 }] },
  "ELU": { label: "ELU", color: "bg-purple-600", category: "Activation", params: [{ name: "alpha", type: "number", default: 1.0 }] },
  "SiLU": { label: "SiLU", color: "bg-purple-600", category: "Activation", params: [] },
  "Hardswish": { label: "HardSwish", color: "bg-purple-600", category: "Activation", params: [{ name: "inplace", type: "bool", default: true }] },
  "Mish": { label: "Mish", color: "bg-purple-600", category: "Activation", params: [] },
  "GELU": { label: "GELU", color: "bg-purple-600", category: "Activation", params: [] },
  "Tanh": { label: "Tanh", color: "bg-purple-600", category: "Activation", params: [] },
  "Sigmoid": { label: "Sigmoid", color: "bg-purple-600", category: "Activation", params: [] },
  "Softmax": { label: "Softmax", color: "bg-purple-600", category: "Activation", params: [{ name: "dim", type: "number", default: 1 }] },

  // Pooling
  "MaxPool2d": { label: "MaxPool", color: "bg-rose-600", category: "Pooling", params: [{ name: "k", type: "number", default: 2 }, { name: "s", type: "number", default: 2 }] },
  "AvgPool2d": { label: "AvgPool", color: "bg-rose-600", category: "Pooling", params: [{ name: "k", type: "number", default: 2 }] },
  "AdaptiveAvg": { label: "AdaptAvg", color: "bg-rose-600", category: "Pooling", params: [{ name: "out", type: "number", default: 1 }] },
  "AdaptiveMaxPool2d": { label: "AdaptMax", color: "bg-rose-600", category: "Pooling", params: [{ name: "out", type: "number", default: 1 }] },

  // Ops
  "Concat": { label: "Concat", color: "bg-amber-600", category: "Ops", params: [{ name: "dim", type: "number", default: 1 }] },
  "Add": { label: "Add", color: "bg-amber-600", category: "Ops", params: [] },
  "Upsample": { label: "Upsample", color: "bg-amber-600", category: "Ops", params: [{ name: "scale", type: "number", default: 2 }, { name: "mode", type: "select", default: "nearest", options: ["nearest", "bilinear", "bicubic"] }] },
  "Identity": { label: "Identity", color: "bg-slate-500", category: "Ops", params: [] },
  "Pad2d": { label: "Pad2d", color: "bg-amber-600", category: "Ops", params: [{ name: "pad", type: "dims4", default: [1, 1, 1, 1] }, { name: "mode", type: "select", default: "constant", options: ["constant", "reflect", "replicate", "circular"] }] },
  "Reshape": { label: "Reshape", color: "bg-amber-600", category: "Ops", params: [{ name: "shape", type: "dimsN", default: [-1, 64] }] },
  "Permute": { label: "Permute", color: "bg-amber-600", category: "Ops", params: [{ name: "dims", type: "dimsN", default: [0, 2, 1] }] },
  "ChannelShuffle": { label: "ChShuffle", color: "bg-amber-600", category: "Ops", params: [{ name: "groups", type: "number", default: 2 }] },
  "Squeeze": { label: "Squeeze", color: "bg-amber-600", category: "Ops", params: [{ name: "dim", type: "number", default: 1 }] },
  "Unsqueeze": { label: "Unsqueeze", color: "bg-amber-600", category: "Ops", params: [{ name: "dim", type: "number", default: 1 }] },
  "PixelShuffle": { label: "PixelShuffle", color: "bg-amber-600", category: "Ops", params: [{ name: "scale", type: "number", default: 2 }] },
};

// 2. TEMPLATES (Structures)
interface BlockTemplate {
  name: string;
  description: string;
  nodes: VisualNode[];
  connections: Connection[];
}

interface VisualNode extends ModelNode { data: Record<string, any>; }
interface Connection { id: string; source: string; target: string; }
interface ModelData {
  id: string;              // 前端使用的ID（格式：server_{db_id} 或 local_*）
  arch_id?: number;        // 数据库中的架构ID
  name: string;
  version: string;
  status: string;
  type: string;
  updated: string;
  description?: string;    // 模型描述
  created?: string;        // 创建时间
  node_count?: number;     // 节点数量
  connection_count?: number; // 连接数量
  nodes: VisualNode[];
  connections: Connection[];
  filename?: string;       // 服务器文件名（保留用于兼容）
}


// --- Internal Reusable Dialog Component ---
interface DialogProps {
    isOpen: boolean;
    type: 'confirm' | 'prompt' | 'alert';
    title: string;
    message?: string;
    defaultValue?: string;
    onClose: () => void;
    onConfirm: (val?: string) => void;
}

const CustomDialog: React.FC<DialogProps> = ({ isOpen, type, title, message, defaultValue, onClose, onConfirm }) => {
    const [inputValue, setInputValue] = useState(defaultValue || '');
    
    useEffect(() => {
        if(isOpen && defaultValue) setInputValue(defaultValue);
    }, [isOpen, defaultValue]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-sm shadow-2xl animate-in fade-in zoom-in duration-200" onClick={(e) => e.stopPropagation()}>
                <div className="flex flex-col items-center text-center mb-6">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 ${type === 'confirm' ? 'bg-rose-900/30 text-rose-500' : 'bg-cyan-900/30 text-cyan-500'}`}>
                        {type === 'confirm' ? <AlertOctagon size={24} /> : <HelpCircle size={24} />}
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
                    {message && <p className="text-sm text-slate-400">{message}</p>}
                </div>
                
                {type === 'prompt' && (
                    <div className="mb-6">
                        <input 
                            autoFocus
                            type="text" 
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-white outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') onConfirm(inputValue);
                            }}
                        />
                    </div>
                )}

                <div className="flex gap-3">
                    <button 
                        onClick={onClose}
                        className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700"
                    >
                        取消
                    </button>
                    <button 
                        onClick={() => onConfirm(inputValue)}
                        className={`flex-1 py-2.5 text-white rounded-lg font-bold transition-colors shadow-lg ${type === 'confirm' ? 'bg-rose-600 hover:bg-rose-500 shadow-rose-900/20' : 'bg-cyan-600 hover:bg-cyan-500 shadow-cyan-900/20'}`}
                    >
                        {type === 'confirm' ? '确认删除' : '确认'}
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- 错误详情弹窗组件 ---
interface ErrorDetailDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  errors: string[];
  warnings: string[];
  onClose: () => void;
}

const ErrorDetailDialog: React.FC<ErrorDetailDialogProps> = ({
  isOpen, title, message, errors, warnings, onClose
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-rose-700/50 p-6 rounded-xl w-full max-w-lg shadow-2xl animate-in fade-in zoom-in duration-200">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-rose-900/30 flex items-center justify-center shrink-0">
            <AlertOctagon size={20} className="text-rose-500" />
          </div>
          <h3 className="text-xl font-bold text-white">{title}</h3>
        </div>

        {message && (
          <p className="text-slate-300 mb-4">{message}</p>
        )}

        {errors.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-rose-400 mb-2">错误详情:</h4>
            <ul className="text-sm text-slate-300 space-y-1 bg-slate-950/50 p-3 rounded-lg max-h-40 overflow-y-auto">
              {errors.map((err, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-rose-500 mt-0.5 shrink-0">•</span>
                  <span className="break-words">{err}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {warnings.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-amber-400 mb-2">警告:</h4>
            <ul className="text-sm text-slate-300 space-y-1 bg-slate-950/50 p-3 rounded-lg max-h-32 overflow-y-auto">
              {warnings.map((warn, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5 shrink-0">•</span>
                  <span className="break-words">{warn}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <button
          onClick={onClose}
          className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-medium transition-colors"
        >
          关闭
        </button>
      </div>
    </div>
  );
};


const ModelBuilder: React.FC = () => {
  // Main View State: 'architectures', 'weights', or 'generated'
  const [activeTab, setActiveTab] = useState<'architectures' | 'weights' | 'generated'>('architectures');

  // Sub View for Architectures: 'list' or 'builder'
  const [archView, setArchView] = useState<'list' | 'builder'>('list');

  // Model State - 从服务器加载
  const [models, setModels] = useState<ModelData[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  // 预设模型状态
  const [presetModels, setPresetModels] = useState<any[]>([]);
  const [isLoadingPresets, setIsLoadingPresets] = useState(false);
  const [showPresets, setShowPresets] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  // 预设模型弹窗状态
  const [showPresetModal, setShowPresetModal] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<{id: number, name: string, description?: string} | null>(null);
  const [isCreatingFromPreset, setIsCreatingFromPreset] = useState(false);

  // Weights State
  const [weights, setWeights] = useState<WeightCheckpoint[]>([]);
  const [isLoadingWeights, setIsLoadingWeights] = useState(false);
  const [rootWeights, setRootWeights] = useState<WeightCheckpoint[]>([]);
  const [weightTree, setWeightTree] = useState<WeightTreeItem[]>([]);
  const [selectedWeight, setSelectedWeight] = useState<WeightTreeItem | null>(null);
  const [weightView, setWeightView] = useState<'list' | 'tree'>('list');

  // Generated Model Files State
  const [generatedFiles, setGeneratedFiles] = useState<Array<{id: number, filename: string, name: string, size: number, created: string}>>([]);

  // 辅助函数：日期格式化 - 显示相对时间
  const formatDate = (dateStr: string): string => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return '今天';
    if (diffDays === 1) return '昨天';
    if (diffDays < 7) return `${diffDays}天前`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}周前`;
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  // 辅助函数：根据模型生成智能标签
  const getModelTags = (model: ModelData): JSX.Element[] => {
    const tags: JSX.Element[] = [];

    // 类型标签
    const typeLower = model.type.toLowerCase();
    if (typeLower.includes('cnn') || typeLower.includes('conv')) {
      tags.push(<span key="cnn" className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-900/30 text-blue-400 border border-blue-500/20">CNN</span>);
    }
    if (typeLower.includes('rnn') || typeLower.includes('lstm') || typeLower.includes('gru')) {
      tags.push(<span key="rnn" className="px-2 py-0.5 rounded text-[10px] font-medium bg-amber-900/30 text-amber-400 border border-amber-500/20">RNN</span>);
    }
    if (typeLower.includes('transformer') || typeLower.includes('attention')) {
      tags.push(<span key="trans" className="px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-900/30 text-emerald-400 border border-emerald-500/20">Transformer</span>);
    }

    // 复杂度标签
    if (model.node_count) {
      if (model.node_count >= 50) {
        tags.push(<span key="complex" className="px-2 py-0.5 rounded text-[10px] font-medium bg-rose-900/30 text-rose-400 border border-rose-500/20">复杂</span>);
      } else if (model.node_count >= 20) {
        tags.push(<span key="medium" className="px-2 py-0.5 rounded text-[10px] font-medium bg-yellow-900/30 text-yellow-400 border border-yellow-500/20">中等</span>);
      } else {
        tags.push(<span key="simple" className="px-2 py-0.5 rounded text-[10px] font-medium bg-green-900/30 text-green-400 border border-green-500/20">简单</span>);
      }
    }

    return tags;
  };

  // 从服务器加载架构列表
  const loadServerArchitectures = async () => {
    setIsLoadingModels(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/architectures`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        // 将服务器架构转换为 ModelData 格式
        const serverModels: ModelData[] = (data.architectures || []).map((arch: any) => ({
          id: `server_${arch.id}`,
          arch_id: arch.id,              // 数据库ID
          name: arch.name,
          version: arch.version,
          status: 'Ready',
          type: arch.type,
          description: arch.description || '',  // 模型描述
          created: arch.created,                // 创建时间
          updated: arch.updated,
          node_count: arch.node_count || 0,     // 节点数量
          connection_count: arch.connection_count || 0, // 连接数量
          nodes: [], // 节点数据按需加载
          connections: [], // 连接数据按需加载
          filename: arch.file_name,  // 文件名（保留用于兼容）
        }));
        setModels(serverModels);
      }
    } catch (error) {
      console.error('加载服务器架构列表失败:', error);
      // 失败时使用默认列表
      setModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  };

  // 加载预设模型列表
  const loadPresetModels = async () => {
    setIsLoadingPresets(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/presets`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setPresetModels(data.presets || []);
      } else {
        setPresetModels([]);
      }
    } catch (error) {
      console.error('加载预设模型失败:', error);
      setPresetModels([]);
    } finally {
      setIsLoadingPresets(false);
    }
  };

  // 从预设模型创建架构 - 打开弹窗
  const handleCreateFromPreset = (presetId: number, presetName: string, presetDescription?: string) => {
    setSelectedPreset({ id: presetId, name: presetName, description: presetDescription });
    setShowPresetModal(true);
  };

  // 确认从预设模型创建架构
  const confirmCreateFromPreset = async (name: string, description: string) => {
    if (!selectedPreset) return;

    setIsCreatingFromPreset(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/presets/${selectedPreset.id}/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({ name, description }),
      });

      if (response.ok) {
        const result = await response.json();
        // 关闭弹窗
        setShowPresetModal(false);
        setSelectedPreset(null);
        // 显示成功通知
        showNotification(`架构「${name}」已创建！`, 'success');
        // 重新加载架构列表
        loadServerArchitectures();
      } else {
        const error = await response.json();
        const errorMsg = error.detail || error.message || '未知错误';
        showNotification(`创建失败: ${errorMsg}`, 'error');
      }
    } catch (error) {
      console.error('创建架构失败:', error);
      showNotification('创建失败，请稍后重试', 'error');
    } finally {
      setIsCreatingFromPreset(false);
    }
  };

  // 关闭预设模型弹窗
  const closePresetModal = () => {
    setShowPresetModal(false);
    setSelectedPreset(null);
  };

  // 获取难度标签样式
  const getDifficultyBadge = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner':
        return 'px-2 py-0.5 rounded text-[10px] font-medium bg-green-900/30 text-green-400 border border-green-500/20';
      case 'intermediate':
        return 'px-2 py-0.5 rounded text-[10px] font-medium bg-yellow-900/30 text-yellow-400 border border-yellow-500/20';
      case 'advanced':
        return 'px-2 py-0.5 rounded text-[10px] font-medium bg-rose-900/30 text-rose-400 border border-rose-500/20';
      default:
        return 'px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400';
    }
  };

  // 获取难度文本
  const getDifficultyText = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '入门';
      case 'intermediate': return '中级';
      case 'advanced': return '高级';
      default: return difficulty;
    }
  };

  // 获取分类图标
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'cnn': return '🔷';
      case 'rnn': return '🔄';
      case 'transformer': return '⚡';
      case 'classification': return '🏷️';
      case 'detection': return '🎯';
      default: return '📦';
    }
  };

  // 筛选预设模型
  const getFilteredPresets = () => {
    return presetModels.filter(preset => {
      if (selectedCategory !== 'all' && preset.category !== selectedCategory) return false;
      if (selectedDifficulty !== 'all' && preset.difficulty !== selectedDifficulty) return false;
      return true;
    });
  };

  // 从服务器加载权重列表
  const loadServerWeights = async () => {
    setIsLoadingWeights(true);
    try {
      // 并行加载所有权重和根节点权重
      const [allResponse, rootsResponse] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/weights`, {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        }),
        fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/weights/roots`, {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` },
        }),
      ]);

      // 处理所有权重
      if (allResponse.ok) {
        const data = await allResponse.json();
        const serverWeights: WeightCheckpoint[] = (data.weights || []).map((w: any) => ({
          id: w.id,
          name: w.name,
          display_name: w.display_name || `${w.name} v${w.version}`,
          description: w.description,
          task_type: w.task_type,
          version: w.version,
          file_name: w.file_name,
          file_size: w.file_size,
          file_size_mb: w.file_size_mb,
          framework: w.framework,
          is_root: w.is_root,
          source_type: w.source_type,
          architecture_id: w.architecture_id,
          created_at: w.created_at,
          architecture: w.task_type === 'classification' ? 'Classifier' :
                       w.task_type === 'detection' ? 'YOLOv8' : 'Unknown',
          format: w.framework === 'pytorch' ? 'PyTorch' : 'ONNX',
          size: w.file_size_mb ? `${w.file_size_mb} MB` : 'Unknown',
        }));
        setWeights(serverWeights);
      }

      // 处理根节点权重
      if (rootsResponse.ok) {
        const rootsData = await rootsResponse.json();
        const rootWeightsData: WeightCheckpoint[] = (rootsData.weights || []).map((w: any) => ({
          id: w.id,
          name: w.name,
          display_name: w.display_name || `${w.name} v${w.version}`,
          description: w.description,
          task_type: w.task_type,
          version: w.version,
          file_name: w.file_name,
          file_size: w.file_size,
          file_size_mb: w.file_size_mb,
          framework: w.framework,
          is_root: w.is_root,
          source_type: w.source_type,
          architecture_id: w.architecture_id,
          created_at: w.created_at,
          architecture: w.task_type === 'classification' ? 'Classifier' :
                       w.task_type === 'detection' ? 'YOLOv8' : 'Unknown',
          format: w.framework === 'pytorch' ? 'PyTorch' : 'ONNX',
          size: w.file_size_mb ? `${w.file_size_mb} MB` : 'Unknown',
        }));
        setRootWeights(rootWeightsData);
      }
    } catch (error) {
      console.error('加载服务器权重列表失败:', error);
      setWeights([]);
      setRootWeights([]);
    } finally {
      setIsLoadingWeights(false);
    }
  };

  // 加载权重树
  const loadWeightTree = async () => {
    try {
      const data = await weightService.getWeightTree();
      setWeightTree(data);
    } catch (error) {
      console.error('加载权重树失败:', error);
    }
  };

  // 选择权重查看详情
  const handleWeightSelect = async (weight: WeightCheckpoint) => {
    try {
      const subtree = await weightService.getWeightSubtree(weight.id);
      setSelectedWeight(subtree);
    } catch (error) {
      console.error('加载权重详情失败:', error);
    }
  };

  // 组件挂载时从服务器加载架构列表和预设模型
  useEffect(() => {
    loadServerArchitectures();
    loadPresetModels();
  }, []);
  
  // Builder State
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [originalFilename, setOriginalFilename] = useState<string | null>(null); // 原始文件名（用于更新）
  const [modelName, setModelName] = useState('Untitled Architecture');
  const [isEditingName, setIsEditingName] = useState(false);
  const [nodes, setNodes] = useState<VisualNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null); 
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 }); 
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [trashHover, setTrashHover] = useState(false);
  const [notification, setNotification] = useState<{msg: string, type: 'error' | 'success' | 'info'} | null>(null);

  // Code Generation State
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [codeMetadata, setCodeMetadata] = useState<any>(null);
  const [isGeneratingCode, setIsGeneratingCode] = useState(false);
  const [showCodePreview, setShowCodePreview] = useState(false);
  const [codeGenerationError, setCodeGenerationError] = useState<string | null>(null);
  // 代码预览来源：'builder' 表示从构建器生成，'library' 表示从模型库打开
  const [codePreviewSource, setCodePreviewSource] = useState<'builder' | 'library'>('builder');
  // 当前预览的文件名（用于library模式下的删除操作）
  const [currentPreviewFilename, setCurrentPreviewFilename] = useState<string | null>(null);
  const [currentPreviewFileId, setCurrentPreviewFileId] = useState<number | null>(null);

  // Weight Upload Dialog State
  const [showWeightUpload, setShowWeightUpload] = useState(false);

  // Node Expansion State (折叠式卡片)
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  // Custom Dialog State
  const [dialog, setDialog] = useState<{
      isOpen: boolean;
      type: 'confirm' | 'prompt' | 'alert';
      title: string;
      message?: string;
      defaultValue?: string;
      data?: any; // To store temporary ID or data needed for the action
      action: 'delete_model' | 'clear_canvas' | 'save_operator' | 'delete_weight';
  }>({
      isOpen: false,
      type: 'alert',
      title: '',
      action: 'clear_canvas'
  });

  // Error Detail Dialog State - 显示详细的错误和警告信息
  const [errorDialog, setErrorDialog] = useState<{
    show: boolean;
    title: string;
    message: string;
    errors: string[];
    warnings: string[];
  }>({
    show: false,
    title: '',
    message: '',
    errors: [],
    warnings: []
  });

  // Custom Templates State (Persisted)
  const [customTemplates, setCustomTemplates] = useState<BlockTemplate[]>(() => {
    try {
        const saved = localStorage.getItem('neurocore_custom_ops');
        if (saved) return JSON.parse(saved);
        return [];
    } catch { return []; }
  });

  // Persist Custom Templates whenever they change
  useEffect(() => {
    localStorage.setItem('neurocore_custom_ops', JSON.stringify(customTemplates));
  }, [customTemplates]);

  // Dragging & Connecting
  const [draggedItem, setDraggedItem] = useState<{type: string, isTemplate: boolean} | null>(null);
  const [movingNodeId, setMovingNodeId] = useState<string | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [drawingConnection, setDrawingConnection] = useState<{ sourceId: string, startX: number, startY: number, currX: number, currY: number } | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);
  const trashRef = useRef<HTMLDivElement>(null);

  // --- Keyboard Shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
        if ((e.key === 'Delete' || e.key === 'Backspace')) {
            if (selectedConnectionId) {
                setConnections(prev => prev.filter(c => c.id !== selectedConnectionId));
                setSelectedConnectionId(null);
            }
            if (selectedNodeId) {
                setNodes(prev => prev.filter(n => n.id !== selectedNodeId));
                setConnections(prev => prev.filter(c => c.source !== selectedNodeId && c.target !== selectedNodeId));
                setSelectedNodeId(null);
            }
        }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedConnectionId, selectedNodeId]);

  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // --- 加载生成的模型文件 ---
  const loadGeneratedFiles = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/generated-files`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        // 后端返回 data.codes，需要映射为前端期望的格式
        const files = (data.codes || []).map((item: any) => ({
          id: item.id,
          filename: item.file_name,
          name: item.name,
          size: item.code_size,
          created: item.created
        }));
        setGeneratedFiles(files);
      }
    } catch (error) {
      console.error('加载生成文件失败:', error);
    }
  };

  // 切换标签时加载对应数据
  useEffect(() => {
    if (activeTab === 'generated') {
      loadGeneratedFiles();
    } else if (activeTab === 'weights') {
      loadServerWeights();
    }
  }, [activeTab]);

  // 切换节点展开状态
  const toggleNodeExpand = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  // --- CODE GENERATION ACTIONS ---

  /**
   * 生成PyTorch代码
   */
  const handleGenerateCode = async () => {
    // 验证是否有节点
    if (nodes.length === 0) {
      showNotification("请先添加节点构建模型", "error");
      return;
    }

    setIsGeneratingCode(true);
    setCodeGenerationError(null);

    try {
      // 准备图数据
      const graphData = {
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          data: n.data
        })),
        connections: connections
      };

      // 调用API生成代码
      const result = await modelsAPI.generatePyTorchCode(graphData, modelName.replace(/\s+/g, '_'));

      // 只要成功返回代码，就视为生成成功
      if (result.code) {
        setGeneratedCode(result.code);
        // 解析层类型分布并添加到元数据
        const metadata = parseMetadataFromCode(result.code);
        metadata.filename = '';
        metadata.validation_passed = result.validation?.valid !== false;
        setCodeMetadata(metadata);
        setCodePreviewSource('builder');
        setCurrentPreviewFilename(null);
        setShowCodePreview(true);

        // 根据验证状态显示不同的提示
        if (result.validation && !result.validation.valid && result.validation.errors?.length > 0) {
          // 有验证错误，但代码已生成
          setCodeGenerationError("代码已生成，但验证发现潜在问题: " + result.validation.errors.join(', '));
          showNotification("代码已生成（有警告）", "info");
        } else {
          // 完全成功
          setCodeGenerationError("");
          showNotification("代码生成成功！", "success");
        }
      } else {
        showNotification("代码生成失败", "error");
      }
    } catch (error: any) {
      console.error('代码生成失败:', error);

      // 解析详细的错误信息
      let detailMsg = '';
      let errorDetails: string[] = [];
      let warnings: string[] = [];

      if (error.detail) {
        if (typeof error.detail === 'string') {
          detailMsg = error.detail;
        } else if (typeof error.detail === 'object') {
          const detail = error.detail;
          detailMsg = detail.message || '代码生成失败';
          errorDetails = detail.errors || [];
          warnings = detail.warnings || [];
        }
      } else if (error.message) {
        detailMsg = error.message;
      }

      // 显示详细错误弹窗
      setErrorDialog({
        show: true,
        title: '代码生成失败',
        message: detailMsg || '代码生成过程中发生错误',
        errors: errorDetails,
        warnings: warnings
      });

      setCodeGenerationError(detailMsg);
      showNotification("代码生成失败，请查看详细信息", "error");
    } finally {
      setIsGeneratingCode(false);
    }
  };

  /**
   * 下载生成的代码
   */
  const handleDownloadCode = () => {
    if (!generatedCode) return;

    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.replace(/\s+/g, '_')}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification("代码已下载", "success");
  };

  /**
   * 复制代码到剪贴板
   */
  const handleCopyCode = async () => {
    if (!generatedCode) return;

    try {
      await navigator.clipboard.writeText(generatedCode);
      showNotification("代码已复制到剪贴板", "success");
    } catch (error) {
      showNotification("复制失败，请手动复制", "error");
    }
  };

  // --- 生成文件操作 ---
  // PyTorch层类型到类别的映射
  const LAYER_TYPE_TO_CATEGORY: Record<string, string> = {
    // 卷积层
    'Conv1d': 'Conv',
    'Conv2d': 'Conv',
    'Conv3d': 'Conv',
    'ConvTranspose1d': 'Conv',
    'ConvTranspose2d': 'Conv',
    'ConvTranspose3d': 'Conv',
    // 池化层
    'MaxPool1d': 'Pool',
    'MaxPool2d': 'Pool',
    'MaxPool3d': 'Pool',
    'AvgPool1d': 'Pool',
    'AvgPool2d': 'Pool',
    'AvgPool3d': 'Pool',
    'AdaptiveAvgPool1d': 'Pool',
    'AdaptiveAvgPool2d': 'Pool',
    'AdaptiveAvgPool3d': 'Pool',
    // 归一化层
    'BatchNorm1d': 'Norm',
    'BatchNorm2d': 'Norm',
    'BatchNorm3d': 'Norm',
    'GroupNorm': 'Norm',
    'InstanceNorm1d': 'Norm',
    'InstanceNorm2d': 'Norm',
    'LayerNorm': 'Norm',
    'Dropout': 'Dropout',
    'DropPath': 'Dropout',
    'Dropout2d': 'Dropout',
    'Dropout3d': 'Dropout',
    // 激活层
    'ReLU': 'Activation',
    'ReLU6': 'Activation',
    'LeakyReLU': 'Activation',
    'PReLU': 'Activation',
    'RReLU': 'Activation',
    'Sigmoid': 'Activation',
    'Tanh': 'Activation',
    'GELU': 'Activation',
    'SiLU': 'Activation',
    'Mish': 'Activation',
    'Softmax': 'Activation',
    'LogSoftmax': 'Activation',
    // 线性层
    'Linear': 'Linear',
    'Flatten': 'Transform',
    'View': 'Transform',
    'Reshape': 'Transform',
    'Transpose': 'Transform',
    'Permute': 'Transform',
    'Cat': 'Transform',
    // 注意力层
    'MultiheadAttention': 'Attention',
    // 其他
    'Sequential': 'Container',
    'ModuleList': 'Container',
    'ModuleDict': 'Container',
  };

  // 从代码中解析模型元数据
  const parseMetadataFromCode = (code: string) => {
    const metadata: any = {
      filename: '',
      layer_count: 0,
      depth: 0,
      num_parameters: 0,
      validation_passed: true,
      layer_types: {} as Record<string, number>  // 层类型分布
    };

    try {
      // 尝试从代码中提取 MODEL_INFO
      const modelInfoMatch = code.match(/MODEL_INFO\s*=\s*\{([^}]+)\}/s);
      if (modelInfoMatch) {
        const infoStr = modelInfoMatch[1];
        const layerCountMatch = infoStr.match(/"layer_count":\s*(\d+)/);
        const numParamsMatch = infoStr.match(/"num_parameters":\s*(\d+)/);
        if (layerCountMatch) metadata.layer_count = parseInt(layerCountMatch[1]) || 0;
        if (numParamsMatch) metadata.num_parameters = parseInt(numParamsMatch[1]) || 0;
      }

      // 计算深度和层类型分布（查找 nn. 开头的行）
      const layerMatches = code.match(/self\.\w+\s*=\s*nn\.(\w+)/g);
      if (layerMatches) {
        metadata.layer_count = Math.max(metadata.layer_count, layerMatches.length);
        metadata.depth = layerMatches.length;

        // 解析每层的类型并归类
        layerMatches.forEach(line => {
          const match = line.match(/nn\.(\w+)/);
          if (match) {
            const layerType = match[1];
            const category = LAYER_TYPE_TO_CATEGORY[layerType] || 'Other';
            metadata.layer_types[category] = (metadata.layer_types[category] || 0) + 1;
          }
        });
      }

      // 尝试提取参数量（从注释中）
      const paramCommentMatch = code.match(/# 参数数量[：:]\s*([\d,]+)/);
      if (paramCommentMatch) {
        metadata.num_parameters = parseInt(paramCommentMatch[1].replace(/,/g, '')) || metadata.num_parameters;
      }
    } catch (e) {
      console.error('解析元数据失败:', e);
    }

    return metadata;
  };

  const handlePreviewFile = async (fileId: number, filename: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/generated-files/${fileId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setGeneratedCode(data.content);
        // 从代码中解析元数据
        const metadata = parseMetadataFromCode(data.content);
        metadata.filename = filename;
        setCodeMetadata(metadata);
        setCurrentPreviewFilename(filename);
        setCurrentPreviewFileId(fileId);
        setCodePreviewSource('library');
        setShowCodePreview(true);
      }
    } catch (error) {
      showNotification("预览文件失败", "error");
    }
  };

  const handleDownloadFile = async (fileId: number, filename: string) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/generated-files/${fileId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([data.content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showNotification("文件已下载", "success");
      }
    } catch (error) {
      showNotification("下载文件失败", "error");
    }
  };

  /**
   * 保存当前预览的代码到库
   */
  const handleSaveToLibrary = async () => {
    if (!generatedCode) return;

    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const safeName = modelName.replace(/\s+/g, '_').replace(/[^\w\-]/g, '_');
      const filename = `${safeName}_${timestamp}.py`;

      // 调用后端保存接口（复用生成接口的保存逻辑）
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/save-code`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({
          code: generatedCode,
          filename: filename,
          model_name: modelName
        }),
      });

      if (response.ok) {
        showNotification("代码已保存到库", "success");
        // 刷新文件列表
        if (activeTab === 'generated') {
          loadGeneratedFiles();
        }
      } else {
        throw new Error('保存失败');
      }
    } catch (error) {
      console.error('保存代码失败:', error);
      showNotification("保存到库失败", "error");
    }
  };

  const handleDeleteFile = async (fileId: number) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/generated-files/${fileId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
      });
      if (response.ok) {
        loadGeneratedFiles();
        showNotification("文件已删除", "success");
      }
    } catch (error) {
      showNotification("删除文件失败", "error");
    }
  };

  /**
   * 删除当前预览的文件并关闭预览
   */
  const handleDeleteCurrentPreview = async () => {
    if (!currentPreviewFileId) return;
    await handleDeleteFile(currentPreviewFileId);
    setShowCodePreview(false);
    setCurrentPreviewFilename(null);
    setCurrentPreviewFileId(null);
  };

  // --- ACTIONS ---

  const handleDeleteModel = (id: string, arch_id: number | undefined, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Open Dialog instead of window.confirm
    setDialog({
        isOpen: true,
        type: 'confirm',
        title: '删除模型',
        message: '确定要删除该模型吗？此操作不可恢复。',
        action: 'delete_model',
        data: { id, arch_id }  // 传递 id 和 arch_id
    });
  };

  const handleDeleteWeight = (id: string) => {
      setDialog({
          isOpen: true,
          type: 'confirm',
          title: '删除权重',
          message: '确定要删除该权重文件吗？这将影响使用此权重的推理服务。',
          action: 'delete_weight',
          data: id
      });
  }

  // --- ALGORITHM: Auto Layout ---
  const computeAutoLayout = (currentNodes: VisualNode[], currentConnections: Connection[]) => {
    if (currentNodes.length === 0) return currentNodes;
    
    // Deep clone to avoid mutating state directly during calculation
    const layoutNodes = JSON.parse(JSON.stringify(currentNodes));
    
    const adj: Record<string, string[]> = {};
    const parents: Record<string, string[]> = {};
    layoutNodes.forEach((n: VisualNode) => { adj[n.id] = []; parents[n.id] = []; });
    currentConnections.forEach(c => {
      if (adj[c.source] && parents[c.target]) {
        adj[c.source].push(c.target);
        parents[c.target].push(c.source);
      }
    });

    // 1. Level Calculation
    const nodeLevels: Record<string, number> = {};
    layoutNodes.forEach((n: VisualNode) => { if (parents[n.id].length === 0) nodeLevels[n.id] = 0; else nodeLevels[n.id] = -1; });

    let changed = true;
    let iterations = 0;
    while(changed && iterations < layoutNodes.length + 2) {
       changed = false;
       iterations++;
       layoutNodes.forEach((n: VisualNode) => {
          if (parents[n.id].length > 0) {
             let maxParentLevel = -1;
             let allParentsVisited = true;
             parents[n.id].forEach(pid => {
                if (nodeLevels[pid] === -1) allParentsVisited = false;
                if (nodeLevels[pid] > maxParentLevel) maxParentLevel = nodeLevels[pid];
             });
             if (allParentsVisited) {
                const newLevel = maxParentLevel + 1;
                if (nodeLevels[n.id] !== newLevel) { nodeLevels[n.id] = newLevel; changed = true; }
             }
          }
       });
    }

    layoutNodes.forEach((n: VisualNode) => { if(nodeLevels[n.id] === -1) nodeLevels[n.id] = 0; });
    const levelGroups: Record<number, string[]> = {};
    Object.entries(nodeLevels).forEach(([nid, lvl]) => { if (!levelGroups[lvl]) levelGroups[lvl] = []; levelGroups[lvl].push(nid as string); });

    const HORIZONTAL_SPACING = NODE_WIDTH + 40; 
    const VERTICAL_SPACING = 140;
    
    // 2. Initial Placement
    layoutNodes.forEach((n: VisualNode) => {
       const lvl = nodeLevels[n.id];
       const rowNodes = levelGroups[lvl];
       const index = rowNodes.indexOf(n.id);
       const rowWidth = rowNodes.length * HORIZONTAL_SPACING;
       const xOffset = (index * HORIZONTAL_SPACING) - (rowWidth / 2) + (HORIZONTAL_SPACING / 2);
       n.x = xOffset - (NODE_WIDTH / 2);
       n.y = lvl * VERTICAL_SPACING;
    });

    // 3. Collision Resolution for Skip Connections
    const skipEdges = currentConnections.filter(c => {
        const s = layoutNodes.find((n: VisualNode) => n.id === c.source);
        const t = layoutNodes.find((n: VisualNode) => n.id === c.target);
        if (!s || !t) return false;
        const sLvl = nodeLevels[s.id];
        const tLvl = nodeLevels[t.id];
        return (tLvl > sLvl + 1); 
    });

    skipEdges.forEach(edge => {
        const s = layoutNodes.find((n: VisualNode) => n.id === edge.source)!;
        const t = layoutNodes.find((n: VisualNode) => n.id === edge.target)!;
        const startLvl = nodeLevels[s.id];
        const endLvl = nodeLevels[t.id];
        const centerX = (s.x + t.x) / 2;

        for (let l = startLvl + 1; l < endLvl; l++) {
            layoutNodes.forEach((n: VisualNode) => {
                if (nodeLevels[n.id] === l) {
                   if (Math.abs(n.x - centerX) < NODE_WIDTH) {
                       n.x += NODE_WIDTH; 
                   }
                }
            });
        }
    });

    return layoutNodes;
  };

  const handleAutoLayoutButton = () => {
    if (nodes.length === 0) return;
    const currentCanvasWidth = canvasRef.current?.clientWidth || 1200;
    const CENTER_X = currentCanvasWidth / 2;
    
    const layouted = computeAutoLayout(nodes, connections);
    
    // Shift to center of screen
    const centered = layouted.map((n: VisualNode) => ({
      ...n,
      x: n.x + CENTER_X,
      y: n.y + 50
    }));

    setNodes(centered);
    setPan({ x: 0, y: 0 });
    setScale(1);
    showNotification("计算图已自动整理", "success");
  };

  // --- SAVE AS OPERATOR (Button Click) ---
  const handleSaveAsBlockClick = () => {
    if (nodes.length === 0) { 
        showNotification("画布为空，无法保存为算子", "error"); 
        return; 
    }
    // Open Dialog
    setDialog({
        isOpen: true,
        type: 'prompt',
        title: '保存为自定义算子',
        message: '请输入算子名称，保存后可重复使用。',
        defaultValue: modelName + " Block",
        action: 'save_operator'
    });
  };

  // --- TRASH CAN CLEAR LOGIC (Button Click) ---
  const handleClearCanvasClick = () => {
    if (nodes.length === 0) return;
    // Open Dialog
    setDialog({
        isOpen: true,
        type: 'confirm',
        title: '清空画布',
        message: '确定要清空当前画布吗？所有未保存的更改将丢失。',
        action: 'clear_canvas'
    });
  };

  // --- UNIFIED DIALOG CONFIRM HANDLER ---
  const handleDialogConfirm = async (val?: string) => {
      // 1. DELETE MODEL
      if (dialog.action === 'delete_model' && dialog.data) {
          const { id, arch_id } = dialog.data;
          // 如果有 arch_id，从服务器删除
          if (arch_id) {
              try {
                  const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/architectures/${arch_id}`, {
                      method: 'DELETE',
                      headers: {
                          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
                      },
                  });
                  if (response.ok) {
                      showNotification("模型已删除", "success");
                      await loadServerArchitectures();  // 刷新列表
                  } else {
                      const error = await response.json();
                      showNotification(`删除失败: ${error.detail || '未知错误'}`, 'error');
                  }
              } catch (error) {
                  console.error('删除模型失败:', error);
                  showNotification('删除失败，请检查网络连接', 'error');
              }
          } else {
              // 默认模型，只从前端状态移除
              setModels(prev => prev.filter(m => m.id !== id));
              showNotification("模型已删除", "success");
          }
      }
      // 2. CLEAR CANVAS
      else if (dialog.action === 'clear_canvas') {
          setNodes([]);
          setConnections([]);
          showNotification("画布已清空", "success");
      }
      // 3. SAVE OPERATOR
      else if (dialog.action === 'save_operator' && val) {
          const name = val.trim();
          if (!name) return; // Should allow re-entry or show error? For now just close or keep open.
          
          if (customTemplates.some(t => t.name === name) || ATOMIC_NODES[name]) {
             showNotification("算子名称已存在", "error"); 
             return;
          }

          // Auto-Format
          const layoutedNodes = computeAutoLayout(nodes, connections);

          // ID Remapping
          const idMap: Record<string, string> = {};
          layoutedNodes.forEach((n: VisualNode, idx: number) => {
              idMap[n.id] = `tpl_${idx}`;
          });

          // Normalization
          const xValues = layoutedNodes.map((n: VisualNode) => n.x);
          const yValues = layoutedNodes.map((n: VisualNode) => n.y);
          const minX = Math.min(...xValues);
          const maxX = Math.max(...xValues);
          const minY = Math.min(...yValues);
          const maxY = Math.max(...yValues);

          const centerX = (minX + maxX + NODE_WIDTH) / 2;
          const centerY = (minY + maxY) / 2;

          // Construct
          const templateNodes = layoutedNodes.map((n: VisualNode) => ({
              ...n,
              id: idMap[n.id],
              x: n.x - centerX + (NODE_WIDTH/2),
              y: n.y - centerY, 
              data: JSON.parse(JSON.stringify(n.data))
          }));

          const templateConnections = connections.map((c, idx) => ({
              id: `c_tpl_${idx}`,
              source: idMap[c.source],
              target: idMap[c.target]
          }));

          const newTemplate: BlockTemplate = {
              name: name,
              description: `Custom block with ${nodes.length} nodes.`,
              nodes: templateNodes,
              connections: templateConnections
          };

          setCustomTemplates(prev => [...prev, newTemplate]);
          
          // Visual Feedback
          const currentCanvasWidth = canvasRef.current?.clientWidth || 1200;
          const visualNodes = layoutedNodes.map((n: VisualNode) => ({
              ...n,
              x: n.x + (currentCanvasWidth / 2),
              y: n.y + 50
          }));
          setNodes(visualNodes);
          setPan({ x: 0, y: 0 });
          setScale(1);

          showNotification(`已保存算子: ${name}`, "success");
      }
      // 4. DELETE WEIGHT
      else if (dialog.action === 'delete_weight' && dialog.data) {
          // Delete weight from server via API
          const weightId = dialog.data as number;
          fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/weights/${weightId}`, {
              method: 'DELETE',
              headers: {
                  'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
              },
          }).then(async (response) => {
              if (response.ok) {
                  // 更新所有相关状态
                  setWeights(prev => prev.filter(w => w.id !== weightId));
                  setRootWeights(prev => prev.filter(w => w.id !== weightId));
                  setWeightTree(prev => {
                      // 递归删除指定ID的节点
                      const removeFromTree = (nodes: WeightTreeItem[]): WeightTreeItem[] => {
                          return nodes.filter(node => {
                              if (node.id === weightId) return false;
                              if (node.children && node.children.length > 0) {
                                  node.children = removeFromTree(node.children);
                              }
                              return true;
                          });
                      };
                      return removeFromTree(prev);
                  });
                  // 如果删除的是当前选中的权重，清空选中状态
                  if (selectedWeight?.id === weightId) {
                      setSelectedWeight(null);
                  }
                  showNotification("权重文件已删除", "success");
              } else {
                  const error = await response.json();
                  showNotification(`删除失败: ${error.detail || '未知错误'}`, 'error');
              }
          }).catch(() => {
              showNotification("删除权重失败", 'error');
          });
      }
      // 5. DELETE GENERATED FILE
      else if (dialog.action === 'delete_file' && dialog.data) {
          const fileId = dialog.data as number;
          handleDeleteFile(fileId);
      }

      setDialog({ ...dialog, isOpen: false });
  };

  const handleCreateNew = () => {
    setActiveModelId(null); setOriginalFilename(null); setModelName("Untitled Architecture");
    setNodes([]); setConnections([]); setScale(1); setPan({ x: 0, y: 0 }); setArchView('builder');
  };

  const handleEditModel = async (model: ModelData) => {
    // 保存原始arch_id（用于保存时更新）
    setOriginalFilename(model.arch_id ? String(model.arch_id) : null);

    // 如果是服务器模型，从服务器获取完整数据
    if (model.arch_id) {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/architectures/${model.arch_id}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setActiveModelId(model.id);
          setModelName(data.name || model.name);
          setNodes(data.nodes || []);
          setConnections(data.connections || []);
          setScale(1);
          setPan({ x: 0, y: 0 });
          setArchView('builder');
          return;
        }
      } catch (error) {
        console.error('加载架构详情失败:', error);
      }
    }

    // 降级：使用本地数据
    setActiveModelId(model.id); setModelName(model.name);
    setNodes(JSON.parse(JSON.stringify(model.nodes || [])));
    setConnections(JSON.parse(JSON.stringify(model.connections || [])));
    setScale(1); setPan({ x: 0, y: 0 }); setArchView('builder');
  };

  const handleSave = async (asNew: boolean = false) => {
    if (!modelName.trim()) {
      showNotification("模型名称不能为空", 'error');
      setIsEditingName(true);
      return;
    }

    try {
      // 另存为时检查重名
      if (asNew) {
        const hasDuplicate = models.some(m => m.name === modelName);
        if (hasDuplicate) {
          showNotification(`已存在名为 "${modelName}" 的模型`, 'error');
          return;
        }
      } else {
        // 普通保存：也需要检查重名（排除当前正在编辑的模型）
        const hasDuplicate = models.some(m => m.name === modelName && m.id !== activeModelId);
        if (hasDuplicate) {
          showNotification(`已存在名为 "${modelName}" 的模型`, 'error');
          return;
        }
      }

      const architecture = {
        name: modelName,
        description: '',
        version: 'v1.0',
        type: 'Custom',
        nodes: nodes,
        connections: connections
      };

      // 构建请求URL：另存为时不指定目标ID，后端根据name生成新记录
      // 普通保存且有arch_id时，指定target_id更新原记录
      let url = `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'}/models/architectures?overwrite=true`;
      if (!asNew && originalFilename) {
        // originalFilename现在存储的是arch_id的字符串形式
        url += `&target_id=${encodeURIComponent(originalFilename)}`;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify(architecture)
      });

      if (response.ok) {
        const result = await response.json();
        const message = result.updated ? '模型已更新' : `已保存: ${result.filename}`;
        showNotification(message, 'success');

        // 更新originalFilename（后续保存会更新这个记录）
        if (!asNew && !originalFilename) {
          // 新建模型首次保存，保存返回的ID
          setOriginalFilename(String(result.id));
        }

        // 刷新服务器架构列表
        await loadServerArchitectures();
      } else {
        const error = await response.json();
        showNotification(`保存失败: ${error.detail || '未知错误'}`, 'error');
      }
    } catch (error: any) {
      console.error('保存模型失败:', error);
      showNotification('保存失败，请检查网络连接', 'error');
    }
  };

  // 导出架构为 JSON 文件
  const handleExportJSON = () => {
    if (!modelName.trim()) {
      showNotification("请先输入模型名称", 'error');
      return;
    }

    const data = {
      name: modelName,
      version: 'v1.0',
      type: 'Custom',
      description: '',
      nodes: nodes,
      connections: connections,
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${modelName.replace(/\s+/g, '_')}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('架构已导出', 'success');
  };

  // 导入架构 JSON 文件
  const handleImportJSON = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e: any) => {
      const file = e.target.files[0];
      if (!file) return;

      try {
        const text = await file.text();
        const data = JSON.parse(text);

        // 验证数据结构
        if (!data.nodes || !Array.isArray(data.nodes)) {
          showNotification('无效的架构文件：缺少 nodes 数据', 'error');
          return;
        }

        // 加载架构
        setModelName(data.name || file.name.replace('.json', ''));
        setNodes(data.nodes);
        setConnections(data.connections || []);
        setScale(1);
        setPan({ x: 0, y: 0 });

        showNotification(`架构已导入: ${data.name || file.name}`, 'success');
      } catch (error) {
        console.error('导入失败:', error);
        showNotification('导入失败：文件格式错误', 'error');
      }
    };
    input.click();
  };

  const getNodeHeight = (node: VisualNode) => {
    const def = ATOMIC_NODES[node.type] || { params: [] };
    const paramsCount = def.params.length;
    const isExpanded = expandedNodes.has(node.id);
    // 折叠式显示：默认只显示前2个参数，展开后显示全部
    const displayCount = isExpanded ? paramsCount : Math.min(paramsCount, 2);
    // 每个参数行约18px高度（包括间距）
    const paramsHeight = displayCount * 18;
    // 基础高度：头部24px + 内边距12px + 参数区域 + 底部padding
    const baseHeight = 24 + 12 + paramsHeight + 8;
    return Math.max(56, baseHeight);
  };

  const handleWheel = (e: React.WheelEvent) => { e.stopPropagation(); const delta = -e.deltaY * 0.001; setScale(s => Math.min(Math.max(0.2, s + delta), 3)); };
  
  const handleDragStart = (e: React.DragEvent, type: string, isTemplate: boolean) => { 
      setDraggedItem({ type, isTemplate }); 
      e.dataTransfer.effectAllowed = 'copy'; 
  };
  
  // --- DROP HANDLER (Revised) ---
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (!draggedItem || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const dropX = (e.clientX - rect.left - pan.x) / scale;
    const dropY = (e.clientY - rect.top - pan.y) / scale; 

    // CASE 1: ATOMIC NODE
    if (!draggedItem.isTemplate) {
        const type = draggedItem.type;
        const def = ATOMIC_NODES[type] || { label: type, color: 'bg-slate-500', category: 'Unknown', params: [] };
        const initData: any = {};
        if (def.params) def.params.forEach(p => initData[p.name] = p.default);
        
        const newNode: VisualNode = { 
            id: `n_${Date.now()}`, 
            type: type, 
            label: def.label, 
            x: dropX - (NODE_WIDTH / 2), 
            y: dropY - 25, 
            inputs: [], outputs: [], data: initData 
        };
        setNodes(prev => [...prev, newNode]);
        setSelectedNodeId(newNode.id);
    } 
    // CASE 2: TEMPLATE (Custom Operator)
    else {
        const type = draggedItem.type;
        const template = customTemplates.find(t => t.name === type);
        
        if (template) {
            // Generate a unique session ID for this instantiation to avoid ID collisions
            const sessionId = Date.now();
            const idMap: Record<string, string> = {};

            // 1. Instantiate Nodes
            const newNodes = template.nodes.map((n) => {
                // Map the template ID (tpl_0) to a new live ID (n_12345_0)
                const newId = `n_${sessionId}_${n.id}`; 
                idMap[n.id] = newId;

                return {
                    ...n,
                    id: newId,
                    // Apply offset: drop position + stored relative position
                    x: dropX + n.x, 
                    y: dropY + n.y,
                    data: JSON.parse(JSON.stringify(n.data)) // Deep copy data
                };
            });

            // 2. Instantiate Connections
            const newConns = template.connections.map((c, idx) => ({
                id: `c_${sessionId}_${idx}`,
                source: idMap[c.source], // Map source to new ID
                target: idMap[c.target]  // Map target to new ID
            }));

            setNodes(prev => [...prev, ...newNodes]);
            setConnections(prev => [...prev, ...newConns]);
            showNotification(`已实例化自定义算子: ${type}`, 'success');
        } else {
            console.error("Template not found for type:", type);
            showNotification("模板加载失败", "error");
        }
    }
    setDraggedItem(null);
  };

  const handleNodeMouseDown = (e: React.MouseEvent, id: string) => { e.stopPropagation(); setMovingNodeId(id); setSelectedNodeId(id); setSelectedConnectionId(null); };
  const handleCanvasMouseDown = (e: React.MouseEvent) => { setIsPanning(true); setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); setSelectedNodeId(null); setSelectedConnectionId(null); };
  
  const handlePortMouseDown = (e: React.MouseEvent, nodeId: string, type: 'in' | 'out') => {
    e.stopPropagation(); e.preventDefault(); 
    if (type === 'out') {
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      const canvasRect = canvasRef.current!.getBoundingClientRect();
      const startX = (rect.left + rect.width/2 - canvasRect.left - pan.x) / scale;
      const startY = (rect.top + rect.height/2 - canvasRect.top - pan.y) / scale;
      setDrawingConnection({ sourceId: nodeId, startX, startY, currX: startX, currY: startY });
    }
  };
  
  const handlePortMouseUp = (e: React.MouseEvent, targetId: string, type: 'in' | 'out') => {
    e.stopPropagation();
    if (drawingConnection && type === 'in' && drawingConnection.sourceId !== targetId) {
      const newConn = { id: `c_${Date.now()}`, source: drawingConnection.sourceId, target: targetId };
      if (!connections.find(c => c.source === newConn.source && c.target === newConn.target)) {
        setConnections(prev => [...prev, newConn]);
      }
    }
    setDrawingConnection(null);
  };
  
  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (isPanning) { setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y }); return; }
    const x = (e.clientX - rect.left - pan.x) / scale;
    const y = (e.clientY - rect.top - pan.y) / scale;

    if (movingNodeId) {
      setNodes(prev => prev.map(n => n.id === movingNodeId ? { ...n, x: x - (NODE_WIDTH/2), y: y - 25 } : n));
      if (trashRef.current) {
         const tr = trashRef.current.getBoundingClientRect();
         setTrashHover(e.clientX >= tr.left && e.clientX <= tr.right && e.clientY >= tr.top && e.clientY <= tr.bottom);
      }
    } else if (drawingConnection) { setDrawingConnection({ ...drawingConnection, currX: x, currY: y }); }
  };
  
  const handleCanvasMouseUp = () => {
    setIsPanning(false);
    if (movingNodeId && trashHover) {
       setNodes(prev => prev.filter(n => n.id !== movingNodeId));
       setConnections(prev => prev.filter(c => c.source !== movingNodeId && c.target !== movingNodeId));
       setSelectedNodeId(null);
       setTrashHover(false);
    }
    setMovingNodeId(null); setDrawingConnection(null);
  };
  
  const getPath = (sx: number, sy: number, tx: number, ty: number) => `M ${sx} ${sy} C ${sx} ${sy + 50}, ${tx} ${ty - 50}, ${tx} ${ty}`;

  // --- RENDER MAIN LAYOUT ---
  return (
    <div className="h-full flex flex-col relative overflow-hidden">
        
        {/* TOP LEVEL NAVIGATION (Global within module) */}
        <div className="flex justify-center items-center py-4 shrink-0 bg-slate-900/80 backdrop-blur border-b border-slate-800 z-10">
            <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
                <button
                    onClick={() => { setActiveTab('architectures'); if(archView === 'builder') setArchView('list'); }}
                    className={`px-4 py-2 rounded-md text-sm font-medium flex items-center transition-all ${activeTab === 'architectures' ? 'bg-slate-800 text-cyan-400 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <Layout size={16} className="mr-2" /> 架构设计
                </button>
                <button
                    onClick={() => setActiveTab('weights')}
                    className={`px-4 py-2 rounded-md text-sm font-medium flex items-center transition-all ${activeTab === 'weights' ? 'bg-slate-800 text-emerald-400 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <Database size={16} className="mr-2" /> 权重库
                </button>
                <button
                    onClick={() => setActiveTab('generated')}
                    className={`px-4 py-2 rounded-md text-sm font-medium flex items-center transition-all ${activeTab === 'generated' ? 'bg-slate-800 text-amber-400 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <Code size={16} className="mr-2" /> 生成的模型
                </button>
            </div>
        </div>

        {/* Global Notification */}
        {notification && (
            <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[300] px-4 py-2 rounded-lg shadow-lg border flex items-center ${
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

        {/* Global Dialog */}
        <CustomDialog
            isOpen={dialog.isOpen}
            type={dialog.type}
            title={dialog.title}
            message={dialog.message}
            defaultValue={dialog.defaultValue}
            onClose={() => setDialog({ ...dialog, isOpen: false })}
            onConfirm={handleDialogConfirm}
        />

        {/* Error Detail Dialog */}
        <ErrorDetailDialog
            isOpen={errorDialog.show}
            title={errorDialog.title}
            message={errorDialog.message}
            errors={errorDialog.errors}
            warnings={errorDialog.warnings}
            onClose={() => setErrorDialog({ ...errorDialog, show: false })}
        />

        {/* CONTENT AREA SWITCH */}
        <div className="flex-1 overflow-hidden relative">
            
            {/* VIEW 1: ARCHITECTURES (List) */}
            {activeTab === 'architectures' && archView === 'list' && (
                <div className="h-full flex flex-col p-8 pt-4 overflow-y-auto custom-scrollbar">
                    <div className="flex justify-between items-center mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">模型架构</h2>
                            <p className="text-slate-400 text-sm">管理神经网络拓扑结构与计算图。</p>
                        </div>
                        <div className="flex gap-3">
                            <button onClick={handleCreateNew} className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center transition-all">
                                <Plus size={18} className="mr-2" /> 新建架构
                            </button>
                        </div>
                    </div>

                    {/* 预设模型区域 */}
                    {presetModels.length > 0 && (
                        <div className="mb-8">
                            {/* 预设模型标题栏 */}
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={() => setShowPresets(!showPresets)}
                                        className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors"
                                    >
                                        <Package size={18} className={showPresets ? "text-cyan-400" : ""} />
                                        <span className="font-medium">预设模型</span>
                                        <span className="px-2 py-0.5 rounded-full bg-cyan-900/30 text-cyan-400 text-xs">
                                            {presetModels.length}
                                        </span>
                                    </button>
                                    {/* 分类筛选 */}
                                    <div className="flex gap-2 ml-4">
                                        {['all', 'cnn', 'rnn', 'transformer', 'classification', 'detection'].map(cat => (
                                            <button
                                                key={cat}
                                                onClick={() => setSelectedCategory(cat)}
                                                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                                                    selectedCategory === cat
                                                        ? 'bg-cyan-600 text-white'
                                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                                                }`}
                                            >
                                                {cat === 'all' ? '全部' :
                                                 cat === 'cnn' ? 'CNN' :
                                                 cat === 'rnn' ? 'RNN' :
                                                 cat === 'transformer' ? 'Transformer' :
                                                 cat === 'classification' ? '分类' :
                                                 cat === 'detection' ? '检测' : cat}
                                            </button>
                                        ))}
                                    </div>
                                    {/* 难度筛选 */}
                                    <div className="flex gap-2 ml-2">
                                        {['all', 'beginner', 'intermediate', 'advanced'].map(diff => (
                                            <button
                                                key={diff}
                                                onClick={() => setSelectedDifficulty(diff)}
                                                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                                                    selectedDifficulty === diff
                                                        ? 'bg-purple-600 text-white'
                                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                                                }`}
                                            >
                                                {diff === 'all' ? '全部难度' :
                                                 diff === 'beginner' ? '入门' :
                                                 diff === 'intermediate' ? '中级' :
                                                 diff === 'advanced' ? '高级' : diff}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* 预设模型卡片列表 */}
                            {showPresets && (
                                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                                    {getFilteredPresets().map(preset => (
                                        <div
                                            key={preset.id}
                                            onClick={() => handleCreateFromPreset(preset.id, preset.name, preset.description)}
                                            className="glass-panel p-4 rounded-lg border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-800/80 transition-all cursor-pointer group"
                                        >
                                            {/* 图标和难度 */}
                                            <div className="flex items-start justify-between mb-3">
                                                <span className="text-2xl">{getCategoryIcon(preset.category)}</span>
                                                <span className={getDifficultyBadge(preset.difficulty)}>
                                                    {getDifficultyText(preset.difficulty)}
                                                </span>
                                            </div>
                                            {/* 名称 */}
                                            <h4 className="text-sm font-bold text-white mb-1 group-hover:text-cyan-400 transition-colors line-clamp-1" title={preset.name}>
                                                {preset.name}
                                            </h4>
                                            {/* 描述 */}
                                            <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed mb-3" title={preset.description}>
                                                {preset.description}
                                            </p>
                                            {/* 标签 */}
                                            <div className="flex flex-wrap gap-1">
                                                {preset.tags.slice(0, 2).map(tag => (
                                                    <span key={tag} className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
                                                        {tag}
                                                    </span>
                                                ))}
                                                {preset.tags.length > 2 && (
                                                    <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-500">
                                                        +{preset.tags.length - 2}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* 分隔线 */}
                    {presetModels.length > 0 && showPresets && (
                        <div className="border-t border-slate-800 my-6"></div>
                    )}

                    {/* 我的架构标题 */}
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                        <GitBranch size={18} className="text-cyan-400" />
                        我的架构
                        <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 text-xs">
                            {models.length}
                        </span>
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {models.map(model => (
                        <div
                            key={model.id}
                            onClick={() => handleEditModel(model)}
                            className="glass-panel p-0 rounded-xl border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all group flex flex-col cursor-pointer relative overflow-hidden min-h-[220px]"
                        >
                            {/* 删除按钮 */}
                            <button
                                onClick={(e) => handleDeleteModel(model.id, model.arch_id, e)}
                                className="absolute top-3 right-3 p-2 text-slate-500 hover:text-rose-400 bg-slate-900/70 hover:bg-rose-900/30 rounded-lg transition-all z-20 opacity-0 group-hover:opacity-100"
                                title="删除模型"
                            >
                                <Trash2 size={14} />
                            </button>

                            {/* 主体内容区 */}
                            <div className="flex-1 p-5 flex flex-col">
                                {/* 顶部：图标 + 名称 + 描述 */}
                                <div className="flex items-start gap-4 mb-3">
                                    {/* 图标容器 */}
                                    <div className="p-3 rounded-xl border border-slate-700/50 group-hover:scale-110 transition-transform shrink-0 bg-gradient-to-br from-cyan-900/30 to-blue-900/30 text-cyan-400">
                                        <GitBranch size={24} />
                                    </div>

                                    {/* 名称和描述 */}
                                    <div className="flex-1 min-w-0">
                                        <h3 className="text-base font-bold text-white mb-1 group-hover:text-cyan-400 transition-colors line-clamp-1" title={model.name}>
                                            {model.name}
                                        </h3>
                                        <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed" title={model.description || '暂无描述'}>
                                            {model.description || '暂无描述信息'}
                                        </p>
                                    </div>
                                </div>

                                {/* 标签区 */}
                                <div className="flex flex-wrap gap-2 mb-4">
                                    <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-purple-900/30 text-purple-400 border border-purple-500/20">
                                        {model.type}
                                    </span>
                                    {getModelTags(model)}
                                </div>

                                {/* 底部统计信息 */}
                                <div className="mt-auto pt-3 border-t border-slate-800/50">
                                    <div className="grid grid-cols-4 gap-2 text-center">
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-slate-600">层数</span>
                                            <span className="text-sm font-semibold text-cyan-400">{model.node_count || model.nodes?.length || 0}</span>
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-slate-600">连接</span>
                                            <span className="text-sm font-semibold text-purple-400">{model.connection_count || model.connections?.length || 0}</span>
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-slate-600">版本</span>
                                            <span className="text-sm font-semibold text-slate-300">{model.version}</span>
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-slate-600">更新</span>
                                            <span className="text-xs text-slate-400 truncate" title={model.updated}>{formatDate(model.updated)}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* 底部渐变条 */}
                            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
                        </div>
                        ))}
                        <div onClick={handleCreateNew} className="p-5 rounded-xl border-2 border-dashed border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/40 cursor-pointer transition-all flex flex-col items-center justify-center text-slate-500 hover:text-cyan-400 group min-h-[200px]">
                            <Plus size={48} className="mb-4 group-hover:scale-110 transition-transform" />
                            <span className="font-medium">新建架构</span>
                        </div>
                    </div>
                </div>
            )}

            {/* VIEW 2: ARCHITECTURE BUILDER (Canvas) */}
            {activeTab === 'architectures' && archView === 'builder' && (
                <div className="absolute inset-0 flex flex-col">
                    {/* Top Bar (Canvas specific) */}
                    <div className="h-14 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between px-6 shrink-0 z-20">
                        <div className="flex items-center space-x-4">
                        <button onClick={() => { setArchView('list'); loadServerArchitectures(); }} className="text-slate-500 hover:text-white"><ArrowRight size={20} className="rotate-180" /></button>
                        <div className="h-6 w-px bg-slate-800"></div>
                        <div className="flex items-center group">
                            <Layers className="mr-2 text-cyan-400" size={20} /> 
                            {isEditingName ? (
                            <input 
                                autoFocus 
                                value={modelName} 
                                onChange={(e) => setModelName(e.target.value)} 
                                onBlur={() => setIsEditingName(false)} 
                                onKeyDown={(e) => e.key === 'Enter' && setIsEditingName(false)} 
                                className="bg-slate-800 text-lg font-bold text-white px-2 py-0.5 rounded border border-cyan-500 outline-none w-64"
                                />
                            ) : (
                            <div className="flex items-center cursor-pointer" onClick={() => setIsEditingName(true)}>
                                <h2 className="text-lg font-bold text-white mr-2 hover:text-cyan-200 transition-colors">{modelName}</h2>
                                <Edit3 size={14} className="text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                            )}
                        </div>
                        </div>
                        <div className="flex space-x-2">
                            <button onClick={handleSaveAsBlockClick} className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-indigo-600/20 hover:text-indigo-400 hover:border-indigo-500/50 text-slate-300 text-xs font-bold rounded border border-slate-600 transition-all"><Package size={14} className="mr-2" /> 存为算子</button>
                            <button onClick={() => handleSave(true)} className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all"><Copy size={14} className="mr-2" /> 另存为</button>
                            <button onClick={() => handleSave(false)} className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-emerald-600/20 hover:text-emerald-400 hover:border-emerald-500/50 text-slate-300 text-xs font-bold rounded border border-slate-600 transition-all"><Save size={14} className="mr-2" /> 保存</button>
                            <div className="w-px h-6 bg-slate-700 mx-1"></div>
                            <button onClick={handleExportJSON} className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all" title="导出为JSON文件"><Download size={14} className="mr-1" /></button>
                            <button onClick={handleImportJSON} className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all" title="导入JSON文件"><Upload size={14} className="mr-1" /></button>
                            <div className="w-px h-6 bg-slate-700 mx-1"></div>
                            <button
                                onClick={handleGenerateCode}
                                disabled={isGeneratingCode || nodes.length === 0}
                                className={`flex items-center px-3 py-1.5 text-xs font-bold rounded shadow-lg transition-all ${
                                    isGeneratingCode || nodes.length === 0
                                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                        : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20'
                                }`}
                            >
                                {isGeneratingCode ? (
                                    <>
                                        <Loader2 size={14} className="mr-2 animate-spin" />
                                        生成中...
                                    </>
                                ) : (
                                    <>
                                        <Code size={14} className="mr-2" />
                                        预览代码
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    <div className="flex-1 flex overflow-hidden relative">
                        {/* Sidebar, Canvas, Right Panel Logic ... same as before but inside this container */}
                        <div className={`bg-slate-950 border-r border-slate-800 flex flex-col z-40 transition-all duration-300 ease-in-out absolute top-0 bottom-0 left-0 h-full shadow-2xl select-none ${showLeftPanel ? 'w-64 translate-x-0' : 'w-64 -translate-x-full'}`}>
                            <div className="p-4 border-b border-slate-800 min-w-[16rem]"><h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">算子库 (Operators)</h3></div>
                            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar min-w-[16rem] space-y-8">
                                <div>
                                    <div className="flex items-center mb-3 text-cyan-500"><Box size={14} className="mr-2" /><h3 className="text-xs font-bold uppercase tracking-wide">PyTorch 原子算子</h3></div>
                                    <div className="space-y-4 pl-2 border-l border-slate-800 ml-1.5">
                                        {['IO', 'Layer', 'Activation', 'Pooling', 'Ops'].map(cat => (
                                            <div key={cat}>
                                                <h4 className="text-[10px] font-bold text-slate-600 uppercase mb-2">{cat}</h4>
                                                <div className="space-y-1">
                                                {Object.entries(ATOMIC_NODES).filter(([_, def]) => def.category === cat).map(([type, _]) => (
                                                    <div key={type} draggable onDragStart={(e) => handleDragStart(e, type, false)} className="px-2 py-1.5 bg-slate-900 border border-slate-800 rounded hover:border-cyan-500/50 cursor-grab active:cursor-grabbing text-xs text-slate-300 hover:text-white select-none transition-colors">{type}</div>
                                                ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div>
                                    <div className="flex items-center mb-3 text-indigo-400"><Package size={14} className="mr-2" /><h3 className="text-xs font-bold uppercase tracking-wide">自定义算子</h3></div>
                                    <div className="pl-2 border-l border-slate-800 ml-1.5">
                                        <div className="space-y-1">
                                            {customTemplates.map(t => (
                                                <div key={t.name} draggable onDragStart={(e) => handleDragStart(e, t.name, true)} className="px-2 py-1.5 bg-slate-900 border border-slate-800 rounded hover:border-indigo-500/50 cursor-grab active:cursor-grabbing text-xs text-indigo-300 hover:text-white select-none transition-colors group flex justify-between items-center">
                                                    <span>{t.name}</span>
                                                    <button onClick={(e) => { e.stopPropagation(); setCustomTemplates(prev => prev.filter(c => c.name !== t.name)); }} className="opacity-0 group-hover:opacity-100 hover:text-rose-400"><X size={10} /></button>
                                                </div>
                                            ))}
                                            {customTemplates.length === 0 && <div className="text-[10px] text-slate-600 italic px-2">暂无自定义算子</div>}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <button onClick={() => setShowLeftPanel(!showLeftPanel)} className={`absolute top-1/2 z-50 p-1 bg-slate-800 border border-slate-700 text-slate-400 hover:text-white rounded-r transition-all duration-300 ease-in-out ${showLeftPanel ? 'left-64' : 'left-0'}`} style={{ transform: 'translateY(-50%)' }}><SidebarIcon size={14} /></button>

                        <div 
                            className={`flex-1 bg-slate-950 relative overflow-hidden bg-grid-pattern bg-[length:20px_20px] transition-all duration-300 ease-in-out ${showLeftPanel ? 'ml-64' : 'ml-0'} ${showRightPanel ? 'mr-72' : 'mr-0'} ${isPanning ? 'cursor-grabbing' : 'cursor-default'}`}
                            ref={canvasRef} onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} onWheel={handleWheel} onMouseDown={handleCanvasMouseDown} onMouseMove={handleCanvasMouseMove} onMouseUp={handleCanvasMouseUp} onMouseLeave={handleCanvasMouseUp}
                        >
                            <div className="absolute top-4 left-8 z-30 flex space-x-2 select-none">
                                <button onClick={handleAutoLayoutButton} className="p-1.5 bg-slate-900/80 backdrop-blur rounded border border-slate-700 text-slate-300 hover:text-cyan-400 transition-colors" title="自动整理"><Layout size={16} /></button>
                                <div className="w-px h-8 bg-slate-700 mx-1"></div>
                                <button onClick={() => setScale(s => Math.min(s + 0.1, 2))} className="p-1.5 bg-slate-900/80 backdrop-blur rounded border border-slate-700 text-slate-300"><ZoomIn size={16} /></button>
                                <button onClick={() => { setScale(1); setPan({x:0, y:0}); }} className="px-2 bg-slate-900/80 backdrop-blur rounded border border-slate-700 text-xs text-slate-300 font-mono">{Math.round(scale * 100)}%</button>
                                <button onClick={() => setScale(s => Math.max(s - 0.1, 0.5))} className="p-1.5 bg-slate-900/80 backdrop-blur rounded border border-slate-700 text-slate-300"><ZoomOut size={16} /></button>
                            </div>
                            <div 
                                ref={trashRef} 
                                onClick={(e) => { e.stopPropagation(); handleClearCanvasClick(); }}
                                title="点击清空画布 / 拖拽节点至此删除"
                                className={`absolute bottom-8 left-8 z-50 w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-300 cursor-pointer ${trashHover ? 'bg-rose-900/50 border-rose-500 scale-110' : 'bg-slate-900/80 border-slate-700 text-slate-500 hover:bg-slate-800 hover:text-rose-400 hover:border-rose-900'}`}
                                style={{ pointerEvents: 'auto' }}
                            >
                                <Trash2 size={24} className={trashHover ? 'text-rose-400' : ''} />
                            </div>
                            
                            {/* Validation Panel */}
                            <div className="absolute bottom-8 right-8 z-30 p-4 glass-panel rounded-lg border border-slate-800 w-64 shadow-2xl select-none">
                                <div className="text-xs text-slate-500 mb-2 uppercase tracking-wide">计算图验证</div>
                                {nodes.length === 0 ? <div className="flex items-center text-slate-500 text-sm font-bold"><Info size={16} className="mr-2" /> 空画布</div> : connections.length >= nodes.length - 1 ? <div className="flex items-center text-emerald-400 text-sm font-bold"><CheckCircle size={16} className="mr-2" /> 结构合法</div> : <div className="flex items-center text-amber-400 text-sm font-bold"><AlertTriangle size={16} className="mr-2" /> 检测到孤立节点</div>}
                            </div>

                            <div style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`, transformOrigin: '0 0', width: '100%', height: '100%', transition: isPanning ? 'none' : 'transform 0.1s ease-out' }} className="relative">
                                <style>{`@keyframes flow-animation { from { stroke-dashoffset: 24; } to { stroke-dashoffset: 0; } } .animate-flow { animation: flow-animation 0.5s linear infinite; }`}</style>
                                <svg className="absolute top-0 left-0 w-full h-full pointer-events-none z-0" style={{ overflow: 'visible' }}>
                                    {connections.map(conn => {
                                        const source = nodes.find(n => n.id === conn.source);
                                        const target = nodes.find(n => n.id === conn.target);
                                        if (!source || !target) return null;
                                        const sourceHeight = getNodeHeight(source);
                                        const sx = source.x + NODE_WIDTH / 2; const sy = source.y + sourceHeight; const tx = target.x + NODE_WIDTH / 2; const ty = target.y;
                                        const pathData = getPath(sx, sy, tx, ty);
                                        const isSelected = selectedConnectionId === conn.id;
                                        const midX = (sx + tx) / 2; const midY = (sy + ty) / 2;
                                        return (
                                        <g key={conn.id} className="pointer-events-auto">
                                            <path d={pathData} stroke="transparent" strokeWidth="20" fill="none" className="cursor-pointer hover:stroke-white/10 transition-colors" onClick={(e) => { e.stopPropagation(); setSelectedConnectionId(conn.id); setSelectedNodeId(null); }} />
                                            <path d={pathData} stroke={isSelected ? "#22d3ee" : "#94a3b8"} strokeWidth={isSelected ? "3" : "2"} strokeDasharray={isSelected ? "8 4" : "none"} fill="none" className={`pointer-events-none transition-all duration-300 ${isSelected ? 'animate-flow' : ''}`} style={{ filter: isSelected ? 'drop-shadow(0 0 4px rgba(34, 211, 238, 0.5))' : 'none', }} />
                                            {isSelected && (<foreignObject x={midX - 12} y={midY - 12} width={24} height={24} className="overflow-visible"><button onMouseDown={(e) => { e.stopPropagation(); setConnections(prev => prev.filter(c => c.id !== conn.id)); setSelectedConnectionId(null); }} className="w-6 h-6 bg-rose-500 hover:bg-rose-600 text-white rounded-full flex items-center justify-center shadow-lg transition-transform hover:scale-110 cursor-pointer"><X size={14} /></button></foreignObject>)}
                                        </g>
                                        );
                                    })}
                                    {drawingConnection && <path d={getPath(drawingConnection.startX, drawingConnection.startY, drawingConnection.currX, drawingConnection.currY)} stroke="#06b6d4" strokeWidth="2" strokeDasharray="5,5" fill="none" className="pointer-events-none" />}
                                </svg>
                                {nodes.map(node => {
                                    const def = ATOMIC_NODES[node.type] || { label: node.type, color: 'bg-slate-600' };
                                    const nodeHeight = getNodeHeight(node);
                                    const isConnected = selectedConnectionId && connections.find(c => c.id === selectedConnectionId && (c.source === node.id || c.target === node.id));
                                    const nodeParams = ATOMIC_NODES[node.type]?.params || [];

                                    return (
                                        <div key={node.id} style={{ left: node.x, top: node.y, width: NODE_WIDTH, height: nodeHeight }} onMouseDown={(e) => handleNodeMouseDown(e, node.id)} onDoubleClick={(e) => { e.stopPropagation(); toggleNodeExpand(node.id); }} className={`absolute p-0 rounded-lg shadow-lg border-2 cursor-move group select-none flex flex-col ${selectedNodeId === node.id ? 'border-white z-20 shadow-[0_0_15px_rgba(34,211,238,0.2)]' : 'border-slate-800 z-10 bg-slate-900'} ${isConnected ? 'border-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.4)]' : ''} ${!movingNodeId ? 'transition-[top,left] duration-300 ease-in-out' : ''}`} title="双击展开/收起参数">
                                        <div className={`h-6 shrink-0 px-2 flex items-center justify-between rounded-t-sm ${def.color} bg-opacity-20`}>
                                            <span className={`text-[10px] font-bold ${def.color.replace('bg-', 'text-')}`}>{def.label}</span>
                                            {!expandedNodes.has(node.id) && nodeParams.length > 2 && (
                                                <span className="text-[9px] text-cyan-400/70 font-medium">
                                                    +{nodeParams.length - 2}
                                                </span>
                                            )}
                                        </div>
                                        <div className="flex-1 p-2 bg-slate-900/90 backdrop-blur rounded-b-sm overflow-hidden">
                                            <div className="text-[9px] text-slate-500 font-mono space-y-1">
                                                {nodeParams.slice(0, expandedNodes.has(node.id) ? nodeParams.length : 2).map(param => {
                                                    const value = node.data[param.name] ?? param.default;
                                                    const displayValue = Array.isArray(value)
                                                        ? `[${value.join(', ')}]`
                                                        : String(value);
                                                    return (
                                                        <div key={param.name} className="flex justify-between items-center">
                                                            <span className="text-slate-500 shrink-0">{param.name}:</span>
                                                            <span className="text-slate-300 truncate ml-2 text-right" title={displayValue}>{displayValue}</span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                        <div onMouseUp={(e) => handlePortMouseUp(e, node.id, 'in')} className="absolute left-1/2 -top-1.5 -ml-2 w-4 h-4 bg-slate-700 rounded-full border border-slate-500 hover:bg-cyan-400 cursor-crosshair z-50 flex items-center justify-center hover:scale-125 transition-transform" title="Input" onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}><div className="w-1.5 h-1.5 bg-white rounded-full opacity-50 pointer-events-none"></div></div>
                                        <div onMouseDown={(e) => handlePortMouseDown(e, node.id, 'out')} className="absolute left-1/2 -bottom-1.5 -ml-2 w-4 h-4 bg-slate-700 rounded-full border border-slate-500 hover:bg-cyan-400 cursor-crosshair z-50 flex items-center justify-center hover:scale-125 transition-transform" title="Output"><div className="w-1.5 h-1.5 bg-white rounded-full opacity-50 pointer-events-none"></div></div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        <div className={`bg-slate-950 border-l border-slate-800 z-40 transition-all duration-300 ease-in-out absolute top-0 bottom-0 right-0 h-full shadow-2xl select-none ${showRightPanel ? 'w-72 translate-x-0' : 'w-72 translate-x-full'}`}>
                            <div className="p-4 overflow-y-auto h-full min-w-[18rem]">
                                <div className="mb-4 pb-2 border-b border-slate-800 font-bold text-sm text-white">属性 (Properties)</div>
                                {selectedNodeId ? (
                                    <div className="space-y-4">
                                        <div className="text-xs text-slate-500">ID: {selectedNodeId}</div>
                                        {(ATOMIC_NODES[nodes.find(n => n.id === selectedNodeId)?.type || '']?.params || []).map(p => {
                                            const nodeData = nodes.find(n => n.id === selectedNodeId)?.data || {};
                                            const value = nodeData[p.name] ?? p.default;
                                            return (
                                                <div key={p.name} className="space-y-1">
                                                    <label className="text-xs text-slate-500">{p.name}</label>
                                                    {p.type === 'bool' ? (
                                                        <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded px-2 py-1">
                                                            <span className="text-xs text-slate-400">{value ? '是' : '否'}</span>
                                                            <input
                                                                type="checkbox"
                                                                checked={!!value}
                                                                onChange={(e) => {
                                                                    setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: e.target.checked } } : n));
                                                                }}
                                                                className="accent-cyan-500 w-4 h-4"
                                                            />
                                                        </div>
                                                    ) : p.type === 'select' ? (
                                                        <select
                                                            className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                                                            value={value}
                                                            onChange={(e) => {
                                                                setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: e.target.value } } : n));
                                                            }}
                                                        >
                                                            {p.options?.map(opt => (
                                                                <option key={opt} value={opt}>{opt}</option>
                                                            ))}
                                                        </select>
                                                    ) : p.type === 'dims4' ? (
                                                        <div className="flex gap-1">
                                                            {['左', '上', '右', '下'].map((label, i) => (
                                                                <div key={i} className="flex-1">
                                                                    <input
                                                                        type="number"
                                                                        className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white text-center"
                                                                        value={Array.isArray(value) ? value[i] : 1}
                                                                        onChange={(e) => {
                                                                            const arr = Array.isArray(value) ? [...value] : [1, 1, 1, 1];
                                                                            arr[i] = parseInt(e.target.value) || 0;
                                                                            setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: arr } } : n));
                                                                        }}
                                                                    />
                                                                    <div className="text-[8px] text-slate-600 text-center">{label}</div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    ) : p.type === 'dimsN' ? (
                                                        <div className="space-y-1">
                                                            <div className="flex flex-wrap gap-1">
                                                                {(Array.isArray(value) ? value : [0, 2, 1]).map((v, i) => (
                                                                    <input
                                                                        key={i}
                                                                        type="number"
                                                                        className="w-12 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white text-center"
                                                                        value={v}
                                                                        onChange={(e) => {
                                                                            const arr = Array.isArray(value) ? [...value] : [];
                                                                            arr[i] = parseInt(e.target.value) || 0;
                                                                            setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: arr } } : n));
                                                                        }}
                                                                    />
                                                                ))}
                                                            </div>
                                                            <div className="flex gap-1">
                                                                <button
                                                                    className="flex-1 px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs rounded"
                                                                    onClick={() => {
                                                                        const arr = Array.isArray(value) ? [...value] : [];
                                                                        arr.push(arr.length);
                                                                        setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: arr } } : n));
                                                                    }}
                                                                >+ 添加</button>
                                                                {Array.isArray(value) && value.length > 1 && (
                                                                    <button
                                                                        className="flex-1 px-2 py-1 bg-slate-800 hover:bg-rose-900/50 text-slate-400 text-xs rounded"
                                                                        onClick={() => {
                                                                            const arr = [...(value as number[])];
                                                                            arr.pop();
                                                                            setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: arr } } : n));
                                                                        }}
                                                                    >- 移除</button>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <input
                                                            className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                                                            type={p.type === 'number' ? 'number' : 'text'}
                                                            value={value}
                                                            onChange={(e) => {
                                                                const val = p.type === 'number' ? parseFloat(e.target.value) : e.target.value;
                                                                setNodes(nodes.map(n => n.id === selectedNodeId ? { ...n, data: { ...n.data, [p.name]: val } } : n));
                                                            }}
                                                        />
                                                    )}
                                                </div>
                                            );
                                        })}
                                        <button onClick={() => setNodes(nodes.filter(n => n.id !== selectedNodeId))} className="w-full py-2 bg-rose-900/20 text-rose-400 border border-rose-900 rounded text-xs mt-4">删除节点</button>
                                    </div>
                                ) : <div className="text-xs text-slate-500">选择节点以编辑</div>}
                            </div>
                        </div>
                        
                        <button onClick={() => setShowRightPanel(!showRightPanel)} className={`absolute top-1/2 z-50 p-1 bg-slate-800 border border-slate-700 text-slate-400 rounded-l transition-all duration-300 ease-in-out ${showRightPanel ? 'right-72' : 'right-0'}`} style={{ transform: 'translateY(-50%)' }}><SidebarIcon size={14} className="rotate-180" /></button>
                    </div>
                </div>
            )}

            {/* VIEW 3: WEIGHT REGISTRY (New) */}
            {activeTab === 'weights' && (
                <div className="h-full flex flex-col overflow-hidden">
                    {/* Header */}
                    <div className="flex justify-between items-center p-8 pt-4 pb-4">
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">权重库</h2>
                            <p className="text-slate-400 text-sm">管理已训练的模型权重文件，支持版本树形结构展示。</p>
                        </div>
                        <div className="flex items-center space-x-3">
                            {/* View Toggle */}
                            <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
                                <button
                                    onClick={() => setWeightView('list')}
                                    className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-all ${
                                        weightView === 'list' ? 'bg-slate-700 text-cyan-400' : 'text-slate-400 hover:text-slate-200'
                                    }`}
                                >
                                    <Database size={14} className="mr-2" /> 列表
                                </button>
                                <button
                                    onClick={() => { setWeightView('tree'); loadWeightTree(); }}
                                    className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center transition-all ${
                                        weightView === 'tree' ? 'bg-slate-700 text-cyan-400' : 'text-slate-400 hover:text-slate-200'
                                    }`}
                                >
                                    <GitBranch size={14} className="mr-2" /> 树形图
                                </button>
                            </div>
                            <button onClick={() => setShowWeightUpload(true)} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl border border-slate-700 flex items-center transition-all">
                                <Upload size={16} className="mr-2" /> 导入权重
                            </button>
                        </div>
                    </div>

                    {/* Content Area */}
                    <div className="flex-1 flex overflow-hidden px-8 pb-8">
                        {/* Left Panel - Weight List or Tree */}
                        <div className={`${selectedWeight ? 'w-1/2 pr-4' : 'w-full'} flex flex-col`}>
                            <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden flex-1 flex flex-col">
                                {isLoadingWeights ? (
                                    <div className="p-12 text-center text-slate-500 flex flex-col items-center">
                                        <Loader2 size={32} className="animate-spin mb-4 opacity-50" />
                                        <p>加载权重列表中...</p>
                                    </div>
                                ) : weightView === 'list' ? (
                                    /* List View */
                                    <>
                                        <div className="overflow-auto flex-1">
                                            <table className="w-full text-left border-collapse">
                                                <thead className="bg-slate-900/80 backdrop-blur text-xs font-bold text-slate-500 uppercase tracking-wider sticky top-0">
                                                    <tr>
                                                        <th className="p-4 border-b border-slate-800">权重名称</th>
                                                        <th className="p-4 border-b border-slate-800">任务类型</th>
                                                        <th className="p-4 border-b border-slate-800">版本</th>
                                                        <th className="p-4 border-b border-slate-800">大小</th>
                                                        <th className="p-4 border-b border-slate-800">来源</th>
                                                        <th className="p-4 border-b border-slate-800">创建时间</th>
                                                        <th className="p-4 border-b border-slate-800 text-right">操作</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-slate-800 text-sm">
                                                    {rootWeights.map(w => (
                                                        <tr key={w.id} className="hover:bg-slate-800/50 transition-colors group">
                                                            <td className="p-4">
                                                                <div className="flex items-center">
                                                                    <div className={`w-8 h-8 rounded flex items-center justify-center mr-3 ${
                                                                        w.source_type === 'trained' ? 'bg-purple-900/20 text-purple-400' : 'bg-emerald-900/20 text-emerald-400'
                                                                    }`}>
                                                                        {w.source_type === 'trained' ? <GitBranch size={16} /> : <Database size={16} />}
                                                                    </div>
                                                                    <div>
                                                                        <div className="font-bold text-white group-hover:text-cyan-400 transition-colors cursor-pointer"
                                                                            onClick={() => handleWeightSelect(w)}>
                                                                            {w.display_name || w.name}
                                                                        </div>
                                                                        <div className="text-xs text-slate-500">{w.file_name}</div>
                                                                    </div>
                                                                </div>
                                                            </td>
                                                            <td className="p-4">
                                                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                                                    w.task_type === 'classification' ? 'bg-cyan-900/30 text-cyan-400 border border-cyan-800' :
                                                                    w.task_type === 'detection' ? 'bg-purple-900/30 text-purple-400 border border-purple-800' :
                                                                    'bg-slate-800 text-slate-400'
                                                                }`}>
                                                                    {w.task_type === 'classification' ? '分类' :
                                                                    w.task_type === 'detection' ? '检测' : w.task_type}
                                                                </span>
                                                            </td>
                                                            <td className="p-4 text-slate-500 font-mono text-xs">{w.version}</td>
                                                            <td className="p-4 text-slate-400 font-mono">{w.file_size_mb?.toFixed(2) || '-'} MB</td>
                                                            <td className="p-4">
                                                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                                                    w.source_type === 'trained' ? 'bg-purple-900/30 text-purple-400' : 'bg-emerald-900/30 text-emerald-400'
                                                                }`}>
                                                                    {w.source_type === 'trained' ? '训练' : '导入'}
                                                                </span>
                                                            </td>
                                                            <td className="p-4 text-slate-500 text-xs">{w.created_at ? new Date(w.created_at).toLocaleDateString() : '-'}</td>
                                                            <td className="p-4 text-right">
                                                                <div className="flex justify-end space-x-2">
                                                                    <button
                                                                        onClick={() => handleWeightSelect(w)}
                                                                        className="p-2 text-slate-500 hover:text-cyan-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors"
                                                                        title="查看版本树"
                                                                    >
                                                                        <FolderOpen size={14} />
                                                                    </button>
                                                                    <button onClick={() => handleDeleteWeight(w.id)} className="p-2 text-slate-500 hover:text-rose-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors" title="删除">
                                                                        <Trash2 size={14} />
                                                                    </button>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                        {rootWeights.length === 0 && (
                                            <div className="p-12 text-center text-slate-500">
                                                <Database size={48} className="mx-auto mb-4 opacity-20" />
                                                <p>暂无权重文件，请完成训练后保存</p>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    /* Tree View */
                                    <div className="flex-1 overflow-auto p-4">
                                        {weightTree.length > 0 ? (
                                            <div className="space-y-2">
                                                {weightTree.map(tree => (
                                                    <div key={tree.id} className="border border-slate-800 rounded-lg overflow-hidden">
                                                        <WeightTreeView
                                                            tree={tree}
                                                            onNodeClick={(node) => setSelectedWeight(node)}
                                                            selectedId={selectedWeight?.id}
                                                        />
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="p-12 text-center text-slate-500">
                                                <GitBranch size={48} className="mx-auto mb-4 opacity-20" />
                                                <p>暂无权重版本树</p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right Panel - Weight Detail */}
                        {selectedWeight && (
                            <div className="w-1/2 pl-4">
                                <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden flex flex-col h-full">
                                    {/* Detail Header */}
                                    <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                                        <h3 className="font-bold text-white">权重详情</h3>
                                        <button onClick={() => setSelectedWeight(null)} className="p-1 text-slate-400 hover:text-white">
                                            <X size={18} />
                                        </button>
                                    </div>

                                    {/* Detail Content */}
                                    <div className="flex-1 overflow-auto p-4">
                                        {/* Basic Info */}
                                        <div className="mb-6">
                                            <h4 className="text-sm font-bold text-slate-500 uppercase mb-3">基本信息</h4>
                                            <div className="space-y-2 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">名称</span>
                                                    <span className="text-white font-medium">{selectedWeight.display_name}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">版本</span>
                                                    <span className="text-cyan-400 font-mono">v{selectedWeight.version}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">任务类型</span>
                                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                                        selectedWeight.task_type === 'classification' ? 'bg-cyan-900/30 text-cyan-400' :
                                                        selectedWeight.task_type === 'detection' ? 'bg-purple-900/30 text-purple-400' :
                                                        'bg-slate-800 text-slate-400'
                                                    }`}>
                                                        {selectedWeight.task_type === 'classification' ? '分类' :
                                                        selectedWeight.task_type === 'detection' ? '检测' : selectedWeight.task_type}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">来源</span>
                                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                                        selectedWeight.source_type === 'trained' ? 'bg-purple-900/30 text-purple-400' : 'bg-emerald-900/30 text-emerald-400'
                                                    }`}>
                                                        {selectedWeight.source_type === 'trained' ? '训练生成' : '导入'}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">文件大小</span>
                                                    <span className="text-slate-300">{selectedWeight.file_size_mb?.toFixed(2)} MB</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-400">创建时间</span>
                                                    <span className="text-slate-300">{selectedWeight.created_at ? new Date(selectedWeight.created_at).toLocaleString() : '-'}</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Version Tree */}
                                        {selectedWeight.children && selectedWeight.children.length > 0 && (
                                            <div className="mb-6">
                                                <h4 className="text-sm font-bold text-slate-500 uppercase mb-3">版本树</h4>
                                                <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/50">
                                                    <WeightTreeView
                                                        tree={selectedWeight}
                                                        onNodeClick={(node) => setSelectedWeight(node)}
                                                        selectedId={selectedWeight?.id}
                                                    />
                                                </div>
                                            </div>
                                        )}

                                        {/* Training Config */}
                                        {selectedWeight.source_type === 'trained' && (
                                            <div>
                                                <h4 className="text-sm font-bold text-slate-500 uppercase mb-3">训练信息</h4>
                                                <div className="p-3 bg-slate-950/50 border border-slate-800 rounded-lg text-sm text-slate-400">
                                                    <p>此权重由训练生成，版本号 v{selectedWeight.version}</p>
                                                    {selectedWeight.source_training_id && (
                                                        <p className="mt-2">来源训练任务ID: {selectedWeight.source_training_id}</p>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* VIEW 4: GENERATED MODEL FILES */}
            {activeTab === 'generated' && (
                <div className="h-full flex flex-col p-8 pt-4 overflow-hidden">
                    <div className="flex justify-between items-center mb-6 shrink-0">
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">生成的模型</h2>
                            <p className="text-slate-400 text-sm">通过模型构建器生成的 PyTorch 模型代码文件。</p>
                        </div>
                        <button
                            onClick={loadGeneratedFiles}
                            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl border border-slate-700 flex items-center transition-all"
                        >
                            <Loader2 size={18} className="mr-2" /> 刷新列表
                        </button>
                    </div>

                    <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden flex-1 overflow-y-auto custom-scrollbar">
                        <table className="w-full text-left border-collapse">
                            <thead className="bg-slate-900/80 backdrop-blur text-xs font-bold text-slate-500 uppercase tracking-wider">
                                <tr>
                                    <th className="p-4 border-b border-slate-800">文件名</th>
                                    <th className="p-4 border-b border-slate-800">大小</th>
                                    <th className="p-4 border-b border-slate-800">创建时间</th>
                                    <th className="p-4 border-b border-slate-800 text-right">操作</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800 text-sm">
                                {generatedFiles.map((file) => (
                                    <tr key={file.id} className="hover:bg-slate-800/50 transition-colors group">
                                        <td className="p-4">
                                            <div className="flex items-center">
                                                <div className="w-8 h-8 rounded bg-amber-900/20 text-amber-400 flex items-center justify-center mr-3">
                                                    <FileText size={16} />
                                                </div>
                                                <div className="font-bold text-white group-hover:text-amber-400 transition-colors font-mono text-xs">
                                                    {file.filename}
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4 text-slate-400 font-mono text-xs">
                                            {(file.size / 1024).toFixed(2)} KB
                                        </td>
                                        <td className="p-4 text-slate-500 text-xs">
                                            {new Date(file.created).toLocaleString('zh-CN')}
                                        </td>
                                        <td className="p-4 text-right">
                                            <div className="flex justify-end space-x-2">
                                                <button
                                                    onClick={() => handlePreviewFile(file.id, file.filename)}
                                                    className="p-2 text-slate-500 hover:text-cyan-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors"
                                                    title="预览代码"
                                                >
                                                    <FileText size={14} />
                                                </button>
                                                <button
                                                    onClick={() => handleDownloadFile(file.id, file.filename)}
                                                    className="p-2 text-slate-500 hover:text-emerald-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors"
                                                    title="下载"
                                                >
                                                    <Download size={14} />
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        setDialog({
                                                            isOpen: true,
                                                            type: 'confirm',
                                                            title: '删除文件',
                                                            message: `确定要删除 ${file.filename} 吗？`,
                                                            action: 'delete_file',
                                                            data: file.id
                                                        });
                                                    }}
                                                    className="p-2 text-slate-500 hover:text-rose-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors"
                                                    title="删除"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {generatedFiles.length === 0 && (
                            <div className="p-12 text-center text-slate-500">
                                <Code size={48} className="mx-auto mb-4 opacity-20" />
                                <p>暂无生成的模型文件</p>
                                <p className="text-xs mt-2">请在"架构设计"中构建模型并点击"预览代码"</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

        </div>

        {/* Code Preview Modal */}
        <CodePreviewModal
          show={showCodePreview}
          code={generatedCode}
          metadata={codeMetadata}
          source={codePreviewSource}
          modelName={modelName}
          filename={currentPreviewFilename ?? undefined}
          error={codeGenerationError}
          onClose={() => setShowCodePreview(false)}
          onSave={codePreviewSource === 'builder' ? handleSaveToLibrary : undefined}
          onDelete={codePreviewSource === 'library' ? handleDeleteCurrentPreview : undefined}
          showNotification={showNotification}
        />

        {/* Preset Model Input Modal */}
        <InputModal
          show={showPresetModal}
          title="从预设模型创建架构"
          presetName={selectedPreset?.name || ''}
          presetDescription={selectedPreset?.description}
          placeholder="请输入新架构的名称"
          onConfirm={confirmCreateFromPreset}
          onClose={closePresetModal}
          loading={isCreatingFromPreset}
        />

        {/* Weight Upload Dialog */}
        <WeightUploadDialog
          isOpen={showWeightUpload}
          onClose={() => setShowWeightUpload(false)}
          onUploadComplete={() => {
            setShowWeightUpload(false);
            loadServerWeights();
          }}
          showNotification={showNotification}
        />
    </div>
  );
};

// ==============================
// Weight Upload Dialog Component
// ==============================

interface WeightUploadDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
  showNotification: (msg: string, type: 'error' | 'success' | 'info') => void;
}

const WeightUploadDialog: React.FC<WeightUploadDialogProps> = ({
  isOpen,
  onClose,
  onUploadComplete,
  showNotification
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [weightName, setWeightName] = useState('');
  const [taskType, setTaskType] = useState<TaskType>('auto');
  const [description, setDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');

  // 支持的文件格式
  const supportedFormats = ['.pt', '.pth', '.pkl', '.onnx'];
  const maxSize = 500 * 1024 * 1024; // 500MB

  // 重置表单
  const resetForm = () => {
    setWeightName('');
    setTaskType('auto');
    setDescription('');
    setSelectedFile(null);
    setUploadProgress(0);
    setError('');
  };

  // 处理文件选择
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 验证文件格式
    const isValidFormat = supportedFormats.some(fmt =>
      file.name.toLowerCase().endsWith(fmt)
    );
    if (!isValidFormat) {
      setError(`不支持的文件格式。支持的格式: ${supportedFormats.join(', ')}`);
      setSelectedFile(null);
      return;
    }

    // 验证文件大小
    if (file.size > maxSize) {
      setError('文件大小超过500MB限制');
      setSelectedFile(null);
      return;
    }

    setError('');
    setSelectedFile(file);

    // 自动填充权重名称（从文件名提取，去除扩展名）
    const nameWithoutExt = file.name.replace(/\.(pt|pth|pkl|onnx)$/i, '');
    setWeightName(nameWithoutExt);

    // 根据文件扩展名自动检测框架
    if (file.name.toLowerCase().endsWith('.onnx')) {
      // ONNX通常是检测模型
      if (taskType === 'auto') {
        setTaskType('detection');
      }
    }
  };

  // 处理上传
  const handleUpload = async () => {
    if (!selectedFile) {
      setError('请选择权重文件');
      return;
    }

    if (!weightName.trim()) {
      setError('请输入权重名称');
      return;
    }

    setIsUploading(true);
    setError('');
    setUploadProgress(0);

    try {
      await weightService.uploadWeight(
        {
          file: selectedFile,
          name: weightName.trim(),
          task_type: taskType,
          description: description.trim() || undefined
        },
        (progress) => setUploadProgress(progress)
      );

      // 上传成功
      showNotification('权重文件上传成功', 'success');
      onUploadComplete();
      resetForm();
    } catch (err: any) {
      setError(err.message || '上传失败，请重试');
    } finally {
      setIsUploading(false);
    }
  };

  // 弹窗关闭时重置
  useEffect(() => {
    if (!isOpen) {
      resetForm();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-md shadow-2xl animate-in fade-in zoom-in duration-200">
        {/* Header */}
        <div className="flex items-center mb-6">
          <div className="w-12 h-12 rounded-full flex items-center justify-center bg-emerald-900/30 text-emerald-500 mr-4">
            <HardDrive size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">导入权重</h3>
            <p className="text-sm text-slate-400">上传训练好的模型权重文件</p>
          </div>
        </div>

        {/* Form */}
        <div className="space-y-4 mb-6">
          {/* 文件选择 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              权重文件 <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <input
                ref={fileInputRef}
                type="file"
                accept={supportedFormats.join(',')}
                onChange={handleFileSelect}
                className="hidden"
                disabled={isUploading}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-left text-slate-400 hover:border-emerald-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {selectedFile ? (
                  <div className="flex items-center">
                    <CheckCircle size={18} className="text-emerald-500 mr-2" />
                    <span className="text-white truncate">{selectedFile.name}</span>
                    <span className="text-slate-500 text-xs ml-auto">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center">
                    <Upload size={18} className="mr-2" />
                    <span>选择权重文件</span>
                  </div>
                )}
              </button>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              支持格式: {supportedFormats.join(', ')} | 最大500MB
            </p>
          </div>

          {/* 权重名称 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              权重名称 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={weightName}
              onChange={(e) => setWeightName(e.target.value)}
              placeholder="输入权重名称"
              disabled={isUploading}
              className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-50"
            />
          </div>

          {/* 任务类型 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              任务类型 <span className="text-red-500">*</span>
            </label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value as TaskType)}
              disabled={isUploading}
              className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-emerald-500 disabled:opacity-50"
            >
              <option value="auto">自动检测</option>
              <option value="classification">分类 (Classification)</option>
              <option value="detection">检测 (Detection)</option>
            </select>
            <p className="text-xs text-slate-500 mt-1">
              {taskType === 'auto' ? '系统将根据模型结构自动检测任务类型' : ''}
            </p>
          </div>

          {/* 描述 */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              描述
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="输入权重描述（可选）"
              rows={2}
              disabled={isUploading}
              className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-50 resize-none"
            />
          </div>

          {/* 上传进度 */}
          {isUploading && (
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>上传中...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-emerald-500 to-cyan-500 h-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* 错误提示 */}
          {error && (
            <div className="p-3 bg-rose-900/20 border border-rose-800 rounded-lg flex items-start">
              <AlertTriangle size={16} className="text-rose-500 mr-2 mt-0.5 flex-shrink-0" />
              <span className="text-sm text-rose-400">{error}</span>
            </div>
          )}
        </div>

        {/* Footer Buttons */}
        <div className="flex justify-end space-x-3">
          <button
            onClick={onClose}
            disabled={isUploading}
            className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={handleUpload}
            disabled={isUploading || !selectedFile}
            className="px-6 py-2 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-medium rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isUploading ? (
              <>
                <Loader2 size={18} className="mr-2 animate-spin" />
                上传中
              </>
            ) : (
              <>
                <Upload size={18} className="mr-2" />
                上传
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelBuilder;