import React, { useState, useRef, useEffect } from 'react';
import { 
  Layers, ArrowRight, Save, Plus, GitBranch, 
  Edit3, Trash2, ZoomIn, ZoomOut, Sidebar as SidebarIcon,
  CheckCircle, AlertTriangle, Code, Info, X, Copy,
  Layout, Package, Box, AlertOctagon, HelpCircle,
  Database, HardDrive, Download, Upload, Tag
} from 'lucide-react';
import { ModelNode, WeightCheckpoint } from '../types';

// Constants for precise layout
const NODE_WIDTH = 160;

// --- MOCK DATA: Weights ---
const INITIAL_WEIGHTS: WeightCheckpoint[] = [
    { id: 'w1', name: 'yolov8n-traffic-best.pt', architecture: 'YOLOv8-Nano', format: 'PyTorch', size: '6.2 MB', accuracy: 'mAP 0.68', created: '2023-10-25', tags: ['Production', 'Traffic'] },
    { id: 'w2', name: 'resnet50-mri-v2.pt', architecture: 'ResNet50', format: 'PyTorch', size: '98 MB', accuracy: 'Acc 94.2%', created: '2023-10-24', tags: ['Medical', 'Best'] },
    { id: 'w3', name: 'yolov8-face-deploy.onnx', architecture: 'YOLOv8-Small', format: 'ONNX', size: '12 MB', accuracy: 'mAP 0.72', created: '2023-10-20', tags: ['Edge', 'Optimized'] },
];

// 1. ATOMIC OPERATORS
const ATOMIC_NODES: Record<string, { label: string, color: string, category: string, params: { name: string, type: 'text'|'number'|'bool'|'select', default: any }[] }> = {
  // IO
  "Input": { label: "Input", color: "bg-slate-600", category: "IO", params: [{ name: "c", type: "number", default: 3 }, { name: "h", type: "number", default: 640 }, { name: "w", type: "number", default: 640 }] },
  // Layers
  "Conv2d": { label: "Conv2d", color: "bg-cyan-600", category: "Layer", params: [{ name: "in", type: "number", default: 3 }, { name: "out", type: "number", default: 64 }, { name: "k", type: "number", default: 3 }, { name: "s", type: "number", default: 1 }, { name: "p", type: "number", default: 1 }] },
  "Linear": { label: "Linear", color: "bg-cyan-600", category: "Layer", params: [{ name: "in_f", type: "number", default: 512 }, { name: "out_f", type: "number", default: 10 }] },
  "BatchNorm2d": { label: "BN2d", color: "bg-cyan-700", category: "Layer", params: [{ name: "num_f", type: "number", default: 64 }] },
  "LayerNorm": { label: "LayerNorm", color: "bg-cyan-700", category: "Layer", params: [{ name: "shape", type: "text", default: "[64]" }] },
  "Dropout": { label: "Dropout", color: "bg-slate-500", category: "Layer", params: [{ name: "p", type: "number", default: 0.5 }] },
  "Flatten": { label: "Flatten", color: "bg-slate-500", category: "Layer", params: [] },
  // Activations
  "ReLU": { label: "ReLU", color: "bg-purple-600", category: "Activation", params: [{ name: "inplace", type: "bool", default: true }] },
  "LeakyReLU": { label: "LReLU", color: "bg-purple-600", category: "Activation", params: [{ name: "slope", type: "number", default: 0.01 }] },
  "SiLU": { label: "SiLU", color: "bg-purple-600", category: "Activation", params: [] },
  "Sigmoid": { label: "Sigmoid", color: "bg-purple-600", category: "Activation", params: [] },
  "Softmax": { label: "Softmax", color: "bg-purple-600", category: "Activation", params: [{ name: "dim", type: "number", default: 1 }] },
  // Pooling
  "MaxPool2d": { label: "MaxPool", color: "bg-rose-600", category: "Pooling", params: [{ name: "k", type: "number", default: 2 }, { name: "s", type: "number", default: 2 }] },
  "AvgPool2d": { label: "AvgPool", color: "bg-rose-600", category: "Pooling", params: [{ name: "k", type: "number", default: 2 }] },
  "AdaptiveAvg": { label: "AdaptAvg", color: "bg-rose-600", category: "Pooling", params: [{ name: "out", type: "number", default: 1 }] },
  // Ops
  "Concat": { label: "Concat", color: "bg-amber-600", category: "Ops", params: [{ name: "dim", type: "number", default: 1 }] },
  "Add": { label: "Add", color: "bg-amber-600", category: "Ops", params: [] },
  "Upsample": { label: "Upsample", color: "bg-amber-600", category: "Ops", params: [{ name: "scale", type: "number", default: 2 }, { name: "mode", type: "text", default: "nearest" }] },
  "Identity": { label: "Identity", color: "bg-slate-500", category: "Ops", params: [] },
  // Heads
  "YOLO Head": { label: "YOLO Head", color: "bg-blue-600", category: "Head", params: [{ name: "nc", type: "number", default: 80 }] },
  "Classify Head": { label: "Cls Head", color: "bg-blue-600", category: "Head", params: [{ name: "nc", type: "number", default: 1000 }] },
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
  id: string;
  name: string;
  version: string;
  status: string;
  type: string;
  updated: string;
  nodes: VisualNode[];
  connections: Connection[];
}

const DEFAULT_MODELS: ModelData[] = [
  { 
    id: 'm1', name: 'YOLOv8-Nano-Base', version: 'v1.0', status: 'Ready', type: 'Detection', updated: '2h ago',
    nodes: [
      { id: 'n1', type: 'Input', label: 'Input', x: 300, y: 50, inputs: [], outputs: [], data: { c: 3, h: 640, w: 640 } },
      { id: 'n2', type: 'Conv2d', label: 'Conv2d', x: 300, y: 150, inputs: [], outputs: [], data: { in: 3, out: 16 } },
      { id: 'n3', type: 'BatchNorm2d', label: 'BN2d', x: 300, y: 250, inputs: [], outputs: [], data: { num_f: 16 } },
      { id: 'n4', type: 'SiLU', label: 'SiLU', x: 300, y: 350, inputs: [], outputs: [], data: {} },
    ],
    connections: [
      { id: 'c1', source: 'n1', target: 'n2' },
      { id: 'c2', source: 'n2', target: 'n3' },
      { id: 'c3', source: 'n3', target: 'n4' },
    ]
  }
];

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
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
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


const ModelBuilder: React.FC = () => {
  // Main View State: 'architectures' or 'weights'
  const [activeTab, setActiveTab] = useState<'architectures' | 'weights'>('architectures');
  
  // Sub View for Architectures: 'list' or 'builder'
  const [archView, setArchView] = useState<'list' | 'builder'>('list');

  // Model State with Persistence
  const [models, setModels] = useState<ModelData[]>(() => {
    try {
      const saved = localStorage.getItem('neurocore_models');
      if (saved) return JSON.parse(saved);
      return DEFAULT_MODELS;
    } catch {
      return DEFAULT_MODELS;
    }
  });

  // Weights State
  const [weights, setWeights] = useState<WeightCheckpoint[]>(INITIAL_WEIGHTS);

  // Persist models whenever they change
  useEffect(() => {
    localStorage.setItem('neurocore_models', JSON.stringify(models));
  }, [models]);
  
  // Builder State
  const [activeModelId, setActiveModelId] = useState<string | null>(null); 
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

  // --- ACTIONS ---
  
  const handleDeleteModel = (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Open Dialog instead of window.confirm
    setDialog({
        isOpen: true,
        type: 'confirm',
        title: '删除模型',
        message: '确定要删除该模型吗？此操作不可恢复。',
        action: 'delete_model',
        data: id
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
  const handleDialogConfirm = (val?: string) => {
      // 1. DELETE MODEL
      if (dialog.action === 'delete_model' && dialog.data) {
          setModels(prev => prev.filter(m => m.id !== dialog.data));
          showNotification("模型已删除", "success");
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
          setWeights(prev => prev.filter(w => w.id !== dialog.data));
          showNotification("权重文件已删除", "success");
      }

      setDialog({ ...dialog, isOpen: false });
  };

  const handleCreateNew = () => {
    setActiveModelId(null); setModelName("Untitled Architecture");
    setNodes([]); setConnections([]); setScale(1); setPan({ x: 0, y: 0 }); setArchView('builder');
  };

  const handleEditModel = (model: ModelData) => {
    setActiveModelId(model.id); setModelName(model.name);
    setNodes(JSON.parse(JSON.stringify(model.nodes || [])));
    setConnections(JSON.parse(JSON.stringify(model.connections || [])));
    setScale(1); setPan({ x: 0, y: 0 }); setArchView('builder');
  };

  const handleSave = (asNew: boolean = false) => {
    if (!modelName.trim()) { showNotification("模型名称不能为空", 'error'); return; }
    const isDuplicate = models.some(m => m.name === modelName && (asNew || m.id !== activeModelId));
    if (isDuplicate) { showNotification("已存在同名模型，请修改名称", 'error'); return; }
    const currentData = { nodes, connections, updated: 'Just now', name: modelName, status: 'Draft', type: 'Custom', version: 'v0.1' };
    if (activeModelId && !asNew) {
        setModels(prev => prev.map(m => m.id === activeModelId ? { ...m, ...currentData } : m));
        showNotification("模型已更新", 'success');
    } else {
        const newId = `m_${Date.now()}`;
        setModels(prev => [{ ...currentData, id: newId }, ...prev]);
        setActiveModelId(newId);
        showNotification("新模型已保存", 'success');
    }
  };

  const getNodeHeight = (node: VisualNode) => {
    const def = ATOMIC_NODES[node.type] || { params: [] };
    const paramsCount = def.params.length;
    const displayCount = Math.min(paramsCount, 4);
    return 24 + Math.max(20, displayCount * 14 + 12);
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
                    onClick={() => { setActiveTab('architectures'); if(archView === 'builder') setArchView('list'); }} // Reset to list when switching back
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
            </div>
        </div>

        {/* Global Notification */}
        {notification && (
            <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[100] px-4 py-2 rounded-lg shadow-lg border flex items-center ${
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
                        <button onClick={handleCreateNew} className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-lg shadow-cyan-900/20 flex items-center transition-all">
                            <Plus size={18} className="mr-2" /> 新建架构
                        </button>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {models.map(model => (
                        <div key={model.id} onClick={() => handleEditModel(model)} className="glass-panel p-5 rounded-xl border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all group flex flex-col cursor-pointer relative overflow-hidden">
                            <button 
                                onClick={(e) => handleDeleteModel(model.id, e)} 
                                className="absolute top-4 right-4 p-2 text-slate-500 hover:text-rose-400 bg-slate-900/50 hover:bg-slate-900 rounded-full transition-all z-20 opacity-0 group-hover:opacity-100"
                                title="删除模型"
                            >
                                <Trash2 size={16} />
                            </button>

                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center space-x-3">
                                    <div className={`p-3 rounded-lg border border-slate-700/50 group-hover:scale-110 transition-transform ${model.status === 'Ready' ? 'bg-cyan-900/20 text-cyan-400' : model.status === 'Training' ? 'bg-emerald-900/20 text-emerald-400 animate-pulse' : 'bg-slate-800 text-slate-500'}`}>
                                        <GitBranch size={24} />
                                    </div>
                                    <div className={`px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider border ${
                                        model.status === 'Ready' ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-400' : 
                                        model.status === 'Training' ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-400 animate-pulse' : 
                                        'bg-slate-800 border-slate-700 text-slate-400'
                                    }`}>
                                        {model.status}
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1">
                                <h3 className="text-lg font-bold text-white mb-1 group-hover:text-cyan-400 transition-colors line-clamp-1" title={model.name}>{model.name}</h3>
                                <div className="flex items-center space-x-2 text-xs text-slate-500 mb-4"><span className="bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">{model.version}</span><span>• {model.type}</span></div>
                                <div className="text-[10px] text-slate-600">{model.nodes?.length || 0} Layers • {model.connections?.length || 0} Connections</div>
                            </div>

                            <div className="pt-4 border-t border-slate-800/50 flex items-center justify-between">
                                <span className="text-[10px] text-slate-600">Updated: {model.updated}</span>
                            </div>
                            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-purple-600 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
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
                        <button onClick={() => setArchView('list')} className="text-slate-500 hover:text-white"><ArrowRight size={20} className="rotate-180" /></button>
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
                            <button onClick={() => handleSave(false)} className="flex items-center px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded shadow-lg shadow-cyan-900/20 transition-all"><Save size={14} className="mr-2" /> 保存</button>
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
                                        {['Layer', 'Activation', 'Pooling', 'Ops', 'Head', 'IO'].map(cat => (
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
                                        <div key={node.id} style={{ left: node.x, top: node.y, width: NODE_WIDTH, height: nodeHeight }} onMouseDown={(e) => handleNodeMouseDown(e, node.id)} className={`absolute p-0 rounded-lg shadow-lg border-2 cursor-move group select-none flex flex-col ${selectedNodeId === node.id ? 'border-white z-20 shadow-[0_0_15px_rgba(34,211,238,0.2)]' : 'border-slate-800 z-10 bg-slate-900'} ${isConnected ? 'border-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.4)]' : ''} ${!movingNodeId ? 'transition-[top,left] duration-300 ease-in-out' : ''}`}>
                                        <div className={`h-6 shrink-0 px-2 flex items-center justify-between rounded-t-sm ${def.color} bg-opacity-20`}><span className={`text-[10px] font-bold ${def.color.replace('bg-', 'text-')}`}>{def.label}</span></div>
                                        <div className="flex-1 p-1.5 bg-slate-900/90 backdrop-blur rounded-b-sm overflow-hidden">
                                            <div className="text-[9px] text-slate-500 font-mono space-y-0.5">
                                                {nodeParams.slice(0, 4).map(param => (
                                                    <div key={param.name} className="flex justify-between">
                                                        <span className="text-slate-500">{param.name}:</span>
                                                        <span className="text-slate-300 truncate ml-1">{String(node.data[param.name] ?? '')}</span>
                                                    </div>
                                                ))}
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
                                        {(ATOMIC_NODES[nodes.find(n => n.id === selectedNodeId)?.type || '']?.params || []).map(p => (
                                        <div key={p.name} className="space-y-1"><label className="text-xs text-slate-500">{p.name}</label>
                                        {p.type === 'bool' ? <input type="checkbox" defaultChecked={nodes.find(n => n.id === selectedNodeId)?.data[p.name]} className="accent-cyan-500" /> : <input className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white" defaultValue={nodes.find(n => n.id === selectedNodeId)?.data[p.name]} />}
                                        </div>
                                        ))}
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
                <div className="h-full flex flex-col p-8 pt-4 overflow-y-auto custom-scrollbar">
                    <div className="flex justify-between items-center mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">权重库</h2>
                            <p className="text-slate-400 text-sm">管理已训练的模型权重文件 (Checkpoints)。</p>
                        </div>
                        <button onClick={() => showNotification("功能开发中: 导入权重", "info")} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl border border-slate-700 flex items-center transition-all">
                            <Upload size={18} className="mr-2" /> 导入权重
                        </button>
                    </div>

                    <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden">
                        <table className="w-full text-left border-collapse">
                            <thead className="bg-slate-900/80 backdrop-blur text-xs font-bold text-slate-500 uppercase tracking-wider">
                                <tr>
                                    <th className="p-4 border-b border-slate-800">Filename</th>
                                    <th className="p-4 border-b border-slate-800">Source Arch</th>
                                    <th className="p-4 border-b border-slate-800">Format</th>
                                    <th className="p-4 border-b border-slate-800">Metric</th>
                                    <th className="p-4 border-b border-slate-800">Size</th>
                                    <th className="p-4 border-b border-slate-800">Created</th>
                                    <th className="p-4 border-b border-slate-800 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800 text-sm">
                                {weights.map(w => (
                                    <tr key={w.id} className="hover:bg-slate-800/50 transition-colors group">
                                        <td className="p-4">
                                            <div className="flex items-center">
                                                <div className="w-8 h-8 rounded bg-emerald-900/20 text-emerald-400 flex items-center justify-center mr-3">
                                                    <HardDrive size={16} />
                                                </div>
                                                <div>
                                                    <div className="font-bold text-white group-hover:text-emerald-400 transition-colors">{w.name}</div>
                                                    <div className="flex mt-1 space-x-1">
                                                        {w.tags.map(tag => (
                                                            <span key={tag} className="text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-400 flex items-center">
                                                                {tag}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4 text-slate-300 font-mono text-xs">{w.architecture}</td>
                                        <td className="p-4 text-slate-400">{w.format}</td>
                                        <td className="p-4 text-emerald-400 font-mono font-bold">{w.accuracy}</td>
                                        <td className="p-4 text-slate-400 font-mono">{w.size}</td>
                                        <td className="p-4 text-slate-500 text-xs">{w.created}</td>
                                        <td className="p-4 text-right">
                                            <div className="flex justify-end space-x-2">
                                                <button className="p-2 text-slate-500 hover:text-cyan-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors" title="Download">
                                                    <Download size={14} />
                                                </button>
                                                <button onClick={() => handleDeleteWeight(w.id)} className="p-2 text-slate-500 hover:text-rose-400 bg-slate-900/50 hover:bg-slate-900 rounded-lg transition-colors" title="Delete">
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {weights.length === 0 && (
                            <div className="p-12 text-center text-slate-500">
                                <Database size={48} className="mx-auto mb-4 opacity-20" />
                                <p>暂无权重文件，请完成训练后保存</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

        </div>
    </div>
  );
};

export default ModelBuilder;