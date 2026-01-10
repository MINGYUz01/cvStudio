
import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  Info,
  Loader2,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  ImageIcon,
  GripVertical,
  FileText
} from 'lucide-react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

import {
  getAugmentationOperators,
  getAugmentationStrategies,
  createAugmentationStrategy,
  updateAugmentationStrategy,
  deleteAugmentationStrategy,
  previewAugmentation
} from '../src/services/augmentation';

import { datasetService } from '../src/services/datasets';

import type {
  AugmentationOperator,
  AugmentationStrategy,
  PipelineItem,
  DatasetItem
} from '../types';

type ParamType = 'range' | 'number' | 'boolean' | 'select' | 'tuple';

interface AugParamDef {
  name: string;
  label_zh: string;
  label_en: string;
  type: ParamType;
  min_value?: number;
  max_value?: number;
  step?: number;
  options?: string[];
  default: any;
  description: string;
}

interface OperatorDef {
  id: string;
  name_zh: string;
  name_en: string;
  category: string;
  category_label_zh: string;
  category_label_en: string;
  description: string;
  params: AugParamDef[];
}

interface ImageInfo {
  id: string;
  path: string;
  filename: string;
}

interface NotificationProps {
  msg: string;
  type: 'error' | 'success' | 'info';
}

// 自定义数字输入组件（带加减按钮）
interface NumberInputProps {
  value: number;
  onChange: (val: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}

function NumberInput({ value, onChange, min, max, step, className }: NumberInputProps) {
  const handleIncrement = () => {
    const newValue = Math.min(max ?? Number.MAX_SAFE_INTEGER, value + (step ?? 1));
    onChange(newValue);
  };

  const handleDecrement = () => {
    const newValue = Math.max(min ?? Number.MIN_SAFE_INTEGER, value - (step ?? 1));
    onChange(newValue);
  };

  return (
    <div className={`flex items-center ${className}`}>
      <button
        type="button"
        onClick={handleDecrement}
        className="px-2 py-1 bg-slate-800 border border-r-0 border-slate-700 rounded-l text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
      >
        −
      </button>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-20 bg-slate-900 border-y border-slate-700 text-center text-xs text-white outline-none focus:border-cyan-500"
        min={min}
        max={max}
        step={step}
      />
      <button
        type="button"
        onClick={handleIncrement}
        className="px-2 py-1 bg-slate-800 border border-l-0 border-slate-700 rounded-r text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
      >
        +
      </button>
    </div>
  );
}

// 可拖拽的流水线项目组件
interface SortablePipelineItemProps {
  item: PipelineItem;
  index: number;
  operatorDef?: OperatorDef;
  isSelected: boolean;
  onSelect: (instanceId: string) => void;
  onRemove: (instanceId: string) => void;
}

function SortablePipelineItem({
  item,
  index,
  operatorDef,
  isSelected,
  onSelect,
  onRemove
}: SortablePipelineItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: item.instanceId });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`p-3 rounded-lg border cursor-pointer relative transition-all group ${
        isSelected
          ? 'bg-cyan-950/40 border-cyan-500/50 shadow-[0_0_10px_rgba(6,182,212,0.1)]'
          : 'bg-slate-900 border-slate-800 hover:border-slate-600'
      }`}
    >
      <div className="flex justify-between items-center">
        <div className="flex items-center flex-1">
          {/* 拖拽手柄 */}
          <button
            {...attributes}
            {...listeners}
            className="mr-2 text-slate-600 hover:text-cyan-400 cursor-grab active:cursor-grabbing p-1"
            title="拖动排序"
          >
            <GripVertical size={14} />
          </button>
          <div
            className="flex-1"
            onClick={() => onSelect(item.instanceId)}
          >
            <div className="flex justify-between items-center mb-1">
              <div className="flex items-center">
                <span className={`flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold mr-2 ${isSelected ? 'bg-cyan-500 text-black' : 'bg-slate-800 text-slate-500'}`}>
                  {index + 1}
                </span>
                <span className={`text-sm font-bold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                  {operatorDef?.name_zh || item.operatorId}
                </span>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); onRemove(item.instanceId); }}
                className="text-slate-600 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X size={14} />
              </button>
            </div>
            <div className="text-[10px] text-slate-500 ml-7 flex items-center">
              <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${item.enabled ? 'bg-emerald-500' : 'bg-slate-600'}`}></div>
              {item.enabled ? '已启用' : '已禁用'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DataAugmentation() {
  const [view, setView] = useState<'list' | 'editor'>('list');
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [previewSeed, setPreviewSeed] = useState(1);
  const [notification, setNotification] = useState<NotificationProps | null>(null);

  // 图片选择状态
  const [datasetImages, setDatasetImages] = useState<ImageInfo[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [isLoadingImages, setIsLoadingImages] = useState(false);

  // 请求取消控制器
  const abortControllerRef = useRef<AbortController | null>(null);

  // 加载状态
  const [isLoadingOperators, setIsLoadingOperators] = useState(false);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // 数据状态
  const [operators, setOperators] = useState<OperatorDef[]>([]);
  const [strategies, setStrategies] = useState<AugmentationStrategy[]>([]);

  // Custom Modal State
  const [deleteModal, setDeleteModal] = useState<{ isOpen: boolean; strategyId: number | null }>({
    isOpen: false,
    strategyId: null
  });
  const [descriptionModal, setDescriptionModal] = useState<{ isOpen: boolean }>({
    isOpen: false
  });

  // Working Draft State
  const [draftStrategy, setDraftStrategy] = useState<AugmentationStrategy | null>(null);
  const [selectedPipelineItemId, setSelectedPipelineItemId] = useState<string | null>(null);

  const [tempName, setTempName] = useState('');
  const [tempDescription, setTempDescription] = useState('');
  const [isEditingName, setIsEditingName] = useState(false);

  // 预览状态
  const [previewResult, setPreviewResult] = useState<any>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);

  // 当前图片的base64（用于流水线为空时显示原图）
  const [currentImageBase64, setCurrentImageBase64] = useState<string | null>(null);

  // 流水线分割面板状态（0-1之间，表示上半部分占的比例）
  const [pipelineSplitRatio, setPipelineSplitRatio] = useState(0.55);
  const [isDraggingSplitter, setIsDraggingSplitter] = useState(false);

  // 加载增强算子列表
  const loadOperators = useCallback(async () => {
    setIsLoadingOperators(true);
    try {
      const response = await getAugmentationOperators();
      if (response.success && response.data) {
        // 展开分类结构
        const ops: OperatorDef[] = [];
        Object.values(response.data).forEach((category: any) => {
          category.operators.forEach((op: AugmentationOperator) => {
            // 转换参数类型：后端的 integer/float 映射到前端的 number
            const convertedParams = op.params.map((p: any) => ({
              ...p,
              type: (p.param_type === 'integer' || p.param_type === 'float')
                ? ('number' as const)
                : (p.param_type as ParamType)
            }));

            ops.push({
              id: op.id,
              name_zh: op.name_zh,
              name_en: op.name_en,
              category: op.category,
              category_label_zh: op.category_label_zh,
              category_label_en: op.category_label_en,
              description: op.description,
              params: convertedParams as AugParamDef[]
            });
          });
        });
        setOperators(ops);
      }
    } catch (error) {
      showNotification('获取增强算子失败: ' + (error as Error).message, 'error');
    } finally {
      setIsLoadingOperators(false);
    }
  }, []);

  // 加载增强策略列表
  const loadStrategies = useCallback(async () => {
    setIsLoadingStrategies(true);
    try {
      const response = await getAugmentationStrategies({ limit: 100 });
      if (response.success && response.data) {
        setStrategies(response.data.strategies);
      }
    } catch (error) {
      showNotification('获取增强策略失败: ' + (error as Error).message, 'error');
    } finally {
      setIsLoadingStrategies(false);
    }
  }, []);

  // 加载数据集列表
  const loadDatasets = useCallback(async () => {
    setIsLoadingDatasets(true);
    try {
      const result = await datasetService.getDatasets({ limit: 100 });
      if (result && result.length > 0) {
        setDatasets(result.map((ds: any) => ({
          id: ds.id.toString(),
          name: ds.name,
          type: ds.format,
          count: ds.num_images,
          size: '0 B',
          lastModified: new Date(ds.created_at).toLocaleDateString(),
          description: ds.description,
          rawMeta: { path: ds.path }
        })));
        if (!selectedDatasetId) {
          setSelectedDatasetId(result[0].id.toString());
        }
      }
    } catch (error) {
      console.error('加载数据集失败:', error);
    } finally {
      setIsLoadingDatasets(false);
    }
  }, [selectedDatasetId]);

  // 加载数据集图片列表
  const loadDatasetImages = useCallback(async (datasetId: string) => {
    if (!datasetId) return;
    setIsLoadingImages(true);
    try {
      const result = await datasetService.getImages(
        parseInt(datasetId),
        { page: 1, page_size: 100 }
      );
      const images = result.images || [];
      setDatasetImages(images);
      setCurrentImageIndex(0);
      setPreviewResult(null); // 清除之前的预览
    } catch (error) {
      console.error('加载图片列表失败:', error);
      setDatasetImages([]);
    } finally {
      setIsLoadingImages(false);
    }
  }, []);

  // 初始化
  useEffect(() => {
    loadOperators();
    loadStrategies();
    loadDatasets();
  }, [loadOperators, loadStrategies, loadDatasets]);

  // 当选择的数据集改变时，加载图片列表
  useEffect(() => {
    if (selectedDatasetId) {
      loadDatasetImages(selectedDatasetId);
    }
  }, [selectedDatasetId, loadDatasetImages]);

  // Notification Helper
  const showNotification = (msg: string, type: 'error' | 'success' | 'info') => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // Actions

  // Delete Strategy
  const handleDeleteClick = (id: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDeleteModal({ isOpen: true, strategyId: id });
  };

  const confirmDelete = async () => {
    if (deleteModal.strategyId === null) return;

    try {
      await deleteAugmentationStrategy(deleteModal.strategyId);
      await loadStrategies();
      showNotification('策略已删除', 'success');
    } catch (error) {
      showNotification('删除失败: ' + (error as Error).message, 'error');
    }
    setDeleteModal({ isOpen: false, strategyId: null });
  };

  // Edit Strategy
  const handleEditStrategy = async (id: number) => {
    const s = strategies.find(strat => strat.id === id);
    if (s) {
      setDraftStrategy({
        ...s,
        pipeline: JSON.parse(JSON.stringify(s.pipeline))
      });
      setTempName(s.name);
      setTempDescription(s.description || '');
      setView('editor');
      setIsEditingName(false);
    }
  };

  // Create Strategy
  const handleCreateStrategy = () => {
    const newStrategy: AugmentationStrategy = {
      id: 0, // 新策略
      user_id: 0,
      name: '新建策略',
      description: '',
      pipeline: [],
      is_default: false,
      created_at: new Date().toISOString(),
      updated_at: null
    };
    setDraftStrategy(newStrategy);
    setTempName('新建策略');
    setTempDescription('');
    setView('editor');
    setIsEditingName(false);
  };

  // Save Strategy
  const handleSave = async (asNew: boolean = false) => {
    if (!draftStrategy) return;

    const nameToSave = tempName.trim();
    if (!nameToSave) {
      showNotification('策略名称不能为空', 'error');
      return;
    }

    setIsSaving(true);
    try {
      const payload = {
        name: nameToSave,
        description: tempDescription.trim(),
        pipeline: draftStrategy.pipeline
      };

      if (asNew || draftStrategy.id === 0) {
        // Create new
        const response = await createAugmentationStrategy(payload);
        if (response.success && response.data) {
          setDraftStrategy(response.data);
          showNotification('策略已保存', 'success');
        }
      } else {
        // Update existing
        const response = await updateAugmentationStrategy(draftStrategy.id, payload);
        if (response.success && response.data) {
          setDraftStrategy(response.data);
          showNotification('策略已更新', 'success');
        }
      }
      await loadStrategies();
      setIsEditingName(false);
    } catch (error) {
      showNotification('保存失败: ' + (error as Error).message, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  // Operator Helpers
  const addOperator = (opId: string) => {
    if (!draftStrategy) return;
    const opDef = operators.find(op => op.id === opId);
    if (!opDef) return;

    const defaultParams: any = {};
    opDef.params.forEach(p => {
      defaultParams[p.name] = p.default;
    });

    const newItem: PipelineItem = {
      instanceId: Date.now().toString(),
      operatorId: opId,
      enabled: true,
      params: defaultParams
    };

    setDraftStrategy(prev => prev ? ({
      ...prev,
      pipeline: [...prev.pipeline, newItem]
    }) : null);
    setSelectedPipelineItemId(newItem.instanceId);
  };

  const removeOperator = (instanceId: string) => {
    setDraftStrategy(prev => prev ? ({
      ...prev,
      pipeline: prev.pipeline.filter(p => p.instanceId !== instanceId)
    }) : null);
    if (selectedPipelineItemId === instanceId) setSelectedPipelineItemId(null);
  };

  // 分隔面板拖拽处理
  const handleSplitterMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingSplitter(true);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const pipelineColumn = document.getElementById('pipeline-column');
      if (!pipelineColumn) return;

      const rect = pipelineColumn.getBoundingClientRect();
      const relativeY = moveEvent.clientY - rect.top;
      const newRatio = Math.max(0.2, Math.min(0.8, relativeY / rect.height));
      setPipelineSplitRatio(newRatio);
    };

    const handleMouseUp = () => {
      setIsDraggingSplitter(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const updateParam = (instanceId: string, param: string, value: any) => {
    setDraftStrategy(prev =>
      prev ? {
        ...prev,
        pipeline: prev.pipeline.map(p =>
          p.instanceId === instanceId
            ? { ...p, params: { ...p.params, [param]: value } }
            : p
        )
      } : null
    );
  };

  const toggleOperator = (instanceId: string) => {
    setDraftStrategy(prev =>
      prev ? {
        ...prev,
        pipeline: prev.pipeline.map(p =>
          p.instanceId === instanceId
            ? { ...p, enabled: !p.enabled }
            : p
        )
      } : null
    );
  };

  // 拖拽传感器配置
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // 拖拽结束处理
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id && draftStrategy) {
      const oldIndex = draftStrategy.pipeline.findIndex(item => item.instanceId === active.id);
      const newIndex = draftStrategy.pipeline.findIndex(item => item.instanceId === over.id);

      setDraftStrategy(prev => prev ? {
        ...prev,
        pipeline: arrayMove(prev.pipeline, oldIndex, newIndex)
      } : null);
    }
  };

  // 加载当前原图（用于流水线为空时显示）
  const loadCurrentImage = useCallback(async (signal?: AbortSignal) => {
    const currentImage = datasetImages[currentImageIndex];
    if (!currentImage) return;

    setIsPreviewLoading(true);
    try {
      // 使用空pipeline调用预览API获取原图
      const response = await previewAugmentation({
        image_path: currentImage.path,
        dataset_id: selectedDatasetId ? parseInt(selectedDatasetId) : undefined,
        pipeline: [], // 空流水线，只返回原图
        seed: previewSeed
      }, signal);

      if (response.success && response.data) {
        setCurrentImageBase64(response.data.original_image);
        setPreviewResult(response.data); // 同时设置previewResult，确保结构一致
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('加载原图失败:', error);
      }
    } finally {
      setIsPreviewLoading(false);
    }
  }, [currentImageIndex, selectedDatasetId, previewSeed]); // 移除 datasetImages 依赖，避免循环

  // 预览增强效果
  const handlePreview = async (signal?: AbortSignal) => {
    if (!draftStrategy || draftStrategy.pipeline.length === 0) {
      return; // 静默返回，不显示通知
    }

    // 使用当前选中的图片
    const currentImage = datasetImages[currentImageIndex];
    if (!currentImage) {
      return;
    }

    setIsPreviewLoading(true);
    try {
      const response = await previewAugmentation({
        image_path: currentImage.path,
        dataset_id: parseInt(selectedDatasetId), // 用于mosaic等多图算子
        pipeline: draftStrategy.pipeline,
        seed: previewSeed
      }, signal);

      if (response.success && response.data) {
        setPreviewResult(response.data);
      }
    } catch (error) {
      // 如果是主动取消的请求，不显示错误
      if ((error as Error).name !== 'AbortError') {
        showNotification('预览失败: ' + (error as Error).message, 'error');
      }
    } finally {
      setIsPreviewLoading(false);
    }
  };

  // 防抖预览函数
  const debouncedPreviewRef = useRef<NodeJS.Timeout | null>(null);

  const triggerPreview = useCallback(() => {
    // 清除之前的定时器
    if (debouncedPreviewRef.current) {
      clearTimeout(debouncedPreviewRef.current);
    }

    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // 设置新的防抖定时器
    debouncedPreviewRef.current = setTimeout(() => {
      const controller = new AbortController();
      abortControllerRef.current = controller;
      handlePreview(controller.signal);
    }, 800); // 800ms防抖延迟
  }, [draftStrategy?.pipeline, currentImageIndex, selectedDatasetId, previewSeed, datasetImages]);

  // 监听pipeline和图片索引变化，自动触发预览或加载原图
  useEffect(() => {
    if (datasetImages.length === 0 || view !== 'editor') {
      return;
    }

    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    if (draftStrategy && draftStrategy.pipeline.length > 0) {
      // 有流水线：触发预览
      triggerPreview();
    } else {
      // 无流水线：直接加载原图
      const controller = new AbortController();
      abortControllerRef.current = controller;
      loadCurrentImage(controller.signal);
    }

    // 清理函数
    return () => {
      if (debouncedPreviewRef.current) {
        clearTimeout(debouncedPreviewRef.current);
      }
    };
  }, [draftStrategy?.pipeline, currentImageIndex, selectedDatasetId, triggerPreview, datasetImages.length, view, loadCurrentImage]);

  // 当流水线为空时，重置预览按钮状态为"预览"
  useEffect(() => {
    if (!draftStrategy?.pipeline.length) {
      setShowOriginal(false);
    }
  }, [draftStrategy?.pipeline.length]);

  // 拖拽时分隔条时防止文字选中
  useEffect(() => {
    if (isDraggingSplitter) {
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'row-resize';
    } else {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }
    return () => {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDraggingSplitter]);

  // Helper to categorize operators
  const categories = Array.from(new Set(operators.map(op => op.category)));

  // Get selected pipeline item
  const selectedItem = draftStrategy?.pipeline.find(
    p => p.instanceId === selectedPipelineItemId
  );
  const selectedOperatorDef = operators.find(
    op => op.id === selectedItem?.operatorId
  );

  return (
    <>
      {/* GLOBAL NOTIFICATION COMPONENT */}
      {notification && (
        <div className={`fixed top-6 left-1/2 -translate-x-1/2 z-[200] px-4 py-2 rounded-lg shadow-lg border flex items-center ${
          notification.type === 'error'
            ? 'bg-rose-900/90 border-rose-500 text-white'
            : notification.type === 'success'
              ? 'bg-emerald-900/90 border-emerald-500 text-white'
              : 'bg-cyan-900/90 border-cyan-500 text-white'
        }`}>
          {notification.type === 'error' ? <AlertTriangle size={16} className="mr-2" /> :
            notification.type === 'success' ? <CheckCircle size={16} className="mr-2" /> :
              <Info size={16} className="mr-2" />
          }
          <span className="text-sm font-medium">{notification.msg}</span>
        </div>
      )}

      {/* Delete Confirmation Modal */}
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
                onClick={() => setDeleteModal({ isOpen: false, strategyId: null })}
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

      {/* Description Edit Modal */}
      {descriptionModal.isOpen && (
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-700 p-6 rounded-xl w-full max-w-md shadow-2xl animate-in fade-in zoom-in duration-200">
            <div className="flex items-center mb-4">
              <div className="w-10 h-10 rounded-lg bg-cyan-900/30 flex items-center justify-center text-cyan-500 mr-3">
                <FileText size={20} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">编辑策略描述</h3>
                <p className="text-xs text-slate-500">为当前策略添加详细说明</p>
              </div>
              <button
                onClick={() => setDescriptionModal({ isOpen: false })}
                className="ml-auto p-1.5 text-slate-500 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            <textarea
              autoFocus
              value={tempDescription}
              onChange={(e) => setTempDescription(e.target.value)}
              placeholder="请输入策略描述，例如：适用于YOLO训练的数据增强流水线，包含马赛克、水平翻转和颜色抖动..."
              className="w-full h-32 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-sm text-white placeholder-slate-500 outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 resize-none custom-scrollbar"
            />

            <div className="flex items-center justify-between mt-2 mb-4">
              <span className="text-xs text-slate-500">{tempDescription.length} 字符</span>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setTempDescription('');
                    setDescriptionModal({ isOpen: false });
                  }}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs rounded border border-slate-700 transition-colors"
                >
                  清空
                </button>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setDescriptionModal({ isOpen: false })}
                className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors border border-slate-700"
              >
                取消
              </button>
              <button
                onClick={() => {
                  setDescriptionModal({ isOpen: false });
                  showNotification('描述已更新', 'success');
                }}
                className="flex-1 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-bold transition-colors shadow-lg shadow-cyan-900/20"
              >
                <Check size={14} className="inline mr-1" /> 确认
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

          {isLoadingStrategies ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={32} className="animate-spin text-cyan-500" />
              <span className="ml-3 text-slate-400">加载策略中...</span>
            </div>
          ) : (
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
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-400 transition-colors pr-8">
                    {strat.name}
                  </h3>
                  <p className="text-sm text-slate-400 mb-6 line-clamp-2 h-10">
                    {strat.description || '暂无描述'}
                  </p>

                  <div className="flex items-center justify-between text-xs text-slate-500 border-t border-slate-800 pt-4">
                    <span className="flex items-center">
                      <Settings2 size={12} className="mr-1" /> {strat.pipeline.length} 算子
                    </span>
                    <span>更新于 {new Date(strat.updated_at || strat.created_at).toLocaleDateString()}</span>
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
          )}
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
                    onBlur={() => setIsEditingName(false)}
                    onKeyDown={(e) => e.key === 'Enter' && setIsEditingName(false)}
                    className="bg-slate-800 text-xl font-bold text-white px-2 py-0.5 rounded border border-cyan-500 outline-none w-96 shadow-[0_0_10px_rgba(6,182,212,0.3)]"
                    placeholder="策略名称"
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
              <div className="h-6 w-px bg-slate-800 mx-2"></div>
              <button
                onClick={() => setDescriptionModal({ isOpen: true })}
                className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all"
                title="编辑策略描述"
              >
                <FileText size={14} className="mr-2" /> 描述
              </button>
              <button
                onClick={() => handleSave(true)}
                disabled={isSaving}
                className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-bold rounded border border-slate-600 transition-all disabled:opacity-50"
                title="另存为新策略"
              >
                <Copy size={14} className="mr-2" /> 另存为
              </button>
              <button
                onClick={() => handleSave(false)}
                disabled={isSaving}
                className="flex items-center px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded shadow-lg shadow-cyan-900/20 transition-all disabled:opacity-50"
              >
                {isSaving ? <Loader2 size={14} className="mr-2 animate-spin" /> : <Save size={14} className="mr-2" />}
                保存
              </button>
            </div>
          </div>

          <div className="flex-1 flex overflow-hidden">
            {/* 1. Library (Left) */}
            <div className="w-72 bg-slate-950 border-r border-slate-800 flex flex-col z-10">
              <div className="p-4 border-b border-slate-800">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">算子库</h3>
              </div>
              <div className="flex-1 overflow-y-auto p-3 custom-scrollbar">
                {isLoadingOperators ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 size={24} className="animate-spin text-cyan-500" />
                  </div>
                ) : (
                  categories.map(cat => {
                    const categoryOperators = operators.filter(op => op.category === cat);
                    if (categoryOperators.length === 0) return null;

                    return (
                      <div key={cat} className="mb-6">
                        <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-2 px-2 flex items-center">
                          <span className="w-1 h-3 bg-cyan-500 rounded-full mr-2"></span>
                          {categoryOperators[0]?.category_label_zh || cat}
                        </h4>
                        <div className="space-y-1">
                          {categoryOperators.map(op => (
                            <div
                              key={op.id}
                              onClick={() => addOperator(op.id)}
                              className="px-3 py-2.5 rounded-lg text-xs text-slate-300 hover:bg-slate-800 hover:text-white cursor-pointer flex justify-between items-center group transition-all border border-transparent hover:border-slate-700"
                              title={op.description}
                            >
                              <span>{op.name_zh}</span>
                              <Plus size={14} className="opacity-0 group-hover:opacity-100 text-cyan-400 transition-opacity" />
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* 2. Pipeline (Middle) */}
            <div id="pipeline-column" className="w-80 bg-slate-900/30 border-r border-slate-800 flex flex-col z-10">
              <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">流水线</h3>
                <span className="text-xs text-slate-500 font-medium">
                  {draftStrategy?.pipeline.length || 0} 个算子
                </span>
              </div>

              {/* Stack List - 动态高度 */}
              <div
                className="overflow-y-auto p-3 space-y-2 custom-scrollbar"
                style={{ height: `calc(${pipelineSplitRatio * 100}% - 57px)` }}
              >
                {draftStrategy && draftStrategy.pipeline.length > 0 ? (
                  <DndContext
                    sensors={sensors}
                    collisionDetection={closestCenter}
                    onDragEnd={handleDragEnd}
                  >
                    <SortableContext
                      items={draftStrategy.pipeline.map(p => p.instanceId)}
                      strategy={verticalListSortingStrategy}
                    >
                      {draftStrategy.pipeline.map((item, idx) => {
                        const opDef = operators.find(op => op.id === item.operatorId);
                        const isSelected = selectedPipelineItemId === item.instanceId;
                        return (
                          <SortablePipelineItem
                            key={item.instanceId}
                            item={item}
                            index={idx}
                            operatorDef={opDef}
                            isSelected={isSelected}
                            onSelect={setSelectedPipelineItemId}
                            onRemove={removeOperator}
                          />
                        );
                      })}
                    </SortableContext>
                  </DndContext>
                ) : (
                  <div className="flex flex-col items-center justify-center h-40 text-slate-600 border-2 border-dashed border-slate-800 rounded-xl m-2">
                    <Plus size={24} className="mb-2 opacity-50" />
                    <span className="text-xs">从左侧添加算子</span>
                  </div>
                )}
              </div>

              {/* 可拖拽分隔条 */}
              <div
                onMouseDown={handleSplitterMouseDown}
                className={`h-1.5 bg-slate-800 hover:bg-cyan-600 cursor-row-resize transition-colors relative group ${
                  isDraggingSplitter ? 'bg-cyan-500' : ''
                }`}
              >
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className={`w-16 h-1 bg-slate-600 rounded-full transition-all ${
                    isDraggingSplitter ? 'bg-cyan-400 scale-125' : 'group-hover:bg-slate-500'
                  }`}></div>
                </div>
              </div>

              {/* Params Editor - 动态高度 */}
              <div
                className="border-t border-slate-800 bg-slate-950 flex flex-col shadow-[0_-5px_15px_rgba(0,0,0,0.2)]"
                style={{ height: `calc(${(1 - pipelineSplitRatio) * 100}% - 6px)` }}
              >
                <div className="p-3 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">参数配置</h3>
                  {selectedItem && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                      {selectedOperatorDef?.name_zh}
                    </span>
                  )}
                </div>
                <div className="flex-1 overflow-y-auto p-5 custom-scrollbar">
                  {selectedItem ? (
                    <div className="space-y-5">
                      {/* 启用开关 - 独立一行 */}
                      <div className="flex items-center pb-3 border-b border-slate-800">
                        <input
                          type="checkbox"
                          checked={selectedItem.enabled}
                          onChange={() => toggleOperator(selectedItem.instanceId)}
                          className="w-4 h-4 rounded border-slate-700 bg-slate-900 accent-cyan-500 focus:ring-0 focus:ring-offset-0"
                        />
                        <span className={`ml-2 text-sm font-medium ${selectedItem.enabled ? 'text-white' : 'text-slate-500'}`}>
                          启用算子
                        </span>
                      </div>

                      {/* 算子描述 - 独立一行 */}
                      {selectedOperatorDef && (
                        <div className="pb-3 border-b border-slate-800">
                          <p className="text-xs text-slate-500 leading-relaxed">
                            {selectedOperatorDef.description}
                          </p>
                        </div>
                      )}
                      {selectedOperatorDef?.params.map(param => (
                        <div key={param.name} className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span className="text-slate-400">{param.label_zh}</span>
                            <span className="text-cyan-400 font-mono">{selectedItem.params[param.name]}</span>
                          </div>
                          {param.type === 'range' ? (
                            <input
                              type="range"
                              min={param.min_value ?? 0}
                              max={param.max_value ?? 100}
                              step={param.step ?? 1}
                              value={selectedItem.params[param.name]}
                              onChange={(e) => updateParam(selectedItem.instanceId, param.name, parseFloat(e.target.value))}
                              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            />
                          ) : param.type === 'number' ? (
                            <NumberInput
                              value={selectedItem.params[param.name]}
                              onChange={(val) => updateParam(selectedItem.instanceId, param.name, val)}
                              min={param.min_value}
                              max={param.max_value}
                              step={param.step}
                              className="flex-1"
                            />
                          ) : (
                            <input
                              type="number"
                              value={selectedItem.params[param.name]}
                              onChange={(e) => updateParam(selectedItem.instanceId, param.name, parseFloat(e.target.value) || 0)}
                              className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-xs text-white outline-none focus:border-cyan-500 transition-colors"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-slate-600">
                      <Settings2 size={32} className="mb-2 opacity-20" />
                      <span className="text-xs">请选择流水线中的算子</span>
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
                  {/* Dataset Selector */}
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
                      {isLoadingDatasets ? (
                        <option>加载数据集中...</option>
                      ) : datasets.length === 0 ? (
                        <option>无可用数据集</option>
                      ) : (
                        datasets.map(d => (
                          <option key={d.id} value={d.id}>
                            {d.name} ({d.count}张)
                          </option>
                        ))
                      )}
                    </select>
                  </div>

                  <div className="h-10 w-px bg-slate-800"></div>

                  {/* Image Navigation */}
                  <div className="flex flex-col">
                    <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 group-hover:text-cyan-400 transition-colors">
                      <ImageIcon size={10} className="inline mr-1" />
                      图片导航
                    </label>
                    <div className="flex items-center space-x-1">
                      {/* 上一张 */}
                      <button
                        onClick={() => setCurrentImageIndex(Math.max(0, currentImageIndex - 1))}
                        disabled={currentImageIndex <= 0 || isLoadingImages}
                        className="flex-1 flex flex-col items-center justify-center bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 hover:border-cyan-500 hover:bg-slate-800/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed group min-w-[60px]"
                        title="上一张"
                      >
                        <ChevronLeft size={16} className="mb-0.5 group-hover:text-cyan-400 transition-colors" />
                        <span className="text-[10px] text-slate-400 group-hover:text-cyan-400 transition-colors">上一张</span>
                      </button>

                      {/* 图片计数 */}
                      <div className="flex flex-col items-center justify-center bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 min-w-[80px]">
                        <span className="text-sm font-bold text-white">
                          {isLoadingImages ? '...' : `${currentImageIndex + 1}`}
                        </span>
                        <span className="text-[10px] text-slate-500">
                          / {datasetImages.length}
                        </span>
                      </div>

                      {/* 下一张 */}
                      <button
                        onClick={() => setCurrentImageIndex(Math.min(datasetImages.length - 1, currentImageIndex + 1))}
                        disabled={currentImageIndex >= datasetImages.length - 1 || isLoadingImages}
                        className="flex-1 flex flex-col items-center justify-center bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 hover:border-cyan-500 hover:bg-slate-800/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed group min-w-[60px]"
                        title="下一张"
                      >
                        <ChevronRight size={16} className="mb-0.5 group-hover:text-cyan-400 transition-colors" />
                        <span className="text-[10px] text-slate-400 group-hover:text-cyan-400 transition-colors">下一张</span>
                      </button>

                      {/* 随机图片 */}
                      <button
                        onClick={() => {
                          if (datasetImages.length > 0) {
                            const randomIndex = Math.floor(Math.random() * datasetImages.length);
                            setCurrentImageIndex(randomIndex);
                          }
                        }}
                        disabled={isLoadingImages || datasetImages.length === 0}
                        className="flex flex-col items-center justify-center bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 hover:border-cyan-500 hover:bg-slate-800/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed group min-w-[60px]"
                        title="随机图片"
                      >
                        <Shuffle size={14} className="mb-0.5 group-hover:text-cyan-400 transition-colors" />
                        <span className="text-[10px] text-slate-400 group-hover:text-cyan-400 transition-colors">随机</span>
                      </button>
                    </div>
                  </div>

                  <div className="h-10 w-px bg-slate-800"></div>

                  {/* Preview Control */}
                  <div className="flex flex-col">
                    <label className="text-[10px] text-slate-500 uppercase font-bold mb-1.5 group-hover:text-cyan-400 transition-colors">
                      <Wand2 size={10} className="inline mr-1" />
                      预览控制
                    </label>
                    <div className="flex items-center space-x-1">
                      {/* 原图/预览切换按钮 */}
                      <button
                        onClick={() => setShowOriginal(!showOriginal)}
                        className={`flex flex-col items-center justify-center border rounded-lg px-3 py-2 transition-all min-w-[70px] ${
                          showOriginal
                            ? 'bg-cyan-600 border-cyan-500 text-white shadow-lg shadow-cyan-900/20'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-cyan-500 hover:bg-slate-800/80'
                        }`}
                        title={showOriginal ? '点击显示预览图' : '点击显示原图'}
                      >
                        {showOriginal ? (
                          <>
                            <Check size={12} className="mb-0.5" />
                            <span className="text-[10px]">原图</span>
                          </>
                        ) : (
                          <>
                            <Wand2 size={12} className="mb-0.5" />
                            <span className="text-[10px]">预览</span>
                          </>
                        )}
                      </button>

                      {/* 手动刷新 */}
                      <button
                        onClick={() => handlePreview()}
                        disabled={isPreviewLoading}
                        className="flex flex-col items-center justify-center bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 hover:border-cyan-500 hover:bg-slate-800/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed group min-w-[60px]"
                        title="手动刷新预览"
                      >
                        {isPreviewLoading ? (
                          <Loader2 size={14} className="mb-0.5 animate-spin group-hover:text-cyan-400 transition-colors" />
                        ) : (
                          <RefreshCw size={14} className="mb-0.5 group-hover:text-cyan-400 transition-colors" />
                        )}
                        <span className="text-[10px] text-slate-400 group-hover:text-cyan-400 transition-colors">刷新</span>
                      </button>
                    </div>
                  </div>
                </div>

                <div className="flex items-center">
                  {isPreviewLoading && (
                    <div className="flex items-center text-xs text-cyan-400">
                      <Loader2 size={12} className="mr-1 animate-spin" />
                      预览中...
                    </div>
                  )}
                </div>
              </div>

              {/* Canvas Area */}
              <div className="flex-1 relative flex flex-col items-center justify-center p-10 bg-grid-pattern bg-slate-950 overflow-hidden">
                {previewResult ? (
                  <>
                    {/* 单图显示 */}
                    <div className="relative">
                      <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur text-[10px] rounded border border-slate-700 z-10">
                        {/* 流水线为空时始终显示"原图"，否则根据showOriginal状态 */}
                        {!draftStrategy?.pipeline.length || showOriginal ? '原图' : '增强后'}
                      </div>
                      <img
                        src={`data:image/jpeg;base64,${!draftStrategy?.pipeline.length || showOriginal ? previewResult.original_image : previewResult.augmented_images[0]?.augmented_data}`}
                        className="max-w-[600px] max-h-[500px] object-contain rounded-lg border border-slate-800"
                        alt="Preview"
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center text-slate-600">
                    {isPreviewLoading ? (
                      <>
                        <Loader2 size={48} className="mb-4 animate-spin text-cyan-500" />
                        <p className="text-sm">正在生成预览...</p>
                      </>
                    ) : draftStrategy && draftStrategy.pipeline.length > 0 ? (
                      <>
                        <Wand2 size={48} className="mb-4 opacity-20" />
                        <p className="text-sm">
                          {datasetImages.length > 0
                            ? '调整参数后预览将自动更新'
                            : '选择数据集以开始预览'}
                        </p>
                      </>
                    ) : (
                      <>
                        <Wand2 size={48} className="mb-4 opacity-20" />
                        <p className="text-sm">添加算子后将自动预览增强效果</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
