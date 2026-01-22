/**
 * 训练配置展示组件（完整版）
 *
 * 以弹窗形式展示完整的训练配置信息，包括：
 * - 源训练信息
 * - 数据集信息
 * - 模型架构信息
 * - 预训练权重信息
 * - 数据增强配置
 * - 超参数配置
 */

import React, { useMemo } from 'react';
import {
  X, Settings, Database, Cpu, Activity,
  Layers, Box, GitBranch, CheckCircle, XCircle
} from 'lucide-react';
import { WeightTrainingConfig, DatasetInfo, ModelArchitectureInfo } from '../src/services/weights';

// 算子ID到中文名称的映射（与后端 augmentation_registry.py 保持一致）
const OPERATOR_NAMES: Record<string, string> = {
  // 几何变换
  'horizontal_flip': '水平翻转',
  'vertical_flip': '垂直翻转',
  'rotate': '随机旋转',
  'scale': '随机缩放',
  'translate': '随机平移',
  'shift_scale_rotate': '综合几何变换',
  'random_resized_crop': '随机裁剪缩放',
  'perspective': '透视变换',
  'shear': '剪切变换',
  'crop': '固定裁剪',
  'elastic_transform': '弹性变换',
  // 颜色变换
  'hsv_color': 'HSV颜色调整',
  'brightness': '亮度调整',
  'contrast': '对比度调整',
  'saturation': '饱和度调整',
  'hue_shift': '色调偏移',
  'rgb_shift': 'RGB通道偏移',
  'to_gray': '转灰度',
  'gamma_correction': 'Gamma校正',
  'auto_contrast': '自动对比度',
  // 模糊与噪声
  'gaussian_blur': '高斯模糊',
  'motion_blur': '运动模糊',
  'gaussian_noise': '高斯噪声',
  'salt_pepper_noise': '椒盐噪声',
  // 其他
  'random_erase': '随机擦除',
  'jpeg_compression': 'JPEG压缩',
  'mosaic': '马赛克增强',
  'copy_paste': 'Copy-Paste增强'
};

/**
 * 获取算子显示名称
 */
function getOperatorDisplayName(item: any, idx: number): string {
  if (item.name) return item.name;
  if (item.type) return item.type;
  if (item.operatorId) return OPERATOR_NAMES[item.operatorId] || item.operatorId;
  return `算子 ${idx + 1}`;
}

/**
 * 训练配置详情类型（来自训练任务API）
 */
interface TrainingConfigDetail {
  id: number;
  name: string;
  description?: string;
  status: string;
  created_at: string;
  hyperparams: Record<string, any>;
  dataset?: DatasetInfo | null;
  model_architecture?: ModelArchitectureInfo | null;
  pretrained_weight?: any;
  augmentation?: any;
}

// 联合类型，支持两种数据格式
type ConfigData = WeightTrainingConfig | TrainingConfigDetail | null;

interface TrainingConfigViewProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  config: ConfigData;
  loading?: boolean;
}

/**
 * 配置类别定义
 */
const PARAM_CATEGORIES = {
  hyperparams: {
    label: '超参数',
    icon: Activity,
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-900/20',
    fields: {
      learning_rate: '学习率',
      batch_size: '批大小',
      epochs: '训练轮数',
      optimizer: '优化器',
      weight_decay: '权重衰减',
      momentum: '动量',
      lr_scheduler: '学习率调度',
      warmup_epochs: '预热轮数',
      gradient_clip: '梯度裁剪',
      label_smoothing: '标签平滑',
    }
  },
  dataset: {
    label: '数据集',
    icon: Database,
    color: 'text-purple-400',
    bgColor: 'bg-purple-900/20',
    fields: {
      train_split: '训练集比例',
      val_split: '验证集比例',
      shuffle: '数据洗牌',
    }
  },
  architecture: {
    label: '模型架构',
    icon: Cpu,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-900/20',
    fields: {
      freeze_backbone: '冻结骨干',
      freeze_epochs: '冻结轮数',
    }
  }
};

/**
 * 渲染参数值
 */
function renderValue(value: any): React.ReactNode {
  if (value === null || value === undefined) {
    return <span className="text-slate-600">-</span>;
  }
  if (typeof value === 'boolean') {
    return value ? (
      <CheckCircle size={14} className="text-emerald-400" />
    ) : (
      <XCircle size={14} className="text-rose-400" />
    );
  }
  if (typeof value === 'object') {
    return (
      <span className="text-slate-400 text-xs">
        {JSON.stringify(value)}
      </span>
    );
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      return <span>{value}</span>;
    }
    if (value < 0.01 || value > 1000) {
      return <span>{value.toExponential(2)}</span>;
    }
    return <span>{value.toFixed(4)}</span>;
  }
  return <span>{String(value)}</span>;
}

/**
 * 获取参数显示标签
 */
function getParamLabel(key: string, category: keyof typeof PARAM_CATEGORIES): string {
  return PARAM_CATEGORIES[category].fields[key as keyof typeof PARAM_CATEGORIES.hyperparams.fields] || key;
}

/**
 * 判断参数属于哪个类别
 */
function getCategoryForKey(key: string): string | null {
  for (const [categoryKey, category] of Object.entries(PARAM_CATEGORIES)) {
    if (category.fields[key as keyof typeof category.fields]) {
      return categoryKey;
    }
  }
  return null;
}

/**
 * 状态徽章组件
 */
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const statusConfig: Record<string, { color: string; label: string }> = {
    completed: { color: 'bg-emerald-900/40 text-emerald-400', label: '已完成' },
    running: { color: 'bg-cyan-900/40 text-cyan-400', label: '运行中' },
    failed: { color: 'bg-rose-900/40 text-rose-400', label: '失败' },
    stopped: { color: 'bg-slate-800 text-slate-400', label: '已停止' },
    pending: { color: 'bg-amber-900/40 text-amber-400', label: '等待中' },
    queued: { color: 'bg-blue-900/40 text-blue-400', label: '队列中' },
    paused: { color: 'bg-yellow-900/40 text-yellow-400', label: '已暂停' },
  };

  const config = statusConfig[status] || { color: 'bg-slate-800 text-slate-400', label: status };

  return (
    <span className={`px-3 py-1 text-xs rounded-full ${config.color}`}>
      {config.label}
    </span>
  );
};

/**
 * 信息项组件
 */
const InfoItem: React.FC<{ label: string; value: any }> = ({ label, value }) => (
  <div className="flex items-center justify-between">
    <span className="text-sm text-slate-400">{label}:</span>
    <span className="text-sm font-medium text-slate-200">{renderValue(value)}</span>
  </div>
);

/**
 * 源训练信息卡片
 */
const SourceTrainingCard: React.FC<{ training: any }> = ({ training }) => (
  <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800">
    <div className="flex items-center justify-between">
      <div>
        <div className="text-sm font-medium text-slate-300">{training.name}</div>
        <div className="text-xs text-slate-600">
          任务ID: {training.id}
          {training.created_at && ` • 创建于 ${new Date(training.created_at).toLocaleString()}`}
        </div>
      </div>
      {training.status && <StatusBadge status={training.status} />}
    </div>
  </div>
);

/**
 * 数据集信息卡片
 */
const DatasetInfoCard: React.FC<{ dataset: DatasetInfo }> = ({ dataset }) => (
  <div>
    <div className="flex items-center gap-2 mb-3 text-purple-400">
      <Database size={16} />
      <span className="text-sm font-bold uppercase">数据集</span>
    </div>
    <div className="p-4 bg-purple-900/20 rounded-lg border border-slate-800">
      <div className="grid grid-cols-2 gap-3">
        <InfoItem label="数据集名称" value={dataset.name} />
        <InfoItem label="格式" value={dataset.format.toUpperCase()} />
        <InfoItem label="图片数量" value={dataset.num_images?.toLocaleString()} />
        <InfoItem label="类别数量" value={dataset.num_classes} />
        {dataset.path && (
          <div className="col-span-2">
            <span className="text-sm text-slate-400">路径:</span>
            <p className="text-xs text-slate-500 mt-1 break-all">{dataset.path}</p>
          </div>
        )}
        {dataset.classes && dataset.classes.length > 0 && (
          <div className="col-span-2">
            <span className="text-sm text-slate-400">类别列表:</span>
            <div className="flex flex-wrap gap-1 mt-2">
              {dataset.classes.map((cls, idx) => (
                <span key={idx} className="px-2 py-0.5 bg-purple-900/40 text-purple-300 text-xs rounded">
                  {cls}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  </div>
);

/**
 * 模型架构信息卡片
 */
const ModelArchitectureCard: React.FC<{ model: ModelArchitectureInfo }> = ({ model }) => (
  <div>
    <div className="flex items-center gap-2 mb-3 text-emerald-400">
      <Cpu size={16} />
      <span className="text-sm font-bold uppercase">模型架构</span>
    </div>
    <div className="p-4 bg-emerald-900/20 rounded-lg border border-slate-800">
      <div className="grid grid-cols-2 gap-3">
        <InfoItem label="模型名称" value={model.name} />
        {model.input_size && (
          <InfoItem label="输入尺寸" value={`[${model.input_size.join(', ')}]`} />
        )}
        {model.task_type && (
          <InfoItem label="任务类型" value={model.task_type} />
        )}
        {model.description && (
          <div className="col-span-2">
            <span className="text-sm text-slate-400">描述:</span>
            <p className="text-sm text-slate-300 mt-1">{model.description}</p>
          </div>
        )}
        {model.file_path && (
          <div className="col-span-2">
            <span className="text-sm text-slate-400">文件路径:</span>
            <p className="text-xs text-slate-500 mt-1 break-all">{model.file_path}</p>
          </div>
        )}
      </div>
    </div>
  </div>
);

/**
 * 预训练权重信息卡片
 */
const PretrainedWeightCard: React.FC<{ weight: any }> = ({ weight }) => (
  <div>
    <div className="flex items-center gap-2 mb-3 text-orange-400">
      <Box size={16} />
      <span className="text-sm font-bold uppercase">预训练权重</span>
    </div>
    <div className="p-4 bg-orange-900/20 rounded-lg border border-slate-800">
      <div className="grid grid-cols-2 gap-3">
        <InfoItem label="权重名称" value={weight.name} />
        <InfoItem label="版本" value={weight.version} />
        <InfoItem label="任务类型" value={weight.task_type} />
        <InfoItem label="来源" value={weight.source_type === 'trained' ? '训练生成' : '用户上传'} />
      </div>
    </div>
  </div>
);

/**
 * 数据增强配置卡片
 */
const AugmentationCard: React.FC<{ augmentation: any }> = ({ augmentation }) => {
  // 渲染 pipeline 配置为友好的列表
  const renderPipelineConfig = (pipeline: any) => {
    if (!pipeline) return null;

    // 如果是数组，显示为增强算子列表
    if (Array.isArray(pipeline)) {
      return (
        <div className="space-y-2">
          <span className="text-xs text-slate-500">增强算子 ({pipeline.length}):</span>
          <div className="grid grid-cols-2 gap-2">
            {pipeline.map((item: any, idx: number) => (
              <div key={idx} className="p-2 bg-slate-900/50 rounded border border-slate-800">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium text-pink-300">
                    {getOperatorDisplayName(item, idx)}
                  </span>
                  <span className={`px-1.5 py-0.5 text-[10px] rounded ${
                    item.enabled !== false
                      ? 'bg-emerald-900/40 text-emerald-400'
                      : 'bg-slate-800 text-slate-500'
                  }`}>
                    {item.enabled !== false ? '启用' : '禁用'}
                  </span>
                </div>
                {item.probability !== undefined && (
                  <div className="text-[10px] text-slate-500">
                    概率: {(item.probability * 100).toFixed(0)}%
                  </div>
                )}
                {item.params && Object.keys(item.params).length > 0 && (
                  <div className="mt-1 text-[10px] text-slate-600">
                    {Object.entries(item.params).slice(0, 3).map(([k, v]) => (
                      <div key={k}>{k}: {String(v)}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    // 如果是对象，显示为键值对
    if (typeof pipeline === 'object') {
      return (
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(pipeline).map(([key, value]) => (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-slate-500">{key}:</span>
              <span className="text-slate-300">{String(value)}</span>
            </div>
          ))}
        </div>
      );
    }

    // 其他情况，显示原始值
    return <pre className="text-xs text-slate-400 font-mono overflow-x-auto">{JSON.stringify(pipeline, null, 2)}</pre>;
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 text-pink-400">
        <Layers size={16} />
        <span className="text-sm font-bold uppercase">数据增强</span>
        <span className={`px-2 py-0.5 rounded-full text-xs ${
          augmentation.enabled
            ? 'bg-emerald-900/40 text-emerald-400'
            : 'bg-slate-800 text-slate-500'
        }`}>
          {augmentation.enabled ? '已启用' : '未启用'}
        </span>
      </div>
      {augmentation.enabled && augmentation.config && (
        <div className="p-4 bg-pink-900/20 rounded-lg border border-slate-800">
          {augmentation.strategy && (
            <div className="mb-3">
              <span className="text-sm text-slate-400">策略名称: </span>
              <span className="text-sm text-slate-200">{augmentation.strategy}</span>
              {augmentation.description && (
                <p className="text-xs text-slate-500 mt-1">{augmentation.description}</p>
              )}
            </div>
          )}
          <div className="mt-3">
            {renderPipelineConfig(augmentation.config)}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * 超参数配置卡片
 */
const HyperparamsCard: React.FC<{ hyperparams: Record<string, any> }> = ({ hyperparams }) => {
  // 按类别分组参数
  const categorizedParams: Record<string, Record<string, any>> = {
    hyperparams: {},
    dataset: {},
    architecture: {},
    other: {}
  };

  Object.entries(hyperparams).forEach(([key, value]) => {
    // 跳过已在其他卡片显示的信息
    if (['augmentation', 'dataset_id', 'model_id', 'pretrained_weight_id', 'task_type', 'device'].includes(key)) {
      return;
    }

    const category = getCategoryForKey(key);
    if (category) {
      categorizedParams[category][key] = value;
    } else {
      categorizedParams.other[key] = value;
    }
  });

  return (
    <div className="space-y-4">
      {/* 超参数分组 */}
      {(Object.entries(PARAM_CATEGORIES) as [keyof typeof PARAM_CATEGORIES, any][]).map(([catKey, category]) => {
        const params = categorizedParams[catKey];
        if (!params || Object.keys(params).length === 0) return null;

        const Icon = category.icon;

        return (
          <div key={catKey}>
            <div className={`flex items-center gap-2 mb-3 ${category.color}`}>
              <Icon size={16} />
              <span className="text-sm font-bold uppercase">{category.label}</span>
              <span className={`px-2 py-0.5 ${category.bgColor} ${category.color} text-xs rounded-full`}>
                {Object.keys(params).length}
              </span>
            </div>
            <div className={`p-4 ${category.bgColor} rounded-lg border border-slate-800`}>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(params).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-sm text-slate-400">
                      {getParamLabel(key, catKey)}:
                    </span>
                    <span className="text-sm font-medium text-slate-200">
                      {renderValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}

      {/* 其他参数 */}
      {Object.keys(categorizedParams.other).length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3 text-slate-400">
            <Settings size={16} />
            <span className="text-sm font-bold uppercase">其他参数</span>
            <span className="px-2 py-0.5 bg-slate-800 text-slate-400 text-xs rounded-full">
              {Object.keys(categorizedParams.other).length}
            </span>
          </div>
          <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-800">
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(categorizedParams.other).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-sm text-slate-500">{key}:</span>
                  <span className="text-sm text-slate-300">{renderValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * 主组件
 */
const TrainingConfigView: React.FC<TrainingConfigViewProps> = ({
  isOpen,
  onClose,
  title = '训练配置',
  config,
  loading = false
}) => {
  // 数据适配：支持两种不同的数据格式
  const adaptedConfig = useMemo(() => {
    if (!config) return null;

    // 如果是 WeightTrainingConfig 类型（有 training_config 字段）
    if ('training_config' in config) {
      return config as WeightTrainingConfig;
    }

    // 如果是 TrainingConfigDetail 类型（有 hyperparams 字段）
    if ('hyperparams' in config) {
      const detail = config as TrainingConfigDetail;
      return {
        weight_id: detail.id,
        weight_name: detail.name,
        training_config: detail.hyperparams || {},
        source_training: {
          id: detail.id,
          name: detail.name,
          status: detail.status,
          created_at: detail.created_at
        },
        dataset: detail.dataset,
        model_architecture: detail.model_architecture,
        pretrained_weight: detail.pretrained_weight,
        augmentation: detail.augmentation
      } as WeightTrainingConfig;
    }

    return config as WeightTrainingConfig;
  }, [config]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 背景遮罩 */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 弹窗内容 */}
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
        {/* 标题栏 */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-900/30 rounded-lg">
              <Settings size={18} className="text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">{title}</h2>
              <p className="text-xs text-slate-500">
                {adaptedConfig?.weight_name || adaptedConfig?.source_training?.name || '配置详情'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        {/* 内容区域 */}
        <div className="p-6 overflow-y-auto max-h-[calc(85vh-80px)]">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mr-3" />
              <span>加载训练配置...</span>
            </div>
          ) : !adaptedConfig ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
              <Settings size={48} className="mb-4 text-slate-700" />
              <p>没有可用的训练配置信息</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* 源训练信息 */}
              {adaptedConfig.source_training && (
                <SourceTrainingCard training={adaptedConfig.source_training} />
              )}

              {/* 数据集信息 */}
              {adaptedConfig.dataset && (
                <DatasetInfoCard dataset={adaptedConfig.dataset} />
              )}

              {/* 模型架构信息 */}
              {adaptedConfig.model_architecture && (
                <ModelArchitectureCard model={adaptedConfig.model_architecture} />
              )}

              {/* 预训练权重信息 */}
              {adaptedConfig.pretrained_weight && (
                <PretrainedWeightCard weight={adaptedConfig.pretrained_weight} />
              )}

              {/* 数据增强配置 */}
              {adaptedConfig.augmentation && (
                <AugmentationCard augmentation={adaptedConfig.augmentation} />
              )}

              {/* 超参数配置 */}
              {adaptedConfig.training_config && Object.keys(adaptedConfig.training_config).length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3 text-cyan-400">
                    <Activity size={16} />
                    <span className="text-sm font-bold uppercase">超参数配置</span>
                  </div>
                  <HyperparamsCard hyperparams={adaptedConfig.training_config} />
                </div>
              )}

              {/* 没有配置信息时的提示 */}
              {!adaptedConfig.dataset && !adaptedConfig.model_architecture && !adaptedConfig.pretrained_weight &&
               (!adaptedConfig.training_config || Object.keys(adaptedConfig.training_config).length === 0) && (
                <div className="flex flex-col items-center justify-center py-8 text-slate-500">
                  <Settings size={32} className="mb-3 text-slate-700" />
                  <p className="text-sm">此权重没有详细的训练配置信息</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 底部按钮 */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white text-sm rounded-lg transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
};

export default TrainingConfigView;
