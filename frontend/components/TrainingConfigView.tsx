/**
 * 训练配置展示组件
 *
 * 以弹窗形式展示单个权重的完整训练配置信息
 */

import React from 'react';
import { X, Settings, Database, Cpu, Activity } from 'lucide-react';
import { WeightTrainingConfig } from '../src/services/weights';
import { WeightTreeSelectOption } from './WeightTreeSelect';

interface TrainingConfigViewProps {
  isOpen: boolean;
  onClose: () => void;
  weight: WeightTreeSelectOption | null;
  config: WeightTrainingConfig | null;
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
      dataset_id: '数据集ID',
      train_split: '训练集比例',
      val_split: '验证集比例',
      augmentation: '数据增强',
      shuffle: '数据洗牌',
    }
  },
  architecture: {
    label: '模型架构',
    icon: Cpu,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-900/20',
    fields: {
      architecture_id: '架构ID',
      input_size: '输入尺寸',
      freeze_backbone: '冻结骨干',
      freeze_epochs: '冻结轮数',
      num_classes: '类别数量',
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
    return (
      <span className={value ? 'text-emerald-400' : 'text-rose-400'}>
        {value ? '是' : '否'}
      </span>
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
    // 格式化数字，保留合适的小数位
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
  return PARAM_CATEGORIES[category].fields[key] || key;
}

/**
 * 判断参数属于哪个类别
 */
function getCategoryForKey(key: string): keyof typeof PARAM_CATEGORIES | null {
  for (const [categoryKey, category] of Object.entries(PARAM_CATEGORIES)) {
    if (category.fields[key as keyof typeof category.fields]) {
      return categoryKey as keyof typeof PARAM_CATEGORIES;
    }
  }
  return null;
}

const TrainingConfigView: React.FC<TrainingConfigViewProps> = ({
  isOpen,
  onClose,
  weight,
  config,
  loading = false
}) => {
  if (!isOpen) return null;

  const trainingConfig = config?.training_config || {};

  // 按类别分组参数
  const categorizedParams: Record<string, Record<string, any>> = {
    hyperparams: {},
    dataset: {},
    architecture: {},
    other: {}
  };

  Object.entries(trainingConfig).forEach(([key, value]) => {
    const category = getCategoryForKey(key);
    if (category) {
      categorizedParams[category][key] = value;
    } else {
      categorizedParams.other[key] = value;
    }
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 背景遮罩 */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 弹窗内容 */}
      <div className="relative w-full max-w-2xl max-h-[80vh] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
        {/* 标题栏 */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-900/30 rounded-lg">
              <Settings size={18} className="text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">训练配置</h2>
              <p className="text-xs text-slate-500">
                {weight ? `${weight.display_name} (v${weight.version})` : '加载中...'}
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
        <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)]">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mr-3" />
              <span>加载训练配置...</span>
            </div>
          ) : !trainingConfig || Object.keys(trainingConfig).length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
              <Settings size={48} className="mb-4 text-slate-700" />
              <p>此权重没有训练配置信息</p>
              {weight?.source_type === 'uploaded' && (
                <p className="text-sm mt-2 text-slate-600">
                  导入的权重不包含训练配置
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {/* 源训练信息 */}
              {config?.source_training && (
                <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800">
                  <div className="text-xs text-slate-500 mb-2">源训练任务</div>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-slate-300">
                        {config.source_training.name}
                      </div>
                      <div className="text-xs text-slate-600">
                        任务ID: {config.source_training.id}
                      </div>
                    </div>
                    <div className="px-3 py-1 bg-purple-900/30 text-purple-400 text-xs rounded-full">
                      训练生成
                    </div>
                  </div>
                </div>
              )}

              {/* 分组显示配置 */}
              {(Object.entries(PARAM_CATEGORIES) as [keyof typeof PARAM_CATEGORIES, any][]).map(([categoryKey, category]) => {
                const params = categorizedParams[categoryKey];
                const hasParams = params && Object.keys(params).length > 0;

                if (!hasParams) return null;

                const Icon = category.icon;

                return (
                  <div key={categoryKey}>
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
                            <span className="text-sm text-slate-400">{getParamLabel(key, categoryKey)}:</span>
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
              {categorizedParams.other && Object.keys(categorizedParams.other).length > 0 && (
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
                          <span className="text-sm text-slate-300">
                            {renderValue(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
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
