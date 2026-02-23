/**
 * 训练配置对比视图组件
 *
 * 用于对比父子节点之间的训练参数差异
 */

import React, { useMemo } from 'react';
import { ArrowRight, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { WeightTreeItem, WeightTrainingConfig, ConfigDiffItem } from '../../types';

interface TrainingConfigDiffViewProps {
  parentWeight: WeightTreeItem | null;
  currentWeight: WeightTreeItem;
  parentConfig: WeightTrainingConfig | null;
  currentConfig: WeightTrainingConfig | null;
  loading?: boolean;
}

/**
 * 定义需要对比的参数
 */
const PARAM_DEFINITIONS = [
  // 超参数
  { key: 'learning_rate', label: '学习率', category: 'hyperparams' as const },
  { key: 'batch_size', label: '批大小', category: 'hyperparams' as const },
  { key: 'epochs', label: '训练轮数', category: 'hyperparams' as const },
  { key: 'optimizer', label: '优化器', category: 'hyperparams' as const },
  { key: 'weight_decay', label: '权重衰减', category: 'hyperparams' as const },
  { key: 'momentum', label: '动量', category: 'hyperparams' as const },
  { key: 'lr_scheduler', label: '学习率调度', category: 'hyperparams' as const },
  // 数据集
  { key: 'dataset_id', label: '数据集', category: 'dataset' as const },
  { key: 'train_split', label: '训练集比例', category: 'dataset' as const },
  { key: 'augmentation', label: '数据增强', category: 'dataset' as const },
  // 架构
  { key: 'architecture_id', label: '模型架构', category: 'architecture' as const },
  { key: 'input_size', label: '输入尺寸', category: 'architecture' as const },
  { key: 'freeze_backbone', label: '冻结骨干', category: 'architecture' as const },
];

/**
 * 计算配置差异
 */
function computeConfigDiff(
  parentConfig: WeightTrainingConfig | null,
  currentConfig: WeightTrainingConfig | null
): ConfigDiffItem[] {
  const diffItems: ConfigDiffItem[] = [];

  const parentParams = parentConfig?.training_config || {};
  const currentParams = currentConfig?.training_config || {};

  PARAM_DEFINITIONS.forEach(param => {
    const parentValue = parentParams[param.key];
    const currentValue = currentParams[param.key];
    const isChanged = JSON.stringify(parentValue) !== JSON.stringify(currentValue);

    diffItems.push({
      key: param.key,
      label: param.label,
      parentValue,
      currentValue,
      isChanged,
      category: param.category
    });
  });

  // 添加额外存在的参数
  const allKeys = new Set([
    ...Object.keys(parentParams),
    ...Object.keys(currentParams)
  ]);
  const definedKeys = new Set(PARAM_DEFINITIONS.map(p => p.key));

  allKeys.forEach(key => {
    if (!definedKeys.has(key)) {
      diffItems.push({
        key,
        label: key,
        parentValue: parentParams[key],
        currentValue: currentParams[key],
        isChanged: JSON.stringify(parentParams[key]) !== JSON.stringify(currentParams[key]),
        category: 'hyperparams'
      });
    }
  });

  return diffItems;
}

/**
 * 渲染参数值
 */
function renderValue(value: any): React.ReactNode {
  if (value === null || value === undefined) {
    return <span className="text-slate-600">-</span>;
  }
  if (typeof value === 'boolean') {
    return <span className={value ? 'text-emerald-400' : 'text-rose-400'}>{value ? '是' : '否'}</span>;
  }
  if (typeof value === 'object') {
    return <span className="text-slate-400 text-xs">{JSON.stringify(value)}</span>;
  }
  return <span>{String(value)}</span>;
}

/**
 * 判断值的增减方向
 */
function getValueDirection(parentValue: any, currentValue: any): 'increase' | 'decrease' | 'same' | 'changed' {
  if (parentValue === null || parentValue === undefined) return 'increase';
  if (currentValue === null || currentValue === undefined) return 'decrease';

  const parentNum = typeof parentValue === 'number' ? parentValue : parseFloat(parentValue);
  const currentNum = typeof currentValue === 'number' ? currentValue : parseFloat(currentValue);

  if (!isNaN(parentNum) && !isNaN(currentNum)) {
    if (currentNum > parentNum) return 'increase';
    if (currentNum < parentNum) return 'decrease';
    return 'same';
  }

  return parentValue === currentValue ? 'same' : 'changed';
}

const TrainingConfigDiffView: React.FC<TrainingConfigDiffViewProps> = ({
  parentWeight,
  currentWeight,
  parentConfig,
  currentConfig,
  loading = false
}) => {
  const diffItems = useMemo(
    () => computeConfigDiff(parentConfig, currentConfig),
    [parentConfig, currentConfig]
  );

  // 按类别分组
  const groupedItems = {
    hyperparams: diffItems.filter(item => item.category === 'hyperparams'),
    dataset: diffItems.filter(item => item.category === 'dataset'),
    architecture: diffItems.filter(item => item.category === 'architecture')
  };

  const hasChanges = diffItems.some(item => item.isChanged);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8 text-slate-500">
        <span>加载配置对比...</span>
      </div>
    );
  }

  // 如果没有父节点或父节点没有配置
  if (!parentWeight || !parentConfig?.training_config) {
    return (
      <div className="p-4 text-center text-slate-500">
        <p className="mb-2">此节点的父节点没有训练配置</p>
        {currentConfig?.training_config && (
          <div className="mt-4 p-3 bg-slate-950/50 rounded border border-slate-800">
            <div className="text-xs text-slate-500 mb-2">当前节点配置</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(currentConfig.training_config).slice(0, 6).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-slate-500">{key}:</span>
                  <span className="text-slate-300">{renderValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 父子节点信息 */}
      <div className="flex items-center gap-2 p-3 bg-slate-950/30 rounded-lg border border-slate-800">
        <div className="flex-1 text-center">
          <div className="text-xs text-slate-500 mb-1">父节点</div>
          <div className="text-sm font-medium text-slate-300 truncate">{parentWeight.display_name}</div>
          <div className="text-xs text-slate-600">v{parentWeight.version}</div>
        </div>
        <ArrowRight size={20} className="text-cyan-500 flex-shrink-0" />
        <div className="flex-1 text-center">
          <div className="text-xs text-slate-500 mb-1">当前节点</div>
          <div className="text-sm font-medium text-cyan-400 truncate">{currentWeight.display_name}</div>
          <div className="text-xs text-slate-600">v{currentWeight.version}</div>
        </div>
      </div>

      {/* 差异统计 */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-500">参数差异</span>
        <div className="flex items-center gap-4">
          {hasChanges ? (
            <span className="text-cyan-400">有 {diffItems.filter(i => i.isChanged).length} 项变更</span>
          ) : (
            <span className="text-emerald-400">配置一致</span>
          )}
        </div>
      </div>

      {/* 参数对比表格 */}
      <div className="space-y-4">
        {(Object.entries(groupedItems) as [string, ConfigDiffItem[]][]).map(([category, items]) => {
          if (items.length === 0) return null;
          const categoryLabel = {
            hyperparams: '超参数',
            dataset: '数据集',
            architecture: '架构'
          }[category];

          return (
            <div key={category}>
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">{categoryLabel}</div>
              <div className="space-y-1">
                {items.map(item => {
                  const direction = getValueDirection(item.parentValue, item.currentValue);

                  return (
                    <div
                      key={item.key}
                      className={`grid grid-cols-12 gap-2 px-3 py-2 rounded text-sm transition-colors ${
                        item.isChanged
                          ? 'bg-cyan-900/20 hover:bg-cyan-900/30'
                          : 'bg-slate-900/30 opacity-60'
                      }`}
                    >
                      {/* 参数名称 */}
                      <div className="col-span-3 text-slate-400">{item.label}</div>

                      {/* 父节点值 */}
                      <div className="col-span-4 text-right">
                        {renderValue(item.parentValue)}
                      </div>

                      {/* 方向指示 */}
                      <div className="col-span-1 flex justify-center">
                        {item.isChanged ? (
                          direction === 'increase' ? (
                            <TrendingUp size={14} className="text-emerald-400" />
                          ) : direction === 'decrease' ? (
                            <TrendingDown size={14} className="text-rose-400" />
                          ) : (
                            <Minus size={14} className="text-amber-400" />
                          )
                        ) : null}
                      </div>

                      {/* 当前节点值 */}
                      <div className="col-span-4 text-right font-medium">
                        <span className={item.isChanged ? 'text-cyan-400' : 'text-slate-400'}>
                          {renderValue(item.currentValue)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* 图例 */}
      <div className="flex items-center gap-4 pt-2 border-t border-slate-800 text-xs text-slate-600">
        <div className="flex items-center gap-1">
          <TrendingUp size={12} className="text-emerald-400" />
          <span>增加</span>
        </div>
        <div className="flex items-center gap-1">
          <TrendingDown size={12} className="text-rose-400" />
          <span>减少</span>
        </div>
        <div className="flex items-center gap-1">
          <Minus size={12} className="text-amber-400" />
          <span>更改</span>
        </div>
      </div>
    </div>
  );
};

export default TrainingConfigDiffView;
