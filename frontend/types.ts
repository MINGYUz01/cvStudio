import React from 'react';

export enum ViewState {
  DASHBOARD = 'dashboard',
  DATASETS = 'datasets',
  DATA_AUGMENTATION = 'data_augmentation',
  MODEL_BUILDER = 'model_builder',
  TRAINING = 'training',
  INFERENCE = 'inference',
  SETTINGS = 'settings'
}

export interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: number; // percentage
  icon: React.ReactNode;
  color?: 'cyan' | 'purple' | 'emerald' | 'rose' | 'amber';
  onClick?: () => void;
}

export interface DatasetStats {
  numClasses: number;
  avgWidth: number;
  avgHeight: number;
  annotationRate: number;  // 标注率（有标注的图片比例，0-1）
}

export interface DatasetItem {
  id: string;
  name: string;
  type: string;  // 支持更多格式: 'YOLO' | 'COCO' | 'VOC' | 'CLASSIFICATION' | 'UNKNOWN' | 'Folder'
  count: number;
  size: string;
  lastModified: string;
  description?: string;  // 数据集描述
  stats?: DatasetStats;  // 统计信息（从meta中提取）
  rawMeta?: Record<string, any>;  // 原始元数据（用于获取更详细的信息）
}

export interface ModelNode {
  id: string;
  type: string;
  label: string;
  x: number;
  y: number;
  inputs: string[];
  outputs: string[];
}

export interface LogEntry {
  id: number;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR';
  message: string;
}

export interface WeightCheckpoint {
  id: number;  // Changed to number to match database ID
  name: string;
  display_name: string;
  description?: string;
  task_type: 'classification' | 'detection';
  version: string;
  file_name: string;
  file_size?: number;
  file_size_mb?: number;
  framework: 'pytorch' | 'onnx';
  created_at: string;
  is_root?: boolean;
  source_type?: 'uploaded' | 'trained';
  architecture_id?: number;
  parent_version_id?: number;
  // Legacy fields for backward compatibility
  architecture?: string;  // Can be derived from task_type
  format?: 'PyTorch' | 'ONNX' | 'TensorRT';  // Mapped from framework
  size?: string;  // Can be derived from file_size_mb
  accuracy?: string;  // Not in DB, optional
  tags?: string[];  // Not in DB, optional
}

// ==================== 权重树形结构相关类型 ====================

export interface WeightTreeItem extends WeightCheckpoint {
  is_root: boolean;
  source_type: 'uploaded' | 'trained';
  source_training_id?: number;
  architecture_id?: number;
  parent_version_id?: number;
  children: WeightTreeItem[];
}

export interface WeightTrainingConfig {
  weight_id: number;
  weight_name: string;
  training_config: Record<string, any> | null;
  source_training: {
    id: number;
    name: string;
    hyperparams: Record<string, any>;
  } | null;
}

export interface WeightForTrainingOption {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  task_type: 'classification' | 'detection';
  version: string;
  file_path: string;
  architecture_id?: number;
  architecture_name?: string;
  created_at: string;
}

export interface WeightRootListResponse {
  weights: WeightCheckpoint[];
  total: number;
}

// ==================== 数据增强相关类型 ====================

export type ParamType = 'boolean' | 'integer' | 'float' | 'range' | 'select';

export interface AugmentationParam {
  name: string;
  label_zh: string;
  label_en: string;
  type: ParamType;
  default: any;
  min_value?: number;
  max_value?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
  description: string;
}

export interface AugmentationOperator {
  id: string;
  name_zh: string;
  name_en: string;
  category: string;
  category_label_zh: string;
  category_label_en: string;
  description: string;
  enabled: boolean;
  params: AugmentationParam[];
}

export interface PipelineItem {
  instanceId: string;
  operatorId: string;
  enabled: boolean;
  params: Record<string, any>;
}

export interface AugmentationStrategy {
  id: number;
  user_id: number;
  name: string;
  description: string | null;
  pipeline: PipelineItem[];
  is_default: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface AugmentationOperatorsResponse {
  [category: string]: {
    category: string;
    label_zh: string;
    label_en: string;
    operators: AugmentationOperator[];
  };
}

export interface AugmentationStrategyListResponse {
  strategies: AugmentationStrategy[];
  total: number;
}

export interface AugmentedImage {
  original_path: string;
  augmented_data: string;  // base64编码
  augmentation_config: any;
  applied_operations: string[];
}

export interface AugmentationPreview {
  original_image: string;  // base64编码
  augmented_images: AugmentedImage[];
  augmentation_summary: {
    total_operators: number;
    applied_operators: number;
    operations_used: string[];
    image_info?: {
      width: number;
      height: number;
      format: string;
      size: number;
    };
  };
}

export interface APIResponse<T> {
  success: boolean;
  message: string;
  data: T;
}