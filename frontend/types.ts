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
  id: string;
  name: string;
  architecture: string; // e.g., 'YOLOv8', 'ResNet50'
  format: 'PyTorch' | 'ONNX' | 'TensorRT';
  size: string;
  accuracy: string; // e.g., 'mAP 0.89'
  created: string;
  tags: string[];
}