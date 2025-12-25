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

export interface DatasetItem {
  id: string;
  name: string;
  type: 'YOLO' | 'COCO' | 'VOC' | 'Folder';
  count: number;
  size: string;
  lastModified: string;
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