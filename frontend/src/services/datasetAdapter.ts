/**
 * 数据集数据适配器
 * 将后端API数据格式转换为前端组件需要的格式
 */

import { Dataset } from './datasets';

/**
 * 数据集格式状态枚举
 */
export enum DatasetFormatStatus {
  STANDARD = 'standard',         // 标准格式（可直接训练）
  NON_STANDARD = 'non_standard', // 非标准格式（可预览，不能训练）
  UNRECOGNIZED = 'unrecognized'  // 未识别格式（可预览，不能训练）
}

/**
 * 统计信息接口
 */
export interface DatasetStats {
  numClasses: number;
  avgWidth: number;
  avgHeight: number;
  annotationRate: number;  // 标注率（有标注的图片比例，0-1）
}

/**
 * 前端组件使用的数据集项格式
 */
export interface DatasetItem {
  id: string;
  name: string;
  type: string;
  count: number;
  size: string;
  lastModified: string;
  description?: string;  // 数据集描述
  stats?: DatasetStats;  // 统计信息（从meta中提取）
  rawMeta?: Record<string, any>;  // 原始元数据（用于获取更详细的信息）
  formatStatus: DatasetFormatStatus;  // 格式状态
  formatConfidence: number;            // 格式识别置信度 0-1
  canTrain: boolean;                   // 是否可用于训练
}

/**
 * 判断数据集格式状态
 */
function getFormatStatus(dataset: Dataset): DatasetFormatStatus {
  // 优先使用 is_standard 字段，其次使用 meta 中的值，最后根据置信度判断
  const isStandard = dataset.is_standard ?? dataset.meta?.is_standard ?? false;
  if (isStandard) {
    return DatasetFormatStatus.STANDARD;
  }

  const confidence = dataset.meta?.format_confidence ?? dataset.format_confidence ?? 0;

  if (confidence >= 0.7) {
    return DatasetFormatStatus.STANDARD;
  } else if (confidence >= 0.3) {
    return DatasetFormatStatus.NON_STANDARD;
  } else {
    return DatasetFormatStatus.UNRECOGNIZED;
  }
}

/**
 * 判断数据集是否可用于训练
 */
function canDatasetTrain(dataset: Dataset): boolean {
  const status = getFormatStatus(dataset);
  const format = dataset.format.toLowerCase();
  const supportedFormats = ['yolo', 'coco', 'voc', 'classification'];
  return status === DatasetFormatStatus.STANDARD && supportedFormats.includes(format);
}

/**
 * 从元数据中提取统计信息
 */
function extractStats(dataset: Dataset): DatasetStats | undefined {
  const meta = dataset.meta;
  if (!meta) return undefined;

  const imageStats = meta.image_stats || {};
  const labelStats = meta.label_stats || {};

  // 类别数：优先使用num_classes字段，其次从label_stats中获取
  const numClasses = dataset.num_classes || Object.keys(labelStats.class_distribution || {}).length || 0;

  // 平均分辨率
  const avgWidth = imageStats.avg_width || 0;
  const avgHeight = imageStats.avg_height || 0;

  // 标注率计算：有标注的图片数 / 总图片数
  // 对于YOLO格式：valid_labels / num_images
  // 对于COCO/VOC格式：annotated_images / num_images
  let annotationRate = 0;
  if (dataset.num_images > 0) {
    const annotatedImages = labelStats.annotated_images || labelStats.valid_labels || 0;
    annotationRate = annotatedImages / dataset.num_images;
  }

  // 对于classification格式，所有图片都有标注（按文件夹分类）
  if (dataset.format === 'classification' && dataset.num_images > 0) {
    annotationRate = 1;
  }

  return {
    numClasses,
    avgWidth,
    avgHeight,
    annotationRate,
  };
}

/**
 * 将后端Dataset转换为前端DatasetItem
 */
export function adaptDatasetToItem(dataset: Dataset): DatasetItem {
  const formatStatus = getFormatStatus(dataset);
  const formatConfidence = dataset.meta?.format_confidence ?? dataset.format_confidence ?? 0;

  return {
    id: dataset.id.toString(),
    name: dataset.name,
    type: dataset.format.toUpperCase(),
    count: dataset.num_images,
    size: formatSize(dataset.meta?.size || 0),
    lastModified: formatAbsoluteDate(dataset.created_at),
    description: dataset.description,
    stats: extractStats(dataset),
    rawMeta: dataset.meta,
    formatStatus,
    formatConfidence,
    canTrain: canDatasetTrain(dataset),
  };
}

/**
 * 批量转换Dataset列表
 */
export function adaptDatasetList(datasets: Dataset[]): DatasetItem[] {
  return datasets.map(adaptDatasetToItem);
}

/**
 * 格式化文件大小
 */
function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * 格式化日期为相对时间
 */
function formatDate(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSeconds < 60) {
    return '刚刚';
  } else if (diffMinutes < 60) {
    return `${diffMinutes}分钟前`;
  } else if (diffHours < 24) {
    return `${diffHours}小时前`;
  } else if (diffDays < 7) {
    return `${diffDays}天前`;
  } else if (diffDays < 30) {
    return `${Math.floor(diffDays / 7)}周前`;
  } else if (diffDays < 365) {
    return `${Math.floor(diffDays / 30)}个月前`;
  } else {
    return `${Math.floor(diffDays / 365)}年前`;
  }
}

/**
 * 格式化绝对日期
 */
export function formatAbsoluteDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default {
  adaptDatasetToItem,
  adaptDatasetList,
  formatSize,
  formatDate,
  formatAbsoluteDate,
};
