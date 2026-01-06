/**
 * 数据集数据适配器
 * 将后端API数据格式转换为前端组件需要的格式
 */

import { Dataset } from './datasets';

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
}

/**
 * 将后端Dataset转换为前端DatasetItem
 */
export function adaptDatasetToItem(dataset: Dataset): DatasetItem {
  return {
    id: dataset.id.toString(),
    name: dataset.name,
    type: dataset.format.toUpperCase(),
    count: dataset.num_images,
    size: formatSize(dataset.meta?.size || 0),
    lastModified: formatDate(dataset.created_at),
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
