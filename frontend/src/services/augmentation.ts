/**
 * 数据增强 API 服务
 */

import type {
  APIResponse,
  AugmentationOperator,
  AugmentationOperatorsResponse,
  AugmentationStrategy,
  AugmentationStrategyListResponse,
  AugmentationPreview,
  PipelineItem
} from '../types';

// API 基础 URL
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

/**
 * 获取认证 Token
 */
function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('access_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
}

/**
 * 处理 API 响应
 */
async function handleResponse<T>(response: Response): Promise<APIResponse<T>> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: '请求失败' }));
    throw new Error(error.detail || error.message || '请求失败');
  }
  return response.json();
}

// ==================== 增强算子相关 ====================

/**
 * 获取所有可用的数据增强算子
 */
export async function getAugmentationOperators(): Promise<APIResponse<AugmentationOperatorsResponse>> {
  const response = await fetch(`${API_BASE}/augmentation/operators`, {
    headers: getAuthHeaders()
  });
  return handleResponse<AugmentationOperatorsResponse>(response);
}

// ==================== 增强策略相关 ====================

/**
 * 获取增强策略列表
 */
export async function getAugmentationStrategies(params?: {
  skip?: number;
  limit?: number;
  search?: string;
}): Promise<APIResponse<AugmentationStrategyListResponse>> {
  const queryParams = new URLSearchParams();
  if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString());
  if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString());
  if (params?.search) queryParams.append('search', params.search);

  const response = await fetch(`${API_BASE}/augmentation/strategies?${queryParams}`, {
    headers: getAuthHeaders()
  });
  return handleResponse<AugmentationStrategyListResponse>(response);
}

/**
 * 获取单个增强策略详情
 */
export async function getAugmentationStrategy(
  strategyId: number
): Promise<APIResponse<AugmentationStrategy>> {
  const response = await fetch(`${API_BASE}/augmentation/strategies/${strategyId}`, {
    headers: getAuthHeaders()
  });
  return handleResponse<AugmentationStrategy>(response);
}

/**
 * 创建新的增强策略
 */
export async function createAugmentationStrategy(data: {
  name: string;
  description?: string;
  pipeline: PipelineItem[];
}): Promise<APIResponse<AugmentationStrategy>> {
  const response = await fetch(`${API_BASE}/augmentation/strategies`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(data)
  });
  return handleResponse<AugmentationStrategy>(response);
}

/**
 * 更新增强策略
 */
export async function updateAugmentationStrategy(
  strategyId: number,
  data: {
    name?: string;
    description?: string;
    pipeline?: PipelineItem[];
  }
): Promise<APIResponse<AugmentationStrategy>> {
  const response = await fetch(`${API_BASE}/augmentation/strategies/${strategyId}`, {
    method: 'PUT',
    headers: getAuthHeaders(),
    body: JSON.stringify(data)
  });
  return handleResponse<AugmentationStrategy>(response);
}

/**
 * 删除增强策略
 */
export async function deleteAugmentationStrategy(
  strategyId: number
): Promise<APIResponse<boolean>> {
  const response = await fetch(`${API_BASE}/augmentation/strategies/${strategyId}`, {
    method: 'DELETE',
    headers: getAuthHeaders()
  });
  return handleResponse<boolean>(response);
}

// ==================== 增强预览相关 ====================

/**
 * 预览数据增强效果
 * @param params 预览参数
 * @param signal 可选的AbortSignal用于取消请求
 */
export async function previewAugmentation(
  params: {
    image_path: string;
    dataset_id?: number;
    pipeline: PipelineItem[];
    seed?: number;
  },
  signal?: AbortSignal
): Promise<APIResponse<AugmentationPreview>> {
  const queryParams = new URLSearchParams();
  queryParams.append('image_path', params.image_path);
  if (params.dataset_id !== undefined) queryParams.append('dataset_id', params.dataset_id.toString());
  if (params.seed !== undefined) queryParams.append('seed', params.seed.toString());

  // pipeline 直接作为请求体发送
  const response = await fetch(`${API_BASE}/augmentation/preview?${queryParams}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(params.pipeline),
    signal  // 支持请求取消
  });
  return handleResponse<AugmentationPreview>(response);
}

/**
 * 使用 Base64 图像进行增强预览（前端生成预览用）
 */
export async function previewAugmentationWithBase64(
  imageBase64: string,
  pipeline: PipelineItem[]
): Promise<AugmentationPreview> {
  // 这个方法在前端实现，使用 Canvas API 进行简单的预览
  // 实际的增强应该通过后端 API 完成

  const img = new Image();
  img.src = `data:image/jpeg;base64,${imageBase64}`;

  return new Promise((resolve, reject) => {
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        reject(new Error('无法创建 Canvas 上下文'));
        return;
      }

      // 应用每个增强操作
      const appliedOperations: string[] = [];

      // 这里简化处理，实际增强应该由后端完成
      // 只做基本的 CSS 滤镜模拟
      let filterString = '';
      let transformString = '';

      for (const item of pipeline) {
        if (!item.enabled) continue;

        const params = item.params;

        switch (item.operatorId) {
          case 'horizontal_flip':
            transformString += 'scaleX(-1) ';
            appliedOperations.push('水平翻转');
            break;
          case 'vertical_flip':
            transformString += 'scaleY(-1) ';
            appliedOperations.push('垂直翻转');
            break;
          case 'rotate':
            const angle = params.angle || 0;
            transformString += `rotate(${angle}deg) `;
            appliedOperations.push(`旋转 ${angle}°`);
            break;
          case 'brightness':
            const brightness = (params.factor || 1) * 100;
            filterString += `brightness(${brightness}%) `;
            appliedOperations.push(`亮度 ${params.factor}`);
            break;
          case 'contrast':
            const contrast = (params.factor || 1) * 100;
            filterString += `contrast(${contrast}%) `;
            appliedOperations.push(`对比度 ${params.factor}`);
            break;
          case 'gaussian_blur':
            const blur = params.sigma || 0;
            filterString += `blur(${blur}px) `;
            if (blur > 0) appliedOperations.push(`高斯模糊 ${blur}`);
            break;
        }
      }

      ctx.filter = filterString;
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.transform(1, 0, 0, 1, 0, 0);
      ctx.translate(-canvas.width / 2, -canvas.height / 2);
      ctx.drawImage(img, 0, 0);

      const augmentedData = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];

      resolve({
        original_image: imageBase64,
        augmented_images: [
          {
            original_path: 'preview',
            augmented_data: augmentedData,
            augmentation_config: {},
            applied_operations: appliedOperations
          }
        ],
        augmentation_summary: {
          total_operators: pipeline.length,
          applied_operators: appliedOperations.length,
          operations_used: appliedOperations,
          image_info: {
            width: img.width,
            height: img.height,
            format: 'JPEG',
            size: imageBase64.length
          }
        }
      });
    };

    img.onerror = () => {
      reject(new Error('图像加载失败'));
    };
  });
}

export default {
  getAugmentationOperators,
  getAugmentationStrategies,
  getAugmentationStrategy,
  createAugmentationStrategy,
  updateAugmentationStrategy,
  deleteAugmentationStrategy,
  previewAugmentation,
  previewAugmentationWithBase64
};
