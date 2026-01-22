/**
 * 权重库相关API服务
 *
 * 提供权重文件的上传、查询、删除等功能
 */

import { apiClient } from './api';

// ==============================
// 类型定义
// ==============================

export type TaskType = 'classification' | 'detection' | 'auto';

export type Framework = 'pytorch' | 'onnx';

// ==============================
// 训练配置详情相关类型
// ==============================

export interface DatasetInfo {
  id: number;
  name: string;
  format: string;
  num_images?: number;
  num_classes?: number;
  classes?: string[];
  path?: string;
}

export interface ModelArchitectureInfo {
  id: number;
  name: string;
  description?: string;
  file_path?: string;
  input_size?: number[];
  task_type?: string;
}

export interface PretrainedWeightInfo {
  id: number;
  name: string;
  display_name: string;
  task_type: string;
  version: string;
  source_type: 'uploaded' | 'trained';
}

export interface AugmentationConfigInfo {
  enabled: boolean;
  strategy?: string;
  strategy_id?: number;
  description?: string;
  config?: Record<string, any> | any[];  // 支持对象和数组（pipeline）
}

export interface WeightUploadData {
  file: File;
  name: string;
  task_type: TaskType;
  description?: string;
  input_size?: number[];
  class_names?: string[];
  normalize_params?: Record<string, any>;
}

export interface WeightLibraryItem {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  task_type: TaskType;
  version: string;
  file_name: string;
  file_size_mb?: number;
  framework: Framework;
  is_auto_detected: boolean;
  is_root?: boolean;
  source_type?: 'uploaded' | 'trained';
  architecture_id?: number;
  parent_version_id?: number;
  created_at: string;
}

export interface WeightLibraryListResponse {
  weights: WeightLibraryItem[];
  total: number;
}

export interface WeightUploadResponse {
  success: boolean;
  message: string;
  weight_id: number;
  weight: WeightLibraryItem;
}

export interface WeightUploadOptions {
  name: string;
  task_type: TaskType;
  description?: string;
  input_size?: number[];
  class_names?: string[];
  normalize_params?: Record<string, any>;
}

// ==============================
// 权重树形结构相关类型
// ==============================

export interface WeightTreeItem extends WeightLibraryItem {
  is_root: boolean;
  source_type: 'uploaded' | 'trained';
  source_training_id?: number;
  architecture_id?: number;
  children: WeightTreeItem[];
}

export interface WeightRootListResponse {
  weights: WeightLibraryItem[];
  total: number;
}

export interface WeightTrainingConfig {
  weight_id: number;
  weight_name: string;
  training_config: Record<string, any> | null;
  source_training: {
    id: number;
    name: string;
    status?: string;
    created_at?: string;
    hyperparams?: Record<string, any>;
  } | null;
  // 完整的关联信息
  dataset?: DatasetInfo | null;
  model_architecture?: ModelArchitectureInfo | null;
  pretrained_weight?: PretrainedWeightInfo | null;
  augmentation?: AugmentationConfigInfo | null;
}

export interface WeightForTrainingOption {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  task_type: TaskType;
  version: string;
  file_path: string;
  architecture_id?: number;
  architecture_name?: string;
  created_at: string;
}

// ==============================
// API方法
// ==============================

class WeightService {
  /**
   * 上传权重文件
   * @param data 上传数据
   * @param onProgress 上传进度回调
   */
  async uploadWeight(
    data: WeightUploadData,
    onProgress?: (progress: number) => void
  ): Promise<WeightUploadResponse> {
    const formData = new FormData();
    formData.append('file', data.file);
    formData.append('name', data.name);
    formData.append('task_type', data.task_type);
    if (data.description) {
      formData.append('description', data.description);
    }
    if (data.input_size) {
      formData.append('input_size', JSON.stringify(data.input_size));
    }
    if (data.class_names) {
      formData.append('class_names', JSON.stringify(data.class_names));
    }
    if (data.normalize_params) {
      formData.append('normalize_params', JSON.stringify(data.normalize_params));
    }

    const token = localStorage.getItem('access_token');
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

    // 使用原生fetch以支持文件上传进度
    return new Promise<WeightUploadResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // 监听上传进度
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = Math.round((e.loaded / e.total) * 100);
          onProgress(progress);
        }
      });

      // 监听完成
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result: WeightUploadResponse = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            const errorMsg = error.detail || error.message || '上传失败';
            reject(new Error(errorMsg));
          } catch {
            reject(new Error(`上传失败 (${xhr.status}): ${xhr.statusText}`));
          }
        }
      });

      // 监听错误
      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，请检查连接'));
      });

      // 监听取消
      xhr.addEventListener('abort', () => {
        reject(new Error('上传已取消'));
      });

      // 发送请求
      xhr.open('POST', `${baseUrl}/weights/upload`);
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }
      xhr.send(formData);
    });
  }

  /**
   * 获取权重列表
   */
  async getWeights(taskType?: TaskType): Promise<WeightLibraryListResponse> {
    const params = taskType ? { task_type: taskType } : undefined;
    return await apiClient.get<WeightLibraryListResponse>('/weights', params);
  }

  /**
   * 获取权重详情
   */
  async getWeight(weightId: number): Promise<WeightLibraryItem> {
    return await apiClient.get<WeightLibraryItem>(`/weights/${weightId}`);
  }

  /**
   * 删除权重
   */
  async deleteWeight(weightId: number): Promise<{ message: string; id: number }> {
    return await apiClient.delete<{ message: string; id: number }>(`/weights/${weightId}`);
  }

  /**
   * 创建权重新版本
   */
  async createWeightVersion(
    weightId: number,
    file: File,
    description?: string
  ): Promise<WeightLibraryItem> {
    const formData = new FormData();
    formData.append('file', file);
    if (description) {
      formData.append('description', description);
    }

    const token = localStorage.getItem('access_token');
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

    return new Promise<WeightLibraryItem>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result: WeightLibraryItem = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            const errorMsg = error.detail || error.message || '创建版本失败';
            reject(new Error(errorMsg));
          } catch {
            reject(new Error(`创建版本失败 (${xhr.status}): ${xhr.statusText}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，请检查连接'));
      });

      xhr.open('POST', `${baseUrl}/weights/${weightId}/versions`);
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }
      xhr.send(formData);
    });
  }

  /**
   * 获取权重版本历史
   */
  async getWeightVersions(weightId: number): Promise<WeightLibraryListResponse> {
    return await apiClient.get<WeightLibraryListResponse>(`/weights/${weightId}/versions`);
  }

  /**
   * 按任务类型获取权重列表
   */
  async getWeightsByTask(taskType: TaskType): Promise<WeightLibraryListResponse> {
    return await apiClient.get<WeightLibraryListResponse>(`/weights/by-task/${taskType}`);
  }

  // ==============================
  // 权重树形结构相关API方法
  // ==============================

  /**
   * 获取根节点权重列表
   */
  async getRootWeights(taskType?: TaskType): Promise<WeightRootListResponse> {
    const params = taskType ? { task_type: taskType } : undefined;
    return await apiClient.get<WeightRootListResponse>('/weights/roots', params);
  }

  /**
   * 获取完整的权重树形结构
   */
  async getWeightTree(): Promise<WeightTreeItem[]> {
    return await apiClient.get<WeightTreeItem[]>('/weights/tree');
  }

  /**
   * 获取指定权重的子树
   */
  async getWeightSubtree(weightId: number): Promise<WeightTreeItem> {
    return await apiClient.get<WeightTreeItem>(`/weights/${weightId}/tree`);
  }

  /**
   * 获取权重的训练配置信息
   */
  async getWeightTrainingConfig(weightId: number): Promise<WeightTrainingConfig> {
    return await apiClient.get<WeightTrainingConfig>(`/weights/${weightId}/config`);
  }

  /**
   * 获取可用于训练的权重列表（根据架构筛选）
   */
  async getWeightsForTraining(
    architectureId?: number,
    taskType?: TaskType
  ): Promise<WeightForTrainingOption[]> {
    const params: any = {};
    if (architectureId) params.architecture_id = architectureId;
    if (taskType) params.task_type = taskType;
    return await apiClient.get<WeightForTrainingOption[]>('/weights/for-training', params);
  }

  /**
   * 获取按模型代码筛选的权重树（用于预训练权重选择）
   * 当提供 modelCodeId 时，只返回该模型训练产生的权重
   */
  async getWeightTreeByArchitecture(
    modelCodeId?: number,
    taskType?: TaskType
  ): Promise<WeightTreeItem[]> {
    const params: any = {};
    if (modelCodeId) params.model_code_id = modelCodeId;
    if (taskType) params.task_type = taskType;
    return await apiClient.get<WeightTreeItem[]>('/weights/tree-by-architecture', params);
  }

  /**
   * 按架构ID筛选权重（旧版API，保留兼容）
   * @deprecated 建议使用 getWeightTreeByArchitecture 并传入 modelCodeId
   */
  async getWeightTreeByArchitectureId(
    architectureId: number,
    taskType?: TaskType
  ): Promise<WeightTreeItem[]> {
    const params: any = { architecture_id: architectureId };
    if (taskType) params.task_type = taskType;
    return await apiClient.get<WeightTreeItem[]>('/weights/tree-by-architecture', params);
  }
}

export const weightService = new WeightService();

export default weightService;
