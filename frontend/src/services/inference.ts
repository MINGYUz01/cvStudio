/**
 * 推理相关API服务
 * 支持权重库管理、推理执行、结果获取
 */

import { apiClient } from './api';

// ===== 类型定义 =====

/**
 * 任务类型
 */
export type TaskType = 'classification' | 'detection' | 'auto';

/**
 * 推理模式
 */
export type InferenceMode = 'single' | 'batch' | 'stream';

/**
 * 推理状态
 */
export type InferenceStatus = 'idle' | 'processing' | 'completed' | 'error';

/**
 * 权重库信息
 */
export interface WeightLibrary {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  task_type: TaskType;
  version: string;
  file_path: string;
  file_name: string;
  file_size?: number;
  file_size_mb?: number;
  framework: 'pytorch' | 'onnx';
  input_size?: [number, number];
  class_names?: string[];
  normalize_params?: {
    mean?: number[];
    std?: number[];
  };
  is_auto_detected: boolean;
  created_at?: string;
  updated_at?: string;
}

/**
 * 权重库创建请求
 */
export interface WeightLibraryCreate {
  name: string;
  task_type: TaskType;
  description?: string;
  input_size?: [number, number];
  class_names?: string[];
  normalize_params?: {
    mean?: number[];
    std?: number[];
  };
}

/**
 * 任务类型检测结果
 */
export interface TaskTypeDetectionResult {
  task_type: TaskType;
  confidence: string;
  output_shape?: number[];
}

/**
 * 版本信息
 */
export interface VersionInfo {
  id: number;
  version: string;
  created_at?: string;
}

/**
 * 权重版本历史
 */
export interface WeightVersionHistory {
  weight_id: number;
  weight_name: string;
  versions: VersionInfo[];
}

/**
 * 推理结果
 */
export interface InferenceResult {
  type: 'detection' | 'classification';
  bbox?: [number, number, number, number]; // x1, y1, x2, y2
  label?: string;
  class_id?: number;
  confidence: number;
  area_percentage?: number;
  pixel_count?: number;
}

/**
 * 推理指标
 */
export interface InferenceMetrics {
  inference_time: number;
  fps?: number;
  preprocessing_time?: number;
  postprocessing_time?: number;
  total_time?: number;
  device: string;
  image_size: [number, number];
}

/**
 * 可视化结果
 */
export interface VisualizationResult {
  image_url?: string;
  overlay_url?: string;
  mask_url?: string;
}

/**
 * 导出数据结果
 */
export interface ExportDataResult {
  json_url?: string;
  yolo_url?: string;
  coco_url?: string;
}

/**
 * 推理响应
 */
export interface InferencePredictResponse {
  task_type: TaskType;
  results: InferenceResult[];
  metrics: InferenceMetrics;
  visualization?: VisualizationResult;
  export_data?: ExportDataResult;
  image_path: string;
  weight_id?: number;
}

/**
 * 推理请求
 */
export interface WeightInferenceRequest {
  weight_id: number;
  image_path?: string;
  confidence_threshold?: number;
  iou_threshold?: number;
  top_k?: number;
  device?: 'auto' | 'cuda' | 'cpu';
  save_visualization?: boolean;
}

/**
 * 批量推理请求
 */
export interface BatchInferenceRequest {
  weight_id: number;
  image_paths: string[];
  confidence_threshold?: number;
  iou_threshold?: number;
}

/**
 * 批量推理响应
 */
export interface BatchInferenceResponse {
  job_id: number;
  status: string;
  total_images: number;
  message: string;
}

// ===== 服务类 =====

/**
 * 推理服务类
 */
class InferenceServiceClass {
  private weightsUrl = '/weights';
  private inferenceUrl = '/inference';

  // ==================== 权重库管理 ====================

  /**
   * 获取权重列表
   * @param taskType 任务类型过滤
   * @param isActive 是否只返回活跃权重
   * @returns 权重列表
   */
  async getWeights(taskType?: TaskType, isActive: boolean = true): Promise<WeightLibrary[]> {
    const params: Record<string, string | number | boolean> = {
      is_active: isActive
    };
    if (taskType) {
      params.task_type = taskType;
    }

    // 后端返回格式: { weights: [...], total: number }
    const response = await apiClient.get<{ weights: WeightLibrary[], total: number }>(`${this.weightsUrl}`, { params });

    // 从响应中提取权重数组
    return response?.weights || [];
  }

  /**
   * 获取单个权重详情
   * @param weightId 权重ID
   * @returns 权重详情
   */
  async getWeight(weightId: number): Promise<WeightLibrary> {
    // 后端直接返回对象，不是 { data: {} } 格式
    const response = await apiClient.get<WeightLibrary>(`${this.weightsUrl}/${weightId}`);
    return response;
  }

  /**
   * 上传权重文件
   * @param file 权重文件
   * @param data 权重元信息
   * @param onProgress 上传进度回调
   * @returns 上传的权重
   */
  async uploadWeight(
    file: File,
    data: WeightLibraryCreate,
    onProgress?: (progress: number) => void
  ): Promise<WeightLibrary> {
    const formData = new FormData();
    formData.append('file', file);
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

    // 使用XMLHttpRequest以支持进度监控
    const token = localStorage.getItem('access_token');
    const baseUrl = apiClient['baseURL'];

    return new Promise<WeightLibrary>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // 构建完整的请求URL
      const requestUrl = `${baseUrl}${this.weightsUrl}/upload`;
      console.log('[上传权重] 请求URL:', requestUrl);
      console.log('[上传权重] 文件名:', file.name, '大小:', file.size);

      // 监听上传进度
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = Math.round((e.loaded / e.total) * 100);
          onProgress(progress);
        }
      });

      // 监听完成
      xhr.addEventListener('load', () => {
        console.log('[上传权重] 响应状态:', xhr.status);
        console.log('[上传权重] 响应内容:', xhr.responseText);

        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            // 后端直接返回对象，不是包裹在 data 中
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
        console.error('[上传权重] 网络错误');
        reject(new Error('网络错误，上传失败'));
      });

      // 发送请求
      xhr.open('POST', requestUrl);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      console.log('[上传权重] Authorization:', `Bearer ${token ? token.substring(0, 20) + '...' : 'null'}`);
      xhr.send(formData);
    });
  }

  /**
   * 删除权重
   * @param weightId 权重ID
   */
  async deleteWeight(weightId: number): Promise<void> {
    await apiClient.delete<{
      success: boolean;
      message: string;
    }>(`${this.weightsUrl}/${weightId}`);
  }

  /**
   * 获取权重版本历史
   * @param weightId 权重ID
   * @returns 版本历史
   */
  async getWeightVersions(weightId: number): Promise<WeightVersionHistory> {
    const response = await apiClient.get<WeightVersionHistory>(
      `${this.weightsUrl}/${weightId}/versions`
    );
    return response;
  }

  /**
   * 创建权重新版本
   * @param weightId 父版本ID
   * @param file 新权重文件
   * @param description 版本描述
   * @param onProgress 上传进度回调
   * @returns 新版本权重
   */
  async createWeightVersion(
    weightId: number,
    file: File,
    description?: string,
    onProgress?: (progress: number) => void
  ): Promise<WeightLibrary> {
    const formData = new FormData();
    formData.append('file', file);
    if (description) {
      formData.append('description', description);
    }

    const token = localStorage.getItem('access_token');

    return new Promise<WeightLibrary>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = Math.round((e.loaded / e.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result.data);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || '创建版本失败'));
          } catch {
            reject(new Error(`创建版本失败 (${xhr.status})`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，创建版本失败'));
      });

      xhr.open('POST', `${apiClient['baseURL']}${this.weightsUrl}/${weightId}/version`);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      xhr.send(formData);
    });
  }

  /**
   * 自动检测权重文件的任务类型
   * @param file 权重文件
   * @param onProgress 上传进度回调
   * @returns 检测结果
   */
  async detectTaskType(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<TaskTypeDetectionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('access_token');

    return new Promise<TaskTypeDetectionResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = Math.round((e.loaded / e.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result.data);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || '检测失败'));
          } catch {
            reject(new Error(`检测失败 (${xhr.status})`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，检测失败'));
      });

      xhr.open('POST', `${apiClient['baseURL']}${this.weightsUrl}/detect-type`);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      xhr.send(formData);
    });
  }

  // ==================== 推理执行 ====================

  /**
   * 使用权重进行单图推理
   * @param request 推理请求
   * @returns 推理结果
   */
  async predict(request: WeightInferenceRequest): Promise<InferencePredictResponse> {
    // 后端直接返回对象
    const response = await apiClient.post<InferencePredictResponse>(
      `${this.inferenceUrl}/predict`, request
    );
    return response;
  }

  /**
   * 使用权重进行批量推理
   * @param request 批量推理请求
   * @returns 批量推理任务信息
   */
  async predictBatch(request: BatchInferenceRequest): Promise<BatchInferenceResponse> {
    const response = await apiClient.post<BatchInferenceResponse>(
      `${this.inferenceUrl}/batch`, request
    );
    return response;
  }

  /**
   * 上传图像并推理
   * @param weightId 权重ID
   * @param file 图像文件
   * @param options 推理选项
   * @param onProgress 上传进度回调
   * @returns 推理结果
   */
  async predictWithImage(
    weightId: number,
    file: File,
    options?: {
      confidence_threshold?: number;
      iou_threshold?: number;
      top_k?: number;
      device?: 'auto' | 'cuda' | 'cpu';
    },
    onProgress?: (progress: number) => void
  ): Promise<InferencePredictResponse> {
    const formData = new FormData();
    // 必须先添加 weight_id，后端需要它
    formData.append('weight_id', String(weightId));
    formData.append('image', file);
    if (options?.confidence_threshold !== undefined) {
      formData.append('confidence_threshold', String(options.confidence_threshold));
    }
    if (options?.iou_threshold !== undefined) {
      formData.append('iou_threshold', String(options.iou_threshold));
    }
    if (options?.top_k !== undefined) {
      formData.append('top_k', String(options.top_k));
    }
    if (options?.device) {
      formData.append('device', options.device);
    }

    const token = localStorage.getItem('access_token');

    return new Promise<InferencePredictResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = Math.round((e.loaded / e.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result.data);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || '推理失败'));
          } catch {
            reject(new Error(`推理失败 (${xhr.status})`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，推理失败'));
      });

      xhr.open('POST', `${apiClient['baseURL']}${this.inferenceUrl}/predict-image`);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      xhr.send(formData);
    });
  }

  /**
   * 获取批量推理任务状态
   * @param jobId 任务ID
   * @returns 任务状态
   */
  async getBatchJobStatus(jobId: number): Promise<{
    job_id: number;
    status: string;
    processed: number;
    total: number;
    progress: number;
  }> {
    const response = await apiClient.get<{
      job_id: number;
      status: string;
      processed: number;
      total: number;
      progress: number;
    }>(`${this.inferenceUrl}/jobs/${jobId}`);
    return response;
  }

  /**
   * 获取批量推理结果
   * @param jobId 任务ID
   * @returns 推理结果列表
   */
  async getBatchResults(jobId: number): Promise<InferencePredictResponse[]> {
    const response = await apiClient.get<InferencePredictResponse[]>(
      `${this.inferenceUrl}/jobs/${jobId}/results`
    );
    // 确保返回数组
    return Array.isArray(response) ? response : [];
  }

  // ==================== 工具方法 ====================

  /**
   * 格式化文件大小
   * @param bytes 字节数
   * @returns 格式化后的字符串
   */
  formatFileSize(bytes?: number): string {
    if (!bytes) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  /**
   * 获取任务类型的中文名称
   * @param taskType 任务类型
   * @returns 中文名称
   */
  getTaskTypeName(taskType: TaskType): string {
    const names: Record<string, string> = {
      classification: '分类',
      detection: '检测',
      auto: '自动检测'
    };
    return names[taskType] || taskType;
  }

  /**
   * 获取任务类型的图标颜色
   * @param taskType 任务类型
   * @returns 颜色类名
   */
  getTaskTypeColor(taskType: TaskType): string {
    const colors: Record<string, string> = {
      classification: 'text-purple-400',
      detection: 'text-cyan-400',
      auto: 'text-amber-400'
    };
    return colors[taskType] || 'text-slate-400';
  }

  /**
   * 获取任务类型的背景颜色
   * @param taskType 任务类型
   * @returns 背景颜色类名
   */
  getTaskTypeBgColor(taskType: TaskType): string {
    const colors: Record<string, string> = {
      classification: 'bg-purple-500/10 border-purple-500/30',
      detection: 'bg-cyan-500/10 border-cyan-500/30',
      auto: 'bg-amber-500/10 border-amber-500/30'
    };
    return colors[taskType] || 'bg-slate-500/10 border-slate-500/30';
  }
}

// 导出单例
export const inferenceService = new InferenceServiceClass();

export default inferenceService;
