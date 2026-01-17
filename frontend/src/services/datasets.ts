/**
 * 数据集相关API服务
 */

import { apiClient } from './api';

// ===== 类型定义 =====

/**
 * 数据集信息接口
 */
export interface Dataset {
  id: number;
  name: string;
  description?: string;
  path: string;
  format: string;
  num_images: number;
  num_classes: number;
  classes: string[];
  meta: Record<string, any>;
  is_active: string;
  is_standard?: boolean;  // 是否为标准格式（可直接用于训练）
  format_confidence?: number;  // 格式识别置信度 (0-1)
  created_at: string;
  updated_at?: string;
}

/**
 * 分页信息
 */
export interface PaginationInfo {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/**
 * 数据集列表响应
 */
export interface DatasetListResponse {
  success: boolean;
  message: string;
  data: Dataset[];
  pagination: PaginationInfo;
}

/**
 * 单个数据集响应
 */
export interface DatasetResponse {
  success: boolean;
  message: string;
  data: Dataset;
}

/**
 * 数据集注册数据
 */
export interface DatasetRegisterData {
  name: string;
  description?: string;
  dataset_path: string;
}

/**
 * 数据集上传数据
 */
export interface DatasetUploadData {
  name: string;
  description?: string;
  files: File[];
}

/**
 * 图像信息
 */
export interface ImageInfo {
  filename: string;
  path: string;
  size: number;
  format: string;
  width: number;
  height: number;
  annotations?: any[];
}

/**
 * 图像列表响应
 */
export interface ImageListResponse {
  success: boolean;
  message: string;
  data: {
    images: ImageInfo[];
    total: number;
    page: number;
    page_size: number;
  };
}

/**
 * 数据集统计信息
 */
export interface DatasetStatistics {
  dataset_id: number;
  num_images: number;
  num_classes: number;
  class_distribution: Record<string, number>;
  image_size_distribution: Record<string, any>;
  format_details: Record<string, any>;
  quality_metrics: Record<string, any>;
}

/**
 * 统计信息响应
 */
export interface StatisticsResponse {
  success: boolean;
  message: string;
  data: DatasetStatistics;
}

/**
 * 分页参数
 */
export interface PaginationParams {
  skip?: number;
  limit?: number;
  format_filter?: string;
}

// ===== 服务类 =====

/**
 * 数据集服务类
 */
class DatasetServiceClass {
  private baseUrl = '/datasets';

  /**
   * 获取数据集列表
   * @param params 分页和过滤参数
   * @returns 数据集列表
   */
  async getDatasets(params?: PaginationParams): Promise<Dataset[]> {
    const response = await apiClient.get<DatasetListResponse>(
      `${this.baseUrl}/`,
      { params }
    );
    return response.data;
  }

  /**
   * 获取数据集详情
   * @param id 数据集ID
   * @returns 数据集详情
   */
  async getDataset(id: number): Promise<Dataset> {
    const response = await apiClient.get<DatasetResponse>(
      `${this.baseUrl}/${id}`
    );
    return response.data;
  }

  /**
   * 上传数据集
   * @param data 上传数据
   * @returns 上传的数据集
   */
  async uploadDataset(data: DatasetUploadData): Promise<Dataset> {
    const formData = new FormData();
    formData.append('name', data.name);
    if (data.description) {
      formData.append('description', data.description);
    }
    data.files.forEach(file => {
      formData.append('files', file);
    });

    // 使用原生fetch以支持文件上传进度
    const token = localStorage.getItem('access_token');
    const response = await fetch(
      `${apiClient['baseURL']}${this.baseUrl}/upload`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || '上传失败');
    }

    const result: DatasetResponse = await response.json();
    return result.data;
  }

  /**
   * 上传数据集压缩包
   * @param name 数据集名称
   * @param description 数据集描述
   * @param archive 压缩包文件
   * @param onProgress 上传进度回调
   * @returns 上传的数据集
   */
  async uploadDatasetArchive(
    name: string,
    description: string | undefined,
    archive: File,
    onProgress?: (progress: number) => void
  ): Promise<Dataset> {
    const formData = new FormData();
    formData.append('name', name);
    if (description) {
      formData.append('description', description);
    }
    formData.append('archive', archive);

    // 使用原生fetch以支持文件上传进度
    const token = localStorage.getItem('access_token');

    // 创建XMLHttpRequest以支持进度监控
    return new Promise<Dataset>((resolve, reject) => {
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
            const result: DatasetResponse = JSON.parse(xhr.responseText);
            resolve(result.data);
          } catch (e) {
            reject(new Error('解析响应失败'));
          }
        } else {
          // 处理错误响应
          try {
            const error = JSON.parse(xhr.responseText);
            // FastAPI 返回格式: { "detail": "错误信息" }
            const errorMsg = error.detail || error.message || '上传失败';
            reject(new Error(errorMsg));
          } catch {
            // 无法解析JSON，使用状态文本
            reject(new Error(`上传失败 (${xhr.status}): ${xhr.statusText}`));
          }
        }
      });

      // 监听错误
      xhr.addEventListener('error', () => {
        reject(new Error('网络错误，上传失败'));
      });

      // 监听取消
      xhr.addEventListener('abort', () => {
        reject(new Error('上传已取消'));
      });

      // 发送请求
      xhr.open('POST', `${apiClient['baseURL']}${this.baseUrl}/upload-archive`);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      xhr.send(formData);
    });
  }

  /**
   * 注册现有数据集
   * @param data 注册数据
   * @returns 注册的数据集
   */
  async registerDataset(data: DatasetRegisterData): Promise<Dataset> {
    const response = await apiClient.post<DatasetResponse>(
      `${this.baseUrl}/register`,
      data
    );
    return response.data;
  }

  /**
   * 更新数据集
   * @param id 数据集ID
   * @param data 更新数据
   * @returns 更新后的数据集
   */
  async updateDataset(
    id: number,
    data: { name?: string; description?: string }
  ): Promise<Dataset> {
    const response = await apiClient.put<DatasetResponse>(
      `${this.baseUrl}/${id}`,
      data
    );
    return response.data;
  }

  /**
   * 删除数据集
   * @param id 数据集ID
   */
  async deleteDataset(id: number): Promise<void> {
    await apiClient.delete<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}`
    );
  }

  /**
   * 重新扫描数据集
   * @param id 数据集ID
   * @returns 重新扫描后的数据集
   */
  async rescanDataset(id: number): Promise<Dataset> {
    const response = await apiClient.post<DatasetResponse>(
      `${this.baseUrl}/${id}/rescan`,
      {}
    );
    return response.data;
  }

  /**
   * 获取数据集统计信息
   * @param id 数据集ID
   * @returns 统计信息
   */
  async getStatistics(id: number): Promise<DatasetStatistics> {
    const response = await apiClient.get<StatisticsResponse>(
      `${this.baseUrl}/${id}/statistics`
    );
    return response.data;
  }

  /**
   * 获取数据集图像列表
   * @param id 数据集ID
   * @param params 分页和排序参数
   * @returns 图像列表
   */
  async getImages(
    id: number,
    params?: {
      page?: number;
      page_size?: number;
      sort_by?: string;
      sort_order?: 'asc' | 'desc';
    }
  ): Promise<ImageListResponse['data']> {
    const response = await apiClient.get<ImageListResponse>(
      `${this.baseUrl}/${id}/images`,
      { params }
    );
    return response.data;
  }

  /**
   * 获取数据集缩略图列表
   * @param id 数据集ID
   * @returns 缩略图路径列表
   */
  async getThumbnails(id: number): Promise<string[]> {
    const response = await apiClient.get<{
      success: boolean;
      message: string;
      data: string[];
    }>(`${this.baseUrl}/${id}/thumbnails`);
    return response.data;
  }

  /**
   * 生成数据集缩略图
   * @param id 数据集ID
   */
  async generateThumbnails(id: number): Promise<void> {
    await apiClient.post<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}/thumbnails/generate`,
      {}
    );
  }

  /**
   * 删除数据集缩略图
   * @param id 数据集ID
   */
  async deleteThumbnails(id: number): Promise<void> {
    await apiClient.delete<{ success: boolean; message: string }>(
      `${this.baseUrl}/${id}/thumbnails`
    );
  }

  /**
   * 获取数据集图像URL
   * @param id 数据集ID
   * @param index 图像索引
   * @returns 图像URL
   */
  getImageUrl(id: number, index: number): string {
    return `${this.baseUrl}/${id}/image-file?index=${index}`;
  }

  /**
   * 获取数据集中所有图像的URL列表
   * @param dataset 数据集对象
   * @returns 图像URL列表
   */
  getImageUrls(dataset: Dataset): string[] {
    const imagePaths = dataset.meta?.image_paths || [];
    return imagePaths.map((_: string, index: number) =>
      this.getImageUrl(dataset.id, index)
    );
  }

  /**
   * 获取数据集目录结构
   * @param id 数据集ID
   * @param maxDepth 最大递归深度
   * @returns 目录结构树
   */
  async getDirectoryStructure(id: number, maxDepth: number = 5): Promise<any> {
    const response = await apiClient.get<{
      success: boolean;
      message: string;
      data: any;
    }>(`${this.baseUrl}/${id}/structure?max_depth=${maxDepth}`);
    return response.data;
  }
}

// 导出单例
export const datasetService = new DatasetServiceClass();

export default datasetService;
